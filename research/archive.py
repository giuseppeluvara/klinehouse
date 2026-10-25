"""Lossless 1-second archive: bounded export, immutable Parquet, atomic catalog.

Only catalogued files are readable. Writers publish a verified generation before
atomically replacing the catalog; crashes can leave an orphan, never half a file.
A shared/exclusive filesystem lock also protects readers during compaction/GC.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS = (
    "symbol",
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
)
SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("open_time", pa.timestamp("us", tz="UTC")),
        ("close_time", pa.timestamp("us", tz="UTC")),
        *((name, pa.float64()) for name in COLUMNS[3:9]),
        ("trades", pa.int32()),
        ("taker_buy_base", pa.float64()),
        ("taker_buy_quote", pa.float64()),
    ]
)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ROW = struct.Struct("!qq6di2d")
DEFAULT_BATCH = 50_000


def utc(value: str | datetime) -> datetime:
    stamp = (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, str)
        else value
    )
    return (
        stamp.replace(tzinfo=timezone.utc)
        if stamp.tzinfo is None
        else stamp.astimezone(timezone.utc)
    )


def symbol_name(value: str) -> str:
    value = value.upper()
    if not re.fullmatch(r"[A-Z0-9]{2,30}", value):
        raise ValueError("invalid Binance symbol")
    return value


def micros(stamp: datetime) -> int:
    delta = utc(stamp) - EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


class Fingerprint:
    """Batch-boundary independent, exact fingerprint of all stored columns."""

    def __init__(self):
        self.digest = hashlib.sha256()
        self.count = 0
        self.first = self.last = None
        self.gaps = 0

    def update(self, rows, *, symbol: str, start: datetime, end: datetime):
        for row in rows:
            stamp = utc(row[1])
            if (
                row[0] != symbol
                or not start <= stamp < end
                or stamp.microsecond
                or (self.last is not None and stamp <= self.last)
            ):
                raise ValueError(
                    "wrong symbol, range, duplicate or unordered 1s candle"
                )
            if not stamp <= utc(row[2]) < stamp + timedelta(seconds=1):
                raise ValueError("archive accepts closed 1-second candles only")
            if any(value is None or not math.isfinite(value) for value in row[3:]):
                raise ValueError("candle numeric fields must be finite and non-null")
            if self.last is not None and stamp != self.last + timedelta(seconds=1):
                self.gaps += 1
            self.digest.update(symbol.encode() + b"\0")
            self.digest.update(_ROW.pack(micros(row[1]), micros(row[2]), *row[3:]))
            self.count += 1
            self.first = self.first or stamp
            self.last = stamp

    def result(self, start, end):
        return {
            "rows": self.count,
            "logical_sha256": self.digest.hexdigest(),
            "first": self.first.isoformat() if self.first else None,
            "last": self.last.isoformat() if self.last else None,
            "gaps": self.gaps,
            "complete": (
                self.count == int((end - start).total_seconds())
                and self.first == start
                and self.last == end - timedelta(seconds=1)
                and self.gaps == 0
            ),
        }


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def archive_lock(root: Path, *, exclusive: bool = False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".archive.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_catalog(root: Path) -> list[dict]:
    path = Path(root) / "catalog.json"
    if not path.exists():
        return []
    catalog = json.loads(path.read_text())
    if (
        catalog["version"] != 1
        or catalog["market"] != "binance_spot"
        or catalog["interval"] != "1s"
    ):
        raise ValueError("unsupported archive format")
    entries = sorted(catalog["files"], key=lambda e: (e["symbol"], e["start"]))
    previous = {}
    for entry in entries:
        file_path(root, entry)  # refuse paths outside the archive
        symbol = symbol_name(entry["symbol"])
        start, end = utc(entry["start"]), utc(entry["end"])
        if start >= end or start < previous.get(symbol, start):
            raise ValueError("overlapping or invalid archive ranges")
        previous[symbol] = end
    return entries


def file_path(root: Path, entry: dict) -> Path:
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("archive path must be relative without parent traversal")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("archive path escapes root")
    return path


def _fsync_directory(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_catalog(root: Path, entries: list[dict]):
    data = {"version": 1, "market": "binance_spot", "interval": "1s", "files": entries}
    fd, name = tempfile.mkstemp(prefix=".catalog-", dir=root)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, root / "catalog.json")
        _fsync_directory(root)
    finally:
        Path(name).unlink(missing_ok=True)


def table_rows(table):
    # At most one export/read batch is converted, never a complete file.
    return zip(*(table.column(name).to_pylist() for name in COLUMNS))


def inspect_file(
    path: Path,
    symbol: str,
    start: datetime,
    end: datetime,
    batch_size: int = DEFAULT_BATCH,
) -> dict:
    fingerprint = Fingerprint()
    with pq.ParquetFile(path) as parquet:
        if parquet.schema_arrow != SCHEMA:
            raise ValueError(f"unexpected archive schema: {path}")
        for batch in parquet.iter_batches(batch_size=batch_size):
            if any(batch.column(i).null_count for i in range(batch.num_columns)):
                raise ValueError(f"null candle fields: {path}")
            fingerprint.update(table_rows(batch), symbol=symbol, start=start, end=end)
    return {
        **fingerprint.result(start, end),
        "sha256": file_sha(path),
        "bytes": path.stat().st_size,
    }


def verify_entry(root: Path, entry: dict) -> dict:
    actual = inspect_file(
        file_path(root, entry), entry["symbol"], utc(entry["start"]), utc(entry["end"])
    )
    for key, value in actual.items():
        if entry.get(key) != value:
            raise ValueError(f"archive verification failed ({key}): {entry['path']}")
    return actual


def _write_batches(root, symbol, start, end, batches, batch_size):
    folder = root / "binance_spot" / symbol / f"{start:%Y-%m}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{start:%Y%m%dT%H%M%S}-{end:%Y%m%dT%H%M%S}-{uuid4().hex}.parquet"
    temporary = path.with_suffix(".partial")
    expected = Fingerprint()
    try:
        with pq.ParquetWriter(
            temporary, SCHEMA, compression="zstd", compression_level=3
        ) as writer:
            for rows in batches:
                if isinstance(rows, (pa.Table, pa.RecordBatch)):
                    if rows.schema != SCHEMA:
                        raise ValueError("unexpected source batch schema")
                    rows = list(table_rows(rows))
                if not rows:
                    continue
                expected.update(rows, symbol=symbol, start=start, end=end)
                table = pa.Table.from_arrays(
                    [
                        pa.array(values, type=field.type)
                        for values, field in zip(zip(*rows), SCHEMA)
                    ],
                    schema=SCHEMA,
                )
                writer.write_table(table, row_group_size=batch_size)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        actual = inspect_file(temporary, symbol, start, end, batch_size)
        if any(
            actual[key] != value for key, value in expected.result(start, end).items()
        ):
            raise ValueError("Parquet read-back differs from source")
        os.replace(temporary, path)
        _fsync_directory(folder)
        return {
            "symbol": symbol,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "path": str(path.relative_to(root)),
            **actual,
        }
    finally:
        try:
            close = getattr(batches, "close", None)
            if close is not None:
                close()
        finally:
            temporary.unlink(missing_ok=True)


def _require_preserved_keys(root, previous, replacement, batch_size):
    """A backfill may correct values, but must retain every archived observation."""

    def keys(parquet):
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["open_time"]):
            yield from batch.column(0).to_pylist()

    with (
        pq.ParquetFile(file_path(root, previous)) as old_file,
        pq.ParquetFile(file_path(root, replacement)) as new_file,
    ):
        new_keys = iter(keys(new_file))
        current = next(new_keys, None)
        for stamp in keys(old_file):
            while current is not None and current < stamp:
                current = next(new_keys, None)
            if current != stamp:
                raise ValueError("source is missing candles from the existing archive")


def db_batches(conn, symbol, start, end, columns=COLUMNS, batch_size=DEFAULT_BATCH):
    if batch_size <= 0 or not columns or any(name not in COLUMNS for name in columns):
        raise ValueError("invalid batch size or columns")
    with conn.cursor(name=f"archive_{uuid4().hex}") as cur:
        cur.itersize = batch_size
        cur.execute(
            f"SELECT {', '.join(columns)} FROM klines "
            "WHERE symbol = %s AND open_time >= %s AND open_time < %s ORDER BY open_time",
            (symbol, start, end),
        )
        while rows := cur.fetchmany(batch_size):
            yield rows


def _export_batches_locked(root, symbol, start, end, batches, batch_size, catalog):
    overlap = [
        entry
        for entry in catalog
        if entry["symbol"] == symbol
        and utc(entry["start"]) < end
        and utc(entry["end"]) > start
    ]
    if any(utc(e["start"]) != start or utc(e["end"]) != end for e in overlap):
        raise ValueError(
            "range overlaps a compacted or partial archive; use its exact range"
        )
    entry = _write_batches(root, symbol, start, end, batches, batch_size)
    if overlap:
        try:
            verify_entry(root, overlap[0])
            if overlap[0]["logical_sha256"] == entry["logical_sha256"]:
                file_path(root, entry).unlink()
                return overlap[0], catalog
            if entry["rows"] < overlap[0]["rows"]:
                raise ValueError("source has fewer rows than the existing archive")
            # Counts alone miss a lost second replaced by a different one.
            _require_preserved_keys(root, overlap[0], entry, batch_size)
        except Exception:
            file_path(root, entry).unlink(missing_ok=True)
            raise
    catalog = [e for e in catalog if e not in overlap] + [entry]
    _publish_catalog(root, catalog)
    return entry, catalog


def export_batches(
    root: Path, symbol: str, start, end, batches, batch_size: int = DEFAULT_BATCH
) -> dict:
    """Publish one range from bounded row or Arrow batches without a database.

    Call once per UTC day for external daily sources. Row tuples use COLUMNS;
    Arrow batches use SCHEMA. Source provenance belongs in a separate manifest.
    """
    symbol, root = symbol_name(symbol), Path(root)
    start, end = utc(start), utc(end)
    if start >= end or start.microsecond or end.microsecond or batch_size <= 0:
        raise ValueError("invalid range or batch size")
    with archive_lock(root, exclusive=True):
        entry, _ = _export_batches_locked(
            root, symbol, start, end, batches, batch_size, read_catalog(root)
        )
    return entry


def export_range(
    conn, symbol: str, start, end, root: Path, batch_size: int = DEFAULT_BATCH
) -> list[dict]:
    """Archive UTC day slices. The supplied connection must be read-only.

    Missing seconds remain missing and are recorded. No database rows are deleted.
    Re-export replaces an identical range; overlapping partial ranges are refused.
    """
    symbol, root = symbol_name(symbol), Path(root)
    start, end = utc(start), utc(end)
    if start >= end or start.microsecond or end.microsecond or batch_size <= 0:
        raise ValueError("invalid range or batch size")
    published = []
    with archive_lock(root, exclusive=True):
        catalog = read_catalog(root)
        while start < end:
            stop = min(
                end, start.replace(hour=0, minute=0, second=0) + timedelta(days=1)
            )
            entry, catalog = _export_batches_locked(
                root,
                symbol,
                start,
                stop,
                db_batches(conn, symbol, start, stop, batch_size=batch_size),
                batch_size,
                catalog,
            )
            published.append(entry)
            start = stop
    return published


def compact_month(
    root: Path, symbol: str, month: str, batch_size=DEFAULT_BATCH
) -> dict:
    """Compact a completely covered, closed UTC month, without loading it in RAM."""
    start = utc(month + "-01")
    end = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    return compact_range(root, symbol, start, end, batch_size)


def compact_range(
    root: Path, symbol: str, start, end, batch_size=DEFAULT_BATCH
) -> dict:
    """Merge adjacent complete files; refuse any hole, overlap or corrupt input."""
    root, symbol = Path(root), symbol_name(symbol)
    start, end = utc(start), utc(end)
    if start >= end or batch_size <= 0:
        raise ValueError("invalid compaction range or batch size")
    if end > datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ):
        raise ValueError("only closed months can be compacted")
    with archive_lock(root, exclusive=True):
        catalog = read_catalog(root)
        entries = sorted(
            [
                e
                for e in catalog
                if e["symbol"] == symbol
                and utc(e["start"]) < end
                and utc(e["end"]) > start
            ],
            key=lambda e: e["start"],
        )
        cursor = start
        for entry in entries:
            verify_entry(root, entry)
            if utc(entry["start"]) != cursor or not entry["complete"]:
                raise ValueError("month is not completely archived")
            cursor = utc(entry["end"])
        if cursor != end:
            raise ValueError("month is not completely archived")
        if len(entries) == 1:
            return entries[0]

        def batches():
            for entry in entries:
                with pq.ParquetFile(file_path(root, entry)) as parquet:
                    for batch in parquet.iter_batches(batch_size=batch_size):
                        yield list(table_rows(batch))

        merged = _write_batches(root, symbol, start, end, batches(), batch_size)
        _publish_catalog(root, [e for e in catalog if e not in entries] + [merged])
        # Shared reader locks guarantee no active query still uses the old files.
        for entry in entries:
            file_path(root, entry).unlink()
        return merged


def garbage_collect(root: Path) -> int:
    """Remove only unreferenced generations under this archive's own layout."""
    root = Path(root)
    with archive_lock(root, exclusive=True):
        active = {file_path(root, e) for e in read_catalog(root)}
        stale = [
            p
            for p in (root / "binance_spot").glob("*/*/*")
            if p.suffix in {".parquet", ".partial"} and p.resolve() not in active
        ]
        for path in stale:
            path.unlink()
        return len(stale)
