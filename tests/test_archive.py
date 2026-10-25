"""Storage correctness tests, independent of the frozen ML bundles."""

from __future__ import annotations

from datetime import timedelta
import fcntl
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import archive, db, market  # noqa: E402
from scripts import prune_archive  # noqa: E402

START = archive.utc("2026-01-01")


def candles(count, start=START, symbol="BTCUSDT"):
    for i in range(count):
        stamp = start + timedelta(seconds=i)
        yield (
            symbol,
            stamp,
            stamp + timedelta(milliseconds=999),
            100.0 + i,
            102.0 + i,
            99.0 + i,
            101.0 + i,
            1.25,
            130.5,
            3,
            0.75,
            78.3,
        )


class Cursor:
    def __init__(self, connection, name):
        assert name, "raw candles must use a server-side cursor"
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.cursor_exits += 1

    def execute(self, sql, parameters):
        symbol, start, end = parameters
        self.connection.queries.append((start, end))
        self.columns = sql.split("SELECT ", 1)[1].split(" FROM", 1)[0].split(", ")
        self.rows = iter(
            row
            for row in self.connection.rows
            if row[0] == symbol and start <= row[1] < end
        )

    def fetchmany(self, size):
        from itertools import islice

        self.connection.sizes.append(size)
        indices = [archive.COLUMNS.index(name) for name in self.columns]
        return [tuple(row[i] for i in indices) for row in islice(self.rows, size)]

    def fetchall(self):
        raise AssertionError("raw fetchall is forbidden")


class Connection:
    def __init__(self, rows):
        self.rows, self.queries, self.sizes = list(rows), [], []
        self.closed = False
        self.cursor_exits = 0

    def cursor(self, name=None):
        return Cursor(self, name)

    def close(self):
        self.closed = True


def export(tmp_path, rows, start=START, seconds=120, batch=17):
    conn = Connection(rows)
    entries = archive.export_range(
        conn, "BTCUSDT", start, start + timedelta(seconds=seconds), tmp_path, batch
    )
    return conn, entries


def test_lossless_export_is_bounded_and_batch_independent(tmp_path):
    rows = list(candles(120))
    conn, entries = export(tmp_path, rows)
    entry = entries[0]
    assert max(conn.sizes) == 17
    assert entry["rows"] == 120 and entry["complete"] and entry["gaps"] == 0
    archive.verify_entry(tmp_path, entry)
    expected = pd.DataFrame(rows, columns=archive.COLUMNS)
    actual = market.load_market(
        "BTCUSDT",
        START,
        START + timedelta(seconds=120),
        archive_root=tmp_path,
        source="archive",
        batch_size=11,
    )
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False)
    _, repeated = export(tmp_path, rows, batch=31)
    assert repeated == entries
    assert len(list(tmp_path.rglob("*.parquet"))) == 1
    assert (
        pq.ParquetFile(archive.file_path(tmp_path, entry))
        .metadata.row_group(0)
        .column(0)
        .compression
        == "ZSTD"
    )


def test_gap_is_preserved_and_partial_minute_is_excluded(tmp_path):
    rows = list(candles(120))
    del rows[13]
    _, entries = export(tmp_path, rows)
    assert not entries[0]["complete"] and entries[0]["gaps"] == 1
    actual = market.load_market(
        "BTCUSDT",
        START,
        START + timedelta(seconds=120),
        timeframe="1m",
        archive_root=tmp_path,
        source="archive",
    )
    assert len(actual) == 1
    assert actual.iloc[0].open_time == START + timedelta(minutes=1)
    assert actual.iloc[0].volume == 60 * 1.25
    assert actual.iloc[0].taker_buy_base == 60 * 0.75


def test_auto_reads_only_missing_ranges_and_complete_archive_needs_no_db(
    tmp_path, monkeypatch
):
    export(tmp_path, candles(60), seconds=60)
    tail = Connection(candles(180))
    frame = market.load_market(
        "BTCUSDT",
        START,
        START + timedelta(seconds=180),
        archive_root=tmp_path,
        conn=tail,
        columns=["open_time", "close"],
    )
    assert tail.queries == [
        (START + timedelta(seconds=60), START + timedelta(seconds=180))
    ]
    assert len(frame) == 180 and frame.open_time.is_unique
    monkeypatch.setattr(
        market,
        "connect",
        lambda **_: pytest.fail("archive-only coverage must not open DB"),
    )
    assert (
        len(
            market.load_market(
                "BTCUSDT", START, START + timedelta(seconds=60), archive_root=tmp_path
            )
        )
        == 60
    )


def test_incomplete_archive_merges_db_without_duplicates_and_db_wins(tmp_path):
    raw = list(candles(120))
    export(tmp_path, raw[:20] + raw[21:])
    changed = [*raw[5]]
    changed[6] = 999.0
    raw[5] = tuple(changed)
    actual = market.load_market(
        "BTCUSDT",
        START,
        START + timedelta(seconds=120),
        archive_root=tmp_path,
        conn=Connection(raw),
    )
    assert len(actual) == 120 and actual.open_time.is_unique
    assert actual.iloc[5].close == 999.0


def test_aggregates_are_causal_and_include_all_flow_fields(tmp_path):
    export(tmp_path, candles(180), seconds=180)
    frame = market.load_market(
        "BTCUSDT",
        START,
        START + timedelta(seconds=119),
        archive_root=tmp_path,
        source="archive",
        timeframe="1m",
    )
    assert len(frame) == 1  # never include the unfinished second minute
    row = frame.iloc[0]
    assert row.open == 100 and row.close == 160 and row.high == 161 and row.low == 99
    assert row.trades == 180 and row.taker_buy_quote == pytest.approx(60 * 78.3)
    projected = market.load_market(
        "BTCUSDT",
        START,
        START + timedelta(seconds=60),
        archive_root=tmp_path,
        source="archive",
        columns=["volume"],
    )
    assert list(projected.columns) == ["volume"]


def test_eager_limit_and_streaming_batches(tmp_path):
    export(tmp_path, candles(120))
    kwargs = {"archive_root": tmp_path, "source": "archive", "batch_size": 13}
    with pytest.raises(MemoryError, match="max_rows"):
        market.load_market(
            "BTCUSDT", START, START + timedelta(seconds=120), max_rows=100, **kwargs
        )
    batches = list(
        market.iter_market("BTCUSDT", START, START + timedelta(seconds=120), **kwargs)
    )
    assert sum(b.num_rows for b in batches) == 120
    assert max(b.num_rows for b in batches) <= 13


def test_compaction_matches_original_and_gc_keeps_active_files(tmp_path):
    rows = list(candles(120))
    _, first = export(tmp_path, rows[:60], seconds=60)
    _, second = export(
        tmp_path, rows[60:], start=START + timedelta(seconds=60), seconds=60
    )
    merged = archive.compact_range(
        tmp_path, "BTCUSDT", START, START + timedelta(seconds=120), batch_size=19
    )
    assert merged["rows"] == 120 and merged["complete"]
    assert not archive.file_path(tmp_path, first[0]).exists()
    assert not archive.file_path(tmp_path, second[0]).exists()
    archive.verify_entry(tmp_path, merged)
    orphan = archive.file_path(tmp_path, merged).with_name("orphan.partial")
    orphan.write_bytes(b"interrupted write")
    assert archive.garbage_collect(tmp_path) == 1
    assert archive.file_path(tmp_path, merged).exists()


def test_failed_publication_preserves_previous_catalog(tmp_path, monkeypatch):
    rows = list(candles(120))
    export(tmp_path, rows)
    original = (tmp_path / "catalog.json").read_bytes()
    changed = list(rows[3])
    changed[6] += 0.1
    rows[3] = tuple(changed)

    def fail(*_):
        raise OSError("simulated interrupted publication")

    monkeypatch.setattr(archive, "_publish_catalog", fail)
    with pytest.raises(OSError):
        export(tmp_path, rows)
    assert (tmp_path / "catalog.json").read_bytes() == original
    for entry in archive.read_catalog(tmp_path):
        archive.verify_entry(tmp_path, entry)
    assert archive.garbage_collect(tmp_path) == 1


def test_duplicates_wrong_interval_and_partial_overlap_are_refused(tmp_path):
    rows = list(candles(120))
    with pytest.raises(ValueError, match="duplicate"):
        export(tmp_path, rows[:10] + [rows[9]] + rows[10:])
    assert not (tmp_path / "catalog.json").exists()
    broken = list(rows[0])
    broken[2] += timedelta(minutes=1)
    with pytest.raises(ValueError, match="1-second"):
        export(tmp_path, [tuple(broken), *rows[1:]])
    export(tmp_path, rows)
    with pytest.raises(ValueError, match="overlap"):
        export(tmp_path, rows, seconds=60)
    with pytest.raises(ValueError, match="fewer rows"):
        export(tmp_path, rows[:60])


def test_checksum_corruption_and_catalog_overlap_are_refused(tmp_path):
    _, entries = export(tmp_path, candles(120))
    path = archive.file_path(tmp_path, entries[0])
    data = bytearray(path.read_bytes())
    data[20] ^= 1
    path.write_bytes(data)
    with pytest.raises(Exception):
        archive.verify_entry(tmp_path, entries[0])
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    catalog["files"].append(catalog["files"][0])
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="overlapping"):
        archive.read_catalog(tmp_path)


def test_legacy_loader_routes_to_archive_and_guards_unbounded_reads(
    tmp_path, monkeypatch
):
    export(tmp_path, candles(120))
    monkeypatch.setenv("ARCHIVE_ROOT", str(tmp_path))
    frame = db.load_klines(
        "BTCUSDT", START.isoformat(), (START + timedelta(seconds=120)).isoformat()
    )
    assert len(frame) == 120
    with pytest.raises(ValueError, match="explicit start/end"):
        db.load_klines("BTCUSDT")


def test_prune_refuses_overlapping_backup_and_short_hot_window(tmp_path):
    with pytest.raises(ValueError, match="separate archive"):
        prune_archive.prune(tmp_path, tmp_path / "backup", apply=True)
    with pytest.raises(ValueError, match="7 hot days"):
        prune_archive.prune(tmp_path, tmp_path.parent / "other", hot_days=1, apply=True)


def test_missing_backup_blocks_prune_before_drop(tmp_path, monkeypatch):
    root, backup = tmp_path / "archive", tmp_path / "backup"
    export(root, candles(120))
    shutil.copytree(root, backup)
    (backup / "catalog.json").unlink()
    statements = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def execute(self, sql, params=None):
            statements.append(sql)

        def fetchall(self):
            sql = statements[-1]
            if "range_start" in sql:
                return [(START, START + timedelta(seconds=120))]
            if "DISTINCT symbol" in sql:
                return [("BTCUSDT",)]
            return []

    class FakeDB:
        def cursor(self, **_):
            return FakeCursor()

        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(prune_archive, "connect", lambda **_: FakeDB())
    with pytest.raises(ValueError, match="backup"):
        prune_archive.prune(root, backup, apply=True)
    assert not any("drop_chunks" in sql for sql in statements)


@pytest.mark.parametrize("extra_backfill", [False, True])
def test_backfill_cannot_exchange_archived_seconds_for_different_seconds(
    tmp_path, extra_backfill
):
    rows = list(candles(120))
    original = rows[:20] + rows[22:]
    _, entries = export(tmp_path, original)
    changed = rows[:5] + rows[6:21] + rows[22:]
    if extra_backfill:
        changed = rows[:5] + rows[6:]
    assert len(changed) >= len(original)
    with pytest.raises(ValueError, match="missing candles"):
        export(tmp_path, changed)
    assert archive.read_catalog(tmp_path) == entries
    assert len(list(tmp_path.rglob("*.parquet"))) == 1
    archive.verify_entry(tmp_path, entries[0])


def test_backfill_adds_missing_seconds_and_permits_price_corrections(tmp_path):
    rows = list(candles(120))
    export(tmp_path, rows[:20] + rows[21:])
    corrected = list(rows[5])
    corrected[6] += 1
    rows[5] = tuple(corrected)
    _, entries = export(tmp_path, rows)
    assert entries[0]["complete"]
    frame = market.load_market(
        "BTCUSDT", START, START + timedelta(seconds=120),
        archive_root=tmp_path, source="archive",
    )
    assert len(frame) == 120 and frame.iloc[5].close == corrected[6]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), None])
def test_nonfinite_source_is_refused_and_cursor_and_partial_file_are_closed(tmp_path, value):
    rows = list(candles(3))
    broken = list(rows[1])
    broken[6] = value
    rows[1] = tuple(broken)
    conn = Connection(rows)
    with pytest.raises(ValueError, match="finite"):
        archive.export_range(conn, "BTCUSDT", START, START + timedelta(seconds=3), tmp_path)
    assert conn.cursor_exits == 1
    assert not (tmp_path / "catalog.json").exists()
    assert not list(tmp_path.rglob("*.partial"))


@pytest.mark.parametrize("damage", ["missing", "truncated"])
def test_damaged_complete_archive_never_silently_falls_back_to_database(tmp_path, damage):
    _, entries = export(tmp_path, candles(120))
    path = archive.file_path(tmp_path, entries[0])
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes()[:-8])
    connection = Connection(candles(120))
    with pytest.raises(ValueError, match="archive missing or changed"):
        market.load_market(
            "BTCUSDT", START, START + timedelta(seconds=120),
            archive_root=tmp_path, conn=connection,
        )
    assert connection.queries == []


@pytest.mark.parametrize("source", ["archive", "database"])
def test_fractional_end_cannot_expose_unclosed_candle_or_minute(tmp_path, source):
    rows = list(candles(60))
    if source == "archive":
        export(tmp_path, rows, seconds=60)
    options = {"archive_root": tmp_path, "source": source, "conn": Connection(rows)}
    end = START + timedelta(seconds=59, milliseconds=500)
    frame = market.load_market("BTCUSDT", START, end, **options)
    assert len(frame) == 59 and all(frame.close_time < end)
    assert market.load_market("BTCUSDT", START, end, timeframe="1m", **options).empty


@pytest.mark.parametrize("field,value", [
    ("close", float("nan")),
    ("volume", None),
    ("close_time", START + timedelta(minutes=1)),
    ("open_time", START + timedelta(milliseconds=1)),
])
def test_invalid_db_inputs_fail_before_aggregation_and_close_resources(
    tmp_path, monkeypatch, field, value
):
    rows = list(candles(60))
    broken = list(rows[0])
    broken[archive.COLUMNS.index(field)] = value
    rows[0] = tuple(broken)
    connection = Connection(rows)
    monkeypatch.setattr(market, "connect", lambda **_: connection)
    monkeypatch.setenv("MARKET_TEMP_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        market.load_market(
            "BTCUSDT", START, START + timedelta(seconds=60),
            source="database", archive_root=tmp_path, timeframe="1m",
        )
    assert connection.closed and connection.cursor_exits == 1
    assert not list(tmp_path.glob("klinehouse-*"))


def test_early_generator_close_releases_archive_lock_and_spool(tmp_path, monkeypatch):
    connection = Connection(candles(120))
    monkeypatch.setattr(market, "connect", lambda **_: connection)
    monkeypatch.setenv("MARKET_TEMP_DIR", str(tmp_path))
    iterator = market.iter_market(
        "BTCUSDT", START, START + timedelta(seconds=120),
        archive_root=tmp_path, batch_size=7,
    )
    assert next(iterator).num_rows == 7
    assert list(tmp_path.glob("klinehouse-*/tail.parquet"))
    assert connection.closed and connection.cursor_exits == 1
    with (tmp_path / ".archive.lock").open("a+b") as lock:
        with pytest.raises(BlockingIOError):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        iterator.close()
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
    assert not list(tmp_path.glob("klinehouse-*"))


def test_public_arrow_batch_import_matches_database_export(tmp_path):
    rows = list(candles(120))
    table = pa.Table.from_arrays(
        [pa.array(values, type=field.type) for values, field in zip(zip(*rows), archive.SCHEMA)],
        schema=archive.SCHEMA,
    )
    imported = archive.export_batches(
        tmp_path / "imported", "BTCUSDT", START, START + timedelta(seconds=120),
        table.to_batches(max_chunksize=19), batch_size=19,
    )
    _, exported = export(tmp_path / "database", rows)
    assert imported["logical_sha256"] == exported[0]["logical_sha256"]
    assert imported["complete"]
    archive.verify_entry(tmp_path / "imported", imported)


def test_interrupted_stream_preserves_catalog_and_removes_partial_file(tmp_path):
    _, original = export(tmp_path, candles(60), seconds=60)
    closed = []

    def interrupted():
        try:
            yield list(candles(5, start=START + timedelta(seconds=60)))
            raise OSError("source stream interrupted")
        finally:
            closed.append(True)

    with pytest.raises(OSError, match="interrupted"):
        archive.export_batches(
            tmp_path, "BTCUSDT", START + timedelta(seconds=60),
            START + timedelta(seconds=120), interrupted(),
        )
    assert closed == [True]
    assert archive.read_catalog(tmp_path) == original
    assert not list(tmp_path.rglob("*.partial"))
    assert len(list(tmp_path.rglob("*.parquet"))) == 1
    archive.verify_entry(tmp_path, original[0])


@pytest.mark.parametrize("absolute", [False, True])
def test_catalog_paths_remain_safe_when_rebased_to_a_backup(tmp_path, absolute):
    _, entries = export(tmp_path, candles(3), seconds=3)
    entry = entries[0]
    # These resolve inside the source, but escape a differently named backup.
    entry["path"] = (
        str(tmp_path / entry["path"]) if absolute
        else f"../{tmp_path.name}/{entry['path']}"
    )
    with pytest.raises(ValueError, match="relative without parent traversal"):
        archive.file_path(tmp_path, entry)


def test_parallel_multifile_reader_preserves_global_order_across_arrow_batches(tmp_path):
    # More than one DuckDB scan task is necessary to expose parallel Arrow
    # vector reordering; the small unit fixtures cannot exercise this path.
    count = 2 * 86400
    export(tmp_path, candles(count), seconds=count, batch=10000)
    columns = ["open_time", "close_time", "close", "volume", "trades", "taker_buy_base"]
    end = START + timedelta(seconds=count)
    position = 0
    iterator = market.iter_market(
        "BTCUSDT", START, end, archive_root=tmp_path, source="archive",
        columns=columns, batch_size=10000,
    )
    try:
        for batch in iterator:
            expected = range(position, position + batch.num_rows)
            assert batch.column("open_time").to_pylist() == [
                START + timedelta(seconds=i) for i in expected
            ]
            assert batch.column("close").to_pylist() == [101 + i for i in expected]
            position += batch.num_rows
    finally:
        iterator.close()
    assert position == count
    minutes = market.load_market(
        "BTCUSDT", START, end, archive_root=tmp_path, source="archive",
        columns=columns, timeframe="1m", batch_size=1000,
    )
    assert len(minutes) == count // 60
    assert minutes.open_time.tolist() == [
        START + timedelta(minutes=i) for i in range(count // 60)
    ]
    assert minutes.close.tolist() == [160 + i * 60 for i in range(count // 60)]
    assert (minutes.volume == 75).all() and (minutes.trades == 180).all()
