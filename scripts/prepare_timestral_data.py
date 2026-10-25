#!/usr/bin/env python3
"""Acquire verified Binance Spot one-second candles without a live database.

ZIP inputs are retained; CSV members are streamed directly to the archive. Existing
inputs and provenance are immutable. Interrupted runs reuse only verified inputs.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import zipfile

import pyarrow as pa
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import archive  # noqa: E402

BASE_URL = "https://data.binance.vision/data/spot/daily/klines"
DOCUMENTATION = "https://github.com/binance/binance-public-data"
MICROSECOND_CUTOFF = archive.utc("2025-01-01")
MAX_CSV_BYTES = 64 * 1024 * 1024


def day_range(start, end):
    start, end = archive.utc(start), archive.utc(end)
    if start >= end or any(
        (v.hour, v.minute, v.second, v.microsecond) != (0, 0, 0, 0)
        for v in (start, end)
    ):
        raise ValueError("start and exclusive end must be ordered UTC midnights")
    while start < end:
        yield start
        start += timedelta(days=1)


def source_url(symbol, day):
    symbol = archive.symbol_name(symbol)
    name = f"{symbol}-1s-{day:%Y-%m-%d}.zip"
    return f"{BASE_URL}/{symbol}/1s/{name}"


def request(session, method, url, **kwargs):
    response = session.request(method, url, timeout=(15, 90), **kwargs)
    response.raise_for_status()
    if not response.url.startswith("https://data.binance.vision/"):
        response.close()
        raise ValueError("unexpected download origin")
    return response


def download_plan(session, symbol, start, end):
    files = []
    for day in day_range(start, end):
        url = source_url(symbol, day)
        with request(session, "HEAD", url) as response:
            size = int(response.headers["Content-Length"])
            if size <= 0:
                raise ValueError("invalid source content length")
            files.append({"day": day.isoformat(), "url": url, "bytes": size})
    download_bytes = sum(item["bytes"] for item in files)
    # Conservative bound for retained ZIPs, Parquet and one in-flight ZIP. CSV is
    # never extracted. Arrow/Python process memory is separate from disk storage.
    estimated_increment = (
        download_bytes + len(files) * 86400 * 240 + max(item["bytes"] for item in files)
    )
    return {
        "files": files,
        "download_bytes": download_bytes,
        "estimated_increment_bytes": estimated_increment,
    }


def parse_checksum(text, filename):
    lines = text.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("checksum must contain one file")
    match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?([^\s]+)", lines[0])
    if not match or match.group(2) != filename:
        raise ValueError("checksum filename or digest is invalid")
    return match.group(1).lower()


def publish_new(temp_path, destination):
    """Publish on the same filesystem without replacing an existing path."""
    os.link(temp_path, destination)
    fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new_bytes(destination, data):
    destination = Path(destination)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        publish_new(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def acquire_zip(session, item, downloads):
    url = item["url"]
    destination = downloads / url.rsplit("/", 1)[-1]
    checksum_path = destination.with_suffix(".zip.CHECKSUM")
    if checksum_path.exists():
        if checksum_path.is_symlink() or checksum_path.stat().st_size > 4096:
            raise ValueError("invalid existing checksum file")
        checksum_text = checksum_path.read_text(encoding="ascii")
    else:
        with request(session, "GET", url + ".CHECKSUM", stream=True) as response:
            content = response.raw.read(4097)
        if len(content) > 4096:
            raise ValueError("checksum response is too large")
        checksum_text = content.decode("ascii")
        parse_checksum(checksum_text, destination.name)
        write_new_bytes(checksum_path, content)
    expected = parse_checksum(checksum_text, destination.name)
    if destination.exists():
        if destination.is_symlink() or archive.file_sha(destination) != expected:
            raise ValueError(
                f"existing ZIP does not match its saved checksum: {destination}"
            )
    else:
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}-", dir=downloads)
        try:
            digest, size = hashlib.sha256(), 0
            with os.fdopen(fd, "wb") as stream:
                with request(session, "GET", url, stream=True) as response:
                    for block in response.iter_content(chunk_size=1 << 20):
                        size += len(block)
                        if size > item["bytes"]:
                            raise ValueError("download exceeded HEAD content length")
                        digest.update(block)
                        stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if size != item["bytes"] or digest.hexdigest() != expected:
                raise ValueError("download size or SHA-256 verification failed")
            publish_new(name, destination)
        finally:
            Path(name).unlink(missing_ok=True)
    return destination, expected


def parse_candle(fields, symbol, day):
    if len(fields) != 12:
        raise ValueError("Binance Spot klines must have exactly 12 columns")
    unit = 1 if day >= MICROSECOND_CUTOFF else 1000
    open_us, close_us = int(fields[0]) * unit, int(fields[6]) * unit
    stamp = archive.EPOCH + timedelta(microseconds=open_us)
    close = archive.EPOCH + timedelta(microseconds=close_us)
    if not day <= stamp < day + timedelta(days=1) or stamp.microsecond:
        raise ValueError("candle is not aligned to its source UTC day")
    if not stamp <= close < stamp + timedelta(seconds=1):
        raise ValueError("source candle does not have a one-second duration")
    values = [float(fields[i]) for i in (1, 2, 3, 4, 5, 7, 9, 10)]
    trades = int(fields[8])
    if (
        any(not math.isfinite(value) or value < 0 for value in values)
        or not 0 <= trades <= 2**31 - 1
    ):
        raise ValueError("invalid finite nonnegative market values or trade count")
    if any(value <= 0 for value in values[:4]):
        raise ValueError("OHLC prices must be positive")
    op, high, low, cl, volume, quote, taker_base, taker_quote = values
    if low > min(op, cl) or high < max(op, cl) or low > high:
        raise ValueError("invalid OHLC ordering")
    # Binance's reserved last field has no archival counterpart. Refuse unknown
    # content rather than silently dropping information introduced by a schema change.
    if float(fields[11]) != 0:
        raise ValueError("reserved Binance ignore field changed")
    return (
        symbol,
        stamp,
        close,
        op,
        high,
        low,
        cl,
        volume,
        quote,
        trades,
        taker_base,
        taker_quote,
    )


def iter_zip_batches(path, symbol, day, batch_size=8192):
    if batch_size < 1 or batch_size > 50000:
        raise ValueError("batch_size must be between 1 and 50000")
    symbol, day = archive.symbol_name(symbol), archive.utc(day)
    with zipfile.ZipFile(path) as bundle:
        members = bundle.infolist()
        expected_member = Path(path).with_suffix(".csv").name
        if (
            len(members) != 1
            or members[0].filename != expected_member
            or members[0].file_size > MAX_CSV_BYTES
        ):
            raise ValueError("ZIP must contain exactly its expected bounded CSV member")
        columns = [[] for _ in archive.COLUMNS]
        last, count = None, 0
        with (
            bundle.open(members[0]) as binary,
            io.TextIOWrapper(binary, encoding="utf-8", newline="") as text,
        ):
            for fields in csv.reader(text):
                row = parse_candle(fields, symbol, day)
                if last is not None and row[1] <= last:
                    raise ValueError("duplicate or unordered source candle")
                last, count = row[1], count + 1
                if count > 86400:
                    raise ValueError("too many candles for one UTC day")
                for target, value in zip(columns, row):
                    target.append(value)
                if len(columns[0]) == batch_size:
                    yield pa.RecordBatch.from_arrays(
                        [
                            pa.array(v, type=f.type)
                            for v, f in zip(columns, archive.SCHEMA)
                        ],
                        schema=archive.SCHEMA,
                    )
                    columns = [[] for _ in archive.COLUMNS]
        if columns[0]:
            yield pa.RecordBatch.from_arrays(
                [pa.array(v, type=f.type) for v, f in zip(columns, archive.SCHEMA)],
                schema=archive.SCHEMA,
            )
        if count == 0:
            raise ValueError("source CSV is empty")


def verify_existing(output, symbol, start, end):
    path = output / "provenance.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    expected = ("binance_public_data", symbol, start.isoformat(), end.isoformat())
    actual = tuple(manifest[key] for key in ("source", "symbol", "start", "end"))
    if actual != expected:
        raise ValueError("output already contains a different immutable dataset")
    for item in manifest["files"]:
        source = output / "downloads" / item["filename"]
        if source.parent.resolve() != (output / "downloads").resolve():
            raise ValueError("invalid saved source path")
        if archive.file_sha(source) != item["sha256"]:
            raise ValueError("retained source ZIP changed")
        archive.verify_entry(output / "archive", item["archive_entry"])
    catalog = archive.read_catalog(output / "archive")
    if catalog != sorted(
        [i["archive_entry"] for i in manifest["files"]],
        key=lambda e: (e["symbol"], e["start"]),
    ):
        raise ValueError("archive catalog differs from immutable dataset provenance")
    return manifest


def prepare(
    output,
    symbol,
    start,
    end,
    *,
    batch_size=8192,
    max_download_mib=512,
    min_free_gib=40,
    plan_only=False,
):
    output, symbol = Path(output), archive.symbol_name(symbol)
    start, end = archive.utc(start), archive.utc(end)
    list(day_range(start, end))  # Validate before any writes or network request.
    if output.exists() and not plan_only:
        existing = verify_existing(output, symbol, start, end)
        if existing is not None:
            return existing
    with requests.Session() as session:
        plan = download_plan(session, symbol, start, end)
        if plan["download_bytes"] > max_download_mib * 1024**2:
            raise ValueError("download exceeds explicit size budget")
        parent = output.resolve()
        while not parent.exists():
            parent = parent.parent
        free = shutil.disk_usage(parent).free
        plan["free_before_bytes"] = free
        if plan_only:
            return plan
        if free - plan["estimated_increment_bytes"] < min_free_gib * 1024**3:
            raise ValueError(
                "insufficient free space for planned increment and free-space target"
            )
        output.mkdir(parents=True, exist_ok=True)
        downloads = output / "downloads"
        downloads.mkdir(exist_ok=True)
        files = []
        began = time.perf_counter()
        for item in plan["files"]:
            day = archive.utc(item["day"])
            before = time.perf_counter()
            source, checksum = acquire_zip(session, item, downloads)
            downloaded = time.perf_counter()
            entry = archive.export_batches(
                root=output / "archive",
                symbol=symbol,
                start=day,
                end=day + timedelta(days=1),
                batches=iter_zip_batches(source, symbol, day, batch_size),
                batch_size=batch_size,
            )
            archive.verify_entry(output / "archive", entry)
            files.append(
                {
                    "day": day.isoformat(),
                    "filename": source.name,
                    "url": item["url"],
                    "checksum_url": item["url"] + ".CHECKSUM",
                    "sha256": checksum,
                    "download_bytes": source.stat().st_size,
                    "archive_entry": entry,
                    "download_seconds": downloaded - before,
                    "archive_verify_seconds": time.perf_counter() - downloaded,
                }
            )
            print(
                json.dumps(
                    {
                        "day": item["day"],
                        "rows": entry["rows"],
                        "complete": entry["complete"],
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
        manifest = {
            "version": 1,
            "source": "binance_public_data",
            "market": "binance_spot",
            "interval": "1s",
            "symbol": symbol,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "documentation": DOCUMENTATION,
            "timestamp_rule": "Spot timestamps use microseconds from 2025-01-01; older files use milliseconds",
            "numeric_storage": "float64, matching the live database archive schema; reserved zero ignore field omitted",
            "local_database_verified": False,
            "batch_size": batch_size,
            "rows": sum(item["archive_entry"]["rows"] for item in files),
            "complete": all(item["archive_entry"]["complete"] for item in files),
            "files": files,
            "plan": plan,
            "elapsed_seconds": time.perf_counter() - began,
            "free_after_bytes": shutil.disk_usage(output).free,
        }
        write_new_bytes(
            output / "provenance.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("research/out/timestral-pilot-20261025")
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default="2026-07-08", help="Exclusive UTC date")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-download-mib", type=float, default=512)
    parser.add_argument("--min-free-gib", type=float, default=40)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Only inspect remote sizes and free disk space",
    )
    args = parser.parse_args()
    result = prepare(
        args.output,
        args.symbol,
        args.start,
        args.end,
        batch_size=args.batch_size,
        max_download_mib=args.max_download_mib,
        min_free_gib=args.min_free_gib,
        plan_only=args.plan,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
