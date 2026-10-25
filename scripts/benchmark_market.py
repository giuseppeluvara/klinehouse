#!/usr/bin/env python3
"""Measure real archive queries in separate processes without collecting rows.

Each sample uses a fresh Python process. The operating-system page cache is not
cleared, so first/repeat samples are not guaranteed cold/warm-cache measurements.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import importlib.metadata
import json
import platform
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import archive  # noqa: E402


def measure_query(scenario):
    from research.market import iter_market
    import pyarrow as pa

    began = time.perf_counter()
    rows = batches = max_rows = max_bytes = 0
    previous_timestamp = None
    start_us = archive.micros(archive.utc(scenario["start"]))
    end_us = archive.micros(archive.utc(scenario["end"]))
    for batch in iter_market(
        scenario["symbol"],
        scenario["start"],
        scenario["end"],
        archive_root=scenario["archive_root"],
        source="archive",
        timeframe=scenario["timeframe"],
        columns=scenario["columns"],
        batch_size=scenario["batch_size"],
    ):
        timestamps = batch.column("open_time")
        if timestamps.null_count:
            raise ValueError("reader returned null open timestamps")
        values = (
            timestamps.cast(pa.timestamp("us", tz="UTC"))
            .cast(pa.int64())
            .to_numpy(zero_copy_only=False)
        )
        if len(values):
            if (
                int(values[0]) < start_us
                or int(values[-1]) >= end_us
                or (
                    previous_timestamp is not None
                    and int(values[0]) <= previous_timestamp
                )
                or (values[1:] <= values[:-1]).any()
            ):
                raise ValueError(
                    "reader returned unordered, duplicate or out-of-range timestamps"
                )
            previous_timestamp = int(values[-1])
        rows += batch.num_rows
        batches += 1
        max_rows = max(max_rows, batch.num_rows)
        max_bytes = max(max_bytes, batch.nbytes)
    elapsed = time.perf_counter() - began
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        **scenario,
        "rows": rows,
        "batches": batches,
        "max_batch_rows": max_rows,
        "max_arrow_batch_bytes": max_bytes,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": rss * (1 if sys.platform == "darwin" else 1024),
        "strict_timestamp_order_verified": True,
    }


def benchmark(root, symbol, start, end, *, batch_size=8192, repeats=2):
    root = Path(root).resolve()
    start, end = archive.utc(start), archive.utc(end)
    if start >= end or batch_size <= 0 or repeats < 1:
        raise ValueError("invalid range, batch size or repeats")
    common = {
        "symbol": archive.symbol_name(symbol),
        "archive_root": str(root),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "batch_size": batch_size,
    }
    scenarios = [
        {
            **common,
            "name": "one_day_raw_all_columns",
            "end": min(start + timedelta(days=1), end).isoformat(),
            "timeframe": "1s",
            "columns": None,
        },
        {
            **common,
            "name": "full_period_raw_two_columns",
            "timeframe": "1s",
            "columns": ["open_time", "close"],
        },
        {
            **common,
            "name": "full_period_minute_six_columns",
            "timeframe": "1m",
            "columns": [
                "open_time",
                "close",
                "volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
            ],
        },
    ]
    samples = []
    for scenario in scenarios:
        for repeat in range(repeats):
            sample = json.loads(
                subprocess.check_output(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        json.dumps(scenario),
                    ],
                    text=True,
                )
            )
            sample["repeat"] = repeat + 1
            samples.append(sample)
            print(json.dumps(sample), file=sys.stderr, flush=True)
    entries = archive.read_catalog(root)
    source_root = Path(__file__).resolve().parents[1]
    return {
        "version": 1,
        "source": "archive",
        "synthetic": False,
        "platform": platform.platform(),
        "python": sys.version,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("duckdb", "pyarrow", "pandas", "numpy")
        },
        "source_code_sha256": {
            name: archive.file_sha(source_root / name)
            for name in (
                "research/archive.py",
                "research/market.py",
                "scripts/benchmark_market.py",
            )
        },
        "archive_catalog_sha256": archive.file_sha(root / "catalog.json"),
        "measurement": "Fresh process per sample; elapsed query only, peak RSS includes interpreter/libraries. "
        "Operating-system page cache is not cleared; repeats do not prove cold/warm cache performance.",
        "row_consumption": "Arrow batches counted and discarded; no complete DataFrame retained",
        "archive_total_bytes": sum(item["bytes"] for item in entries),
        "archive_total_files": len(entries),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default="2026-07-08")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(measure_query(json.loads(args.worker))))
        return
    if args.archive_root is None or args.output is None:
        parser.error("--archive-root and --output are required")
    if args.output.exists():
        parser.error(
            "output already exists; choose a new filename to preserve previous measurements"
        )
    result = benchmark(
        args.archive_root,
        args.symbol,
        args.start,
        args.end,
        batch_size=args.batch_size,
        repeats=args.repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
