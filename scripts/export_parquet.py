#!/usr/bin/env python3
"""Export closed UTC day slices to the verified, lossless 1s archive."""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def export(symbol: str, start: str, end: str, out_dir: Path, batch_size=50_000):
    from research.archive import export_range, utc
    from research.db import connect

    if utc(end) > datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ):
        raise ValueError(
            "export only closed UTC days; keep today's candles in the live database"
        )
    with closing(connect(read_only=True)) as conn:
        entries = export_range(conn, symbol, start, end, out_dir, batch_size)
    for entry in entries:
        print(
            f"{entry['symbol']} {entry['start']}: {entry['rows']:,} rows "
            f"({entry['bytes']:,} bytes), complete={entry['complete']}"
        )
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="inclusive UTC start")
    parser.add_argument("--end", required=True, help="exclusive UTC end")
    parser.add_argument("--out", default="archive")
    parser.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args()
    export(args.symbol, args.start, args.end, Path(args.out), args.batch_size)


if __name__ == "__main__":
    main()
