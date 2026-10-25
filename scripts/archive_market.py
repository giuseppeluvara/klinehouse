#!/usr/bin/env python3
"""Sync, verify and compact the 1s archive. Never deletes database rows."""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import signal
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.archive import (  # noqa: E402
    archive_lock,
    compact_month,
    export_range,
    file_path,
    garbage_collect,
    read_catalog,
    utc,
    verify_entry,
)
from research.db import _load_dotenv, connect  # noqa: E402


def sync(root, *, symbols=None, start=None, end=None, batch_size=50_000):
    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end = utc(end) if end else today
    if end > today or end != end.replace(hour=0, minute=0, second=0, microsecond=0):
        raise ValueError("sync end must be a closed UTC day boundary")
    if start and utc(start) != utc(start).replace(
        hour=0, minute=0, second=0, microsecond=0
    ):
        raise ValueError("sync start must be a UTC day boundary")
    if start and utc(start) >= end:
        raise ValueError("start must precede end")
    with closing(connect(read_only=True)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol, min(open_time) FROM klines GROUP BY symbol ORDER BY symbol"
            )
            bounds = cur.fetchall()  # one row per symbol, never the raw candles
        available = {symbol for symbol, _ in bounds}
        if symbols and set(symbols) - available:
            raise ValueError(
                f"symbols absent from database: {set(symbols) - available}"
            )
        for symbol, first in bounds:
            if symbols and symbol not in symbols:
                continue
            cursor = (
                utc(start)
                if start
                else utc(first).replace(hour=0, minute=0, second=0, microsecond=0)
            )
            while cursor < end:
                stop = cursor + timedelta(days=1)
                with archive_lock(root):
                    covered = next(
                        (
                            e
                            for e in read_catalog(root)
                            if e["symbol"] == symbol
                            and utc(e["start"]) <= cursor
                            and utc(e["end"]) >= stop
                            and e["complete"]
                        ),
                        None,
                    )
                    if covered:
                        path = file_path(root, covered)
                        if not path.exists() or path.stat().st_size != covered["bytes"]:
                            raise ValueError(
                                "archive changed; run verify before syncing"
                            )
                if not covered:
                    entries = export_range(conn, symbol, cursor, stop, root, batch_size)
                    print(
                        f"archived {symbol} {cursor.date()}: {entries[0]['rows']:,} rows",
                        flush=True,
                    )
                conn.rollback()  # release each read transaction, including skipped days
                cursor = stop


def main():
    env = {**_load_dotenv(), **os.environ}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(env.get("ARCHIVE_ROOT", "archive"))
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sync_parser = sub.add_parser(
        "sync", help="archive all closed days still present in the DB"
    )
    sync_parser.add_argument(
        "--symbols", help="comma-separated symbols; default: all DB symbols"
    )
    sync_parser.add_argument("--start")
    sync_parser.add_argument("--end")
    sync_parser.add_argument("--batch-size", type=int, default=50_000)
    sync_parser.add_argument("--loop", action="store_true")
    sync_parser.add_argument("--poll-seconds", type=float, default=3600)
    sub.add_parser("verify", help="streaming full checksum and candle validation")
    comp = sub.add_parser("compact", help="merge a complete closed UTC month")
    comp.add_argument("--symbol", required=True)
    comp.add_argument("--month", required=True, help="YYYY-MM")
    sub.add_parser(
        "gc", help="remove orphan archive generations, never live files or DB rows"
    )
    args = parser.parse_args()
    if args.command == "verify":
        with archive_lock(args.root):
            for entry in read_catalog(args.root):
                verify_entry(args.root, entry)
                print(f"verified {entry['path']}")
    elif args.command == "compact":
        print(compact_month(args.root, args.symbol, args.month))
    elif args.command == "gc":
        print(f"removed {garbage_collect(args.root)} orphan files")
    else:
        if args.poll_seconds <= 0 or args.batch_size <= 0:
            parser.error("poll interval and batch size must be positive")
        stop = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())
        while not stop.is_set():
            try:
                sync(
                    args.root,
                    symbols=args.symbols.upper().split(",") if args.symbols else None,
                    start=args.start,
                    end=args.end,
                    batch_size=args.batch_size,
                )
            except Exception as exc:
                if not args.loop:
                    raise
                print(
                    f"archive sync failed; database untouched: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            if not args.loop or stop.wait(args.poll_seconds):
                break


if __name__ == "__main__":
    main()
