#!/usr/bin/env python3
"""Opt-in, backed-up and verified Timescale chunk pruning. Dry-run by default."""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb  # noqa: E402

from research.archive import (  # noqa: E402
    COLUMNS,
    Fingerprint,
    archive_lock,
    db_batches,
    file_path,
    file_sha,
    read_catalog,
    table_rows,
    utc,
    verify_entry,
)
from research.db import connect  # noqa: E402


def prune(root, backup_root, *, hot_days=7, apply=False):
    root, backup_root = Path(root).resolve(), Path(backup_root).resolve()
    if hot_days < 7:
        raise ValueError(
            "keep at least 7 hot days, beyond the built-in aggregate refresh windows"
        )
    if (
        root == backup_root
        or root in backup_root.parents
        or backup_root in root.parents
    ):
        raise ValueError(
            "backup must be a separate archive copy, preferably on another device"
        )
    cutoff = datetime.now(timezone.utc) - timedelta(days=hot_days)
    removed = []
    # Stable catalog/files and backup throughout validation and deletion.
    with archive_lock(root, exclusive=True), archive_lock(backup_root):
        catalog, backup = read_catalog(root), read_catalog(backup_root)
        backups = {e["path"]: e for e in backup}
        verified = set()
        with closing(connect(read_only=not apply)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT range_start, range_end FROM timescaledb_information.chunks "
                    "WHERE hypertable_schema = 'public' AND hypertable_name = 'klines' "
                    "AND range_end <= %s ORDER BY range_start",
                    (cutoff,),
                )
                chunks = cur.fetchall()
            if not apply:
                return [
                    {
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "action": "would verify and drop",
                    }
                    for start, end in chunks
                ]
            conn.rollback()
            # Refresh existing aggregate materializations before removing raw data.
            # CALL is outside a transaction; it may manage its own transaction.
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT format('%I.%I', view_schema, view_name) "
                    "FROM timescaledb_information.continuous_aggregates "
                    "WHERE hypertable_schema = 'public' AND hypertable_name = 'klines'"
                )
                aggregates = [row[0] for row in cur.fetchall()]
                for start, end in chunks:
                    for view in aggregates:
                        cur.execute(
                            "CALL refresh_continuous_aggregate(%s::regclass, %s::timestamptz, %s::timestamptz)",
                            (view, start, end),
                        )
            conn.autocommit = False
            for start, end in chunks:
                start, end = utc(start), utc(end)
                try:
                    with conn.cursor() as cur:
                        cur.execute("SET LOCAL lock_timeout = '5s'")
                        # Block late backfills/inserts while comparing this chunk;
                        # the existing collector buffers temporarily in Redis.
                        cur.execute("LOCK TABLE klines IN SHARE MODE")
                        cur.execute(
                            "SELECT DISTINCT symbol FROM klines WHERE open_time >= %s AND open_time < %s",
                            (start, end),
                        )
                        symbols = [row[0] for row in cur.fetchall()]
                    for symbol in symbols:
                        entries = [
                            e
                            for e in catalog
                            if e["symbol"] == symbol
                            and utc(e["start"]) < end
                            and utc(e["end"]) > start
                        ]
                        if not entries:
                            raise ValueError(f"no archive for {symbol} {start}")
                        for entry in entries:
                            if entry["path"] not in verified:
                                verify_entry(root, entry)
                                other = backups.get(entry["path"])
                                if (
                                    other != entry
                                    or file_sha(file_path(backup_root, other))
                                    != entry["sha256"]
                                ):
                                    raise ValueError(
                                        "backup is missing, stale or corrupt"
                                    )
                                verified.add(entry["path"])
                        expected, actual = Fingerprint(), Fingerprint()
                        for rows in db_batches(conn, symbol, start, end):
                            expected.update(rows, symbol=symbol, start=start, end=end)
                        with duckdb.connect(
                            config={"memory_limit": "256MB", "threads": "2"}
                        ) as engine:
                            engine.read_parquet(
                                [str(file_path(root, e)) for e in entries],
                                hive_partitioning=False,
                            ).create_view("archived")
                            result = engine.sql(
                                f"SELECT {', '.join(COLUMNS)} FROM archived "
                                "WHERE symbol = ? AND open_time >= ? AND open_time < ? "
                                "ORDER BY open_time",
                                params=[symbol, start, end],
                            )
                            reader = (
                                result.to_arrow_reader(50_000)
                                if hasattr(result, "to_arrow_reader")
                                else result.fetch_record_batch(50_000)
                            )
                            for batch in reader:
                                actual.update(
                                    table_rows(batch),
                                    symbol=symbol,
                                    start=start,
                                    end=end,
                                )
                        if expected.result(start, end) != actual.result(start, end):
                            raise ValueError(
                                f"archive differs from current DB: {symbol} {start}; nothing dropped"
                            )
                    if not symbols:
                        raise ValueError("empty candidate chunk; inspect manually")
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT drop_chunks('klines', newer_than => %s::timestamptz, "
                            "older_than => %s::timestamptz)",
                            (start, end),
                        )
                        dropped = [row[0] for row in cur.fetchall()]
                    conn.commit()
                    removed.extend(dropped)
                except BaseException:
                    conn.rollback()
                    raise
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("archive"))
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--hot-days", type=int, default=7)
    parser.add_argument(
        "--apply", action="store_true", help="actually drop chunks after all checks"
    )
    args = parser.parse_args()
    print(prune(args.root, args.backup_root, hot_days=args.hot_days, apply=args.apply))


if __name__ == "__main__":
    main()
