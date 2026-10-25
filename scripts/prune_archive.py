#!/usr/bin/env python3
"""Verify archive, backup, policies and current DB before planning chunk pruning."""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import duckdb  # noqa: E402
from research.archive import (  # noqa: E402
    COLUMNS, Fingerprint, archive_lock, db_batches, file_path, read_catalog,
    table_rows, utc,
)
from research.archive_backup import separate_roots, verify_copy  # noqa: E402
from research.db import connect  # noqa: E402


def check_policies(conn, hot_days):
    """Fail closed on native retention or refresh windows reaching pruned data."""
    with conn.cursor() as cur:
        cur.execute(
            "WITH RECURSIVE related(schema_name, table_name) AS ("
            " SELECT 'public'::text COLLATE \"C\", 'klines'::text COLLATE \"C\" UNION "
            " SELECT c.materialization_hypertable_schema::text COLLATE \"C\", "
            " c.materialization_hypertable_name::text COLLATE \"C\" "
            " FROM timescaledb_information.continuous_aggregates c "
            " JOIN related r ON c.hypertable_schema = r.schema_name "
            " AND c.hypertable_name = r.table_name) "
            "SELECT j.job_id, j.proc_name, j.scheduled, j.config "
            "FROM timescaledb_information.jobs j JOIN related r "
            "ON j.hypertable_schema = r.schema_name AND j.hypertable_name = r.table_name"
        )
        policies = cur.fetchall()
        for job_id, name, scheduled, config in policies:
            if not scheduled:
                continue
            if name == "policy_retention":
                raise ValueError(f"active native retention job {job_id}; review/remove it before pruning")
            if name == "policy_refresh_continuous_aggregate":
                offset = config.get("start_offset")
                end_offset = config.get("end_offset")
                if not isinstance(offset, str) or not isinstance(end_offset, str):
                    raise ValueError(f"unbounded/unknown aggregate refresh job {job_id}")
                cur.execute(
                    "SELECT %s::interval > %s::interval AND %s::interval >= interval '0' "
                    "AND %s::interval < %s::interval",
                    (offset, end_offset, end_offset, offset, f"{hot_days} days"),
                )
                if not cur.fetchone()[0]:
                    raise ValueError(f"aggregate refresh job {job_id} reaches the pruning window")
            elif name not in {"policy_compression", "policy_reorder"}:
                raise ValueError(f"unknown scheduled job {job_id} ({name}); review before pruning")
    return policies


def aggregate_plan(conn, chunks):
    """Only fixed UTC buckets aligned with chunk bounds are automatically safe.

    Refuse custom origins/timezones/hierarchies rather than partially refreshing
    buckets across chunks whose raw data may already have been pruned.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT format('%I.%I', view_schema, view_name), view_definition "
            "FROM timescaledb_information.continuous_aggregates "
            "WHERE hypertable_schema = 'public' AND hypertable_name = 'klines'"
        )
        aggregates = cur.fetchall()
        for view, definition in aggregates:
            match = re.search(
                r"time_bucket\('([^']+)'::interval,\s*(?:\w+\.)?open_time\)", definition
            )
            if not match:
                raise ValueError(f"custom aggregate bucket needs manual review: {view}")
            width = match.group(1)
            cur.execute(
                "SELECT %s::interval > interval '0' AND "
                "EXTRACT(year FROM %s::interval) = 0 AND EXTRACT(month FROM %s::interval) = 0",
                (width, width, width),
            )
            if not cur.fetchone()[0]:
                raise ValueError(f"calendar/invalid aggregate bucket needs manual review: {view}")
            cur.execute(
                "SELECT coalesce(bool_and(time_bucket(%s::interval, boundary) = boundary), true) "
                "FROM unnest(%s::timestamptz[]) AS boundary",
                (width, [t for pair in chunks for t in pair]),
            )
            if not cur.fetchone()[0]:
                raise ValueError(f"chunk boundaries do not align with aggregate {view}")
        cur.execute(
            "SELECT child.view_name FROM timescaledb_information.continuous_aggregates child "
            "JOIN timescaledb_information.continuous_aggregates parent "
            "ON child.hypertable_schema = parent.materialization_hypertable_schema "
            "AND child.hypertable_name = parent.materialization_hypertable_name "
            "WHERE parent.hypertable_schema = 'public' AND parent.hypertable_name = 'klines'"
        )
        if cur.fetchall():
            raise ValueError("hierarchical aggregates require manual review before pruning")
    return [view for view, _ in aggregates]


def source_fingerprint(conn, symbol, start, end):
    result = Fingerprint()
    with closing(db_batches(conn, symbol, start, end)) as batches:
        for rows in batches:
            result.update(rows, symbol=symbol, start=start, end=end)
    return result.result(start, end)


def verify_chunk(conn, root, backup_root, catalog, backups, verified, start, end):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT symbol FROM klines WHERE open_time >= %s AND open_time < %s",
            (start, end),
        )
        symbols = sorted(row[0] for row in cur.fetchall())
    if not symbols:
        raise ValueError("empty candidate chunk; inspect manually")
    fingerprints = {}
    for symbol in symbols:
        entries = [e for e in catalog if e["symbol"] == symbol
                   and utc(e["start"]) < end and utc(e["end"]) > start]
        if not entries:
            raise ValueError(f"no archive for {symbol} {start}")
        for entry in entries:
            if entry["path"] not in verified:
                verify_copy(root, backup_root, entry, backups.get(entry["path"]))
                verified.add(entry["path"])
        expected = source_fingerprint(conn, symbol, start, end)
        actual = Fingerprint()
        with duckdb.connect(config={"memory_limit": "256MB", "threads": "2"}) as engine:
            engine.read_parquet(
                [str(file_path(root, e)) for e in entries], hive_partitioning=False,
            ).create_view("archived")
            result = engine.sql(
                f"SELECT {', '.join(COLUMNS)} FROM archived "
                "WHERE symbol = ? AND open_time >= ? AND open_time < ? ORDER BY open_time",
                params=[symbol, start, end],
            )
            reader = (result.to_arrow_reader(50_000) if hasattr(result, "to_arrow_reader")
                      else result.fetch_record_batch(50_000))
            with closing(reader):
                for batch in reader:
                    actual.update(table_rows(batch), symbol=symbol, start=start, end=end)
        if expected != actual.result(start, end):
            raise ValueError(f"archive differs from current DB: {symbol} {start}; nothing dropped")
        fingerprints[symbol] = expected
    return fingerprints


def prune(root, backup_root, *, hot_days=7, apply=False):
    if hot_days < 7:
        raise ValueError("keep at least 7 hot days, beyond the built-in aggregate refresh windows")
    root, backup_root = separate_roots(root, backup_root)
    if not (root / "catalog.json").exists() or not (backup_root / "catalog.json").exists():
        raise ValueError("archive or backup catalog is missing")
    cutoff = datetime.now(timezone.utc) - timedelta(days=hot_days)
    removed, plan = [], []
    with archive_lock(root, exclusive=True), archive_lock(backup_root):
        catalog = read_catalog(root)
        backups = {e["path"]: e for e in read_catalog(backup_root)}
        verified = set()
        with closing(connect(read_only=not apply)) as conn:
            conn.set_session(isolation_level="REPEATABLE READ", readonly=not apply)
            check_policies(conn, hot_days)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT range_start, range_end FROM timescaledb_information.chunks "
                    "WHERE hypertable_schema = 'public' AND hypertable_name = 'klines' "
                    "AND range_end <= %s ORDER BY range_start", (cutoff,),
                )
                chunks = [(utc(a), utc(b)) for a, b in cur.fetchall()]
            aggregates = aggregate_plan(conn, chunks)
            fingerprints = {}
            # Both modes really compare every column, symbol and backup to a DB snapshot.
            for start, end in chunks:
                fingerprints[(start, end)] = verify_chunk(
                    conn, root, backup_root, catalog, backups, verified, start, end,
                )
                plan.append({
                    "start": start.isoformat(), "end": end.isoformat(),
                    "action": "verified; would drop after locked revalidation",
                    "symbols": list(fingerprints[(start, end)]),
                    "rows": sum(v["rows"] for v in fingerprints[(start, end)].values()),
                    "aggregates_to_refresh": aggregates,
                    "source_verified": True, "backup_verified": True, "policies_checked": True,
                })
            if not apply:
                return plan
            conn.rollback()
            conn.set_session(isolation_level="READ COMMITTED")
            for start, end in chunks:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SET LOCAL lock_timeout = '5s'")
                        cur.execute("LOCK TABLE klines IN SHARE MODE")
                    check_policies(conn, hot_days)
                    latest = verify_chunk(conn, root, backup_root, catalog, backups, verified, start, end)
                    if latest != fingerprints[(start, end)]:
                        raise ValueError("source changed since validation; nothing dropped for this chunk")
                    # Keep the raw-table SHARE lock on this connection throughout
                    # refresh and deletion. CALL uses another connection because
                    # Timescale may commit internally. Backfills cannot race it.
                    if aggregates:
                        with closing(connect(read_only=False)) as refresher:
                            refresher.autocommit = True
                            with refresher.cursor() as refresh_cursor:
                                refresh_cursor.execute("SET lock_timeout = '5s'")
                                for view in aggregates:
                                    refresh_cursor.execute(
                                        "CALL refresh_continuous_aggregate(%s::regclass, %s::timestamptz, %s::timestamptz)",
                                        (view, start, end),
                                    )
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT drop_chunks('klines', newer_than => %s::timestamptz, "
                            "older_than => %s::timestamptz)", (start, end),
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
    parser.add_argument("--apply", action="store_true", help="actually drop chunks after all checks")
    args = parser.parse_args()
    print(json.dumps(prune(args.root, args.backup_root, hot_days=args.hot_days, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
