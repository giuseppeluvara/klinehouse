#!/usr/bin/env python3
"""Bounded, read-only storage inventory; no credentials are included in output."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.archive import archive_lock, read_catalog  # noqa: E402
from research.db import connect  # noqa: E402


def audit(root):
    root = Path(root)
    report = {"observed_at": datetime.now(timezone.utc).isoformat(),
              "platform": platform.system(), "machine": platform.machine(),
              "free_disk_bytes": shutil.disk_usage(Path.cwd()).free,
              "docker_cli_available": shutil.which('docker') is not None,
              "archive_present": (root / 'catalog.json').is_file()}
    if report['archive_present']:
        with archive_lock(root):
            files = read_catalog(root)
            report['archive'] = {"files": len(files), "bytes": sum(e['bytes'] for e in files),
                                 "rows": sum(e['rows'] for e in files),
                                 "symbols": sorted({e['symbol'] for e in files}),
                                 "incomplete_files": sum(not e['complete'] for e in files)}
    os.environ.setdefault('PGCONNECT_TIMEOUT', '5')
    try:
        with closing(connect(read_only=True)) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = '15s'")
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
            version = cur.fetchone()
            cur.execute('SELECT pg_database_size(current_database())')
            report['database'] = {"reachable": True, "bytes": cur.fetchone()[0],
                                  "timescaledb_version": version[0] if version else None}
            # Loose index scan over the (symbol, open_time) primary key: avoid a
            # full historical GROUP BY merely to discover a few market symbols.
            cur.execute("""
                WITH RECURSIVE symbols AS (
                    (SELECT symbol FROM klines ORDER BY symbol LIMIT 1)
                    UNION ALL
                    SELECT (SELECT k.symbol FROM klines k WHERE k.symbol > s.symbol
                            ORDER BY k.symbol LIMIT 1)
                    FROM symbols s WHERE s.symbol IS NOT NULL
                ) SELECT symbol FROM symbols WHERE symbol IS NOT NULL
            """)
            symbols = [r[0] for r in cur.fetchall()]
            bounds = {}
            for symbol in symbols:
                cur.execute('SELECT open_time, close_time FROM klines WHERE symbol=%s ORDER BY open_time LIMIT 1', (symbol,))
                first = cur.fetchone()
                cur.execute('SELECT open_time, close_time FROM klines WHERE symbol=%s ORDER BY open_time DESC LIMIT 1', (symbol,))
                last = cur.fetchone()
                bounds[symbol] = {'first_open': first[0], 'last_open': last[0], 'last_close': last[1],
                                  'first_candle_seconds': (first[1] - first[0]).total_seconds(),
                                  'last_candle_seconds': (last[1] - last[0]).total_seconds()}
            report['database']['bounds'] = bounds
            if version:
                cur.execute("SELECT range_start, range_end, is_compressed FROM timescaledb_information.chunks WHERE hypertable_schema='public' AND hypertable_name='klines' ORDER BY range_start")
                report['database']['chunks'] = [dict(zip(('start', 'end', 'compressed'), row)) for row in cur.fetchall()]
                cur.execute("SELECT job_id, proc_name, scheduled, config FROM timescaledb_information.jobs WHERE hypertable_name IS NOT NULL ORDER BY job_id")
                report['database']['policies'] = [dict(zip(('job_id', 'procedure', 'scheduled', 'config'), row)) for row in cur.fetchall()]
                cur.execute("SELECT * FROM hypertable_detailed_size('public.klines')")
                names = [c.name for c in cur.description]
                report['database']['hypertable_sizes'] = [dict(zip(names, row)) for row in cur.fetchall()]
    except Exception as exc:
        # libpq exception text may contain a remote hostname or user; retain only
        # the class and SQLSTATE. Full diagnostics remain local to the operator.
        report['database'] = {'reachable': report.get('database', {}).get('reachable', False),
                              'audit_complete': False, 'error_type': type(exc).__name__,
                              'sqlstate': getattr(exc, 'pgcode', None)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('archive'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    text = json.dumps(result, indent=2, default=str) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
