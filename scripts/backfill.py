#!/usr/bin/env python3
"""Heal *interior* gaps in the klines table by back-filling from Binance REST.

The collector's startup gap recovery only covers the *trailing* gap (the latest
stored candle -> now). It cannot fill an interior hole — one that opened while
the collector was down and that newer streamed data has since grown past. This
tool scans for every interior gap and fetches the missing candles via REST.

It is idempotent (``ON CONFLICT DO NOTHING``), safe to run while the collector
is live, and can be scheduled (cron) to self-heal periodically.

Usage:
  python scripts/backfill.py                          # all SYMBOLS, KLINE_INTERVAL from .env
  python scripts/backfill.py --symbol BTCUSDT --interval 1s
  python scripts/backfill.py --min-gap 2              # only gaps >= 2 intervals
  python scripts/backfill.py --dry-run               # report gaps, fetch nothing
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values


def _ssl_context() -> ssl.SSLContext:
    """Prefer certifi's CA bundle; some Python installs have a broken default
    trust store that rejects valid certs."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_SSL_CTX = _ssl_context()

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
REST_PATH = "/api/v3/klines"
PAGE = 1000
_UNITS = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}

INSERT_SQL = """
INSERT INTO klines (
    symbol, open_time, close_time, open, high, low, close,
    volume, quote_volume, trades, taker_buy_base, taker_buy_quote
) VALUES %s
ON CONFLICT (symbol, open_time) DO NOTHING
"""


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    env: dict[str, str] = {}
    if path.exists():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip("'").strip('"')
    return env


def interval_ms(interval: str) -> int:
    return int(interval[:-1]) * _UNITS[interval[-1]]


def connect(env: dict[str, str]):
    return psycopg2.connect(
        host=env.get("PGHOST", "localhost"),
        port=int(env.get("PGPORT", env.get("POSTGRES_PORT_HOST", "5432"))),
        user=env["POSTGRES_USER"], password=env["POSTGRES_PASSWORD"],
        dbname=env["POSTGRES_DB"],
    )


def find_gaps(conn, symbol: str, step_ms: int, min_gap: int) -> list[tuple[int, int]]:
    """Return [(prev_open_ms, next_open_ms)] for interior gaps >= min_gap steps."""
    threshold_s = min_gap * step_ms / 1000.0
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH g AS (
                SELECT open_time,
                       lag(open_time) OVER (ORDER BY open_time) AS prev
                FROM klines WHERE symbol = %s
            )
            SELECT (EXTRACT(EPOCH FROM prev) * 1000)::BIGINT,
                   (EXTRACT(EPOCH FROM open_time) * 1000)::BIGINT
            FROM g
            WHERE prev IS NOT NULL
              AND EXTRACT(EPOCH FROM (open_time - prev)) > %s
            ORDER BY open_time
            """,
            (symbol, threshold_s),
        )
        return [(int(p), int(n)) for p, n in cur.fetchall()]


def fetch_klines(base_url: str, symbol: str, interval: str,
                 start_ms: int, end_ms: int) -> list[list]:
    """Fetch a single REST page [start_ms, end_ms] (inclusive), with retries."""
    params = urllib.parse.urlencode({
        "symbol": symbol, "interval": interval,
        "startTime": start_ms, "endTime": end_ms, "limit": PAGE,
    })
    url = f"{base_url}{REST_PATH}?{params}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30, context=_SSL_CTX) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
            if e.code == 429:  # rate limited
                wait = int(e.headers.get("Retry-After", "2"))
                print(f"  rate-limited, sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    return []


def rest_row_to_tuple(symbol: str, row: list) -> tuple:
    # REST: [open_time, o, h, l, c, vol, close_time, quote_vol, trades, tbb, tbq, _]
    return (
        symbol, row[0], row[6],
        float(row[1]), float(row[2]), float(row[3]), float(row[4]),
        float(row[5]), float(row[7]), int(row[8]), float(row[9]), float(row[10]),
    )


def backfill_gap(conn, base_url: str, symbol: str, interval: str,
                 step_ms: int, prev_ms: int, next_ms: int, sleep_s: float) -> int:
    """Fill (prev_ms, next_ms) exclusive. Returns candles inserted."""
    start = prev_ms + step_ms
    end = next_ms - step_ms
    inserted = 0
    while start <= end:
        rows = fetch_klines(base_url, symbol, interval, start, end)
        if not rows:
            break
        tuples = [
            rest_row_to_tuple(symbol, r) for r in rows
            if start <= r[0] <= end  # stay strictly inside the gap
        ]
        if tuples:
            with conn.cursor() as cur:
                template = ("(%s, to_timestamp(%s::float/1000) AT TIME ZONE 'UTC', "
                            "to_timestamp(%s::float/1000) AT TIME ZONE 'UTC', "
                            "%s,%s,%s,%s,%s,%s,%s,%s,%s)")
                # page_size >= len(tuples) -> single statement, so rowcount is the
                # true number of rows inserted (ON CONFLICT skips are excluded).
                execute_values(cur, INSERT_SQL, tuples, template=template,
                               page_size=len(tuples))
                inserted += cur.rowcount
            conn.commit()
        last_open = rows[-1][0]
        nxt = last_open + step_ms
        if nxt <= start or len(rows) < PAGE:
            start = nxt
            if len(rows) < PAGE:
                break
        else:
            start = nxt
        time.sleep(sleep_s)
    return inserted


def main() -> int:
    env = {**load_env()}
    ap = argparse.ArgumentParser(description="Back-fill interior gaps from Binance REST")
    ap.add_argument("--symbol", action="append", help="symbol(s); default = SYMBOLS in .env")
    ap.add_argument("--interval", default=env.get("KLINE_INTERVAL", "1s"))
    ap.add_argument("--min-gap", type=int, default=1, help="min gap size in intervals (default 1)")
    ap.add_argument("--sleep", type=float, default=0.12, help="seconds between REST pages")
    ap.add_argument("--from", dest="from_ts", default=None,
                    help="seed history from this UTC time (e.g. 2026-06-10 or "
                         "2026-06-10T00:00:00) up to the earliest existing candle")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from_ms = None
    if args.from_ts:
        import calendar
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                from_ms = calendar.timegm(time.strptime(args.from_ts, fmt)) * 1000
                break
            except ValueError:
                continue
        if from_ms is None:
            print(f"could not parse --from '{args.from_ts}'", file=sys.stderr)
            return 2

    base_url = env.get("REST_BASE_URL", "https://api.binance.com")
    symbols = args.symbol or [s.strip().upper() for s in env.get("SYMBOLS", "BTCUSDT").split(",") if s.strip()]
    step_ms = interval_ms(args.interval)

    conn = connect(env)
    grand_total = 0
    try:
        for symbol in symbols:
            gaps = find_gaps(conn, symbol, step_ms, args.min_gap)
            if from_ms is not None:
                with conn.cursor() as cur:
                    cur.execute("SELECT (EXTRACT(EPOCH FROM min(open_time)) * 1000)::BIGINT "
                                "FROM klines WHERE symbol = %s", (symbol,))
                    row = cur.fetchone()
                first = row[0] if row and row[0] is not None else int(time.time() * 1000)
                if from_ms < first:
                    # leading gap: fill [from_ms, first) as a synthetic gap
                    gaps.insert(0, (from_ms - step_ms, int(first)))
            missing = sum((n - p) // step_ms - 1 for p, n in gaps)
            print(f"\n{symbol} [{args.interval}]: {len(gaps)} interior gap(s), "
                  f"~{missing:,} candles missing", flush=True)
            if args.dry_run:
                for p, n in gaps[:20]:
                    dur = (n - p) / 1000
                    print(f"  gap {dur:>10.0f}s  "
                          f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(p/1000))} -> "
                          f"{time.strftime('%H:%M:%S', time.gmtime(n/1000))}")
                continue
            for i, (p, n) in enumerate(gaps, 1):
                got = backfill_gap(conn, base_url, symbol, args.interval, step_ms, p, n, args.sleep)
                grand_total += got
                print(f"  [{i}/{len(gaps)}] gap {(n-p)/1000:.0f}s -> recovered {got:,} candles", flush=True)
        print(f"\nDone. Total recovered: {grand_total:,} candles.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
