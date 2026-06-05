#!/usr/bin/env python3
"""Export a date range of klines to Parquet for cold archival or downstream
analysis (Spark, Databricks, DuckDB, pandas, etc.).

Usage
-----
    python scripts/export_parquet.py \\
        --symbol BTCUSDT \\
        --start 2024-01-01 \\
        --end   2024-02-01 \\
        --out   exports/

Environment variables
---------------------
All POSTGRES_* variables from .env are read automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print(
            "pyarrow is required for Parquet export.\n"
            "Install it with:  pip install pyarrow",
            file=sys.stderr,
        )
        sys.exit(1)


def _dsn() -> str:
    user = os.environ.get("POSTGRES_USER", "klines")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "klines")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def export(symbol: str, start: str, end: str, out_dir: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import psycopg

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{symbol}_{start}_{end}.parquet".replace(":", "-")

    query = """
        SELECT
            symbol,
            open_time,
            close_time,
            open, high, low, close,
            volume, quote_volume, trades,
            taker_buy_base, taker_buy_quote
        FROM klines
        WHERE symbol = %s
          AND open_time >= %s::TIMESTAMPTZ
          AND open_time <  %s::TIMESTAMPTZ
        ORDER BY open_time
    """

    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (symbol, start, end))
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]

    if not rows:
        print(f"No data found for {symbol} between {start} and {end}.")
        return

    columns = {name: [row[i] for row in rows] for i, name in enumerate(col_names)}
    table = pa.table(columns)
    pq.write_table(table, out_file, compression="snappy")
    print(f"Exported {len(rows):,} rows → {out_file}")


def main() -> None:
    _require_pyarrow()

    parser = argparse.ArgumentParser(description="Export klines to Parquet.")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. BTCUSDT")
    parser.add_argument("--start",  required=True, help="Start date (inclusive) ISO-8601, e.g. 2024-01-01")
    parser.add_argument("--end",    required=True, help="End date (exclusive) ISO-8601, e.g. 2024-02-01")
    parser.add_argument("--out",    default="exports", help="Output directory (default: exports/)")
    args = parser.parse_args()

    export(
        symbol=args.symbol.upper(),
        start=args.start,
        end=args.end,
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
