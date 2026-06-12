"""Database access for the research layer.

Reads the same TimescaleDB instance the collector writes to. Connection
parameters are taken from the project ``.env`` (or the process environment),
so no separate configuration is needed.

The loader is *gap-aware*: real-world collection is never perfectly
continuous, so :func:`add_segments` labels maximal runs of consecutive
candles. Honest research must be done *within* a continuous segment — never
across a multi-hour gap, which would invent price moves that never happened.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd

try:
    import psycopg2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    psycopg2 = None

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# Columns we always pull; order matters for downstream feature code.
KLINE_COLUMNS = [
    "symbol", "open_time", "close_time",
    "open", "high", "low", "close",
    "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote",
]


def _load_dotenv(path: Path = _ENV_PATH) -> dict[str, str]:
    """Minimal ``.env`` parser (no external dependency)."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip("'").strip('"')
    return env


def connect():
    """Open a psycopg2 connection to the klines database.

    Host/port default to the host-published values (``localhost`` /
    ``POSTGRES_PORT_HOST``) so the tool runs from the host machine against the
    Dockerised database. Override with ``PGHOST`` / ``PGPORT`` if needed.
    """
    if psycopg2 is None:
        raise RuntimeError(
            "psycopg2 is required for live DB access: pip install psycopg2-binary"
        )
    env = {**_load_dotenv(), **os.environ}
    return psycopg2.connect(
        host=env.get("PGHOST", "localhost"),
        port=int(env.get("PGPORT", env.get("POSTGRES_PORT_HOST", "5432"))),
        user=env["POSTGRES_USER"],
        password=env["POSTGRES_PASSWORD"],
        dbname=env["POSTGRES_DB"],
    )


def load_klines(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    conn=None,
) -> pd.DataFrame:
    """Load raw 1-candle rows for ``symbol`` into a DataFrame, ordered by time.

    ``start`` / ``end`` are optional ISO timestamps (inclusive lower / exclusive
    upper bound on ``open_time``).
    """
    own = conn is None
    conn = conn or connect()
    try:
        clauses = ["symbol = %s"]
        params: list = [symbol]
        if start:
            clauses.append("open_time >= %s")
            params.append(start)
        if end:
            clauses.append("open_time < %s")
            params.append(end)
        where = " AND ".join(clauses)
        sql = (
            f"SELECT {', '.join(KLINE_COLUMNS)} FROM klines "
            f"WHERE {where} ORDER BY open_time"
        )
        with warnings.catch_warnings():
            # pandas warns about non-SQLAlchemy DBAPI connections; psycopg2 works fine.
            warnings.simplefilter("ignore", UserWarning)
            df = pd.read_sql(sql, conn, params=params,
                             parse_dates=["open_time", "close_time"])
    finally:
        if own:
            conn.close()
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Offline fallback: load a CSV exported with the same columns."""
    return pd.read_csv(path, parse_dates=["open_time", "close_time"]).sort_values(
        "open_time"
    ).reset_index(drop=True)


def add_segments(df: pd.DataFrame, max_gap_s: float = 5.0) -> pd.DataFrame:
    """Label maximal runs of consecutive candles.

    A new segment starts whenever the gap to the previous candle exceeds
    ``max_gap_s`` (default 5s for 1s data). Adds an integer ``seg`` column.
    """
    df = df.sort_values("open_time").reset_index(drop=True)
    gap = df["open_time"].diff().dt.total_seconds()
    df["seg"] = (gap.isna() | (gap > max_gap_s)).cumsum()
    return df


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-segment row count, time span and coverage — the honest data map."""
    rows = []
    for seg, g in df.groupby("seg"):
        span = (g["open_time"].max() - g["open_time"].min()).total_seconds()
        rows.append({
            "seg": seg,
            "rows": len(g),
            "start": g["open_time"].min(),
            "end": g["open_time"].max(),
            "span_h": round(span / 3600, 2),
            "coverage_pct": round(100 * len(g) / (span + 1), 1) if span else 100.0,
        })
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def largest_segments(df: pd.DataFrame, min_rows: int = 10_000) -> list[int]:
    """Segment ids worth analysing, largest first."""
    summary = segment_summary(df)
    return summary[summary["rows"] >= min_rows]["seg"].tolist()
