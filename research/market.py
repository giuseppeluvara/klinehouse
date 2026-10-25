"""Bounded SQL queries over verified archive files and the recent DB tail."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import os
from pathlib import Path
import tempfile

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .archive import (
    COLUMNS,
    DEFAULT_BATCH,
    SCHEMA,
    archive_lock,
    db_batches,
    file_path,
    read_catalog,
    symbol_name,
    utc,
)
from .db import _load_dotenv, connect

TIMEFRAMES = {
    "1s": 1,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}
_AGG = {
    "symbol": "first(symbol)",
    "open_time": "bucket",
    "close_time": "max(close_time)",
    "open": "arg_min(open, open_time)",
    "high": "max(high)",
    "low": "min(low)",
    "close": "arg_max(close, open_time)",
    "volume": "sum(volume)",
    "quote_volume": "sum(quote_volume)",
    "trades": "sum(trades)",
    "taker_buy_base": "sum(taker_buy_base)",
    "taker_buy_quote": "sum(taker_buy_quote)",
}


def _missing_ranges(start, end, entries):
    cursor = start
    for entry in sorted(entries, key=lambda e: e["start"]):
        if not entry["complete"]:
            continue
        left, right = max(start, utc(entry["start"])), min(end, utc(entry["end"]))
        if left > cursor:
            yield cursor, left
        cursor = max(cursor, right)
    if cursor < end:
        yield cursor, end


@contextmanager
def _query(
    symbol,
    start,
    end,
    timeframe,
    columns,
    archive_root,
    conn,
    source,
    batch_size,
    memory_limit,
):
    symbol, start, end = symbol_name(symbol), utc(start), utc(end)
    if start >= end or batch_size <= 0 or timeframe not in TIMEFRAMES:
        raise ValueError("invalid range, batch size or timeframe")
    if source not in {"auto", "archive", "database"}:
        raise ValueError("source must be auto, archive or database")
    columns = list(COLUMNS if columns is None else columns)
    if (
        not columns
        or len(set(columns)) != len(columns)
        or any(c not in COLUMNS for c in columns)
    ):
        raise ValueError("invalid or duplicate columns")
    seconds = TIMEFRAMES[timeframe]
    needed = list(dict.fromkeys(["symbol", "open_time", *columns]))
    env = {**_load_dotenv(), **os.environ}
    root = Path(archive_root or env.get("ARCHIVE_ROOT", "archive"))
    with (
        archive_lock(root),
        tempfile.TemporaryDirectory(
            prefix="klinehouse-", dir=os.environ.get("MARKET_TEMP_DIR")
        ) as temp,
    ):
        entries = (
            []
            if source == "database"
            else [
                e
                for e in read_catalog(root)
                if e["symbol"] == symbol
                and utc(e["start"]) < end
                and utc(e["end"]) > start
            ]
        )
        files = []
        for entry in entries:
            path = file_path(root, entry)
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise ValueError(f"archive missing or changed; run verify: {path}")
            files.append(str(path))
        engine = duckdb.connect(config={"memory_limit": memory_limit, "threads": "2"})
        try:
            engine.execute("SET TimeZone='UTC'")
            engine.execute("SET temp_directory = ?", [temp])
            engine.execute("SET max_temp_directory_size='20GB'")
            engine.execute("SET preserve_insertion_order=false")
            select = ", ".join(f'"{name}"' for name in needed)
            parts = []
            if files:
                engine.read_parquet(files, hive_partitioning=False).create_view(
                    "archive_files"
                )
                parts.append(
                    f"SELECT {select}, 0 AS priority FROM archive_files "
                    "WHERE symbol = ? AND open_time >= ? AND open_time < ?"
                )
            parameters = [symbol, start, end] if files else []
            ranges = (
                []
                if source == "archive"
                else list(_missing_ranges(start, end, entries))
            )
            own_connection = conn is None
            if ranges:
                connection = conn if conn is not None else connect(read_only=True)
                try:
                    spool = Path(temp) / "tail.parquet"
                    schema = pa.schema([SCHEMA.field(name) for name in needed])
                    with pq.ParquetWriter(spool, schema, compression="zstd") as writer:
                        for left, right in ranges:
                            for rows in db_batches(
                                connection, symbol, left, right, needed, batch_size
                            ):
                                table = pa.Table.from_arrays(
                                    [
                                        pa.array(values, type=field.type)
                                        for values, field in zip(zip(*rows), schema)
                                    ],
                                    schema=schema,
                                )
                                writer.write_table(table, row_group_size=batch_size)
                    engine.read_parquet(
                        str(spool), hive_partitioning=False
                    ).create_view("db_tail")
                    parts.append(f"SELECT {select}, 1 AS priority FROM db_tail")
                finally:
                    if own_connection:
                        connection.close()
            if not parts:
                engine.register(
                    "empty_candles", pa.Table.from_batches([], schema=SCHEMA)
                )
                parts = [f"SELECT {select}, 0 AS priority FROM empty_candles"]
            union = " UNION ALL ".join(parts)
            # A DB candle wins in partially archived ranges. Complete archive
            # ranges are never fetched from PostgreSQL in the first place.
            raw = (
                f"SELECT {select} FROM ({union}) QUALIFY "
                "row_number() OVER (PARTITION BY symbol, open_time ORDER BY priority DESC) = 1"
            )
            output = ", ".join(f'"{name}"' for name in columns)
            if seconds == 1:
                sql = f"SELECT {output} FROM ({raw}) ORDER BY open_time"
            else:
                aggregates = ", ".join(f'{_AGG[name]} AS "{name}"' for name in columns)
                sql = (
                    f"SELECT {aggregates} FROM (SELECT *, "
                    f"time_bucket(INTERVAL '{seconds} seconds', open_time) AS bucket FROM ({raw})) "
                    f"GROUP BY bucket HAVING count(*) = {seconds} "
                    "AND min(open_time) = bucket "
                    f"AND max(open_time) = bucket + INTERVAL '{seconds - 1} seconds' "
                    "ORDER BY bucket"
                )
            yield engine.sql(sql, params=parameters)
        finally:
            engine.close()


def iter_market(
    symbol: str,
    start,
    end,
    *,
    timeframe="1s",
    columns=None,
    archive_root=None,
    conn=None,
    source="auto",
    batch_size=DEFAULT_BATCH,
    memory_limit="256MB",
):
    """Yield Arrow batches, with UTC [start,end) semantics and bounded memory.

    Only complete UTC-aligned buckets are returned for aggregated timeframes.
    Close the generator when stopping early to release locks/connections.
    The DuckDB limit excludes Python/Arrow allocations; output batches are bounded
    separately. Large sorts/deduplication can spill to MARKET_TEMP_DIR.
    """
    with _query(
        symbol,
        start,
        end,
        timeframe,
        columns,
        archive_root,
        conn,
        source,
        batch_size,
        memory_limit,
    ) as result:
        reader = (
            result.to_arrow_reader(batch_size)
            if hasattr(result, "to_arrow_reader")
            else result.fetch_record_batch(batch_size)
        )
        yield from reader


def load_market(
    symbol: str, start, end, *, max_rows=1_000_000, **kwargs
) -> pd.DataFrame:
    """Convenience DataFrame with a hard row cap; use iter_market for large jobs."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    batches, count = [], 0
    iterator = iter_market(symbol, start, end, **kwargs)
    try:
        for batch in iterator:
            count += batch.num_rows
            if count > max_rows:
                raise MemoryError(
                    "result exceeds max_rows; narrow the range, aggregate or use iter_market"
                )
            batches.append(batch)
    finally:
        iterator.close()
    if batches:
        return pa.Table.from_batches(batches).to_pandas()
    return pd.DataFrame(columns=kwargs.get("columns") or COLUMNS)


def iter_market_windows(symbol, start, end, *, window_seconds=86400, **kwargs):
    """Load sequential windows; callers can retain their own bounded lookback."""
    start, end = utc(start), utc(end)
    seconds = TIMEFRAMES.get(kwargs.get("timeframe", "1s"), 0)
    if window_seconds <= 0 or not seconds or window_seconds % seconds:
        raise ValueError("window_seconds must be a positive multiple of the timeframe")
    if int(start.timestamp()) % seconds or start.microsecond:
        raise ValueError("window start must align to the timeframe")
    while start < end:
        stop = min(end, start + timedelta(seconds=window_seconds))
        yield load_market(symbol, start, stop, **kwargs)
        start = stop
