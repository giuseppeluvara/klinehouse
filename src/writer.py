"""Writer: pops batches from Redis and persists them to TimescaleDB."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import psycopg
import redis.asyncio as aioredis

from . import config

logger = logging.getLogger(__name__)

_INSERT_SQL = """
INSERT INTO klines (
    symbol, open_time, close_time,
    open, high, low, close,
    volume, quote_volume, trades,
    taker_buy_base, taker_buy_quote
) VALUES (
    %(symbol)s,
    to_timestamp(%(open_time)s::DOUBLE PRECISION / 1000) AT TIME ZONE 'UTC',
    to_timestamp(%(close_time)s::DOUBLE PRECISION / 1000) AT TIME ZONE 'UTC',
    %(open)s, %(high)s, %(low)s, %(close)s,
    %(volume)s, %(quote_volume)s, %(trades)s,
    %(taker_buy_base)s, %(taker_buy_quote)s
)
ON CONFLICT (symbol, open_time) DO NOTHING
"""


def _record_to_params(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol":          record["symbol"],
        "open_time":       int(record["open_time"]),
        "close_time":      int(record["close_time"]),
        "open":            float(record["open"]),
        "high":            float(record["high"]),
        "low":             float(record["low"]),
        "close":           float(record["close"]),
        "volume":          float(record["volume"]),
        "quote_volume":    float(record["quote_volume"]),
        "trades":          int(record["trades"]),
        "taker_buy_base":  float(record["taker_buy_base"]),
        "taker_buy_quote": float(record["taker_buy_quote"]),
    }


class Writer:
    def __init__(self, redis_client: aioredis.Redis, db_conn: psycopg.AsyncConnection) -> None:
        self._redis = redis_client
        self._db = db_conn
        self._stop_event = asyncio.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        while True:
            if self._stop_event.is_set():
                await self._flush_remaining()
                return
            try:
                await self._run_batch_cycle()
            except asyncio.CancelledError:
                logger.info("writer cancelled — flushing remaining buffer")
                await self._flush_remaining()
                return
            except Exception as exc:
                logger.error("writer error: %s", exc)
                await asyncio.sleep(1)

    async def _run_batch_cycle(self) -> None:
        """Wait up to BATCH_INTERVAL_SECONDS then flush whatever is buffered."""
        deadline = asyncio.get_event_loop().time() + config.BATCH_INTERVAL_SECONDS
        batch: list[dict[str, Any]] = []

        while len(batch) < config.BATCH_SIZE:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break

            raw = await self._redis.blpop(config.REDIS_KLINES_KEY, timeout=min(remaining, 1))
            if raw is None:
                if asyncio.get_event_loop().time() >= deadline:
                    break
                continue
            _, value = raw
            try:
                record = json.loads(value)
                batch.append(_record_to_params(record))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("writer: invalid record dropped: %s", exc)

        if batch:
            await self._write_batch(batch)

    async def _write_batch(self, batch: list[dict[str, Any]]) -> None:
        async with self._db.cursor() as cur:
            await cur.executemany(_INSERT_SQL, batch)
        await self._db.commit()
        logger.info("writer: wrote %d rows to timescaledb", len(batch))

    async def _flush_remaining(self) -> None:
        """Drain the entire Redis buffer on shutdown."""
        batch: list[dict[str, Any]] = []
        while True:
            raw = await self._redis.lpop(config.REDIS_KLINES_KEY)
            if raw is None:
                break
            try:
                record = json.loads(raw)
                batch.append(_record_to_params(record))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("writer flush: invalid record dropped: %s", exc)

            if len(batch) >= config.BATCH_SIZE:
                await self._write_batch(batch)
                batch = []

        if batch:
            await self._write_batch(batch)

        logger.info("writer: graceful flush complete")
