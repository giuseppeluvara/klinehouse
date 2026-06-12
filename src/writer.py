"""Writer: pops batches from Redis and persists them to TimescaleDB.

Resilient to database outages: if a write fails (e.g. the database restarts or
the connection drops), the popped records are pushed back to the head of the
Redis queue *before* reconnecting, so no candle is ever lost — it is simply
retried once the database is reachable again.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
    def __init__(self, redis_client: aioredis.Redis, dsn: str) -> None:
        self._redis = redis_client
        self._dsn = dsn
        self._db: psycopg.AsyncConnection | None = None
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
        # connection is closed by main() on shutdown path via _flush_remaining

    # -- connection management -------------------------------------------------

    async def _ensure_connection(self) -> None:
        if self._db is None or self._db.closed:
            await self._reconnect()

    async def _reconnect(self) -> None:
        """(Re)establish the DB connection with exponential backoff.

        Blocks the writer loop while the database is unreachable; buffered
        records stay safely in Redis until it returns.
        """
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                if self._db is not None:
                    try:
                        await self._db.close()
                    except Exception:
                        pass
                self._db = await psycopg.AsyncConnection.connect(self._dsn, autocommit=False)
                logger.info("writer: connected to timescaledb")
                return
            except Exception as exc:
                logger.error("writer: DB connect failed: %s — retry in %.0fs", exc, backoff)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, config.RECONNECT_MAX_BACKOFF)

    # -- batching --------------------------------------------------------------

    async def _run_batch_cycle(self) -> None:
        """Wait up to BATCH_INTERVAL_SECONDS then flush whatever is buffered."""
        deadline = asyncio.get_event_loop().time() + config.BATCH_INTERVAL_SECONDS
        raws: list[str] = []

        while len(raws) < config.BATCH_SIZE:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break

            raw = await self._redis.blpop(config.REDIS_KLINES_KEY, timeout=min(remaining, 1))
            if raw is None:
                if asyncio.get_event_loop().time() >= deadline:
                    break
                continue
            _, value = raw
            raws.append(value)

        if raws:
            await self._write_raws(raws)

    async def _write_raws(self, raws: list[str]) -> None:
        """Parse and persist raw JSON records, re-queueing them on DB failure."""
        batch: list[dict[str, Any]] = []
        for value in raws:
            try:
                batch.append(_record_to_params(json.loads(value)))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("writer: invalid record dropped: %s", exc)
        if not batch:
            return

        try:
            await self._ensure_connection()
            assert self._db is not None
            async with self._db.cursor() as cur:
                await cur.executemany(_INSERT_SQL, batch)
            await self._db.commit()
            logger.info("writer: wrote %d rows to timescaledb", len(batch))
        except Exception as exc:
            # Never lose data: put the raw records back at the head of the queue,
            # then reconnect so the next cycle retries them (ON CONFLICT makes
            # this idempotent).
            await self._requeue(raws)
            logger.error("writer: DB write failed, re-queued %d records: %s", len(raws), exc)
            await self._reconnect()

    async def _requeue(self, raws: list[str]) -> None:
        try:
            await self._redis.lpush(config.REDIS_KLINES_KEY, *reversed(raws))
        except Exception as exc:  # pragma: no cover - Redis itself down
            logger.error("writer: re-queue failed (%d records at risk): %s", len(raws), exc)

    async def _flush_remaining(self) -> None:
        """Drain the Redis buffer on shutdown, preserving data on DB failure."""
        while True:
            raws: list[str] = []
            for _ in range(config.BATCH_SIZE):
                raw = await self._redis.lpop(config.REDIS_KLINES_KEY)
                if raw is None:
                    break
                raws.append(raw)
            if not raws:
                break

            batch = []
            for value in raws:
                try:
                    batch.append(_record_to_params(json.loads(value)))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    logger.warning("writer flush: invalid record dropped: %s", exc)
            if not batch:
                continue
            try:
                await self._ensure_connection()
                assert self._db is not None
                async with self._db.cursor() as cur:
                    await cur.executemany(_INSERT_SQL, batch)
                await self._db.commit()
                logger.info("writer: flushed %d rows to timescaledb", len(batch))
            except Exception as exc:
                # DB unreachable at shutdown: leave records in Redis (RDB-persisted)
                # rather than dropping them, and stop trying.
                await self._requeue(raws)
                logger.error("writer: flush failed, %d records left in Redis: %s", len(raws), exc)
                break

        if self._db is not None:
            try:
                await self._db.close()
            except Exception:
                pass
        logger.info("writer: graceful flush complete")
