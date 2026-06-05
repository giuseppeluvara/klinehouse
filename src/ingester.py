"""WebSocket ingester: connects to Binance combined streams, pushes closed
klines to Redis, runs heartbeat, and fills gaps via REST on (re)connect."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx
import redis.asyncio as aioredis
import websockets
from websockets.exceptions import ConnectionClosed

from . import config

logger = logging.getLogger(__name__)

# Binance REST endpoint for historical klines
_KLINES_PATH = "/api/v3/klines"
# Maximum candles per REST request
_REST_PAGE_SIZE = 1000


def _build_ws_url() -> str:
    streams = "/".join(
        f"{sym.lower()}@kline_{config.KLINE_INTERVAL}" for sym in config.SYMBOLS
    )
    return f"{config.WS_BASE_URL}/stream?streams={streams}"


def _interval_to_ms(interval: str) -> int:
    """Convert a Binance interval string to milliseconds."""
    units = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000, "M": 2_592_000_000}
    unit = interval[-1]
    amount = int(interval[:-1])
    return amount * units[unit]


def _kline_to_record(k: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": k["s"],
        "open_time": k["t"],
        "close_time": k["T"],
        "open": k["o"],
        "high": k["h"],
        "low": k["l"],
        "close": k["c"],
        "volume": k["v"],
        "quote_volume": k["q"],
        "trades": k["n"],
        "taker_buy_base": k["V"],
        "taker_buy_quote": k["Q"],
    }


class GapFiller:
    """Fetches missing klines via REST and pushes them to Redis."""

    def __init__(self, redis_client: aioredis.Redis, db_conn: Any) -> None:
        self._redis = redis_client
        self._db = db_conn

    async def fill(self, symbol: str) -> None:
        last_ms = await self._latest_open_time_ms(symbol)
        if last_ms is None:
            logger.info("gap_fill symbol=%s no existing data, skipping", symbol)
            return

        now_ms = int(time.time() * 1000)
        interval_ms = _interval_to_ms(config.KLINE_INTERVAL)
        start_ms = last_ms + interval_ms

        if start_ms >= now_ms:
            logger.debug("gap_fill symbol=%s no gap detected", symbol)
            return

        total = 0
        url = f"{config.REST_BASE_URL}{_KLINES_PATH}"
        async with httpx.AsyncClient(timeout=30) as client:
            while start_ms < now_ms:
                params = {
                    "symbol": symbol,
                    "interval": config.KLINE_INTERVAL,
                    "startTime": start_ms,
                    "limit": _REST_PAGE_SIZE,
                }
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                rows = resp.json()
                if not rows:
                    break

                records = []
                for row in rows:
                    # REST format: [open_time, open, high, low, close, volume,
                    #               close_time, quote_vol, trades, taker_base,
                    #               taker_quote, ignore]
                    records.append(json.dumps({
                        "symbol": symbol,
                        "open_time": row[0],
                        "close_time": row[6],
                        "open": row[1],
                        "high": row[2],
                        "low": row[3],
                        "close": row[4],
                        "volume": row[5],
                        "quote_volume": row[7],
                        "trades": row[8],
                        "taker_buy_base": row[9],
                        "taker_buy_quote": row[10],
                    }))

                if records:
                    await self._redis.rpush(config.REDIS_KLINES_KEY, *records)  # type: ignore[arg-type]
                    total += len(records)

                last_row_open_time = rows[-1][0]
                start_ms = last_row_open_time + interval_ms

                if len(rows) < _REST_PAGE_SIZE:
                    break

        if total:
            logger.info("gap_fill symbol=%s recovered=%d candles", symbol, total)

    async def _latest_open_time_ms(self, symbol: str) -> int | None:
        async with await self._db.cursor() as cur:
            await cur.execute(
                "SELECT EXTRACT(EPOCH FROM MAX(open_time))::BIGINT * 1000 "
                "FROM klines WHERE symbol = %s",
                (symbol,),
            )
            row = await cur.fetchone()
            if row and row[0] is not None:
                return int(row[0])
            return None


class Ingester:
    def __init__(self, redis_client: aioredis.Redis, db_conn: Any) -> None:
        self._redis = redis_client
        self._db = db_conn
        self._filler = GapFiller(redis_client, db_conn)
        self._stop_event = asyncio.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                await self._connect_and_stream()
                backoff = 1.0  # reset on clean disconnect
            except asyncio.CancelledError:
                logger.info("ingester cancelled")
                return
            except Exception as exc:
                logger.error("ingester error: %s — reconnecting in %.0fs", exc, backoff)

            if self._stop_event.is_set():
                break

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass

            backoff = min(backoff * 2, config.RECONNECT_MAX_BACKOFF)

    async def _connect_and_stream(self) -> None:
        url = _build_ws_url()
        logger.info("connecting to %s", url)

        # Fill gaps for all symbols before streaming
        await self._fill_all_gaps()

        async with websockets.connect(
            url,
            ping_interval=config.HEARTBEAT_INTERVAL,
            ping_timeout=config.HEARTBEAT_INTERVAL * 2,
            close_timeout=10,
        ) as ws:
            logger.info("connected symbols=%s interval=%s", config.SYMBOLS, config.KLINE_INTERVAL)
            backoff_reset_task = asyncio.create_task(self._heartbeat_loop())
            try:
                async for raw in ws:
                    if self._stop_event.is_set():
                        break
                    await self._handle_message(raw)
            except ConnectionClosed as exc:
                # Binance closes connections every ~24h; treat as normal
                logger.info("ws connection closed: %s", exc)
            finally:
                backoff_reset_task.cancel()
                try:
                    await backoff_reset_task
                except asyncio.CancelledError:
                    pass

    async def _fill_all_gaps(self) -> None:
        for symbol in config.SYMBOLS:
            try:
                await self._filler.fill(symbol)
            except Exception as exc:
                logger.error("gap_fill symbol=%s error: %s", symbol, exc)

    async def _handle_message(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("received non-JSON message: %r", raw)
            return

        # Combined stream wraps payload under "data"
        data = msg.get("data", msg)
        if data.get("e") != "kline":
            return

        k = data["k"]
        if not k.get("x"):
            # Candle not closed yet
            return

        record = _kline_to_record(k)
        await self._redis.rpush(config.REDIS_KLINES_KEY, json.dumps(record))
        logger.debug("queued closed candle symbol=%s open_time=%s", record["symbol"], record["open_time"])

    async def _heartbeat_loop(self) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            while not self._stop_event.is_set():
                ts = time.time()
                await self._redis.set(config.REDIS_HEARTBEAT_KEY, ts)

                if config.HEALTHCHECK_URL:
                    try:
                        await client.get(config.HEALTHCHECK_URL)
                    except Exception as exc:
                        logger.warning("healthcheck ping failed: %s", exc)

                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=config.HEARTBEAT_INTERVAL,
                    )
                except asyncio.TimeoutError:
                    pass
