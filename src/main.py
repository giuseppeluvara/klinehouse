"""Entry point: sets up DB/Redis connections, applies policies, runs
ingester + writer as concurrent asyncio tasks, handles graceful shutdown."""

from __future__ import annotations

import asyncio
import logging
import signal

import psycopg
import redis.asyncio as aioredis

from . import config
from .ingester import Ingester
from .writer import Writer

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


async def _apply_db_policies(conn: psycopg.AsyncConnection) -> None:
    """Add compression and optional retention policies (idempotent)."""
    async with conn.cursor() as cur:
        # Compression policy
        await cur.execute(
            """
            SELECT add_compression_policy(
                'klines',
                compress_after => %s::INTERVAL,
                if_not_exists  => TRUE
            )
            """,
            (config.COMPRESSION_AFTER,),
        )
        logger.info("compression policy: %s", config.COMPRESSION_AFTER)

        # Retention policy (only if configured)
        if config.RETENTION_PERIOD:
            await cur.execute(
                """
                SELECT add_retention_policy(
                    'klines',
                    drop_after    => %s::INTERVAL,
                    if_not_exists => TRUE
                )
                """,
                (config.RETENTION_PERIOD,),
            )
            logger.info("retention policy: %s", config.RETENTION_PERIOD)
        else:
            logger.info("retention policy: disabled (keeping all raw data)")

    await conn.commit()


async def _wait_for_db(dsn: str, retries: int = 30, delay: float = 2.0) -> psycopg.AsyncConnection:
    for attempt in range(1, retries + 1):
        try:
            conn = await psycopg.AsyncConnection.connect(dsn, autocommit=False)
            logger.info("connected to timescaledb")
            return conn
        except Exception as exc:
            if attempt == retries:
                raise
            logger.warning("db not ready (attempt %d/%d): %s — retrying in %.0fs", attempt, retries, exc, delay)
            await asyncio.sleep(delay)
    raise RuntimeError("could not connect to database")  # unreachable but satisfies type checker


async def main() -> None:
    logger.info(
        "starting collector symbols=%s interval=%s",
        config.SYMBOLS,
        config.KLINE_INTERVAL,
    )

    db_conn = await _wait_for_db(config.DB_DSN)
    await _apply_db_policies(db_conn)

    redis_client: aioredis.Redis = aioredis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        decode_responses=True,
    )

    ingester = Ingester(redis_client, db_conn)
    writer = Writer(redis_client, db_conn)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown(signame: str) -> None:
        logger.info("received %s — initiating graceful shutdown", signame)
        stop_event.set()
        ingester.request_stop()
        writer.request_stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown, sig.name)

    ingester_task = asyncio.create_task(ingester.run(), name="ingester")
    writer_task = asyncio.create_task(writer.run(), name="writer")

    # Block until a shutdown signal is received
    await stop_event.wait()

    logger.info("waiting for tasks to finish…")
    await asyncio.gather(ingester_task, writer_task, return_exceptions=True)

    await redis_client.aclose()
    await db_conn.close()
    logger.info("shutdown complete")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
