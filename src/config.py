"""Centralised configuration loaded from environment variables."""

from __future__ import annotations

import os


def _get(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def _require(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Required environment variable '{key}' is not set.")
    return value


# ---------------------------------------------------------------------------
# Symbols & interval
# ---------------------------------------------------------------------------
SYMBOLS: list[str] = [s.strip().upper() for s in (_get("SYMBOLS", "BTCUSDT") or "BTCUSDT").split(",") if s.strip()]
KLINE_INTERVAL: str = _get("KLINE_INTERVAL", "1s") or "1s"

# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
BATCH_SIZE: int = int(_get("BATCH_SIZE", "50") or 50)
BATCH_INTERVAL_SECONDS: float = float(_get("BATCH_INTERVAL_SECONDS", "2") or 2)

# ---------------------------------------------------------------------------
# TimescaleDB policies
# ---------------------------------------------------------------------------
COMPRESSION_AFTER: str = _get("COMPRESSION_AFTER", "7 days") or "7 days"
RETENTION_PERIOD: str | None = _get("RETENTION_PERIOD") or None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
WS_BASE_URL: str = _get("WS_BASE_URL", "wss://stream.binance.com:9443") or "wss://stream.binance.com:9443"
REST_BASE_URL: str = _get("REST_BASE_URL", "https://api.binance.com") or "https://api.binance.com"

# ---------------------------------------------------------------------------
# Reconnection
# ---------------------------------------------------------------------------
RECONNECT_MAX_BACKOFF: float = float(_get("RECONNECT_MAX_BACKOFF", "60") or 60)

# ---------------------------------------------------------------------------
# Heartbeat & monitoring
# ---------------------------------------------------------------------------
HEARTBEAT_INTERVAL: float = float(_get("HEARTBEAT_INTERVAL", "10") or 10)
HEALTHCHECK_URL: str | None = _get("HEALTHCHECK_URL") or None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = (_get("LOG_LEVEL", "INFO") or "INFO").upper()

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
POSTGRES_HOST: str = _get("POSTGRES_HOST", "timescaledb") or "timescaledb"
POSTGRES_PORT: int = int(_get("POSTGRES_PORT", "5432") or 5432)
POSTGRES_USER: str = _require("POSTGRES_USER")
POSTGRES_PASSWORD: str = _require("POSTGRES_PASSWORD")
POSTGRES_DB: str = _require("POSTGRES_DB")

# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
REDIS_HOST: str = _get("REDIS_HOST", "redis") or "redis"
REDIS_PORT: int = int(_get("REDIS_PORT", "6379") or 6379)

# ---------------------------------------------------------------------------
# Internal constants (not user-facing)
# ---------------------------------------------------------------------------
REDIS_KLINES_KEY = "klines:buffer"
REDIS_HEARTBEAT_KEY = "ingester:last_beat"

# DSN for psycopg
DB_DSN = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
