# klinehouse

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)

Real-time Binance OHLCV kline collector that streams closed candlesticks via WebSocket and persists them into TimescaleDB, building a complete, gap-free historical dataset that runs unattended.

---

## Features

- **Multi-symbol**: collect any number of Binance symbols simultaneously over a single combined WebSocket.
- **All intervals**: `1s`, `1m`, `5m`, `1h`, … — any Binance-supported interval, set via env var.
- **Gap-free**: on every (re)connect the collector back-fills missing candles via REST before resuming the stream.
- **Automatic reconnection**: exponential back-off (1 s → capped at `RECONNECT_MAX_BACKOFF`). Binance's ~24 h connection limit is handled transparently.
- **Idempotent writes**: `ON CONFLICT DO NOTHING` — safe to restart at any time, no duplicate rows.
- **Continuous aggregates**: pre-built 1 m / 5 m / 1 h / 1 d OHLCV materialized views with automatic refresh policies.
- **Compression & retention**: TimescaleDB compression enabled by default; optional data-expiry policy.
- **Graceful shutdown**: `SIGTERM` / `SIGINT` flushes the in-memory Redis buffer to the DB before exiting.
- **Monitoring**: optional external ping (Healthchecks.io / Uptime Kuma) and a Docker `healthcheck` on the collector container.
- **No API key required**: market data streams are public; authentication is never requested or stored.
- **Multi-arch**: runs on Apple Silicon (`arm64`) and x86-64 Linux without changes.

---

## Requirements

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/install/) ≥ 2.20 (included in Docker Desktop)

No other local dependencies.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/giuseppeluvara/klinehouse.git
cd klinehouse

# 2. Configure
cp .env.example .env
# Edit .env — set POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB at minimum

# 3. Start
docker compose up -d

# 4. Watch logs
docker compose logs -f collector

# 5. Verify rows are accumulating
docker compose exec timescaledb psql -U $POSTGRES_USER -d $POSTGRES_DB \
  -c "SELECT symbol, count(*), max(open_time) FROM klines GROUP BY symbol;"
```

---

## Configuration

All settings are passed as environment variables (in `.env` or the shell). Every variable has a default except the three required database credentials.

| Variable | Default | Description |
|---|---|---|
| `SYMBOLS` | `BTCUSDT` | Comma-separated list of symbols, e.g. `BTCUSDT,ETHUSDT` |
| `KLINE_INTERVAL` | `1s` | Any Binance-supported interval (`1s` `1m` `5m` `1h` …) |
| `BATCH_SIZE` | `50` | Flush when this many rows are buffered in Redis |
| `BATCH_INTERVAL_SECONDS` | `2` | Flush after this many seconds even if `BATCH_SIZE` not reached |
| `COMPRESSION_AFTER` | `7 days` | Compress TimescaleDB chunks older than this |
| `RETENTION_PERIOD` | *(empty)* | Drop data older than this (e.g. `90 days`). Empty = keep all data |
| `WS_BASE_URL` | `wss://stream.binance.com:9443` | WebSocket base URL (override for testnet) |
| `REST_BASE_URL` | `https://api.binance.com` | REST base URL (override for testnet) |
| `RECONNECT_MAX_BACKOFF` | `60` | Maximum reconnection back-off in seconds |
| `HEARTBEAT_INTERVAL` | `10` | Seconds between heartbeats written to Redis |
| `HEALTHCHECK_URL` | *(empty)* | HTTP ping URL on each heartbeat. Empty = disabled |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `POSTGRES_USER` | — | **Required.** Database username |
| `POSTGRES_PASSWORD` | — | **Required.** Database password |
| `POSTGRES_DB` | — | **Required.** Database name |
| `POSTGRES_HOST` | `timescaledb` | Database hostname (internal Docker network name) |
| `POSTGRES_PORT` | `5432` | Database port (internal) |
| `POSTGRES_PORT_HOST` | `5432` | Host-side published port (change to avoid clashes) |
| `REDIS_HOST` | `redis` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |

---

## Architecture

```
Binance combined WebSocket  (<symbol>@kline_<interval>, one stream per symbol)
        │  closed candles only (k.x == true)
        ▼
   [ ingester ]  ──RPUSH──►  Redis list "klines:buffer"
        │  heartbeat ping                │
        │  (HEALTHCHECK_URL)           BLPOP batches
        ▼                                ▼
  Redis key                        [ writer ]  ──INSERT ON CONFLICT DO NOTHING──►  TimescaleDB
  "ingester:last_beat"                                                                  │
                                                                          continuous aggregates
                                                                          klines_1m / 5m / 1h / 1d
                                                                          compression policy
                                                                          optional retention policy
```

The ingester and writer run as two `asyncio` tasks inside a single container (`collector`).
TimescaleDB and Redis each run in their own container with named volumes for persistence.

### Gap recovery

On every (re)connection the ingester:
1. reads the latest `open_time` for each symbol from TimescaleDB;
2. calls `GET /api/v3/klines` with paginated REST requests to fetch the missing range;
3. pushes the recovered candles into Redis — the writer persists them like any live candle.

This means a network outage or a container restart never leaves a gap in the time series.

---

## Example queries

```sql
-- Latest rows per symbol
SELECT symbol, open_time, close, volume
FROM klines
ORDER BY symbol, open_time DESC
LIMIT 20;

-- 1-minute OHLCV aggregates (last 10 buckets)
SELECT * FROM klines_1m
WHERE symbol = 'BTCUSDT'
ORDER BY bucket DESC
LIMIT 10;

-- Gap detection — any open_time jump larger than the collection interval
SELECT
    symbol,
    open_time,
    open_time - lag(open_time) OVER (PARTITION BY symbol ORDER BY open_time) AS gap
FROM klines
ORDER BY gap DESC NULLS LAST
LIMIT 10;

-- Row count per symbol
SELECT symbol, count(*), min(open_time), max(open_time)
FROM klines
GROUP BY symbol;
```

---

## Customization

### Adding or changing symbols

Set `SYMBOLS` in `.env` and restart the collector:

```env
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
```

### Changing the collection interval

Set `KLINE_INTERVAL` in `.env`:

```env
KLINE_INTERVAL=1m
```

The same interval is used for both the WebSocket stream and REST gap-filling.

### Editing continuous aggregates

The four materialized views (`klines_1m`, `klines_5m`, `klines_1h`, `klines_1d`) are defined in
`db/init.sql`. To add or change a bucket:

1. Edit `db/init.sql`.
2. Apply the change to the database.

**Fresh setup (simplest):** tear down and recreate the volume so `init.sql` runs from scratch:
```bash
docker compose down -v && docker compose up -d
```

**Existing deployment (no data loss):** alter the views in place via TimescaleDB without touching
the raw `klines` hypertable:
```sql
DROP MATERIALIZED VIEW klines_1m CASCADE;
CREATE MATERIALIZED VIEW klines_1m ...;  -- paste the updated definition from init.sql
```
Repeat for each view you changed. The raw tick data is never affected.

### Enabling data retention

Set `RETENTION_PERIOD` in `.env` to a PostgreSQL interval string:

```env
RETENTION_PERIOD=90 days
```

Leave it empty (the default) to keep all raw data indefinitely.

### Precision

Prices are stored as `DOUBLE PRECISION` (64-bit float) for analytics performance and efficient
TimescaleDB compression. If exact decimal precision is required (e.g. for settlement calculations),
change every price column in `db/init.sql` to `NUMERIC(20,8)` and reset the volume.

### Parquet export

Export a date range to Parquet for cold archival or offline analysis:

```bash
pip install pyarrow
python scripts/export_parquet.py \
  --symbol BTCUSDT \
  --start  2024-01-01 \
  --end    2024-02-01 \
  --out    exports/
```

The output file can be read with pandas, DuckDB, Spark, or any Parquet-compatible tool.

---

## Deployment

The project runs identically on a local machine, a cloud VPS, or a dedicated server.
To move an existing installation:

```bash
# On the new host
git clone https://github.com/giuseppeluvara/klinehouse.git
cd klinehouse
cp /path/to/old/.env .env   # copy your existing configuration
docker compose up -d
```

Gap recovery ensures no candles are missed during the migration window.

For long-running production deployments:
- Use a systemd service or a managed container platform to restart Docker Compose on host reboot.
- Point `HEALTHCHECK_URL` at a monitoring service (Healthchecks.io, Uptime Kuma) to be alerted if the collector stops sending heartbeats.

---

## Verification checklist

```bash
# 1. All containers healthy
docker compose ps

# 2. Rows accumulating
docker compose exec timescaledb psql -U $POSTGRES_USER -d $POSTGRES_DB \
  -c "SELECT symbol, count(*), max(open_time) FROM klines GROUP BY symbol;"

# 3. Aggregates available
docker compose exec timescaledb psql -U $POSTGRES_USER -d $POSTGRES_DB \
  -c "SELECT * FROM klines_1m ORDER BY bucket DESC LIMIT 5;"

# 4. Gap check (no row should have a gap larger than the collection interval)
docker compose exec timescaledb psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
  SELECT symbol, open_time,
         open_time - lag(open_time) OVER (PARTITION BY symbol ORDER BY open_time) AS gap
  FROM klines ORDER BY gap DESC NULLS LAST LIMIT 10;"

# 5. Stop the collector for ~60 s, restart, verify the gap is filled
docker compose stop collector
sleep 90
docker compose start collector
# Re-run query 4 — no gap should appear for the pause window
```

---

## License

[MIT](LICENSE)
