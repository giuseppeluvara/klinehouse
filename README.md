# klinehouse

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)

Real-time Binance OHLCV kline collector that streams closed candlesticks via WebSocket and persists them into TimescaleDB, building a complete, gap-free historical dataset that runs unattended — **plus a research layer that turns that data into honest, cost-aware microstructure analytics, a Grafana radar, and a delta-neutral funding-carry money tool.**

---

## Features

**Collector**
- **Multi-symbol**: collect any number of Binance symbols simultaneously over a single combined WebSocket.
- **All intervals**: `1s`, `1m`, `5m`, `1h`, … — any Binance-supported interval, set via env var.
- **Gap-free**: on every (re)connect the collector back-fills missing candles via REST before resuming the stream; `scripts/backfill.py` also heals *interior* gaps anywhere in history.
- **Automatic reconnection**: exponential back-off for both the WebSocket and the **database connection** — a DB restart never loses in-flight candles (they are re-queued to Redis).
- **Idempotent writes**: `ON CONFLICT DO NOTHING` — safe to restart at any time, no duplicate rows.
- **Continuous aggregates**: pre-built 1 m / 5 m / 1 h / 1 d OHLCV materialized views with automatic refresh policies.
- **Compression, retention & backups**: TimescaleDB compression by default; optional retention; restore-tested logical backups (`scripts/backup.sh` + opt-in sidecar).
- **Graceful shutdown / monitoring / no API key / multi-arch**: as you'd expect for a 24-7 unattended service.

**Research & money layer** (`research/`, `db/analytics.sql`, `grafana/`)
- **Microstructure analytics**: order-flow imbalance, taker pressure, realized-volatility — as Python features and queryable SQL views.
- **Honest signal harness**: information coefficient, decile monotonicity, **cost-aware backtests** that always show the breakeven fee.
- **Execution-alpha engine**: causal, order-aware child-order scheduler that beats TWAP on short windows (validated, ~+0.1 bps).
- **Grafana radar**: provisioned order-flow & volatility dashboard.
- **💰 Funding-carry tools**: a scanner that ranks live delta-neutral carry opportunities (the one genuinely profitable, market-neutral strategy found) and a paper carry book validated on real funding settlements.

See **[`research/README.md`](research/README.md)** for the full, honest "can this make money?" write-up.

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

### Lightweight 1-second archive

Keep the live collector and recent Timescale data, archive closed UTC days as
verified Zstandard Parquet, and query history directly with DuckDB. Export and
query paths process bounded batches; the archive retains every original column,
including taker-buy volumes. Nothing deletes database rows by default.

```bash
pip install -r requirements-storage.txt
python scripts/archive_market.py --root archive sync
python scripts/archive_market.py --root archive verify
```

See **[docs/storage.md](docs/storage.md)** for streaming queries, the optional
archive worker, monthly compaction and backed-up, verified chunk pruning.
The existing `scripts/export_parquet.py --symbol ... --start ... --end ... --out ...`
command remains available, now writing the same verified day-partitioned archive.

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
- After extended downtime, run `python scripts/backfill.py` to heal any **interior** gaps via REST. The collector's own recovery only fills the *trailing* gap (latest candle → now); this tool finds and fills holes anywhere in the history (idempotent, safe to run live, cron-friendly).

For a bounded leading-history seed, reuse the same tool:

```bash
python scripts/backfill.py --from 2026-05-09
```

The command keeps serial, idempotent writes and stops itself at 15 GB by
default (`--max-db-bytes`) without stopping the live collector. It also refuses
to start with less than 20 GiB free on the host.

---

## Research, analytics & making money

The `research/` layer turns the collected data into decisions. It is built on
**intellectual honesty**: every profitability number is net of fees, and the
search reports what *doesn't* work as loudly as what does.

```bash
pip install -r research/requirements.txt

# Honest microstructure report (IC, deciles, cost-aware backtest, vol forecast)
python -m research.report --symbol BTCUSDT          # -> research/out/report.md + figures

# Execution-alpha engine (causal, order-aware) — beats TWAP on short windows
python -m research.execution_engine --symbol BTCUSDT --side buy --qty 10 --window 60s

# 💰 The money-maker: rank live delta-neutral funding-carry opportunities
python -m research.funding_scanner --top 15
python -m research.carry_book simulate DEXEUSDT --days 14 --notional 10000
```

**The honest verdict** (full detail in [`research/README.md`](research/README.md)):

- Directional scalping, cross-asset stat-arb and retail market-making **do not
  beat fees** on these efficient markets — proven, not assumed.
- What *does* pay is **delta-neutral funding-carry** (long spot + short perp,
  collect funding, no price view): a real, market-neutral risk premium, ~8 %/yr
  net on liquid names in calm markets and far higher in leverage-frenzy regimes.
- Plus two operational edges: **execution-cost reduction** (~+0.1 bps on flow you
  already trade) and the **proprietary 1-second order-flow dataset** itself.

No strategy here places real orders: the tools are paper/advisory by design (the
collector never holds API keys).

---

## Dashboard (Grafana)

A provisioned **order-flow & volatility radar** ships with the stack.

```bash
# 1. create the analytics views the dashboard reads
docker compose exec -T timescaledb psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" < db/analytics.sql
docker compose exec timescaledb psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "CALL refresh_continuous_aggregate('flow_1m', NULL, NULL);"

# 2. start Grafana
docker compose up -d grafana
open http://localhost:3000      # default login: admin / admin
```

The dashboard (folder *klinehouse*) has a `$symbol` selector and panels for
price vs cumulative order-flow imbalance, the `ofi_norm` buyers-vs-sellers bar,
the realized-volatility radar, taker-buy ratio, and live gauges. Datasource and
dashboard are provisioned as code under `grafana/`. Set `GRAFANA_PORT` /
`GRAFANA_ADMIN_PASSWORD` in `.env`.

---

## Durability & backups

What is protected, and how:

| Failure | Outcome | Mechanism |
|---|---|---|
| Collector crash / restart | No data loss | Closed candles are committed; on restart the collector back-fills any trailing gap via REST. |
| **Database** restart / brief outage | No data loss, auto-recovery | The writer **re-queues** in-flight records to Redis and **reconnects** with backoff; committed data is on the named volume (WAL/ACID — a crash never truncates the table). |
| `docker compose down` (no `-v`) | No data loss | Named volume `timescaledb_data` persists. |
| `docker compose down -v`, volume removal, disk failure | Data lost **unless backed up** | Use the backups below. |

**Automated backups** (logical, compressed, restore-tested):

```bash
# One-off (cron-friendly)
scripts/backup.sh                         # -> ./backups/klines_<ts>.dump

# Or run a backup sidecar with the stack (daily by default)
docker compose --profile backup up -d
```

Tune via env: `BACKUP_INTERVAL_SECONDS` (default `86400`), `BACKUP_RETENTION` (default `14`).

**Restore** into a side database (your live data is never overwritten):

```bash
scripts/restore.sh backups/klines_<ts>.dump            # -> <db>_restored
# verify, then promote by stopping the collector and renaming databases
```

The restore script follows the official TimescaleDB `pre/post_restore` procedure,
so hypertables, continuous aggregates and compression are all reinstated.

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

## Reproducible Timestral pilot

The separate [Timestral experiment](docs/timestral.md) compares zero-shot price
and past-covariate forecasts against simple baselines. The
[2026-10-25 report](reports/timestral-20261025/README.md) includes real Binance
data measurements, saved forecasts, limitations and reproducible commands.
Historical scores are exploratory; no trading edge or prospective completion
is claimed.

The [four-hour and daily follow-up](docs/timestral-long-horizon.md) builds hourly
inputs from 10.5 million verified second candles and evaluates 552 four-hour and
92 daily anchors. Its [report and complete forecast CSV](reports/timestral-long-horizon-20261025/README.md)
record the observed errors, uncertainty and resource use.

The [frozen adaptation experiment](docs/timestral-adaptation.md) selects a
forecast rule on October–December 2025 data and evaluates it on separate
January–March 2026 samples. Its [results, forecasts and independent audits](reports/timestral-adaptation-20261025/README.md)
compare the selected rule with unchanged price using a predeclared evidence gate.
The historical design was informed by the later May–July scores; it is not a
prospective validation.

The [diagnosis and repair](docs/timestral-repair.md) adds baseline-aware monthly
selection, causal volatility-scaled intervals and a direct horizon-return
hypothesis. Its [executed diagnostic report](reports/timestral-diagnosis-20261026/README.md)
preserves the negative point-forecast result and reports interval improvements
on previously observed history; prospective validation remains pending.

The [root-cause diagnostics](docs/timestral-diagnostics.md) correct the attribution
of interval scores, test numerical equivalence and known synthetic signals, and
compare overlapping versus disjoint horizon returns on identical past windows.
The [executed report](reports/timestral-root-cause-20261026/README.md) retains both
the improvements over the flawed representation and the losses to unchanged price.

The [capacity, calibration, funding and history comparisons](reports/timestral-capacity-20261026/README.md)
test four further explanations with registered protocols and paired samples.
Neither a larger Timestral checkpoint, loss-aligned calibration, settled funding,
nor three years of training data passes the historical screen. The report includes
compressed forecasts and an independent arithmetic and causality audit; it does
not claim a functioning trading strategy.

The [direction/variance decomposition and transfer tests](reports/timestral-decomposition-20261027/README.md)
find useful historical four-hour variance forecasts on BTC and ETH using a
calibrated Gamma model. Direction remains unvalidated, and the full ETH/SOL
transfer gate fails because SOL does not pass every cohort. The report preserves
the initial failed screen, explains the separate mean and expected-log outputs,
and includes an independently verified single-origin forecast command.

The [minute direction and selective reversal experiment](reports/timestral-microdirection-20261027/README.md)
tests a different hypothesis using existing second-level BTC candles and new,
pre-protected native minute ETH/SOL data. Minute flow and last-ten-second flow
fail the BTC screen. A frozen sign-symmetric reversal rule passes the individual
historical ETH screen at 52.74% accuracy on 24.37% of five-minute anchors with a
one-minute delay. BTC and SOL fail; the combined transfer gate remains false.
Saved predictions, independent reconstruction and a historical forecast preview
are included. This is a selective research candidate, not prospective validation.

The [shared-model generalization study](reports/timestral-universal-20261027/README.md)
fits one common directional model across BTC, ETH and SOL. All three pass the
January-July 2026 screen, but locked replications in 2024 and 2025 fail the combined
criterion. A fixed nonlinear follow-up also fails the full nine-cohort gate.
Independent audits and a shared-fit historical preview preserve the positive
recent results and the limits of generalization without opening protected data.

## Adaptive direction follow-up

The [adaptive direction study](reports/timestral-adaptive-direction-20261027/README.md)
tests daily probability calibration and changing mixtures of frozen experts on
the same nine 2024–2026 asset/cohort comparisons. Both pass five of nine direction
gates; neither establishes a general improvement. The report separates flat-price
calibration from up/down prediction and includes independent causal replay audits.

## Derivatives information and up/down direction

The [premium-index study](reports/timestral-premium-direction-20261027/README.md)
adds a new perpetual-futures source while keeping flat-price probabilities fixed.
Across all nine 2024–2026 asset/cohort checks, it finds no verified incremental
up/down benefit over the spot-only control. All original evaluation anchors,
source gaps, frozen protocols and independent audits are preserved.

The [futures volume and aggressor-flow study](reports/timestral-futures-flow-20261027/README.md)
compares spot, spot with futures activity, and spot with activity plus taker flow.
Across January–July 2024, 2025 and 2026, all three incremental-information
comparisons fail in every asset/year cohort. The flow model passes five of nine
original direction gates versus six for spot. Ninety checksummed source files,
63 monthly fits and independent raw-data/score audits preserve this negative
result without opening protected holdouts.

The [recurring-regime study](reports/timestral-regime-direction-20261027/README.md)
separates changing state frequencies from changing within-state fixed-rule
accuracy. Long context and conditional slopes fail to establish a general signal;
the regime model fits all 21 training windows better but scores worse in all
nine subsequent cohorts. The report preserves the registered comparisons,
diagnostic uncertainty and independent reconstruction without protected holdouts.

## License

[MIT](LICENSE)
