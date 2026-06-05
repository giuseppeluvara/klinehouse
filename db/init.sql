-- =============================================================================
-- Binance kline collector — TimescaleDB schema
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Raw klines table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS klines (
    symbol          TEXT               NOT NULL,
    open_time       TIMESTAMPTZ        NOT NULL,
    close_time      TIMESTAMPTZ        NOT NULL,
    -- Prices are received as strings from Binance and stored as DOUBLE PRECISION
    -- for analytics and efficient TimescaleDB compression.
    -- If exact decimal precision is required, change to NUMERIC(20,8) here and
    -- in the continuous-aggregate definitions below.
    open            DOUBLE PRECISION   NOT NULL,
    high            DOUBLE PRECISION   NOT NULL,
    low             DOUBLE PRECISION   NOT NULL,
    close           DOUBLE PRECISION   NOT NULL,
    volume          DOUBLE PRECISION   NOT NULL,
    quote_volume    DOUBLE PRECISION   NOT NULL,
    trades          INTEGER            NOT NULL,
    taker_buy_base  DOUBLE PRECISION   NOT NULL,
    taker_buy_quote DOUBLE PRECISION   NOT NULL,
    PRIMARY KEY (symbol, open_time)
);

-- Convert to hypertable, partitioned by open_time
SELECT create_hypertable('klines', 'open_time', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- Compression
-- ---------------------------------------------------------------------------
ALTER TABLE klines SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby   = 'open_time DESC'
);

-- Compress chunks older than COMPRESSION_AFTER (injected at runtime by the
-- collector; default 7 days).  The policy is added by the collector on
-- startup so it can read the env variable.

-- ---------------------------------------------------------------------------
-- Continuous aggregates
-- ---------------------------------------------------------------------------
-- These views are a customization point: edit time_bucket sizes and
-- aggregated columns to match your analysis needs, then restart the
-- collector to re-apply refresh policies.

CREATE MATERIALIZED VIEW IF NOT EXISTS klines_1m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 minute', open_time)   AS bucket,
    first(open,  open_time)              AS open,
    max(high)                            AS high,
    min(low)                             AS low,
    last(close,  open_time)              AS close,
    sum(volume)                          AS volume,
    sum(quote_volume)                    AS quote_volume,
    sum(trades)                          AS trades
FROM klines
GROUP BY symbol, bucket
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS klines_5m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('5 minutes', open_time)  AS bucket,
    first(open,  open_time)              AS open,
    max(high)                            AS high,
    min(low)                             AS low,
    last(close,  open_time)              AS close,
    sum(volume)                          AS volume,
    sum(quote_volume)                    AS quote_volume,
    sum(trades)                          AS trades
FROM klines
GROUP BY symbol, bucket
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS klines_1h
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 hour', open_time)     AS bucket,
    first(open,  open_time)              AS open,
    max(high)                            AS high,
    min(low)                             AS low,
    last(close,  open_time)              AS close,
    sum(volume)                          AS volume,
    sum(quote_volume)                    AS quote_volume,
    sum(trades)                          AS trades
FROM klines
GROUP BY symbol, bucket
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS klines_1d
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 day', open_time)      AS bucket,
    first(open,  open_time)              AS open,
    max(high)                            AS high,
    min(low)                             AS low,
    last(close,  open_time)              AS close,
    sum(volume)                          AS volume,
    sum(quote_volume)                    AS quote_volume,
    sum(trades)                          AS trades
FROM klines
GROUP BY symbol, bucket
WITH NO DATA;

-- Refresh policies (aggregate 1 minute behind realtime to avoid partial buckets)
SELECT add_continuous_aggregate_policy('klines_1m',
    start_offset => INTERVAL '10 minutes',
    end_offset   => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('klines_5m',
    start_offset => INTERVAL '30 minutes',
    end_offset   => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('klines_1h',
    start_offset => INTERVAL '2 hours',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('klines_1d',
    start_offset => INTERVAL '3 days',
    end_offset   => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE);
