-- =============================================================================
-- klinehouse analytics layer — make the order-flow data concrete in SQL/Grafana
-- =============================================================================
-- These objects are *additive* and safe to (re)run: they only read `klines`.
-- They expose the microstructure quantities that the research layer proved
-- meaningful — order-flow imbalance, taker pressure, realized volatility —
-- directly as queryable views, ready to chart in Grafana.
--
-- Apply with:
--   docker exec -i klinehouse-timescaledb-1 psql -U $POSTGRES_USER -d $POSTGRES_DB < db/analytics.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1) Per-minute order-flow continuous aggregate
-- ---------------------------------------------------------------------------
-- Only raw sums live in the materialized view (the safe subset for continuous
-- aggregates); ratios are derived in the view below. `taker_buy_base` is the
-- aggressive-buy volume; total minus that is the aggressive-sell volume.
CREATE MATERIALIZED VIEW IF NOT EXISTS flow_1m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 minute', open_time) AS bucket,
    first(open, open_time)             AS open,
    last(close, open_time)             AS close,
    sum(volume)                        AS volume,
    sum(quote_volume)                  AS quote_volume,
    sum(taker_buy_base)                AS taker_buy_base,
    sum(taker_buy_quote)               AS taker_buy_quote,
    sum(trades)                        AS trades,
    count(*)                           AS n_candles,
    count(*) FILTER (WHERE trades = 0) AS empty_candles
FROM klines
GROUP BY symbol, bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('flow_1m',
    start_offset      => INTERVAL '30 minutes',
    end_offset        => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists     => TRUE);

-- ---------------------------------------------------------------------------
-- 2) Derived order-flow metrics (ratios) on top of the aggregate
-- ---------------------------------------------------------------------------
--   taker_buy_ratio : share of volume from aggressive buyers (0.5 = balanced)
--   ofi_norm        : net signed volume / total volume, in [-1, 1]
--                     (>0 = buyers in control, <0 = sellers in control)
--   signed_volume   : aggressive buy minus aggressive sell base volume
CREATE OR REPLACE VIEW flow_metrics_1m AS
SELECT
    symbol,
    bucket,
    open,
    close,
    volume,
    quote_volume,
    trades,
    (2 * taker_buy_base - volume)                              AS signed_volume,
    CASE WHEN volume > 0 THEN taker_buy_base / volume END      AS taker_buy_ratio,
    CASE WHEN volume > 0
         THEN (2 * taker_buy_base - volume) / volume END       AS ofi_norm,
    empty_candles,
    n_candles
FROM flow_1m;

-- ---------------------------------------------------------------------------
-- 3) Per-minute realized volatility from the raw 1s returns (computed live)
-- ---------------------------------------------------------------------------
-- Realized volatility needs intra-minute return ordering, so it is a plain
-- view over `klines` rather than a continuous aggregate. In bps.
-- Returns spanning a collection gap are excluded (a 2-day gap is not a real
-- 1-second move): a return is only counted when the previous candle is within
-- 5 seconds. Widen this guard if collecting at a coarser KLINE_INTERVAL.
CREATE OR REPLACE VIEW realized_vol_1m AS
WITH r AS (
    SELECT
        symbol,
        open_time,
        ln(close / NULLIF(lag(close) OVER w, 0)) AS logret,
        EXTRACT(EPOCH FROM (open_time - lag(open_time) OVER w)) AS dt_s
    FROM klines
    WINDOW w AS (PARTITION BY symbol ORDER BY open_time)
)
SELECT
    symbol,
    time_bucket('1 minute', open_time) AS bucket,
    sqrt(sum(logret * logret)) * 1e4    AS realized_vol_bps,
    count(*)                            AS n
FROM r
WHERE logret IS NOT NULL AND dt_s <= 5
GROUP BY symbol, bucket;

-- ---------------------------------------------------------------------------
-- Example queries (for Grafana panels or ad-hoc analysis)
-- ---------------------------------------------------------------------------
-- Latest order-flow pressure (last hour):
--   SELECT bucket, close, ofi_norm, taker_buy_ratio
--   FROM flow_metrics_1m
--   WHERE symbol = 'BTCUSDT' AND bucket > now() - INTERVAL '1 hour'
--   ORDER BY bucket;
--
-- Cumulative signed flow vs price (Grafana dual-axis):
--   SELECT bucket, close,
--          sum(signed_volume) OVER (PARTITION BY symbol ORDER BY bucket) AS cum_flow
--   FROM flow_metrics_1m WHERE symbol = 'BTCUSDT' ORDER BY bucket;
--
-- Volatility radar (where was it stormy?):
--   SELECT bucket, realized_vol_bps FROM realized_vol_1m
--   WHERE symbol = 'BTCUSDT' ORDER BY realized_vol_bps DESC LIMIT 20;
