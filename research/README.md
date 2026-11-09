# klinehouse — research & analytics layer

This is the part of the project that turns the collected candles into **decisions
and value**, not just rows in a table. It is a small, honest quantitative
research toolkit that runs against the same TimescaleDB the collector fills.

Its guiding principle is **intellectual honesty**: every profitability figure is
reported *net of fees*, every claim is bounded by how much real continuous data
backs it, and the toolkit is deliberately built to expose a signal that is
statistically real but economically useless — because that is exactly what the
data shows at first, and pretending otherwise is how people lose money.

---

## TL;DR — what the data actually says

Analysed on ~6 continuous days of 1-second BTCUSDT data (≈522k rows, after
REST gap-healing) plus ~2.5 days of ETHUSDT/SOLUSDT; findings reproduce across
independent sub-periods and, more weakly, across symbols (see the *data map*
caveat below):

| Question | Honest answer |
|---|---|
| Is there a real predictive signal? | **Yes.** Order-flow imbalance (built from `taker_buy_base`) predicts the next **2–5 second** return. Spearman IC ≈ **+0.09 to +0.16**, with perfectly **monotonic deciles**, reproduced on both independent segments. |
| Can you scalp it for profit? | **No — structurally.** The gross edge is **~0.1–0.15 bps per trade**; breakeven fee is **~0.05–0.07 bps/side**. The cheapest real crypto fee (~2 bps/side) is **30–200× larger**. It is not a tuning problem; tuning cannot close a 100× gap. |
| Why does it die? | The edge **decays to ~0 by the 1-minute horizon** (minute-level IC ≈ 0.00). It lives in seconds, where you'd pay the fee dozens of times. |
| So where *is* the value? | Not in predicting price. In a **structural risk premium** (funding) + operational edges. See below. |

---

## Making money — the honest verdict

The historical experiments do not establish a currently profitable strategy.
The carry claim below was revised after a full two-leg accounting audit on
2026-11-04; see `reports/carry-audit-20261104/results.json`.

| Approach | Observed limit |
|---|---|
| Directional scalping on order-flow | Measured gross edge is smaller than modeled fees. |
| Cross-asset stat-arb | Existing experiments do not establish net profitability. |
| Retail market-making | Fill probability, adverse selection and fees remain binding. |
| Funding carry | A research candidate. Positive funding alone is insufficient: basis, fees, slippage, capital and margin matter. |

`funding_scanner.py` is an exploratory ranking of historical/current funding,
not a forward yield estimate. Current rates use the configured settlement
interval; historical rates use elapsed time between actual settlements.
`carry_book.py` is a funding-only illustration using fixed notional; it omits
basis PnL, mark-dependent settlement notional, capital and liquidation risk.
Its `net` field is not a complete strategy PnL or a validated return on capital.

`carry_accounting.py` separately accounts for fixed matched base quantities,
spot and short-perp PnL, actual settlement marks, four fill costs and capital
cost. `scripts/audit_carry_history.py` preserves raw responses and hashes for
three fixed 30-day windows on BTC/ETH/SOL. All nine windows lost money under
the declared assumptions. Candle opens remain price proxies; intraperiod
liquidation and funding completeness are not certified. These are historical
diagnostics, not prospective validation.

### The three further places real value lives

1. **Execution alpha — money on flow you trade anyway.**
   If you already need to buy or sell (DCA, rebalancing, entering a multi-hour
   position), you pay the fee *regardless*. Timing the child orders of that parent
   order toward favourable seconds — using the same order-flow signal — lowers your
   average buy price / raises your average sell price for free.
   Measured **causally** (1-second reaction lag, no intra-second peeking) vs a TWAP
   baseline by `execution_engine.py`: **~+0.10 to +0.12 bps for short windows
   (≤ ~2 min), t ≈ 6–9 on BTCUSDT** (weaker but still significant on ETH/SOL,
   t ≈ 2–3); the edge **vanishes by 5-minute windows**. That is ~€10 per €1M of
   flow — small, but free and scaling with size.
   ⚠️ An *intra-second look-ahead* schedule shows ~+0.6 bps, but that assumes you
   can act on a second's order flow within that same second; it is **not
   achievable** and the engine reports it only as a ceiling. Win rate is just
   below 50% with a positive mean — a few good calls outweigh many small misses.

2. **Volatility & liquidity forecasting — risk, sizing, gating.**
   Return *direction* is barely predictable; return *magnitude* is strongly
   forecastable (volatility clusters). A plain EWMA forecast orders calm vs stormy
   periods with **rank-IC ≈ +0.40 to +0.50** out-of-sample. That powers position
   sizing, stop placement, "don't trade into a vol spike" gating, and an
   options/perp risk view — none of which require beating fees on direction.

3. **The dataset as a proprietary asset.**
   Gap-free 1-second OHLCV **with order-flow** is scarce: Binance's public REST
   1s history is short-lived, so the only way to own a long, clean record is to
   *collect it continuously* — which is exactly what klinehouse does. The longer
   it runs, the more it becomes a moat for any of the above.

---

## Module map

| File | Role |
|---|---|
| `db.py` | Gap-aware loading from TimescaleDB (or CSV). `add_segments` / `segment_summary` label the continuous runs so analysis never crosses a collection gap. |
| `features.py` | Causal microstructure features: signed volume, taker ratio, normalised order-flow imbalance (`ofi_n*`), rolling realized vol, trade intensity. Forward returns (targets) are kept strictly separate, with a 1s execution lag. |
| `evaluate.py` | The honest harness: information coefficient (non-overlapping t-stats), decile monotonicity, **cost-aware backtest**, breakeven-fee, fee sweep. |
| `volatility.py` | EWMA volatility forecast + out-of-sample skill evaluation. |
| `execution.py` | Research measure of signal-tilted vs TWAP fills (contemporaneous = look-ahead upper bound). |
| `execution_engine.py` | **Production** execution-alpha engine: causal, predictive, order-aware child-order scheduler + live advisor + honest validation/window-sweep. |
| `funding_scanner.py` | **💰 The money tool.** Live scan of Binance funding × spot/perp liquidity × persistence → ranked, risk-adjusted, fee-net delta-neutral carry opportunities. |
| `carry_book.py` | Paper carry book: tracks delta-neutral funding harvest on **real settlements** (forward) + instant historical what-if. |
| `report.py` | CLI that runs everything and writes `out/report.md` + figures. |
| `../db/analytics.sql` | The same ideas as **queryable SQL views** (`flow_metrics_1m`, `realized_vol_1m`) — ready for Grafana, no Python needed. |

---

## Quick start

```bash
# from the project root, with the collector stack already up
pip install -r research/requirements.txt

# full report against the live database
python -m research.report --symbol BTCUSDT

# ...or fully offline from a CSV export
docker exec klinehouse-timescaledb-1 psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "\copy (SELECT * FROM klines ORDER BY open_time) TO STDOUT WITH CSV HEADER" > /tmp/klines.csv
python -m research.report --symbol BTCUSDT --csv /tmp/klines.csv
```

Output lands in `research/out/`: a `report.md` and PNG figures (information-
coefficient curve, decile bars, fee-sweep, price-vs-cumulative-flow).

### Frozen BTC sequence experiment

`sequence_experiment.py` is the single path for the separate, predictive-only
comparison specified in `../PLAN.md`. It uses Binance Spot 1-minute archives,
hourly non-overlapping BTC labels, the same causal `120x12` tensor for every
model, three fixed walk-forward folds and Transformer seeds 7/17/29.

```bash
python -m research.sequence_experiment prepare --cutoff 2026-08-07T00:00:00Z
python -m research.sequence_experiment build
python -m research.sequence_experiment run
```

`run` writes chronological OOF probabilities, trains the final models exactly
once on the frozen pre-T0 sample IDs and leaves the prospective test pending.
Only at T1, after preparing and building the extended source, run `final`; a
second opening is refused. Generated material lives in `research/out/sequence/`.

### Making money — funding carry (no DB needed, live Binance data)

```bash
# rank live delta-neutral carry opportunities, risk-adjusted, net of fees
python -m research.funding_scanner --top 15

# what would a $10k delta-neutral carry have earned over the last 14 days?
python -m research.carry_book simulate DEXEUSDT --days 14 --notional 10000

# track it forward on real funding settlements (paper)
python -m research.carry_book open DEXEUSDT --notional 10000
python -m research.carry_book status      # re-run over the days to watch it accrue
```

### SQL-only analytics (Grafana-ready)

```bash
docker exec -i klinehouse-timescaledb-1 psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" < db/analytics.sql
docker exec klinehouse-timescaledb-1 psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "CALL refresh_continuous_aggregate('flow_1m', NULL, NULL);"
```

Then query `flow_metrics_1m` (order-flow pressure per minute) and
`realized_vol_1m` (volatility radar). Example panels are documented inline in
`db/analytics.sql`.

---

## Honesty caveats (read before trusting any number)

- **Data map.** After REST gap-healing (`scripts/backfill.py`) BTCUSDT is now one
  continuous ~6-day block; ETH/SOL have ~2.5 days each. That is enough to
  characterise microstructure and validate intraday timing, **not** enough for
  multi-week or regime-dependent conclusions. Let the collector run for weeks
  before drawing stronger ones. `report.py` prints the live data map first, on purpose.
- **Symbol coverage.** The signal is strongest and best-sampled on BTCUSDT;
  ETH/SOL reproduce it more weakly (less data, wider spreads). Re-run per symbol
  before generalising.
- **No live-trading frictions modelled** beyond fees: queue position, partial
  fills, and latency would make naive scalping *worse*, never better. The
  execution-alpha result is the conservative, robust one.
- **Backtest realism.** A 1-second execution lag is applied and trades are
  non-overlapping, but fills are valued at the candle close (no order-book
  simulation). Treat absolute PnL as indicative; treat the *sign and magnitude
  relative to fees* as the real conclusion.
