# Separate prospective depth-information research: capture active, efficacy untested

The preceding direction diagnosis found a short live regime in which the frozen
hourly-center reversal signal ranked directions poorly. A causal orientation
replay did not produce stable improvement over the nine historical cohorts; see
[the complete diagnosis](../direction-orientation-20261030-v3/README.md).

This next hypothesis adds a genuinely different input: visible resting bid/ask
depth. The [procedure](protocol-plan.json) was registered before the new capture
outcomes. It is separate from the running seven-day shadow pilot. **No new fitted
model or candidate forecast has yet been produced.**

## Acquisition actually executed

A bounded, anonymous GET-only collector is running locally from an immutable
runtime copy in `research/out/depth-research-20261030/`. It reads neither the
original pilot nor historical archives, models, environment files or credentials.
It collects three depth20 snapshots every five minutes, beginning ten seconds
before the anchor, plus a server-time clock probe. Requests and full-body client
receipts must precede the anchor; post-COMMIT observations establish persistence.
The source, quantity-derived features and sequence/hash evidence remain append-only.

The first trio at **2026-10-30 15:35 UTC (16:35 Europe/Rome)** passed independent
verification. All body receipts, primary commits, acknowledgement commits and the
separately synced proof precede the anchor. The minimum acknowledgement lead was
**9.098548 seconds**; the proof-write lead was **9.097863 seconds**. The independently
recalculated UTC clock-offset bound was **0.396738 seconds**.

| Asset | Best-level QI1 | Twenty-level QI20 | Spread, basis points |
| --- | ---: | ---: | ---: |
| BTCUSDT | -0.987111 | -0.985032 | 0.001291 |
| ETHUSDT | -0.948073 | -0.305621 | 0.044847 |
| SOLUSDT | -0.305614 | +0.281242 | 1.058481 |

These are captured feature values, not forecasts or accuracy results. Independent
80-digit Decimal arithmetic matched with maximum discrepancy **4.393e-51**.
The audit checked the raw JSON/body hashes, 20 valid levels per side, the config
seed, all twelve initial events and the proof's retained chain tip. Evidence:
[first capture proof](first-cycle.json), [write receipt](first-cycle-write-receipt.json),
[independent audit](first-cycle-independent-audit.json).

QI20 is `(sum_bid_quantity20-sum_ask_quantity20)/(sum_bid_quantity20+sum_ask_quantity20)`.
All features come from the **same** depth response. Asynchronous depth and
bookTicker responses are not combined. QI is static visible queue imbalance,
not trade flow, event-level order-flow imbalance or open interest. The REST
response has no event timestamp: client receipt proves availability, not the
freshness of the exchange's internal book. `lastUpdateId` is a sequence identifier.

The fixed resource budget is 32 MiB with a 4 MiB reserve and a stop below 40 GiB
free; nothing is deleted. The end is **2026-11-05 11:10 UTC**. The original pilot's
service, runtime, model, protocol and journal remain unchanged. No order endpoint
or account access exists. The capture is a local process, not an installed
autostart service; its status/process must be checked rather than assumed alive.
The HTTP connect/read timeouts do not constitute an absolute total-response
deadline, but a late or clock-unhealthy observation remains explicitly ineligible.

## Registered future model and unfinished work

The first scheduled anchor fixes the training cutoff at **2026-10-31 15:35 UTC
(16:35 Europe/Rome)**. Only the newly captured first 24 hours, eligible original
baseline probabilities and labels actually matured **and received** by that cutoff
may fit one shared coefficient. The model is

`q_new = sigmoid(logit(q_original_baseline) + beta*QI20)`.

The penalty is fixed at `0.25*beta²/2`, beta is bounded to [-8,8], and the issued
flat probability stays unchanged. Flat labels do not train direction; moving
observations have equal total weight per anchor across assets. At least 300 moving
labels pooled and 100 per asset are required; otherwise beta is frozen to zero
without extending the window. The coefficient must be durably frozen once before
any future candidate emission. There is no post-evaluation refitting or threshold
search.

The pure model helper and its chronology/optimality tests are implemented. The
durable model-freeze, separate prospective forecast and evaluation runner is
**still being implemented**. Acquisition success is not model-validation success.
The original baseline, matched price control and OI/position control must be
evaluated on identical future candidate emissions; missing/fallback/late cases
stay visible in coverage. Any failed asset remains a failed criterion. Neither
the warmup fit nor the already inspected orientation replay is confirmation.

The complete local diagnostic/capture suite passed **174 tests**, including
30 capture tests and nine one-coefficient depth-model tests. An independent
review checked post-fsync UTC/monotonic clock guards, orphan restart behavior,
missing records, API allowlists and resource limits before activation. Existing
environments were reused; no historical book dataset or AI model was downloaded.

## Reproduce safely into a new directory

```sh
.venv-timestral/bin/python scripts/capture_depth_research.py init \
  --root research/out/depth-fresh
.venv-timestral/bin/python scripts/capture_depth_research.py run \
  --root research/out/depth-fresh
```

Each init refuses an existing directory. Do not replace the active run to repeat
a test. Future data cannot backfill a past book. The active collector's local
session/process details are retained separately in its runtime receipts.

## Sources and limits

[Binance Spot market API](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market)
documents public depth and bookTicker. The depth20 request weight is five;
the [server-time endpoint](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/general)
adds one per cohort. That is sixteen weight units per five-minute trio.

[Cont et al.](https://arxiv.org/abs/1011.6402) study short-interval contemporaneous
order-flow price impact in equities; [Gould and Bonart](https://arxiv.org/abs/1512.03492)
study next-tick queue-imbalance prediction for Nasdaq stocks. These motivate
testing a different information source but do **not** establish predictive power
after a one-minute delay over this five-minute crypto target. This short new
experiment cannot establish statistical superiority, profitability or universal
cross-year performance.
