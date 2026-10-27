# Futures volume and aggressor flow: no stable incremental direction signal

The registered comparison completed on **January–July 2024, 2025 and 2026** for
BTCUSDT, ETHUSDT and SOLUSDT. Neither futures volume intensity nor aggressor flow
establishes additional directional information in any of the nine asset/year
cohorts. This experiment does not provide a general solution to up/down prediction.

| Registered check | Passed cohorts |
| --- | ---: |
| Volume intensity improves spot | 0/9 |
| Flow improves spot | 0/9 |
| Flow improves the volume control | 0/9 |
| Original selective direction gate: spot | 6/9 |
| Original selective direction gate: spot + volume | 6/9 |
| Original selective direction gate: spot + volume + flow | 5/9 |

Both combined gates in [results.json](results.json) are false. The flow arm loses
the original SOL 2025 direction gate. All nine paired Brier **and** log-loss
confidence intervals include zero for **each** of the three comparisons. The
result is therefore not merely a failure of a minimum-accuracy threshold.

## What was tested

Three models share the same training samples, parameters, monthly fit schedule,
evaluation anchors and frozen flat-price probabilities:

1. **Spot:** the existing 23 spot features.
2. **Volume:** spot plus four interactions of futures volume/trade intensity with
   spot direction. Intensity is relative to the preceding 120 minutes.
3. **Flow:** volume plus eight features using the futures taker-buy imbalance over
   1/5/30/120 minutes, a short/long flow difference, an intensity interaction and
   other-asset flow. Base-volume sums weight the multi-minute imbalance.

The source is actual USD-M perpetual **trade klines**, distinct from the previous
premium-index study. The official schema supplies total base volume, trade count
and taker-buy base volume. Aggressive selling is total volume minus taker-buy
volume. Every trade has a buyer and seller; this measures the aggressor side,
not counts of traders, long/short positions, opening/closing trades or order-book
liquidity. See the [Binance public-data schema](https://github.com/binance/binance-public-data)
and [official futures market-data documentation](https://developers.binance.info/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data).

The target remains the sign of the **five-minute spot return**, from anchor+1min
to anchor+6min. Futures information ends at anchor−1min: the latest admitted
minute opens at anchor−2min. Activity references exclude that latest minute;
flow windows include it. All normalizations use the preceding training window.
Each monthly model trains on 90 days of matured, non-flat labels pooled across
three assets, with fixed sign-reflected logistic regression. There are **63 fits**.
The twelve additions are odd under simultaneous price/aggressor reflection.

The [protocol](protocol.json), code hashes and input hashes were recorded before
new source acquisition and scoring. No model or threshold search followed the
results. Volume must improve spot; flow must improve both controls. The original
all-anchor selective direction gate remains an additional requirement.

## Conditional directional results

Accuracy below is calculated on **all non-flat outcomes**, without confidence
selection. It cannot identify future flat prices or demonstrate tradable coverage.
Brier gain means control loss minus flow-model loss; positive is better. Gains
and intervals are multiplied by 10,000 for readability, **not return basis points**.

| Cohort | Flow accuracy | Brier gain vs spot ×10⁴ | 99.444% interval ×10⁴ | Gain vs volume ×10⁴ | Positive months vs spot |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024-BTCUSDT | 51.76% | -0.044 | [-0.895, +0.841] | +0.196 | 3/7 |
| 2024-ETHUSDT | 51.32% | -0.597 | [-1.885, +0.619] | -0.357 | 3/7 |
| 2024-SOLUSDT | 51.08% | -0.205 | [-1.283, +1.010] | -0.035 | 3/7 |
| 2025-BTCUSDT | 50.95% | +0.281 | [-0.764, +1.357] | +0.222 | 5/7 |
| 2025-ETHUSDT | 51.54% | +0.246 | [-0.648, +1.075] | -0.036 | 4/7 |
| 2025-SOLUSDT | 50.97% | -0.119 | [-0.973, +0.735] | +0.020 | 3/7 |
| 2026-BTCUSDT | 51.96% | +0.049 | [-1.352, +1.155] | +0.584 | 5/7 |
| 2026-ETHUSDT | 52.50% | +0.307 | [-1.003, +1.548] | +0.471 | 4/7 |
| 2026-SOLUSDT | 52.20% | +0.384 | [-1.050, +1.534] | +0.455 | 5/7 |

Small gains appear in 2026, while all three 2024 point estimates against spot are
negative. No paired gain is distinguishable from zero under the registered
seven-day block bootstrap. The evidence supports **no demonstrated incremental
benefit for these representations, delay and horizon**. It does not establish
that all possible futures information is useless, or that markets are inherently
unpredictable. Another source or horizon requires a separate protocol; selecting
variants against these same outcomes would remain exploratory.

## Coverage, protection and limitations

All **90 monthly ZIPs**, October of the previous year through July of each
evaluation year, passed official SHA256 checksums. They contain **3,944,160 minute
rows**, with **zero missing minutes**. The 121-minute context rule is tested on
synthetic gaps; on unavailable contexts both futures arms use the spot probability.
No evaluation context needed this fallback in the real study. All **550,362**
original asset/anchor evaluations were retained.

No market source or outcome from **2026-08-01 onward** was opened for this study.
Previously frozen experiments and user files remain unchanged. Years mean the
specified January–July cohorts, not full calendar years. The historical spot
outcomes have been explored before; this is not untouched or prospective
validation. Archive revisions and publication latency are unknown, so causal
replay is conditional on the declared lag rather than verified historical vintages.

Conditional uncertainty uses 4,000 paired seven-day moving-block replicates and
a nine-cohort correction per endpoint. It does not correct jointly over all
endpoints, comparisons or earlier experiments; correlated assets are not treated
as independent replications. Models are not refitted inside the bootstrap.
Flat-price probability stays identical across arms; original selective scores
include flats. No trading profitability, costs, deployment or live orders are claimed.

## Executed verification and resource use

The [independent audit](verification.json) parses all raw ZIPs separately,
reconstructs features with an explicit minute grid and convolutions, checks all
63 fit memberships and convex optimality gradients, replays saved coefficients,
and recomputes all 27 unconditional cohort metrics and 27 conditional comparisons.
It does not call the production feature builder, reader or fitter. Maximum fit
gradient: **8.865×10⁻⁸**. Hashes, chronology, labels, unchanged flat probabilities,
source bounds and every saved probability are checked.

Local relevant suite: **273 passed, 8 skipped** (database integration requires
TimescaleDB). CI runs those integration tests and the saved-evidence audit.
Runtime: acquisition/validation **127.39s**, fit/evaluation **23.57s**, independent
raw audit **36.21s**. Peak fit RSS **1.33GiB**; peak audit RSS **1.38GiB**.
See [runtime.json](runtime.json) for the reused environment.

Raw downloads occupy **155.66MiB**; the reproducible public bundle about
**54.62MiB**, approximately **210.3MiB** before Git objects. No new model, environment
or container was installed locally. Internal free space measured **40.53GiB**
before acquisition and **43.29GiB** afterward; changes outside this experiment
mean the free-space delta is not a measurement of its footprint. No cleanup ran.

## Reproduce

Audit committed arrays, coefficients and metrics without downloads or fitting:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/verify_futures_flow.py
```

Add `--raw` to independently validate the retained ZIPs in
`research/out/futures-flow-20261027/raw`. Fresh experiments must use a **new**
output directory; existing outputs use exclusive writes:

```sh
python scripts/run_futures_flow.py register --root research/out/futures-flow-replay
python scripts/run_futures_flow.py prepare --root research/out/futures-flow-replay
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python scripts/run_futures_flow.py run --root research/out/futures-flow-replay
python scripts/verify_futures_flow.py --root research/out/futures-flow-replay --raw
```

The manifest freezes the executed experiment artifacts. This report, runtime
receipt and independent verification receipt were added afterward.

## CI portability diagnosis

The first publication run passed **274 tests**, including TimescaleDB integration,
but stopped in an older frozen tree audit before reaching the new futures audit.
On BTC 2024-06-13 00:45 UTC, feature `log1p(trades)` for 656 trades equals a saved
tree split, 6.48768401848461. Raising that feature by exactly one ULP reproduces
all three discrepant probabilities from CI exactly, including the maximum
8.80241439e-6 difference. This is an arithmetic branch-boundary issue; it is
separate from the negative futures-information result.

The earlier passing and failing CI runs used the same NumPy 2.5.4 release.
Disabling AVX-512 did **not** resolve the failure; that initial runtime hypothesis
was rejected. NumPy documents [platform-sensitive transcendental differences](https://github.com/numpy/numpy/issues/23523), but disabling one instruction set is
insufficient to guarantee identical rounding across mathematical libraries.

The dedicated [portable auditor](../../scripts/verify_history_portable.py) now
reconstructs only the integer-count logarithm with Decimal at 48 and 80 digits,
requires both precisions to round to the same binary64 value, and then runs the
existing strict history/tree audits. The canonical value for 656 trades is
`0x1.9f3637078b16bp+2`. This is the same mathematical feature with deterministic
rounding, not a changed statistical model. All other features, labels, frozen
trees, saved predictions, score gates and comparison tolerances are unchanged.
The wrapper restores the original reconstruction function after the audit.

Seven focused regression cases check the split boundary, invalid counts and
invariance of every other feature. The full portable history/tree audit passed
locally, including all 21 frozen tree fits and all nine cohorts; see
[portable-history-verification.json](portable-history-verification.json).
The initial failed CI runs are retained as part of the publication history.
