# Traded perpetual lead-lag information: registered negative result

The new futures-price signal does **not** establish stable improvement in direction, Brier or log loss across the nine explored asset/year cohorts. All nine original selective gates still pass, as they already did for the frozen baseline. Passing those old gates does not demonstrate added information.

## Design and scope

We tested whether actual traded USD-M perpetual prices lead Binance spot prices. The earlier futures-flow study excluded futures prices; the premium-index study used a different source. We reused the 90 checksummed monthly trade-kline archives already present, extracting 3,944,160 close prices into nine small arrays. There were no downloads or installations.

The evaluated periods are January through July of 2024, 2025 and 2026, for BTCUSDT, ETHUSDT and SOLUSDT: 550,362 unchanged forecasts. These are three seven-month periods, not three complete years. All market sources and outcomes remain before 2026-08-01. None of the protected holdouts was opened. These cohorts have already informed many earlier studies; this is sequential retrospective research, not independent confirmation.

The new signal contains the futures-minus-spot log-return difference over 5 and 60 minutes, divided by a causal spot-volatility scale. The latest new-input candle opens two minutes before the forecast anchor and closes more than a minute before it. The volatility denominator excludes the latest return. Missing context would trigger an exact baseline fallback; there were zero missing evaluation contexts.

Each month uses only matured non-flat outcomes from the preceding 90 days, pooled across the three assets with equal UTC-day weights. Each arm adds exactly two ridge-penalized offset-logistic coefficients, with fixed penalty 0.25, train-only RMS normalization, clipping, no intercept, no asset identifier and no parameter search. The baseline uniform-center coefficient remains frozen; the flat probability remains the same fixed mixture used in the preceding flat-repair study. The training baseline offset is computed using that month's fitted coefficient on its past training rows; it is not an out-of-fold forecast sequence.

The primary **lead_lag** arm is compared both to the unchanged baseline and to **price_control**, a correction with the same two-coefficient budget using only spot returns over the same horizons. This separates adding fitting flexibility from introducing futures-price inputs. The model, runner, independent auditor, tests and input hashes were frozen in [protocol.json](protocol.json) before [opened.json](opened.json). No settings were changed after scores.

## Registered comparisons

Full scores use all three classes, with Brier defined as the sum of squared probability errors. Conditional direction scores use only observed moving outcomes: single-square binary Brier, binary log loss, and accuracy with ties assigned up. Conditioning does not make future unchanged-price outcomes observable.

Each paired gate requires positive lower confidence bounds for every metric and positive gains in at least five of seven months for every metric. We used 12,000 paired seven-day moving-block replicates, seed 271032, retaining the conservative quantiles 1/1080 and 1079/1080. These intervals do not jointly correct for every endpoint or the long preceding research history. No bootstrap refitting was performed. The fixed baseline 52% selection is a diagnostic scope only.

| Comparison | Full Brier improves at point estimate | Full log loss improves at point estimate | Direction accuracy improves at point estimate | Full paired gates | Direction paired gates |
|---|---:|---:|---:|---:|---:|
| lead lag vs baseline | 2/9 | 2/9 | 2/9 | 0/9 | 0/9 |
| lead lag vs price control | 1/9 | 1/9 | 3/9 | 0/9 | 0/9 |
| price control vs baseline | 7/9 | 7/9 | 5/9 | 0/9 | 0/9 |

All three forecasters pass 9/9 original gates. The primary arm passes 0/18 full paired gates and 0/18 direction paired gates against its two controls; `stable_improvement` remains **false**. Neither primary selected-scope comparison passes any gate. The spot-only correction passes one selected diagnostic, ETH 2025; it is not promoted as a general solution. On that same fixed subset of 13,279 forecasts, lead-lag significantly regresses versus price control on both scores: Brier gain -0.00017997 (interval -0.00037145 to -0.00001357), log-loss gain -0.00018119 (interval -0.00037427 to -0.00001381).

## Every cohort, primary arm versus frozen baseline

Positive gains mean improvement; Brier/log-loss gain is control minus candidate. Accuracy gain is candidate minus control in percentage points. Differences are small and these point estimates alone do not establish a benefit.

| Cohort | Full Brier gain | Full log-loss gain | Direction accuracy gain (pp) |
|---|---:|---:|---:|
| 2024-BTCUSDT | -0.00001369 | -0.00001378 | -0.05396 |
| 2024-ETHUSDT | -0.00000503 | -0.00000505 | -0.10131 |
| 2024-SOLUSDT | -0.00000212 | -0.00000210 | -0.05954 |
| 2025-BTCUSDT | -0.00000963 | -0.00000960 | -0.03284 |
| 2025-ETHUSDT | -0.00000153 | -0.00000157 | +0.00656 |
| 2025-SOLUSDT | +0.00000209 | +0.00000212 | -0.09303 |
| 2026-BTCUSDT | -0.00000807 | -0.00000810 | -0.00492 |
| 2026-ETHUSDT | +0.00001985 | +0.00001992 | +0.08539 |
| 2026-SOLUSDT | -0.00002279 | -0.00002379 | -0.01363 |

ETH 2026 is the only cohort with positive point gains in all three headline metrics versus baseline; its confidence bounds still do not establish improvement, and its full Brier/log-loss point estimates are worse than the matched spot-only control. No primary all-anchor score confidence interval excludes zero in either direction. The tiny local gains therefore do not identify a stable source of directional information.

## What this resolves, and what remains open

The result rejects this particular delayed 5/60-minute traded-price correction as a stable improvement under the frozen design. It does not prove that every use of futures prices is uninformative. The fitted short-horizon coefficient changes sign across monthly windows, including within years. That is a descriptive sign of instability, not proof of its economic cause or a license to flip the signal using later outcomes. We did not tune the lag, horizons or penalty after seeing it.

Open interest and long/short position ratios were considered before this experiment but were not downloaded or scored. The [Binance archive discussion](https://github.com/binance/binance-public-data/issues/491) reports changes in five-minute timestamp conventions during the allowed 2026 period. We selected existing trade candles on source-integrity grounds before scoring, rather than assuming immediately available position snapshots. Any future position-information study still needs an explicit timestamp/revision audit before a prediction claim.

The [official Binance public-data repository](https://github.com/binance/binance-public-data) describes the archives and checksums; the [official USD-M market-data specification](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data) defines trade-kline open/close times and prices. The declared one-minute delay makes this a causal replay conditional on source availability, but historical release vintages and archive revisions are not independently certified. There is no profitability, execution-cost, prospective-validation or universal-superiority claim.

## Verification and reproduction

- Independent source-only audit re-decoded all 90 original ZIPs and compared 3,944,160 prices and timestamps; [source-verification.json](source-verification.json).
- Independent replay audit reconstructed features, 42 fitted optima without refitting, 63 historical-frequency windows, all predictions, 27 original evaluations and 81 paired comparisons; [verification.json](verification.json). Maximum fit-gradient residual: 1.49e-9.
- Six new synthetic tests cover timing, gaps, future-data invariance, day weighting, independent gradients and protected-date rejection. Local storage/Timestral suite: 325 passed, 8 database-integration skips; CI runs the integration checks separately.
- Replay: 16.77 seconds, maximum RSS 970,457,088 bytes. Independent replay audit: 15.26 seconds, maximum RSS 1,022,836,736 bytes. Existing Python environment, no swaps in measured runs; [runtime.json](runtime.json).

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-timestral/bin/python scripts/verify_lead_lag.py
# Local optional audit of the already cached original ZIPs:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-timestral/bin/python scripts/verify_lead_lag.py --sources-only --raw
```

To reproduce fitting, use a new report directory under the repository and run `scripts/run_lead_lag.py prepare`, `register`, and `run` with the same `--root`; these commands use exclusive writes and never replace the frozen bundle. The raw ZIPs remain in the preexisting private cache, while prepared prices and their provenance are committed for CI. Results, fits, predictions and original failures are all retained.
