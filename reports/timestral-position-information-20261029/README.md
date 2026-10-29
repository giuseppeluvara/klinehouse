# Open interest and position information: timing audit and conditional replay

**No stable improvement was found.** The new two-feature model retains all nine original selective gates, but passes none of the paired full-score or direction-improvement gates against either control. Its timestamp assumptions are also not historically certified. The empirical failure and the unresolved availability are recorded separately; neither can be presented as a verified predictive solution.

## Timestamp verification came first

Before fitting or opening new candidate scores, we registered source acquisition, downloaded the admitted data, reconstructed their timing diagnostics, obtained independent open-interest observations, and completed independent source audits. Only then were the model, comparisons, code and input hashes frozen in [protocol.json](protocol.json). All 550,362 original forecast anchors remain in evaluation: BTCUSDT, ETHUSDT and SOLUSDT, January–July 2024, 2025 and 2026. These are seven-month periods, not complete years. No market data or outcome from 2026-08-01 onward was opened.

We downloaded 2,730 daily Binance USD-M metrics ZIPs, covering October of the preceding year through July 30 of each evaluation year. All official SHA-256 checksums passed. July 31 files were deliberately excluded because a label at 23:55 could represent an August 1 snapshot under a five-minute shift. This exclusion never removes evaluation anchors: both fitted corrections fall back to the unchanged baseline when context is absent.

The source arrays preserve raw labels, file dates, row numbers and six numeric columns. There are 785,842 rows, 398 missing expected grid rows, 518 rows with invalid required OI/position fields, and no duplicate or off-grid labels in this acquired snapshot. The implementation nevertheless rejects every copy of a duplicate, does not round off-grid timestamps, and never fills a missing observation. Missing unused columns do not invalidate otherwise valid required fields.

### Flow timing is evidence about flow only

We compared the archive taker ratio with the buy/sell ratio independently reconstructed from the previously frozen one-minute futures trade candles, separately for `[T−5m,T)` and `[T,T+5m)`. The source-only rules require at least 200 finite pairs, correlation at least 0.98, median relative error at most 0.05, and a fivefold error advantage over the alternative. These thresholds were fixed after the small source pilot and before the bulk audit or model scores.

Across the 2,730 symbol-days, 456 match the preceding window, 2,218 the following window, and 56 remain unclassified. In the acquired 2024 cohort, matching preceding-window days extend through March 3, while following-window matches begin March 4 for all three assets. The later cohorts have only following-window matches or unclassified days. This describes the current acquired objects; it does not establish the historical archive revision process. We did not reproduce the universal June 25, 2026 transition reported in [Binance issue #491](https://github.com/binance/binance-public-data/issues/491), and we did not apply that claimed date as a correction.

The [current Binance API specification](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data) describes taker timestamps as period start and OI/top-position timestamps as period end. A flow alignment therefore cannot certify the other columns. Definitions and point-in-time availability are also still discussed in [issue #513](https://github.com/binance/binance-public-data/issues/513).

### Independent OI samples

The [Tardis raw openInterest channel](https://docs.tardis.dev/historical-data-details/binance-futures) supplies native observation timestamps and local receipt timestamps. Ninety-six bounded seven-minute captures on four pre-holdout dates yielded 22,026 snapshots. We compared 24 fixed hourly archive labels per date and asset (288 labels total) with the nearest native observation within 15 seconds, under two predeclared hypotheses: T and T+300 seconds. No prediction outcomes were involved.

| Sample date | Favored OI alignment | Paired quantity comparisons favoring it |
|---|---|---:|
| 2024-01-01 | T | 55/58 |
| 2025-01-01 | T+5 minutes | 70/72 |
| 2026-06-01 | T+5 minutes | 53/72 |
| 2026-07-01 | T+5 minutes | 48/72 |

Log-change discrepancies between consecutive sampled observations favor the same alignment for all three assets on every sampled date. This supports a shift already present by January 2025; it does not identify the OI transition date. Only 2/288 quantities were seen exactly, and ETH June/July 2026 has persistent quantity discrepancies of approximately 0.092%/0.075%. These are sample alignment findings, not exact all-history identities.

Observed native-to-receipt delays were 0.407–10.852 seconds for these independently captured snapshots. They do not bound publication time of the Binance metrics files. No independent historical top-position capture was found. Position measurement time, archive publication time, historical field mapping and revision vintages therefore remain unresolved. Raw third-party captures remain in the private research cache; [oi-timestamp-audit.json](oi-timestamp-audit.json) contains derived comparisons, source hashes and explicit limits. An offline verifier reproduces the full private comparison and the committed summary.

## Frozen simple model

This is an explicitly **conditional** replay with a fixed assumed 15-minute lag. It does not convert that assumption into a timing certificate. Each forecast requires exactly 13 unique valid five-minute labels spanning one hour and ending at `anchor−900 seconds`. A plausible content timestamp `label+300` reaching the protected boundary is rejected. No per-period retiming, latency search or post-result parameter changes were performed.

The two new raw features are:

1. Hourly change in log OI quantity, multiplied by the clipped current normalized own-asset hourly spot return.
2. Hourly change in the archive top-position-ratio imbalance `(R−1)/(R+1)`, where R is `sum_toptrader_long_short_ratio`.

OI quantity is used rather than notional value to avoid a mechanical price term. The position field is treated as an archive proxy where historical semantics remain unresolved. Both features are clipped at ±8. The matched price-only control uses normalized spot returns over five and sixty minutes with the same two-coefficient budget.

Both arms reuse the preceding study's fixed ridge-offset logistic algorithm: penalty 0.25, no intercept or asset identifiers, monthly 90-day matured non-flat training, pooled assets with equal UTC-day weights, train-only RMS normalization and ±3 clipping. The monthly uniform-center baseline coefficient and mixed flat probability remain unchanged. All 42 fits use matching eligible training rows; no fit is silently replaced by baseline. The baseline training offset is evaluated with that month's past-trained coefficient, not an out-of-fold stack.

Incomplete context in any asset triggers the same baseline fallback for both arms on all three assets. There are 7,971 fallback forecasts among the unchanged 550,362 anchors: 1,963 per asset in 2024, 410 per asset in 2025, and 284 per asset in 2026. The matched price control is re-fitted on this study's shared membership, so it is not identical to the preceding lead-lag study's control.

## Results

Full Brier is the sum of three squared class-probability errors; full log loss uses natural logarithms. Conditional direction metrics use observed non-flat outcomes only: binary single-square Brier, binary log loss and accuracy, with ties assigned up. This conditional evaluation does not supply a future-flat oracle.

Each paired gate requires positive lower confidence bounds for every metric and positive gains in at least five of seven months for every metric. We retained 12,000 paired seven-day moving-block replicates, seed 271032, quantiles 1/1080 and 1079/1080. The correction is conservative for the 18 primary cohort/control comparisons per endpoint/scope, but not joint across all endpoints and sequential research history. There is no bootstrap refitting.

| Comparison | Positive full Brier point gain | Positive full log-loss point gain | Positive direction accuracy point gain | Full paired gates | Direction paired gates |
|---|---:|---:|---:|---:|---:|
| positions vs baseline | 2/9 | 2/9 | 3/9 | 0/9 | 0/9 |
| positions vs price control | 2/9 | 2/9 | 6/9 | 0/9 | 0/9 |
| price control vs baseline | 6/9 | 6/9 | 5/9 | 0/9 | 0/9 |

All three models retain 9/9 original selective gates. The new-information arm passes **0/18 full-score and 0/18 direction gates** against its two controls. No selected-scope diagnostic gate passes. `empirical_stable_improvement=false`, `timing_certified=false`, and `stable_improvement=false`.

| Cohort | Full Brier gain vs baseline | Full log-loss gain vs baseline | Direction accuracy gain (percentage points) |
|---|---:|---:|---:|
| 2024-BTCUSDT | -0.00000086 | -0.00000084 | +0.00000 |
| 2024-ETHUSDT | -0.00000533 | -0.00000536 | -0.00817 |
| 2024-SOLUSDT | -0.00000114 | -0.00000116 | -0.02811 |
| 2025-BTCUSDT | -0.00002206 | -0.00002211 | -0.07554 |
| 2025-ETHUSDT | +0.00000095 | +0.00000094 | -0.06234 |
| 2025-SOLUSDT | +0.00001240 | +0.00001250 | -0.02160 |
| 2026-BTCUSDT | -0.00000100 | -0.00000101 | +0.04596 |
| 2026-ETHUSDT | -0.00000394 | -0.00000395 | +0.08046 |
| 2026-SOLUSDT | -0.00000181 | -0.00000200 | +0.07325 |

Positive score gains mean control minus candidate; positive accuracy gains mean candidate minus control. Score improvements occur only in ETH/SOL 2025, while accuracy improvements occur only in the three 2026 cohorts. There is **no cohort with positive point gains in all three headline metrics versus baseline**. BTC 2024 accuracy is tied.

BTC 2025 significantly regresses against baseline on both full scores: Brier gain -0.00002206 (interval -0.00004095 to -0.00000278) and log-loss gain -0.00002211 (interval -0.00004105 to -0.00000279). Both scores worsen in every one of its seven months. Its conditional probability scores also significantly regress. No directional-accuracy confidence interval establishes an improvement in any cohort. None of these failures is hidden by a timing-based success flag.

## What was verified

- All 2,730 official checksums, source structures, retained rows and 5,460 daily flow comparisons were independently reconstructed, including a local re-decode of every original ZIP.
- The separate offline OI audit reproduced 96 captures, 22,026 snapshots, 12 source ZIPs and 288 labels, without a network call or outcome access during verification.
- The independent model audit reconstructed conditional features, all 42 fit gradients without re-fitting, 63 past-frequency windows, every saved prediction, 27 original evaluations and 81 paired comparisons. Maximum gradient residual was 3.94e-9.
- Six fits report `optimizer_success=false`; all 42 satisfy the frozen numerical acceptance rule (gradient maximum at most 1e-7) and its independent reconstruction. The raw optimizer flags remain saved rather than being relabeled as successes.
- Local storage/Timestral suite: 334 passed, 8 database-integration skips. Nine new tests cover exact timing, missing/duplicate/off-grid rows, future-data invariance, unused-column missingness, source allowlists and protected-row rejection before numeric parsing. CI runs the database integration and prepared-source/model audits.
- Measured model runtime: 16.92 seconds, maximum RSS 981,975,040 bytes; independent full audit: 19.55 seconds, maximum RSS 1,030,864,896 bytes. Metrics ZIPs: 31,210,298 bytes; OI probe downloads: 403,047 bytes. Existing environment reused without installations.

```bash
# Portable verification of committed prepared sources, features, optima and scores:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-timestral/bin/python scripts/verify_position_information.py
# Local independent raw-source verification using the preserved private cache:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-timestral/bin/python scripts/verify_position_information.py --sources-only --raw
# Local offline independent OI-sample verification:
.venv-timestral/bin/python scripts/audit_open_interest_sample.py --verify
```

The acquisition pipeline is `scripts/prepare_position_information.py register`, `download`, `audit`; the model pipeline is `scripts/run_position_information.py register`, `run`. Reproduction must use a fresh report directory via `--root`, preserve/copy the source-only OI evidence there, and retain all failed comparisons. Existing outputs are never overwritten. CI uses committed prepared arrays and hashed OI evidence; it does not claim to re-download or independently observe the private OI snapshots.

The source audit prevents an unjustified common timestamp correction. It does not resolve the historical availability of long/short positions. This negative conditional experiment is not a deployable forecasting system, prospective validation or profitability evidence; protected holdouts and every preceding frozen study remain unchanged.
