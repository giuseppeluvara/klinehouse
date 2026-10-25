# Frozen adaptation and a separate historical test

This follow-up tries to improve BTCUSDT forecasts at four hours and one day.
It compares the selected adaptation with an unchanged-price forecast on samples
that are not used for fitting, normalization, selection or residual calibration.
The earlier one-hour pilot, May–July long-horizon experiment and protected
prospective bundles remain unchanged.

## Chronology and its limits

| Role | UTC interval, exclusive end |
| --- | --- |
| Raw development source and context | September 1, 2025 – January 1, 2026 |
| Initial fit | October 1 – December 1, 2025 |
| Candidate selection | December 1, 2025 – January 1, 2026 |
| Selected candidate refit | October 1, 2025 – January 1, 2026 |
| Separate historical test | January 1 – April 1, 2026 |

Fit labels must mature by the validation boundary; all development labels must
mature by the test boundary. The January 1 context reuses only the final 256
development hours, beginning December 21 at 08:00 UTC. The test has 540 four-hour
and 90 daily anchors if complete. Targets do not overlap within a horizon; the
two horizons share market observations.

The earlier experiments reserve August 8, 2026 through February 4, 2027. This
experiment does not access that period. Its design is motivated by the already
observed May–July 2026 scores, so this earlier holdout is **historical evidence
on separate samples**, not proof of a choice that would actually have been made
in January 2026. Model pretraining contamination cannot be excluded. No result
is a prospective result or a profit calculation.

## Fixed candidates and information

The original pinned Timestral-5M is reused, without fine-tuning, on CPU with two
threads. Complete hourly inputs retain the three established covariates from
exactly 3,600 closed one-second OHLCV candles. A missing or invalid hour resets
context and invalidates affected targets. There is no interpolation.

The nine candidates, in deterministic tie order, are:

1. Original Timestral price prediction.
2. Original Timestral prediction with past covariates.
3. Price prediction multiplied by 0.25, plus half the median training residual.
4. The same correction with multiplier 0.5.
5. The same correction with multiplier 1.
6. Ridge combination of the two original predictions, penalty 1.
7. The same combination with penalty 10.
8. Ridge combination plus six past-state features, penalty 1.
9. The same state model with penalty 10.

The six state features are trailing 4h, 24h and 168h log returns; the square root
of summed intrahour realized variances over the last 24 hours; the simple mean
of 24 hourly taker-imbalance ratios; and mean hourly log-trade intensity over
24 hours minus its 168-hour mean. Intrahour realized variance excludes the
cross-hour second return, as in the earlier fixed feature definition.

Ridge minimizes mean squared error plus the stated coefficient penalty, with
an unpenalized intercept. Feature means and standard deviations use only the
fit data; standardized inputs are clipped to [-8, 8]. Every candidate receives
the same complete samples. Each horizon independently selects the lowest
December MAE, using the listed order for exact ties. Even if no candidate beats
unchanged price in validation, the best candidate is evaluated honestly on the
test; its validation gate is reported, not used to suppress the comparison.

Only the chosen rule is refitted to October–December. Parameters, scaler,
feature order, candidate list, code, sample IDs and data hashes are frozen before
the dedicated test dataset directory exists. Test outcomes never update these
parameters. A first-opening receipt prevents silently replacing a completed or
partial test run; an infrastructure failure must be documented before any new
technical execution, without changing the chosen model.

Unchanged-price interval quantiles are estimated from October–December returns.
Adapted interval quantiles add the selected candidate's December out-of-sample
residual quantiles to its refitted point forecast. December also selects the
candidate, and refitting changes the point model, so these intervals are not
guaranteed calibrated. Their median need not equal the separate point forecast.
Original Timestral uses its native quantiles. All interval distributions remain
frozen throughout the test.

## Predeclared comparison

The primary loss is absolute log-return error, paired on identical anchors.
Report MAE, RMSE, sign accuracy on non-neutral forecasts, empirical 90% interval
coverage, pinball loss, monthly scores and all individual price forecasts.

The fixed uncertainty calculation resamples 13 calendar blocks of seven days
starting January 1, including the final six-day block. It uses 10,000 draws,
seed 17, and weights all sampled anchors equally. For the two primary
adapted-minus-unchanged comparisons, report two-sided 97.5% percentile intervals
(Bonferroni for the two horizons). Other comparisons use descriptive 95%
intervals. Few blocks and dependence across weeks limit this approximation.

A horizon passes the historical evidence gate only if all these hold:

- At least 480 four-hour or 80 daily complete test samples.
- Strictly lower MAE than unchanged price.
- The upper endpoint of its adjusted interval is below zero.
- Lower MAE in at least two of the three test months.
- RMSE no worse than unchanged price.

The result can be negative. No candidate, feature, penalty, threshold or period
is changed after opening the test. Passing this historical gate would still not
establish prospective accuracy or trading profitability.

## Run in order

Reuse `.venv-timestral` and the existing checkpoint. Apply the Mac storage guard
before acquisition. The two raw datasets contain 212 days in total; official HEAD
metadata budget approximately 510 MiB of ZIP downloads, about 1.2 GiB for retained
ZIPs plus Parquet based on earlier measurements, and a conservative 4.6 GiB bound.
No CSV is extracted. Raw data, model cache and working output stay outside Git.

```bash
.venv-timestral/bin/python scripts/prepare_timestral_data.py \
  --start 2025-09-01 --end 2026-01-01 \
  --output research/out/timestral-adaptation-20261025/development

.venv-timestral/bin/python scripts/run_timestral_adaptation.py development \
  --config configs/timestral_adaptation.json \
  --output research/out/timestral-adaptation-20261025/development-run

.venv-timestral/bin/python scripts/run_timestral_adaptation.py freeze \
  --development-run research/out/timestral-adaptation-20261025/development-run \
  --output research/out/timestral-adaptation-20261025/frozen

# Only after the immutable selection and protocol have been written:
.venv-timestral/bin/python scripts/prepare_timestral_data.py \
  --start 2026-01-01 --end 2026-04-01 \
  --output research/out/timestral-adaptation-20261025/test

.venv-timestral/bin/python scripts/run_timestral_adaptation.py holdout \
  --frozen research/out/timestral-adaptation-20261025/frozen \
  --output research/out/timestral-adaptation-20261025/test-run

.venv-timestral/bin/python scripts/analyze_timestral_adaptation.py \
  --run research/out/timestral-adaptation-20261025/test-run \
  --output research/out/timestral-adaptation-20261025/analysis
```

For an independent reproduction, change the dedicated output roots in a copied
configuration before starting. Existing artifacts are never overwritten. The
development hourly cache is small, checksummed and derived once from verified
seconds; it provides the exact final context without duplicating December's raw
files. Runtime and peak RSS refer to the entire process, not just model weights.
