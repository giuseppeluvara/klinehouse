# Diagnose before changing the forecasting model

The [executed root-cause report](../reports/timestral-root-cause-20261026/README.md)
separates a scoring error, numeric/integration checks and a representation hypothesis.
Earlier frozen experiments and reports remain unchanged. These diagnostic runs
use previously inspected history and cannot supply independent validation.

## Scoring correction

The earlier static baseline had a zero point forecast but its quantile triplet
contained the historical empirical median. The dynamic baseline had a zero
median. Both are legitimate distributions to score, but their **three-quantile
average does not isolate interval quality**. Describing that average as an
interval-only improvement was incorrect.

`research.timestral_diagnostics.interval_scores` reports coverage, width, the
90% interval score, tail-only pinball and median loss separately. For alpha=0.1:

```
IS = upper - lower + 20*max(lower-y, 0) + 20*max(y-upper, 0)
tail_pinball = (loss_0.05 + loss_0.95)/2 = IS/40
```

`compare_intervals` also computes a version with identical point/median values.
It never mutates the saved forecasts. This correction changes attribution and
reported improvement percentages; it does not create a directional signal.

## Numerical and positive/negative controls

The first registered diagnostic compares the existing log-price path with an
affine equivalent on all 1,484 January–July anchors: center on the last log close
in float64, multiply by 10,000, then cast to float32. The same 256 closes and
forecast endpoint are used. No calibration, selection or new data are involved.

Constant, increasing/decreasing linear and four periodic signals supply known
positive controls. Independent seeded Gaussian random walks supply a negative
control with known population conditional median zero. A finite sample can still
favor the model by chance; the negative control is not a performance guarantee.

## Overlap hypothesis and paired ablation

Adjacent rolling h-hour returns share h-1 increments. Even independent hourly
increments produce lag-one correlation (h-1)/h. That is not predictability of
the next full h hours. The first diagnostic measures both lag one and lag h,
and relates the actual saved forecasts to past and future returns.

The follow-up ablation is registered **after** those diagnostic results and
**before** its own inference. It is one hypothesis, not a hyperparameter sweep:

- Both variants see the same 64*h+1 consecutive past hourly closes.
- Dense returns sample the h-hour difference every hour (63*h+1 inputs) and
  forecast h steps. The last step is the target.
- Disjoint returns sample that same series every h hours (64 inputs) and forecast
  one step. This is the same future target, log(P[t+h-1]/P[t-1]).
- Both use the pinned Timestral-5M, basis-point inputs and identical UTC anchors.

Changing cadence also changes token count and discards intermediate observations;
the experiment does not isolate overlap from every possible representation effect.
Both variants use 64 horizon periods of history, versus the old 256 hours. At
24h that is 64 days. Comparing only the new disjoint result with the old report
would confound history length. The paired dense result is therefore mandatory.

All monthly results and paired weekly bootstrap intervals are retained. There is
no automatic deployment or winner selection. The bootstrap adjusts four
comparisons within this ablation, not the full sequence of research decisions.

## Reproduce

Use the existing pinned environment and cached weights. New directories are
mandatory; protocol, input and code hashes are checked before inference.

```bash
.venv-timestral/bin/python scripts/diagnose_timestral.py register \
  --output research/out/root-cause-reproduction
.venv-timestral/bin/python scripts/diagnose_timestral.py run \
  --output research/out/root-cause-reproduction
.venv-timestral/bin/python scripts/diagnose_timestral_overlap.py register \
  --output research/out/overlap-reproduction
.venv-timestral/bin/python scripts/diagnose_timestral_overlap.py run \
  --output research/out/overlap-reproduction
.venv-timestral/bin/python -m pytest -q \
  tests/test_timestral_diagnostics.py tests/test_timestral_cadence.py
```

The scripts use the already published hourly cache and forecast records. They
do not download data or open the protected August 2026–February 2027 period.
