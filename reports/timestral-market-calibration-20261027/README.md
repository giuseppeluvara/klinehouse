# Prequential calibration: lower underconfidence, no general improvement

The common-hour signal was held fixed and calibrated with one multiplicative
factor learned only from earlier out-of-training predictions with matured labels.
The uncertainty-shrunk correction improves fixed-bin calibration error versus
the overly conservative base in **8/9** cohorts at point estimate. It nevertheless
worsens Brier versus the already frozen unpenalized model in **8/9** cohorts.
It passes **5/9** original direction gates and **0/9** improvement gates against
the unpenalized control. The registered general-solution criterion fails.

All **550,362** evaluation anchors cover BTCUSDT, ETHUSDT and SOLUSDT in
**January–July 2024, 2025 and 2026**. The periods were previously explored;
registration before this run is not independent or prospective confirmation.

## Registered calibration

The [protocol](protocol.json) freezes all code, sources, controls and criteria.
The 21 January–July base fits are reused byte-for-byte from the
[common-hour experiment](../timestral-market-reversal-20261027/README.md).
Six additional base fits generate November/December warmup forecasts. They use
the identical trailing-90-day procedure on the available history starting in
October; these warmup windows contain about 31/61 days, explicitly shorter than
90. No warmup forecast uses its own or later outcomes.

For each evaluation month, the calibrator uses the preceding **56 days** of
non-flat outcomes and their previously issued base logits. Every base model's
training labels have matured by its month start. Calibration labels must mature
by the new month start, including the six-minute target delay. There are at least
42 contiguous calibration days; all actual fits have 56. The three assets are
pooled with equal total weight per UTC day. Calibration includes all finite past
forecasts, while base fitting retains its original futures-availability filter.

The raw scalar factor in [0,8] minimizes day-weighted log loss of
`sigmoid(factor × base_logit)`. There is no intercept or asset-specific adjustment.
The primary version shrinks that factor toward identity one when its estimated
uncertainty is large:

`weight = max(0, 1 − factor_variance / (raw_factor − 1)²)`

`adjusted_factor = 1 + weight × (raw_factor − 1)`.

Daily logistic scores estimate factor variance using a Bartlett covariance
estimate with six daily lags and the scalar Hessian. Fixed numerical floors and
the no-information identity fallback are specified in the protocol. This is a
fixed heuristic, **not a guarantee** of calibration or protection against
overfitting. No window, variance setting or factor bound was selected afterward.

Raw and uncertainty-shrunk corrections are both retained. New probabilities keep
the frozen flat-price probability in all original evaluations. The original
five-minute spot target, from anchor+1min to anchor+6min, and the 52% selection
threshold are unchanged. All 21 factors are positive, and **zero** up/down
decisions change in every cohort: this experiment changes confidence, not ranking.

## Results

Brier gain below is control loss minus adjusted-calibrator loss, multiplied by
10,000 for readability. These are **score units, not trading basis points**.

| Cohort | Gain vs conservative base ×10⁴ | Gain vs unpenalized ×10⁴ | Fixed-bin ECE: base → adjusted | Original gate |
| --- | ---: | ---: | ---: | --- |
| 2024 BTC | +1.686 | +0.031 | 1.282% → 0.529% | pass |
| 2024 ETH | +1.166 | -0.085 | 0.928% → 0.215% | pass |
| 2024 SOL | +0.444 | -0.208 | 0.639% → 0.323% | pass |
| 2025 BTC | -0.549 | -0.342 | 0.152% → 0.596% | fail |
| 2025 ETH | +1.686 | -0.259 | 1.064% → 0.555% | pass |
| 2025 SOL | -0.017 | -0.182 | 0.401% → 0.347% | fail |
| 2026 BTC | +0.415 | -0.521 | 0.732% → 0.376% | fail |
| 2026 ETH | +1.471 | -0.421 | 0.964% → 0.301% | pass |
| 2026 SOL | +0.544 | -0.452 | 0.850% → 0.359% | fail |

Conditional information gates pass **3/9** versus conservative base, **6/9**
versus 50/50, and **0/9** versus each of unpenalized, raw calibration and flow.
Every comparison also tests historical frequency. The raw-calibration arm passes
5/9 original gates as well. The unpenalized frozen control passed 6/9.
All 45 paired comparisons, original scores and fixed-bin counts are in
[results.json](results.json). Calibration success and general-solution flags
remain false.

Primary mean confidence increases from approximately 50.5–50.6% to 51.25–51.76%.
That corrects much of the aggregate underconfidence but overshoots for several
cohorts. Fixed-bin ECE is descriptive, bin-dependent and subject to sampling
noise; it is not a significance test or a complete calibration guarantee.
Proper scoring comparisons remain necessary even when ECE gets smaller.

The second factor operates on logits whose underlying coefficient already
changes monthly. That introduces another estimated scale and a 56-day estimation
window. Scale drift is a plausible concern, **not an isolated causal explanation**
for the observed losses. A separately registered
[direct uncertainty adjustment](../timestral-uncertain-market-20261027/README.md)
tests an alternative with only one final directional coefficient. This experiment
and its negative results remain unchanged.

## Verification and limits

The [independent auditor](verification.json) reconstructs features, all 27 base
fit memberships and optimality conditions, all 21 prequential calibration
memberships, the covariance estimate using a separate matrix formulation,
probabilities, bins and scores. No fitting is rerun. Maximum optimality residual:
**2.136×10⁻¹⁴**. Tests cover synthetic miscalibration recovery, symmetry, identity
fallback, maturity, out-of-training scope, protected boundaries and deliberate
corruption of the uncertainty receipt.

Paired uncertainty uses the unchanged 4,000 seven-day block bootstrap and
nine-cohort correction per endpoint. It does not jointly correct endpoints,
comparisons or this sequence of research attempts. Previously explored outcomes
remain unsuitable for claiming untouched validation. Conditional diagnostics
exclude realized flats without implying that flats can be known in advance.
Historical publication vintage remains unknown. No trading/profit claim is made.

No market source or outcome from **2026-08-01 onward** was opened. Older frozen
bundles and user files remain unchanged. No downloads or installations were
needed. Execution took **7.49s**, peak RSS **0.78GiB**; audit **7.16s**, peak RSS
**0.69GiB**. Bundle approximately **9.1MiB**, reusing source arrays.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/verify_market_calibration.py
```

To repeat execution use a fresh directory with the runner's `register` and `run`
commands and pass `--root` to the verifier. Existing outputs are exclusive-write.
This report and verification/runtime receipts were added after the execution
manifest.
