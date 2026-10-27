# Direct uncertainty adjustment: useful correction, incomplete solution

Replacing excessive fixed regularization with measured uncertainty retains
**one final common-hour coefficient**. Compared with the conservative base,
fixed-bin calibration error and Brier score improve in **8/9** cohorts at point
estimate; **5/9** corrected conditional improvement gates pass. The original
direction gates improve from 0/9 for that conservative base to **6/9**.

This does **not** establish a general solution or an improvement over the frozen
unpenalized model: the latter also passes 6/9 original gates, and **0/9** paired
improvement gates against it pass. Both registered calibration-success and
general-solution flags remain false. The result is a limited historical
calibration correction, not a validated trading system.

## Why a second calibration experiment

The preceding [prequential factor experiment](../timestral-market-calibration-20261027/README.md)
improved aggregate calibration but worsened Brier against the unpenalized model
in 8/9 cohorts. This motivated a separately registered adjustment of the original
coefficient itself. The follow-up is explicitly informed by previous outcomes;
it is not independent confirmation. The [protocol](protocol.json) was frozen
before this run, and no setting changed after the results.

## Fixed one-coefficient rule

The signal is unchanged: negate the equally weighted mean of the three assets'
causal RMS-normalized trailing-hour returns, clipped to ±3. A positive scalar
coefficient maps it to conditional up probability through a sigmoid. The frozen
unpenalized coefficient is reused for every month without optimization or an
extra fitted temperature factor.

Using exactly its original trailing-90-day matured non-flat training samples,
pooled across BTC/ETH/SOL with equal total weight per UTC day, compute the daily
logistic scores and average Hessian. A six-lag Bartlett covariance estimate
gives the coefficient variance. The fixed correction is:

`weight = max(0, 1 − estimated_variance / raw_beta²)`

`adaptive_beta = weight × raw_beta`.

Numerical floors and the zero-information fallback are in the protocol. This
heuristic shrinks uncertain estimates toward neutral probability. It does not
prove that future calibration error or overfitting cannot increase. Twenty-one
adjustment states are saved; there are no newly optimized coefficients, asset
offsets, regimes, intercepts or parameter searches.

All actual weights lie between **0.9558 and 0.9933**: the measured uncertainty
supports a small reduction of the raw coefficient, rather than the much stronger
fixed penalty used earlier. The coefficients stay positive. **No up/down
decision changes** in any cohort. This preserves directional accuracy, while
changing confidence and which cases exceed the original selection threshold.

## Results across the three years

The scope is January–July **2024, 2025 and 2026**, for all three assets; all
**550,362** original anchors and their five-minute targets are retained. Frozen
flat probabilities and original 52% selection requirements are unchanged.
Conditional directional accuracy therefore remains **50.66–51.78%**. The nine
accuracy intervals remain above 50%, but proper-score gates can still fail.

Brier gain is control loss minus adaptive loss, multiplied by 10,000. These are
**score units, not trading-return basis points**. ECE uses the preregistered
confidence edges 50%, 51%, 52%, 54%, 58%, 66%, 100%, with bin counts retained.

| Cohort | Gain vs conservative base ×10⁴ | Gain vs unpenalized ×10⁴ | Fixed-bin ECE: base → adaptive | Original gate |
| --- | ---: | ---: | ---: | --- |
| 2024 BTC | +1.639 | -0.016 | 1.282% → 0.525% | pass |
| 2024 ETH | +1.245 | -0.006 | 0.928% → 0.207% | pass |
| 2024 SOL | +0.659 | +0.006 | 0.639% → 0.124% | pass |
| 2025 BTC | -0.142 | +0.065 | 0.152% → 0.551% | fail |
| 2025 ETH | +1.869 | -0.076 | 1.064% → 0.656% | pass |
| 2025 SOL | +0.204 | +0.039 | 0.401% → 0.324% | fail |
| 2026 BTC | +0.948 | +0.011 | 0.732% → 0.190% | pass |
| 2026 ETH | +1.874 | -0.019 | 0.964% → 0.356% | pass |
| 2026 SOL | +1.006 | +0.010 | 0.850% → 0.159% | fail |

The paired information gates pass **5/9** versus conservative base and **7/9**
versus 50/50, each also requiring improvement against historical frequency.
They pass **0/9** versus unpenalized, prior adjusted temperature and flow.
All 45 comparisons, intervals and reliability tables are in
[results.json](results.json).

The adaptive coefficient improves Brier against the previous temperature
correction in 8/9 cohorts at point estimate, but no corresponding strict
information gate passes. Against unpenalized, Brier improves in 5/9 and worsens
in 4/9 at point estimate; the small differences do not establish superiority.
Both adaptive and unpenalized fail the same three original gates: BTC 2025,
SOL 2025 and SOL 2026.

Mean confidence is **51.21–51.55%**, compared with approximately 50.5–50.6% for
the conservative base. Original-threshold coverage is **7.59–27.05%**, clearing
the 5% minimum everywhere. Thus insufficient coverage is repaired; the remaining
failures concern predictive strength and uncertainty in score improvement.
For BTC/SOL 2025, selected accuracy remains below 52%. The selected Brier-gain
intervals for the failed cohorts include zero.

This separates the issues: a calibration correction can repair excessive
underconfidence and coverage while leaving a weak or variable directional
advantage unresolved. Since positive scalar rescaling leaves the direction
unchanged, it cannot manufacture stronger discrimination. A smaller ECE alone
does not demonstrate better proper scores, lack of overfitting or profitability.

## Verification and limits

The [independent audit](verification.json) reconstructs features, all 21 frozen
raw-fit memberships/gradients and adjustment states, covariance via a separate
matrix formulation, saved predictions, all scores and fixed-bin diagnostics.
It never refits a model. Maximum raw-fit residual: **1.560×10⁻¹⁴**. Synthetic
tests cover strong/weak-signal shrinkage, reflection, maturity and forecasting
boundaries. The combined relevant local suite passed **290 tests**, with eight
database integration tests reserved for CI.

Uncertainty retains the 4,000 paired seven-day block bootstrap and nine-cohort
correction per endpoint. There is no joint correction over endpoints, comparisons
or the sequence of hypotheses tried on these years. Calibration ECE is a
bin-dependent descriptive statistic, not a significance test. Outcomes were
previously explored, and these results are not untouched or prospective
validation. Excluding realized flat returns cannot establish tradable coverage.
Publication vintage is unknown; no orders, costs or profitability claim apply.

No market source or outcome from **2026-08-01 onward** was used. Earlier frozen
artifacts, models and user files remain unchanged. No downloads or installations
were needed. Execution: **6.88s**, peak RSS **0.78GiB**; independent audit:
**7.04s**, peak RSS **0.68GiB**. The bundle is approximately **5.2MiB** before
documentation/receipts/Git objects, with source arrays reused.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/verify_uncertain_market.py
```

For a fresh replay:

```sh
.venv-timestral/bin/python scripts/run_uncertain_market.py register --root research/out/uncertain-market-replay
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/run_uncertain_market.py run --root research/out/uncertain-market-replay
.venv-timestral/bin/python scripts/verify_uncertain_market.py --root research/out/uncertain-market-replay
```

The execution manifest remains frozen. This report and runtime/verification
receipts were added afterward. No unsuccessful result or cohort was discarded.
