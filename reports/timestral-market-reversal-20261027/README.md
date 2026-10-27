# A simple common-hour directional signal, with limited historical support

A **one-coefficient** reversal model based on the common market move over the
past hour passes the registered conditional comparison against **50/50 and past
frequency in all nine asset/year cohorts**. Conditional accuracy ranges from
**50.66% to 51.78%**, with all nine 99.444% intervals above 50% under the registered
per-endpoint correction. This identifies a modest historical directional signal.

It does **not** meet the registered general-solution criterion. The primary model
passes **0/9** improvement gates against the frozen 35-feature flow model and
**0/9** original selective direction gates. Fixed strong regularization makes
the probabilities too close to 50%; it does not improve the predicted direction.
The unpenalized diagnostic arm passes **6/9** original gates, also short of all
nine. The general-solution flag remains false.

## Why this follow-up was tested

The [first simplification](../timestral-simple-direction-20261027/README.md)
removed the common one-minute movement and failed. An inspection of already
explored flow-model coefficients showed negative average own/other 60-minute
slopes in each of the three years. This motivated testing that shared slower
component alone. It is a **sequential exploratory hypothesis**, informed by
previous outcomes and fitted models, not independent confirmation.

The [new protocol](protocol.json) froze this follow-up before its execution. No
parameter, horizon, confidence threshold or gate was changed after its scores.

## One parameter, causal inputs

Each asset's trailing-hour log return is normalized by its causal prior-minute
60-minute RMS times `sqrt(60)`, using the frozen feature implementation. The
three normalized returns are averaged equally and negated, then clipped to ±3.
Thus a common downward hour predicts a small upward tendency and vice versa.
All spot bars close through the forecast anchor; the target is the following
five-minute return from anchor+1min to anchor+6min. This is a hypothesis of
short-term reversal, not evidence of its economic cause.

`P(up | non-flat) = sigmoid(beta × negative_common_hour_return)`.

The scalar beta is shared across all assets and constrained to [0,8]. Every
month uses only matured non-flat labels in the preceding 90 days, with equal
total weight per UTC training day across the pooled assets. There is no
intercept, asset-specific parameter, fitted regime or interaction. No fitted
normalization is added. The exact one-dimensional convex optimum is solved by
its derivative; beta zero would return 50/50. The imposed nonnegative sign is
not itself evidence of stability; the subsequent evaluations supply the evidence.

Three arms were fixed:

- `market1`: the corresponding common one-minute signal, penalty 0.25.
- `market60_unshrunk`: common-hour signal, penalty zero.
- `market60`, primary: common-hour signal, penalty 0.25.

The objective is day-weighted log loss plus `penalty × beta² / 2`. The penalty
was reused unchanged from the first simplification, not selected using these
scores. The upper bound eight is a numerical guard. All **550,362** original
evaluation anchors are retained, with unchanged flat probabilities for original
all-anchor direction gates. Training membership matches the frozen flow control.
Futures data determine that membership and supply the saved control; new models
use no futures features. All sources and environments were reused.

## Three-year evaluation

Periods are **January–July 2024, 2025 and 2026**, not full calendar years.
Accuracy below is conditional on non-flat outcomes without confidence selection.
Brier gain is control loss minus primary loss; positive is better. Values ×10⁴
are **score units, not trading-return basis points**.

| Cohort | Accuracy | 99.444% accuracy interval | Brier gain vs 50/50 ×10⁴ | Brier gain vs flow ×10⁴ |
| --- | ---: | ---: | ---: | ---: |
| 2024 BTC | 51.782% | [51.312%, 52.318%] | +2.119 | -2.299 |
| 2024 ETH | 51.429% | [50.982%, 51.966%] | +1.843 | +0.156 |
| 2024 SOL | 51.140% | [50.797%, 51.580%] | +1.451 | +0.352 |
| 2025 BTC | 50.660% | [50.132%, 51.228%] | +0.855 | +0.358 |
| 2025 ETH | 51.572% | [50.951%, 52.158%] | +2.306 | -3.243 |
| 2025 SOL | 50.909% | [50.258%, 51.519%] | +1.104 | -0.406 |
| 2026 BTC | 51.376% | [50.723%, 51.902%] | +2.215 | -2.333 |
| 2026 ETH | 51.607% | [51.065%, 52.140%] | +2.875 | -6.365 |
| 2026 SOL | 51.494% | [50.946%, 52.073%] | +2.271 | -4.474 |

All nine conditional gates against 50/50 also require positive corrected lower
bounds for Brier/log-loss gains against historical frequency, at least five
positive monthly Brier gains against the named control, and at least 100 forecasts
in each direction. All pass. Versus common one-minute reversal **8/9** pass;
BTC 2025 does not. Versus frozen flow **0/9** pass. All 36 comparisons, uncertainty
intervals and 27 original cohort/arm evaluations are in [results.json](results.json).

## What simplification fixes, and what it does not

All 21 common-hour monthly fits support a positive beta. The regularized
coefficients range from **0.02260 to 0.04449**; unpenalized ones from **0.05756 to
0.10976**. With positive scalar coefficients, regularization changes probability
magnitude but leaves the up/down decisions identical. It cannot improve the
directional accuracy by itself.

Primary mean confidence is only **50.50–50.64%**, below observed accuracy in all
nine cohorts by **0.15–1.28 percentage points**. This is evidence of conservative
confidence on this aggregate diagnostic, not a complete calibration assessment.
The original 52% unconditional-probability threshold selects only
**0.036–0.948%** of anchors, below its required 5% coverage everywhere. Hence all
nine primary original gates fail. Reducing overconfidence too aggressively has
created underconfidence rather than a general solution.

Regularization worsens Brier versus the unpenalized common-hour arm in **8/9**
cohorts at point estimate; BTC 2025 is the exception. The unpenalized arm passes
six original gates but fails BTC 2025, SOL 2025 and SOL 2026. Their selected-sample
gain intervals include zero; the first two also fall below 52% selected accuracy.
Neither arm was retuned or promoted after seeing this distinction.

The useful finding is narrower: a single common-hour signal preserves a modest
directional advantage across the nine explored cohorts. Its strength still varies
and it has not established superiority to the richer control. This does not show
why market participants generate the effect, nor establish future profitability.

## Verification and limits

The [independent audit](verification.json) reconstructs source features, scalar
signals, all 63 training memberships, day weights, constrained-optimum conditions,
saved forecasts, calibration summaries and all registered metrics. It does not
refit or use production feature/model readers. Maximum KKT residual:
**1.869×10⁻¹⁴**. Synthetic tests verify the planted relative mechanism, coefficient
shrinkage, symmetry, unsupported-orientation fallback, maturity/boundary rules,
equal day weights and horizon/asset invariance of the common signal.
The full local relevant suite passed **284 tests**, with eight Timescale integration
tests skipped locally; CI runs those against real TimescaleDB and both new audits.

Uncertainty uses 4,000 paired seven-day moving blocks with nine-cohort correction
per endpoint. It does **not** jointly correct endpoints, comparisons or the many
sequential hypotheses explored on these years. Passing this retrospective test
does not constitute untouched or prospective confirmation. Flat outcomes are
excluded only from conditional diagnostics; that cannot identify future flats or
demonstrate trading coverage. Publication latency/historical vintage remain
unknown. No transaction-cost or profitability claim is made.

No market source or outcome from **2026-08-01 onward** was opened. Frozen earlier
bundles and user files remain unchanged. No orders, installations or downloads
were made. Execution took **8.13s**, peak RSS **0.84GiB**; the independent audit
took **6.72s**, peak RSS **0.69GiB**. Bundle approximately **12.4MiB** before
report/receipts/Git objects. See [runtime.json](runtime.json).

## Reproduce

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/verify_market_reversal.py
```

Re-execution requires a new directory; frozen outputs are never overwritten:

```sh
.venv-timestral/bin/python scripts/run_market_reversal.py register --root research/out/market-replay
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/run_market_reversal.py run --root research/out/market-replay
.venv-timestral/bin/python scripts/verify_market_reversal.py --root research/out/market-replay
```

This report and verification/runtime receipts were added after the execution
manifest. Both experiments, including the rejected first hypothesis, remain
available. A functioning general predictor has **not** been established.
