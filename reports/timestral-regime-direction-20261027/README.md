# Recurring regimes: diagnosed overfitting, no general direction solution

The fixed three-arm experiment completed on **January–July 2024, 2025 and 2026**,
for BTCUSDT, ETHUSDT and SOLUSDT. Adding recurring trend/activity regimes does
not resolve directional generalization: **0/9** cohorts pass either incremental
regime-information comparison. The new model passes **3/9** original selective
direction gates, versus **5/9** for the frozen-compatible futures-flow control
and **5/9** with long context alone. Both combined gates remain false.

The diagnosis narrows the problem. The two fixed simple signals keep the same
aggregate orientation across all nine cohorts. Their small accuracy changes are
mostly within the four measured conditions, rather than changes in how often
those conditions occur. A more flexible model fits past data better but scores
worse later. These observations do not identify the economic cause of drift,
nor distinguish every within-condition change from sampling noise.

## Registered comparison

The [protocol](protocol.json) and code/input hashes were frozen before execution.
All source data came from previously frozen, pre-protected arrays. No data,
model or environment downloads were needed. Three models use identical monthly
90-day training windows, pooled assets, matured non-flat labels and fixed
sign-symmetric logistic regression parameters:

- **Flow, 35 features:** the previous spot/futures representation.
- **Long, 39 features:** flow plus normalized signed 4-hour/24-hour returns of the
  asset and the other two assets.
- **Regime, 71 features:** long context plus 32 interactions. Eight directional
  signals interact with four centered state indicators, allowing conditional
  slopes while retaining common coefficients and regularization.

The four states are fixed combinations of 4-hour trend strength and trading
activity. Trend means absolute net log return divided by the square root of
summed squared minute log returns is at least one. Active means the latest
hour's mean trade count is at least the preceding day's mean, with that latest
hour excluded from the reference. Thresholds are not estimated from outcomes.
The earliest context needs 1,501 complete spot minutes.

Spot context ends at the anchor; futures context ends one minute earlier. The
target remains the sign of the five-minute spot return from anchor+1min to
anchor+6min. Flat-price probabilities are identical across all models. All
**550,362** original evaluation cases remain; no evaluation context was missing.
The flow control reproduces the previous study's saved probabilities.

The general claim requires all nine original direction gates and all nine
conditional information gates against **both** flow and long controls. No
threshold, state definition, model or feature was changed after scores opened.

## Directional results

Accuracy is conditional on non-flat outcomes, without confidence selection.
Brier gain is control loss minus regime loss: negative is worse. Values are
multiplied by 10,000 for readability and are **not trading-return basis points**.

| Cohort | Regime accuracy | Brier gain vs flow ×10⁴ | 99.444% interval ×10⁴ | Original direction gate |
| --- | ---: | ---: | ---: | --- |
| 2024-BTCUSDT | 51.84% | -0.582 | [-2.590, +1.166] | fail |
| 2024-ETHUSDT | 51.18% | -0.407 | [-2.027, +0.795] | fail |
| 2024-SOLUSDT | 51.20% | -0.184 | [-1.175, +0.844] | fail |
| 2025-BTCUSDT | 51.02% | -0.874 | [-2.210, +0.403] | fail |
| 2025-ETHUSDT | 51.66% | -0.279 | [-1.740, +1.110] | pass |
| 2025-SOLUSDT | 51.25% | -0.087 | [-2.186, +1.888] | fail |
| 2026-BTCUSDT | 51.78% | -1.117 | [-2.799, +0.866] | fail |
| 2026-ETHUSDT | 52.17% | -2.335 | [-3.785, -0.749] | pass |
| 2026-SOLUSDT | 52.38% | -0.746 | [-2.529, +1.119] | pass |

Both Brier and log loss are worse than flow in **all nine** cohorts. ETH 2026's
Brier interval is entirely negative under the registered per-endpoint correction.
The other eight Brier intervals include zero. Long context alone also fails all
nine incremental-information gates. See [results.json](results.json) for all
27 conditional comparisons and all 27 original cohort/model evaluations.

## Why the apparent signal changes

Two fixed, untrained rules make the comparison interpretable: predict the
opposite sign of the latest spot return; or predict the sign of the lagged
futures aggressor imbalance. Ties and flat outcomes are excluded only from this
descriptive diagnostic, with all exclusions recorded. This cannot identify
future flats or establish tradable coverage.

For each rule, accuracy is a weighted average of accuracy inside the four
states. Relative to 2024, the exact decomposition is:

`change = sum((new_weight − old_weight) × old_accuracy)`
`       + sum(new_weight × (new_accuracy − old_accuracy))`.

The first term is the changed mix of observed conditions; the second is the
observed change within those conditions. Below, components are percentage
points for **2026 minus 2024**. The corresponding 2025 decomposition is also
saved in [diagnosis.json](diagnosis.json).

| Asset | Fixed rule | 2024 accuracy | 2025 accuracy | 2026 accuracy | Mix component, pp | Within-state component, pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | spot_reversal | 50.675% | 50.562% | 50.927% | -0.0221 | +0.2738 |
| BTCUSDT | futures_flow | 49.657% | 49.649% | 49.375% | -0.0147 | -0.2673 |
| ETHUSDT | spot_reversal | 50.461% | 50.423% | 51.130% | +0.0012 | +0.6682 |
| ETHUSDT | futures_flow | 49.609% | 49.523% | 49.242% | +0.0094 | -0.3759 |
| SOLUSDT | spot_reversal | 50.296% | 50.128% | 50.852% | -0.0073 | +0.5640 |
| SOLUSDT | futures_flow | 49.888% | 49.716% | 49.266% | +0.0080 | -0.6305 |

Spot reversal has point accuracy above 50% in all nine cohorts; following the
futures flow has point accuracy below 50% in all nine. Thus the aggregate **sign
is not flipping** for these two particular rules. The strength is weak and
variable. Some 95% seven-day block intervals include 50%; apparent differences
between individual states must not be treated as established mechanisms.
Neither pooled averages nor selected favorable states replace the registered
nine-cohort requirement.

The measured state mix explains little of the year-to-year point changes. This
rejects the simple descriptive explanation that these four conditions merely
occur in different proportions. It does **not** prove a causal market mechanism:
unmeasured conditions, execution delay and sampling variation remain possible.

After the negative results, an explicitly **post-hoc diagnostic**, not used for
model selection, compared fit and subsequent losses. The regime model improves
training log loss in **21/21** monthly windows against each simpler control, but
worsens subsequent cohort log loss in **9/9** against flow. Mean training loss
gain versus flow is 0.0003814. This is concrete evidence of overfitting for this
candidate, rather than evidence that increasing model complexity solves drift.
All 63 training losses are recorded by the independent audit.

## Verification and limits

Five synthetic regression tests check causality, complete context, reflection,
exact decomposition and recovery of a planted signal whose direction flips by
known state. The planted-signal success validates the implementation, not the
existence of the same mechanism in markets.

The [independent audit](verification.json) reconstructs contexts with direct
convolutions, independently checks states/interactions, all 63 training
memberships and fit gradients, all saved probabilities, the frozen flow control,
all scores, marginal diagnostic intervals and all 12 decompositions. Maximum
optimality residual is **1.381×10⁻⁷**. No fitting is rerun in the verifier.
The additional training-loss diagnostic does not change frozen result artifacts.
Local relevant tests: **278 passed, 8 skipped**; CI includes TimescaleDB integration
and this saved-evidence audit.

No market source/outcome from **2026-08-01 onward** was used. Old protocols,
models, artifacts and user files remain unchanged. The years cover January–July,
not full years. Outcomes were previously explored, so this remains retrospective
research. The conditional gate uses 4,000 paired seven-day blocks and nine-cohort
correction per endpoint, not a joint correction across all research history.
Diagnostic 95% intervals are marginal and uncorrected; the descriptive
decomposition is not a causal or significance test. Archive publication latency
and historical revisions remain unknown. No trading-profitability claim is made.

Execution took **66.73s**, peak RSS **2.64GiB**; independent audit **15.11s**, peak
RSS **2.60GiB**. The new bundle is approximately **13.2MiB** before Git objects,
with existing source arrays reused. See [runtime.json](runtime.json).

## Reproduce

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/verify_regime_direction.py
```

To execute again, use a fresh directory; existing artifacts use exclusive writes:

```sh
python scripts/run_regime_direction.py register --root research/out/regime-replay
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python scripts/run_regime_direction.py run --root research/out/regime-replay
python scripts/verify_regime_direction.py --root research/out/regime-replay
```

The manifest freezes the executed experiment. This report, runtime and independent
verification receipts were added afterward. A general functioning predictor has
**not** been established; reusing these outcomes to pick a winning regime would
not supply the missing validation.
