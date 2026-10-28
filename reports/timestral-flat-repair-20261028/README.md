# Flat-probability shrinkage repairs all original historical gates

The fixed primary correction passes **9/9 original gates**, repairing SOL 2025
while retaining the other eight passes. It adds **zero fitted parameters** and
leaves every conditional up/down probability unchanged. On SOL 2025 it also
improves Brier and log loss with positive paired confidence bounds on both all
cases and the exact old selected subset. This is more than a selection change.

It is **not universally better** than the preceding model: all-case log loss
worsens significantly for BTC 2024 and BTC 2026. The registered `repair_success`
is true, while `calibration_success` and `general_solution` remain **false**.
Passing historical gates does not establish prospective or trading performance.

## Frozen primary and ablation

The previous experiment isolated a flat/moving loss penalty in SOL 2025 that
offset its improved conditional direction. This follow-up chooses its frozen
uniform-center direction arm because of those earlier results; that choice is
explicitly sequential and retrospective. The [new protocol](protocol.json) and
code/input hashes were saved before opening the correction's scores.

Let `q` be the frozen uniform-center conditional up probability, `f` the existing
flat probability, and `h` the existing per-asset historical flat frequency:

* **Primary mixed:** `flat = (f + h) / 2`.
* **Secondary frequency:** `flat = h`, eliminating the complex flat model.
* **Old control:** `flat = f`.

Recombine as `P(down) = (1 − flat) × (1 − q)`, `P(up) = (1 − flat) × q`.
The half-weight was fixed before scores, not optimized. No new direction fit,
calibration coefficient, asset exception, horizon, confidence threshold or
parameter sweep is introduced. The secondary remains an ablation rather than
being promoted after observing its results.

Historical frequencies are already-issued monthly baseline probabilities:
`(class_count + 1) / (training_count + 3)`, estimated separately per asset on
the preceding 90 days with target end no later than the month start. The
independent verifier reconstructs all **63 asset/month windows** directly from
source prices and checks exact memberships, labels and frequency values.
Frequency adaptation uses past outcomes; no future flat labels enter predictions.

All **550,362** original anchors are retained for January–July **2024, 2025 and
2026** and BTC/ETH/SOL. The target remains the five-minute spot return from
anchor +60 seconds to +360 seconds. This is not a full-year evaluation.

## Nine-cohort results

Original gates are unchanged: at least 5% selected coverage, at least 52%
selected accuracy, original confidence bounds, positive selected Brier gain
against historical frequency, no log-loss degradation against that baseline,
at least 100 calls in each direction and at least five positive months with
sufficient calls. New flat probabilities can change threshold selection even
though conditional direction is identical.

The following gains compare mixed with the old model on **all identical cases**,
not each model's different selected subset. They are multiplied by 10,000 for
readability and are score units, not trading-return basis points.

| Cohort | Mixed selected accuracy | Mixed coverage | All-case Brier gain ×10⁴ | All-case log-loss gain ×10⁴ | Original gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 2024 BTC | 53.746% | 19.324% | -0.1185 | -7.0887 | pass |
| 2024 ETH | 53.412% | 18.727% | +0.0934 | +4.0325 | pass |
| 2024 SOL | 52.357% | 10.788% | -0.1903 | +2.2744 | pass |
| 2025 BTC | 53.370% | 22.115% | -0.1574 | -2.7711 | pass |
| 2025 ETH | 54.665% | 21.749% | +0.0803 | +4.7793 | pass |
| 2025 SOL | 52.521% | 14.713% | +0.3422 | +13.3415 | pass |
| 2026 BTC | 53.389% | 35.381% | -0.0883 | -5.5284 | pass |
| 2026 ETH | 54.155% | 35.040% | +0.0093 | +0.7818 | pass |
| 2026 SOL | 53.253% | 12.462% | +3.3638 | +13.4633 | pass |

Mixed improves all-case Brier in **5/9** and log loss in **6/9** at point estimate.
The paired improvement gate passes **4/9** on all cases (ETH 2024, ETH/SOL 2025,
SOL 2026), and **1/9** on the exact old selected subset (SOL 2025). Frequency-only
also passes 9/9 original gates, but only 1/9 paired improvement gates in either
scope. Its all-case Brier and log loss improve in only 3/9 cohorts each.

The mixed/old all-case log-loss-gain intervals are entirely negative for BTC
2024, **[-0.00174615, -0.00014231]**, and BTC 2026,
**[-0.00111398, -0.00006431]**, under the registered 99.444% per-cohort endpoint
intervals. The unchanged original gates do not require improvement against the
old candidate; they use historical frequency. Keeping these separate prevents
the 9/9 count from hiding a real regression.

All 36 paired scope/arm comparisons, 27 original evaluations and 27 loss
decompositions are in [results.json](results.json).

## SOL 2025: repair survives a fixed-selection comparison

Mixed selects **8,983** cases, versus 10,015 previously. Its accuracy is
**52.521%**, with original 98.333% interval **[51.298%, 53.838%]** and coverage
**14.713%**. The selected relative Brier-improvement interval against historical
frequency is **[0.1402%, 0.7797%]**. Log loss is **0.75252595**, better than the
baseline's **0.75464384** on those same new selected cases. All original criteria
therefore pass without threshold changes.

To rule out an apparent gain caused solely by reselecting cases, the protocol
also froze the previous selection. On those **same 10,015 cases**, mixed versus
old gives:

| Metric | Gain | Paired 99.444% interval |
| --- | ---: | --- |
| Three-class Brier | +0.00005141 | [+0.00001993, +0.00008242] |
| Three-class log loss | +0.00243298 | [+0.00091009, +0.00389109] |

Across all 61,055 SOL 2025 cases, Brier gain is **0.00003422** with interval
**[0.00001623, 0.00005447]**, and log-loss gain **0.00133415** with interval
**[0.00065971, 0.00211285]**. The mixed Brier gain is positive in all seven months
on the full cohort and six months on the old selected subset.

The planned loss decomposition confirms the mechanism of this numerical repair.
On the mixed-selected cases, the direction contribution improves loss against
historical frequency by **0.00232805**. The remaining flat/moving penalty is
only **0.00021017**, yielding total improvement **0.00211789**. Previously, on
the old selected cases, the flat penalty was **0.00271070**, exceeding the
direction gain **0.00236655**. These subsets differ; the paired fixed-selection
comparison above is the controlled evidence of improvement. Conditional
direction itself did not improve in this experiment.

## What remains unresolved

The correction resolves the remaining original historical gate, but not every
remaining research requirement. The flat-probability improvement is not uniform
across assets and years, and two BTC log-loss regressions are statistically
visible. Conditional direction is preserved exactly, so its earlier **5/9**
information-gate result against the common-hour control is also unchanged.
Calibration cannot repair that missing directional evidence by rescaling the
flat class. The registered general criterion explicitly retains this requirement
rather than redefining success after a favorable 9/9 original-gate result.

No per-asset exceptions were introduced to remove the observed regressions.
Zero new fitted parameters limits flexibility but does not eliminate selection
overfitting from repeated studies on the same outcomes. All years have already
been explored. No untouched or prospective performance is claimed, and no
protected holdout was used to resolve these limits.

## Independent verification and resources

The [audit](verification.json) reconstructs historical frequency from source
prices, recombines the fixed probability rules independently, checks that every
conditional up/down probability and direction is preserved, and recomputes all
scores, decompositions and gates. Paired uncertainty is independently aggregated
through a matrix of sampled-day counts. No model is refit.

Paired comparisons use 4,000 seven-day moving-block replicates, seed 271032,
with nine-cohort correction separately per endpoint. Original intervals remain
98.333% under their frozen three-asset convention, without a joint correction
over all nine year/asset cohorts. Neither scheme jointly corrects across
endpoints, scopes, arms and the sequential research history.

Three new tests cover valid probabilities and reflection, exact preservation of
conditional direction, rejection of invalid inputs, fixed-subset flat/direction
loss accounting and the zero-gain identity control. The relevant local suite
passed **301 tests**, with eight database integration tests reserved for CI.

No market source or outcome from **2026-08-01 onward** was opened. Previous
frozen models, protocols, data, results, credentials and user files remain
unchanged. No downloads, installations, orders or deployments were required.
Run time: **5.08s**, peak RSS **0.26GiB**; independent audit: **5.77s**, peak RSS
**0.18GiB**. See [runtime.json](runtime.json).

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/verify_flat_repair.py
```

Fresh replay into a new directory:

```sh
.venv-timestral/bin/python scripts/run_flat_repair.py register --root research/out/flat-repair-replay
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/run_flat_repair.py run --root research/out/flat-repair-replay
.venv-timestral/bin/python scripts/verify_flat_repair.py --root research/out/flat-repair-replay
```

The execution manifest is frozen. This report and runtime/audit receipts were
added afterward; no failed comparison was removed.
