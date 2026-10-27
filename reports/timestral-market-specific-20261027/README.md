# Asset-specific hourly information: verified, insufficient improvement

Adding the asset's own hourly movement does **not** resolve the three failing
cohorts. Both registered rules still pass **6/9 original gates**, failing BTC
2025, SOL 2025 and SOL 2026. The primary consensus rule passes **0/9** conditional
improvement gates against the frozen common-hour model. The secondary equal
blend passes **1/9**, ETH 2026. The registered general-solution flag is **false**.

This experiment identifies why a plausible disagreement filter has little effect
on the failures. It does not establish a working general predictor. Results are
retrospective, on January–July of 2024, 2025 and 2026, not three complete years.

## What the earlier failures mean

The frozen common-hour model produces the same conditional direction probability
for all three assets. Its failure can reflect weak binary discrimination, the
unchanged-price class, or uncertainty in proper-score improvement. These must be
measured separately. The following diagnostics use its original 52% unconditional
direction-probability selection, before either new rule is applied:

| Cohort | Selected cases | Accuracy excluding realized flats | Accuracy including flats as errors | Selected flat share | Selected nonflat cases with asset/common sign disagreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTC 2025 | 9,751 | 52.029% | 51.933% | 0.185% | 6 |
| SOL 2025 | 6,743 | 52.096% | 51.431% | 1.275% | 1 |
| SOL 2026 | 4,632 | 53.573% | 52.440% | 2.116% | 1 |

Disagreement accounts for roughly 9–14% of all anchors across the nine cohorts,
and common-signal accuracy there is close to 50%. However, almost none of these
cases pass the original confidence threshold in the three failing cohorts.
Neutralizing them cannot materially repair that selected subset. This is a
descriptive diagnosis, not a causal or significance claim.

For BTC/SOL 2025 the selected all-class accuracy misses 52%. SOL 2026 already
clears that point threshold; its selected Brier-improvement interval includes
zero. The same uncertainty issue affects the other two failures. Excluding
realized flats after the event is not a usable forecast rule. Flat probabilities
remain frozen here so that a new flat model cannot obscure the directional test.
The complete nine-cohort diagnostic is in [diagnosis.json](diagnosis.json).

## Registered rules and controls

The [protocol](protocol.json) and code/input hashes were saved before new model
scores were computed. Earlier failure identities and descriptive diagnostics
informed the hypothesis, so registration does not make these reused outcomes an
untouched validation set. No settings changed after this run.

From the existing causal features, let `common` be the negative mean of the
three RMS-normalized trailing-hour returns, clipped to ±3. Let `own` be the
asset's negative normalized trailing-hour return, also clipped to ±3.

* **Primary consensus:** retain the frozen uncertainty-adjusted common probability
  when `common * own > 0`; otherwise use conditional up probability 0.5. No new
  parameter is fitted. Every evaluation anchor remains in scoring. Neutral cases
  count as up under the frozen tie convention; that is not directional ability.
  They cannot pass the original unconditional 52% selection.
* **Secondary blend:** use `(common + own) / 2` with one shared nonnegative
  logistic coefficient. Monthly fits use the prior 90 days of matured nonflat
  labels pooled across three assets, equal UTC-day weights and the same training
  availability mask as the previous studies. The frozen six-lag daily Bartlett
  uncertainty correction then shrinks the coefficient. There are 21 scalar fits
  and adjustments, with no fitted blend weights, asset IDs, regimes or sweep.

Both rules use the same frozen flat probabilities, all **550,362** original
anchors, and the five-minute target from anchor +60 seconds to +360 seconds.
Controls are the frozen adaptive common model, neutral 50/50, futures-flow model
and historical frequency. Seven registered comparisons per cohort are retained.
The primary general criterion requires all nine original gates and all nine
conditional gates against common, neutral and flow. The secondary is not promoted
after observing its scores.

Keeping model complexity low limits one source of overfitting; it does not prove
that overfitting has been eliminated, especially after repeated research on the
same years.

## All nine results

Brier gain is the common-model conditional loss minus the candidate loss,
multiplied by 10,000. These are score units, not trading-return basis points.
Selected accuracy includes unchanged-price outcomes as errors. Candidate
selections differ, so those accuracy columns alone are not a paired comparison.

| Cohort | Consensus Brier gain ×10⁴ | Blend Brier gain ×10⁴ | Consensus selected accuracy | Blend selected accuracy | Original gates, both rules |
| --- | ---: | ---: | ---: | ---: | --- |
| 2024 BTC | -0.0640 | +0.0928 | 53.080% | 53.206% | pass |
| 2024 ETH | +0.0083 | +0.0065 | 53.340% | 53.226% | pass |
| 2024 SOL | -0.0233 | +0.2482 | 53.275% | 52.921% | pass |
| 2025 BTC | +0.0628 | +0.1555 | 51.914% | 51.998% | fail |
| 2025 ETH | +0.0777 | +0.2004 | 53.730% | 53.815% | pass |
| 2025 SOL | -0.0614 | -0.0230 | 51.439% | 51.590% | fail |
| 2026 BTC | +0.0313 | -0.1334 | 52.977% | 52.839% | pass |
| 2026 ETH | +0.0760 | +0.4788 | 53.781% | 53.689% | pass |
| 2026 SOL | -0.0234 | +0.2586 | 52.429% | 52.147% | fail |

Consensus improves conditional Brier at point estimate in **5/9** cohorts;
blending does so in **7/9**, worsening SOL 2025 and BTC 2026. Neither improves
every cohort even before uncertainty is considered. Both pass **7/9** conditional
information gates against neutral and **0/9** against futures flow. Consensus
passes **0/9** against blend. All 63 comparisons and 18 original evaluations are
saved in [results.json](results.json).

All three failed cohorts retain selected Brier-gain intervals spanning zero:

| Cohort | Consensus gain interval, % | Blend gain interval, % |
| --- | ---: | ---: |
| BTC 2025 | [-0.1504, +0.4585] | [-0.1023, +0.4606] |
| SOL 2025 | [-0.1060, +0.4770] | [-0.0666, +0.4441] |
| SOL 2026 | [-0.0224, +1.1442] | [-0.0224, +1.0072] |

These are the frozen original 98.333% intervals for relative Brier improvement
against historical frequency on each candidate's selected subset. They differ
from the paired conditional comparisons, which use 99.444% intervals and absolute
loss differences. No threshold was weakened to count a failure as a success.

The remaining gap is evidence of a reliable advantage on the selected cases
where asset and market already agree. Reweighting their hourly returns does not
supply enough such evidence. These observations do not identify a unique missing
market variable or prove that a stable directional predictor is impossible.
Any later hypothesis needs separately registered evaluation; searching blend
weights or thresholds on these scores would add selection overfitting.

## Verification, reproducibility and limits

The [independent audit](verification.json) reconstructs source features and
labels, all monthly training memberships, scalar-fit gradients, daily covariance
through a separate matrix formulation, probabilities, diagnosis counts and all
reported scores. It does not refit models. The maximum raw-fit residual is
**2.205×10⁻¹⁴**. New tests cover fixed blending, sign reflection, consensus
neutrality and the distinction between binary accuracy and flat-class errors.
The relevant local suite passed **293 tests**, with eight database integration
tests skipped locally and included in CI.

The conditional comparisons retain 4,000 paired seven-day bootstrap blocks and
nine-cohort correction per endpoint. There is no joint correction across all
arms, endpoints or sequential hypotheses. Existing outcome reuse and unknown
publication vintage prevent an untouched/prospective validation claim. No costs,
execution or profitability claim follows from these scores.

No market source or outcome from **2026-08-01 onward** was opened. Earlier frozen
models, data, protocols, results, credentials and user files remain unchanged.
No downloads or installations were required. Run time was **8.00s**, peak RSS
**0.79GiB**; independent audit **7.92s**, peak RSS **0.69GiB**. Machine-readable
measurements are in [runtime.json](runtime.json).

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/verify_market_specific.py
```

For a fresh replay in a new directory:

```sh
.venv-timestral/bin/python scripts/run_market_specific.py register --root research/out/market-specific-replay
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/run_market_specific.py run --root research/out/market-specific-replay
.venv-timestral/bin/python scripts/verify_market_specific.py --root research/out/market-specific-replay
```

The execution manifest is frozen. This report and verification/runtime receipts
were added afterward. No failed cohort or comparison was discarded.
