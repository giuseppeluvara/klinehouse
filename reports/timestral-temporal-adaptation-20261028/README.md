# Temporal adaptation with the same six experts does not establish stable improvement

Three prespecified adaptation rules preserve **9/9 original historical gates**
each, but none improves direction, full Brier and log loss stably across the
nine asset/year cohorts. Every arm has **0/36 full-score comparison gates** and
**0/18 directional comparison gates**. The primary share rule's
**`primary_stable_improvement` is false**; both secondary arms also fail.

The negative conclusion is not based only on stringent intervals. Against the
same-pool cumulative control, each method improves full Brier and conditional
directional Brier/log loss in only **1/9** cohorts, SOL 2024. In the six cohorts
from 2025–2026, all three methods worsen these three proper scores at point
estimate against both controls. Accuracy gains vary, and none has a positive
registered lower confidence bound. Some overall log-loss gains are real and
localized; they do not form a stable directional improvement.

## Frozen, matched-pool comparison

The [protocol](protocol.json), code, verifier and input hashes were recorded
before new forecast scores were opened. Exactly the same six issued forecast
sequences are used in every arm: `common_old`, `common_mixed`, `uniform_old`,
`uniform_mixed`, `equal` and `logloss_adaptive`. Their internal fits and adaptation
are unchanged. Only the outer combination's treatment of past performance
changes. This removes the pool-expansion confound in the preceding experiment.

The control `cumulative` is the exact saved defensive forecast from the prior
multiscore study, with this same six-expert pool and forecast solver.
`uniform_mixed` is the second, stronger static comparator. The data retain all
**550,362** original anchors: January–July **2024, 2025 and 2026**, BTC/ETH/SOL,
not complete years. The five-minute target remains anchor +60 to +360 seconds.
There are **1,651,086 new arm forecasts** across the three methods.

Each arm maintains two delayed lanes per asset/year, with twelve equal evaluator
priors per lane: six experts times log loss and full three-class Brier, both
with learning rate one. Forecast `i` can use only the mature same-lane outcome
at `i-2`. The feedback increment is that arm's own issued forecast loss minus
each expert's loss, separately for each score. The frozen defensive probability
solver and all numerical tolerances remain identical across arms and control.

| Rule | Registered temporal update |
| --- | --- |
| **share — primary** | After mature feedback, redistribute a fraction `alpha = 1 - exp(-dt / 30 days)` uniformly among experts within each score, preserving that score's total weight. |
| **discount — secondary** | Multiply both stored log-regret and newly received feedback by `rho = 2^(-dt / 30 days)`, a 30-day half-life measured from forecast anchors. |
| **monthly — secondary** | At the first forecast of each new UTC month in each lane, reset all twelve weights and discard pending feedback from an old-month anchor. That old forecast remains scored. |

`dt` is elapsed wall time between successive forecasts in the same lane, so gaps
are handled explicitly. Share's 30 days is a refresh time constant, not a
half-life; because uniform redistribution includes the current expert, its
actual switch probability is `5 alpha / 6`. These are three fixed rules at
stated time scales, not a search over equivalent or optimal memory lengths.
The monthly rule has two coldstarts per month, all included in evaluation.
No arm is selected for an individual asset/year after seeing results.

The expert-sharing idea follows
[Herbster and Warmuth](https://mwarmuth.bitbucket.io/pubs/J39.pdf), while the
reduced influence of older performance follows the discounted-loss setting of
[Chernov and Zhdanov](https://arxiv.org/pdf/1005.1918). These sources motivate
adaptation; they do not establish that these chosen rules improve these data.
The study scores full-period losses without discounting the reported errors.

## Success criteria and uncertainty

Original coverage, selection, baseline and accuracy gates are unchanged. New
comparisons use identical cases for each pair:

- Full three-class Brier and log loss on all anchors, and on the exact frozen
  `uniform_mixed` selection with maximum up/down probability at least 52%.
- Conditional directional Brier (single binary square), log loss and accuracy
  on all moving outcomes. A conditional up probability of exactly 0.5 predicts
  up. Accuracy gains are method minus control; loss gains are control minus method.

Each new comparison requires positive lower confidence bounds for **every**
metric and at least five positive monthly gains for **every** metric. An arm's
stable-improvement criterion requires all nine original passes, all 36
full-score comparison passes (nine cohorts, two scopes, two controls), and all
18 direction passes. The primary was fixed as share; secondary outcomes do
not replace it. These criteria were registered before evaluation and do not
rewrite the earlier experiments' general-solution definitions.

Intervals use **12,000 paired seven-day moving-block resamples**, seed 271032,
with quantiles `1/1080` and `1079/1080`: **99.8148148%** intervals correcting
across 27 arm/cohort comparisons for each fixed endpoint, control and scope.
This does not jointly correct over endpoints, controls, scopes or the history
of sequential research. There is no bootstrap refitting. The old original-gate
98.333% intervals retain their earlier convention. These repeatedly explored
cohorts remain retrospective evidence, not independent confirmation.

## Counts of improvements

These counts are **positive mean gains**, not significance or gate passes.
The conditional Brier and conditional log-loss counts coincide in every row.

| Method | Control | Full Brier | Full log loss | Each directional proper score | Direction accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| share | cumulative | 1/9 | 3/9 | 1/9 | 5/9 |
| discount | cumulative | 1/9 | 3/9 | 1/9 | 4/9 |
| monthly | cumulative | 1/9 | 1/9 | 1/9 | 4/9 |
| share | uniform_mixed | 1/9 | 4/9 | 1/9 | 3/9 |
| discount | uniform_mixed | 1/9 | 4/9 | 1/9 | 3/9 |
| monthly | uniform_mixed | 1/9 | 4/9 | 1/9 | 2/9 |

All three methods retain nine original passes, as does cumulative; this pass
count alone therefore establishes no incremental benefit. All full-score and
directional joint gates fail. On the frozen selection, no endpoint for any
method/control has a significantly positive gain. All 27 original evaluations,
108 paired full-score comparisons and 54 directional comparisons are retained
in [results.json](results.json).

For the **primary share arm versus cumulative**, full-score gains are multiplied
by 10,000 for readability. Accuracy changes are percentage points on the same
moving cases, not each method's potentially different selected forecasts.
Positive values indicate improvement; score units are not trading-return basis
points.

| Cohort | Full Brier gain x10^4 | Full log-loss gain x10^4 | Direction accuracy change (pp) |
| --- | ---: | ---: | ---: |
| 2024 BTC | -0.0982 | -0.4048 | +0.0474 |
| 2024 ETH | -0.0259 | -0.4061 | +0.0964 |
| 2024 SOL | +0.1752 | +3.2170 | +0.0066 |
| 2025 BTC | -0.2035 | +0.0634 | +0.0887 |
| 2025 ETH | -0.3460 | -0.8843 | +0.0295 |
| 2025 SOL | -0.1736 | -0.5701 | -0.0714 |
| 2026 BTC | -0.1785 | -0.2191 | -0.0476 |
| 2026 ETH | -0.4516 | -0.3246 | -0.1034 |
| 2026 SOL | -0.3107 | +0.0999 | -0.1005 |

## Local benefit and repeated regression

**SOL 2024** is the only cohort with positive full Brier and both directional
proper-score gains for all methods against both controls. Against cumulative,
all three full log-loss gains have positive confidence bounds. For share the
gain is **+0.000321697**, interval **[+0.000018038, +0.000681519]**. Its Brier
gain is **+0.0000175175**, interval **[-0.0000085446, +0.0000414187]**; the
joint gate correctly fails. Discount and monthly have analogous localized
log-loss improvements without significant full Brier improvement.

**ETH 2025** significantly regresses in log loss for every method against both
controls, on all cases and on the frozen selection. Against cumulative:

| Method | Full log-loss gain | Paired 99.8148148% interval |
| --- | ---: | --- |
| share | -0.000088434 | [-0.000132092, -0.000039287] |
| discount | -0.000044013 | [-0.000067974, -0.000017130] |
| monthly | -0.000117982 | [-0.000187974, -0.000042078] |

Share and discount also significantly regress in both conditional proper scores
against uniform_mixed and in selected Brier against both controls in ETH 2025.
No significant directional-accuracy gain or loss is established for any arm
against either control. Nevertheless, the 2026 accuracy changes are negative
for **every method and asset against both controls**, so the point estimates
also fail to show the desired cross-period stability. Against cumulative:

| Cohort | Share accuracy change (pp) | Discount accuracy change (pp) | Monthly accuracy change (pp) |
| --- | ---: | ---: | ---: |
| 2024 BTC | +0.0474 | +0.0425 | +0.1275 |
| 2024 ETH | +0.0964 | +0.1062 | -0.0196 |
| 2024 SOL | +0.0066 | -0.0099 | +0.0281 |
| 2025 BTC | +0.0887 | +0.0361 | +0.1150 |
| 2025 ETH | +0.0295 | +0.0230 | +0.0919 |
| 2025 SOL | -0.0714 | -0.0415 | -0.0664 |
| 2026 BTC | -0.0476 | -0.0788 | -0.0279 |
| 2026 ETH | -0.1034 | -0.0016 | -0.0673 |
| 2026 SOL | -0.1005 | -0.0239 | -0.1261 |

These findings do not show that every adaptation method is ineffective, or that
no directional information exists. They show that these three preregistered
ways of changing outer memory, with the fixed six experts, do not solve the
requested stability problem. There is no evidence here to promote any arm as a
uniformly better replacement for cumulative or uniform_mixed. No post-hoc
window sweep, asset exception or winner-per-year rule was introduced.

## Verification, resources and reproduction

The [independent verifier](verification.json) reconstructs temporal weights in
probability space using the saved issued forecasts and only mature same-lane
labels, independently of the production log-weight updates. It checks all
three possible-outcome potentials, skipped feedback and monthly resets, then
recomputes every metric and interval using sampled-day count matrices. It also
reconstructs all **63 historical-frequency training windows** from source prices.

Every local potential constraint passes: maximum ratio **1 + 1.802e-12**,
maximum cohort/arm numerical slack **2.835e-12**, no solver fallback. Monthly
has **12 resets per cohort**, **108** in total, in addition to the initial lane
priors. The earlier constant full-cohort `2 ln(12)` regret guarantee is **not
claimed for these forgetting/sharing/resetting variants**. Local feasibility
is a technical property, separate from empirical full-period improvement.

Nine new tests cover mature feedback and future-prefix invariance, share's
preservation of each score's weight mass, recovery of suppressed experts,
wall-time discounting, both monthly lane resets, discarded old-month feedback,
reflection, protected inputs and exact equivalence to cumulative within one
month. Before registration, synthetic cross-month forecasts and bootstrap
comparisons passed the independent verifier. The relevant local suite passed
**319 tests**, with **8 database integration tests skipped locally** and included
in CI.

Previous frozen code, experiments, models, data, credentials and user files are
unchanged. No source or outcome from **2026-08-01 onward** was opened. Replay
of all three arms took **312.41s**, maximum RSS **0.40 GiB**; the independent
audit took **29.09s**, **0.32 GiB**. Saved outputs occupy approximately
**36.7 MiB**. Existing environments and data were reused without downloads or
installations. See [runtime.json](runtime.json).

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/verify_temporal_adaptation.py
```

Fresh replay into an unused directory:

```sh
.venv-timestral/bin/python scripts/run_temporal_adaptation.py register --root research/out/temporal-replay
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/run_temporal_adaptation.py run --root research/out/temporal-replay
.venv-timestral/bin/python scripts/verify_temporal_adaptation.py --root research/out/temporal-replay
```

The execution manifest is frozen; this report and runtime/verification receipts
were added afterward. All failed comparisons remain in the published results.
