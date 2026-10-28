# Simultaneous protection of log loss and Brier: a bounded path, not universal dominance

The new defensive forecast has a verified cumulative-regret bound for **both
full three-class log loss and Brier** against six previous forecasters, and
retains **9/9 original historical gates**. It does **not** establish empirical
universal superiority. Compared with the preceding log-loss-only adaptive model,
it improves both full scores only in **SOL 2024**, worsens both point estimates
in the other eight cohorts, and significantly worsens both scores in ETH 2025.
The registered **`general_solution` remains false**.

This experiment therefore does not justify replacing the previous model on the
claim of uniformly better forecasts. Its successful new property is simultaneous
worst-case protection, not a demonstrated general performance advantage.

## What can and cannot be universal

Strict superiority for every possible future outcome is impossible even for
one logarithmic-score comparison. To be no worse than a forecast `q` for every
outcome, a new probability vector `p` must satisfy `p[y] >= q[y]` for every `y`.
Both vectors sum to one, so they must be identical: no strict improvement is
possible. This simple argument concerns unrestricted outcome-wise dominance;
it does not rule out useful expected improvements under assumptions, or passing
a specified finite empirical benchmark.

A feasible universal statement is relative: for any admitted sequence, keep
the excess cumulative loss against a declared expert pool bounded. Here the
same predictions satisfy that requirement simultaneously for two proper scores.
The bound allows a finite learning cost, and says nothing about absolute quality
when all experts share a blind spot. Strict empirical improvement, stability of
the directional signal and independent future validation remain separate goals.

## Registered experiment

The [protocol](protocol.json) and all code/input hashes were frozen before
opening the new scores. No parameter search or post-result candidate changes
were performed. All **550,362** anchors are retained for January–July **2024,
2025 and 2026**, BTC/ETH/SOL. These are seven-month cohorts, not full years.
The target is unchanged: the five-minute return from anchor +60 to +360 seconds.
These outcomes have already been explored in earlier studies; this is sequential
retrospective research, not untouched confirmation.

Six experts supply frozen three-class forecasts: `common_old`, `common_mixed`,
`uniform_old`, `uniform_mixed`, their static equal mixture `equal`, and the
preceding `logloss_adaptive`. There is no new market information or model fit.
Each asset/year starts two independent lanes at equal **1/12** evaluator priors:
six experts times two scoring rules. Lane `i % 2` updates only with the mature
label from `i - 2`; all initial forecasts are scored. There is no forgetting,
lookback choice, asset exception or learning-rate sweep.

For each expert and score, the log weight increases by **learner loss minus
expert loss**, with learning rate one. The learner term cannot be dropped across
different scoring rules. Given normalized weights, the next probability vector
is found by ensuring the summed exponential regret potential is at most one
for **each of the three possible outcomes**, before observing the actual one.
A damped Newton solver uses a deterministic convex fallback if needed. It must
certify the declared numerical tolerance or abort; solver success alone is not
accepted as evidence. No fallback was needed in the historical run.

## Two simultaneous certificates

The multi-loss construction uses the expert/evaluator framework in
[Chernov et al., section 5](https://arxiv.org/pdf/1003.2218).
The full multiclass Brier convention and learning-rate-one result are described
by [Vovk and Zhdanov](https://jmlr.org/papers/volume10/vovk09a/vovk09a.pdf).
Our two-lane, six-expert implementation and numerical certificates are evaluated
separately from those general results.

For either score `m`, every full-cohort prefix satisfies

```
L_m(learner) <= min_k L_m(expert_k) + 2 ln(12) + numerical_slack
2 ln(12) = 4.9698133.
```

The comparator may be a different expert for each score. Log loss uses natural
logarithms; Brier is the sum of the three squared class errors. The allowance
is cumulative score units, not a percentage, accuracy or trading return.
For the approximately 61,000 forecasts per cohort, the base allowance per
forecast is at most **0.00008140** for either score. The expanded protection
costs a looser log-loss bound than the previous four-expert `2 ln(4)` guarantee.
It has no direction/flat factorization certificate from the earlier method.

The saved per-step slack is the positive part of the logarithm of the largest
outcome potential ratio. Its maximum cohort sum is **5.73e-12**. The independent
audit checks all three outcomes on all **550,362** forecasts and both score
bounds at every prefix. Its maximum potential ratio is **1 + 1.864e-12**,
within the registered `1e-10` tolerance. Final regrets are:

| Cohort | Cumulative log-loss regret | Cumulative Brier regret |
| --- | ---: | ---: |
| 2024 BTC | 3.166519 | 1.378186 |
| 2024 ETH | 3.292230 | 1.455259 |
| 2024 SOL | 3.867201 | 0.638348 |
| 2025 BTC | 2.975372 | 1.704853 |
| 2025 ETH | 3.369414 | 1.583099 |
| 2025 SOL | 3.341225 | 1.510006 |
| 2026 BTC | 3.227398 | 1.424634 |
| 2026 ETH | 1.541292 | 1.698287 |
| 2026 SOL | 3.095368 | 1.851638 |

All certificates pass; the maximum prefix excess over its allowance is
**-1.08665**. No observed regret is concealed by averaging across cohorts.
Final labels are used to certify realized performance after the forecasts,
never to construct earlier forecasts. These bounds do not cover switching
experts, selected subsets, separate months, the best possible model or arbitrary
other performance measures.

## Empirical results and regressions

All original gates, thresholds, baselines and coverage rules are unchanged.
The table compares defensive with **logloss_adaptive on identical full samples**.
Positive gains are better; `x10^4` is a display scale for score units. Accuracy
and coverage refer to defensive's own original threshold selection.

| Cohort | Selected accuracy | Coverage | Full Brier gain x10^4 | Full log-loss gain x10^4 | Original gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 2024 BTC | 53.674% | 18.766% | -0.0132 | -0.0815 | pass |
| 2024 ETH | 53.237% | 17.775% | -0.0119 | -0.0869 | pass |
| 2024 SOL | 52.606% | 11.041% | +0.2701 | +3.0419 | pass |
| 2025 BTC | 53.339% | 21.310% | -0.0311 | -0.0370 | pass |
| 2025 ETH | 54.381% | 21.384% | -0.0300 | -0.0978 | pass |
| 2025 SOL | 52.686% | 14.210% | -0.0263 | -0.0976 | pass |
| 2026 BTC | 53.578% | 33.716% | -0.0014 | -0.0761 | pass |
| 2026 ETH | 54.167% | 33.902% | -0.0497 | -0.0570 | pass |
| 2026 SOL | 53.004% | 11.776% | -0.0539 | -0.0529 | pass |

Against logloss_adaptive, SOL 2024 improves full Brier by **0.00002701**
(interval **[0.00000760, 0.00004913]**) and log loss by **0.00030419**
(**[0.00003990, 0.00067954]**). Its full paired improvement gate passes.
ETH 2025 significantly worsens Brier by **0.00000300** and log loss by
**0.00000978**; gain intervals are **[-0.00000610, -0.00000053]** and
**[-0.00001682, -0.00000419]**. Its regressions also occur on the frozen
selection. All other full-score point estimates against this comparator worsen.

| Comparator | Full Brier improves | Full log loss improves | Full paired gates | Frozen-selected paired gates |
| --- | ---: | ---: | ---: | ---: |
| common_old | 9/9 | 9/9 | 4/9 | 3/9 |
| common_mixed | 9/9 | 9/9 | 4/9 | 3/9 |
| uniform_old | 3/9 | 6/9 | 1/9 | 0/9 |
| uniform_mixed | 1/9 | 4/9 | 0/9 | 0/9 |
| equal | 8/9 | 8/9 | 2/9 | 2/9 |
| logloss_adaptive | 1/9 | 1/9 | 1/9 | 0/9 |

Frozen-selected comparisons retain the exact previous uniform_mixed selection
for every control. Conditional direction information gates are still **5/9
against common**, **0/9 against uniform**. All 63 original evaluations, 108
paired comparisons and 18 conditional comparisons are in
[results.json](results.json). `all_certificates` and `all_original_gates` are
true; the other empirical success flags remain false.

## What the remaining gap means

The six-expert pool contains the same hindsight-best expert for both full
scores in every cohort: uniform_old for BTC, uniform_mixed for ETH and SOL,
except SOL 2024 where equal is best. This is a descriptive result using the
whole evaluation period, not a valid way to select a forecast beforehand.
The residual gap to that comparator is therefore a cost of learning which
existing forecast to follow in these data; it is not evidence that the two
cohort-optimal comparators inherently conflict.

The SOL 2024 gain cannot be attributed solely to adding the second objective:
the pool also grew from four to six experts, admitting equal, which was absent
from the old adaptive pool. No matched-pool ablation was registered. Neither
changing the objective nor expanding the pool has been isolated as the cause.

A separate, explicitly post-hoc diagnosis of when the learning cost occurs is
saved in [learning-diagnosis.json](learning-diagnosis.json). It is descriptive
only and does not select another candidate or alter the registered result.
The first seven days account for **16.2% of summed log-loss excess** and **14.1%
of Brier excess** against the cohort-specific hindsight comparators. These are
signed accounting proportions, not causal attribution. January contains a
majority of the cohort's excess in only **4/9** cohorts for either score.
February–July excess remains positive in **all nine** cohorts for both scores.
For SOL 2024, log-loss excess is **-9.6146** in January and **+13.4818** afterward.
The excess is not confined to initialization. This decomposition cannot
determine whether initialization choices cause later losses or whether a warm
start, forgetting rate or restart would help.

The evidence supports a narrower research route: retain the simultaneous-score
constraints, then separately test adaptation to changes in relative expert
performance, with a rule fixed before its scores are opened. Any expanded expert
pool also needs a matched-pool comparison to distinguish new information from a
new weighting method. No such further candidate was evaluated or selected here.
Repeated improvements on these explored cohorts cannot establish future
superiority; that would require separately authorized independent validation.
The protected holdouts remain closed. Even independent success would support a
specified performance claim, not dominance under every possible future sequence.

## Verification and reproducibility

The [independent audit](verification.json) does not import the production
optimizer or use its A/B potential reduction. It reconstructs strict same-lane
prefix regrets, evaluates all twelve exponential evaluator terms for each
possible outcome, checks numerical slack and both loss bounds, reconstructs
all 63 historical-frequency windows, and recomputes every score/gate.

Five tests cover causal-prefix invariance, late feedback, outcome reflection,
identical experts, simultaneous regret, extreme probabilities, numerical fallback
and invalid/protected inputs. A separate 61,000-step synthetic comparison passed
the independent certificate audit before registration. The relevant local
suite passed **310 tests**, with **8 database tests skipped locally** and
included in CI.

Paired uncertainty uses the frozen 4,000 seven-day blocks, seed 271032, with
99.444% nine-cohort correction separately per endpoint. Original intervals
retain their prior 98.333% convention. There is no joint correction across
endpoints, controls, scopes or sequential studies, and bootstrap scores do not
refit the adaptive learner. No prospective performance or profitability is
claimed.

No market source or outcome from **2026-08-01 onward** was opened. Previous
frozen code, models, protocols, data, credentials and user files are preserved.
Replay took **94.32s**, maximum RSS **0.34 GiB**; independent audit **14.75s**,
**0.34 GiB**. The new saved bundle is approximately **13.2 MiB**. No downloads
or installations were required. See [runtime.json](runtime.json).

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/verify_multiscore.py
  .venv-timestral/bin/python scripts/diagnose_multiscore.py --verify
```

Fresh replay into an unused directory:

```sh
.venv-timestral/bin/python scripts/run_multiscore.py register --root research/out/multiscore-replay
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/run_multiscore.py run --root research/out/multiscore-replay
.venv-timestral/bin/python scripts/verify_multiscore.py --root research/out/multiscore-replay
```

The execution manifest remains frozen. This report, runtime/audit receipts and
the explicitly post-hoc learning diagnosis were added after the registered run.
