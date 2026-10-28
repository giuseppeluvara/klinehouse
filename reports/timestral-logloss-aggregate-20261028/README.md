# Causal aggregation retains nine gates with a finite-sequence log-loss bound

The registered adaptive rule passes **9/9 original gates**, recovers the two
significant BTC log-loss regressions relative to the preceding mixed model, and
has a mathematical cumulative-loss bound independently verified on all nine
cohorts. It does **not** satisfy the registered general-superiority criterion.
Relative to that mixed model, Brier worsens in eight cohorts and ETH 2025 has a
small but statistically visible log-loss regression. No candidate dominates all
comparators on all required metrics.

The useful result is narrower and precise: without knowing which of four fixed
experts will perform best, this causal mixture incurs at most **2 ln(4) =
2.772589 cumulative log-loss units** above the best one in each asset/year
sequence. On these cohorts that is at most **0.00004542 average loss units**.
This is a relative loss guarantee, not a guarantee of accurate or profitable
forecasts. All four experts can be wrong together.

## Frozen procedure and timing

The [protocol](protocol.json), code and input hashes were saved before opening
new aggregate scores. The experts were chosen from earlier results on these
same outcomes, so the study remains sequential retrospective research.
All **550,362** original anchors are retained: January–July **2024, 2025 and
2026**, BTC/ETH/SOL, not complete calendar years. The target is the five-minute
spot return from anchor +60 to +360 seconds.

Four experts form a Cartesian product:

| Direction | Old flat probability | Half old / half historical-frequency flat |
| --- | --- | --- |
| Frozen common-hour signal | `common_old` | `common_mixed` |
| Frozen uniform price-path center | `uniform_old` | `uniform_mixed` |

The last expert is the preceding flat-repair primary. All component probabilities
are reused exactly. A static equal four-way mixture is the planned ablation.
No new source, fit, asset exception or parameter search is introduced.

Each asset/year starts two independent weight vectors at equal 1/4 priors.
Forecast index `i` uses lane `i % 2`; immediately before issuing it, that lane
multiplies its weights by each expert's probability of the observed label at
`i - 2`, then normalizes in log space. Anchors are at least 300 seconds apart;
that label ends at least 240 seconds before this forecast. The latest
`i - 1` outcome can still be unavailable and is not used. Both initial forecasts
and all coldstart costs are retained. Other-lane feedback is ignored even if
available. There is no forgetting rate, rolling window, learned learning rate
or switching rule. The same update applies to every asset/year, with separate
states; this adds adaptive flexibility rather than eliminating model complexity.

## What the bound proves

For each lane, the product of aggregate realized-label probabilities equals
the arithmetic mean of the four experts' corresponding products. Taking
negative natural logarithms gives the exact identity

```
L_lane = -log(mean_k exp(-L_lane,k)) <= L_lane,k + ln(4).
L_total <= min_k L_total,k + 2 ln(4).
```

The sum over two lanes handles delayed labels without an independence
assumption. The standard Bayesian-mixture identity is described by
[Orseau et al. (2017), section 2](https://proceedings.mlr.press/v76/orseau17a/orseau17a.pdf),
and delayed replicated aggregation by
[Korotin et al. (2018), section 4.2](https://proceedings.mlr.press/v91/korotin18a/korotin18a.pdf).
The two-lane specialization and Cartesian component calculation here are
verified directly from the saved forecasts and outcomes.

The Cartesian posterior factorizes into separate direction and flat/moving
weights. Flat outcomes cannot change the direction weights. The conditional
direction cumulative loss on moving outcomes and the flat/moving cumulative
loss each incur at most **2 ln(2) = 1.386294** above their better fixed component.
The final labels enter the after-the-fact certificate; they never enter earlier
forecasts.

This statement concerns the full sequence and the four specified fixed experts.
It does not cover the best possible model, best switching expert, best convex
mixture, each month, selected subsets, Brier, accuracy, calibration, or returns.
It does not guarantee strict improvement. The adaptive total loss is slightly
above the best fixed expert in every observed cohort, as permitted by the bound.

| Cohort | Best fixed expert in hindsight | Total regret | Direction regret | Flat/moving regret |
| --- | --- | ---: | ---: | ---: |
| 2024 BTC | uniform_old | 2.666833 | 1.280539 | 1.386294 |
| 2024 ETH | uniform_mixed | 2.759370 | 1.373090 | 1.386280 |
| 2024 SOL | common_mixed | 2.138231 | 0.754846 | 1.383385 |
| 2025 BTC | uniform_old | 2.749507 | 1.383011 | 1.366496 |
| 2025 ETH | uniform_mixed | 2.772362 | 1.386072 | 1.386290 |
| 2025 SOL | uniform_mixed | 2.745266 | 1.358972 | 1.386294 |
| 2026 BTC | uniform_old | 2.762710 | 1.376416 | 1.386294 |
| 2026 ETH | uniform_mixed | 1.192981 | 1.386210 | -0.193230 |
| 2026 SOL | uniform_mixed | 2.772503 | 1.386208 | 1.386294 |

All regrets are cumulative natural-log score units. The independent audit's
maximum absolute mixture-identity discrepancy is **2.04e-10**; all total and
component bounds pass. Negative component regret is possible: the two lanes
can benefit from different experts. Per-cohort totals are in
[evidence.json](evidence.json).

## Empirical comparisons

Original gates and thresholds remain unchanged: coverage at least 5%, selected
accuracy at least 52%, required confidence bounds and baseline comparisons,
at least 100 calls each direction and at least five positive months with enough
calls. All nine adaptive gates pass. Original passes for controls are 6/9
common_old, 8/9 common_mixed, 8/9 uniform_old, 9/9 uniform_mixed and 9/9 equal.
Therefore 9/9 alone does not distinguish this adaptive rule from the simplest
static ablation.

The following paired gains compare adaptive with **uniform_mixed on all
identical anchors**. Positive means improvement. Gains are multiplied by 10,000
for readability; they are score units, not trading-return basis points.
Accuracy and coverage describe the adaptive model's own threshold selection.

| Cohort | Selected accuracy | Coverage | Brier gain x10^4 | Log-loss gain x10^4 | Original gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 2024 BTC | 53.643% | 18.996% | -0.0930 | +6.6539 | pass |
| 2024 ETH | 53.149% | 17.989% | -0.2253 | -0.4498 | pass |
| 2024 SOL | 52.823% | 11.579% | +0.4286 | -0.0100 | pass |
| 2025 BTC | 53.383% | 21.667% | -0.0907 | +2.3208 | pass |
| 2025 ETH | 54.443% | 21.474% | -0.2293 | -0.4541 | pass |
| 2025 SOL | 52.561% | 14.484% | -0.2211 | -0.4496 | pass |
| 2026 BTC | 53.503% | 34.041% | -0.1436 | +5.0759 | pass |
| 2026 ETH | 54.135% | 34.207% | -0.2285 | -0.1954 | pass |
| 2026 SOL | 53.237% | 11.992% | -0.2494 | -0.4541 | pass |

Against uniform_mixed, log loss improves in **3/9** cohorts, all BTC, while
Brier improves in **1/9**, SOL 2024. The 99.444% all-case log-loss-gain intervals
are positive for BTC 2024 **[0.00009807, 0.00171524]** and BTC 2026
**[0.00006404, 0.00104119]**. The earlier fixed correction harmed those cohorts;
the causal weighting recovers much of that penalty. It does not beat the old
uniform model there: the bounded learning cost remains.

ETH 2025 instead has a significant regression versus uniform_mixed:
**-0.00004541**, interval **[-0.00008870, -0.00000637]**. It also worsens Brier
and log loss significantly on the frozen-selected subset. No adverse comparison
has been removed. SOL 2025 retains its original gate, with **52.561%** accuracy
on 8,843 selected forecasts and **14.484%** coverage.

| Comparator | Full Brier improves | Full log loss improves | Full paired gates | Frozen-selected paired gates |
| --- | ---: | ---: | ---: | ---: |
| common_old | 8/9 | 9/9 | 4/9 | 3/9 |
| common_mixed | 9/9 | 8/9 | 3/9 | 3/9 |
| uniform_old | 3/9 | 6/9 | 1/9 | 0/9 |
| uniform_mixed | 1/9 | 3/9 | 0/9 | 0/9 |
| static equal | 8/9 | 8/9 | 2/9 | 2/9 |

The frozen selection is precisely the preceding uniform_mixed threshold subset,
used unchanged for every paired control. Conditional up/down information gates
pass **5/9 against common**, **0/9 against uniform**. Mixing introduces no new
market information and did not resolve the missing directional evidence in
this study.
All 54 original evaluations, 90 paired comparisons and 18 conditional
comparisons are retained in [results.json](results.json).

`all_original_gates` is true; `all_paired_gates`,
`all_direction_information_gates`, and **`general_solution` are false**.
The loss bound is a successful separate property, never substituted for those
registered empirical criteria.

## Interpretation and verification

The opposing flat-probability choices explain part of the earlier instability:
old flat has lower cumulative log loss in all BTC cohorts; mixed flat has lower
cumulative log loss in the ETH/SOL cohorts. The causal rule learns those
differences from mature labels, with a
bounded cost, instead of manually choosing per-asset exceptions from the full
results. Uniform direction has lower cumulative conditional log loss in eight
cohorts; common has lower loss in SOL 2024. This diagnoses the observed
comparator conflict; it does not establish
why the underlying market process changes, nor show that these preferences
will persist. Accumulating evidence without forgetting can react slowly after
a regime reversal; the bound compares fixed experts, not switching experts.

Paired comparisons use 4,000 seven-day blocks with seed 271032 and 99.444%
intervals, correcting across nine cohorts per endpoint. Original 98.333%
intervals retain their earlier convention. Neither correction is joint across
endpoints, controls, scopes or this repeated research history. Bootstrap scores
are not obtained by refitting the adaptive learner. These are historical
comparisons, not untouched confirmation or a prospective performance estimate.

The [independent audit](verification.json) reconstructs weights using separate
binary-component log odds and strict prefix sums, rather than importing the
production learner. It verifies chronology, all hashes, outcome availability,
probabilities, the loss identity, component certificates and every score/gate.
It also reconstructs all **63 past-frequency training windows** from source
prices. Four new tests cover causal-prefix invariance, delayed feedback,
factorization, sign reflection, bounds, flat-label direction invariance and
invalid/protected inputs. The relevant local suite passed **305 tests**, with
**8 database tests skipped locally** and included in CI.

No market source or outcome from **2026-08-01 onward** was opened. All previous
frozen artifacts/code, credentials and user files are preserved. Run: **13.55s**,
maximum RSS **0.31 GiB**; independent audit: **12.10s**, **0.23 GiB**. The new
prediction bundle is approximately **28.9 MiB**; no downloads or installations
were needed. See [runtime.json](runtime.json).

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/verify_logloss_aggregate.py
```

Fresh replay into an unused directory:

```sh
.venv-timestral/bin/python scripts/run_logloss_aggregate.py register --root research/out/logloss-replay
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/run_logloss_aggregate.py run --root research/out/logloss-replay
.venv-timestral/bin/python scripts/verify_logloss_aggregate.py --root research/out/logloss-replay
```

The execution manifest is frozen. This report and runtime/audit receipts were
added afterward; the failed general criterion is retained.
