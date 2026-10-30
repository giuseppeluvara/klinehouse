# Direction failure diagnosis and an independently audited orientation replay

The live models' sub-50% accuracy is primarily an observed reversal/continuation
mismatch in this short sample, not just class imbalance or a scoring bug. The
hourly price-center baseline is explicitly contrarian. Its OI correction is too
small to change that behavior on most observations. Historical selected accuracy
and live all-moving accuracy also used different denominators, but applying the
original selection does **not** repair the live failure.

A separately registered causal orientation mixture improves the inspected live
BTC and SOL replay, but not ETH against the price control, and passes **0/9**
historical joint improvement comparisons. It is not a verified replacement for
the running pilot. Every negative result is retained in [results.json](results.json).

## What was diagnosed

The immutable prospective checkpoint was captured on **2026-10-30 at 14:57:45 UTC**
(15:57:45 Europe/Rome). It contains 963 eligible forecasts and 960 outcomes: 320
per asset, with eight flat SOL outcomes. There are three future targets and no
overdue unobserved outcomes. Sources and outcomes were collected prospectively;
this subsequent diagnosis and new-model replay are **post hoc**.

| Asset | Baseline moving accuracy | Balanced accuracy | AUC | Original 52% selection |
| --- | ---: | ---: | ---: | ---: |
| BTC | 144/320 = 45.00% | 45.47% | 0.4486 | 45/108 = 41.67% |
| ETH | 148/320 = 46.25% | 46.30% | 0.4766 | 51/106 = 48.11% |
| SOL | 141/312 = 45.19% | 45.85% | 0.4622 | 15/36 = 41.67% |

The original selection is exactly `max(p_down,p_up) >= .52`, using full three-class
probabilities. It selected 33.75%, 33.13% and 11.25% of these outcomes. No realized
flat is selected here. The historical 52--54% figures referred to selected cases,
including flat outcomes as errors; historical all-moving baseline accuracy was
only approximately 50.8--51.9%. These measures must not be conflated.

The model predicts against the lagged hourly return on approximately 81--83% of
cases. The target instead continues its sign on 57.19% of BTC, 51.56% of ETH and
55.13% of SOL moving cases. The exact decomposition

`accuracy - .5 = 2*(actual_up_share-.5)*(predicted_up_share-.5) + 2*Cov(actual_up,predicted_up)`

attributes only -0.494, -0.049 and -0.678 percentage points to class/prediction
imbalance, against total deficits of -5.000, -3.750 and -4.808 points. The rest
comes from the negative empirical association of decisions and outcomes.
Six-hour UTC strata suggest an important between-period component: stratified
AUC is 0.4906/0.5051/0.5164. This does not establish a permanent regime change.

Baseline mean conditional confidence is only about 51.8%, so extreme probability
confidence is not the principal problem. Changing the flat prior cancels out of
`q = p_up/(p_down+p_up)`; multiplying logits by a positive calibration factor
preserves every direction and the ranking. Neither creates missing discrimination.

OI changes only 7/5/10 baseline decisions, for net success changes +1/+5/+2. The
frozen coefficients and feature clipping imply a maximum conditional-probability
change below 0.375 percentage points. Both corrections leave the decisions on
the baseline's selected subset exactly unchanged in this checkpoint. These are
small corrections to a contrarian model, not independent up/down classifiers.

The model's fit cutoff is **2026-07-01**, using preceding observations, not a fit
through the end of July. Its parameters and scales are 121 days old on October30.
That is a deliberate pilot constraint and a possible source of drift, not proof
that refitting would solve the observed ranking failure. Source/target alignment
checks did not reveal a sign or one-minute target mismatch. Historical/live OI
publication and semantic equivalence remain uncertified.

The three assets are highly dependent: realized directions agree on about
84--86% of anchors, while baseline directions agree across all three on 91.25%.
The 952 moving asset/outcome pairs are not 952 independent trials. No statistical
superiority or tradable advantage is inferred from this short sample.

## Separate orientation experiment

The [protocol](protocol.json) and exact [source copies](frozen-source/) were frozen
before this version's scores. The experts preserve the issued flat probability
and use conditional direction `q`, `1-q`, and `0.5`. Each asset/cohort starts two
equal-prior index-parity learners. A lane updates only on its own earlier moving
outcomes whose receipt precedes issuance; late pending feedback is retained and
consumed once. Flat outcomes do not update direction weights. There is no
forgetting rate, rolling window, fitted intercept or hyperparameter search.

The probability is computed as
`q_new = .5 + (weight_reversal - weight_continuation)*(q-.5)`.
Equal orientation weights are exactly neutral. Neutral tie-up accuracy is a
counting convention, not directional information. The method changes the
orientation/intensity of an existing signal; it adds no new market information.

Historical evaluation retains all **550,362** original January--July forecast
anchors across 2024, 2025 and 2026. The final 2024/25 July31 target ends on August1
at00:01, as in the original saved experiment. Every target ends before the
protected **2026-08-01** boundary. Historical receipt availability at target end
is conditional, not historically publication-certified. The live replay instead
uses actual recorded issue and outcome-receipt times. It does not insert or
replace any forecast in the live journal.

| Live checkpoint asset | Adaptive moving accuracy | Three-class Brier | Natural log-loss |
| --- | ---: | ---: | ---: |
| BTC | 171/320 = 53.44% | 0.499438305 | 0.694121778 |
| ETH | 149/320 = 46.56% | 0.500292437 | 0.696473205 |
| SOL | 162/312 = 51.92% | 0.523960923 | 0.794150424 |

This is a retrospectively motivated replay, **not prospective evidence** for the
adaptive method. ETH still trails price control in accuracy and neutral in all
three point measures. Static mirroring is diagnostic, never a hindsight-selected
replacement. All unchanged controls, individual selections, fixed-baseline
selections, flat counts and pending outcomes remain in the saved results.

On historical data, Brier and log-loss are slightly worse than the baseline in
**all nine cohorts**. Direction improves in four and worsens in five. The joint
criterion against baseline and price control passes zero cohorts. Approximate
full Brier regressions are 0.0000352--0.0000359 per observation; natural log-loss
regressions are 0.0000358--0.0000360. Learning the orientation has a cost when the
original contrarian expert already performs best over a long sequence. No
mathematical loss bound is substituted for the failed empirical criterion.

## Verification, preservation and reproduction

The [independent verifier](verification.json) imports neither the learner, runner
nor production scoring. It reconstructs same-lane posteriors from receipt-sorted
likelihood-ratio prefix sums, independently checks source membership, the live
hash chain, post-commit confirmations and price/label provenance, and recomputes
all metrics and selection flags. Maximum weight discrepancy is 4.071e-10;
maximum adaptive-probability discrepancy is 2.087e-11.

The new diagnostic CLI independently reproduces confusion matrices, AUC,
imbalance/association decomposition and selection on the preserved checkpoint:

```sh
.venv-timestral/bin/python scripts/analyze_shadow_direction.py \
  --checkpoint research/out/direction-diagnosis-20261030/live-checkpoint \
  --output research/out/direction-diagnosis-20261030/live-diagnosis-NEW.json

.venv-timestral/bin/python scripts/verify_direction_orientation.py \
  --root reports/direction-orientation-20261030-v3 \
  --output research/out/direction-diagnosis-20261030/orientation-audit-NEW.json
```

Verifier output refuses overwriting; supply a fresh output filename when one is
already present. For a fresh complete replay:

```sh
.venv-timestral/bin/python scripts/run_direction_orientation.py register \
  --root research/out/orientation-fresh
.venv-timestral/bin/python scripts/run_direction_orientation.py run \
  --root research/out/orientation-fresh
```

The relevant local suite passed **135 tests**, including the 29 new diagnostic,
orientation and independent-verifier tests and existing shadow tests. No live
runtime/model/protocol/service, credentials, user file or protected holdout was
changed. The pilot continues unchanged. Existing environments and saved arrays
were reused; no model or dataset was downloaded.

An initial registration stopped before scoring on an overly restrictive year-local
target guard. A second execution exposed machine-epsilon direction on theoretically
neutral mixtures. Both registrations/results/source evidence are retained under
`research/out/direction-diagnosis-20261030/retained-*`; this final version fixes the
guard and exact-neutral arithmetic without retuning the empirical hypothesis.

## Remaining research

A functioning general replacement has **not** been established. An information
source absent from these models is bid/ask depth and queue imbalance. It is not
equivalent to OI changes or candle taker volume. Existing evidence for equities
concerns contemporaneous order-flow impact or the next price tick, not a verified
five-minute crypto predictor: [Cont et al.](https://arxiv.org/abs/1011.6402),
[Gould and Bonart](https://arxiv.org/abs/1512.03492).
Public-source feasibility is being checked separately. Any next experiment must
register features, timing, controls and criteria before its new outcomes, preserve
actual receipts, and leave the ongoing seven-day pilot unchanged.
