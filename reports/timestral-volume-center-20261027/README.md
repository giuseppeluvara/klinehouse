# Price-path centers improve direction; volume weighting adds no benefit

The registered uniform and volume-weighted hourly price-center rules both pass
**8/9 original direction gates**, versus 6/9 for the frozen endpoint-hour signal.
BTC 2025 and SOL 2026 now pass. SOL 2025 remains a failure because combined
three-class log loss is slightly worse than historical frequency, despite better
directional loss and selected accuracy above 52%.

The useful new representation is the **intra-hour price path**. Volume weighting
worsens conditional Brier against the uniform center in **all nine cohorts**.
The primary volume-information hypothesis fails, and the registered
**general-solution flag remains false**. The uniform ablation is not promoted to
a validated general model after seeing its better scores.

## Hypothesis and fixed comparison

The prior agreement study found that disagreement barely affected selected cases
in the failing groups. The new question is whether the distance from the price
center during the hour adds information when asset and market already agree.
Identical endpoint returns can hide different paths and volume concentration.
This is a testable representation hypothesis, not a claim about market causality.

The [protocol](protocol.json), code and input hashes were frozen before new
candidate scores. For each asset, using the last 60 closed spot minute candles:

* **Uniform center:** mean log close.
* **Volume center:** sum of base volume times log close, divided by total base
  volume; fall back to the uniform center if total volume is zero.

These are geometric centers of candle closes, **not execution VWAP**, trade ticks
or order-book information. Each displacement is
`−2 × (current_log_close − center) / (prior_60_minute_RMS × sqrt(60))`, clipped
to ±8. The three displacements are averaged and clipped to ±3. The fixed factor
two roughly aligns the uniform center with the endpoint displacement for a
linear price path; it was not optimized against outcomes.

Where the frozen own-hour and common-hour signals agree in sign, use the new
market-center displacement. Otherwise retain the old common-hour signal. Each
representation gets exactly **one shared nonnegative logistic coefficient**, fit
monthly on the same prior 90 days of matured nonflat labels, pooled across the
three assets with equal total UTC-day weights. The existing training-availability
mask, scalar optimizer and six-lag Bartlett uncertainty shrinkage are unchanged.
The coefficient applies globally, so reestimation can also change confidence on
disagreement cases. There are 42 scalar fits and 42 adjustments, no fitted mixture
weights, intercepts, asset IDs, regime coefficients or parameter sweep.

The uniform arm separates path information from volume weighting. Other controls
are the frozen adaptive common-hour model, neutral 50/50, futures-flow model and
historical frequency. All **550,362** original anchors are retained for
January–July **2024, 2025 and 2026**, with the same five-minute target from anchor
+60 seconds to +360 seconds and unchanged flat probabilities. There are
**484,663** agreement anchors, selected solely from information available at the
anchor. Both full-sample and agreement-subset paired scores are recorded.

The primary general criterion requires all nine volume original gates plus all
nine conditional improvement gates against **both common and uniform**, on
**both the full sample and agreement subset**. Other comparisons are diagnostic.
No criterion was weakened after results. Low dimension limits model flexibility
but does not prove overfitting cannot increase after sequential hypothesis reuse.

## Results on every cohort

Brier gains below are control loss minus candidate loss, multiplied by 10,000;
these are score units, not trading-return basis points. Conditional scores use
all nonflat outcomes. Selected accuracy includes realized flats as errors and
uses each candidate's fixed original 52% unconditional threshold; the selected
subsets differ and their accuracies alone are not a paired comparison.

| Cohort | Uniform gain vs common ×10⁴ | Volume gain vs common ×10⁴ | Volume gain vs uniform ×10⁴ | Uniform selected accuracy | Volume selected accuracy | Original gates, both |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2024 BTC | +0.8456 | +0.7397 | -0.1058 | 53.580% | 53.920% | pass |
| 2024 ETH | +0.8210 | +0.4335 | -0.3875 | 53.404% | 53.012% | pass |
| 2024 SOL | -0.1721 | -0.6681 | -0.4960 | 52.773% | 52.336% | pass |
| 2025 BTC | +1.1310 | +0.9709 | -0.1601 | 53.335% | 53.202% | pass |
| 2025 ETH | +1.6293 | +1.3817 | -0.2476 | 54.746% | 54.655% | pass |
| 2025 SOL | +1.0836 | +0.9917 | -0.0919 | 52.611% | 52.272% | fail |
| 2026 BTC | +1.4204 | +0.9878 | -0.4326 | 53.411% | 53.156% | pass |
| 2026 ETH | +2.9361 | +2.6261 | -0.3100 | 54.188% | 53.827% | pass |
| 2026 SOL | +2.1728 | +1.9801 | -0.1927 | 52.511% | 52.482% | pass |

Both representations improve conditional Brier over common at point estimate in
8/9 cohorts, with SOL 2024 worse. The paired information gate passes **5/9** for
uniform versus common (ETH/SOL 2025 and all assets 2026) and **4/9** for volume
versus common (ETH/SOL 2025 and ETH/SOL 2026). Volume passes **8/9** against neutral,
**0/9** against futures flow and **0/9** against uniform.

On the agreement subset, volume beats common under the full information gate in
the same four cohorts and beats uniform in **0/9**. Agreement-subset Brier gains
against common are +0.8447, +0.5005, −0.7942, +1.1058, +1.6057, +1.1390, +1.1106,
+2.9175 and +2.2580 (×10⁴), in table order. Thus the full-sample improvement is
also present in the intended subgroup, but is neither universal nor attributable
to volume weighting. All 63 comparisons, 18 original evaluations, confidence
intervals, monthly scores and calibration tables are in [results.json](results.json).

The uniform/common paired Brier intervals for 2024 BTC, ETH and SOL and 2025 BTC
still include zero. In particular, the SOL 2024 point deterioration has interval
[-1.0956, +0.7528] ×10⁴. A better gate count does not establish improvement over
the previous model in every year/asset, nor prospective performance.

## What still blocks SOL 2025

For volume, selected accuracy is 52.272%, with original 98.333% interval
[51.237%, 53.454%]. Its selected Brier-improvement interval against historical
frequency is positive, [0.1422%, 0.6889%]. However, combined log loss is
**0.75764868**, compared with baseline **0.75751377**. The unchanged criterion
therefore correctly rejects it. Uniform also fails combined log loss:
**0.75481524**, versus baseline **0.75447110**.

After observing that failure, a separate **post-hoc diagnostic** decomposed loss
for all nine cohorts and both arms; it did not alter predictions or gates.
Three-class log loss is the flat/moving binary loss plus, on moving cases, the
conditional up/down loss. Components below are gains against historical frequency,
averaged over the candidate's selected cases; positive is better:

| SOL 2025 arm | Direction component gain | Flat/moving component gain | Total gain |
| --- | ---: | ---: | ---: |
| Uniform | +0.00236655 | -0.00271070 | -0.00034415 |
| Volume | +0.00214979 | -0.00228470 | -0.00013491 |

The frozen flat/moving probability component offsets the improved conditional
direction component. The subsets contain 133 flats among 10,015 uniform-selected
cases and 144 among 10,191 volume-selected cases. This accounting does not prove
that a new flat predictor will generalize, and removing realized flats cannot
serve as a forecast rule. A later flat-calibration study would require its own
registered test. The diagnostic is reproducible with
[diagnose_volume_center.py](../../scripts/diagnose_volume_center.py) and saved in
[loss-decomposition.json](loss-decomposition.json); both component sums are checked
against the independently audited original losses.

## Verification and limits

The [independent audit](verification.json) uses convolution to reconstruct
centers, while production uses bounded batches of direct local-window sums.
Before registration, source-only comparisons exposed cumulative-sum cancellation
(maximum signal discrepancy about 1.2×10⁻⁷). Direct sums fixed it before any new
candidate scores were opened; all three periods then agreed within 9.2×10⁻¹³.
A regression test protects small current volumes from large distant past volumes.

The audit independently checks all 42 training memberships and raw-fit gradients,
uncertainty covariance, saved probabilities, agreement masks, scores and gates.
Maximum raw-fit residual is **7.258×10⁻¹⁵**. It does not refit models. Five new
tests cover volume sensitivity with unchanged price paths, volume-unit invariance,
future perturbations, sign reflection, zero-volume fallback, protected-source
rejection and agreement routing. The relevant local suite passed **298 tests**;
eight database integration tests are reserved for CI.

Conditional intervals retain 4,000 paired seven-day bootstrap replicates and
nine-cohort correction per endpoint; original gate intervals remain 98.333%.
There is no joint correction across comparisons, subsets, endpoints or the
sequence of hypotheses tested on these years. These are repeatedly explored
outcomes, not pristine or prospective validation. Unknown publication vintage,
realized-flat conditioning and absent execution costs rule out a profitability
claim. No model deployment or orders were performed.

No market source or outcome from **2026-08-01 onward** was opened. Older frozen
experiments and user files are unchanged. Existing sources and environments were
reused, without downloads or installations. Execution took **9.59s**, peak RSS
**0.81GiB**; independent audit **9.60s**, peak RSS **0.70GiB**. Measurements are in
[runtime.json](runtime.json).

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/verify_volume_center.py
.venv-timestral/bin/python scripts/diagnose_volume_center.py --verify
```

Fresh replay, retaining the published experiment:

```sh
.venv-timestral/bin/python scripts/run_volume_center.py register --root research/out/volume-center-replay
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/run_volume_center.py run --root research/out/volume-center-replay
.venv-timestral/bin/python scripts/verify_volume_center.py --root research/out/volume-center-replay
```

The original execution manifest is frozen. Documentation, runtime/audit receipts
and the explicitly post-hoc loss decomposition were added after execution.
