# One-coefficient relative reversal: rejected

This registered simplification was executed on January–July 2024, 2025 and 2026,
for BTCUSDT, ETHUSDT and SOLUSDT. The primary relative-reversal model passes
**0/9** conditional gates against each control and **0/9** original direction
gates. It does not provide a general up/down solution. All negative results are
retained in [results.json](results.json).

## Fixed experiment

The [protocol](protocol.json) froze code, inputs and criteria before this run.
The hypothesis was informed by earlier exploration of these outcomes; this is
not an untouched confirmation sample. Sources are previously frozen arrays only.
All **550,362** original anchors and the five-minute spot target, from anchor+1min
to anchor+6min, are unchanged. No data or environment downloads were needed.

Three models each have **one scalar coefficient**, shared across assets:

- `reversal`: the negative own one-minute return, normalized by causal past
  60-minute RMS and clipped to ±3; penalty 0.25.
- `relative_unshrunk`: `(mean_other_return1 − own_return1) / sqrt(2)`, clipped
  to ±3; penalty zero.
- `relative`, primary: the same relative signal, penalty 0.25.

For signal `z`, `P(up | non-flat) = sigmoid(beta × z)`. Each month fits beta in
[0,8] on the preceding 90 days of matured non-flat labels, with equal total
weight per UTC day across the three assets. The objective is weighted log loss
plus `penalty × beta² / 2`. There is no intercept, asset ID, fitted normalization,
regime, interaction or parameter search. An exact one-dimensional convex
derivative root or boundary solves the fit. A coefficient of zero returns 50/50;
the orientation cannot be flipped to fit a different period. The upper bound is
a numerical guard, not a calibrated confidence limit.

Training membership matches the frozen flow control, including its futures
availability requirement. New predictions use spot inputs only. Existing flat
probabilities remain identical in original all-anchor evaluations. The 50/50
control and historical frequency are separate controls.

## Results and interpretation

| Cohort | Primary conditional accuracy | Brier gain vs 50/50 ×10⁴ | Brier gain vs flow ×10⁴ |
| --- | ---: | ---: | ---: |
| 2024 BTC | 50.316% | +0.008 | -4.410 |
| 2024 ETH | 50.269% | +0.009 | -1.678 |
| 2024 SOL | 50.064% | +0.016 | -1.083 |
| 2025 BTC | 49.658% | -0.020 | -0.516 |
| 2025 ETH | 49.979% | -0.001 | -5.550 |
| 2025 SOL | 50.000% | +0.004 | -1.507 |
| 2026 BTC | 49.975% | -0.001 | -4.549 |
| 2026 ETH | 50.564% | +0.050 | -9.189 |
| 2026 SOL | 50.174% | +0.035 | -6.710 |

Every accuracy interval includes 50%; every Brier-gain interval against 50/50
includes zero. Values ×10⁴ are score units, **not trading basis points**.
The primary model has zero original-threshold selection coverage in all cohorts.
Five of seven 2025 monthly relative fits collapse to beta zero. Those ties are
counted explicitly; the scoring convention predicts up at exactly 50%, which is
not directional information.

Mean confidence is only 50.004–50.034%. Shrinkage improves Brier relative to the
unshrunk version in six of nine cohorts at point estimate, but no corresponding
information gate passes. Both other arms also pass 0/9 original gates. Merely
moving probabilities toward 50% does not create discrimination. Removing the
common one-minute move has not retained a useful standalone signal here.

This negative result motivated a separately registered
[common-hour reversal experiment](../timestral-market-reversal-20261027/README.md).
It did not trigger changes to this protocol, model or results.

## Verification and limits

The [independent auditor](verification.json) reconstructs features using the
existing independent source auditor, checks all 63 fit memberships and scalar
optimality conditions without fitting, reconstructs probabilities and all scores,
and verifies hashes and chronology. Maximum KKT residual: **2.274×10⁻¹⁴**.
Synthetic tests check symmetry, cancellation of common moves, equal day weights,
shrinkage, unsupported-orientation fallback, maturity and protected boundaries,
and recovery of a planted relative signal. These tests establish implementation
behavior, not market efficacy.

The original gates are unchanged. Conditional tests use 4,000 paired seven-day
blocks with 99.444% intervals, correcting nine cohorts separately per endpoint.
They do not jointly correct the multiple endpoints, controls or sequential
research history. Conditional evaluation excludes realized flat outcomes and
cannot establish tradable coverage. These are retrospective, previously explored
periods, not prospective evidence. Source publication vintage remains unknown.
No market source/outcome from **2026-08-01 onward** was opened; old artifacts and
user files are preserved. No orders, costs or profitability claim are involved.

Execution: **7.74s**, peak RSS **0.83GiB**. Audit: **6.17s**, peak RSS **0.70GiB**.
Bundle approximately **10.2MiB** before report/receipts/Git objects, reusing sources.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/verify_simple_direction.py
```

Re-execution requires a fresh output directory:

```sh
.venv-timestral/bin/python scripts/run_simple_direction.py register --root research/out/simple-replay
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-timestral/bin/python scripts/run_simple_direction.py run --root research/out/simple-replay
.venv-timestral/bin/python scripts/verify_simple_direction.py --root research/out/simple-replay
```

The manifest freezes execution artifacts; this report and verification/runtime
receipts were added afterward.
