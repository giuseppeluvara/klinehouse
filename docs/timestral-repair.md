# Diagnosis and repair of forecast selection and intervals

This separate replay fixes two operational weaknesses of the earlier experiment:
selection could only choose a nonzero model, even when no model had evidence of
beating unchanged price, and December residual intervals stayed fixed when market
volatility changed. The earlier frozen code, source data, reports and holdouts
remain unchanged. Lack of fine-tuning is not itself an implementation error, and
adding a larger model would not establish a predictive signal.

Executed results, saved forecasts and independent replay checks are in the
[diagnosis report](../reports/timestral-diagnosis-20261026/README.md).

## Point forecast hypothesis

Both point models see exactly the same 256 complete hourly closes before the
anchor. The original model forecasts log-price levels. The new model receives
the rolling horizon log returns, in basis points, calculated within that same
window. For horizon h, there are 256-h observed returns. The official API pads
these inputs with missing indicators; no missing market data is imputed.

It forecasts **h steps**, using the final predicted horizon return directly.
At that endpoint the return is log(P[t+h-1]/P[t-1]), matching the original target.
Forecasting one step would be the wrong target, and summing one-hour marginal
quantiles would not produce quantiles of the horizon return. Tests cover both
the horizon alignment and the basis-point conversion.

The pinned Timestral-5M code and weights are reused on CPU with two threads.
There is one new point hypothesis, no fine-tuning, new download or parameter
sweep. All candidate scores are reported, including losses to the baseline.

## Baseline-aware selection

At each month's first forecast, selection uses only the previous three complete
calendar months of matured, nonoverlapping outcomes. It stays fixed for the
month. Minimum history is 60 daily or 360 four-hour observations, with at least
20 days represented in each month. Both Timestral candidates must satisfy:

- Strictly lower mean absolute error than unchanged price.
- RMSE no worse, and MAE better in at least two of three months.
- Negative upper endpoint of a paired weekly block-bootstrap interval.

The bootstrap uses 2,000 draws, seed 17 and 98.75% two-sided intervals to adjust
the two candidates by two horizons at each decision. It does not control all
repeated monthly decisions or establish independence across weeks. Among passing
candidates the lower MAE wins, with fixed order for ties. If none passes, the
selected point is unchanged price and the status explicitly says no evidence.
This is an abstention rule, not a successful learned forecast or trading edge.

## Intervals that respond to past volatility

At each anchor, a 90-day window includes only already matured residuals. Each
absolute residual is divided by the volatility scale available at its original
forecast origin. The scale is the norm of the last 24 intrahour realized
volatilities, multiplied by sqrt(h/24), with a numerical floor of 1e-8.

The current interval radius is the order statistic at rank ceil((n+1)*0.90)
of normalized absolute residuals, multiplied by the current past-only scale.
The point forecast is unchanged by interval calibration. Insufficient history
returns unavailable intervals explicitly. No exchangeability or coverage
guarantee is asserted for this time series or for selected models.

The replay includes static and dynamic intervals around the **same zero-return
point**, isolating interval calibration from point accuracy. Coverage, width and
pinball loss must be read together; wider intervals alone do not prove quality.

## Evidence status and execution

January–March and May–July 2026 results had already been observed. April's source
prices had already been read as model context. This replay uses October–December
2025 as initial history and evaluates January–July 2026. It is diagnostic and
causal in its computation, but **not a new independent holdout**. Prospective
validation remains pending. The protected August 2026–February 2027 experiment
is not opened. No claim of successful future validation is made.

The small hourly cache is derived from the three existing verified archives,
with source and feature checksums in `cache-provenance.json`. The saved protocol
binds that cache, the plan and execution code before replay. Outputs and opening
receipts cannot be overwritten. New runs require a separate directory.

```bash
.venv-timestral/bin/python scripts/prepare_timestral_repair_cache.py \
  --root research/out/timestral-repair-reproduction
.venv-timestral/bin/python scripts/run_timestral_repair.py register \
  --root research/out/timestral-repair-reproduction
.venv-timestral/bin/python scripts/run_timestral_repair.py run \
  --root research/out/timestral-repair-reproduction
.venv-timestral/bin/python -m pytest -q tests/test_timestral_repair.py
```

These commands use a new directory and the existing raw archives; preparation
refuses to replace an existing directory. The cache contains exactly 8,016
consecutive hours from September 1, 2025 through August 1, 2026 (exclusive).
The executed directory is `research/out/timestral-diagnosis-20261026`. Its forecasts
completed, but a NumPy boolean blocked final JSON serialization. That type was
corrected and covered by a regression test; analysis was recovered from the
unchanged saved predictions, without rerunning inference. The original code,
empty failed analysis, error log and explicit recovery receipt are retained.
Raw sources and weights stay outside Git. The functions in `research.timestral_repair` can be reused with a causal
stream, but this task does not deploy or schedule a live service.
