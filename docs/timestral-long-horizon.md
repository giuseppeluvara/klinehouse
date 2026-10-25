# Historical forecasts at four hours and one day

This experiment measures BTCUSDT Spot movements over **4 hours and 24 hours**.
Its inputs remain verified Binance one-second OHLCV candles. Complete UTC hours
are constructed from those seconds before inference: longer forecast horizons do
not require feeding millions of second observations directly into the model.
The original one-hour Timestral pilot and the earlier frozen experiments remain
unchanged. This experiment ends before their August 2026 protected holdout.

## Fixed protocol

The configuration is `configs/timestral_long_horizon.json`.

| Choice | Value |
| --- | --- |
| Source | Binance Spot BTCUSDT, complete 1-second candles |
| Source interval | 2026-04-01 through 2026-08-01, exclusive end, UTC |
| Evaluation | 2026-05-01 through 2026-08-01, exclusive end |
| Working resolution | One hour, exactly 3,600 consecutive closed seconds |
| Context | 256 complete hours, 10 days and 16 hours |
| Horizons | 4 hours and 24 hours, evaluated separately |
| Anchor spacing | 4h for the 4h task; 24h for the daily task |
| Expected evaluation anchors | 552 at 4h and 92 at 24h, before exclusions |
| Model | Pinned Timestral-5M, zero-shot, CPU, two threads |
| Baselines | Unchanged price and mean context log-return extrapolation |
| Variants | Price alone; price plus three past-only second-derived covariates |
| Baseline calibration | Separate expanding matured residual histories per horizon |
| Minimum calibration | 12 fully matured past forecasts |
| Reporting folds | May, June and July 2026, with no tuning between folds |

At an anchor `t`, context hours cover `[t-256h,t)`. The target for horizon `H`
is `log(close(t+H-1h) / close(t-1h))`; each hour's close is the final second's
close. Its outcome becomes available at `t+H`. The horizon is a movement duration,
not a claim that the last observed price can be executed after inference.

The price model receives 256 hourly log-closes. The covariate model adds:

- Realized volatility from all 3,599 within-hour second-to-second log returns.
- `log1p` of the total hourly number of trades.
- Hourly taker imbalance `(2*taker_buy_base-volume)/volume`, zero at zero volume.

Every feature precedes its anchor. No future covariates, global fitted scalers,
interpolation, training, hyperparameter search, or post-result model selection are
used. The same target/context and eligible anchors are used by all four models.
Non-overlap holds separately within each horizon; the 4h and 24h tasks share
market observations and are not independent replications.

An incomplete hour invalidates any affected context or target. Missing data are
reported explicitly, not filled. A gap can affect many anchors because the
context spans over ten days. Report planned anchors, saved predictions, valid
outcomes and exclusions separately. Calibration uses only residuals whose
`label_end <= anchor`; histories and counts are saved separately for each horizon.
The daily baseline begins with few tail observations, so its empirical 90%
interval is not guaranteed to achieve 90% coverage.

## Metrics and interpretation

Report return MAE/RMSE, sign accuracy, pinball loss, 90% coverage/width, quantile
crossing, and monthly scores, separately for 4h and 24h. Preserve every prediction,
all model quantiles, actual outcome, reference price and implied forecast price.
The unchanged-price forecast is neutral; its three-sign accuracy near zero must
not be presented as a directional trading benchmark. Also report the fraction of
non-neutral predictions and accuracy conditional on making one.

Before evaluation, the descriptive uncertainty analysis is fixed to a paired
bootstrap of seven-day calendar blocks starting at the evaluation boundary,
2,000 draws, seed 7. Resample whole blocks within each horizon, including the
short final block. Compare each Timestral variant with both baselines and compare
covariates with price-only. Report the 2.5% and 97.5% percentiles of paired MAE
differences. These approximate intervals preserve within-week dependence but do
not establish independence across weeks, correct for multiple comparisons, or
remove potential pretraining contamination. Do not interpret a small interval as
proof of a tradable advantage.

Model and checkpoint identity, installation, license, and contamination caveats
are documented in [the original Timestral guide](timestral.md). Reuse that exact
local checkpoint and environment. Historical forecasts are exploratory and their
actual emission timestamps are the current run time. There are no orders,
profitability calculations, or completed prospective results here.

## Reproduce

Run the Mac storage guard before downloading this separate dataset. The importer
checks the source-size budget and the 40 GiB free-space planning target, retains
checksummed ZIPs, and streams CSV rows without extracting full CSV files.

```bash
.venv-timestral/bin/python scripts/prepare_timestral_data.py \
  --start 2026-04-01 --end 2026-08-01 \
  --output research/out/timestral-long-horizon-20261025

.venv-timestral/bin/python -m research.timestral_long_horizon preregister \
  --config configs/timestral_long_horizon.json \
  --output research/out/timestral-long-horizon-20261025/run

.venv-timestral/bin/python -m research.timestral_long_horizon backtest \
  --output research/out/timestral-long-horizon-20261025/run

.venv-timestral/bin/python scripts/analyze_timestral_long_horizon.py \
  --run research/out/timestral-long-horizon-20261025/run \
  --output research/out/timestral-long-horizon-20261025/analysis
```

Registration verifies the source archive and writes an immutable protocol with
source, code, model and environment identity before forecasting. Each run uses a
new directory; reruns cannot replace old forecasts. Use a new directory name for
an independent reproduction. Raw data, model cache and run outputs are ignored by
Git; the final report contains the compact reproducibility evidence.

The analysis command verifies saved output hashes and independently recalculates
core metrics before exporting `forecasts.csv`, `analysis.json`, and
`forecast-comparison.png`. Plotting uses Matplotlib; the observed package versions
are pinned in `requirements-timestral-long-horizon.txt`.
