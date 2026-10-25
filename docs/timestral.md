# First Timestral experiment

This experiment is separate from `PLAN.md`, `RAPID_EVIDENCE.md`,
`PROFITABILITY_SPRINT.md`, and their frozen data and models. It performs zero-shot
forecasting on Binance Spot OHLCV **candles at one-second resolution**, aggregated
causally to one minute. It does not place orders or estimate trading profit.

## Model and reproducibility

The official [Mistral repository](https://github.com/mistralai/timestral) exposes
`TimestralForecaster.from_pretrained` and `forecaster.forecast`. This experiment
pins code to `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` (v1.0.0) and
[`mistralai/Timestral-5M`](https://huggingface.co/mistralai/Timestral-5M/tree/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f)
weights to `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f`. The loader checks the
installed Git revision, verifies expected weight/configuration SHA-256 before model
loading, and records SHA-256 of locally resolved checkpoint files.
It uses CPU, two PyTorch threads, float32 inference, and a 256-step context.
The official scaler uses float64 internally, which makes this implementation's
MPS path unsuitable; CPU is the fixed first configuration.

Weights use the [Modified MIT license](https://huggingface.co/mistralai/Timestral-5M/blob/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f/LICENSE),
which includes a commercial revenue condition: organizations above USD 20 million
revenue in the preceding calendar month require a separate commercial license.
Preserve the actual license with redistributed weights; this is not unrestricted
MIT licensing.

Mistral describes synthetic pretraining and subsequent training on real and
synthetic series in its [announcement](https://mistral.ai/news/timestral/), but does
not disclose a complete real-data inventory or cutoff there. Historical overlap
cannot be excluded. Every historical result is therefore labelled **exploratory**;
a successful historical score would not establish an uncontaminated trading edge.

## Registered protocol

`configs/timestral_smoke.json` fixes these choices before inference:

| Item | First configuration |
| --- | --- |
| Market | Binance Spot BTCUSDT |
| Source period | 2026-07-01 00:00 through 2026-07-08 00:00 UTC, exclusive end |
| Evaluation period | 2026-07-02 00:00 through 2026-07-08 00:00 UTC |
| Working bars / context | One minute / 256 complete minutes |
| Forecast horizon / anchor spacing | 60 minutes / 60 minutes, whole UTC hours |
| Target | `log(close(t+59m) / close(t-1m))` |
| Calibration | Expanding previous baseline residuals with `label_end <= t`; minimum 12 |
| Folds | Six daily UTC evaluation blocks; no tuning or model selection |
| Quantiles scored | 0.05, 0.50, 0.95 |

With complete data, the first usable calibration anchor is July 1 at 05:00 UTC;
19 matured samples precede the first evaluated anchor. Evaluation comprises 144
anchors on identical contexts for all four alternatives. These are deliberately
small initial samples: empirical tail quantiles and coverage are noisy.

At anchor `t`, only seconds in `[t-256m,t)` enter the context. The forecast target
uses complete candles in `[t,t+60m)` and becomes available at `t+60m`. Missing or
invalid seconds invalidate their minute. Any minute gap resets the rolling window,
so it invalidates every affected context **and** target; nothing is forward-filled.
Raw times must be increasing and each source close time must fall within its own
second. All usable anchors are exactly one hour apart, so labels do not overlap.

The four alternatives are:

1. **Unchanged price:** point return zero; empirical quantiles of previously
   observed returns.
2. **Local drift:** mean minute log-return in the current context times 60;
   quantiles from expanding, previously matured out-of-sample forecast residuals
   added to the current drift forecast.
3. **Timestral price:** the 256 log-closes, without extra covariates.
4. **Timestral covariates:** identical log-closes with three past-only channels:
   realized volatility from the 59 within-minute close-to-close log returns,
   `log1p(sum(trades))`, and `(2*sum(taker_buy_base)-sum(volume))/sum(volume)`
   (zero when volume is zero).

There are no future-known covariates, fitted global normalizers, extra assets, or
hyperparameter searches. Timestral receives `(1,1,256)` target values and optional
`(1,3,256)` past covariates. Its last horizon step supplies return quantiles after
subtracting the last observed log-close. Median is the model's point prediction.
Symmetric averaging is enabled; nonnegative clipping is disabled for log-prices.

The baseline calibration is prequential: each historical forecast is computed
from its own past context, and its residual only becomes usable when its entire
target has matured. Calibration expands within and across daily folds. A baseline
score never gates Timestral inference.

Metrics are MAE/RMSE in **log-return units**, strict sign accuracy (zero is its own
sign), mean and individual pinball loss, central 90% empirical coverage, interval
width, and crossing among all 21 model quantiles (three for baselines). All 21
terminal model quantiles and their levels are saved. Quantiles are not interpreted
as calibrated probabilities of profit. Unchanged-price sign accuracy will usually
be zero under this explicit three-sign convention.

## Run on a verified archive

Apply the Mac storage guard before creating the environment/downloading weights.
The environment should reuse suitable installed packages where available.

```bash
python -m venv --system-site-packages .venv-timestral
.venv-timestral/bin/python -m pip install -r requirements-timestral.txt
```

Prepare and verify the real source archive with public Binance data (no database
required), or use the database archival workflow in `docs/storage.md`:

```bash
.venv-timestral/bin/python scripts/prepare_timestral_data.py \
  --start 2026-07-01 --end 2026-07-08 \
  --output research/out/timestral-pilot-20261025
.venv-timestral/bin/python - <<'PYMODEL'
from huggingface_hub import snapshot_download
snapshot_download("mistralai/Timestral-5M",
    revision="bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
    allow_patterns=["*.json", "*.safetensors", "LICENSE", "README.md"],
    cache_dir="research/out/timestral-pilot-20261025/model-cache")
PYMODEL
```

The importer verifies official ZIP checksums and preserves a provenance manifest.
Downloads are Spot klines, not trade ticks or order-book events. The manifest
records source units: Binance public Spot timestamps since 2025 are microseconds;
the collector's older millisecond timestamps remain valid in their own archives.
Do not mix this research copy with an existing production archive.
 The example
configuration expects `research/out/timestral-pilot-20261025/archive` and the pinned
checkpoint in `research/out/timestral-pilot-20261025/model-cache`. The first model
download must explicitly name `WEIGHT_REVISION`; subsequent runs are local-only.

```bash
.venv-timestral/bin/python -m research.timestral_experiment preregister \
  --config configs/timestral_smoke.json \
  --output research/out/timestral-pilot-20261025/run

.venv-timestral/bin/python -m research.timestral_experiment backtest \
  --output research/out/timestral-pilot-20261025/run

.venv-timestral/bin/python -m pytest -q tests/test_timestral_experiment.py
```

Preregistration exclusively creates the output directory, read-only `protocol.json`
and its hash, before model loading or outcome evaluation. It records configuration,
source catalog hash, code/reader hashes, model identity, baselines, metrics, and environment.
Relevant archive files are fully verified against their catalog under the shared
archive lock before preregistration and again before inference.
Changing protocol/code or rerunning into existing outputs is refused. Use a new
explicitly named run if an error requires a corrected protocol; retain failed runs.
Local hashes/read-only flags are audit aids, not independently trusted timestamps.

Historical `predictions.jsonl` and `outcomes.jsonl` are separate append-only files.
Every evaluated prediction is flushed and fsynced before its outcome is recorded.
The historical reader necessarily visits target data to check completeness, but the
forecast function receives context and matured calibration only. Historical issue
times are real run times and are never represented as historical live emissions.

`results.json` contains common sample IDs, aggregate/per-day scores, stream/data
fingerprints, prediction/outcome/checkpoint hashes, package versions, model timing,
wall time and process peak RSS. The raw reader hashes each selected source row and
each derived minute. It retains one Arrow batch plus at most 316 minutes, with
explicit period/sample budgets; only the small calibrated errors and scored records
accumulate. DuckDB's buffer limit is additional to Python, Arrow and model memory.

## Prospective evaluation

Prepare a **new** configuration and directory for a forthcoming whole-hour UTC
anchor. Set `end` to that anchor, retain enough earlier data for calibration, and
select a source exposing the just-closed second (`auto`/`database` for a live tail).
Preregister before the anchor. `evaluation_start` remains an explicit historical
calibration boundary, and should lie between `start` and `end`. The existing July
pilot configuration is not a live prospective configuration.

```bash
.venv-timestral/bin/python -m research.timestral_experiment preregister \
  --config /path/to/prospective-config.json --output /path/to/new-run

# Run immediately after the registered UTC anchor closes.
.venv-timestral/bin/python -m research.timestral_experiment prospect \
  --output /path/to/new-run --anchor YYYY-MM-DDTHH:00:00Z

# Only after the full 60-minute target has matured:
.venv-timestral/bin/python -m research.timestral_experiment score-prospective \
  --output /path/to/new-run
```

`prospect` queries exclusively before the anchor and saves no future outcome.
It refuses old/future anchors and refuses publication if data preparation plus
inference finishes 60 seconds or more after the anchor. It records actual emission,
inference duration, and a lower bound for execution strictly after emission; that
bound is not a fill estimate. Forecasts are relative to the last context close,
which cannot automatically be used as an executable price after inference.

`score-prospective` reads the target only after maturity and writes a separate,
exclusive outcome file. A command being implemented or a forecast being saved
does not constitute a completed prospective evaluation. No scheduler or automatic
ongoing forecast collection is activated by these commands.
