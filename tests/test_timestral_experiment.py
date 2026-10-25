"""Causality and publication contracts; no checkpoint or model import required."""

from datetime import timedelta
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_experiment as experiment

START = experiment.utc("2026-07-01T00:00:00Z")


def config(tmp_path):
    root = tmp_path / "archive"
    root.mkdir(exist_ok=True)
    (root / "catalog.json").write_text(json.dumps({"version": 1, "market": "binance_spot", "interval": "1s", "files": []}))
    return {"symbol": "BTCUSDT", "start": experiment.stamp(START),
            "end": experiment.stamp(START + timedelta(hours=4)),
            "evaluation_start": experiment.stamp(START + timedelta(hours=2)),
            "archive_root": str(root), "source": "archive", "lookback": 32,
            "horizon": 60, "stride": 60, "min_calibration": 1, "max_minutes": 300,
            "max_samples": 5, "batch_size": 17, "torch_threads": 1,
            "max_emission_delay_seconds": 60, "local_files_only": True}


def minutes(count=240):
    return [experiment.Minute(START + timedelta(minutes=i), 10 + 0.001 * np.sin(i / 13),
                              (0.01, float(i), 0.1)) for i in range(count)]


def seconds(count):
    rows = []
    for i in range(count):
        current = START + timedelta(seconds=i)
        rows.append({"open_time": current, "close_time": current + timedelta(milliseconds=999),
                     "close": float(100 + i), "volume": 2.0, "trades": 3,
                     "taker_buy_base": 1.5})
    return rows


def batches(rows, size=17):
    for i in range(0, len(rows), size):
        yield pa.RecordBatch.from_pylist(rows[i:i + size])


def test_streaming_features_are_batch_invariant_and_use_only_within_minute_returns():
    rows = seconds(120)
    audit1, audit2 = {}, {}
    first = list(experiment.minute_features(batches(rows, 17), audit1))
    second = list(experiment.minute_features(batches(rows, 60), audit2))
    assert first == second and audit1 == audit2
    assert audit1["raw_rows"] == 120 and audit1["valid_minutes"] == 2
    assert first[0].features == pytest.approx((np.sqrt(np.square(np.diff(np.log(np.arange(100, 160)))).sum()), np.log1p(180), 0.5))
    # Alter the last second of minute zero: minute one's features do not change.
    rows[59]["close"] *= 2
    changed = list(experiment.minute_features(batches(rows), {}))
    assert changed[1] == first[1]


def test_missing_seconds_and_unclosed_candles_invalidate_minutes():
    rows = seconds(180)
    del rows[5]
    rows[-1]["close_time"] += timedelta(seconds=1)
    audit = {}
    actual = list(experiment.minute_features(batches(rows), audit))
    assert [item.time for item in actual] == [START + timedelta(minutes=1)]
    assert audit["invalid_minutes"] == 2
    with pytest.raises(ValueError, match="strictly increasing"):
        list(experiment.minute_features(batches(seconds(10) + seconds(10)), {}))


def test_gap_invalidates_both_context_and_target(tmp_path):
    cfg = config(tmp_path)
    whole = list(experiment.sample_windows(iter(minutes()), cfg, {}))
    assert [item[0].hour for item in whole] == [1, 2, 3]
    incomplete = [item for item in minutes() if item.time != START + timedelta(minutes=115)]
    audit = {}
    actual = list(experiment.sample_windows(iter(incomplete), cfg, audit))
    # Minute 115 is in anchor 01's target and anchor 02's context.
    assert [item[0].hour for item in actual] == [3]
    assert audit["minute_gap_resets"] == 1


def test_calibration_excludes_future_labels_and_includes_equal_boundary(tmp_path):
    cfg, context = config(tmp_path), minutes(32)
    anchor = START + timedelta(hours=2)
    calibration = experiment.Calibration()
    calibration.add(anchor, 0.01, 0.002)
    before, evidence = calibration.forecast(anchor, context, cfg)
    calibration.add(anchor + timedelta(microseconds=1), 1000, -1000)
    after, repeated = calibration.forecast(anchor, context, cfg)
    assert after == before and evidence == repeated
    assert evidence == {"count": 1, "latest_label_end": experiment.stamp(anchor)}
    assert before["unchanged"]["point"] == 0
    assert before["unchanged"]["quantiles"] == [0.01] * 3


def test_preregistration_is_exclusive_and_tampering_is_rejected(tmp_path):
    cfg, output = config(tmp_path), tmp_path / "run"
    protocol = experiment.preregister(cfg, output)
    assert experiment.load_protocol(output) == protocol
    with pytest.raises(FileExistsError):
        experiment.preregister(cfg, output)
    path = output / "protocol.json"
    path.chmod(0o644)
    path.write_text(path.read_text().replace("BTCUSDT", "ETHUSDT"))
    with pytest.raises(ValueError, match="protocol changed"):
        experiment.load_protocol(output)


class FakeModel:
    def __init__(self, cfg):
        self.calls = []

    def predict(self, context, cfg, covariates):
        self.calls.append((list(context), covariates))
        return {"point": 0.0, "quantiles": [-0.01, 0.0, 0.01]}


def test_backtest_same_samples_and_forecasts_are_published_before_outcomes(tmp_path, monkeypatch):
    cfg, output, calls = config(tmp_path), tmp_path / "run", []
    model = FakeModel(cfg)
    experiment.preregister(cfg, output)
    monkeypatch.setattr(experiment, "read_minutes", lambda *args, **kwargs: (item for item in minutes()))
    original_append = experiment.append_record

    def append(handle, record):
        calls.append(Path(handle.name).name)
        original_append(handle, record)

    monkeypatch.setattr(experiment, "append_record", append)
    result = experiment.backtest(output, model_factory=lambda cfg: model)
    assert calls == ["predictions.jsonl", "outcomes.jsonl"] * 2
    records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(result["sample_ids"]) == 2 and len(model.calls) == 4
    for i, record in enumerate(records):
        assert set(record["predictions"]) == set(experiment.MODELS)
        assert experiment.utc(record["calibration"]["latest_label_end"]) <= experiment.utc(record["anchor"])
        for context, _ in model.calls[i * 2:i * 2 + 2]:
            assert context[-1].time < experiment.utc(record["anchor"])
            assert len(context) == cfg["lookback"]
    assert all(metrics["n"] == 2 for metrics in result["metrics"].values())
    with pytest.raises(FileExistsError):
        experiment.backtest(output, model_factory=lambda cfg: pytest.fail("rerun loaded model"))


def test_timestral_official_api_contract_without_importing_model(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    monkeypatch.setattr(experiment, "CHECKPOINT_SHA256", {"config.json": experiment.digest_file(checkpoint / "config.json")})
    calls = []
    monkeypatch.setattr(experiment, "verify_model_installation", lambda: None)

    class Tensor:
        def __getitem__(self, key):
            return self
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return np.array([9.0, 10.0, 11.0])

    class Forecaster:
        supports_covariates = True
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return cls()
        def forecast(self, series, **kwargs):
            calls.append((series, kwargs))
            return SimpleNamespace(quantiles=Tensor(), quantile_levels=experiment.LEVELS)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None, manual_seed=lambda _: None))
    monkeypatch.setitem(sys.modules, "timestral", SimpleNamespace(TimestralForecaster=Forecaster))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **k: str(checkpoint)))
    model = experiment.Timestral(cfg)
    prediction = model.predict(minutes(32), cfg, True)
    assert calls[0][1]["revision"] == experiment.WEIGHT_REVISION
    assert calls[0][1]["device"] == "cpu" and calls[0][1]["local_files_only"] is True
    series, kwargs = calls[1]
    assert series.shape == (1, 1, 32)
    assert kwargs["past_covariates"].shape == (1, 3, 32)
    assert "known_covariates" not in kwargs
    assert kwargs["prediction_length"] == 60 and kwargs["clip_nonnegative"] is False
    assert prediction["point"] == pytest.approx(10 - minutes(32)[-1].log_close)


def test_prospective_reads_only_past_and_refuses_late_emission(tmp_path, monkeypatch):
    cfg, output = config(tmp_path), tmp_path / "run"
    experiment.preregister(cfg, output)
    anchor = experiment.utc(cfg["end"])
    ends = []

    def read(config, audit, end=None):
        ends.append(end)
        yield from minutes()

    monkeypatch.setattr(experiment, "read_minutes", read)
    record = experiment.prospect(output, anchor, FakeModel, now=lambda: anchor + timedelta(seconds=2))
    assert ends == [anchor]
    assert "return" not in record and "outcome" not in record
    assert experiment.utc(record["issued_at"]) < experiment.utc(record["label_end"])
    assert experiment.utc(record["earliest_execution_ts"]) > experiment.utc(record["issued_at"])
    with pytest.raises(ValueError, match="not mature"):
        experiment.score_prospective(output, now=lambda: anchor + timedelta(minutes=59))
    with pytest.raises(ValueError, match="emission window"):
        experiment.prospect(output, anchor, FakeModel, now=lambda: anchor + timedelta(seconds=61))
    other = tmp_path / "late-run"
    experiment.preregister(cfg, other)
    clock = iter([anchor + timedelta(seconds=1), anchor + timedelta(seconds=61)])
    with pytest.raises(ValueError, match="missed"):
        experiment.prospect(other, anchor, FakeModel, now=lambda: next(clock))
    assert not (other / "prospective-prediction.json").exists()


def test_metrics_measure_return_quantiles_and_crossing():
    record = {"predictions": {name: {"point": 0.01, "quantiles": [-0.01, 0.01, 0.03]} for name in experiment.MODELS}}
    result = experiment.metric_summary([(record, 0.02)])
    assert result["timestral_price"]["return_mae"] == pytest.approx(0.01)
    assert result["timestral_price"]["coverage90"] == 1
    assert result["timestral_price"]["quantile_crossing"] == 0
    record["predictions"]["timestral_price"]["quantiles"] = [0.03, 0.01, -0.01]
    assert experiment.metric_summary([(record, 0.02)])["timestral_price"]["quantile_crossing"] == 1
