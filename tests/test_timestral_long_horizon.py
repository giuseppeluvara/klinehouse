"""Causal multi-horizon streaming and immutable publication, without a checkpoint."""

from datetime import timedelta
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_long_horizon as experiment

START = experiment.utc("2026-04-01T00:00:00Z")


def config(tmp_path):
    root = tmp_path / "archive"
    root.mkdir(exist_ok=True)
    (root / "catalog.json").write_text(json.dumps({"version": 1, "market": "binance_spot", "interval": "1s", "files": []}))
    return {"symbol": "BTCUSDT", "start": experiment.stamp(START),
            "end": experiment.stamp(START + timedelta(days=6)),
            "evaluation_start": experiment.stamp(START + timedelta(days=3)),
            "archive_root": str(root), "source": "archive", "lookback": 32,
            "horizons": [4, 24], "min_calibration": 1, "max_hours": 200,
            "max_samples": 50, "batch_size": 8192, "torch_threads": 1,
            "local_files_only": True, "model_cache": str(tmp_path / "cache")}


def hours(count=144):
    return [experiment.Hour(START + timedelta(hours=i), 10 + 0.002 * np.sin(i / 7),
                            (0.01, float(i), 0.1)) for i in range(count)]


def seconds(count=7200):
    return [{"open_time": START + timedelta(seconds=i),
             "close_time": START + timedelta(seconds=i, milliseconds=999),
             "close": 100 + i / 1000, "volume": 2., "trades": 3,
             "taker_buy_base": 1.5} for i in range(count)]


def batches(rows, size=8192):
    for i in range(0, len(rows), size):
        yield pa.RecordBatch.from_pylist(rows[i:i + size])


class FakeModel:
    def __init__(self, cfg):
        self.calls = []

    def predict(self, context, cfg, covariates):
        self.calls.append((list(context), cfg["horizon"], covariates))
        point = float(np.diff([item.log_close for item in context]).mean())
        return {"point": point, "quantiles": [point - .01, point, point + .01],
                "all_quantile_levels": [.01, .05, .5, .95, .99],
                "all_return_quantiles": [point - .02, point - .01, point, point + .01, point + .02]}


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def run(tmp_path, monkeypatch, series=None):
    cfg, output = config(tmp_path), tmp_path / "run"
    experiment.preregister(cfg, output)
    # A real stream is a generator with close(); preserve that interface.
    monkeypatch.setattr(experiment, "read_hours", lambda *args: (item for item in (hours() if series is None else series)))
    model = FakeModel(cfg)
    result = experiment.backtest(output, lambda _: model)
    return cfg, output, result, model


def test_hour_features_exact_seconds_batch_invariant_and_past_only():
    source = seconds()
    first_audit, second_audit = {}, {}
    first = list(experiment.hourly_features(batches(source, 8192), first_audit))
    second = list(experiment.hourly_features(batches(source, 137), second_audit))
    assert first == second and len(first) == 2
    for key in ("source_rows_sha256", "source_column_sha256", "hourly_features_sha256", "raw_rows", "valid_hours", "invalid_hours"):
        assert first_audit[key] == second_audit[key]
    expected_rv = np.sqrt(np.square(np.diff(np.log(100 + np.arange(3600) / 1000))).sum())
    assert first[0].features == pytest.approx((expected_rv, np.log1p(10800), .5))
    source[3599]["close"] *= 2
    changed = list(experiment.hourly_features(batches(source), {}))
    assert changed[1] == first[1]
    assert first_audit["raw_rows"] == 7200 and first_audit["invalid_hours"] == 0


@pytest.mark.parametrize("corruption", ["missing", "shifted", "unclosed", "negative", "infinite", "excess_taker"])
def test_incomplete_or_invalid_hour_is_never_used(corruption):
    source = seconds()
    if corruption == "missing":
        del source[23]
    elif corruption == "shifted":
        source[23]["open_time"] += timedelta(microseconds=1)
    elif corruption == "unclosed":
        source[23]["close_time"] = source[23]["open_time"] + timedelta(seconds=1)
    elif corruption == "negative":
        source[23]["close"] = -1.
    elif corruption == "infinite":
        source[23]["close"] = float("inf")
    else:
        source[23]["taker_buy_base"] = 20.
    audit = {}
    actual = list(experiment.hourly_features(batches(source, 511), audit))
    assert [item.time for item in actual] == [START + timedelta(hours=1)]
    assert audit["invalid_hours"] == 1


def test_duplicate_or_unsorted_seconds_fail_and_nulls_fail():
    source = seconds(3)
    with pytest.raises(ValueError, match="strictly increasing"):
        list(experiment.hourly_features(batches(source + source), {}))
    source[0]["close"] = None
    with pytest.raises(ValueError, match="null source"):
        list(experiment.hourly_features(batches(source), {}))


def test_preregister_exclusive_and_imported_code_drift_rejected(tmp_path, monkeypatch):
    cfg, output = config(tmp_path), tmp_path / "run"
    protocol = experiment.preregister(cfg, output)
    assert experiment.load_protocol(output) == protocol
    assert "timestral_experiment.py" in protocol["code_sha256"]
    with pytest.raises(FileExistsError):
        experiment.preregister(cfg, output)
    original = experiment.code_hashes()
    monkeypatch.setattr(experiment, "code_hashes", lambda: {**original, "timestral_experiment.py": "changed"})
    with pytest.raises(ValueError, match="imported code changed"):
        experiment.load_protocol(output)


def test_protocol_tampering_and_invalid_config(tmp_path):
    cfg, output = config(tmp_path), tmp_path / "run"
    experiment.preregister(cfg, output)
    path = output / "protocol.json"
    path.chmod(0o644)
    path.write_text(path.read_text().replace("BTCUSDT", "ETHUSDT"))
    with pytest.raises(ValueError, match="protocol changed"):
        experiment.load_protocol(output)
    with pytest.raises(ValueError, match="UTC midnights"):
        experiment.validate_config({**cfg, "end": "2026-04-07T01:00:00Z"})
    with pytest.raises(ValueError, match="horizons"):
        experiment.validate_config({**cfg, "horizons": [4, 60]})


def test_multi_horizon_exact_anchors_endpoints_prices_and_calibration(tmp_path, monkeypatch):
    cfg, output, result, model = run(tmp_path, monkeypatch)
    predictions, outcomes = records(output / "predictions.jsonl"), records(output / "outcomes.jsonl")
    assert len(predictions) == len(outcomes) == 21
    assert len(model.calls) == 42
    assert set(row["sample_id"] for row in predictions) == set(row["sample_id"] for row in outcomes)
    source = {item.time: item for item in hours()}
    outcome_map = {row["sample_id"]: row for row in outcomes}
    for prediction in predictions:
        anchor, horizon = experiment.utc(prediction["anchor"]), prediction["horizon_hours"]
        assert int(anchor.timestamp()) % (3600 * horizon) == 0
        assert set(prediction["predictions"]) == set(experiment.MODELS)
        assert experiment.utc(prediction["calibration"]["latest_label_end"]) <= anchor
        outcome = outcome_map[prediction["sample_id"]]
        expected = source[anchor + timedelta(hours=horizon - 1)].log_close - source[anchor - timedelta(hours=1)].log_close
        assert outcome["return"] == pytest.approx(expected)
        for name, forecast in prediction["predictions"].items():
            assert forecast["point_price"] == pytest.approx(np.exp(prediction["reference_log_close"] + forecast["point"]))
            assert len(forecast["all_price_quantiles"]) == (5 if name.startswith("timestral") else 3)
    for horizon, count in ((4, 18), (24, 3)):
        assert result["horizons"][str(horizon)]["expected_anchors_if_complete"] == count
        assert result["horizons"][str(horizon)]["scored_anchors"] == count
        assert all(item["n"] == count for item in result["horizons"][str(horizon)]["metrics"].values())
        assert max(experiment.utc(row["label_end"]) for row in predictions if row["horizon_hours"] == horizon) == experiment.utc(cfg["end"])
        history = records(output / f"calibration-{horizon}h.jsonl")
        assert all(row["horizon_hours"] == horizon for row in history)
        assert len(history) == result["horizons"][str(horizon)]["final_calibration_count"]
        assert result["horizons"][str(horizon)]["metrics"]["unchanged"]["direction_accuracy_when_called"] is None
    assert len(records(output / "forecast-rows.jsonl")) == 21 * 4
    with pytest.raises(FileExistsError):
        experiment.backtest(output, lambda _: pytest.fail("rerun loaded model"))


def test_predictions_published_before_any_target_hour_is_consumed(tmp_path, monkeypatch):
    cfg, output, consumed = config(tmp_path), tmp_path / "run", []
    experiment.preregister(cfg, output)

    def stream(*args):
        for hour in hours():
            consumed.append(hour.time)
            yield hour

    monkeypatch.setattr(experiment, "read_hours", stream)
    original_append, published = experiment.shared.append_record, set()

    def append(handle, record):
        if Path(handle.name).name == "predictions.jsonl":
            assert consumed[-1] < experiment.utc(record["anchor"])
            published.add(record["sample_id"])
        if Path(handle.name).name == "outcomes.jsonl":
            assert record["sample_id"] in published
            assert consumed[-1] + timedelta(hours=1) == experiment.utc(record["label_end"])
        original_append(handle, record)

    monkeypatch.setattr(experiment.shared, "append_record", append)
    experiment.backtest(output, FakeModel)


def test_gap_invalidates_emitted_targets_and_resets_context(tmp_path, monkeypatch):
    gap = START + timedelta(days=3, hours=2)
    series = [item for item in hours() if item.time != gap]
    cfg, output, result, model = run(tmp_path, monkeypatch, series)
    outcomes = records(output / "outcomes.jsonl")
    invalid = [item for item in outcomes if item["status"] == "invalid_target"]
    assert {item["horizon_hours"] for item in invalid} == {4, 24}
    predictions = records(output / "predictions.jsonl")
    assert result["data"]["hour_gap_resets"] == 1
    for prediction in predictions:
        start, anchor = experiment.utc(prediction["input_start"]), experiment.utc(prediction["anchor"])
        assert not start <= gap < anchor
    for horizon in (4, 24):
        history = records(output / f"calibration-{horizon}h.jsonl")
        assert not any(experiment.utc(item["anchor"]) <= gap < experiment.utc(item["label_end"]) for item in history)
        assert result["horizons"][str(horizon)]["emitted_predictions"] == result["horizons"][str(horizon)]["scored_anchors"] + 1


def test_missing_final_hour_invalidates_targets_without_retroactive_prediction_filter(tmp_path, monkeypatch):
    _, output, result, _ = run(tmp_path, monkeypatch, hours()[:-1])
    invalid = [item for item in records(output / "outcomes.jsonl") if item["status"] == "invalid_target"]
    assert len(invalid) == 2
    assert {item["horizon_hours"] for item in invalid} == {4, 24}
    assert result["horizons"]["4"]["emitted_predictions"] == 18
    assert result["horizons"]["24"]["emitted_predictions"] == 3


def test_future_labels_never_affect_baseline_calibration():
    context, anchor = hours(32), START + timedelta(days=3)
    calibration = experiment.shared.Calibration()
    calibration.add(anchor, .03, .01)
    cfg = {"horizon": 24, "min_calibration": 1}
    before = calibration.forecast(anchor, context, cfg)
    calibration.add(anchor + timedelta(seconds=1), 1000, -1000)
    assert calibration.forecast(anchor, context, cfg) == before
    assert before[1] == {"count": 1, "latest_label_end": experiment.stamp(anchor)}


def test_full_production_calendar_has_552_four_hour_and_92_daily_anchors(tmp_path, monkeypatch):
    cfg, output = config(tmp_path), tmp_path / "run"
    cfg.update(end="2026-08-01T00:00:00Z", evaluation_start="2026-05-01T00:00:00Z",
               lookback=256, min_calibration=12, max_hours=3000, max_samples=800)
    experiment.preregister(cfg, output)
    monkeypatch.setattr(experiment, "read_hours", lambda *args: (item for item in hours(2928)))
    result = experiment.backtest(output, FakeModel)
    assert result["data"]["candidate_anchors_including_warmup"] == 779
    assert result["data"]["expected_hours"] == 2928
    assert result["data"]["expected_raw_seconds"] == 10540800
    for horizon, expected, calibration in ((4, 552, 116), (24, 92, 19)):
        summary = result["horizons"][str(horizon)]
        assert summary["expected_anchors_if_complete"] == summary["scored_anchors"] == expected
        assert summary["first_calibration_count"] == calibration
        assert summary["skipped_missing_context_or_anchor"] == summary["skipped_insufficient_calibration"] == 0
