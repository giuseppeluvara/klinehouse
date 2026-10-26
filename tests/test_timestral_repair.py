"""Causal admission, horizon-target semantics and volatility-aware intervals."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from research.timestral_long_horizon import Hour
from research import timestral_repair as repair
from scripts import run_timestral_repair as runner

UTC = timezone.utc
JANUARY = datetime(2026, 1, 1, tzinfo=UTC)


def history(horizon=24, quality="bad"):
    start = datetime(2025, 10, 1, tzinfo=UTC)
    rows = []
    for i in range(92 * 24 // horizon):
        anchor = start + timedelta(hours=i * horizon)
        y = (.01 if i % 2 else -.013) * (1 + i % 5 / 10)
        points = {"timestral_price": -y, "timestral_returns": -2 * y}
        if quality == "good":
            points["timestral_returns"] = y * .9
        elif quality == "tie":
            points = dict.fromkeys(points, 0.)
        rows.append({"sample_id": str(i), "horizon_hours": horizon,
                     "anchor": anchor.isoformat(), "label_end": (anchor + timedelta(hours=horizon)).isoformat(),
                     "actual_log_return": y, "scale": .01,
                     "predictions": {key: {"point": value} for key, value in points.items()}})
    return rows


@pytest.mark.parametrize("horizon", [4, 24])
def test_baseline_is_selected_when_models_lose_and_future_perfection_cannot_change_it(horizon):
    rows = history(horizon)
    chosen = repair.choose_model(rows, JANUARY, horizon)
    assert chosen["selected"] == "unchanged"
    assert chosen["status"] == "baseline_no_evidence"
    future = {**rows[-1], "sample_id": "future", "anchor": JANUARY.isoformat(),
              "label_end": (JANUARY + timedelta(hours=horizon)).isoformat(), "actual_log_return": 1e20}
    assert repair.choose_model([*rows, future], JANUARY, horizon) == chosen
    assert repair.choose_model([*rows, future], JANUARY + timedelta(days=15), horizon) == chosen


@pytest.mark.parametrize("horizon", [4, 24])
def test_evidence_can_admit_a_real_signal_instead_of_always_forcing_baseline(horizon):
    result = repair.choose_model(history(horizon, "good"), JANUARY, horizon)
    assert result["selected"] == "timestral_returns"
    assert all(result["candidate_checks"]["timestral_returns"]["checks"].values())
    assert result["latest_label_end"] <= JANUARY.isoformat()


def test_ties_and_small_samples_never_promote_model():
    assert repair.choose_model(history(quality="tie"), JANUARY, 24)["selected"] == "unchanged"
    result = repair.choose_model(history()[-31:], JANUARY, 24)
    assert result["selected"] == "unchanged" and result["status"] == "insufficient_history"


def test_every_selection_flag_can_be_persisted_as_strict_json():
    for quality in ("bad", "good", "tie"):
        decision = repair.choose_model(history(quality=quality), JANUARY, 24)
        assert json.loads(json.dumps(decision, allow_nan=False)) == decision
        assert all(type(flag) is bool for candidate in decision["candidate_checks"].values()
                   for flag in candidate["checks"].values())


def test_conformal_rank_and_present_volatility_change_radius_without_future_labels():
    rows = history()
    result = repair.calibrated_interval(rows, JANUARY, 24, "unchanged", 0., .01)
    assert result["samples"] == 90 and result["rank"] == 82
    residuals = sorted(abs(row["actual_log_return"]) / row["scale"] for row in rows[-90:])
    assert result["quantiles"] == [-residuals[81] * .01, 0., residuals[81] * .01]
    wide = repair.calibrated_interval(rows, JANUARY, 24, "unchanged", 0., .02)
    np.testing.assert_allclose(wide["quantiles"], np.asarray(result["quantiles"]) * 2)
    future = {**rows[-1], "sample_id": "future", "anchor": JANUARY.isoformat(),
              "label_end": (JANUARY + timedelta(days=1)).isoformat(), "actual_log_return": 10000.}
    assert repair.calibrated_interval([*rows, future], JANUARY, 24, "unchanged", 0., .01) == result
    assert result["coverage_guaranteed"] is False


def test_insufficient_calibration_and_invalid_history_are_explicit():
    rows = history()
    assert repair.calibrated_interval(rows[-10:], JANUARY, 24, "unchanged", 0., .01)["quantiles"] is None
    with pytest.raises(ValueError, match="duplicate"):
        repair.matured([*rows, rows[-1]], JANUARY, 24)
    with pytest.raises(ValueError, match="scale"):
        repair.calibrated_interval(rows, JANUARY, 24, "unchanged", 0., 0.)
    bad = {**rows[-1], "label_end": (JANUARY + timedelta(hours=1)).isoformat()}
    with pytest.raises(ValueError, match="label interval"):
        repair.matured([bad], JANUARY, 24)


class ArrayTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, key):
        return ArrayTensor(self.values[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeModel:
    def __init__(self):
        self.forecaster = self
        self.inputs = []

    def predict(self, hours, config, covariates):
        assert len(hours) == 256 and covariates is False
        return {"point": .001, "quantiles": [-.01, .001, .01]}

    def forecast(self, inputs, **kwargs):
        self.inputs.append((inputs.copy(), kwargs.copy()))
        values = np.broadcast_to(np.array([-50., 7., 80.]), (1, 1, kwargs["prediction_length"], 3))
        return SimpleNamespace(quantile_levels=[.05, .5, .95], quantiles=ArrayTensor(values))


@pytest.mark.parametrize("horizon", [4, 24])
def test_direct_horizon_target_uses_correct_lag_endpoint_and_units(horizon):
    hours = [Hour(JANUARY - timedelta(hours=256 - i), 10 + i * .001, (.01, 1., 0.))
             for i in range(256)]
    model = FakeModel()
    result, scale = repair.forecast_candidates(model, hours, JANUARY, horizon)
    assert model.inputs[0][0].shape == (1, 1, 256 - horizon)
    np.testing.assert_allclose(model.inputs[0][0], horizon * 10)
    assert model.inputs[0][1]["prediction_length"] == horizon
    assert model.inputs[0][1]["clip_nonnegative"] is False
    assert result["timestral_returns"]["point"] == .0007
    assert scale == pytest.approx(.01 * np.sqrt(horizon))
    with pytest.raises(ValueError, match="strictly before"):
        repair.past_inputs(hours, JANUARY - timedelta(hours=1), horizon)


def test_replay_emits_predictions_before_outcomes_and_future_change_leaves_origin_unchanged():
    start = datetime(2025, 10, 1, tzinfo=UTC)
    first = start - timedelta(hours=280)
    end = JANUARY + timedelta(days=2)
    hours = [Hour(first + timedelta(hours=i), 10 + .001 * np.sin(i), (.005, 1., 0.))
             for i in range(int((end - first).total_seconds() / 3600))]
    protocol = {"fit_start": start.isoformat(), "evaluation_start": JANUARY.isoformat(), "evaluation_end": end.isoformat()}
    events = []
    _, rows, decisions = runner.replay(hours, protocol, FakeModel(), lambda kind, row: events.append((kind, row.copy())))
    assert len(rows) == 14 and len(decisions) == 2
    for i in range(0, len(events), 2):
        assert events[i][0] == "prediction" and "actual_log_return" not in events[i][1]
        assert events[i + 1][0] == "outcome" and events[i][1]["sample_id"] == events[i + 1][1]["sample_id"]
    altered = [Hour(x.time, x.log_close + (1 if x.time >= JANUARY else 0), x.features) for x in hours]
    _, changed, _ = runner.replay(altered, protocol, FakeModel(), lambda *_: None)
    for h in (4, 24):
        before = next(x for x in rows if x["anchor"] == "2026-01-01T00:00:00Z" and x["horizon_hours"] == h)
        after = next(x for x in changed if x["anchor"] == before["anchor"] and x["horizon_hours"] == h)
        assert before["predictions"] == after["predictions"]
        assert before["selected"] == after["selected"]
        assert before["actual_log_return"] != after["actual_log_return"]


def registered_fixture(root):
    content = json.dumps({"time": "2025-09-01T00:00:00Z", "log_close": 10., "features": [.01, 1., 0.]}) + "\n"
    (root / "hours.jsonl").write_text(content)
    (root / "cache-provenance.json").write_text(json.dumps({"hours_sha256": hashlib.sha256(content.encode()).hexdigest()}))
    (root / "plan.json").write_text(json.dumps({"status": "diagnostic"}))
    runner.register(root)


def test_registered_inputs_cannot_be_changed_or_protocol_replaced(tmp_path):
    registered_fixture(tmp_path)
    assert runner.verified_protocol(tmp_path)["status"].startswith("diagnostic")
    with pytest.raises(FileExistsError):
        runner.register(tmp_path)
    (tmp_path / "plan.json").write_text('{"status":"changed"}')
    with pytest.raises(ValueError, match="changed"):
        runner.verified_protocol(tmp_path)


def test_opening_receipt_prevents_rerun_before_model_initialization(tmp_path):
    registered_fixture(tmp_path)
    (tmp_path / "run/opened.json").write_text('{}')
    def factory(_):
        pytest.fail("model must not be initialized on a duplicate opening")
    with pytest.raises(FileExistsError):
        runner.run(tmp_path, model_factory=factory)
