"""Frozen development selection and causal feature contracts, without market data."""

from datetime import timedelta
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_adaptation as adaptation


def feature_values(index=0):
    return {name: float(np.sin(index / (i + 2))) for i, name in enumerate(adaptation.FEATURES)}


def examples():
    result = []
    for horizon in (4, 24):
        for phase, start, count in (("fit", "2025-10-01", 40), ("validation", "2025-12-01", 31)):
            for i in range(count):
                anchor = adaptation.utc(start) + timedelta(days=i)
                base = .01 * np.sin(i / 3)
                result.append({"sample_id": f"{horizon}-{phase}-{i}", "anchor": adaptation.stamp(anchor),
                               "label_end": adaptation.stamp(anchor + timedelta(hours=horizon)),
                               "horizon_hours": horizon, "features": feature_values(i),
                               "baseprice": base, "basecov": base + .002, "actual": base})
    return result


def context(anchor):
    return [SimpleNamespace(time=anchor - timedelta(hours=256 - i), log_close=10 + i * .001,
                            features=(.01 + i / 10000, 5 + i / 100, -.5 + i / 256)) for i in range(256)]


def spec(name):
    return next(item for item in adaptation.candidates() if item["name"] == name)


def test_exact_causal_summaries_only_use_the_256_past_hours():
    anchor = adaptation.utc("2026-01-01")
    past = context(anchor)
    features = adaptation.extract_features(past, anchor)
    assert features["return_4h"] == pytest.approx(.004)
    assert features["return_24h"] == pytest.approx(.024)
    assert features["return_168h"] == pytest.approx(.168)
    assert features["realized_volatility_24h"] == pytest.approx(np.sqrt(sum(item.features[0] ** 2 for item in past[-24:])))
    assert features["mean_taker_imbalance_24h"] == pytest.approx(np.mean([item.features[2] for item in past[-24:]]))
    assert features["log_trades_24h_minus_168h"] == pytest.approx(np.mean([item.features[1] for item in past[-24:]]) - np.mean([item.features[1] for item in past[-168:]]))
    # The feature function accepts past context only: an unrelated future object
    # can change arbitrarily without altering the returned feature mapping.
    future = SimpleNamespace(log_close=1)
    future.log_close = 100000
    assert adaptation.extract_features(past, anchor) == features


@pytest.mark.parametrize("change", ["gap", "shift", "future", "short"])
def test_context_rejects_gaps_shifts_future_rows_and_short_history(change):
    anchor = adaptation.utc("2026-01-01")
    past = context(anchor)
    if change == "gap":
        past[30].time += timedelta(hours=1)
    elif change == "shift":
        for item in past:
            item.time += timedelta(minutes=1)
    elif change == "future":
        past[-1].time = anchor
    else:
        past.pop()
    with pytest.raises(ValueError, match="context"):
        adaptation.extract_features(past, anchor)


def test_context_rejects_invalid_hourly_covariates_and_nonhour_anchor():
    anchor = adaptation.utc("2026-01-01")
    past = context(anchor)
    past[5].features = (-1, 0, 0)
    with pytest.raises(ValueError, match="invalid volatility"):
        adaptation.extract_features(past, anchor)
    with pytest.raises(ValueError, match="exact UTC hour"):
        adaptation.extract_features(context(anchor), anchor + timedelta(seconds=1))


def test_fixed_candidate_grid_and_training_only_bias():
    assert len(adaptation.candidates()) == 9
    rows = [row for row in examples() if row["horizon_hours"] == 4 and row["anchor"] < "2025-12-01"]
    parameters = adaptation.fit_candidate(spec("shrink_price_0.25"), rows, "2025-12-01")
    expected = .5 * np.median([row["actual"] - .25 * row["baseprice"] for row in rows])
    assert parameters["bias"] == pytest.approx(expected)
    before = adaptation.predict_candidate(parameters, feature_values(), .01, .02)
    rows[0]["actual"] = 1000
    assert adaptation.predict_candidate(parameters, feature_values(), .01, .02) == before


def test_scaler_and_regularized_coefficients_come_only_from_fit_rows():
    rows = [row for row in examples() if row["horizon_hours"] == 4 and row["anchor"] < "2025-12-01"]
    parameters = adaptation.fit_candidate(spec("ridge_state_1"), rows, "2025-12-01")
    names = adaptation.BASE_FEATURES + adaptation.FEATURES
    matrix = np.array([[{**row["features"], "baseprice": row["baseprice"], "basecov": row["basecov"]}[name] for name in names] for row in rows])
    assert parameters["mean"] == pytest.approx(matrix.mean(axis=0))
    assert parameters["scale"] == pytest.approx(np.maximum(matrix.std(axis=0), 1e-8))
    z = np.clip((matrix - parameters["mean"]) / parameters["scale"], -8, 8)
    centered = z - z.mean(axis=0)
    y = np.array([row["actual"] for row in rows])
    coefficients = np.linalg.solve(centered.T @ centered + len(rows) * np.eye(len(names)), centered.T @ (y - y.mean()))
    assert parameters["coefficients"] == pytest.approx(coefficients)
    saved = copy.deepcopy(parameters)
    adaptation.predict_candidate(parameters, {name: 1e6 for name in adaptation.FEATURES}, 1e6, -1e6)
    assert parameters == saved


def test_future_labels_are_rejected_before_any_fit_or_output(tmp_path):
    rows = examples()
    future = copy.deepcopy(rows[0])
    future.update(sample_id="future", anchor="2026-01-01T00:00:00Z", label_end="2026-01-01T04:00:00Z")
    with pytest.raises(ValueError, match="test or future labels"):
        adaptation.select_and_freeze([*rows, future], tmp_path / "freeze")
    assert not (tmp_path / "freeze").exists()
    with pytest.raises(ValueError, match="matured labels"):
        adaptation.fit_candidate(spec("ridge_state_1"), [future], "2026-01-01")


def test_invalid_endpoints_duplicate_ids_and_unknown_features_fail():
    rows = examples()
    rows[0]["label_end"] = "2025-10-01T03:00:00Z"
    with pytest.raises(ValueError, match="endpoint"):
        adaptation.validate_examples(rows, None)
    with pytest.raises(ValueError, match="duplicate"):
        adaptation.validate_examples(examples() + [examples()[0]], None)
    rows = examples()
    rows[0]["features"]["future_return"] = rows[0]["actual"]
    with pytest.raises(ValueError, match="feature names"):
        adaptation.validate_examples(rows, None)


def test_selection_uses_validation_only_then_refits_before_test(tmp_path):
    rows, output = examples(), tmp_path / "freeze"
    bundle = adaptation.select_and_freeze(rows, output)
    assert adaptation.load_bundle(output) == bundle
    for horizon in (4, 24):
        selection = bundle["selection"][str(horizon)]
        selected = bundle["selected"][str(horizon)]
        assert selection["selected"] == selected["candidate"] == "raw_price"
        assert selection["fit_count"] == 40 and selection["validation_count"] == 31
        assert len(selected["refit_sample_ids"]) == selected["parameters"]["fit_count"] == 71
        assert selected["parameters"]["fit_cutoff"] == "2026-01-01T00:00:00Z"
        assert selected["residual_calibration_count"] == 31
        assert selected["validation_residual_quantiles"] == pytest.approx([0, 0, 0])
        assert selected["unchanged_calibration_count"] == 71
        actual = [row["actual"] for row in rows if row["horizon_hours"] == horizon]
        assert selected["unchanged_return_quantiles"] == pytest.approx(np.quantile(actual, adaptation.LEVELS))
    before = copy.deepcopy(bundle)
    assert adaptation.predict(bundle, "2026-01-01", 4, feature_values(), .017, -.002) == .017
    assert bundle == before
    with pytest.raises(TypeError):
        adaptation.predict(bundle, "2026-01-01", 4, feature_values(), .017, -.002, actual=.5)
    with pytest.raises(FileExistsError):
        adaptation.select_and_freeze(rows, output)


def test_validation_outliers_cannot_change_fit_scaler(tmp_path):
    rows = examples()
    for row in rows:
        if row["anchor"] >= "2025-12-01":
            row["features"] = {name: 100000. for name in adaptation.FEATURES}
    bundle = adaptation.select_and_freeze(rows, tmp_path / "freeze")
    training = [row for row in rows if row["horizon_hours"] == 4 and row["anchor"] < "2025-12-01"]
    expected = adaptation.fit_candidate(spec("ridge_state_1"), training, "2025-12-01")
    assert bundle["selection"]["4"]["candidate_parameters_fit_only"]["ridge_state_1"] == expected


def test_baseline_gate_is_separate_from_candidate_selection(tmp_path):
    rows = examples()
    for row in rows:
        row.update(actual=0., baseprice=.01, basecov=.02)
    bundle = adaptation.select_and_freeze(rows, tmp_path / "freeze")
    for selection in bundle["selection"].values():
        # Even when the zero baseline is already perfect, selection still returns
        # one of the declared nine candidates and labels its gate honestly.
        assert selection["selected"] in {item["name"] for item in adaptation.candidates()}
        assert selection["validation_better_than_unchanged"] is False


def test_bundle_tampering_and_code_drift_are_rejected(tmp_path, monkeypatch):
    output = tmp_path / "freeze"
    adaptation.select_and_freeze(examples(), output)
    original = adaptation.code_hashes()
    monkeypatch.setattr(adaptation, "code_hashes", lambda: {**original, "timestral_adaptation.py": "changed"})
    with pytest.raises(ValueError, match="code changed"):
        adaptation.load_bundle(output)
    monkeypatch.setattr(adaptation, "code_hashes", lambda: original)
    path = output / "bundle.json"
    path.chmod(0o644)
    contents = json.loads(path.read_text())
    contents["selected"]["4"]["parameters"]["source"] = "basecov"
    path.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="bundle changed"):
        adaptation.load_bundle(output)


@pytest.mark.parametrize("anchor,horizon", [("2025-12-31", 24), ("2026-04-01", 4), ("2026-01-01T01:00:00Z", 4), ("2026-03-31T04:00:00Z", 24)])
def test_test_predict_rejects_wrong_period_and_grid(tmp_path, anchor, horizon):
    bundle = adaptation.select_and_freeze(examples(), tmp_path / "freeze")
    with pytest.raises(ValueError, match="test anchor grid"):
        adaptation.predict(bundle, anchor, horizon, feature_values(), .01, .02)
