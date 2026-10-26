"""Independent scoring identities and controlled transformations."""

from types import SimpleNamespace

import numpy as np
import pytest

from research import timestral_diagnostics as diagnosis
from tests.test_timestral_repair import ArrayTensor
from scripts import diagnose_timestral as runner


def test_interval_score_penalizes_width_and_misses_with_correct_units():
    # Below, inside and above [-1, 1]: interval scores 22, 2, 22.
    q = [[-1., 0., 1.]] * 3
    result = diagnosis.interval_scores([-2., 0., 2.], q)
    assert result["interval_score90"] == pytest.approx(46 / 3)
    assert result["coverage90"] == pytest.approx(1 / 3)
    assert result["interval_score90"] == pytest.approx(40 * result["tail_pinball"])
    assert result["median_pinball"] == pytest.approx(2 / 3)


def test_median_change_cannot_be_mistaken_for_improved_interval_quality():
    first = diagnosis.interval_scores([.5], [[-1., 0., 1.]])
    second = diagnosis.interval_scores([.5], [[-1., .5, 1.]])
    assert first["all_quantile_pinball"] > second["all_quantile_pinball"]
    assert first["tail_pinball"] == second["tail_pinball"]
    assert first["interval_score90"] == second["interval_score90"]


def test_comparison_exposes_median_mismatch_without_mutating_predictions():
    row = {"actual_log_return": .3, "predictions": {
        "unchanged_static": {"point": 0., "quantiles": [-1., .5, 1.]},
        "unchanged_dynamic": {"point": 0., "quantiles": [-1., 0., 1.]}}}
    result = diagnosis.compare_intervals([row])
    assert result["static_median_minus_point"] == [.5]
    assert result["matched_median_static"] == result["dynamic"]
    assert row["predictions"]["unchanged_static"]["quantiles"][1] == .5
    row["predictions"]["unchanged_dynamic"]["point"] = .1
    with pytest.raises(ValueError, match="point forecasts differ"):
        diagnosis.compare_intervals([row])


@pytest.mark.parametrize("actual,quantiles", [([], []), ([0.], [[1., 0., -1.]]),
                                             ([float('nan')], [[-1., 0., 1.]])])
def test_invalid_scores_fail_explicitly(actual, quantiles):
    with pytest.raises(ValueError):
        diagnosis.interval_scores(actual, quantiles)


def test_affine_probe_centers_before_float32_cast_and_restores_return_units():
    captured = []
    def forecast(values, **kwargs):
        captured.append((values.copy(), kwargs))
        result = np.broadcast_to([-100., 10., 200.], (1, 1, kwargs["prediction_length"], 3))
        return SimpleNamespace(quantile_levels=[.05, .5, .95], quantiles=ArrayTensor(result))
    model = SimpleNamespace(forecaster=SimpleNamespace(forecast=forecast))
    # These sub-ULP changes would disappear if levels were cast before centering.
    series = 10. + np.arange(256) * 1e-9
    prediction = diagnosis.centered_forecast(model, series, 24)
    assert captured[0][0][0, 0, -2] != 0
    assert captured[0][0][0, 0, -1] == 0
    assert captured[0][1] == {"prediction_length": 24, "symmetric_averaging": True, "clip_nonnegative": False}
    assert prediction == {"point": .001, "quantiles": [-.01, .001, .02]}


def test_overlap_is_large_even_when_future_increments_are_independent():
    values = np.cumsum(np.random.default_rng(17).normal(size=200000))
    result = diagnosis.overlap_statistics(values, 24)
    assert result["rolling_lag1_correlation"] == pytest.approx(23 / 24, abs=.015)
    assert result["rolling_lag_h_correlation"] == pytest.approx(0, abs=.03)


def test_synthetic_controls_have_known_endpoints_and_reproducible_noise():
    first, second = dict(diagnosis.synthetic_cases()), dict(diagnosis.synthetic_cases())
    assert len(first) == 135
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert first["linear_up"][279] - first["linear_up"][255] == pytest.approx(.0048)
    assert first["linear_down"][259] - first["linear_down"][255] == pytest.approx(-.0008)
    assert first["constant"][279] == first["constant"][255]


def test_diagnostic_protocol_detects_changes_and_cannot_be_overwritten(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "run").mkdir(parents=True)
    for name in runner.INPUTS:
        (source / name).write_text('{}\n')
    monkeypatch.setattr(runner, "SOURCE", source)
    runner.register(output)
    runner.verify(output)
    with pytest.raises(FileExistsError):
        runner.register(output)
    (source / "hours.jsonl").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="changed"):
        runner.verify(output)
