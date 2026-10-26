"""Ensure the overlap ablation changes cadence, not target or past boundary."""

from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from research import timestral_cadence as cadence
from research.timestral_experiment import utc
from research.timestral_long_horizon import Hour
from tests.test_timestral_repair import ArrayTensor


@pytest.mark.parametrize("h", [4, 24])
def test_same_past_window_disjoint_is_exact_subsequence_and_target_matches(h):
    anchor = utc("2026-01-01")
    n = 64 * h + 1
    values = 10 + np.arange(n + h) * .0002
    hours = [Hour(anchor - timedelta(hours=n-i), float(v), (0., 0., 0.)) for i, v in enumerate(values[:n])]
    inputs = cadence.horizon_inputs(hours, anchor, h)
    dense, dense_steps = inputs["dense"]
    disjoint, disjoint_steps = inputs["disjoint"]
    assert len(dense) == 63*h + 1 and len(disjoint) == 64
    assert dense_steps == h and disjoint_steps == 1
    np.testing.assert_array_equal(dense[::h], disjoint)
    # Both model endpoints correspond to the full h new increments.
    actual = values[n+h-1] - values[n-1]
    assert actual == pytest.approx(h * .0002)
    np.testing.assert_allclose(disjoint / 10000, actual)
    with pytest.raises(ValueError, match="before anchor"):
        cadence.horizon_inputs(hours, anchor + timedelta(hours=1), h)


def test_forecasts_use_last_step_and_convert_units_once():
    h, anchor = 24, utc("2026-01-01")
    n = 64*h + 1
    hours = [Hour(anchor - timedelta(hours=n-i), 10+i*.0002, (0., 0., 0.)) for i in range(n)]
    calls = []
    def forecast(series, **kwargs):
        steps = kwargs["prediction_length"]
        calls.append((series.shape[-1], steps))
        q = np.stack([np.arange(steps)-100, np.arange(steps)+10, np.arange(steps)+100], axis=-1)
        return SimpleNamespace(quantile_levels=[.05, .5, .95], quantiles=ArrayTensor(q[None, None]))
    model = SimpleNamespace(forecaster=SimpleNamespace(forecast=forecast))
    result = cadence.forecast(model, hours, anchor, h)
    assert calls == [(1513, 24), (64, 1)]
    assert result["dense"]["point"] == .0033
    assert result["disjoint"]["point"] == .001


def test_paired_bootstrap_uses_matching_outcomes_and_loss_direction():
    start = utc("2026-01-01")
    rows = [{"anchor": (start + timedelta(days=i)).isoformat(), "actual": .01,
             "predictions": {"disjoint": {"point": .01}, "dense": {"point": -.01}}} for i in range(70)]
    result = cadence.paired_mae_interval(rows, "disjoint", "dense")
    assert result["mae_difference"] == pytest.approx(-.02)
    np.testing.assert_allclose(result["interval98_75"], [-.02, -.02])
