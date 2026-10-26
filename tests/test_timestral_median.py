"""Maturity, loss alignment and incremental model information."""

from datetime import timedelta
import copy

import numpy as np
import pytest

from research import timestral_median as median
from research.timestral_experiment import utc
from research.timestral_long_horizon import Hour


def examples():
    start = utc("2025-10-01")
    rows = []
    for i in range(90):
        signal = np.sin(i * 1.37)
        rows.append(
            {
                "sample_id": str(i),
                "anchor": (start + timedelta(days=i)).isoformat(),
                "label_end": (start + timedelta(days=i + 1)).isoformat(),
                "horizon_hours": 24,
                "scale": 0.01,
                "features": [0.0, 0.0, 0.0, 0.0, 0.0, signal, 0.0],
                "actual": 0.01 * signal,
            }
        )
    return rows, start + timedelta(days=90)


def test_model_feature_contributes_only_when_it_contains_real_signal():
    rows, cutoff = examples()
    state = median.fit(rows, cutoff, 24, False)
    model = median.fit(rows, cutoff, 24, True)
    state_error = np.mean(
        [
            abs(median.predict(state, r["features"], r["scale"]) - r["actual"])
            for r in rows
        ]
    )
    model_error = np.mean(
        [
            abs(median.predict(model, r["features"], r["scale"]) - r["actual"])
            for r in rows
        ]
    )
    assert model_error < state_error * 0.1
    assert model["objective"] <= model["zero_objective"]
    assert median.predict(model, [0, 0, 0, 0, 0, 0.7, 0], 0.02) == pytest.approx(
        0.014, abs=0.001
    )


def test_absolute_error_fit_resists_a_single_large_outcome():
    rows, cutoff = examples()
    for r in rows:
        r["features"] = [0.0] * 7
        r["actual"] = 0.0
    rows[-1]["actual"] = 1.0
    fitted = median.fit(rows, cutoff, 24, False)
    assert median.predict(fitted, [0.0] * 7, 0.01) == pytest.approx(0, abs=1e-12)


def test_future_labels_duplicates_and_short_history_are_rejected():
    rows, cutoff = examples()
    with pytest.raises(ValueError, match="matured"):
        median.fit(rows, cutoff - timedelta(hours=1), 24, True)
    with pytest.raises(ValueError, match="duplicate"):
        median.fit([*rows, rows[-1]], cutoff, 24, True)
    with pytest.raises(ValueError, match="insufficient"):
        median.fit(rows[-10:], cutoff, 24, True)
    bad = copy.deepcopy(rows)
    bad[0]["scale"] = 0.0
    with pytest.raises(ValueError, match="invalid training"):
        median.fit(bad, cutoff, 24, True)


def test_features_use_only_past_and_fixed_units():
    anchor = utc("2026-01-01")
    hours = [
        Hour(anchor - timedelta(hours=256 - i), 10 + i * 0.001, (0.01, 2.0, 0.2))
        for i in range(256)
    ]
    predictions = {
        "timestral_price": {"point": 0.002},
        "timestral_returns": {"point": 0.003},
    }
    values = median.feature_vector(hours, anchor, 0.01, predictions)
    np.testing.assert_allclose(values, [0.4, 2.4, 16.8, 0.2, 0.0, 0.2, 0.3])
    with pytest.raises(ValueError, match="precede anchor"):
        median.feature_vector(hours, anchor - timedelta(hours=1), 0.01, predictions)
