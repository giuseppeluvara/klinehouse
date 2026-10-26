"""Settlement availability and nonlinear signal controls."""

from datetime import timedelta
import copy

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from research import timestral_funding as funding
from research.timestral_experiment import utc


def settlements():
    start = utc("2025-12-31")
    return [
        {
            "time": (start + timedelta(hours=8 * i, milliseconds=1)).isoformat(),
            "interval_hours": 8.0,
            "rate": 0.0001 * (i + 1),
        }
        for i in range(8)
    ]


def test_funding_availability_respects_settlement_lag_and_milliseconds():
    rows = settlements()
    anchor = utc("2026-01-02T01:00:00Z")
    value, latest = funding.funding_features(rows, anchor)
    assert latest == rows[5]["time"]
    assert value[0] == pytest.approx(rows[5]["rate"] / 8 * 10000)
    assert (
        funding.funding_features(rows, anchor + timedelta(milliseconds=1))[1]
        == rows[6]["time"]
    )
    changed = copy.deepcopy(rows)
    for r in changed[6:]:
        r["rate"] = 1e6
    assert funding.funding_features(changed, anchor) == (value, latest)


def test_stale_and_incomplete_funding_are_not_silently_carried_forward():
    rows = settlements()
    with pytest.raises(ValueError, match="stale"):
        funding.funding_features(rows, utc("2026-01-05"))
    with pytest.raises(ValueError, match="insufficient"):
        funding.funding_features(rows[:2], utc("2026-01-01"))
    with pytest.raises(ValueError, match="gap"):
        funding.funding_features([*rows[:4], rows[5]], utc("2026-01-02T01:00:00Z"))


def examples():
    start = utc("2025-10-01")
    rows = []
    for i in range(90):
        signal = 1 if i % 2 else -1
        rows.append(
            {
                "sample_id": str(i),
                "anchor": (start + timedelta(days=i)).isoformat(),
                "label_end": (start + timedelta(days=i + 1)).isoformat(),
                "horizon_hours": 24,
                "features": [0.0] * 7 + [signal, signal],
                "actual": signal * 0.01,
                "scale": 0.01,
            }
        )
    return rows, start + timedelta(days=90)


def test_added_information_can_predict_a_real_control_signal():
    rows, cutoff = examples()
    with threadpool_limits(2):
        state = funding.fit(rows, cutoff, 24, "state")
        model = funding.fit(rows, cutoff, 24, "state_timestral_funding")
        state_error = np.mean(
            [
                abs(funding.predict(state, r["features"], r["scale"]) - r["actual"])
                for r in rows
            ]
        )
        model_error = np.mean(
            [
                abs(funding.predict(model, r["features"], r["scale"]) - r["actual"])
                for r in rows
            ]
        )
    assert model_error < 0.2 * state_error


def test_future_outcomes_and_overlapping_targets_are_rejected():
    rows, cutoff = examples()
    with pytest.raises(ValueError, match="matured"):
        funding.fit(rows, cutoff - timedelta(hours=1), 24, "state")
    rows[1]["anchor"] = rows[0]["anchor"]
    with pytest.raises(ValueError, match="target intervals"):
        funding.fit(rows, cutoff, 24, "state")
