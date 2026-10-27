"""Causality, known-signal recovery, target identity and honest screening."""

from datetime import timedelta
import copy
import math

import numpy as np
import pytest

from research import timestral_decomposition as d
from research.timestral_experiment import utc, stamp


def candles(n=1550):
    start = utc("2023-01-01")
    return [
        {
            "time": stamp(start + timedelta(hours=i)),
            "close": math.exp(9 + 0.0001 * i),
            "high": math.exp(9 + 0.0001 * i + 0.001),
            "low": math.exp(9 + 0.0001 * i - 0.001),
            "volume": 100.0,
            "taker": 50.0,
            "trades": 10.0,
        }
        for i in range(n)
    ]


def synthetic(n=1100):
    rng = np.random.default_rng(109)
    start = utc("2021-01-01")
    rows = []
    for i in range(n):
        x = rng.normal(size=11)
        variance = 0.001 * math.exp(0.8 * x[0] - 0.3 * x[1])
        up = rng.random() < 1 / (1 + math.exp(-2 * x[0]))
        anchor = start + timedelta(days=i)
        rows.append(
            {
                "sample_id": str(i),
                "anchor": stamp(anchor),
                "label_end": stamp(anchor + timedelta(days=1)),
                "horizon_hours": 24,
                "state": x.tolist(),
                "variance_features": x[:3].tolist(),
                "return": 0.01 if up else -0.01,
                "variance": variance,
                "variance_persistence": 0.001,
                "variance_ewma": 0.001,
            }
        )
    return rows


def test_exact_variance_target_is_not_squared_terminal_return():
    rows, _ = d.examples(candles(750))
    for r in rows:
        h = r["horizon_hours"]
        assert r["return"] == pytest.approx(h * 0.0001)
        assert r["variance"] == pytest.approx(h * 0.0001**2)
        assert r["variance"] != pytest.approx(r["return"] ** 2, abs=1e-12)
        assert r["variance_persistence"] == pytest.approx(r["variance"])


def test_future_price_poison_cannot_change_origin_features():
    source = candles(750)
    a, _ = d.examples(source)
    changed = copy.deepcopy(source)
    anchor = a[0]["anchor"]
    for row in changed:
        if row["time"] >= anchor:
            for key in ("close", "low", "high"):
                row[key] *= 2
    b, _ = d.examples(changed)
    for key in ("state", "variance_features", "variance_persistence", "variance_ewma"):
        assert a[0][key] == b[0][key]
    assert a[0]["variance"] != b[0]["variance"]
    with pytest.raises(ValueError, match="closed past"):
        d.origin(source[:721], utc(source[721]["time"]) - timedelta(hours=1), 4)


def test_gap_removes_only_affected_windows_and_recovers():
    source = candles()
    full, _ = d.examples(source)
    missing = source[:760] + source[761:]
    partial, audit = d.examples(missing)
    assert audit["missing_context"] > 0 and audit["missing_target"] > 0
    assert partial[-1] == full[-1]
    gap = utc(source[760]["time"])
    assert all(
        not utc(r["anchor"]) - timedelta(hours=721) <= gap < utc(r["label_end"])
        for r in partial
    )


def test_models_recover_injected_direction_and_variance():
    rows = synthetic()
    cutoff = utc(rows[800]["anchor"])
    params = d.fit(rows[:800], cutoff, 24)
    evaluated = [{**r, "predictions": d.predict(params, r)} for r in rows[800:]]
    y = np.array([r["return"] > 0 for r in evaluated])
    prob = np.array([r["predictions"]["direction"] for r in evaluated])
    assert np.mean((prob - y) ** 2) < 0.20
    actual = np.array([r["variance"] for r in evaluated])
    predicted = np.array([r["predictions"]["variance"] for r in evaluated])
    assert d.losses(actual, predicted, "variance")[0].mean() < 0.01
    assert params["direction"]["iterations"] < 1000
    assert params["variance"]["iterations"] < 1000


def test_unmatured_duplicate_and_overlapping_labels_rejected():
    rows = synthetic(800)
    cutoff = utc(rows[-1]["label_end"])
    for bad in [rows + [rows[-1]], rows[:]]:
        if len(bad) == 800:
            bad = copy.deepcopy(bad)
            bad[-1]["label_end"] = stamp(cutoff + timedelta(hours=1))
        with pytest.raises(ValueError):
            d.fit(bad, cutoff, 24)
    bad = copy.deepcopy(rows)
    bad[-1]["anchor"] = stamp(utc(bad[-2]["anchor"]) + timedelta(hours=4))
    bad[-1]["label_end"] = stamp(utc(bad[-1]["anchor"]) + timedelta(hours=24))
    with pytest.raises(ValueError, match="overlapping"):
        d.fit(bad, cutoff, 24)


def test_bad_scoring_values_rejected_and_qlike_zero_when_exact():
    assert d.losses([0.1, 0.2], [0.1, 0.2], "variance")[0].tolist() == [0, 0]
    with pytest.raises(ValueError):
        d.losses([1], [0], "variance")
    with pytest.raises(ValueError):
        d.losses([1], [1.1], "direction")
    with pytest.raises(ValueError):
        d.losses([1], [float("nan")], "direction")


def test_identical_baseline_never_passes_historical_screen():
    rows = synthetic(400)
    for r in rows:
        r["predictions"] = {
            "direction": 0.5,
            "direction_chance": 0.5,
            "direction_prior": 0.5,
            "variance": 0.001,
            "variance_persistence": 0.001,
            "variance_ewma": 0.001,
            "variance_mean": 0.001,
        }
    for kind in ("direction", "variance"):
        result = d.score(rows, kind)
        assert not result["passes_both_baselines"]
        assert not result["trading_validation"]
        assert all(s["gain"] == 0 for s in result["screens"].values())


def test_protected_source_dates_rejected():
    source = candles(2)
    source[-1]["time"] = "2026-08-01T00:00:00Z"
    with pytest.raises(ValueError, match="protected"):
        d.examples(source)
