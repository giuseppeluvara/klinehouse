"""Premium timing, source semantics and conditional direction isolation."""

import numpy as np
import pytest

from research import timestral_premium_direction as model
from research import timestral_premium_direction_gaps as gaps
from scripts.run_premium_direction import decode, epoch, url


def test_premium_features_exclude_latest_minute_and_future():
    t = epoch(2024, 1) + np.arange(400) * 60
    sources = {
        s: {"time": t, "close": np.sin(np.arange(400) / 20) * 0.0001}
        for s in model.shared.SYMBOLS
    }
    anchor = np.array([t[200]])
    x = np.zeros((3, 1, 23))
    _, first = model.premium_features(sources, anchor, x)
    for d in sources.values():
        d["close"][t >= anchor[0] - 60] = 0.0009
    valid, second = model.premium_features(sources, anchor, x)
    assert valid.all()
    np.testing.assert_array_equal(first, second)
    # The last admitted candle opens two minutes before the anchor.
    assert first[0, 0, 0] == pytest.approx(np.sin(198 / 20))


def test_premium_gap_and_protected_time_fail_closed():
    t = epoch(2024, 1) + np.arange(400) * 60
    data = {s: {"time": t.copy(), "close": np.zeros(400)} for s in model.shared.SYMBOLS}
    data[model.shared.SYMBOLS[0]]["time"][300] += 60
    with pytest.raises(ValueError):
        model.premium_features(data, np.array([t[200]]), np.zeros((3, 1, 23)))
    with pytest.raises(ValueError):
        url("BTCUSDT", 2026, 8)


def test_premium_can_be_negative_and_ignored_fields_are_not_volume():
    t = epoch(2024, 1) * 1000
    row = [
        str(t),
        "-0.0003",
        "-0.0001",
        "-0.0004",
        "-0.0002",
        "0",
        str(t + 59999),
        "0",
        "12",
        "0",
        "0",
        "0",
    ]
    assert decode(row, 2024, 1) == (t // 1000, -0.0002)
    row[0] = str(t * 1000)
    with pytest.raises(ValueError):
        decode(row, 2024, 1)


def test_new_information_can_predict_direction_without_changing_flat():
    rng = np.random.default_rng(473)
    x = rng.normal(size=(2500, 35))
    y = (x[:, 23] + rng.normal(size=len(x)) * 0.1 > 0).astype(int)
    cutoff = epoch(2024, 1)
    fit = model.fit(x, y, np.full(len(y), cutoff), cutoff)
    held = rng.normal(size=(500, 35))
    p = model.predict(fit, held, np.full(500, cutoff))
    assert ((p > 0.5) == (held[:, 23] > 0)).mean() > 0.95
    mirrored = held.copy()
    mirrored[:, model.ODD] *= -1
    np.testing.assert_allclose(
        model.predict(fit, mirrored, np.full(500, cutoff)), 1 - p, atol=1e-5
    )
    flat = np.linspace(0.01, 0.2, 500)
    combined = model.combine(p, flat)
    np.testing.assert_array_equal(combined[:, 1], flat)
    np.testing.assert_allclose(combined.sum(1), 1)
    with pytest.raises(ValueError):
        model.fit(x, y, np.full(len(y), cutoff + 1), cutoff)


def test_conditional_metric_keeps_flat_count_separate():
    t = epoch(2024, 1) + np.arange(35 * 288) * 300
    y = np.resize(np.array([0, 1, 2]), len(t))
    p = np.where(y == 2, 0.6, 0.4)
    result = model.conditional_metrics(
        y, p, np.full(len(t), 0.5), np.full(len(t), 0.5), t
    )
    assert result["n"] + result["flats"] == len(t)
    assert result["accuracy"] == 1
    assert result["brier_gain_vs_spot"] > 0


def test_missing_premium_context_is_marked_without_filling():
    t = epoch(2024, 1) + np.arange(400) * 60
    source = {
        s: {"time": t.copy(), "close": np.sin(np.arange(400) / 20) * 0.0001}
        for s in model.shared.SYMBOLS
    }
    anchor = t[[200, 300]]
    x = np.zeros((3, 2, 23))
    _, original = model.premium_features(source, anchor, x)
    data = source[model.shared.SYMBOLS[0]]
    for k in data:
        data[k] = np.delete(data[k], 150)
    valid, remaining = gaps.premium_features(source, anchor, x)
    np.testing.assert_array_equal(valid, [False, True])
    np.testing.assert_allclose(remaining[:, 0], original[:, 1])


def test_gap_safe_context_still_ignores_future_values():
    t = epoch(2024, 1) + np.arange(400) * 60
    d = {s: {"time": t, "close": np.ones(400) * 0.0001} for s in model.shared.SYMBOLS}
    a = np.array([t[200]])
    x = np.zeros((3, 1, 23))
    _, first = gaps.premium_features(d, a, x)
    for source in d.values():
        source["close"][199:] *= 4
    _, second = gaps.premium_features(d, a, x)
    np.testing.assert_array_equal(first, second)
