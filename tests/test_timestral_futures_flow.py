"""Aggressor semantics, delayed availability, gaps and directional symmetry."""

import numpy as np
import pytest

from research import timestral_futures_flow as model
from scripts.run_futures_flow import decode, epoch, url


def inputs():
    t = epoch(2024, 1) + np.arange(400) * 60
    source = {
        s: {
            "time": t.copy(),
            "volume": np.arange(400, dtype=float) + 1,
            "taker": (np.arange(400) + 1) * (0.5 + np.sin(np.arange(400) / 20) * 0.4),
            "trades": np.arange(400) + 2,
        }
        for s in model.shared.SYMBOLS
    }
    return t, source


def test_flow_is_volume_weighted_and_activity_excludes_current_bar():
    t, data = inputs()
    x = np.ones((3, 1, 23))
    valid, f = model.futures_features(data, t[[200]], x)
    assert valid.all()
    v, b = data["BTCUSDT"]["volume"], data["BTCUSDT"]["taker"]
    assert f[0, 0, 0] == pytest.approx(np.log(v[198] / v[78:198].mean()))
    for column, horizon in zip(range(4, 8), (1, 5, 30, 120)):
        sl = slice(199 - horizon, 199)
        assert f[0, 0, column] == pytest.approx(2 * b[sl].sum() / v[sl].sum() - 1)
    assert abs(f[0, 0, 7] - np.mean(2 * b[79:199] / v[79:199] - 1)) > 0.01


def test_latest_closed_bar_and_future_cannot_change_features():
    t, data = inputs()
    x = np.ones((3, 1, 23))
    _, original = model.futures_features(data, t[[200]], x)
    for d in data.values():
        for k in ("volume", "taker", "trades"):
            d[k][199:] *= 10
    _, actual = model.futures_features(data, t[[200]], x)
    np.testing.assert_array_equal(original, actual)


def test_gap_excludes_all_three_assets_until_full_context_recovers():
    t, data = inputs()
    x = np.ones((3, 3, 23))
    anchors = t[[121, 200, 300]]
    valid, before = model.futures_features(data, anchors, x)
    np.testing.assert_array_equal(valid, [False, True, True])
    for k in data["BTCUSDT"]:
        data["BTCUSDT"][k] = np.delete(data["BTCUSDT"][k], 150)
    valid, after = model.futures_features(data, anchors, x)
    np.testing.assert_array_equal(valid, [False, False, True])
    np.testing.assert_allclose(after[:, 0], before[:, 1])


def test_zero_volume_is_finite_with_zero_flow():
    t, data = inputs()
    for d in data.values():
        for k in ("volume", "taker", "trades"):
            d[k][:] = 0
    _, f = model.futures_features(data, t[[200]], np.ones((3, 1, 23)))
    assert np.isfinite(f).all()
    np.testing.assert_array_equal(f[:, :, 4:], 0)


def test_price_and_aggressor_reflection_negates_every_added_feature():
    t, data = inputs()
    x = np.random.default_rng(42).normal(size=(3, 1, 23))
    _, f = model.futures_features(data, t[[200]], x)
    for d in data.values():
        d["taker"] = d["volume"] - d["taker"]
    mirrored = x.copy()
    mirrored[:, :, list(model.shared.ODD)] *= -1
    _, reflected = model.futures_features(data, t[[200]], mirrored)
    np.testing.assert_allclose(reflected, -f, atol=1e-12)
    # Futures aggressor volume is excluded from the volume-only arm.
    _, buy_only_changed = model.futures_features(data, t[[200]], x)
    np.testing.assert_array_equal(buy_only_changed[:, :, :4], f[:, :, :4])


def test_raw_volume_fields_and_protected_months():
    t = epoch(2024, 1) * 1000
    row = [
        str(t),
        "10",
        "12",
        "9",
        "11",
        "100",
        str(t + 59999),
        "1100",
        "30",
        "65",
        "715",
        "0",
    ]
    assert decode(row, 2024, 1) == (t // 1000, 100, 65, 30)
    row[9] = "101"
    with pytest.raises(ValueError):
        decode(row, 2024, 1)
    with pytest.raises(ValueError):
        url("BTCUSDT", 2026, 8)
    t, data = inputs()
    data["BTCUSDT"]["time"][-1] = model.shared.BOUNDARY
    with pytest.raises(ValueError):
        model.futures_features(data, t[[200]], np.ones((3, 1, 23)))
