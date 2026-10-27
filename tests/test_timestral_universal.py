import numpy as np
import pytest

from research import timestral_universal as u


def data():
    rng = np.random.default_rng(31)
    result = {}
    for name in u.SYMBOLS:
        n = 400
        result[name] = {
            "time": 1767225600 + np.arange(n) * 60,
            "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n))),
            "volume": np.full(n, 2.0),
            "taker": rng.uniform(0, 2, n),
            "trades": rng.integers(1, 100, n).astype(float),
        }
    return result


def test_features_have_no_future_or_cross_asset_future_leakage():
    d = data()
    first = u.examples(d)
    for a in d.values():
        a["close"][201:] *= 2
        a["trades"][201:] *= 3
        a["taker"][201:] *= 0.1
    later = u.examples(d)
    old = first["anchor"] <= 1767225600 + 201 * 60
    np.testing.assert_array_equal(first["x"][:, old], later["x"][:, old])


def test_shared_market_context_requires_aligned_inputs():
    d = data()
    d["ETHUSDT"]["time"][100] += 60
    with pytest.raises(ValueError, match="complete"):
        u.examples(d)


def test_protected_sources_and_targets_rejected():
    d = data()
    for a in d.values():
        a["time"] += u.BOUNDARY - a["time"][-1]
    with pytest.raises(ValueError, match="protected"):
        u.examples(d)
    with pytest.raises(ValueError, match="protected"):
        u.predict({"cutoff": 0}, np.zeros((1, 23)), np.array([u.BOUNDARY - 300]))


def test_target_retains_ties_and_excludes_first_future_minute():
    d = data()
    s = u.examples(d)
    index = (s["anchor"][0] - 1767225600) // 60 - 1
    for a in d.values():
        a["close"][index + 1] = 200
        a["close"][index + 6] = 200
    s = u.examples(d)
    assert np.all(s["y"][:, 0] == 1)
    assert s["end"][0] - s["anchor"][0] == 360


def test_augmentation_reflects_only_directional_features():
    x = np.arange(69, dtype=float).reshape(3, 23)
    a, b = u.augment(x, np.array([0, 1, 2]))
    assert np.array_equal(b, [0, 1, 2, 2, 1, 0])
    assert np.array_equal(a[3:, list(u.ODD)], -x[:, list(u.ODD)])
    even = [j for j in range(23) if j not in u.ODD]
    assert np.array_equal(a[3:, even], x[:, even])


def test_classifier_learns_a_known_cross_market_signal_symmetrically():
    rng = np.random.default_rng(2)
    x = np.zeros((1000, 23))
    x[:, 13] = rng.normal(size=1000)
    x[:, 22] = abs(x[:, 13])
    y = np.where(x[:, 13] > 0.25, 2, np.where(x[:, 13] < -0.25, 0, 1))
    f = u.fit(x, y, np.zeros(1000), 10)
    p = u.predict(f, x, np.full(1000, 11))
    assert np.mean(p.argmax(1) == y) > 0.95
    mirror = x.copy()
    mirror[:, 13] *= -1
    q = u.predict(f, mirror, np.full(1000, 11))
    np.testing.assert_allclose(q, p[:, ::-1], atol=1e-7)
    with pytest.raises(ValueError, match="matured"):
        u.fit(x, y, np.full(1000, 11), 10)


def test_abstention_does_not_pass_without_predictions():
    y = np.tile([0, 1, 2], 100)
    p = np.full((300, 3), 1 / 3)
    result = u.evaluate(y, p, p, p, np.arange(300) * 300 + 1767225600)
    assert result["gate"] is False
    assert result["selected"] == 0
