import numpy as np
import pytest

from research.timestral_volume_center import centers, signals, SYMBOLS, BOUNDARY


def source():
    t = np.arange(180) * 60 + 1704067200
    price = np.exp(np.arange(180) * 0.001)
    return {
        s: dict(time=t.copy(), close=price.copy(), volume=np.ones(180)) for s in SYMBOLS
    }


def test_volume_changes_path_center_with_identical_prices():
    d = source()
    t = np.array([d[SYMBOLS[0]]["time"][100] + 60])
    a = centers(d, t)
    np.testing.assert_allclose(a["volume"], a["uniform"])
    for row in d.values():
        row["volume"][96:101] = 100
    b = centers(d, t)
    np.testing.assert_allclose(a["uniform"], b["uniform"])
    assert abs(b["volume"][0]) < abs(a["volume"][0])
    for row in d.values():
        row["volume"] *= 17
    np.testing.assert_allclose(centers(d, t)["volume"], b["volume"], atol=1e-12)


def test_future_values_cannot_change_past_features_and_reflection():
    d = source()
    t = np.array([d[SYMBOLS[0]]["time"][100] + 60])
    before = centers(d, t)
    for row in d.values():
        row["close"][101:] *= 100
        row["volume"][101:] *= 100
    after = centers(d, t)
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])
    for row in d.values():
        row["close"] = 1 / row["close"]
    mirrored = centers(d, t)
    for name in before:
        np.testing.assert_allclose(mirrored[name], -before[name], atol=1e-12)


def test_zero_volume_fallback_and_protected_rejection():
    d = source()
    t = np.array([d[SYMBOLS[0]]["time"][100] + 60])
    for row in d.values():
        row["volume"][:] = 0
    result = centers(d, t)
    np.testing.assert_array_equal(result["volume"], result["uniform"])
    d[SYMBOLS[0]]["time"][-1] = BOUNDARY
    with pytest.raises(ValueError):
        centers(d, t)


def test_large_old_volume_cannot_erase_small_current_window():
    d = source()
    t = np.array([d[SYMBOLS[0]]["time"][100] + 60])
    expected = centers(d, t)
    for row in d.values():
        row["volume"][:10] = 1e20
    actual = centers(d, t)
    np.testing.assert_array_equal(actual["volume"], expected["volume"])


def test_fixed_agreement_routing_and_sign_reflection():
    x = np.zeros((3, 3, 23))
    x[..., 3] = [1, -1, 0]
    x[..., 16] = 1
    c = {"volume": np.array([0.2, 0.4, 0.6])}
    result, agree = signals(x, c)
    np.testing.assert_array_equal(agree[0], [True, False, False])
    np.testing.assert_allclose(result["volume"][0], [0.2, -1 / 3, -2 / 3])
    reflected, same = signals(-x, {"volume": -c["volume"]})
    np.testing.assert_array_equal(agree, same)
    np.testing.assert_allclose(reflected["volume"], -result["volume"])
