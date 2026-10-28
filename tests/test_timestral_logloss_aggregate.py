import numpy as np
import pytest

from research.timestral_logloss_aggregate import experts, replay, evidence, BOUNDARY


def sample(n=100):
    rng = np.random.default_rng(17)
    common, uniform = rng.uniform(0.3, 0.7, (2, n))
    flat, frequency = rng.uniform(0.01, 0.1, (2, n))
    p = experts(common, uniform, flat, frequency)
    y = rng.integers(0, 3, n)
    t = 1704067200 + np.arange(n) * 300
    return p, y, t, t + 360


def test_delayed_feedback_and_future_predictions_cannot_leak():
    p, y, t, end = sample()
    before, w = replay(p, y, t, end)
    y2 = y.copy()
    y2[10] = (y2[10] + 1) % 3
    after, w2 = replay(p, y2, t, end)
    np.testing.assert_array_equal(before[:12], after[:12])
    assert not np.array_equal(w[12], w2[12])
    changed = p.copy()
    changed[40:] = changed[40:, :, ::-1]
    future, _ = replay(changed, y, t, end)
    np.testing.assert_array_equal(before[:40], future[:40])
    np.testing.assert_array_equal(w[:2], np.full((2, 4), 0.25))


def test_factorization_reflection_and_finite_sequence_bound():
    p, y, t, end = sample(1000)
    a, w = replay(p, y, t, end)
    matrix = w.reshape(-1, 2, 2)
    np.testing.assert_allclose(
        matrix, matrix.sum(2)[:, :, None] * matrix.sum(1)[:, None, :], atol=1e-12
    )
    reflected, _ = replay(p[:, :, ::-1], 2 - y, t, end)
    np.testing.assert_allclose(reflected, a[:, ::-1], atol=1e-12)
    e = evidence(p, y, a)
    assert e["regret"] <= e["regret_bound"] + 1e-9
    assert e["direction_regret"] <= e["component_bound"] + 1e-9
    assert e["flat_regret"] <= e["component_bound"] + 1e-9
    assert abs(e["telescoping_gap"]) < 1e-9


def test_flat_label_does_not_update_directional_weights():
    p, y, t, end = sample(4)
    y[0] = 1
    _, w = replay(p, y, t, end)
    np.testing.assert_allclose(w[2].reshape(2, 2).sum(1), [0.5, 0.5], atol=1e-14)


def test_rejects_immature_spacing_and_protected_targets():
    p, y, t, end = sample()
    with pytest.raises(ValueError):
        replay(p, y, t, t + 700)
    with pytest.raises(ValueError):
        replay(p, y, t // 2, t // 2 + 360)
    t = BOUNDARY - 300 + np.arange(len(y)) * 300
    with pytest.raises(ValueError):
        replay(p, y, t, t + 360)
