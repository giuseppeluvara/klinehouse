import numpy as np
import pytest

from research.timestral_multiscore import BOUNDARY, forecast, potential, replay


def sample(n=200, k=6):
    rng = np.random.default_rng(38)
    p = rng.dirichlet([1, 0.3, 1], (n, k))
    y = rng.integers(0, 3, n)
    t = 1704067200 + 300 * np.arange(n)
    return p, y, t, t + 360


def test_delayed_feedback_and_expert_future_invariance():
    p, y, t, end = sample()
    a, _, _ = replay(p, y, t, end)
    changed = y.copy()
    changed[10] = (changed[10] + 1) % 3
    b, _, _ = replay(p, changed, t, end)
    np.testing.assert_array_equal(a[:12], b[:12])
    assert not np.array_equal(a[12], b[12])
    p[40:] = p[40:, :, ::-1]
    future, _, _ = replay(p, y, t, end)
    np.testing.assert_array_equal(a[:40], future[:40])


def test_all_outcomes_and_both_cumulative_regret_bounds():
    p, y, t, end = sample(1000)
    a, slack, diagnostics = replay(p, y, t, end)
    np.testing.assert_allclose(a.sum(1), 1, atol=1e-14)
    assert np.all(a > 0)
    n, k, _ = p.shape
    logloss = -np.log(p[np.arange(n)[:, None], np.arange(k), y[:, None]])
    brier = (p**2).sum(2) + 1 - 2 * p[np.arange(n)[:, None], np.arange(k), y[:, None]]
    own_log = -np.log(a[np.arange(n), y])
    own_brier = (a**2).sum(1) + 1 - 2 * a[np.arange(n), y]
    bound = 2 * np.log(2 * k) + slack.sum()
    assert own_log.sum() - logloss.sum(0).min() <= bound + 1e-9
    assert own_brier.sum() - brier.sum(0).min() <= bound + 1e-9
    for lane in (0, 1):
        idx = np.arange(lane, n, 2)
        logs = np.stack(
            (own_log[idx, None] - logloss[idx], own_brier[idx, None] - brier[idx]),
            axis=1,
        )
        prefixes = np.concatenate((np.zeros((1, 2, k)), logs.cumsum(0)[:-1]), axis=0)
        for j, log_weights in zip(idx, prefixes):
            w = np.exp(log_weights - log_weights.max())
            w /= w.sum()
            pa = w[0] @ p[j]
            pb = w[1] @ np.exp(-(p[j] ** 2).sum(1)[:, None] - 1 + 2 * p[j])
            assert potential(a[j], pa, pb).max() <= np.exp(slack[j]) + 1e-11
    assert diagnostics["maximum_potential_ratio"] <= 1 + 1e-10


def test_reflection_identical_experts_and_degenerate_metric_weights():
    p, y, t, end = sample()
    a, _, _ = replay(p, y, t, end)
    reflected, _, _ = replay(p[:, :, ::-1], 2 - y, t, end)
    np.testing.assert_allclose(reflected, a[:, ::-1], atol=2e-10)
    identical = np.broadcast_to(p[:, :1], p.shape).copy()
    same, _, _ = replay(identical, y, t, end)
    np.testing.assert_allclose(same, p[:, 0], atol=2e-12)
    experts = np.array([[1 - 2e-8, 1e-8, 1e-8], [1e-8, 1 - 2e-8, 1e-8]])
    weights = np.array([[0.0, 0.0], [0.5, 0.5]])
    result, slack, _, _, ratio = forecast(experts, weights)
    pa = weights[0] @ experts
    pb = weights[1] @ np.exp(-(experts**2).sum(1)[:, None] - 1 + 2 * experts)
    assert potential(result, pa, pb).max() <= np.exp(slack) + 1e-12
    assert potential(result, pa, pb).max() == ratio


def test_rejects_invalid_or_protected_inputs():
    p, y, t, end = sample()
    with pytest.raises(ValueError):
        replay(p, y, t, end + 1)
    with pytest.raises(ValueError):
        replay(p, y, t // 2, t // 2 + 360)
    with pytest.raises(ValueError):
        replay(p * 0.9, y, t, end)
    future = BOUNDARY - 300 + np.arange(len(y)) * 300
    with pytest.raises(ValueError):
        replay(p, y, future, future + 360)


def test_near_vertex_experts_and_extreme_potential_weights():
    rng = np.random.default_rng(922)
    for _ in range(128):
        expert = np.maximum(rng.dirichlet([0.05, 0.05, 0.05], 6), 1e-12)
        expert /= expert.sum(1, keepdims=True)
        logs = rng.uniform(-800, 0, (2, 6))
        weights = np.exp(logs - logs.max())
        weights /= weights.sum()
        prediction, slack, _, _, ratio = forecast(expert, weights)
        assert np.isfinite(prediction).all() and np.all(prediction > 0)
        assert abs(prediction.sum() - 1) < 1e-12
        assert ratio <= 1 + 1e-10
        assert slack == max(0, np.log(ratio))
