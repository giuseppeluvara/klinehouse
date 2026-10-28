import numpy as np
import pytest
from scipy.special import logsumexp

from research.timestral_multiscore import forecast, replay as cumulative
from research.timestral_temporal_adaptation import (
    BOUNDARY,
    METHODS,
    PERIOD_SECONDS,
    replay,
    update,
)


def sample(n=100, k=6):
    rng = np.random.default_rng(381)
    expert = rng.dirichlet([1, 0.3, 1], (n, k))
    y = rng.integers(0, 3, n)
    anchor = 1704067200 + 300 * np.arange(n)
    return expert, y, anchor, anchor + 360


@pytest.mark.parametrize("method", METHODS)
def test_only_mature_feedback_and_current_experts_affect_forecasts(method):
    expert, y, anchor, end = sample()
    before, _, diagnostics = replay(expert, y, anchor, end, method)
    changed = y.copy()
    changed[10] = (changed[10] + 1) % 3
    after, _, _ = replay(expert, changed, anchor, end, method)
    np.testing.assert_array_equal(before[:12], after[:12])
    assert not np.array_equal(before[12], after[12])
    changed_expert = expert.copy()
    changed_expert[40:] = changed_expert[40:, :, ::-1]
    future, _, _ = replay(changed_expert, y, anchor, end, method)
    np.testing.assert_array_equal(before[:40], future[:40])
    assert diagnostics["updates"] == len(y) - 2
    assert diagnostics["skipped_feedback"] == diagnostics["resets"] == 0


def test_share_preserves_each_score_mass_and_restores_all_experts():
    logs = np.array([[0.0, -10.0, -30.0], [4.0, -100.0, -600.0]])
    delta = np.array([[0.02, -0.05, 0.1], [-0.3, 0.05, -0.08]])
    elapsed = 7200
    actual = update(logs, delta, elapsed, "share")
    updated = logs + delta
    alpha = -np.expm1(-elapsed / PERIOD_SECONDS)
    raw = np.exp(updated)
    expected = (1 - alpha) * raw + alpha * raw.mean(1, keepdims=True)
    np.testing.assert_allclose(np.exp(actual), expected, atol=1e-13)
    np.testing.assert_allclose(
        logsumexp(actual, axis=1), logsumexp(updated, axis=1), atol=1e-14
    )
    assert np.all(np.exp(actual) >= alpha * raw.mean(1, keepdims=True) * (1 - 1e-13))
    # An old expert can recover even after its relative weight underflows.
    extreme = update(
        np.array([[0.0, -1000.0], [-30.0, -2000.0]]), np.zeros((2, 2)), elapsed, "share"
    )
    assert np.isfinite(extreme).all()


def test_discount_uses_wall_time_and_ages_newly_matured_feedback():
    logs = np.array([[1.0, 2.0], [3.0, 4.0]])
    delta = np.array([[0.3, -0.2], [0.1, -0.4]])
    np.testing.assert_array_equal(
        update(logs, delta, PERIOD_SECONDS, "discount"), 0.5 * (logs + delta)
    )
    np.testing.assert_array_equal(
        update(logs, delta, 2 * PERIOD_SECONDS, "discount"), 0.25 * (logs + delta)
    )
    first = update(logs, delta, PERIOD_SECONDS, "discount")
    second = update(first, delta, PERIOD_SECONDS, "discount")
    np.testing.assert_allclose(second, 0.25 * logs + 0.75 * delta, atol=1e-15)


def test_monthly_resets_both_lanes_and_discards_old_month_feedback():
    expert, y, _, _ = sample(8)
    boundary = int(np.datetime64("2024-02-01T00:00:00", "s").astype(np.int64))
    anchor = boundary - 600 + 300 * np.arange(len(y))
    end = anchor + 360
    actual, _, diagnostics = replay(expert, y, anchor, end, "monthly")
    changed = y.copy()
    changed[:2] = (changed[:2] + 1) % 3
    ignored, _, _ = replay(expert, changed, anchor, end, "monthly")
    np.testing.assert_array_equal(actual, ignored)
    weights = np.full((2, expert.shape[1]), 1 / (2 * expert.shape[1]))
    for i in (0, 1, 2, 3):
        expected, _, _, _, _ = forecast(expert[i], weights)
        np.testing.assert_array_equal(actual[i], expected)
    assert diagnostics["resets"] == diagnostics["skipped_feedback"] == 2
    assert diagnostics["updates"] == 4
    # In a single month the rule is exactly the previously frozen cumulative learner.
    expert, y, anchor, end = sample()
    monthly, _, _ = replay(expert, y, anchor, end, "monthly")
    old, _, _ = cumulative(expert, y, anchor, end)
    np.testing.assert_array_equal(monthly, old)


@pytest.mark.parametrize("method", METHODS)
def test_reflection_validity_and_invalid_or_protected_inputs(method):
    expert, y, anchor, end = sample()
    actual, slack, diagnostics = replay(expert, y, anchor, end, method)
    reflected, _, _ = replay(expert[:, :, ::-1], 2 - y, anchor, end, method)
    np.testing.assert_allclose(reflected, actual[:, ::-1], atol=2e-10)
    np.testing.assert_allclose(actual.sum(1), 1, atol=1e-14)
    assert np.all(actual > 0) and np.all(slack >= 0)
    assert diagnostics["maximum_potential_ratio"] <= 1 + 1e-10
    for bad_expert, bad_anchor, bad_end in (
        (expert * 0.9, anchor, end),
        (expert, anchor // 2, anchor // 2 + 360),
        (expert, anchor, end + 1),
    ):
        with pytest.raises(ValueError):
            replay(bad_expert, y, bad_anchor, bad_end, method)
    future = BOUNDARY - 300 + 300 * np.arange(len(y))
    with pytest.raises(ValueError):
        replay(expert, y, future, future + 360, method)
