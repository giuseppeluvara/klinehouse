import numpy as np
import pytest

from research.direction_orientation import replay


def sample(n=20):
    p = np.tile([0.35, 0.05, 0.60], (n, 1))
    issue = np.arange(n) * 300.0 + 10
    return p, issue, np.arange(n) * 300.0 + 360, np.full(n, 2)


def test_maturity_and_future_prefix_invariance():
    p, issue, at, y = sample()
    result = replay(p, issue, at, y)
    np.testing.assert_array_equal(result["feedback_count"][:4], [0, 0, 1, 1])
    np.testing.assert_allclose(result["weights"][:2], 1 / 3)
    changed = y.copy()
    changed[10:] = 0
    other = replay(p, issue, at, changed)
    np.testing.assert_array_equal(result["adaptive"][:12], other["adaptive"][:12])
    short = replay(p[:10], issue[:10], at[:10], y[:10])
    np.testing.assert_array_equal(result["adaptive"][:10], short["adaptive"])


def test_actual_late_receipt_is_retained_but_not_used_early():
    p, issue, at, y = sample(8)
    at[0] = issue[6]
    result = replay(p, issue, at, y)
    np.testing.assert_array_equal(result["feedback_count"][::2], [0, 0, 1, 3])
    # At index 4 only index 2 is observed; the late index 0 enters at index 6.
    assert result["weights"][2, 0] == pytest.approx(1 / 3)
    assert result["weights"][6, 0] > result["weights"][4, 0]


def test_reflection_symmetry_and_flat_invariance():
    p, issue, at, y = sample()
    result = replay(p, issue, at, y)
    reflected = replay(p[:, [2, 1, 0]], issue, at, 2 - y)
    np.testing.assert_allclose(result["adaptive"][:, [2, 1, 0]], reflected["adaptive"])
    np.testing.assert_array_equal(result["adaptive"][:, 1], p[:, 1])
    flat = replay(p, issue, at, np.ones(len(y), dtype=int))
    np.testing.assert_allclose(flat["weights"], 1 / 3)
    assert not flat["feedback_count"].any()


def test_learns_continuation_when_reversal_is_wrong_without_changing_flat():
    p, issue, at, y = sample(200)
    result = replay(p, issue, at, np.zeros(len(y), dtype=int))
    assert result["weights"][-1, 1] > 0.99
    assert result["adaptive"][-1, 0] > result["adaptive"][-1, 2]
    np.testing.assert_array_equal(result["adaptive"][:, 1], p[:, 1])


def test_unknown_outcomes_are_never_invented():
    p, issue, at, y = sample()
    y[:] = -1
    at[:] = np.nan
    result = replay(p, issue, at, y)
    assert not result["feedback_count"].any()
    np.testing.assert_allclose(result["adaptive"][:, 0], result["adaptive"][:, 2])


def test_exact_neutrality_does_not_acquire_a_roundoff_direction():
    q = 0.123456789
    p = np.array([[0.99 * (1 - q), 0.01, 0.99 * q]])
    result = replay(p, [10], [360], [2])
    assert result["adaptive"][0, 0] == result["adaptive"][0, 2]


def test_rejects_invalid_or_backdated_outcomes():
    p, issue, at, y = sample()
    at[0] = issue[0]
    with pytest.raises(ValueError):
        replay(p, issue, at, y)
