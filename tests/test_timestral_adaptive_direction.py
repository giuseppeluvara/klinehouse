"""Causality, delayed outcomes and calibration under a synthetic regime change."""

import numpy as np
import pytest

from research import timestral_adaptive_direction as model


def stream(days=42):
    t = 1704067200 + np.arange(days * 288) * 300
    p = np.tile([0.09, 0.02, 0.89], (len(t), 1))
    rng = np.random.default_rng(812)
    y = rng.choice(3, len(t), p=[0.09, 0.02, 0.89])
    shift = min(len(t), 14 * 288)
    y[shift:] = rng.choice(3, len(t) - shift, p=[0.49, 0.02, 0.49])
    return p, y, t, t + 360


def test_adaptive_probability_recovers_after_synthetic_shift():
    p, y, t, end = stream()
    adaptive, fixed, states = model.replay(p, y, t, end)
    tail = slice(-7 * 288, None)
    target = np.eye(3)[y[tail]]
    assert (
        np.square(adaptive[tail] - target).mean()
        < np.square(fixed[tail] - target).mean()
    )
    assert states[-1]["alpha"] < states[0]["alpha"] / 2
    np.testing.assert_allclose(adaptive.sum(1), 1, atol=1e-15)
    np.testing.assert_array_equal(adaptive[: 7 * 288], p[: 7 * 288])


def test_future_outcomes_cannot_change_current_predictions():
    p, y, t, end = stream(12)
    cutoff = t[0] + 9 * model.DAY
    a, f, states = model.replay(p, y, t, end)
    changed = y.copy()
    changed[end > cutoff] = 2 - changed[end > cutoff]
    b, g, _ = model.replay(p, changed, t, end)
    np.testing.assert_array_equal(a[t < cutoff + model.DAY], b[t < cutoff + model.DAY])
    np.testing.assert_array_equal(f, g)
    for state in states:
        assert state["max_end"] <= state["cutoff"]


def test_sign_reflection_and_flat_calibration_are_symmetric():
    p, y, t, end = stream(9)
    a, f, states = model.replay(p, y, t, end)
    b, g, mirror = model.replay(p[:, ::-1], 2 - y, t, end)
    np.testing.assert_allclose(a, b[:, ::-1], atol=1e-12)
    np.testing.assert_allclose(f, g[:, ::-1], atol=1e-12)
    np.testing.assert_allclose(
        [s["alpha"] for s in states], [s["alpha"] for s in mirror]
    )


def test_immature_training_outcome_is_rejected():
    p, y, t, end = stream(8)
    with pytest.raises(ValueError, match="matured"):
        model.fit(p, y, t, end, t[0] + 7 * model.DAY)


@pytest.mark.parametrize("kind", ["protected", "unaligned", "probability", "label"])
def test_invalid_stream_fails_closed(kind):
    p, y, t, end = stream(8)
    if kind == "protected":
        t += model.BOUNDARY - t[-1]
        end = t + 360
    elif kind == "unaligned":
        t[3] += 1
    elif kind == "probability":
        p[0] = [0.1, 0.1, 0.1]
    else:
        y[0] = 4
    with pytest.raises(ValueError):
        model.replay(p, y, t, end)


def test_identity_and_bounds():
    p = np.array([[0.2, 0.1, 0.7], [0.6, 0.3, 0.1]])
    np.testing.assert_allclose(model.transform(p, 1, 0), p)
    q = model.transform(p, 0, 0)
    np.testing.assert_allclose(q[:, 0], q[:, 2])
    with pytest.raises(ValueError):
        model.transform(p, 1.1, 0)
