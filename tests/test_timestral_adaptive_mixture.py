"""Changing-expert selection, maturity and future-data isolation."""

import numpy as np
import pytest

from research import timestral_adaptive_mixture_refined as model


def stream():
    t = 1704067200 + np.arange(42 * 288) * 300
    experts = np.tile(
        [[0.1, 0.05, 0.85], [0.85, 0.05, 0.1], [0.45, 0.1, 0.45], [0.4, 0.2, 0.4]],
        (len(t), 1, 1),
    )
    y = np.full(len(t), 2)
    y[14 * 288 :] = 0
    return experts, y, t, t + 360


def test_mixture_switches_to_expert_matching_new_regime():
    experts, y, t, end = stream()
    a, f, states = model.replay(experts, y, t, end)
    assert states[0]["weights"][0] > 0.9
    assert states[-1]["weights"][1] > 0.9
    assert a[-1].argmax() == 0 and f[-1].argmax() == 2
    np.testing.assert_allclose(a.sum(1), 1, atol=1e-12)


def test_future_labels_do_not_change_earlier_mixture():
    experts, y, t, end = stream()
    experts, y, t, end = [x[: 10 * 288] for x in (experts, y, t, end)]
    cutoff = t[0] + 8 * 86400
    a, f, _ = model.replay(experts, y, t, end)
    changed = y.copy()
    changed[end > cutoff] = 0
    b, g, _ = model.replay(experts, changed, t, end)
    np.testing.assert_array_equal(a[t < cutoff + 86400], b[t < cutoff + 86400])
    np.testing.assert_array_equal(f, g)


def test_fit_rejects_unmatured_labels_and_bad_expert_shape():
    e, y, t, end = stream()
    with pytest.raises(ValueError):
        model.fit(e[:3000], y[:3000], t[:3000], end[:3000], t[0] + 7 * 86400)
    with pytest.raises(ValueError):
        model.replay(e[:, :3], y, t, end)


def test_near_equal_experts_with_rare_class_pass_optimality():
    rng = np.random.default_rng(1848)
    t = 1704067200 + np.arange(2015) * 300
    e = rng.uniform(0.45, 0.55, (len(t), 4, 3))
    e[:, :, 1] *= 0.02
    e /= e.sum(2, keepdims=True)
    y = rng.choice(3, len(t), p=[0.48, 0.01, 0.51])
    state = model.fit(e, y, t, t + 360, t[0] + 7 * 86400)
    assert state["kkt_residual"] < 2e-6
