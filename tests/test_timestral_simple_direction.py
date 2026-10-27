"""Shrinkage, symmetry, maturity and the intended relative-value mechanism."""

import numpy as np
import pytest
from research import timestral_simple_direction as model


def batch(n=3000):
    rng = np.random.default_rng(921)
    x = np.zeros((n, 14))
    market, residual = rng.normal(size=(2, n))
    x[:, 0], x[:, 13] = market + residual, market
    z = model.signal(x, True)
    y = (rng.random(n) < 1 / (1 + np.exp(-0.5 * z))).astype(int)
    anchors = 1704067200 - 60 * np.arange(n, 0, -1) - 360
    return x, z, y, anchors


def test_relative_signal_cancels_common_move_and_is_bounded_odd():
    x, z, _, _ = batch()
    shifted = x.copy()
    shifted[:, [0, 13]] += 2
    np.testing.assert_allclose(model.signal(shifted, True), z, atol=1e-15)
    np.testing.assert_array_equal(model.signal(-x, True), -z)
    assert np.max(abs(z)) <= 3


def test_one_coefficient_shrinkage_and_reflection():
    _, z, y, t = batch()
    raw = model.fit(z, y, t, t + 360, 1704067200, 0.0)
    shrunk = model.fit(z, y, t, t + 360, 1704067200, 0.25)
    reflected = model.fit(-z, 1 - y, t, t + 360, 1704067200, 0.25)
    assert 0 < shrunk["beta"] < raw["beta"] < 1
    assert shrunk["beta"] == pytest.approx(reflected["beta"], abs=1e-12)
    pred_t = np.full(len(z), 1704067200)
    p = model.predict(shrunk, z, pred_t)
    np.testing.assert_allclose(model.predict(shrunk, -z, pred_t), 1 - p)


def test_wrong_orientation_collapses_to_neutral_and_days_have_equal_weight():
    _, z, y, t = batch()
    state = model.fit(-z, y, t, t + 360, 1704067200, 0.25)
    assert state["beta"] == 0
    w = model.day_weights(np.array([0, 60, 86400]))
    np.testing.assert_array_equal(w, [0.25, 0.25, 0.5])


def test_mature_labels_and_protected_prediction_boundary():
    _, z, y, t = batch()
    with pytest.raises(ValueError):
        model.fit(z, y, t, np.full(len(t), 1704067201), 1704067200, 0.25)
    state = model.fit(z, y, t, t + 360, 1704067200, 0.25)
    with pytest.raises(ValueError):
        model.predict(state, z, np.full(len(z), model.BOUNDARY - 300))
    with pytest.raises(ValueError):
        model.predict(state, z, np.full(len(z), state["cutoff"] - 1))


def test_planted_relative_signal_generalizes_with_one_parameter():
    x, z, y, t = batch(10000)
    train = np.arange(len(y)) < 7000
    state = model.fit(z[train], y[train], t[train], t[train] + 360, 1704067200, 0.25)
    held = ~train
    p = model.predict(state, z[held], np.full(held.sum(), 1704067200))
    assert ((p > 0.5) == y[held]).mean() > 0.54
    assert ((p > 0.5) == y[held]).mean() > ((-x[held, 0] > 0) == y[held]).mean()
