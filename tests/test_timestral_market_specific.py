"""No learned agreement threshold, fixed blending and explicit flat accounting."""

import numpy as np
from research import timestral_market_specific as model


def test_signals_are_bounded_and_reflect_without_fitted_mixture_weight():
    x = np.random.default_rng(415).normal(size=(3, 100, 23))
    c, o, b = model.signals(x)
    cc, oo, bb = model.signals(-x)
    np.testing.assert_array_equal(cc, -c)
    np.testing.assert_array_equal(oo, -o)
    np.testing.assert_array_equal(bb, -b)
    np.testing.assert_array_equal(b, (c + o) / 2)
    assert np.max(abs(b)) <= 3


def test_consensus_abstains_on_disagreement_and_zero_without_removing_rows():
    q = np.array([0.6, 0.4, 0.55, 0.45, 0.5])
    c = np.array([1, -1, 1, -1, 0])
    o = np.array([2, -2, -2, 2, 1])
    p = model.consensus(q, c, o)
    np.testing.assert_array_equal(p, [0.6, 0.4, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(model.consensus(1 - q, -c, -o), 1 - p)


def test_diagnostic_separates_flat_penalty_from_direction_errors():
    y = np.array([2, 1, 2, 0])
    q = np.array([0.6, 0.6, 0.6, 0.4])
    f = np.zeros(4)
    c = np.array([1, 1, 1, -1])
    o = np.array([1, 1, -1, -1])
    d = model.diagnose(y, q, f, c, o)
    assert d["selected_binary_accuracy"] == 1
    assert d["selected_all_accuracy"] == 0.75
    assert d["selected_flat_rate"] == 0.25
    assert d["groups"]["disagree"]["own_accuracy"] == 0
    assert sum(g["all_n"] for g in d["groups"].values()) == 4
