"""Causal contexts, reflected regimes, and recovery of state-dependent signals."""

import numpy as np
import pytest
from research import timestral_regime_direction as model


def source():
    t = 1704067200 + np.arange(2200) * 60
    return t, {
        s: dict(
            time=t.copy(),
            close=np.exp(5 + np.arange(2200) * 1e-5),
            trades=np.full(2200, 100.0),
        )
        for s in model.shared.SYMBOLS
    }


def test_context_excludes_future_and_reference_excludes_latest_hour():
    t, d = source()
    a = t[[1900]]
    for s in d.values():
        s["trades"][1840:1900] = 200
    valid, features, states = model.context(d, a)
    assert valid.all()
    np.testing.assert_array_equal(states, 3)
    for s in d.values():
        s["close"][1900:] *= 2
        s["trades"][1900:] *= 100
    v, f, state = model.context(d, a)
    np.testing.assert_array_equal(f, features)
    np.testing.assert_array_equal(state, states)
    for s in d.values():
        s["trades"][1840:1900] = 50
    _, _, state = model.context(d, a)
    np.testing.assert_array_equal(state, 2)


def test_context_needs_full_history_and_reflects_direction_only():
    t, d = source()
    a = t[[1000, 1900]]
    valid, f, state = model.context(d, a)
    np.testing.assert_array_equal(valid, [False, True])
    for s in d.values():
        s["close"] = 1 / s["close"]
    v, mirrored, reflected_state = model.context(d, a)
    np.testing.assert_allclose(mirrored, -f, atol=1e-8)
    np.testing.assert_array_equal(state, reflected_state)
    d["BTCUSDT"]["time"][-1] = model.shared.BOUNDARY
    with pytest.raises(ValueError):
        model.context(d, a)


def test_regime_interactions_are_centered_and_sign_symmetric():
    x = np.random.default_rng(7).normal(size=(3, 20, 39))
    state = np.tile(np.arange(20) % 4, (3, 1))
    extended = model.interact(x, state)
    assert extended.shape == (3, 20, 71)
    np.testing.assert_allclose(
        extended[:, :, 39:].reshape(3, 20, 8, 4).sum(3), 0, atol=1e-15
    )
    mirror = x.copy()
    odd = list(model.shared.ODD) + list(range(23, 39))
    mirror[:, :, odd] *= -1
    other = model.interact(mirror, state)
    allodd = list(model.shared.ODD) + list(range(23, 71))
    np.testing.assert_allclose(other[:, :, allodd], -extended[:, :, allodd])


def test_regime_fit_recovers_a_signal_that_flips_in_known_conditions():
    rng = np.random.default_rng(745)

    def batch(n):
        x = np.zeros((1, n, 39))
        x[0, :, 0] = rng.normal(size=n)
        state = rng.integers(0, 4, size=(1, n))
        label = (x[0, :, 0] * np.where(state[0] < 2, 1, -1) > 0).astype(int)
        return model.interact(x, state)[0], label

    x, y = batch(2400)
    held, label = batch(600)
    cutoff = 1704067200
    base = model.fit(x[:, :39], y, np.full(len(y), cutoff), cutoff)
    regime = model.fit(x, y, np.full(len(y), cutoff), cutoff)
    pred = model.predict(regime, held, np.full(len(held), cutoff))
    plain = model.predict(base, held[:, :39], np.full(len(held), cutoff))
    assert ((pred >= 0.5) == label).mean() > 0.95
    assert ((plain >= 0.5) == label).mean() < 0.60
    with pytest.raises(ValueError):
        model.fit(x, y, np.full(len(y), cutoff + 1), cutoff)


def test_drift_decomposition_separates_mix_from_conditional_change():
    d = {}
    for s in model.shared.SYMBOLS:
        for year, weights, rates in [
            (2024, [0.25] * 4, [0.4, 0.5, 0.6, 0.7]),
            (2025, [0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]),
            (2026, [0.25] * 4, [0.5, 0.6, 0.7, 0.8]),
        ]:
            cells = [dict(weight=w, accuracy=g) for w, g in zip(weights, rates)]
            d[f"{year}-{s}"] = {
                name: dict(accuracy=float(np.dot(weights, rates)), regimes=cells)
                for name in ("spot_reversal", "futures_flow")
            }
    r = model.drift(d)
    for name, z in r.items():
        assert z["total"] == pytest.approx(z["composition"] + z["within"])
        assert z[
            "within" if name.startswith("2025") else "composition"
        ] == pytest.approx(0)
