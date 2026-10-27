"""Noise-dependent shrinkage without a second calibration coefficient."""

import numpy as np
import pytest
from research import timestral_uncertain_market as model
from research.timestral_simple_direction import fit


def test_strong_signal_survives_and_reflection_preserves_shrinkage():
    rng = np.random.default_rng(991)
    cutoff = 1704067200
    t = (
        cutoff
        - 90 * 86400
        + np.repeat(np.arange(90), 100) * 86400
        + np.tile(np.arange(100), 90) * 300
    )
    z = np.clip(rng.normal(size=len(t)), -3, 3)
    y = (rng.random(len(t)) < 1 / (1 + np.exp(-0.3 * z))).astype(int)
    raw = fit(z, y, t, t + 360, cutoff, 0)
    s = model.adjust(raw, z, y, t, t + 360, cutoff)
    assert 0.9 < s["weight"] < 1
    assert 0 < s["beta"] < raw["beta"]
    mirrored = model.adjust(raw, -z, 1 - y, t, t + 360, cutoff)
    assert mirrored["variance"] == pytest.approx(s["variance"])
    np.testing.assert_allclose(
        model.predict(s, z, np.full(len(z), cutoff)),
        1 - model.predict(s, -z, np.full(len(z), cutoff)),
    )
    weak = {**raw, "beta": 0.001}
    assert model.adjust(weak, z, y, t, t + 360, cutoff)["beta"] == 0
    with pytest.raises(ValueError):
        model.adjust(raw, z, y, t, np.full(len(t), cutoff + 1), cutoff)
