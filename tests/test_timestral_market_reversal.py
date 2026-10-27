"""Common signal is identical across assets and respects its fixed horizon."""

import numpy as np
import pytest
from research import timestral_market_reversal as model


def test_market_signal_pools_assets_equally_and_uses_one_registered_horizon():
    x = np.zeros((3, 10, 23))
    r = np.random.default_rng(728).normal(size=(3, 10))
    for a in range(3):
        x[a, :, 3] = r[a]
        x[a, :, 16] = np.mean([r[b] for b in range(3) if b != a], axis=0)
    z = model.signal(x, "market60")
    for a in range(3):
        np.testing.assert_allclose(z[a], -r.mean(0))
    np.testing.assert_allclose(model.signal(-x, "market60"), -z)
    np.testing.assert_array_equal(model.signal(x, "market1"), 0)
    x[:, :, 1] = 100
    np.testing.assert_array_equal(model.signal(x, "market60"), z)
    with pytest.raises(ValueError):
        model.signal(x, "market15")
