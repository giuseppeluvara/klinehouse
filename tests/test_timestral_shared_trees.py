import numpy as np
import pytest

from research import timestral_shared_trees as t


def test_tree_export_and_reflection_on_known_nonlinear_signal():
    rng = np.random.default_rng(27)
    x = np.zeros((1000, 23))
    x[:, 13] = rng.normal(size=1000)
    x[:, 22] = abs(x[:, 13])
    y = np.where(x[:, 13] > 0.3, 2, np.where(x[:, 13] < -0.3, 0, 1))
    f = t.fit(x, y, np.zeros(1000), 10)
    p = t.predict(f, x, np.full(1000, 11))
    assert np.mean(p.argmax(1) == y) > 0.9
    mirror = x.copy()
    mirror[:, 13] *= -1
    np.testing.assert_allclose(
        t.predict(f, mirror, np.full(1000, 11)), p[:, ::-1], atol=1e-12
    )
    with pytest.raises(ValueError, match="matured"):
        t.fit(x, y, np.full(1000, 11), 10)
    with pytest.raises(ValueError, match="protected"):
        t.predict(f, x, np.full(1000, 1785542400))


def test_malformed_cyclic_tree_fails_closed():
    f = {"initial": [0, 0, 0], "trees": [[[[0, 0, 0, 0, 0, 0]]] * 3]}
    with pytest.raises(ValueError, match="cyclic"):
        t.raw_predict(f, np.zeros((1, 23)))
