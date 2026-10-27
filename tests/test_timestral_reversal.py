import numpy as np
import pytest

from research import timestral_reversal as r


def test_sign_symmetry_and_flat_input_abstention():
    cells = np.zeros(200, dtype=int)
    signs = np.tile([-1, 1], 100)
    labels = np.where(signs < 0, 2, 0)
    f = r.fit(cells, signs, labels, np.zeros(200), 10)
    p, selected = r.predict(
        f, np.zeros(3, dtype=int), np.array([-1, 1, 0]), np.full(3, 11)
    )
    assert np.allclose(p[0, 2], p[1, 2, ::-1])
    assert p[0, 2, 2] > 0.9 and p[1, 2, 0] > 0.9
    assert selected.tolist() == [True, True, False]
    assert np.array_equal(p[2, 2], f["frequency"])


def test_flat_targets_are_kept_and_future_labels_refused():
    cells = np.zeros(100, dtype=int)
    f = r.fit(cells, np.ones(100), np.ones(100, dtype=int), np.zeros(100), 10)
    p, selected = r.predict(f, cells, np.ones(100), np.full(100, 11))
    assert (p[:, 2, 1] > 0.9).all()
    assert not selected.any()
    with pytest.raises(ValueError, match="immature"):
        r.fit(cells, np.ones(100), np.ones(100, dtype=int), np.full(100, 11), 10)


def test_no_selected_calls_cannot_pass():
    y = np.tile([0, 2], 100)
    p = np.tile([0.495, 0.01, 0.495], (200, 3, 1))
    result = r.evaluate(
        y, p, 1767225600 + np.arange(200) * 86400, np.zeros(200, dtype=bool)
    )
    assert result["gate"] is False
    assert result["coverage"] == 0
