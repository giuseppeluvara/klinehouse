import numpy as np
import pytest

from research.timestral_flat_repair import probabilities, paired, decomposition


def test_flat_shrink_preserves_direction_and_reflection():
    q = np.array([0.6, 0.3, 0.5])
    f = np.array([0.1, 0.01, 0.2])
    base = np.array([0.05, 0.02, 0.1])
    p = probabilities(q, f, base)
    reflected = probabilities(1 - q, f, base)
    for name, value in p.items():
        np.testing.assert_allclose(value.sum(1), 1)
        np.testing.assert_allclose(value[:, 2] / (1 - value[:, 1]), q)
        np.testing.assert_allclose(reflected[name], value[:, ::-1])
    np.testing.assert_allclose(p["mixed"], (p["old"] + p["frequency"]) / 2)


def test_rejects_invalid_probabilities():
    q = np.array([0.5, 0.6])
    for f in (np.array([0.0, 0.1]), np.array([np.nan, 0.2]), np.array([0.1])):
        with pytest.raises(ValueError):
            probabilities(q, f, q)


def test_fixed_subset_logloss_difference_is_only_flat_component():
    y = np.tile([0, 1, 2, 0, 2], 14)
    anchors = (
        1704067200
        + np.repeat(np.arange(14) * 86400, 5)
        + np.tile(np.arange(5) * 300, 14)
    )
    q = np.full(len(y), 0.57)
    f = np.full(len(y), 0.001)
    base = np.full(len(y), 0.2)
    p = probabilities(q, f, base)
    mask = np.ones(len(y), dtype=bool)
    comparison = paired(y, p["mixed"], p["old"], anchors, mask)
    np.testing.assert_allclose(
        comparison["logloss_gain"], comparison["flat_logloss_gain"], atol=1e-14
    )
    identity = paired(y, p["old"], p["old"], anchors, mask)
    assert identity["brier_gain"] == 0 and not identity["improvement_gate"]
    d = decomposition(y, p["mixed"], p["old"], mask)
    np.testing.assert_allclose(d["direction_gain"], 0, atol=1e-14)
    np.testing.assert_allclose(d["total_gain"], comparison["logloss_gain"], atol=1e-14)
