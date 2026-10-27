"""Count-logarithm rounding stays stable at frozen decision boundaries."""

import numpy as np
import pytest

from scripts.verify_history_portable import (
    count_log1p,
    reconstruct,
    original_reconstruct,
)
from scripts.verify_universal_direction import ASSETS


def test_count_log_rounds_known_split_without_vector_math():
    # The next binary64 value flips the archived BTC June 2024 tree branch.
    expected = float.fromhex("0x1.9f3637078b16bp+2")
    result = count_log1p(np.full((3, 20), 656.0))
    np.testing.assert_array_equal(result, expected)
    assert not (np.nextafter(expected, np.inf) <= expected)
    np.testing.assert_array_equal(count_log1p(np.array([0, 656, 0])), [0, expected, 0])


@pytest.mark.parametrize("bad", [[-1], [0.5], [np.nan], [np.inf], [2**53]])
def test_non_count_inputs_fail_closed(bad):
    with pytest.raises(ValueError):
        count_log1p(bad)


def test_reconstruction_changes_only_the_count_logarithm():
    t = 1704067200 + np.arange(300) * 60
    data = [
        dict(
            time=t,
            close=100 + np.sin(np.arange(300) / 17),
            volume=np.ones(300),
            taker=np.full(300, 0.4),
            trades=np.full(300, 656.0),
        )
        for _ in ASSETS
    ]
    old = original_reconstruct(data)
    new = reconstruct(data)
    for index in (0, 1, 3):
        np.testing.assert_array_equal(new[index], old[index])
    cols = [i for i in range(23) if i != 8]
    np.testing.assert_array_equal(new[2][:, :, cols], old[2][:, :, cols])
    np.testing.assert_array_equal(
        new[2][:, :, 8], float.fromhex("0x1.9f3637078b16bp+2")
    )
