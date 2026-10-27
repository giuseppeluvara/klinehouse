import pytest

from scripts.run_direction_transfer import decode


def test_native_minute_decode_and_bad_close_time():
    values = [
        "1767225600000000",
        "100",
        "101",
        "99",
        "100",
        "2",
        "1767225659999999",
        "200",
        "3",
        "1",
        "100",
        "0",
    ]
    assert decode(values, 2026, 1) == (1767225600, 100.0, 2.0, 1.0, 3.0)
    values[6] = "1767225599999999"
    assert decode(values, 2026, 1) is None


def test_native_minute_rejects_protected_or_misplaced_source():
    values = [
        "1767225600000000",
        "100",
        "101",
        "99",
        "100",
        "2",
        "1767225659999999",
        "200",
        "3",
        "1",
        "100",
        "0",
    ]
    with pytest.raises(ValueError, match="misplaced"):
        decode(values, 2026, 2)
    values[0] = "1785542400000000"
    with pytest.raises(ValueError, match="protected"):
        decode(values, 2026, 8)
