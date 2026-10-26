"""Native timestamps, causal features and local rather than global gap handling."""

from datetime import timedelta
import copy
import math

import numpy as np
import pytest

from research import timestral_history as history
from research.timestral_experiment import utc
from scripts.prepare_timestral_history import timestamp, decode_row


def test_malformed_maintenance_placeholder_becomes_a_gap():
    raw = "1608559200000,22646.53,22646.53,22646.53,22646.53,0,1608558440521,0,0,0,0,0".split(
        ","
    )
    assert decode_row(raw, 2020, 12) is None
    with pytest.raises(ValueError, match="misplaced"):
        decode_row(raw, 2021, 1)


def hours():
    start = utc("2023-12-01")
    return [
        {
            "time": (start + timedelta(hours=i)).isoformat(),
            "close": math.exp(10 + 0.0001 * i),
            "high": math.exp(10 + 0.0001 * i + 0.001),
            "low": math.exp(10 + 0.0001 * i - 0.001),
            "volume": 100.0,
            "taker": 60.0,
            "trades": 50.0,
        }
        for i in range(500)
    ]


def test_native_timestamp_units_change_without_shifting_candles():
    assert (
        timestamp("1735689600000") == timestamp("1735689600000000") == utc("2025-01-01")
    )


def test_features_and_targets_are_causal_with_exact_endpoint():
    source = hours()
    anchor = utc(source[169]["time"])
    values, scale = history.features(source[:169], anchor, 24)
    assert scale == pytest.approx(0.0001 * np.sqrt(24))
    assert values[0] == pytest.approx(0.0004 / scale)
    assert values[6] == pytest.approx(0.2)
    rows, _ = history.examples(source)
    r = rows[0]
    assert r["actual"] == pytest.approx(0.0001 * r["horizon_hours"])
    changed = copy.deepcopy(source)
    for row in changed:
        if utc(row["time"]) >= utc(r["anchor"]):
            row["close"] *= 2
            row["high"] *= 2
            row["low"] *= 2
    altered, _ = history.examples(changed)
    assert altered[0]["features"] == r["features"]
    assert altered[0]["actual"] != r["actual"]


def test_gap_excludes_affected_windows_and_recovers_later():
    source = hours()
    full, _ = history.examples(source)
    missing = source[:200] + source[201:]
    partial, audit = history.examples(missing)
    assert audit["missing_context"] > 0 and audit["missing_target"] > 0
    assert len(partial) < len(full)
    assert partial[-1] == full[-1]
    for row in partial:
        anchor = utc(row["anchor"])
        gap = utc(source[200]["time"])
        assert (
            not anchor - timedelta(hours=169)
            <= gap
            < anchor + timedelta(hours=row["horizon_hours"])
        )
