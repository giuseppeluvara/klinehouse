from datetime import datetime, timezone

import numpy as np
import pytest

from scripts.run_universal_history import decode, examples


@pytest.mark.parametrize("year", [2024, 2025])
def test_native_source_timestamp_units(year):
    t = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())
    unit = 1000 if year == 2024 else 1000000
    row = [
        str(t * unit),
        "100",
        "101",
        "99",
        "100",
        "2",
        str((t + 60) * unit - 1),
        "200",
        "3",
        "1",
        "100",
        "0",
    ]
    assert decode(row, year, 1) == (t, 100.0, 2.0, 1.0, 3.0)
    row[6] = str(t * unit - 1)
    assert decode(row, year, 1) is None


def test_wrong_or_protected_source_month_rejected():
    row = [
        "1785542400000000",
        "100",
        "101",
        "99",
        "100",
        "2",
        "1785542459999999",
        "200",
        "3",
        "1",
        "100",
        "0",
    ]
    with pytest.raises(ValueError, match="unregistered"):
        decode(row, 2026, 8)


def test_cross_asset_gaps_exclude_crossing_contexts_and_targets():
    rng = np.random.default_rng(34)
    data = {}
    t = 1704067200 + np.arange(300) * 60
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        data[symbol] = {
            "time": t.copy(),
            "close": np.exp(rng.normal(0, 0.01, 300)),
            "volume": np.ones(300),
            "taker": np.full(300, 0.5),
            "trades": np.full(300, 10.0),
        }
    data["ETHUSDT"] = {k: np.delete(v, 120) for k, v in data["ETHUSDT"].items()}
    s, cells, signs = examples(data)
    assert not np.any((s["anchor"] > t[120] - 300) & (s["anchor"] <= t[120] + 62 * 60))
    assert s["x"].shape[:2] == cells.shape == signs.shape
    assert len(s["anchor"]) > 0
