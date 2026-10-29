import copy

import numpy as np
import pytest

from research import shadow_model as live
from research import timestral_lead_lag as historical_price
from research import timestral_position_information as historical_positions
from research import timestral_universal as historical
from research import timestral_volume_center as historical_center
from scripts.freeze_shadow_model import freeze


def source_fixture():
    rng = np.random.default_rng(7821)
    time = live.FROZEN_CUTOFF + np.arange(140) * 60
    sources = {}
    for a, symbol in enumerate(live.SYMBOLS):
        returns = rng.normal(0, 0.0003, len(time))
        returns[::9] = 0
        volume = rng.uniform(0, 100, len(time))
        sources[symbol] = dict(
            time=time,
            close=(a + 1) * 10 * np.exp(np.cumsum(returns)),
            volume=volume,
            taker=volume * rng.uniform(0, 1, len(time)),
            trades=rng.integers(0, 1000, len(time)),
        )
    return sources


def metric_fixture(anchor):
    return {
        symbol: [
            dict(
                label=anchor - 4500 + j * 300,
                oi=1000 + 3 * j,
                position_ratio=1 + j * 0.01,
            )
            for j in range(13)
        ]
        for symbol in live.SYMBOLS
    }


def context(source, anchor):
    return {
        symbol: {key: value[data["time"] < anchor] for key, value in data.items()}
        for symbol, data in source.items()
    }


def test_live_features_reproduce_frozen_formulas_before_boundary():
    sources = source_fixture()
    sample = historical.examples(sources)
    for index in (0, 4, 10):
        anchor = sample["anchor"][index]
        x, signal, price = live.spot_features(context(sources, anchor), anchor)
        np.testing.assert_allclose(x, sample["x"][:, index], atol=2e-12, rtol=2e-12)
        expected, _ = historical_center.signals(
            sample["x"][:, index : index + 1],
            historical_center.centers(sources, np.array([anchor])),
        )
        np.testing.assert_allclose(
            signal, expected["uniform"][:, 0], atol=2e-12, rtol=2e-12
        )
        for a, symbol in enumerate(live.SYMBOLS):
            expected_price, _, valid = historical_price.features(
                sources[symbol], sources[symbol], np.array([anchor])
            )
            assert valid[0]
            np.testing.assert_allclose(
                price[a], expected_price[0], atol=2e-12, rtol=2e-12
            )


def test_live_positions_match_exact_historical_lag_and_fallback():
    anchor = live.FROZEN_CUTOFF + 7200
    metrics = metric_fixture(anchor)
    hour = np.array([2.0, -9.0, 0.0])
    actual, valid, _ = live.position_features(metrics, anchor, hour)
    assert valid.all()
    for a, symbol in enumerate(live.SYMBOLS):
        rows = metrics[symbol]
        values = np.ones((13, 6))
        values[:, 0] = [r["oi"] for r in rows]
        values[:, 3] = [r["position_ratio"] for r in rows]
        label = np.array([r["label"] for r in rows])
        source = dict(
            label=label, values=values, file_day=label // 86400, row=np.arange(13)
        )
        expected, okay = historical_positions.metrics_features(
            source, np.array([anchor]), hour[a : a + 1]
        )
        assert okay[0]
        np.testing.assert_allclose(actual[a], expected[0], atol=1e-14, rtol=0)
    metrics["ETHUSDT"].append(metrics["ETHUSDT"][3].copy())
    _, valid, reasons = live.position_features(metrics, anchor, hour)
    assert valid.tolist() == [True, False, True]
    assert reasons[1]


def test_snapshot_and_predictions_use_frozen_states_and_global_fallback(tmp_path):
    path = tmp_path / "snapshot.json"
    snapshot = freeze(path)
    with pytest.raises(FileExistsError):
        freeze(path)
    source = source_fixture()
    anchor = live.FROZEN_CUTOFF + 100 * 60
    spot = context(source, anchor)
    metrics = metric_fixture(anchor)
    full = live.predict(snapshot, spot, metrics, anchor)
    for symbol in live.SYMBOLS:
        for p in full[symbol]["models"].values():
            assert sum(p) == pytest.approx(1, abs=1e-14)
            assert min(p) > 0
        assert full[symbol]["diagnostics"]["correction_available"]
        assert (
            full[symbol]["models"]["baseline"][1]
            == full[symbol]["models"]["positions"][1]
        )
    metrics["BTCUSDT"].pop(5)
    fallback = live.predict(snapshot, spot, metrics, anchor)
    for symbol in live.SYMBOLS:
        assert not fallback[symbol]["diagnostics"]["correction_available"]
        assert (
            fallback[symbol]["models"]["baseline"] == full[symbol]["models"]["baseline"]
        )
        assert len({tuple(v) for v in fallback[symbol]["models"].values()}) == 1
    broken = copy.deepcopy(snapshot)
    broken["corrections"]["positions"]["cutoff"] = anchor
    with pytest.raises(ValueError, match="cutoff"):
        live.predict(broken, spot, metrics, anchor)


def test_live_dates_allowed_without_modifying_historical_boundary(tmp_path):
    snapshot = freeze(tmp_path / "snapshot.json")
    source = source_fixture()
    anchor = live.FROZEN_CUTOFF + 100 * 60
    spot = context(source, anchor)
    metrics = metric_fixture(anchor)
    original = live.predict(snapshot, spot, metrics, anchor)
    shift = 200 * 86400
    shifted_spot = copy.deepcopy(spot)
    for d in shifted_spot.values():
        d["time"] += shift
    shifted_metrics = copy.deepcopy(metrics)
    for rows in shifted_metrics.values():
        for row in rows:
            row["label"] += shift
    future = live.predict(snapshot, shifted_spot, shifted_metrics, anchor + shift)
    for symbol in live.SYMBOLS:
        assert original[symbol]["models"] == future[symbol]["models"]
    with pytest.raises(ValueError, match="protected"):
        historical_price.features(
            shifted_spot["BTCUSDT"], shifted_spot["BTCUSDT"], np.array([anchor + shift])
        )


def test_gap_unclosed_and_invalid_spot_rows_are_rejected():
    source = source_fixture()
    anchor = live.FROZEN_CUTOFF + 100 * 60
    spot = context(source, anchor)
    for problem in ("gap", "future", "price", "taker"):
        bad = copy.deepcopy(spot)
        d = bad["ETHUSDT"]
        if problem == "gap":
            bad["ETHUSDT"] = {k: np.delete(v, -5) for k, v in d.items()}
        elif problem == "future":
            d["time"][-1] = anchor
        elif problem == "price":
            d["close"][-2] = 0
        else:
            d["taker"][-3] = d["volume"][-3] + 1
        with pytest.raises(ValueError):
            live.spot_features(bad, anchor)


def test_target_uses_exact_equality_without_a_deadband():
    assert live.outcome_label(100, 100) == 1
    assert live.outcome_label(100, 100.00000001) == 2
    assert live.outcome_label(100, 99.99999999) == 0
    assert live.outcome_label("1000000000000000.01", "1000000000000000.02") == 2
    assert live.outcome_label("1000000000000000.02", "1000000000000000.01") == 0
    assert live.outcome_label("100.000", "100.0") == 1
    with pytest.raises(ValueError):
        live.outcome_label(0, 1)
    with pytest.raises(ValueError):
        live.outcome_label("NaN", "1")
