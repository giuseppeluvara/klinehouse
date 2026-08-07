from __future__ import annotations
# ruff: noqa: E701, E702

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import sequence_experiment as seq


def frame(start="2020-01-01", minutes=500, price=100.0):
    first = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    times = first + np.arange(minutes, dtype=np.int64) * seq.MINUTE_MS
    close = price * np.exp(np.arange(minutes) * 1e-6)
    return pd.DataFrame({"time": times, "open": close, "high": close * 1.001,
        "low": close * 0.999, "close": close, "volume": np.full(minutes, 10.0),
        "quote_volume": close * 10, "taker_buy_base": np.full(minutes, 4.0)})


def frames(minutes=500):
    return {symbol: frame(minutes=minutes, price=100 + 10 * i)
            for i, symbol in enumerate(seq.SYMBOLS)}


def cutoff(data):
    return pd.to_datetime(data["BTCUSDT"]["time"].iloc[-1] + seq.MINUTE_MS,
                          unit="ms", utc=True).isoformat()


def test_feature_prefix_invariance_and_support_bar():
    original = frame(minutes=200); changed = original.copy()
    _, before = seq.feature_arrays(original); changed.loc[150:, "close"] *= 5
    _, after = seq.feature_arrays(changed)
    np.testing.assert_array_equal(before[:150], after[:150])
    assert np.isnan(before[0, 0]) and np.isfinite(before[1:, 0]).all()


def test_complete_future_and_hourly_nonoverlap():
    source = frames(); data = seq.build_arrays(source, cutoff(source))
    end = int(pd.Timestamp(cutoff(source)).timestamp() * 1000)
    assert len(data["anchor"]) > 2
    assert (data["label_end"] <= end).all()
    np.testing.assert_array_equal(np.diff(data["anchor"]),
                                  np.full(len(data["anchor"]) - 1, 60 * seq.MINUTE_MS))
    np.testing.assert_array_equal(data["label_end"], data["anchor"] + 60 * seq.MINUTE_MS)


def test_t_minus_121_gap_excludes_anchor_locally():
    complete = frames(); baseline = seq.build_arrays(complete, cutoff(complete))
    target = int(baseline["anchor"][1]); broken = frames()
    missing = target - (seq.LOOKBACK + 1) * seq.MINUTE_MS
    broken["ETHUSDT"] = broken["ETHUSDT"][broken["ETHUSDT"]["time"] != missing]
    rebuilt = seq.build_arrays(broken, cutoff(complete))
    assert target in baseline["anchor"] and target not in rebuilt["anchor"]
    assert len(rebuilt["anchor"]) < len(baseline["anchor"])


def test_cross_asset_input_is_synchronized_and_strictly_causal():
    original = frames(); first = seq.build_arrays(original, cutoff(original))
    target = int(first["anchor"][1]); changed = frames()
    changed["SOLUSDT"].loc[changed["SOLUSDT"]["time"] >= target, "close"] *= 10
    second = seq.build_arrays(changed, cutoff(original))
    i, j = np.searchsorted(first["anchor"], target), np.searchsorted(second["anchor"], target)
    np.testing.assert_array_equal(first["x"][i], second["x"][j])
    assert first["x"].shape[1:] == (120, 12)


def test_target_uses_open_t_and_close_t_plus_59():
    source = frames(); preliminary = seq.build_arrays(source, cutoff(source))
    target = int(preliminary["anchor"][1]); btc = source["BTCUSDT"]
    btc.loc[btc["time"] == target, "open"] = 100.0
    btc.loc[btc["time"] == target + 59 * seq.MINUTE_MS, "close"] = 100 * np.exp(seq.THRESHOLD)
    result = seq.build_arrays(source, cutoff(source)); index = np.searchsorted(result["anchor"], target)
    assert result["y"][index] == 2


def test_purge_uses_real_support_intervals():
    start = pd.Timestamp("2020-01-01", tz="UTC"); end = pd.Timestamp("2024-01-01", tz="UTC")
    anchors = (pd.date_range(start, end, freq="h", inclusive="left").asi8 // 1_000_000).astype(np.int64)
    label_end = anchors + seq.HOUR_MS if hasattr(seq, "HOUR_MS") else anchors + 60 * seq.MINUTE_MS
    folds = seq.temporal_folds(anchors, label_end, end.isoformat())
    assert len(folds) == 3
    for fold in folds:
        support_start, support_end = anchors[fold.train] - 120 * seq.MINUTE_MS, label_end[fold.train]
        assert np.all((support_end <= fold.protected_start) | (support_start >= fold.protected_end))
        assert not np.intersect1d(anchors[fold.train], anchors[fold.validation]).size


def test_scaler_and_prior_use_training_only():
    train = np.zeros((4, 120, 12), np.float32); validation = np.full((2, 120, 12), 99, np.float32)
    train[:, :, 0] = np.arange(4)[:, None]; mean, std = seq.fit_scaler(train)
    assert mean[0] == pytest.approx(1.5) and mean[1] == 0 and not np.isclose(mean[0], np.r_[train, validation].mean())
    prior, predictions = seq.prior_probabilities(np.array([0, 0, 1, 2]), 3)
    np.testing.assert_allclose(prior, [0.5, 0.25, 0.25]); np.testing.assert_array_equal(predictions[0], prior)


def test_hash_and_oof_bootstrap_are_reproducible():
    array = np.arange(20, dtype=np.int64); assert seq._array_hash(array) == seq._array_hash(array.copy())
    source = frames(); first, second = seq.build_arrays(source, cutoff(source)), seq.build_arrays(source, cutoff(source))
    assert seq._array_hash(*first.values()) == seq._array_hash(*second.values())
    rows = pd.DataFrame({"anchor": pd.date_range("2024-01-01", periods=12, freq="7D", tz="UTC"),
        "fold": np.repeat([1, 2, 3], 4), "loss_transformer": np.linspace(0.8, 1.0, 12),
        "loss_logistic": np.ones(12)})
    assert seq.paired_bootstrap(rows, 100) == seq.paired_bootstrap(rows, 100)


def test_all_transformer_seeds_contribute_to_ensemble():
    pytest.importorskip("torch"); rng = np.random.default_rng(4)
    x = rng.normal(size=(6, 120, 12)).astype(np.float32); y = np.array([0, 1, 2, 0, 1, 2])
    mean, std = seq.fit_scaler(x); probabilities = []
    assert sum(parameter.numel() for parameter in seq.make_transformer().parameters()) < 25_000
    assert seq.SPEC["transformer"]["epochs"] == 10
    for seed in seq.SEEDS:
        model = seq.train_transformer(x, y, mean, std, seed, "cpu", batch=3, epochs=1)
        probabilities.append(seq.transformer_predict(model, x, mean, std, "cpu", batch=3))
    ensemble = np.mean(probabilities, axis=0)
    np.testing.assert_allclose(ensemble, sum(probabilities) / len(seq.SEEDS))
    assert tuple(seq.SPEC["transformer"]["seeds"]) == seq.SEEDS


def test_exclusive_freeze_and_final_open(tmp_path, monkeypatch):
    path = tmp_path / "freeze.json"; seq._json(path, {"state": "pending"}, exclusive=True)
    with pytest.raises(FileExistsError): seq._json(path, {"state": "changed"}, exclusive=True)
    monkeypatch.setattr(seq, "ROOT", tmp_path); (tmp_path / "final-test.json").write_text(json.dumps({"opened": True}))
    with pytest.raises(FileExistsError): seq.final_test()
