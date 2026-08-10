from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import profitability as profit


def raw_frames(start="2024-01-01", minutes=900, price=40_000.0):
    first = int(pd.Timestamp(start, tz="UTC").timestamp())
    time = first + np.arange(minutes, dtype=np.int64) * 60
    close = price * np.exp(np.arange(minutes) * 1e-6)
    trade = pd.DataFrame({
        "time": time, "open": close, "high": close * 1.0002,
        "low": close * 0.9998, "close": close, "volume": np.full(minutes, 2.0),
    })
    mark = pd.DataFrame({
        "time": time, "open": close, "high": close * 1.0001,
        "low": close * 0.9999, "close": close * 1.00001,
        "volume": np.zeros(minutes),
    })
    count = pd.DataFrame({"time": time, "trade_count": np.full(minutes, 3.0)})
    cvd = pd.DataFrame({"time": time, "buy_volume": np.full(minutes, 1.2),
                        "sell_volume": np.full(minutes, 0.8)})
    return {"trade": trade, "mark": mark, "trade_count": count, "cvd": cvd}


def synthetic_arrays(frames=None):
    frames = raw_frames() if frames is None else frames
    aligned = profit.aligned_features(frames)
    start = pd.Timestamp("2024-01-01T02:00:00Z")
    end = pd.Timestamp("2024-01-01T10:00:00Z")
    return profit.build_arrays(aligned, start, end)


def ledger(folds=range(1, 7), per_fold=20, net=0.02):
    rows = []
    for fold in folds:
        base = pd.Timestamp("2024-01-01", tz="UTC") + pd.DateOffset(months=3 * fold)
        for index in range(per_fold):
            anchor = int((base + pd.Timedelta(days=index * 3)).timestamp())
            rows.append({
                "fold": fold, "anchor": anchor, "net_usd": net,
                "gross_usd": net + 0.01, "net_bps": net * 100,
                "gross_bps": (net + 0.01) * 100, "fees_usd": 0.002,
                "funding_usd": 0.001, "adverse_selection_usd": 0.004,
            })
    return pd.DataFrame(rows)


def test_candle_and_analytics_parsers_normalize_seconds_and_milliseconds():
    candles = {"candles": [{"time": 1_700_000_000_000, "open": "100", "high": "102",
        "low": "99", "close": "101", "volume": "2"}]}
    parsed = profit.parse_candles(candles, "trade")
    assert parsed.time.tolist() == [1_700_000_000]
    count = profit.parse_analytics({"result": {"timestamp": [1_700_000_000], "data": [4]}},
                                   "trade_count")
    cvd = profit.parse_analytics({"result": {"timestamp": [1_700_000_000_000],
        "data": {"buy_volume": ["2"], "sell_volume": ["1"]}}}, "cvd")
    assert count.trade_count.tolist() == [4]
    assert cvd.time.tolist() == [1_700_000_000]


def test_pagination_advances_and_deduplicates(monkeypatch):
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    calls = []

    def response(_client, _url, params=None):
        cursor = params["from"]
        calls.append(cursor)
        times = [cursor, cursor + 60]
        rows = [{"time": value * 1000, "open": "100", "high": "101", "low": "99",
                 "close": "100", "volume": "1"} for value in times]
        return {"candles": rows, "more_candles": cursor == int(start.timestamp())}

    monkeypatch.setattr(profit, "get_json", response)
    frame, raw = profit.download_series(object(), "trade", start, start + pd.Timedelta(minutes=4))
    assert calls == [int(start.timestamp()), int(start.timestamp()) + 120]
    assert raw == len(frame) == 4
    assert frame.time.is_unique


def test_instrument_requires_linear_lot_and_eea_metadata():
    row = {"symbol": profit.SYMBOL, "type": "flexible_futures", "base": "BTC",
        "quote": "USD", "contractSize": 1, "contractValueTradePrecision": 4,
        "tradeable": True, "isExpired": False, "platformsPermitted": ["europa"],
        "countriesBanned": []}
    assert profit.validate_instrument(row)["passed"]
    row["contractValueTradePrecision"] = 3
    assert not profit.validate_instrument(row)["passed"]


def test_funding_gap_is_explicit_data_no_go():
    export = pd.DataFrame({"timestamp": pd.to_datetime([
        "2022-04-01T00:00:00Z", "2026-02-01T00:00:00Z"]),
        "relative_rate": [0.0, 0.0]})
    api = pd.DataFrame({"time": [int(pd.Timestamp("2026-02-12T13:00:00Z").timestamp()),
                                      int(pd.Timestamp("2026-07-01T00:00:00Z").timestamp())],
                        "relative_rate": [0.0, 0.0]})
    result = profit.funding_precheck(export, api, pd.Timestamp("2022-04-01T00:00:00Z"),
                                     pd.Timestamp("2026-07-01T00:00:00Z"))
    assert not result["passed"]
    assert result["gap_hours"] == pytest.approx(277)
    assert result["overlap_rows"] == 0


def test_funding_overlap_must_agree_exactly_within_tolerance():
    timestamps = pd.to_datetime(["2022-04-01T00:00:00Z", "2026-07-01T00:00:00Z"])
    export = pd.DataFrame({"timestamp": timestamps, "relative_rate": [0.1, 0.2]})
    api = pd.DataFrame({"time": timestamps.astype("int64") // 1_000_000_000,
                        "relative_rate": [0.1, 0.2]})
    result = profit.funding_precheck(export, api, timestamps[0], timestamps[-1])
    assert result["passed"] and result["overlap_rows"] == 2
    api.loc[1, "relative_rate"] += 1e-6
    assert not profit.funding_precheck(export, api, timestamps[0], timestamps[-1])["passed"]


def test_features_are_causal_and_no_forward_fill_crosses_a_gap():
    frames = raw_frames()
    complete = synthetic_arrays(frames)
    target = int(complete["anchor"][2])
    changed = raw_frames()
    changed["cvd"] = changed["cvd"][changed["cvd"].time != target - 60]
    rebuilt = synthetic_arrays(changed)
    assert target in complete["anchor"] and target not in rebuilt["anchor"]
    future = raw_frames()
    future["mark"].loc[future["mark"].time >= target, "close"] *= 5
    altered = synthetic_arrays(future)
    before = complete["x"][np.searchsorted(complete["anchor"], target)]
    after = altered["x"][np.searchsorted(altered["anchor"], target)]
    np.testing.assert_array_equal(before, after)


def test_dataset_is_right_censored_and_reproducible():
    first, second = synthetic_arrays(), synthetic_arrays()
    assert profit.array_hash(*first.values()) == profit.array_hash(*second.values())
    cutoff = int(pd.Timestamp("2024-01-01T10:00:00Z").timestamp())
    assert np.all(first["anchor"] + 4 * 3600 <= cutoff)
    assert first["x"].shape[1:] == (120, 7) and first["aggregate"].shape[1] == 28


def test_stop_is_applied_using_mark_then_next_trade_open():
    frames = raw_frames()
    anchor = int(pd.Timestamp("2024-01-01T03:00:00Z").timestamp())
    trigger = anchor + 30 * 60
    frames["mark"].loc[frames["mark"].time == trigger, "low"] = 39_000
    frames["trade"].loc[frames["trade"].time == trigger + 60, "open"] = 39_500
    arrays = synthetic_arrays(frames)
    row = np.searchsorted(arrays["anchor"], anchor)
    assert arrays["stop_time_1"][row] == trigger + 60
    assert arrays["stop_exit_1"][row] == pytest.approx(min(arrays["entry"][row] * 0.99, 39_500))


def test_full_four_hour_purge_and_quarter_right_boundary():
    start, end = pd.Timestamp("2022-04-01T00:00:00Z"), pd.Timestamp("2026-07-01T00:00:00Z")
    anchors = pd.date_range(start, end, freq="h", inclusive="left").astype("int64") // 1_000_000_000
    folds = profit.temporal_folds(anchors.to_numpy(), start, end)
    assert len(folds) == 9
    for fold in folds:
        boundary = int(fold.start.timestamp())
        assert np.all(anchors.to_numpy()[fold.train] + 4 * 3600 <= boundary - 120 * 60)
        assert np.all(anchors.to_numpy()[fold.validation] + 4 * 3600 <= int(fold.end.timestamp()))


def test_linear_quantity_pnl_and_fee_schedules():
    qty = profit.quantity(63_000)
    assert qty == pytest.approx(0.0015) and qty * 63_000 <= 100
    maker = profit.trade_economics(63_000, 63_630, qty, False, 0.01)
    taker = profit.trade_economics(63_000, 62_370, qty, True, 0.01)
    assert maker["gross_usd"] == pytest.approx(qty * 630)
    assert maker["fees_usd"] == pytest.approx(qty * 63_000 * 0.0002 + qty * 63_630 * 0.0002)
    assert taker["fees_usd"] == pytest.approx(qty * 63_000 * 0.0002 + qty * 62_370 * 0.0005)
    assert maker["adverse_selection_usd"] == pytest.approx(qty * 63_000 * 0.0004)


def test_funding_floor_ignores_credits_and_uses_elapsed_hours():
    funding = pd.DataFrame({"time": [3600, 7200, 10_800],
                            "relative_rate": [-0.001, 0.00005, -0.002]})
    cost, rate, rows = profit.funding_cost(funding, 0, 10_800, 100)
    assert rows == 3 and rate == pytest.approx(0.0003) and cost == pytest.approx(0.03)
    funding.loc[1, "relative_rate"] = 0.0005
    assert profit.funding_cost(funding, 0, 10_800, 100)[1] == pytest.approx(0.0005)


def test_backtest_is_long_only_thresholded_and_non_overlapping():
    arrays = synthetic_arrays()
    rows = np.arange(len(arrays["anchor"]))
    funding = pd.DataFrame({"time": arrays["anchor"], "relative_rate": np.zeros(len(rows))})
    scores = np.arange(len(rows), dtype=float)
    result = profit.backtest(arrays, rows, scores, 0.5, 4, 1, funding)
    assert (result.score > 0.5).all()
    assert np.all(result.anchor.iloc[1:].to_numpy() >= result.exit_time.iloc[:-1].to_numpy())
    assert (result.entry_notional_usd <= 100 + 1e-9).all()


def test_bootstrap_and_gate_are_reproducible_and_corrected():
    rows = ledger()
    first = profit.candidate_summary(rows, list(range(1, 7)))
    second = profit.candidate_summary(rows, list(range(1, 7)))
    assert first == second and first["passed"]
    assert first["ci9917"][0] > 0 and first["ci95"][0] > 0
    failed = profit.candidate_summary(rows.iloc[:80], list(range(1, 7)))
    assert not failed["gate"]["at_least_100_trades"] and not failed["passed"]


def test_candidate_selection_uses_corrected_lower_bound_then_frozen_tie_break():
    def summary(lower, passed=True): return {"passed": passed, "ci9917": [lower, lower + 2]}
    values = {"transformer_1h": summary(5.0), "hgb_4h": summary(4.5),
              "ridge_4h": summary(4.1), "ridge_1h": summary(4.1)}
    assert profit.select_candidate(values) == "ridge_1h"
    values["transformer_1h"] = summary(6.0)
    assert profit.select_candidate(values) == "transformer_1h"
    assert profit.select_candidate({"ridge_1h": summary(1, False)}) is None


def test_transformer_is_two_layer_regressor_and_all_seeds_are_frozen():
    torch = pytest.importorskip("torch")
    model = profit.make_transformer()
    assert len(model.encoder.layers) == 2 and profit.SEEDS == (7, 17, 29)
    assert sum(parameter.numel() for parameter in model.parameters()) < 25_000
    x = np.random.default_rng(1).normal(size=(4, 120, 7)).astype(np.float32)
    mean, std = profit.fit_scaler(x, axes=(0, 1))
    fitted = profit.train_transformer(x, np.arange(4, dtype=np.float32), mean, std,
                                      seed=7, device="cpu", epochs=1, batch=2)
    prediction = profit.transformer_predict(fitted, x, mean, std, "cpu", batch=2)
    assert prediction.shape == (4,) and torch.isfinite(torch.from_numpy(prediction)).all()


def test_run_refuses_data_no_go_and_existing_results(tmp_path, monkeypatch):
    monkeypatch.setattr(profit, "OUT", tmp_path)
    manifest = {"protocol_sha256": profit.sha(profit.PROTOCOL), "state": "data_no_go"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="data gate forbids training"):
        profit.run("cpu")
    (tmp_path / "report.json").write_text("{}")
    with pytest.raises(FileExistsError, match="already exist"):
        profit.run("cpu")


def test_module_has_no_paper_websocket_or_order_execution_path():
    source = inspect.getsource(profit)
    assert "import websockets" not in source
    assert "sendorder" not in source.lower()
    assert not any(name.startswith(("paper", "order", "websocket")) for name in dir(profit))
