"""Synthetic checks for the independent historical auditor; no market reads."""

import copy
from datetime import timedelta

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from research import timestral_adaptation as production
from scripts import analyze_timestral_adaptation as analysis
from scripts import verify_timestral_adaptation as verifier


def raw_hour(path, duplicate=False):
    start = verifier.utc("2025-09-01T00:00:00Z")
    index = np.arange(3600)
    timestamps = int(start.timestamp() * 1_000_000) + index * 1_000_000
    if duplicate:
        timestamps[-1] = timestamps[-2]
    prices = 100 * np.exp(.0001 * np.sin(index / 13))
    volume = 1 + index / 3600
    table = pa.table({"symbol": ["BTCUSDT"] * 3600,
                      "open_time": pa.array(timestamps).cast(pa.timestamp("us", tz="UTC")),
                      "close_time": pa.array(timestamps + 999999).cast(pa.timestamp("us", tz="UTC")),
                      "close": prices, "volume": volume, "trades": index % 17,
                      "taker_buy_base": volume * .6})
    pq.write_table(table, path)
    return start, prices, volume


@pytest.mark.parametrize("connection_timezone", ["UTC", "Asia/Kolkata"])
def test_direct_sql_and_arrow_reconstruct_raw_hour(tmp_path, connection_timezone):
    path = tmp_path / "hour.parquet"
    start, prices, _ = raw_hour(path)
    with duckdb.connect() as connection:
        connection.execute("SET TimeZone = ?", [connection_timezone])
        rows = verifier.direct_hours(connection, path, start, start + timedelta(hours=1))
    assert len(rows) == 1
    assert rows[0]["log_close"] == pytest.approx(np.log(prices[-1]))
    assert rows[0]["features"][0] == pytest.approx(np.sqrt(sum(np.diff(np.log(prices)) ** 2)))
    assert rows[0]["features"][2] == pytest.approx(.2)


def test_sql_cannot_hide_duplicate_second_or_missing_endpoint(tmp_path):
    path = tmp_path / "bad.parquet"
    start, _, _ = raw_hour(path, duplicate=True)
    with duckdb.connect() as connection, pytest.raises(ValueError, match="missing, duplicated"):
        verifier.direct_hours(connection, path, start, start + timedelta(hours=1))


def examples():
    rows = []
    for horizon in (4, 24):
        for day in range(92):
            anchor = verifier.utc("2025-10-01T00:00:00Z") + timedelta(days=day)
            features = {name: float(np.sin((day + i) / (i + 2))) for i, name in enumerate(verifier.FEATURES)}
            features[verifier.FEATURES[-1]] = 1.0
            # A train outlier with a constant column exercises clipping/scaling.
            base = 50.0 if day == 0 else .01 * np.sin(day)
            actual = .002 + .003 * np.sin(day / 4)
            rows.append({"sample_id": f"{horizon}-{day}", "anchor": verifier.stamp(anchor),
                         "label_end": verifier.stamp(anchor + timedelta(hours=horizon)), "horizon_hours": horizon,
                         "features": features, "baseprice": base, "basecov": .5 * base, "actual": actual})
    return sorted(rows, key=lambda row: (row["horizon_hours"], row["anchor"], row["sample_id"]))


def test_augmented_ridge_handles_clipping_and_keeps_mse_penalty_scale():
    rows = examples()[:92]
    spec = next(item for item in verifier.candidate_specs() if item["name"] == "ridge_state_1")
    independent = verifier.independent_fit(spec, rows)
    expected = production.fit_candidate(spec, rows, "2026-01-01")
    verifier.compare_parameters(independent, expected)
    duplicated = verifier.independent_fit(spec, rows + rows)
    verifier.close(independent["coefficients"], duplicated["coefficients"], "doubling n changed ridge solution")
    assert independent["intercept"] == pytest.approx(duplicated["intercept"], abs=1e-13)
    assert abs(independent["intercept"] - independent["target_mean"]) > 1e-8
    tampered = copy.deepcopy(expected)
    tampered["intercept"] += .0001
    with pytest.raises(ValueError, match="intercept"):
        verifier.compare_parameters(independent, tampered)


def frozen_case(tmp_path):
    rows = examples()
    bundle = production.select_and_freeze(rows, tmp_path / "frozen")
    test_rows = []
    for horizon in (4, 24):
        row = copy.deepcopy(next(row for row in rows if row["horizon_hours"] == horizon))
        row["anchor"] = "2026-01-01T00:00:00Z"
        parameters = bundle["selected"][str(horizon)]
        point = production.predict(bundle, row["anchor"], horizon, row["features"], row["baseprice"], row["basecov"])
        row["predictions"] = {"unchanged": {"point": 0., "quantiles": parameters["unchanged_return_quantiles"]},
                              "adapted": {"point": point, "quantiles": (point + np.array(parameters["validation_residual_quantiles"])).tolist()}}
        test_rows.append(row)
    return bundle, rows, test_rows


def test_independent_nine_candidate_selection_refit_and_test_predictions(tmp_path):
    bundle, rows, test_rows = frozen_case(tmp_path)
    report, validation = verifier.verify_selection(bundle, rows, test_rows)
    assert len(validation) == 62
    for horizon in (4, 24):
        assert report[str(horizon)]["fit_samples"] == 61
        assert report[str(horizon)]["validation_samples"] == 31
        assert report[str(horizon)]["refit_samples"] == 92
        assert report[str(horizon)]["maximum_adapted_point_difference"] < 1e-12


@pytest.mark.parametrize("kind", ["fit_label", "score", "ranking", "samples", "quantile", "prediction"])
def test_independent_audit_rejects_maturity_selection_and_output_changes(tmp_path, kind):
    bundle, rows, test_rows = frozen_case(tmp_path)
    if kind == "fit_label":
        bundle["selection"]["4"]["candidate_parameters_fit_only"]["raw_price"]["latest_fit_label_end"] = "2026-01-01T00:00:00Z"
    elif kind == "score":
        bundle["selection"]["4"]["candidates"][0]["return_mae"] += .01
    elif kind == "ranking":
        bundle["selected"]["4"]["candidate"] = "wrong"
    elif kind == "samples":
        bundle["selection"]["4"]["fit_sample_ids"].pop()
    elif kind == "quantile":
        bundle["selected"]["4"]["validation_residual_quantiles"][0] += .01
    else:
        test_rows[0]["predictions"]["adapted"]["point"] += .01
    with pytest.raises(ValueError):
        verifier.verify_selection(bundle, rows, test_rows)


def test_independent_metrics_months_and_gate_reject_report_drift():
    rows = []
    protocol = {"test_start": "2026-01-01T00:00:00Z", "test_end": "2026-04-01T00:00:00Z",
                "selected_candidates": {"4": "fixture", "24": "fixture"}}
    for horizon in (4, 24):
        for day in range(90):
            anchor = verifier.utc(protocol["test_start"]) + timedelta(days=day)
            outcome = .01 if day % 2 else -.02
            forecasts = {name: {"point": outcome * scale, "quantiles": [-.03, outcome * scale, .03]}
                         for name, scale in zip(verifier.MODELS, (0., .1, .2, .5))}
            rows.append({"sample_id": f"{horizon}-{day}", "anchor": verifier.stamp(anchor),
                         "horizon_hours": horizon, "reference_price": 100.,
                         "actual_price": float(100 * np.exp(outcome)), "actual_log_return": outcome,
                         "predictions": forecasts})
    document = {"horizons": {str(h): analysis.summarize([row for row in rows if row["horizon_hours"] == h], h, protocol) for h in (4, 24)},
                "all_horizons_pass": False}
    audited = verifier.verify_analysis(document, rows, protocol)
    assert audited["4"]["gate_passed"] is False  # Only 90 of the required 480 samples.
    assert audited["24"]["gate_passed"] is True
    for kind in ("overall", "monthly", "gate", "selection"):
        altered = copy.deepcopy(document)
        if kind == "overall":
            altered["horizons"]["4"]["metrics"]["adapted"]["return_mae"] += .001
        elif kind == "monthly":
            altered["horizons"]["24"]["monthly"]["2026-02"]["timestral_price"]["coverage90"] = 0
        elif kind == "gate":
            altered["horizons"]["24"]["gate"]["passed"] = False
        else:
            altered["horizons"]["24"]["selected_candidate"] = "wrong"
        with pytest.raises(ValueError):
            verifier.verify_analysis(altered, rows, protocol)
