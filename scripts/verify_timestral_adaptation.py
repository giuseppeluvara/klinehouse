#!/usr/bin/env python3
"""Independent read-only verification of the completed adaptation experiment.

Uses direct Parquet/SQL and NumPy; never imports experiment readers or models.
Run only after the separate historical test has completed. The output is exclusive.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FEATURES = ("return_4h", "return_24h", "return_168h", "realized_volatility_24h",
            "mean_taker_imbalance_24h", "log_trades_24h_minus_168h")
BASE = ("baseprice", "basecov")
COLUMNS = ("open_time", "close_time", "close", "volume", "trades", "taker_buy_base")
HOUR = timedelta(hours=1)
LEVELS = [.05, .5, .95]
MODELS = ("unchanged", "timestral_price", "timestral_covariates", "adapted")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def stamp(value):
    return value.isoformat().replace("+00:00", "Z")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def close(actual, expected, message, atol=1e-12, rtol=1e-9):
    first, second = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(first.shape == second.shape and np.isfinite(first).all() and np.isfinite(second).all()
            and np.allclose(first, second, atol=atol, rtol=rtol), message)


def indexed(rows):
    result = {row["sample_id"]: row for row in rows}
    require(len(result) == len(rows), "duplicate sample IDs")
    return result


def checked_path(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe manifest path")
    result = (root / path).resolve()
    require(result.is_relative_to(root.resolve()), "manifest path escapes root")
    return result


def verify_outputs(run):
    protocol, results = read_json(run / "protocol.json"), read_json(run / "results.json")
    require(sha(run / "protocol.json") == results["protocol_sha256"], "run protocol checksum mismatch")
    for name, digest in results["output_sha256"].items():
        require(sha(checked_path(run, name)) == digest, f"run output checksum mismatch: {name}")
    return protocol, results


def direct_hours(connection, path, start, end, column_hashes=None):
    """Bound memory to one daily file and validate features through independent SQL."""
    sql = """
        WITH seconds AS (
          SELECT date_trunc('hour', open_time) AS hour, epoch_us(open_time) AS ts,
                 epoch_us(close_time) AS closed, symbol, close, volume, trades, taker_buy_base,
                 lag(close) OVER (PARTITION BY date_trunc('hour', open_time) ORDER BY open_time) AS prior
          FROM read_parquet(?)
        )
        SELECT hour, count(*), count(DISTINCT ts), min(ts), max(ts),
               count(*) FILTER (WHERE symbol IS NULL OR symbol != 'BTCUSDT' OR ts % 1000000 != 0
                  OR closed < ts OR closed >= ts + 1000000 OR close <= 0 OR volume < 0
                  OR trades < 0 OR taker_buy_base < 0 OR taker_buy_base > volume + 1e-10
                  OR NOT isfinite(close) OR NOT isfinite(volume) OR NOT isfinite(taker_buy_base)),
               arg_max(close, ts),
               sqrt(sum(CASE WHEN prior IS NULL THEN 0 ELSE pow(ln(close / prior), 2) END)),
               sum(volume), sum(trades), sum(taker_buy_base)
        FROM seconds GROUP BY hour ORDER BY hour
    """
    aggregate = connection.execute(sql, [str(path)]).fetchall()
    with pq.ParquetFile(path) as parquet:
        table = parquet.read(columns=list(COLUMNS))
    require(all(table[name].null_count == 0 for name in COLUMNS), "null raw feature input")
    arrays = []
    for name in COLUMNS:
        column = table[name]
        array = (column.cast(pa.int64()).to_numpy().astype("<i8", copy=False) if name.endswith("time")
                 else column.to_numpy().astype("<f8", copy=False))
        arrays.append(array)
        if column_hashes is not None:
            column_hashes[name].update(array.tobytes())
    expected_count = int((end - start).total_seconds())
    require(expected_count <= 86400 and expected_count % 3600 == 0, "audit expects files of at most one full day")
    require(len(arrays[0]) == expected_count, "raw row count differs from complete file interval")
    expected = int(start.timestamp() * 1_000_000) + np.arange(expected_count, dtype=np.int64) * 1_000_000
    require(np.array_equal(arrays[0], expected), "raw seconds missing, duplicated or unordered")
    require(len(aggregate) == expected_count // 3600, "SQL hourly count mismatch")
    result = []
    for i, sql_row in enumerate(aggregate):
        opened = start + i * HOUR
        ts = int(opened.timestamp() * 1_000_000)
        require(sql_row[0] == opened and sql_row[1:6] == (3600, 3600, ts, ts + 3_599_000_000, 0),
                "SQL found incomplete or invalid raw hour")
        close_values, volume, trades, taker = [values[i * 3600:(i + 1) * 3600] for values in arrays[2:]]
        logs = np.log(close_values)
        volume_total = float(np.sum(volume))
        features = [float(np.linalg.norm(np.diff(logs))), float(np.log1p(np.sum(trades))),
                    float((2 * np.sum(taker) - volume_total) / volume_total) if volume_total else 0.]
        # The independent SQL uses log ratios, unlike the experiment's log differences.
        close(features, [sql_row[7], np.log1p(sql_row[9]),
                         (2 * sql_row[10] - sql_row[8]) / sql_row[8] if sql_row[8] else 0], "SQL/raw features disagree")
        close(close_values[-1], sql_row[6], "SQL endpoint price differs")
        # Preserve canonical floating-point serialization for the separate hash check.
        features[0] = float(np.sqrt(np.square(np.diff(logs)).sum()))
        result.append({"time": stamp(opened), "log_close": float(logs[-1]), "features": features})
    return result


def verify_archive(folder, protocol, results, start, end, connection):
    archive = folder / "archive"
    catalog = read_json(archive / "catalog.json")
    require(sha(archive / "catalog.json") == protocol["source"]["catalog_sha256"], "source catalog checksum mismatch")
    require((catalog["version"], catalog["market"], catalog["interval"]) == (1, "binance_spot", "1s"), "unexpected source format")
    entries = sorted(catalog["files"], key=lambda item: item["start"])
    require(entries == sorted(protocol["source_verified_entries"], key=lambda item: item["start"]), "protocol source entries differ")
    manifest = read_json(folder / "provenance.json")
    require((manifest["source"], manifest["market"], manifest["interval"], manifest["symbol"])
            == ("binance_public_data", "binance_spot", "1s", "BTCUSDT"), "unexpected source identity")
    require(utc(manifest["start"]) == start and utc(manifest["end"]) == end, "source period differs")
    receipts = sorted(manifest["files"], key=lambda item: item["archive_entry"]["start"])
    require([item["archive_entry"] for item in receipts] == entries, "provenance/archive binding differs")
    column_hashes = {name: hashlib.sha256() for name in COLUMNS}
    hours, cursor = [], start
    for entry, receipt in zip(entries, receipts):
        left, right = utc(entry["start"]), utc(entry["end"])
        require(left == cursor and right - left == timedelta(days=1), "source calendar gaps or overlapping entries")
        require(entry["symbol"] == "BTCUSDT" and entry["complete"] and entry["rows"] == 86400, "incomplete archive entry")
        name = f"BTCUSDT-1s-{left:%Y-%m-%d}.zip"
        url = f"https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/{name}"
        require(receipt["filename"] == name and receipt["url"] == url and receipt["checksum_url"] == url + ".CHECKSUM", "unexpected source receipt")
        require(sha(folder / "downloads" / name) == receipt["sha256"], "retained source ZIP checksum mismatch")
        checksum = (folder / "downloads" / f"{name}.CHECKSUM").read_text().split()
        require(len(checksum) == 2 and checksum[0].lower() == receipt["sha256"] and checksum[1].lstrip("*") == name, "official checksum receipt differs")
        path = checked_path(archive, entry["path"])
        require(sha(path) == entry["sha256"] and path.stat().st_size == entry["bytes"], "Parquet bytes differ")
        hours.extend(direct_hours(connection, path, left, right, column_hashes))
        cursor = right
    require(cursor == end and manifest["rows"] == len(hours) * 3600, "source does not cover its requested interval")
    digests = {name: state.hexdigest() for name, state in column_hashes.items()}
    data = results["data"]
    require(digests == data["source_column_sha256"], "direct source-column hash differs")
    require(hashlib.sha256(canonical(digests)).hexdigest() == data["source_rows_sha256"], "source row hash differs")
    feature_digest = hashlib.sha256()
    for hour in hours:
        feature_digest.update(canonical([hour["time"], hour["log_close"], hour["features"]]) + b"\n")
    require(feature_digest.hexdigest() == data["hourly_features_sha256"], "hourly feature hash differs")
    return hours, {"files": len(entries), "raw_rows": len(hours) * 3600, "hours": len(hours),
                   "catalog_sha256": sha(archive / "catalog.json"), "provenance_sha256": sha(folder / "provenance.json"),
                   "source_rows_sha256": data["source_rows_sha256"], "hourly_features_sha256": feature_digest.hexdigest()}


def reconstruct_features(context):
    require(len(context) == 256, "incomplete past context")
    log_closes = np.array([hour["log_close"] for hour in context])
    state = np.array([hour["features"] for hour in context])
    return dict(zip(FEATURES, [*(float(log_closes[-1] - log_closes[-lag - 1]) for lag in (4, 24, 168)),
                               float(np.linalg.norm(state[-24:, 0])), float(np.mean(state[-24:, 2])),
                               float(np.mean(state[-24:, 1]) - np.mean(state[-168:, 1]))]))


def verify_examples(predictions, outcomes, combined, hours, start, end, test=False):
    pred, out, examples = indexed(predictions), indexed(outcomes), indexed(combined)
    require(pred.keys() == out.keys() == examples.keys(), "missing predictions or scored outcomes")
    lookup = {utc(hour["time"]): hour for hour in hours}
    expected = {(horizon, stamp(start + i * timedelta(hours=horizon))) for horizon in (4, 24)
                for i in range(int((end - start).total_seconds() / (horizon * 3600)))}
    actual_anchors = {(row["horizon_hours"], row["anchor"]) for row in predictions}
    require(len(actual_anchors) == len(predictions) and actual_anchors == expected, "unexpected prediction anchor calendar")
    maximum = 0.
    rebuilt = []
    for key, row in pred.items():
        anchor, horizon = utc(row["anchor"]), row["horizon_hours"]
        label_end = anchor + timedelta(hours=horizon)
        context = [lookup[anchor - timedelta(hours=256 - i)] for i in range(256)]
        # Reconstruct complete target support, not only its last available price.
        future = [lookup[anchor + i * HOUR] for i in range(horizon)]
        require(row["label_end"] == stamp(label_end) and row["input_start"] == context[0]["time"]
                and row["input_end_exclusive"] == stamp(anchor), "input or target boundary differs")
        context_hash = hashlib.sha256(canonical([[hour["time"], hour["log_close"], hour["features"]] for hour in context])).hexdigest()
        require(context_hash == row["input_sha256"], "raw-derived context hash differs")
        sample_hash = hashlib.sha256(canonical(["BTCUSDT", row["anchor"], horizon, context_hash])).hexdigest()
        require(sample_hash == key, "raw-derived sample ID differs")
        features = reconstruct_features(context)
        close([row["features"][name] for name in FEATURES], list(features.values()), "causal raw-derived features differ")
        reference = context[-1]["log_close"]
        target = future[-1]["log_close"]
        close([row["reference_log_close"], row["reference_price"]], [reference, np.exp(reference)], "reference price differs")
        outcome, example = out[key], examples[key]
        require(outcome["status"] == "scored" and example == {**row, **outcome, **({} if test else {"actual": outcome["actual_log_return"]})}, "saved combined example differs")
        require(all(outcome[field] == row[field] for field in ("anchor", "label_end", "horizon_hours")), "outcome metadata differs")
        close([outcome["actual_price"], outcome["actual_log_return"]], [np.exp(target), target - reference], "actual price/target differs")
        maximum = max(maximum, abs(outcome["actual_log_return"] - (target - reference)))
        close([row["baseprice"], row["basecov"]], [row["predictions"][name]["point"] for name in ("timestral_price", "timestral_covariates")], "base forecasts differ from stored Timestral points")
        rebuilt.append({"sample_id": key, "anchor": row["anchor"], "label_end": row["label_end"],
                        "horizon_hours": horizon, "features": row["features"], "baseprice": row["baseprice"],
                        "basecov": row["basecov"], "actual": outcome["actual_log_return"]})
    return sorted(rebuilt, key=lambda row: (row["horizon_hours"], row["anchor"], row["sample_id"])), maximum


def candidate_specs():
    return [{"name": "raw_price", "kind": "raw", "source": "baseprice"},
            {"name": "raw_covariates", "kind": "raw", "source": "basecov"},
            *[{"name": f"shrink_price_{scale:g}", "kind": "shrink", "scale": scale, "bias_fraction": .5} for scale in (.25, .5, 1.)],
            *[{"name": f"ridge_{group}_{penalty:g}", "kind": "ridge", "group": group, "penalty": penalty, "clip_z": 8.}
              for group in ("combo", "state") for penalty in (1., 10.)]]


def matrix(rows, names):
    return np.array([[({**row["features"], "baseprice": row["baseprice"], "basecov": row["basecov"]})[name] for name in names] for row in rows])


def independent_fit(spec, rows):
    """Use augmented least squares, independent of the production normal equations."""
    actual = np.array([row["actual"] for row in rows])
    fitted = dict(spec)
    if spec["kind"] == "shrink":
        residuals = actual - spec["scale"] * np.array([row["baseprice"] for row in rows])
        fitted.update(training_median_residual=float(np.median(residuals)), bias=.5 * float(np.median(residuals)))
    elif spec["kind"] == "ridge":
        names = BASE + (FEATURES if spec["group"] == "state" else ())
        values = matrix(rows, names)
        center, scale = np.mean(values, axis=0), np.maximum(np.std(values, axis=0), 1e-8)
        values = np.clip((values - center) / scale, -8., 8.)
        design = np.column_stack([np.ones(len(rows)), values])
        regularizer = np.diag([0., *([np.sqrt(len(rows) * spec["penalty"])] * len(names))])
        coefficients = np.linalg.lstsq(np.vstack([design, regularizer]), np.concatenate([actual, np.zeros(len(names) + 1)]), rcond=None)[0]
        fitted.update(feature_order=list(names), mean=center.tolist(), scale=scale.tolist(),
                      coefficients=coefficients[1:].tolist(), intercept=float(coefficients[0]), target_mean=float(actual.mean()))
    return fitted


def independent_predict(parameters, rows):
    if parameters["kind"] == "raw":
        return np.array([row[parameters["source"]] for row in rows])
    if parameters["kind"] == "shrink":
        return parameters["scale"] * np.array([row["baseprice"] for row in rows]) + parameters["bias"]
    values = np.clip((matrix(rows, parameters["feature_order"]) - parameters["mean"]) / parameters["scale"], -8., 8.)
    return values @ parameters["coefficients"] + parameters["intercept"]


def compare_parameters(rebuilt, saved):
    for name, value in rebuilt.items():
        if isinstance(value, (float, int)) or (isinstance(value, list) and value and isinstance(value[0], (float, int))):
            close(value, saved[name], f"frozen parameter differs: {name}", atol=1e-11)
        else:
            require(value == saved[name], f"frozen parameter metadata differs: {name}")


def verify_selection(bundle, rows, test_rows):
    require(bundle["candidates"] == candidate_specs(), "candidate grid differs")
    require(bundle["causal_feature_order"] == list(FEATURES), "feature schema differs")
    require(bundle["development_examples_sha256"] == hashlib.sha256(canonical(rows)).hexdigest(), "development example hash differs")
    config, report, all_validation = bundle["config"], {}, []
    for horizon in (4, 24):
        group = [row for row in rows if row["horizon_hours"] == horizon]
        fit = [row for row in group if utc(row["anchor"]) < utc(config["validation_start"])]
        validation = [row for row in group if utc(row["anchor"]) >= utc(config["validation_start"])]
        require(all(utc(row["label_end"]) <= utc(config["validation_start"]) for row in fit), "future label in initial fit")
        require(all(utc(row["label_end"]) <= utc(config["test_start"]) for row in group), "test label in selection/refit")
        selected, selection = bundle["selected"][str(horizon)], bundle["selection"][str(horizon)]
        require([row["sample_id"] for row in fit] == selection["fit_sample_ids"] and [row["sample_id"] for row in validation] == selection["validation_sample_ids"], "selection sample IDs differ")
        actual = np.array([row["actual"] for row in validation])
        errors, validation_predictions = [], {}
        for rank, spec in enumerate(candidate_specs()):
            parameters = independent_fit(spec, fit)
            saved_parameters = selection["candidate_parameters_fit_only"][spec["name"]]
            compare_parameters(parameters, saved_parameters)
            require(saved_parameters["fit_count"] == len(fit)
                    and saved_parameters["fit_cutoff"] == config["validation_start"]
                    and saved_parameters["latest_fit_label_end"] == max(row["label_end"] for row in fit),
                    "initial fit maturity/count differs")
            predicted = independent_predict(parameters, validation)
            validation_predictions[spec["name"]] = predicted
            metrics = [np.abs(actual - predicted).mean(), np.sqrt(np.square(actual - predicted).mean()), (np.sign(actual) == np.sign(predicted)).mean()]
            saved = selection["candidates"][rank]
            require(saved["candidate"] == spec["name"] and saved["rank_for_ties"] == rank and saved["n"] == len(validation), "candidate ranking metadata differs")
            close(metrics, [saved[name] for name in ("return_mae", "return_rmse", "direction_accuracy")], "validation metrics differ")
            errors.append((float(metrics[0]), rank, spec))
        best = min(errors, key=lambda item: item[:2])[2]
        require(best["name"] == selected["candidate"] == selection["selected"], "independent candidate selection differs")
        require(selected["refit_sample_ids"] == [row["sample_id"] for row in group] and selected["refit_examples_sha256"] == hashlib.sha256(canonical(group)).hexdigest(), "refit examples differ")
        refitted = independent_fit(best, group)
        compare_parameters(refitted, selected["parameters"])
        require(selected["parameters"]["fit_count"] == len(group)
                and selected["parameters"]["fit_cutoff"] == config["test_start"]
                and selected["parameters"]["latest_fit_label_end"] == max(row["label_end"] for row in group), "refit maturity/count differs")
        residual_quantiles = np.quantile(actual - validation_predictions[best["name"]], LEVELS)
        unchanged_quantiles = np.quantile([row["actual"] for row in group], LEVELS)
        close(residual_quantiles, selected["validation_residual_quantiles"], "validation residual quantiles differ")
        close(unchanged_quantiles, selected["unchanged_return_quantiles"], "unchanged quantiles differ")
        require(selected["residual_calibration_count"] == len(validation)
                and selected["residual_calibration_latest_label_end"] == max(row["label_end"] for row in validation)
                and selected["unchanged_calibration_count"] == len(group)
                and selected["unchanged_calibration_cutoff"] == config["test_start"], "calibration maturity/count differs")
        for i, row in enumerate(validation):
            all_validation.append({"sample_id": row["sample_id"], "predictions": {name: values[i] for name, values in validation_predictions.items()}})
        test = [row for row in test_rows if row["horizon_hours"] == horizon]
        predicted = independent_predict(refitted, test)
        for point, row in zip(predicted, test):
            close(point, row["predictions"]["adapted"]["point"], "frozen adapted test point differs", atol=1e-11)
            close(point + residual_quantiles, row["predictions"]["adapted"]["quantiles"], "frozen adapted quantiles differ", atol=1e-11)
            close(row["predictions"]["unchanged"]["point"], 0., "unchanged point differs", atol=0, rtol=0)
            close(row["predictions"]["unchanged"]["quantiles"], unchanged_quantiles, "unchanged test quantiles differ")
        report[str(horizon)] = {"fit_samples": len(fit), "validation_samples": len(validation), "refit_samples": len(group),
                                "selected": best["name"], "test_predictions_verified": len(test),
                                "validation_mae_margin_to_runner_up": sorted(item[0] for item in errors)[1] - min(item[0] for item in errors),
                                "maximum_adapted_point_difference": float(max(abs(point - row["predictions"]["adapted"]["point"]) for point, row in zip(predicted, test)))}
    return report, all_validation


def independent_metrics(rows):
    """Scalar arithmetic/fsum independently checks vectorized analysis metrics."""
    actual = [row["actual_log_return"] for row in rows]
    result = {}
    for name in MODELS:
        point = [row["predictions"][name]["point"] for row in rows]
        quantiles = [row["predictions"][name]["quantiles"] for row in rows]
        errors = [forecast - observed for forecast, observed in zip(point, actual)]
        price_errors = [row["reference_price"] * math.exp(prediction) - row["actual_price"] for row, prediction in zip(rows, point)]
        called = [i for i, value in enumerate(point) if value != 0]
        correct = [((forecast > 0) - (forecast < 0)) == ((observed > 0) - (observed < 0)) for forecast, observed in zip(point, actual)]
        losses = [max(level * (observed - prediction), (level - 1) * (observed - prediction))
                  for observed, forecasts in zip(actual, quantiles) for level, prediction in zip(LEVELS, forecasts)]
        result[name] = {
            "samples": len(rows), "return_mae": math.fsum(abs(value) for value in errors) / len(rows),
            "return_rmse": math.sqrt(math.fsum(value * value for value in errors) / len(rows)),
            "price_mae_usdt": math.fsum(abs(value) for value in price_errors) / len(rows),
            "price_rmse_usdt": math.sqrt(math.fsum(value * value for value in price_errors) / len(rows)),
            "direction_accuracy": sum(correct) / len(rows), "direction_calls": len(called),
            "direction_call_fraction": len(called) / len(rows),
            "direction_accuracy_when_called": sum(correct[i] for i in called) / len(called) if called else None,
            "pinball_mean": math.fsum(losses) / len(losses),
            "coverage90": sum(forecasts[0] <= observed <= forecasts[2] for forecasts, observed in zip(quantiles, actual)) / len(rows),
            "interval_width90": math.fsum(forecasts[2] - forecasts[0] for forecasts in quantiles) / len(rows),
            "quantile_crossing": sum(forecasts[0] > forecasts[1] or forecasts[1] > forecasts[2] for forecasts in quantiles) / len(rows),
        }
    baseline = result["unchanged"]["return_mae"]
    for scores in result.values():
        scores["mae_skill_vs_unchanged"] = 1 - scores["return_mae"] / baseline if baseline else None
    return result


def compare_metrics(actual, saved):
    require(actual.keys() == saved.keys(), "analysis model set differs")
    for model, metrics in actual.items():
        for name, value in metrics.items():
            if value is None:
                require(saved[model][name] is None, f"analysis null metric differs: {model}/{name}")
            else:
                close(value, saved[model][name], f"analysis metric differs: {model}/{name}", atol=1e-10)


def verify_analysis(analysis, rows, protocol):
    report = {}
    for horizon in (4, 24):
        group = [row for row in rows if row["horizon_hours"] == horizon]
        saved = analysis["horizons"][str(horizon)]
        scores = independent_metrics(group)
        compare_metrics(scores, saved["metrics"])
        require(saved["samples"] == len(group) and saved["sample_ids"] == [row["sample_id"] for row in group], "analysis test sample IDs/count differs")
        require(saved["selected_candidate"] == protocol["selected_candidates"][str(horizon)], "analysis selected candidate differs")
        months = sorted({row["anchor"][:7] for row in group})
        require(set(months) == set(saved["monthly"]), "analysis month set differs")
        wins = 0
        for month in months:
            monthly = independent_metrics([row for row in group if row["anchor"][:7] == month])
            compare_metrics(monthly, saved["monthly"][month])
            wins += monthly["adapted"]["return_mae"] < monthly["unchanged"]["return_mae"]
        require(wins == saved["monthly_mae_wins"], "monthly win count differs")
        contrasts = saved["paired_comparison"]["contrasts"]
        for contrast, values in contrasts.items():
            candidate, reference = contrast.split("_minus_")
            close(values["mean_mae_difference"], scores[candidate]["return_mae"] - scores[reference]["return_mae"], "paired mean MAE difference differs")
        interval = contrasts["adapted_minus_unchanged"]["percentile97_5"]
        require(len(interval) == 2 and np.isfinite(interval).all() and interval[0] <= interval[1], "invalid primary interval")
        criteria = {
            "minimum_samples": len(group) >= {4: 480, 24: 80}[horizon],
            "strict_mae_improvement": scores["adapted"]["return_mae"] < scores["unchanged"]["return_mae"],
            "adjusted_upper_ci_below_zero": interval[1] < 0,
            "at_least_two_of_three_months_improve": len(months) == 3 and wins >= 2,
            "rmse_no_worse": scores["adapted"]["return_rmse"] <= scores["unchanged"]["return_rmse"],
        }
        require(criteria == saved["gate"]["criteria"] and all(criteria.values()) == saved["gate"]["passed"], "historical gate differs")
        report[str(horizon)] = {"models_compared": list(MODELS), "months_compared": months,
                                "metrics_per_model": list(scores["adapted"]), "gate_recalculated": True,
                                "gate_passed": all(criteria.values()), "primary_interval_source": "saved analysis, not independently resampled"}
    require(analysis["all_horizons_pass"] == all(value["gate_passed"] for value in report.values()), "combined gate differs")
    return report


def verify(root, output):
    root, output = Path(root).resolve(), Path(output)
    require(not output.exists(), "audit output already exists")
    started = time.perf_counter()
    dev_run, test_run, frozen = root / "development-run", root / "test-run", root / "frozen"
    dev_protocol, dev_results = verify_outputs(dev_run)
    protocol, results = verify_outputs(test_run)
    bundle, registered = read_json(frozen / "bundle.json"), read_json(frozen / "protocol.json")
    require(sha(frozen / "bundle.json") == registered["freeze_sha256"] == read_json(frozen / "bundle.sha256.json")["sha256"], "bundle checksum differs")
    require(sha(frozen / "protocol.json") == protocol["frozen_protocol_sha256"] == read_json(frozen / "protocol.sha256.json")["sha256"], "frozen protocol checksum differs")
    require(read_json(frozen / "selection.json") == bundle["selection"], "saved selection differs")
    config = registered["config"]
    require(config == dev_protocol["config"] == protocol["config"], "configuration differs between stages")
    selections = {key: value["candidate"] for key, value in bundle["selected"].items()}
    require(selections == registered["selected_candidates"] == protocol["selected_candidates"], "frozen/runtime selected candidate differs")
    for name, digest in registered["code_sha256"].items():
        require(sha(checked_path(Path(__file__).resolve().parents[1], name)) == digest, f"frozen execution code differs: {name}")
    require(sha(dev_run / "results.json") == registered["development_results_sha256"], "development result changed after freeze")
    require(dev_results["output_sha256"] == registered["development_output_sha256"], "development outputs changed after freeze")
    require(dev_results["checkpoint_files_sha256"] == results["checkpoint_files_sha256"], "model checkpoint changed between development and test")
    checkpoint = Path(config["model_cache"]) / "models--mistralai--Timestral-5M" / "snapshots" / "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"
    pinned = {"model.safetensors": "7ace108c4987e7f2a99a75a445240019d1d31c2117139d1bd05ffdfd25aca738",
              "config.json": "b5f27f61513f8fd1a3aa996f3720d3679cb40bce5bcb08eb3dc320ce726e7c2f"}
    for name, digest in results["checkpoint_files_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and sha(checkpoint / name) == digest, "checkpoint file checksum differs")
    require(all(results["checkpoint_files_sha256"].get(name) == digest for name, digest in pinned.items()), "pinned Timestral model identity differs")
    opened = read_json(frozen / "test-opened.json")
    test_source = read_json(root / "test" / "provenance.json")
    require(utc(bundle["frozen_at"]) <= utc(registered["registered_at"]) <= utc(test_source["retrieved_at"]) <= utc(opened["opened_at"]), "acquisition/freeze/open timestamps out of order")
    require(opened["frozen_protocol_sha256"] == sha(frozen / "protocol.json") and Path(opened["output"]).resolve() == test_run, "test opening record differs")
    require(sha(root / "test" / "provenance.json") == protocol["source_provenance_sha256"], "test source provenance changed")
    require(utc(config["test_end"]) < utc("2026-08-08T00:00:00Z"), "protected holdout boundary violated")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=1")
        connection.execute("SET memory_limit='128MB'")
        connection.execute("SET preserve_insertion_order=true")
        connection.execute("SET temp_directory=''")
        dev_hours, dev_source = verify_archive(root / "development", dev_protocol, dev_results,
                                              utc(config["development_start"]), utc(config["test_start"]), connection)
        test_hours, test_source_audit = verify_archive(root / "test", protocol, results,
                                                     utc(config["test_start"]), utc(config["test_end"]), connection)
    finally:
        connection.close()
    require(dev_hours == read_rows(dev_run / "hours.jsonl"), "independent hourly reconstruction differs from development cache")
    dev_rows, dev_error = verify_examples(read_rows(dev_run / "predictions.jsonl"), read_rows(dev_run / "outcomes.jsonl"), read_rows(dev_run / "examples.jsonl"),
                                          dev_hours, utc(config["fit_start"]), utc(config["test_start"]))
    predictions = read_rows(test_run / "predictions.jsonl")
    require(all(utc(row["computed_at"]) >= utc(opened["opened_at"]) and row["historical_exploratory"] is True for row in predictions), "incorrect test computation timestamps/status")
    evaluation = read_rows(test_run / "evaluation.jsonl")
    _, test_error = verify_examples(predictions, read_rows(test_run / "outcomes.jsonl"), evaluation,
                                    dev_hours + test_hours, utc(config["test_start"]), utc(config["test_end"]), test=True)
    ranking, validation_predictions = verify_selection(bundle, dev_rows, predictions)
    saved_validation = indexed(read_rows(frozen / "development-forecasts.jsonl"))
    require(len(validation_predictions) == len(saved_validation), "validation forecast count differs")
    for row in validation_predictions:
        saved = saved_validation[row["sample_id"]]
        close(list(row["predictions"].values()), [saved["predictions"][name] for name in row["predictions"]], "saved development forecasts differ")
    analysis = read_json(root / "analysis" / "analysis.json")
    require(analysis["protocol_sha256"] == results["protocol_sha256"] and analysis["input_sha256"] == results["output_sha256"], "analysis input hashes differ")
    require(analysis["analysis_code_sha256"] == registered["code_sha256"]["scripts/analyze_timestral_adaptation.py"], "analysis code hash differs")
    metrics_audit = verify_analysis(analysis, evaluation, protocol)
    report = {"passed": True, "verified_at": stamp(datetime.now(timezone.utc)), "seconds": time.perf_counter() - started,
              "verifier_path": str(Path(__file__).resolve()), "verifier_sha256": sha(__file__),
              "method": "direct per-file DuckDB SQL and PyArrow, NumPy causal reconstruction; ridge by augmented least squares",
              "source": {"development": dev_source, "test": test_source_audit}, "horizons": ranking,
              "analysis_sha256": sha(root / "analysis" / "analysis.json"), "metric_checks": metrics_audit,
              "maximum_target_return_difference": max(dev_error, test_error),
              "protocol_sha256": sha(test_run / "protocol.json"), "frozen_protocol_sha256": sha(frozen / "protocol.json"),
              "bundle_sha256": sha(frozen / "bundle.json"), "results_sha256": sha(test_run / "results.json"),
              "output_sha256": results["output_sha256"],
              "checkpoint_files_sha256": results["checkpoint_files_sha256"],
              "limits": ["Timestral inference is not rerun; its saved outputs and identities are hash-verified.",
                         "Timestamp evidence verifies recorded ordering, not an external proof of historical ignorance.",
                         "This historical audit establishes no prospective, statistical-independence or profitability guarantee."]}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.output)
    print(json.dumps({key: result[key] for key in ("passed", "seconds", "horizons")}, indent=2))


if __name__ == "__main__":
    main()
