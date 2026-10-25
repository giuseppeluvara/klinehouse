"""Small frozen forecast adaptation fitted and selected before an untouched test.

Nine candidates use Timestral endpoint forecasts and six causal summaries of
hourly bars derived from raw seconds. Selection uses December 2025 only. The
selected rule is refitted on October--December and cannot learn from test labels.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .timestral_experiment import canonical, digest_file, exclusive_json, stamp, utc

FEATURES = ("return_4h", "return_24h", "return_168h", "realized_volatility_24h",
            "mean_taker_imbalance_24h", "log_trades_24h_minus_168h")
BASE_FEATURES = ("baseprice", "basecov")
DEFAULT_CONFIG = {
    "fit_start": "2025-10-01T00:00:00Z", "validation_start": "2025-12-01T00:00:00Z",
    "test_start": "2026-01-01T00:00:00Z", "test_end": "2026-04-01T00:00:00Z",
    "horizons": [4, 24], "min_train": 30, "min_validation": 20,
}
LEVELS = (.05, .5, .95)


def candidates():
    """Deterministic order also breaks exact validation-MAE ties by simplicity."""
    return [
        {"name": "raw_price", "kind": "raw", "source": "baseprice"},
        {"name": "raw_covariates", "kind": "raw", "source": "basecov"},
        *({"name": f"shrink_price_{scale:g}", "kind": "shrink", "scale": scale,
           "bias_fraction": .5} for scale in (.25, .5, 1.)),
        *({"name": f"ridge_{group}_{penalty:g}", "kind": "ridge", "group": group,
           "penalty": penalty, "clip_z": 8.} for group in ("combo", "state") for penalty in (1., 10.)),
    ]


def validate_config(config):
    config = {**DEFAULT_CONFIG, **(config or {})}
    times = [utc(config[key]) for key in ("fit_start", "validation_start", "test_start", "test_end")]
    if not all(left < right for left, right in zip(times, times[1:])):
        raise ValueError("require fit_start < validation_start < test_start < test_end")
    if any(t.hour or t.minute or t.second or t.microsecond for t in times):
        raise ValueError("split boundaries must be UTC midnights")
    if config["horizons"] != [4, 24]:
        raise ValueError("adaptation requires horizons [4,24]")
    for key in ("min_train", "min_validation"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"invalid {key}")
    for key in ("fit_start", "validation_start", "test_start", "test_end"):
        config[key] = stamp(config[key])
    return config


def code_hashes():
    root = Path(__file__).parent
    return {name: digest_file(root / name) for name in ("timestral_adaptation.py", "timestral_experiment.py")}


def extract_features(context, anchor):
    """Use exactly 256 complete past hours; reject shifts, gaps and future input."""
    anchor = utc(anchor)
    if anchor.minute or anchor.second or anchor.microsecond:
        raise ValueError("feature anchor must be an exact UTC hour")
    if len(context) != 256:
        raise ValueError("adaptation context must contain exactly 256 hours")
    expected = anchor - timedelta(hours=256)
    if any(utc(item.time) != expected + timedelta(hours=i) for i, item in enumerate(context)):
        raise ValueError("context must be consecutive hourly bars strictly before anchor")
    closes = np.asarray([item.log_close for item in context], dtype=float)
    state = np.asarray([item.features for item in context], dtype=float)
    if state.shape != (256, 3) or not np.isfinite(closes).all() or not np.isfinite(state).all():
        raise ValueError("non-finite or invalid hourly state")
    if (state[:, :2] < 0).any() or (np.abs(state[:, 2]) > 1 + 1e-10).any():
        raise ValueError("invalid volatility, trade intensity, or imbalance")
    values = [*(float(closes[-1] - closes[-1 - lag]) for lag in (4, 24, 168)),
              float(np.sqrt(np.square(state[-24:, 0]).sum())),
              float(state[-24:, 2].mean()), float(state[-24:, 1].mean() - state[-168:, 1].mean())]
    if not np.isfinite(values).all():
        raise ValueError("non-finite derived features")
    return dict(zip(FEATURES, values))


def _feature_vector(features, baseprice, basecov, names):
    if set(features) != set(FEATURES):
        raise ValueError("feature names differ from the frozen causal schema")
    values = {**features, "baseprice": baseprice, "basecov": basecov}
    if not np.isfinite([float(value) for value in values.values()]).all():
        raise ValueError("non-finite adaptation inputs")
    return np.array([values[name] for name in names], dtype=float)


def validate_examples(examples, config):
    """The fitting entry point rejects every label reaching into the test."""
    config = validate_config(config)
    start, split, cutoff = (utc(config[k]) for k in ("fit_start", "validation_start", "test_start"))
    rows, identities, anchors = [], set(), set()
    for original in examples:
        row = dict(original)
        anchor, end, horizon = utc(row["anchor"]), utc(row["label_end"]), row["horizon_hours"]
        if not isinstance(horizon, int) or horizon not in config["horizons"] or isinstance(horizon, bool):
            raise ValueError("unknown horizon")
        if not start <= anchor < cutoff or end > cutoff:
            raise ValueError("test or future labels are forbidden in development examples")
        if int(anchor.timestamp()) % (3600 * horizon) or anchor.microsecond:
            raise ValueError("anchor is not on its nonoverlapping UTC horizon grid")
        if end != anchor + timedelta(hours=horizon):
            raise ValueError("label endpoint does not match its horizon")
        if anchor < split < end:
            raise ValueError("fit label crosses into validation")
        if not isinstance(row["sample_id"], str) or not row["sample_id"]:
            raise ValueError("missing sample ID")
        if row["sample_id"] in identities or (horizon, anchor) in anchors:
            raise ValueError("duplicate sample ID or horizon anchor")
        identities.add(row["sample_id"])
        anchors.add((horizon, anchor))
        values = _feature_vector(row["features"], row["baseprice"], row["basecov"], BASE_FEATURES + FEATURES)
        if not np.isfinite(float(row["actual"])):
            raise ValueError("non-finite development target")
        rows.append({"sample_id": row["sample_id"], "anchor": stamp(anchor), "label_end": stamp(end),
                     "horizon_hours": horizon, "features": dict(zip(FEATURES, values[2:].tolist())),
                     "baseprice": float(values[0]), "basecov": float(values[1]), "actual": float(row["actual"])})
    return sorted(rows, key=lambda row: (row["horizon_hours"], row["anchor"], row["sample_id"]))


def fit_candidate(specification, rows, cutoff):
    """Estimate a single rule using only labels mature at the explicit cutoff."""
    if not rows or any(utc(row["label_end"]) > utc(cutoff) for row in rows):
        raise ValueError("fit requires nonempty, already matured labels")
    specification = dict(specification)
    if specification not in candidates():
        raise ValueError("candidate is not in the fixed grid")
    actual = np.array([row["actual"] for row in rows], dtype=float)
    result = {**specification, "fit_count": len(rows), "fit_cutoff": stamp(cutoff),
              "latest_fit_label_end": max(row["label_end"] for row in rows)}
    if specification["kind"] == "shrink":
        residual = actual - specification["scale"] * np.array([row["baseprice"] for row in rows])
        result["training_median_residual"] = float(np.median(residual))
        result["bias"] = specification["bias_fraction"] * result["training_median_residual"]
    elif specification["kind"] == "ridge":
        names = BASE_FEATURES + (FEATURES if specification["group"] == "state" else ())
        matrix = np.array([_feature_vector(row["features"], row["baseprice"], row["basecov"], names) for row in rows])
        mean, scale = matrix.mean(axis=0), np.maximum(matrix.std(axis=0), 1e-8)
        standardized = np.clip((matrix - mean) / scale, -specification["clip_z"], specification["clip_z"])
        standardized_mean = standardized.mean(axis=0)
        centered_inputs = standardized - standardized_mean
        centered = actual - actual.mean()
        coefficients = np.linalg.solve(centered_inputs.T @ centered_inputs + len(rows) * specification["penalty"] * np.eye(len(names)), centered_inputs.T @ centered)
        result.update(feature_order=list(names), mean=mean.tolist(), scale=scale.tolist(),
                      coefficients=coefficients.tolist(), intercept=float(actual.mean() - standardized_mean @ coefficients),
                      target_mean=float(actual.mean()),
                      objective="mean squared error + penalty * sum(coefficient squared); unpenalized intercept")
    return result


def predict_candidate(parameters, features, baseprice, basecov):
    _feature_vector(features, baseprice, basecov, BASE_FEATURES + FEATURES)
    kind = parameters["kind"]
    if kind == "raw":
        result = baseprice if parameters["source"] == "baseprice" else basecov
    elif kind == "shrink":
        result = parameters["scale"] * baseprice + parameters["bias"]
    elif kind == "ridge":
        values = _feature_vector(features, baseprice, basecov, parameters["feature_order"])
        standardized = np.clip((values - np.asarray(parameters["mean"])) / parameters["scale"], -parameters["clip_z"], parameters["clip_z"])
        result = parameters["intercept"] + standardized @ np.asarray(parameters["coefficients"])
    else:
        raise ValueError("unknown frozen candidate")
    if not np.isfinite(result):
        raise ValueError("non-finite adapted prediction")
    return float(result)


def _metrics(actual, predicted):
    error = predicted - actual
    return {"n": len(actual), "return_mae": float(np.abs(error).mean()),
            "return_rmse": float(np.sqrt(np.square(error).mean())),
            "direction_accuracy": float((np.sign(actual) == np.sign(predicted)).mean())}


def select_and_freeze(examples, output, config=None, metadata=None):
    """Select on validation, refit before test, and publish an exclusive bundle.

The caller must invoke this before reading test data. This function cannot
consume any test targets and records the actual wall-clock freeze time.
"""
    config = validate_config(config)
    rows = validate_examples(examples, config)
    selected, selection, development_predictions = {}, {}, []
    for horizon in config["horizons"]:
        group = [row for row in rows if row["horizon_hours"] == horizon]
        train = [row for row in group if utc(row["anchor"]) < utc(config["validation_start"])]
        validation = [row for row in group if utc(row["anchor"]) >= utc(config["validation_start"])]
        if len(train) < config["min_train"] or len(validation) < config["min_validation"]:
            raise ValueError(f"insufficient fit or validation samples for {horizon}h")
        actual = np.array([row["actual"] for row in validation])
        scores, forecasts, fitted = [], {}, {}
        for rank, specification in enumerate(candidates()):
            parameters = fit_candidate(specification, train, config["validation_start"])
            prediction = np.array([predict_candidate(parameters, row["features"], row["baseprice"], row["basecov"]) for row in validation])
            forecasts[specification["name"]], fitted[specification["name"]] = prediction, parameters
            scores.append({"candidate": specification["name"], "rank_for_ties": rank, **_metrics(actual, prediction)})
        best = min(scores, key=lambda score: (score["return_mae"], score["rank_for_ties"]))
        best_name = best["candidate"]
        specification = next(spec for spec in candidates() if spec["name"] == best_name)
        reference = _metrics(actual, np.zeros_like(actual))
        raw = next(score for score in scores if score["candidate"] == "raw_price")
        selection[str(horizon)] = {
            "selection_metric": "minimum validation MAE; exact ties use fixed candidate order",
            "selected": best_name, "fit_count": len(train), "validation_count": len(validation),
            "fit_sample_ids": [row["sample_id"] for row in train],
            "validation_sample_ids": [row["sample_id"] for row in validation],
            "candidates": scores, "unchanged_validation": reference,
            "validation_better_than_unchanged": best["return_mae"] < reference["return_mae"],
            "validation_better_than_raw_price": best["return_mae"] < raw["return_mae"],
            "candidate_parameters_fit_only": fitted,
            "gate_policy": "validation gates are descriptive; the best fixed-grid candidate is tested regardless",
        }
        residual = actual - forecasts[best_name]
        selected[str(horizon)] = {
            "candidate": best_name, "parameters": fit_candidate(specification, group, config["test_start"]),
            "refit_sample_ids": [row["sample_id"] for row in group],
            "refit_examples_sha256": hashlib.sha256(canonical(group)).hexdigest(),
            "residual_quantile_levels": list(LEVELS),
            "validation_residual_quantiles": np.quantile(residual, LEVELS).tolist(),
            "residual_calibration_count": len(validation),
            "residual_calibration_latest_label_end": max(row["label_end"] for row in validation),
            "residual_calibration_note": "December residuals from the October-November fit; December also selected the candidate; refitted point uses October-December; no coverage guarantee",
            "unchanged_return_quantiles": np.quantile([row["actual"] for row in group], LEVELS).tolist(),
            "unchanged_calibration_count": len(group),
            "unchanged_calibration_cutoff": config["test_start"],
        }
        for index, row in enumerate(validation):
            development_predictions.append({"sample_id": row["sample_id"], "anchor": row["anchor"],
                                            "label_end": row["label_end"], "horizon_hours": horizon,
                                            "actual": row["actual"],
                                            "predictions": {name: float(values[index]) for name, values in forecasts.items()}})
    bundle = {
        "schema": 1, "frozen_at": stamp(datetime.now(timezone.utc)), "config": config,
        "candidates": candidates(), "causal_feature_order": list(FEATURES), "code_sha256": code_hashes(),
        "numpy_version": np.__version__, "development_examples_sha256": hashlib.sha256(canonical(rows)).hexdigest(),
        "selection": selection, "selected": selected, "metadata": metadata or {},
        "test_policy": "parameters, scalers, selected candidate and residual distributions frozen; no test-label updates",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    exclusive_json(output / "bundle.json", bundle)
    exclusive_json(output / "bundle.sha256.json", {"sha256": digest_file(output / "bundle.json")})
    exclusive_json(output / "selection.json", selection)
    with (output / "development-forecasts.jsonl").open("x") as handle:
        for record in development_predictions:
            handle.write(canonical(record).decode() + "\n")
    (output / "development-forecasts.jsonl").chmod(0o444)
    return bundle


def load_bundle(output):
    output = Path(output)
    if digest_file(output / "bundle.json") != json.loads((output / "bundle.sha256.json").read_text())["sha256"]:
        raise ValueError("adaptation bundle changed after freezing")
    bundle = json.loads((output / "bundle.json").read_text())
    if bundle["code_sha256"] != code_hashes():
        raise ValueError("adaptation or imported code changed after freezing")
    validate_config(bundle["config"])
    if bundle["candidates"] != candidates() or bundle["causal_feature_order"] != list(FEATURES):
        raise ValueError("candidate or feature schema changed after freezing")
    return bundle


def predict(bundle, anchor, horizon, features, baseprice, basecov):
    """Pure frozen test prediction: no outcome argument or fitting operation."""
    anchor = utc(anchor)
    if not isinstance(horizon, int) or horizon not in bundle["config"]["horizons"] or isinstance(horizon, bool):
        raise ValueError("unknown horizon")
    if (not utc(bundle["config"]["test_start"]) <= anchor
            or anchor + timedelta(hours=horizon) > utc(bundle["config"]["test_end"])
            or int(anchor.timestamp()) % (3600 * horizon) or anchor.microsecond):
        raise ValueError("prediction is outside the frozen test anchor grid")
    return predict_candidate(bundle["selected"][str(horizon)]["parameters"], features, baseprice, basecov)
