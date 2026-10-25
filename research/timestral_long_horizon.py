"""Preregistered historical 4-hour and 1-day forecasts built from raw 1s candles.

Hourly bars require all 3,600 exact UTC seconds. A single chronological stream
issues forecasts before consuming target hours; each horizon calibrates only on
its own matured residuals. This is an exploratory historical simulation, not a
prospective trading record. The earlier experiment remains frozen and unchanged.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa

from . import timestral_experiment as shared

FEATURES = ("intrahour_realized_volatility", "log1p_trades", "taker_imbalance")
HOUR_US = 3_600_000_000
SOURCE_COLUMNS = shared.SOURCE_COLUMNS
MODELS = shared.MODELS
utc, stamp, canonical = shared.utc, shared.stamp, shared.canonical


def validate_config(config):
    required = {"symbol", "start", "end", "evaluation_start", "archive_root", "source",
                "lookback", "horizons", "min_calibration", "max_hours", "max_samples",
                "batch_size", "torch_threads", "model_cache", "local_files_only"}
    if not required <= config.keys():
        raise ValueError(f"missing config fields: {sorted(required - config.keys())}")
    if config["symbol"] != "BTCUSDT" or config["source"] != "archive":
        raise ValueError("this protocol requires the BTCUSDT archive")
    if config["horizons"] != [4, 24] or config["local_files_only"] is not True:
        raise ValueError("require horizons=[4,24] hours and local_files_only=true")
    for key in ("lookback", "min_calibration", "max_hours", "max_samples", "batch_size", "torch_threads"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"invalid {key}")
    if config["lookback"] < 32:
        raise ValueError("lookback must be at least 32 hourly bars")
    start, evaluation, end = (utc(config[key]) for key in ("start", "evaluation_start", "end"))
    if not start < evaluation < end:
        raise ValueError("require start < evaluation_start < end")
    if any(value.hour or value.minute or value.second or value.microsecond for value in (start, evaluation, end)):
        raise ValueError("period boundaries must be UTC midnights")
    if (end - start).total_seconds() / 3600 > config["max_hours"]:
        raise ValueError("period exceeds the preregistered hour budget")
    return config


def code_hashes():
    root = Path(__file__).parent
    names = ("timestral_long_horizon.py", "timestral_experiment.py", "archive.py", "market.py", "db.py")
    return {**{name: shared.digest_file(root / name) for name in names},
            "scripts/analyze_timestral_long_horizon.py": shared.digest_file(root.parent / "scripts/analyze_timestral_long_horizon.py")}


def preregister(config, output):
    config = validate_config(dict(config))
    for key in ("archive_root", "model_cache"):
        config[key] = str(Path(config[key]).resolve())
    snapshot, entries = shared.verified_source_snapshot(config)
    protocol = {
        "schema": 1, "registered_at": stamp(datetime.now(timezone.utc)), "config": config,
        "model": {"id": shared.MODEL_ID, "code_revision": shared.CODE_REVISION,
                  "weight_revision": shared.WEIGHT_REVISION, "device": "cpu",
                  "precision": "float32", "zero_shot": True, "symmetric_averaging": True,
                  "clip_nonnegative": False},
        "source": snapshot, "source_verified_entries": entries, "code_sha256": code_hashes(),
        "environment": shared.environment(), "models": list(MODELS), "features": list(FEATURES),
        "quantiles_for_metrics": list(shared.LEVELS),
        "target": "log(close at anchor+H hours minus 1 second / close at anchor minus 1 second)",
        "context": "lookback complete hourly bars from exactly 3600 raw UTC seconds each; all input timestamps precede anchor",
        "hourly_features": "RV=sqrt(sum of squared 3599 within-hour log-close returns); log1p(sum trades); (2*sum taker_buy_base-sum volume)/sum volume, or zero for zero volume",
        "anchors": "separate UTC grids for H=4 and H=24 hours; stride=H; target_end<=end",
        "calibration": "separate expanding empirical residual history per horizon; only label_end<=anchor; minimum config.min_calibration",
        "baseline": "unchanged=0; local_drift=H*mean(diff(context log-close)); baseline interval quantiles from matured residuals",
        "chronology": "mature previous labels, then publish current predictions, then consume future hours; missing targets invalidate all model scores equally",
        "split": "evaluation_start<=anchor and anchor+H<=end; monthly UTC descriptive folds, no tuning or model selection",
        "inference": "Timestral receives context only; all available model quantiles saved; endpoint forecasts only",
        "historical_status": "exploratory: pretraining contamination cannot be excluded; no prospective or PnL claim",
        "comparisons": "paired MAE-difference bootstrap of seven-day calendar blocks starting at evaluation_start; include short final block; seed 7, 2000 draws; descriptive only",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    shared.exclusive_json(output / "protocol.json", protocol)
    shared.exclusive_json(output / "protocol.sha256.json", {"sha256": shared.digest_file(output / "protocol.json")})
    return protocol


def load_protocol(output):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    if shared.digest_file(output / "protocol.json") != json.loads((output / "protocol.sha256.json").read_text())["sha256"]:
        raise ValueError("protocol changed after preregistration")
    if code_hashes() != protocol["code_sha256"]:
        raise ValueError("experiment or imported code changed after preregistration; create a distinct run")
    validate_config(protocol["config"])
    return protocol


@dataclass(frozen=True)
class Hour:
    time: datetime
    log_close: float
    features: tuple[float, float, float]


def hourly_features(batches, audit):
    """Vectorized bounded batches and one hour, with batch-independent hashes.

The input stream hash covers six normalized little-endian columns separately:
timestamps are int64 UTC microseconds; numeric projected values are float64.
Column digests are combined in canonical JSON, independent of batch boundaries.
The archive manifest separately pins all original OHLCV columns and file bytes.
"""
    previous, current_hour, pieces = None, None, []
    hashes = {name: hashlib.sha256() for name in SOURCE_COLUMNS}
    feature_hash = hashlib.sha256()
    audit.update(raw_rows=0, valid_hours=0, invalid_hours=0, max_input_batch_rows=0)

    def finish():
        if not pieces:
            return None
        columns = [np.concatenate([part[i] for part in pieces]) for i in range(len(SOURCE_COLUMNS))]
        opened, closed, close, volume, trades, taker = columns
        aligned = len(opened) == 3600 and np.array_equal(opened, current_hour * HOUR_US + np.arange(3600) * 1_000_000)
        valid = aligned and np.all((closed >= opened) & (closed < opened + 1_000_000))
        valid = valid and all(np.isfinite(values).all() for values in columns[2:])
        valid = valid and (close > 0).all() and all((values >= 0).all() for values in columns[3:])
        valid = valid and (taker <= volume + 1e-10).all()
        if not valid:
            audit["invalid_hours"] += 1
            return None
        logs = np.log(close)
        total_volume = float(volume.sum())
        features = (float(np.sqrt(np.square(np.diff(logs)).sum())), float(np.log1p(trades.sum())),
                    float((2 * taker.sum() - total_volume) / total_volume) if total_volume else 0.0)
        result = Hour(datetime.fromtimestamp(current_hour * 3600, timezone.utc), float(logs[-1]), features)
        feature_hash.update(canonical([stamp(result.time), result.log_close, result.features]) + b"\n")
        audit["valid_hours"] += 1
        return result

    for batch in batches:
        columns = []
        for name in SOURCE_COLUMNS:
            column = batch.column(batch.schema.get_field_index(name))
            if column.null_count:
                raise ValueError(f"null source values in {name}")
            if name in ("open_time", "close_time"):
                values = column.cast(pa.timestamp("us", tz="UTC")).cast(pa.int64()).to_numpy(zero_copy_only=False).astype("<i8", copy=False)
            else:
                values = column.to_numpy(zero_copy_only=False).astype("<f8", copy=False)
            hashes[name].update(values.tobytes())
            columns.append(values)
        opened = columns[0]
        if not len(opened):
            continue
        if np.any(np.diff(opened) <= 0) or (previous is not None and opened[0] <= previous):
            raise ValueError("source timestamps must be strictly increasing")
        previous = int(opened[-1])
        audit["raw_rows"] += len(opened)
        audit["max_input_batch_rows"] = max(audit["max_input_batch_rows"], len(opened))
        keys = opened // HOUR_US
        boundaries = [0, *(np.flatnonzero(np.diff(keys)) + 1).tolist(), len(keys)]
        for left, right in zip(boundaries, boundaries[1:]):
            key = int(keys[left])
            if current_hour is not None and key != current_hour:
                item = finish()
                if item:
                    yield item
                pieces = []
            current_hour = key
            pieces.append(tuple(values[left:right] for values in columns))
    item = finish()
    if item:
        yield item
    column_hashes = {key: value.hexdigest() for key, value in hashes.items()}
    audit.update(source_column_sha256=column_hashes,
                 source_rows_sha256=hashlib.sha256(canonical(column_hashes)).hexdigest(),
                 hourly_features_sha256=feature_hash.hexdigest(),
                 source_hash_encoding="canonical JSON of per-column SHA256 over little-endian int64 UTC microseconds or float64 values")


def read_hours(config, audit):
    from .market import iter_market
    batches = iter_market(config["symbol"], config["start"], config["end"],
                          archive_root=config["archive_root"], source="archive", timeframe="1s",
                          columns=list(SOURCE_COLUMNS), batch_size=config["batch_size"])
    try:
        yield from hourly_features(batches, audit)
    finally:
        batches.close()


def forecast(anchor, context, calibration, config, horizon, model):
    """Only past context is accepted; no outcome or future-window argument exists."""
    local = {**config, "horizon": horizon}
    predictions, evidence = calibration.forecast(anchor, context, local)
    elapsed = {}
    for name, covariates in (("timestral_price", False), ("timestral_covariates", True)):
        started = time.perf_counter()
        predictions[name] = model.predict(context, local, covariates)
        elapsed[name] = time.perf_counter() - started
    reference = context[-1].log_close
    for prediction in predictions.values():
        prediction.setdefault("all_quantile_levels", list(shared.LEVELS))
        prediction.setdefault("all_return_quantiles", list(prediction["quantiles"]))
        prediction["point_price"] = float(np.exp(reference + prediction["point"]))
        prediction["point_return_percent"] = float(np.expm1(prediction["point"]) * 100)
        prediction["price_quantiles"] = np.exp(reference + np.asarray(prediction["quantiles"])).tolist()
        prediction["all_price_quantiles"] = np.exp(reference + np.asarray(prediction["all_return_quantiles"])).tolist()
        if not np.isfinite(prediction["all_price_quantiles"]).all():
            raise ValueError("non-finite forecast price")
    context_hash = hashlib.sha256(canonical([[stamp(x.time), x.log_close, x.features] for x in context])).hexdigest()
    sample_id = hashlib.sha256(canonical([config["symbol"], stamp(anchor), horizon, context_hash])).hexdigest()
    return {"sample_id": sample_id, "anchor": stamp(anchor), "horizon_hours": horizon,
            "label_end": stamp(anchor + timedelta(hours=horizon)), "input_start": stamp(context[0].time),
            "input_end_exclusive": stamp(anchor), "input_sha256": context_hash,
            "reference_log_close": reference, "reference_price": float(np.exp(reference)),
            "calibration": evidence, "predictions": predictions, "inference_seconds": elapsed,
            "fold": anchor.strftime("%Y-%m"), "mode": "historical_exploratory",
            "computed_at": stamp(datetime.now(timezone.utc))}


def flat_forecasts(record):
    for name, prediction in record["predictions"].items():
        yield {"sample_id": record["sample_id"], "anchor": record["anchor"], "label_end": record["label_end"],
               "horizon_hours": record["horizon_hours"], "model": name, "reference_price": record["reference_price"],
               "point_log_return": prediction["point"], "point_return_percent": prediction["point_return_percent"],
               "point_price": prediction["point_price"],
               **{f"price_q{int(level * 100):02d}": value for level, value in zip(shared.LEVELS, prediction["price_quantiles"])},
               **{f"log_return_q{int(level * 100):02d}": value for level, value in zip(shared.LEVELS, prediction["quantiles"])}}


def metrics(records):
    result = shared.metric_summary(records)
    actual = np.array([item[1] for item in records])
    reference = np.array([item[0]["reference_price"] for item in records])
    actual_price = reference * np.exp(actual)
    for name, summary in result.items():
        predicted = np.array([item[0]["predictions"][name]["point"] for item in records])
        called = predicted != 0
        price_error = reference * np.exp(predicted) - actual_price
        summary.update(price_mae=float(np.abs(price_error).mean()),
                       price_rmse=float(np.sqrt(np.square(price_error).mean())),
                       direction_calls=int(called.sum()),
                       direction_call_fraction=float(called.mean()),
                       direction_accuracy_when_called=float((np.sign(predicted[called]) == np.sign(actual[called])).mean()) if called.any() else None)
    for name, summary in result.items():
        summary["mae_skill_vs_unchanged"] = 1 - summary["return_mae"] / result["unchanged"]["return_mae"] if result["unchanged"]["return_mae"] else None
        summary["rmse_skill_vs_unchanged"] = 1 - summary["return_rmse"] / result["unchanged"]["return_rmse"] if result["unchanged"]["return_rmse"] else None
    return result


def backtest(output, model_factory=shared.Timestral):
    from contextlib import ExitStack
    output = Path(output)
    protocol = load_protocol(output)
    config = protocol["config"]
    started = time.perf_counter()
    if shared.verified_source_snapshot(config)[0] != protocol["source"]:
        raise ValueError("source catalog changed after preregistration")
    source_verification_seconds = time.perf_counter() - started
    end, evaluation = utc(config["end"]), utc(config["evaluation_start"])
    horizons = config["horizons"]
    calibration = {h: shared.Calibration() for h in horizons}
    pending, records = {}, {h: [] for h in horizons}
    emitted = {h: [] for h in horizons}
    audit = {"hour_gap_resets": 0, "invalid_targets": {str(h): 0 for h in horizons},
             "insufficient_calibration": {str(h): 0 for h in horizons}}
    files = ["predictions.jsonl", "outcomes.jsonl", "forecast-rows.jsonl", *(f"calibration-{h}h.jsonl" for h in horizons)]
    with ExitStack() as stack:
        # Reserve every file before loading the model; never overwrite a partial run.
        handles = {name: stack.enter_context((output / name).open("x")) for name in files}
        model = model_factory(config)
        buffer, previous, count = deque(maxlen=config["lookback"]), None, 0
        hours = read_hours(config, audit)
        stack.callback(hours.close)

        def invalidate(horizon, reason):
            item = pending.pop(horizon)
            if item["record"] is not None:
                audit["invalid_targets"][str(horizon)] += 1
                shared.append_record(handles["outcomes.jsonl"], {
                    "sample_id": item["record"]["sample_id"], "anchor": stamp(item["anchor"]),
                    "horizon_hours": horizon, "label_end": stamp(item["label_end"]),
                    "status": "invalid_target", "reason": reason})

        for hour in hours:
            if not utc(config["start"]) <= hour.time < end:
                raise ValueError("hour outside preregistered range")
            if previous is not None and hour.time <= previous:
                raise ValueError("hours must be strictly increasing")
            if previous is not None and hour.time != previous + timedelta(hours=1):
                buffer.clear()
                audit["hour_gap_resets"] += 1
                for horizon in list(pending):
                    invalidate(horizon, "missing or incomplete raw-second hour in target")
            previous = hour.time
            buffer.append(hour)
            now = hour.time + timedelta(hours=1)
            # First mature earlier outcomes; equal-boundary labels are now observable.
            for horizon, item in list(pending.items()):
                if item["label_end"] == now:
                    outcome = hour.log_close - item["reference_log_close"]
                    calibration[horizon].add(now, outcome, item["drift"])
                    history = {"anchor": stamp(item["anchor"]), "label_end": stamp(now),
                               "horizon_hours": horizon, "phase": item["phase"],
                               "return": outcome, "drift_prediction": item["drift"],
                               "drift_residual": outcome - item["drift"]}
                    shared.append_record(handles[f"calibration-{horizon}h.jsonl"], history)
                    if item["record"] is not None:
                        record = item["record"]
                        shared.append_record(handles["outcomes.jsonl"], {
                            "sample_id": record["sample_id"], "anchor": record["anchor"],
                            "horizon_hours": horizon, "label_end": stamp(now), "status": "scored",
                            "return": outcome, "return_percent": float(np.expm1(outcome) * 100),
                            "actual_price": float(np.exp(hour.log_close))})
                        records[horizon].append((record, outcome))
                    del pending[horizon]
            if len(buffer) != config["lookback"]:
                continue
            context = list(buffer)
            for horizon in horizons:
                if int(now.timestamp()) % (3600 * horizon) or now + timedelta(hours=horizon) > end:
                    continue
                count += 1
                if count > config["max_samples"]:
                    raise ValueError("sample budget exceeded")
                record = None
                if now >= evaluation:
                    if sum(row[0] <= now for row in calibration[horizon].records) < config["min_calibration"]:
                        audit["insufficient_calibration"][str(horizon)] += 1
                    else:
                        record = forecast(now, context, calibration[horizon], config, horizon, model)
                        shared.append_record(handles["predictions.jsonl"], record)
                        emitted[horizon].append(record)
                        for row in flat_forecasts(record):
                            shared.append_record(handles["forecast-rows.jsonl"], row)
                if horizon in pending:
                    raise ValueError("overlapping pending label in a nonoverlapping horizon")
                pending[horizon] = {"anchor": now, "label_end": now + timedelta(hours=horizon),
                                    "reference_log_close": context[-1].log_close,
                                    "drift": shared.drift(context, horizon), "record": record,
                                    "phase": "evaluation" if now >= evaluation else "warmup"}
        for horizon in list(pending):
            invalidate(horizon, "source ended before the complete target")
    if shared.source_snapshot(config) != protocol["source"]:
        raise ValueError("source catalog changed during evaluation")
    for name in files:
        (output / name).chmod(0o444)
    summaries = {}
    for horizon, evaluated in records.items():
        folds = sorted({item[0]["fold"] for item in evaluated})
        expected = int((end - evaluation).total_seconds() // (horizon * 3600))
        summaries[str(horizon)] = {
            "expected_anchors_if_complete": expected, "scored_anchors": len(evaluated),
            "emitted_predictions": len(emitted[horizon]),
            "skipped_missing_context_or_anchor": expected - len(emitted[horizon]) - audit["insufficient_calibration"][str(horizon)],
            "skipped_insufficient_calibration": audit["insufficient_calibration"][str(horizon)],
            "invalidated_targets": audit["invalid_targets"][str(horizon)],
            "sample_ids": [item[0]["sample_id"] for item in evaluated],
            "metrics": metrics(evaluated) if evaluated else {},
            "monthly": {fold: metrics([item for item in evaluated if item[0]["fold"] == fold]) for fold in folds},
            "first_calibration_count": emitted[horizon][0]["calibration"]["count"] if emitted[horizon] else None,
            "final_calibration_count": len(calibration[horizon].records),
            "inference_seconds": {name: {
                "sum": float(sum(item["inference_seconds"][name] for item in emitted[horizon])),
                "median": float(np.median([item["inference_seconds"][name] for item in emitted[horizon]])) if emitted[horizon] else None,
            } for name in ("timestral_price", "timestral_covariates")},
        }
    audit["candidate_anchors_including_warmup"] = count
    audit["expected_hours"] = int((end - utc(config["start"])).total_seconds() // 3600)
    audit["expected_raw_seconds"] = audit["expected_hours"] * 3600
    if "valid_hours" in audit:
        audit["observed_hour_buckets"] = audit["valid_hours"] + audit["invalid_hours"]
        audit["entirely_missing_hours"] = audit["expected_hours"] - audit["observed_hour_buckets"]
    summary = {
        "status": "historical_exploratory", "contamination_excluded": False,
        "protocol_sha256": shared.digest_file(output / "protocol.json"),
        "output_sha256": {name: shared.digest_file(output / name) for name in files},
        "data": audit, "horizons": summaries,
        "checkpoint_files_sha256": getattr(model, "checkpoint_files", {}),
        "environment": shared.environment(), "source_verification_seconds": source_verification_seconds,
        "total_seconds": time.perf_counter() - started, "peak_rss_bytes": shared.peak_rss_bytes(),
    }
    shared.exclusive_json(output / "results.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("preregister")
    register.add_argument("--config", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("backtest")
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "preregister":
        result = preregister(json.loads(args.config.read_text()), args.output)
        print(json.dumps({"registered": str(args.output), "config": result["config"]}, indent=2))
    else:
        result = backtest(args.output)
        print(json.dumps({"horizons": {h: {"scored_anchors": x["scored_anchors"], "metrics": x["metrics"]}
                                      for h, x in result["horizons"].items()},
                          "total_seconds": result["total_seconds"], "peak_rss_bytes": result["peak_rss_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
