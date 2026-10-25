#!/usr/bin/env python3
"""Build development forecasts, freeze adaptations, then open a separate holdout."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_experiment as shared  # noqa: E402
from research import timestral_long_horizon as hourly  # noqa: E402
from research import timestral_adaptation as adaptation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("unchanged", "timestral_price", "timestral_covariates", "adapted")


def code_hashes():
    names = ("scripts/run_timestral_adaptation.py", "scripts/analyze_timestral_adaptation.py",
             "research/timestral_adaptation.py", "research/timestral_long_horizon.py",
             "research/timestral_experiment.py", "research/archive.py", "research/market.py",
             "research/db.py", "scripts/prepare_timestral_data.py")
    return {name: shared.digest_file(ROOT / name) for name in names}


def validate_config(config):
    config = dict(config)
    if config["symbol"] != "BTCUSDT" or config["horizons"] != [4, 24] or config["lookback"] != 256:
        raise ValueError("the fixed protocol requires BTCUSDT, horizons [4,24], and 256 hours")
    dates = [shared.utc(config[key]) for key in
             ("development_start", "fit_start", "validation_start", "test_start", "test_end")]
    if dates != sorted(set(dates)) or any(t.hour or t.minute or t.second or t.microsecond for t in dates):
        raise ValueError("split boundaries must be strictly increasing UTC midnights")
    if dates[-1] > shared.utc("2026-08-08"):
        raise ValueError("do not open the earlier protected prospective holdout")
    if dates[1] - dates[0] < timedelta(hours=256):
        raise ValueError("insufficient development context")
    if config["local_files_only"] is not True:
        raise ValueError("reuse the pinned local checkpoint")
    for key in ("development_archive", "test_archive", "model_cache"):
        config[key] = str(Path(config[key]).resolve())
    for key in ("batch_size", "torch_threads"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"invalid {key}")
    return config


def source_config(config, phase):
    return {**config, "source": "archive", "archive_root": config[f"{phase}_archive"],
            "start": config["development_start"] if phase == "development" else config["test_start"],
            "end": config["test_start"] if phase == "development" else config["test_end"]}


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_public_provenance(source, entries):
    """Bind the official ZIP receipts to exactly the verified Parquet entries."""
    root = Path(source["archive_root"]).parent
    provenance = json.loads((root / "provenance.json").read_text())
    expected = {"source": "binance_public_data", "market": "binance_spot", "interval": "1s",
                "symbol": source["symbol"]}
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("unexpected public data source identity")
    if any(shared.utc(provenance[key]) != shared.utc(source[key]) for key in ("start", "end")):
        raise ValueError("provenance range differs from protocol")
    items = provenance["files"]
    def order(entry):
        return entry["symbol"], entry["start"], entry["end"]
    if sorted([item["archive_entry"] for item in items], key=order) != sorted(entries, key=order):
        raise ValueError("official provenance is not bound to verified archive entries")
    days = int((shared.utc(source["end"]) - shared.utc(source["start"])).days)
    expected_days = [shared.utc(source["start"]) + timedelta(days=i) for i in range(days)]
    if sorted(shared.utc(item["day"]) for item in items) != expected_days:
        raise ValueError("source receipts do not cover the complete requested calendar")
    for item in items:
        day = shared.utc(item["day"])
        name = f"{source['symbol']}-1s-{day:%Y-%m-%d}.zip"
        url = f"https://data.binance.vision/data/spot/daily/klines/{source['symbol']}/1s/{name}"
        if item["filename"] != name or item["url"] != url or item["checksum_url"] != url + ".CHECKSUM":
            raise ValueError("source receipt is not the expected official daily ZIP")
        if shared.digest_file(root / "downloads" / name) != item["sha256"]:
            raise ValueError("retained official ZIP changed")
    return provenance


def base_forecast(context, anchor, horizon, config, model):
    if len(context) != config["lookback"] or context[-1].time + timedelta(hours=1) != anchor:
        raise ValueError("context must end immediately before anchor")
    if any(b.time - a.time != timedelta(hours=1) for a, b in zip(context, context[1:])):
        raise ValueError("context contains a gap")
    features = adaptation.extract_features(context, anchor)
    local = {**config, "horizon": horizon}
    predictions = {name: model.predict(context, local, covariates)
                   for name, covariates in (("timestral_price", False), ("timestral_covariates", True))}
    for value in predictions.values():
        if not np.isfinite([value["point"], *value["quantiles"]]).all():
            raise ValueError("nonfinite base forecast")
    context_hash = hashlib.sha256(shared.canonical(
        [[shared.stamp(x.time), x.log_close, x.features] for x in context])).hexdigest()
    sample_id = hashlib.sha256(shared.canonical([config["symbol"], shared.stamp(anchor), horizon, context_hash])).hexdigest()
    return {"sample_id": sample_id, "anchor": shared.stamp(anchor),
            "label_end": shared.stamp(anchor + timedelta(hours=horizon)), "horizon_hours": horizon,
            "input_start": shared.stamp(context[0].time), "input_end_exclusive": shared.stamp(anchor),
            "input_sha256": context_hash, "reference_log_close": context[-1].log_close,
            "reference_price": float(np.exp(context[-1].log_close)), "features": features,
            "baseprice": predictions["timestral_price"]["point"],
            "basecov": predictions["timestral_covariates"]["point"], "predictions": predictions,
            "computed_at": shared.stamp(datetime.now(timezone.utc)), "historical_exploratory": True}


def issue(context, anchor, end, config, model, pending, handle, adapt=None):
    for horizon in config["horizons"]:
        if int(anchor.timestamp()) % (horizon * 3600) or anchor + timedelta(hours=horizon) > end:
            continue
        if horizon in pending:
            raise ValueError("nonoverlapping forecast still pending")
        record = base_forecast(list(context), anchor, horizon, config, model)
        if adapt is not None:
            record = adapt(record)
        shared.append_record(handle, record)
        pending[horizon] = record


def mature(hour, pending, outcomes, scored, development=False):
    now = hour.time + timedelta(hours=1)
    for horizon, record in list(pending.items()):
        if shared.utc(record["label_end"]) != now:
            continue
        actual = hour.log_close - record["reference_log_close"]
        outcome = {key: record[key] for key in ("sample_id", "anchor", "label_end", "horizon_hours")}
        outcome.update(status="scored", actual_log_return=actual, actual_price=float(np.exp(hour.log_close)))
        shared.append_record(outcomes, outcome)
        combined = {**record, **outcome}
        if development:
            combined["actual"] = actual
        shared.append_record(scored, combined)
        del pending[horizon]


def invalidate(pending, outcomes, audit, reason):
    for record in pending.values():
        shared.append_record(outcomes, {**{key: record[key] for key in
            ("sample_id", "anchor", "label_end", "horizon_hours")}, "status": "invalid_target", "reason": reason})
        audit["invalid_targets"] += 1
    pending.clear()


def finish_results(output, protocol, audit, files, started, model, source):
    if shared.source_snapshot(source) != protocol["source"]:
        raise ValueError("source changed during the run")
    rows = read_jsonl(output / files[-1])
    counts = {str(h): sum(r["horizon_hours"] == h for r in rows) for h in (4, 24)}
    predictions = read_jsonl(output / "predictions.jsonl")
    evaluation_start = protocol.get("test_start", protocol["config"]["fit_start"])
    expected_hours = int((shared.utc(source["end"]) - shared.utc(source["start"])).total_seconds() // 3600)
    audit.update(expected_hours=expected_hours, expected_raw_seconds=expected_hours * 3600,
                 entirely_missing_hours=expected_hours - audit.get("valid_hours", 0) - audit.get("invalid_hours", 0))
    coverage = {}
    for horizon in (4, 24):
        expected = int((shared.utc(source["end"]) - shared.utc(evaluation_start)).total_seconds() // (horizon * 3600))
        emitted = sum(r["horizon_hours"] == horizon for r in predictions)
        coverage[str(horizon)] = {"expected": expected, "emitted": emitted, "scored": counts[str(horizon)],
                                  "missing_context": expected - emitted, "invalid_target": emitted - counts[str(horizon)]}
    for name in files:
        (output / name).chmod(0o444)
    result = {"protocol_sha256": shared.digest_file(output / "protocol.json"),
              "output_sha256": {name: shared.digest_file(output / name) for name in files},
              "counts": counts, "coverage": coverage, "data": audit, "total_seconds": time.perf_counter() - started,
              "peak_rss_bytes": shared.peak_rss_bytes(), "environment": shared.environment(),
              "checkpoint_files_sha256": getattr(model, "checkpoint_files", {}),
              "prospective": False}
    shared.exclusive_json(output / "results.json", result)
    return result


def development(config, output, model_factory=shared.Timestral):
    config = validate_config(config)
    source = source_config(config, "development")
    started = time.perf_counter()
    snapshot, entries = shared.verified_source_snapshot(source)
    verify_public_provenance(source, entries)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"config": config, "source": snapshot, "source_verified_entries": entries,
                "code_sha256": code_hashes(), "created_at": shared.stamp(datetime.now(timezone.utc)),
                "purpose": "development only; no held-out data are available to selection"}
    shared.exclusive_json(output / "protocol.json", protocol)
    files = ["hours.jsonl", "predictions.jsonl", "outcomes.jsonl", "examples.jsonl"]
    audit = {"hour_gap_resets": 0, "invalid_targets": 0}
    with ExitStack() as stack:
        handles = {name: stack.enter_context((output / name).open("x")) for name in files}
        model = model_factory(config)
        hours = hourly.read_hours(source, audit)
        stack.callback(hours.close)
        context, pending, previous = deque(maxlen=config["lookback"]), {}, None
        for hour in hours:
            if previous is not None and hour.time != previous + timedelta(hours=1):
                invalidate(pending, handles["outcomes.jsonl"], audit, "missing source hour")
                context.clear()
                audit["hour_gap_resets"] += 1
            if not shared.utc(source["start"]) <= hour.time < shared.utc(source["end"]):
                raise ValueError("development hour outside fixed range")
            previous = hour.time
            shared.append_record(handles["hours.jsonl"], {"time": shared.stamp(hour.time),
                                 "log_close": hour.log_close, "features": list(hour.features)})
            context.append(hour)
            mature(hour, pending, handles["outcomes.jsonl"], handles["examples.jsonl"], development=True)
            now = hour.time + timedelta(hours=1)
            if len(context) == config["lookback"] and now >= shared.utc(config["fit_start"]):
                issue(context, now, shared.utc(config["test_start"]), config, model, pending, handles["predictions.jsonl"])
        invalidate(pending, handles["outcomes.jsonl"], audit, "source ended before target")
    return finish_results(output, protocol, audit, files, started, model, source)


def verify_development(path):
    path = Path(path)
    results = json.loads((path / "results.json").read_text())
    if shared.digest_file(path / "protocol.json") != results["protocol_sha256"]:
        raise ValueError("development protocol hash mismatch")
    for name, expected in results["output_sha256"].items():
        if Path(name).name != name or shared.digest_file(path / name) != expected:
            raise ValueError("development output changed")
    return json.loads((path / "protocol.json").read_text()), results


def freeze(development_run, output):
    development_run, output = Path(development_run).resolve(), Path(output)
    dev_protocol, dev_results = verify_development(development_run)
    config = validate_config(dev_protocol["config"])
    if dev_protocol["code_sha256"] != code_hashes():
        raise ValueError("development code changed; do not silently reinterpret its outputs")
    if Path(config["test_archive"]).parent.exists():
        raise ValueError("freeze the selection before acquiring the held-out archive")
    bundle = adaptation.select_and_freeze(
        read_jsonl(development_run / "examples.jsonl"), output, config,
        metadata={"development_run": str(development_run), "development_results_sha256":
                  shared.digest_file(development_run / "results.json"), "code_sha256": code_hashes()})
    protocol = {
        "registered_at": shared.stamp(datetime.now(timezone.utc)), "config": config,
        "test_start": config["test_start"], "test_end": config["test_end"],
        "selected_candidates": {key: item["candidate"] for key, item in bundle["selected"].items()},
        "freeze_sha256": shared.digest_file(output / "bundle.json"), "code_sha256": code_hashes(),
        "development_run": str(development_run),
        "development_results_sha256": shared.digest_file(development_run / "results.json"),
        "development_output_sha256": dev_results["output_sha256"],
        "candidate_count": 9, "selection": "lowest December MAE per horizon; fixed listed-order tie break",
        "refit": "selected candidate fitted once to October-December; no test-label updates",
        "test_counts_if_complete": {"4": 540, "24": 90},
        "primary": "paired adapted-minus-unchanged absolute log-return error",
        "uncertainty": {"calendar_block_days": 7, "bootstrap_draws": 10000, "seed": 17,
                        "primary_interval_quantiles": [0.0125, 0.9875],
                        "interpretation": "approximate descriptive 97.5% intervals; Bonferroni for two horizons"},
        "evidence_gate": {"minimum_counts": {"4": 480, "24": 80}, "lower_mae": True,
                          "upper_interval_below_zero": True, "monthly_mae_wins_at_least": 2,
                          "rmse_not_worse": True},
        "interval_calibration": "unchanged uses October-December returns; adapted uses selected December validation residuals; frozen, selection-biased and not guaranteed calibrated",
        "historical_limit": "independent samples for this selection, but design was motivated by later May-July 2026 scores; not a prospective January decision; pretraining contamination not excluded",
        "protected_period": "2026-08-08 through 2027-02-04 is not accessed",
    }
    shared.exclusive_json(output / "protocol.json", protocol)
    shared.exclusive_json(output / "protocol.sha256.json", {"sha256": shared.digest_file(output / "protocol.json")})
    for path in output.iterdir():
        if path.is_file():
            path.chmod(0o444)
    return protocol


def load_freeze(output):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    if shared.digest_file(output / "protocol.json") != json.loads((output / "protocol.sha256.json").read_text())["sha256"]:
        raise ValueError("frozen protocol changed")
    if shared.digest_file(output / "bundle.json") != protocol["freeze_sha256"] or code_hashes() != protocol["code_sha256"]:
        raise ValueError("frozen parameters or code changed")
    return protocol, adaptation.load_bundle(output)


def add_adaptation(record, bundle):
    horizon = record["horizon_hours"]
    selected = bundle["selected"][str(horizon)]
    point = adaptation.predict(bundle, record["anchor"], horizon, record["features"],
                               record["baseprice"], record["basecov"])
    record["predictions"]["unchanged"] = {"point": 0.0, "quantiles": selected["unchanged_return_quantiles"]}
    record["predictions"]["adapted"] = {
        "point": point,
        "quantiles": (point + np.asarray(selected["validation_residual_quantiles"])).tolist(),
        "candidate": selected["candidate"],
    }
    if set(record["predictions"]) != set(MODEL_NAMES):
        raise ValueError("unequal model set")
    for pred in record["predictions"].values():
        if not np.isfinite([pred["point"], *pred["quantiles"]]).all():
            raise ValueError("nonfinite adapted forecast")
    return record


def holdout(frozen, output, model_factory=shared.Timestral):
    frozen, output = Path(frozen), Path(output)
    protocol, bundle = load_freeze(frozen)
    config = validate_config(protocol["config"])
    started = time.perf_counter()
    development_run = Path(protocol["development_run"])
    if shared.digest_file(development_run / "results.json") != protocol["development_results_sha256"]:
        raise ValueError("development evidence changed after freeze")
    _, dev_results = verify_development(development_run)
    if dev_results["output_sha256"] != protocol["development_output_sha256"]:
        raise ValueError("development inputs changed after freeze")
    tail = read_jsonl(development_run / "hours.jsonl")[-config["lookback"]:]
    context = deque([hourly.Hour(shared.utc(x["time"]), float(x["log_close"]), tuple(x["features"]))
                     for x in tail], maxlen=config["lookback"])
    start, end = shared.utc(config["test_start"]), shared.utc(config["test_end"])
    if len(context) != config["lookback"] or context[-1].time + timedelta(hours=1) != start:
        raise ValueError("development cache does not provide the exact test context")
    if any(b.time - a.time != timedelta(hours=1) for a, b in zip(context, list(context)[1:])):
        raise ValueError("test warmup context has a gap")
    source = source_config(config, "test")
    provenance = json.loads((Path(config["test_archive"]).parent / "provenance.json").read_text())
    if shared.utc(provenance["retrieved_at"]) < shared.utc(protocol["registered_at"]):
        raise ValueError("holdout was acquired before selection freeze")
    if shared.utc(provenance["start"]) != start or shared.utc(provenance["end"]) != end:
        raise ValueError("holdout provenance range differs from protocol")
    output.mkdir(parents=True, exist_ok=False)
    shared.exclusive_json(frozen / "test-opened.json", {
        "opened_at": shared.stamp(datetime.now(timezone.utc)), "output": str(output.resolve()),
        "frozen_protocol_sha256": shared.digest_file(frozen / "protocol.json")})
    snapshot, entries = shared.verified_source_snapshot(source)
    verify_public_provenance(source, entries)
    runtime_protocol = {**protocol, "frozen_protocol_sha256": shared.digest_file(frozen / "protocol.json"),
                        "source": snapshot, "source_verified_entries": entries,
                        "source_provenance_sha256": shared.digest_file(Path(config["test_archive"]).parent / "provenance.json")}
    shared.exclusive_json(output / "protocol.json", runtime_protocol)
    files = ["predictions.jsonl", "outcomes.jsonl", "evaluation.jsonl"]
    audit = {"hour_gap_resets": 0, "invalid_targets": 0}
    with ExitStack() as stack:
        handles = {name: stack.enter_context((output / name).open("x")) for name in files}
        model = model_factory(config)
        hours = hourly.read_hours(source, audit)
        stack.callback(hours.close)
        pending, previous = {}, context[-1].time
        def adapted(record):
            return add_adaptation(record, bundle)
        issue(context, start, end, config, model, pending, handles["predictions.jsonl"], adapted)
        for hour in hours:
            if not start <= hour.time < end:
                raise ValueError("test hour outside fixed range")
            if hour.time <= previous:
                raise ValueError("test hours not strictly increasing")
            if hour.time != previous + timedelta(hours=1):
                invalidate(pending, handles["outcomes.jsonl"], audit, "missing source hour")
                context.clear()
                audit["hour_gap_resets"] += 1
            previous = hour.time
            context.append(hour)
            mature(hour, pending, handles["outcomes.jsonl"], handles["evaluation.jsonl"])
            if len(context) == config["lookback"]:
                issue(context, hour.time + timedelta(hours=1), end, config, model, pending,
                      handles["predictions.jsonl"], adapted)
        invalidate(pending, handles["outcomes.jsonl"], audit, "source ended before target")
    return finish_results(output, runtime_protocol, audit, files, started, model, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dev = sub.add_parser("development")
    dev.add_argument("--config", type=Path, required=True)
    dev.add_argument("--output", type=Path, required=True)
    lock = sub.add_parser("freeze")
    lock.add_argument("--development-run", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    test = sub.add_parser("holdout")
    test.add_argument("--frozen", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "development":
        result = development(json.loads(args.config.read_text()), args.output)
    elif args.command == "freeze":
        result = freeze(args.development_run, args.output)
        print(json.dumps({"selected_candidates": result["selected_candidates"],
                          "registered_at": result["registered_at"], "freeze_sha256": result["freeze_sha256"]}, indent=2))
        return
    else:
        result = holdout(args.frozen, args.output)
    print(json.dumps({key: result[key] for key in ("counts", "total_seconds", "peak_rss_bytes")}, indent=2))


if __name__ == "__main__":
    main()
