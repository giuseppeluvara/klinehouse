"""Small, preregistered zero-shot Timestral experiment on real 1-second candles.

No orders, PnL, fine-tuning, or dependencies on the frozen research experiments.
The data reader retains one Arrow batch and at most lookback + horizon minutes.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np


MODEL_ID = "mistralai/Timestral-5M"
CODE_REVISION = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
WEIGHT_REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"
CHECKPOINT_SHA256 = {
    "model.safetensors": "7ace108c4987e7f2a99a75a445240019d1d31c2117139d1bd05ffdfd25aca738",
    "config.json": "b5f27f61513f8fd1a3aa996f3720d3679cb40bce5bcb08eb3dc320ce726e7c2f",
}
LEVELS = (0.05, 0.5, 0.95)
MODELS = ("unchanged", "local_drift", "timestral_price", "timestral_covariates")
FEATURES = ("intraminute_realized_volatility", "log1p_trades", "taker_imbalance")
SOURCE_COLUMNS = ("open_time", "close_time", "close", "volume", "trades", "taker_buy_base")


def utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def stamp(value):
    return utc(value).isoformat().replace("+00:00", "Z")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path, value):
    """Create once; never replace a protocol, forecast, or result."""
    with Path(path).open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    Path(path).chmod(0o444)


def environment():
    versions = {}
    for name in ("numpy", "torch", "timestral", "duckdb", "pyarrow", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    direct_url = None
    try:
        direct_url = importlib.metadata.distribution("timestral").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        pass
    return {"python": sys.version, "platform": platform.platform(), "versions": versions,
            "timestral_direct_url": json.loads(direct_url) if direct_url else None}


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def verify_model_installation():
    """Do not silently label an arbitrary installed package as the pinned model."""
    try:
        direct_url = importlib.metadata.distribution("timestral").read_text("direct_url.json")
        actual = json.loads(direct_url or "{}").get("vcs_info", {}).get("commit_id")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("install the pinned Timestral revision from requirements-timestral.txt") from exc
    if actual != CODE_REVISION:
        raise ValueError(f"Timestral installation must resolve to commit {CODE_REVISION}")


def validate_config(config):
    required = {"symbol", "start", "end", "evaluation_start", "archive_root", "source",
                "lookback", "horizon", "stride", "min_calibration", "max_minutes",
                "max_samples", "batch_size", "torch_threads", "max_emission_delay_seconds"}
    if not required <= config.keys():
        raise ValueError(f"missing config fields: {sorted(required - config.keys())}")
    if config["symbol"] != "BTCUSDT" or config["horizon"] != 60 or config["stride"] != 60:
        raise ValueError("this first protocol requires BTCUSDT and horizon=stride=60 minutes")
    if config["source"] not in {"archive", "database", "auto"}:
        raise ValueError("invalid source")
    for key in ("lookback", "min_calibration", "max_minutes", "max_samples", "batch_size", "torch_threads"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"invalid {key}")
    if config["lookback"] < 32 or not 0 < config["max_emission_delay_seconds"] <= 60:
        raise ValueError("lookback must be >=32 and emission delay within (0,60] seconds")
    start, evaluation, end = (utc(config[key]) for key in ("start", "evaluation_start", "end"))
    if not start < evaluation < end:
        raise ValueError("require start < evaluation_start < end")
    if any(value.second or value.microsecond for value in (start, evaluation, end)):
        raise ValueError("period boundaries must be whole UTC minutes")
    if (end - start).total_seconds() / 60 > config["max_minutes"]:
        raise ValueError("period exceeds the preregistered minute budget")
    return config


def source_snapshot(config):
    root = Path(config["archive_root"])
    catalog = root / "catalog.json"
    if config["source"] == "archive" and not catalog.is_file():
        raise ValueError(f"archive catalog does not exist: {catalog}")
    return {"archive_root": str(root.resolve()),
            "catalog_sha256": digest_file(catalog) if catalog.is_file() else None,
            "source": config["source"]}


def verified_source_snapshot(config):
    from .archive import archive_lock, read_catalog, verify_entry
    root = Path(config["archive_root"])
    with archive_lock(root):
        snapshot = source_snapshot(config)
        entries = [entry for entry in read_catalog(root)
                   if entry["symbol"] == config["symbol"] and utc(entry["start"]) < utc(config["end"])
                   and utc(entry["end"]) > utc(config["start"])] if config["source"] != "database" else []
        for entry in entries:
            verify_entry(root, entry)
        return snapshot, entries


def reader_code_hashes():
    root = Path(__file__).parent
    return {name: digest_file(root / name) for name in ("archive.py", "market.py", "db.py")}


def preregister(config, output):
    config = validate_config(dict(config))
    config["archive_root"] = str(Path(config["archive_root"]).resolve())
    if config.get("model_cache"):
        config["model_cache"] = str(Path(config["model_cache"]).resolve())
    snapshot, entries = verified_source_snapshot(config)
    protocol = {
        "schema": 1, "registered_at": stamp(datetime.now(timezone.utc)), "config": config,
        "model": {"id": MODEL_ID, "code_revision": CODE_REVISION, "weight_revision": WEIGHT_REVISION,
                  "device": "cpu", "precision": "float32", "zero_shot": True,
                  "symmetric_averaging": True, "clip_nonnegative": False},
        "source": snapshot, "source_verified_entries": entries, "reader_code_sha256": reader_code_hashes(),
        "environment": environment(),
        "experiment_code_sha256": digest_file(__file__),
        "target": "log(close(anchor+59m)/close(anchor-1m)); available at anchor+60m",
        "context": "complete 1-second OHLCV candles aggregated in [anchor-lookback,anchor)",
        "features": list(FEATURES), "models": list(MODELS), "quantiles": list(LEVELS),
        "baseline": "unchanged=0; local_drift=horizon*mean(diff(context log-close)); expanding empirical prequential residual quantiles",
        "split": "hourly non-overlapping labels; expanding calibration with label_end<=anchor; daily UTC evaluation folds",
        "metrics": ["return_mae", "return_rmse", "direction_accuracy", "pinball", "coverage90", "interval_width90", "quantile_crossing"],
        "historical_status": "exploratory: pretraining contamination cannot be excluded",
        "execution": "prediction time includes inference; no execution-price or PnL claims; earliest execution strictly after issue time",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    exclusive_json(output / "protocol.json", protocol)
    protocol_hash = digest_file(output / "protocol.json")
    exclusive_json(output / "protocol.sha256.json", {"sha256": protocol_hash})
    (output / "protocol.json").chmod(0o444)
    (output / "protocol.sha256.json").chmod(0o444)
    return protocol


def load_protocol(output):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    if digest_file(output / "protocol.json") != json.loads((output / "protocol.sha256.json").read_text())["sha256"]:
        raise ValueError("protocol changed after preregistration")
    if digest_file(__file__) != protocol["experiment_code_sha256"]:
        raise ValueError("experiment code changed after preregistration; create a distinct run")
    if reader_code_hashes() != protocol["reader_code_sha256"]:
        raise ValueError("market reader code changed after preregistration; create a distinct run")
    validate_config(protocol["config"])
    return protocol


@dataclass(frozen=True)
class Minute:
    time: datetime
    log_close: float
    features: tuple[float, float, float]


def minute_features(batches, audit):
    """Consume bounded Arrow batches, dropping incomplete/invalid minutes.

    Realized volatility uses the 59 within-minute close-to-close log returns.
    It intentionally excludes the previous minute's last second.
    """
    minute, rows, previous = None, [], None
    source_hash, feature_hash = hashlib.sha256(), hashlib.sha256()
    audit.update(raw_rows=0, valid_minutes=0, invalid_minutes=0)

    def finish():
        if not rows:
            return None
        start = rows[0][0].replace(second=0, microsecond=0)
        valid = len(rows) == 60 and all(row[0] == start + timedelta(seconds=i) for i, row in enumerate(rows))
        valid = valid and all(row[0] <= row[1] < row[0] + timedelta(seconds=1) for row in rows)
        values = np.array([row[2:] for row in rows], dtype=np.float64)
        valid = valid and np.isfinite(values).all() and (values[:, 0] > 0).all()
        valid = valid and (values[:, 1:] >= 0).all() and (values[:, 3] <= values[:, 1] + 1e-10).all()
        if not valid:
            audit["invalid_minutes"] += 1
            return None
        logs = np.log(values[:, 0])
        volume, trades, taker = values[:, 1:].sum(axis=0)
        features = (float(np.sqrt(np.square(np.diff(logs)).sum())), float(np.log1p(trades)),
                    float((2 * taker - volume) / volume) if volume else 0.0)
        result = Minute(start, float(logs[-1]), features)
        feature_hash.update(canonical([stamp(result.time), result.log_close, result.features]) + b"\n")
        audit["valid_minutes"] += 1
        return result

    for batch in batches:
        columns = batch.to_pydict()
        for raw in zip(*(columns[key] for key in SOURCE_COLUMNS)):
            current = utc(raw[0])
            if previous is not None and current <= previous:
                raise ValueError("source timestamps must be strictly increasing")
            previous = current
            row = (current, utc(raw[1]), *raw[2:])
            # Canonical JSON rejects NaN/Inf before any model sees the source.
            source_hash.update(canonical([stamp(row[0]), stamp(row[1]), *row[2:]]) + b"\n")
            audit["raw_rows"] += 1
            key = current.replace(second=0, microsecond=0)
            if minute is not None and key != minute:
                item = finish()
                if item:
                    yield item
                rows = []
            minute = key
            rows.append(row)
    item = finish()
    if item:
        yield item
    audit.update(source_rows_sha256=source_hash.hexdigest(), minute_features_sha256=feature_hash.hexdigest())


def read_minutes(config, audit, end=None):
    from .market import iter_market
    iterator = iter_market(config["symbol"], config["start"], end or config["end"],
                           archive_root=config["archive_root"], source=config["source"],
                           timeframe="1s", columns=list(SOURCE_COLUMNS), batch_size=config["batch_size"])
    try:
        yield from minute_features(iterator, audit)
    finally:
        iterator.close()


def sample_windows(minutes, config, audit):
    size = config["lookback"] + config["horizon"]
    buffer = deque(maxlen=size)
    count, samples, previous = 0, 0, None
    for minute in minutes:
        count += 1
        if count > config["max_minutes"]:
            raise ValueError("minute budget exceeded")
        if previous is not None and minute.time != previous + timedelta(minutes=1):
            buffer.clear()
            audit["minute_gap_resets"] = audit.get("minute_gap_resets", 0) + 1
        previous = minute.time
        buffer.append(minute)
        if len(buffer) != size:
            continue
        anchor = buffer[config["lookback"]].time
        if int(anchor.timestamp()) % (60 * config["stride"]):
            continue
        samples += 1
        if samples > config["max_samples"]:
            raise ValueError("sample budget exceeded")
        window = list(buffer)
        yield anchor, window[:config["lookback"]], window[config["lookback"]:]


def drift(context, horizon):
    return float(horizon * np.diff([item.log_close for item in context]).mean())


class Calibration:
    """Only already matured prequential baseline errors enter calibration."""

    def __init__(self):
        self.records = []

    def add(self, label_end, outcome, drift_prediction):
        self.records.append((label_end, float(outcome), float(outcome - drift_prediction)))

    def forecast(self, anchor, context, config):
        eligible = [item for item in self.records if item[0] <= anchor]
        if len(eligible) < config["min_calibration"]:
            raise ValueError(f"insufficient past calibration: {len(eligible)}")
        trend = drift(context, config["horizon"])
        return {
            "unchanged": {"point": 0.0, "quantiles": np.quantile([x[1] for x in eligible], LEVELS).tolist()},
            "local_drift": {"point": trend, "quantiles": (trend + np.quantile([x[2] for x in eligible], LEVELS)).tolist()},
        }, {"count": len(eligible), "latest_label_end": stamp(max(x[0] for x in eligible))}


class Timestral:
    def __init__(self, config):
        verify_model_installation()
        import torch
        from timestral import TimestralForecaster
        torch.set_num_threads(config["torch_threads"])
        torch.manual_seed(7)
        np.random.seed(7)
        from huggingface_hub import snapshot_download
        checkpoint = Path(snapshot_download(MODEL_ID, revision=WEIGHT_REVISION,
                                          cache_dir=config.get("model_cache"),
                                          local_files_only=config.get("local_files_only", True)))
        self.checkpoint_files = {str(path.relative_to(checkpoint)): digest_file(path)
                                 for path in sorted(checkpoint.rglob("*")) if path.is_file()}
        for name, expected in CHECKPOINT_SHA256.items():
            if self.checkpoint_files.get(name) != expected:
                raise ValueError(f"pinned checkpoint checksum mismatch: {name}")
        self.forecaster = TimestralForecaster.from_pretrained(
            MODEL_ID, device="cpu", max_context=config["lookback"], revision=WEIGHT_REVISION,
            cache_dir=config.get("model_cache"), local_files_only=True)
        if not self.forecaster.supports_covariates:
            raise ValueError("pinned Timestral checkpoint does not support past covariates")

    def predict(self, context, config, covariates):
        series = np.asarray([item.log_close for item in context], dtype=np.float32)[None, None, :]
        kwargs = {"past_covariates": np.asarray([item.features for item in context], dtype=np.float32).T[None, :, :]} if covariates else {}
        result = self.forecaster.forecast(series, prediction_length=config["horizon"],
                                           symmetric_averaging=True, clip_nonnegative=False, **kwargs)
        levels = np.asarray(result.quantile_levels)
        values = result.quantiles[0, 0, -1, :].detach().cpu().numpy().astype(float)
        indices = [int(np.flatnonzero(np.isclose(levels, level))[0]) for level in LEVELS]
        quantiles = values[indices] - context[-1].log_close
        if not np.isfinite(values).all():
            raise ValueError("non-finite model output")
        return {"point": float(quantiles[1]), "quantiles": quantiles.tolist(),
                "all_quantile_levels": levels.tolist(),
                "all_return_quantiles": (values - context[-1].log_close).tolist()}


def predict_all(anchor, context, calibration, config, model):
    predictions, evidence = calibration.forecast(anchor, context, config)
    elapsed = {}
    for name, covariates in (("timestral_price", False), ("timestral_covariates", True)):
        started = time.perf_counter()
        predictions[name] = model.predict(context, config, covariates)
        elapsed[name] = time.perf_counter() - started
    context_hash = hashlib.sha256(canonical([[stamp(x.time), x.log_close, x.features] for x in context])).hexdigest()
    sample_id = hashlib.sha256(canonical([config["symbol"], stamp(anchor), config["horizon"], context_hash])).hexdigest()
    return {"sample_id": sample_id, "anchor": stamp(anchor),
            "label_end": stamp(anchor + timedelta(minutes=config["horizon"])),
            "input_start": stamp(context[0].time), "last_input_close": stamp(anchor),
            "input_sha256": context_hash, "reference_log_close": context[-1].log_close,
            "calibration": evidence, "predictions": predictions, "inference_seconds": elapsed,
            "fold": anchor.strftime("%Y-%m-%d")}


def metric_summary(records):
    if not records:
        raise ValueError("no common complete evaluation samples")
    result = {}
    actual = np.array([item[1] for item in records])
    for name in MODELS:
        point = np.array([item[0]["predictions"][name]["point"] for item in records])
        quantiles = np.array([item[0]["predictions"][name]["quantiles"] for item in records])
        all_quantiles = np.array([item[0]["predictions"][name].get("all_return_quantiles", item[0]["predictions"][name]["quantiles"]) for item in records])
        error, residual = point - actual, actual[:, None] - quantiles
        pinball = np.maximum(np.array(LEVELS) * residual, (np.array(LEVELS) - 1) * residual)
        result[name] = {"n": len(actual), "return_mae": float(np.abs(error).mean()),
                        "return_rmse": float(np.sqrt(np.square(error).mean())),
                        "direction_accuracy": float((np.sign(point) == np.sign(actual)).mean()),
                        "pinball_mean": float(pinball.mean()),
                        "pinball_by_quantile": dict(zip(map(str, LEVELS), pinball.mean(axis=0).tolist())),
                        "coverage90": float(((quantiles[:, 0] <= actual) & (actual <= quantiles[:, 2])).mean()),
                        "interval_width90": float((quantiles[:, 2] - quantiles[:, 0]).mean()),
                        "quantile_crossing": float((np.diff(all_quantiles, axis=1) < 0).any(axis=1).mean())}
    return result


def append_record(handle, record):
    handle.write(canonical(record).decode() + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def backtest(output, model_factory=Timestral):
    output = Path(output)
    protocol = load_protocol(output)
    config = protocol["config"]
    if verified_source_snapshot(config)[0] != protocol["source"]:
        raise ValueError("source catalog changed after preregistration")
    started = time.perf_counter()
    # Reserve output before loading a model: reruns cannot overwrite previous forecasts.
    with (output / "predictions.jsonl").open("x") as predictions_file, (output / "outcomes.jsonl").open("x") as outcomes_file:
        model = model_factory(config)
        calibration, audit, records = Calibration(), {}, []
        minutes = read_minutes(config, audit)
        try:
            for anchor, context, target in sample_windows(minutes, config, audit):
                if anchor >= utc(config["evaluation_start"]):
                    record = predict_all(anchor, context, calibration, config, model)
                    record.update(mode="historical_exploratory", issued_at=stamp(datetime.now(timezone.utc)))
                    # Publish prediction first; the forecast function never receives target rows.
                    append_record(predictions_file, record)
                    outcome = target[-1].log_close - context[-1].log_close
                    append_record(outcomes_file, {"sample_id": record["sample_id"], "return": outcome})
                    records.append((record, outcome))
                else:
                    outcome = target[-1].log_close - context[-1].log_close
                calibration.add(anchor + timedelta(minutes=config["horizon"]), outcome, drift(context, config["horizon"]))
        finally:
            minutes.close()
    if source_snapshot(config) != protocol["source"]:
        raise ValueError("source catalog changed during evaluation")
    (output / "predictions.jsonl").chmod(0o444)
    (output / "outcomes.jsonl").chmod(0o444)
    folds = sorted({item[0]["fold"] for item in records})
    summary = {"status": "historical_exploratory", "contamination_excluded": False,
               "protocol_sha256": digest_file(output / "protocol.json"),
               "predictions_sha256": digest_file(output / "predictions.jsonl"),
               "outcomes_sha256": digest_file(output / "outcomes.jsonl"),
               "sample_ids": [item[0]["sample_id"] for item in records], "data": audit,
               "metrics": metric_summary(records),
               "folds": {fold: metric_summary([item for item in records if item[0]["fold"] == fold]) for fold in folds},
               "checkpoint_files_sha256": getattr(model, "checkpoint_files", {}),
               "environment": environment(), "total_seconds": time.perf_counter() - started,
               "peak_rss_bytes": peak_rss_bytes(),
               "inference_seconds": {name: {"sum": float(sum(item[0]["inference_seconds"][name] for item in records)),
                                            "median": float(np.median([item[0]["inference_seconds"][name] for item in records]))}
                                     for name in ("timestral_price", "timestral_covariates")}}
    exclusive_json(output / "results.json", summary)
    return summary


def prospect(output, anchor, model_factory=Timestral, now=lambda: datetime.now(timezone.utc)):
    output, anchor = Path(output), utc(anchor)
    protocol = load_protocol(output)
    config = protocol["config"]
    if anchor != utc(config["end"]) or int(anchor.timestamp()) % 3600:
        raise ValueError("prospective anchor must equal protocol end and be an exact UTC hour")
    if not anchor <= now() < anchor + timedelta(seconds=config["max_emission_delay_seconds"]):
        raise ValueError("anchor is not within the preregistered emission window")
    started = time.perf_counter()
    model, calibration, audit = model_factory(config), Calibration(), {}
    tail = deque(maxlen=config["lookback"])

    def observed():
        previous = None
        stream = read_minutes(config, audit, end=anchor)
        try:
            for minute in stream:
                if previous is not None and minute.time != previous + timedelta(minutes=1):
                    tail.clear()
                previous = minute.time
                tail.append(minute)
                yield minute
        finally:
            stream.close()

    stream = observed()
    try:
        for past_anchor, context, target in sample_windows(stream, config, audit):
            calibration.add(past_anchor + timedelta(minutes=config["horizon"]),
                            target[-1].log_close - context[-1].log_close, drift(context, config["horizon"]))
    finally:
        stream.close()
    if len(tail) != config["lookback"] or tail[-1].time != anchor - timedelta(minutes=1):
        raise ValueError("latest prospective context is missing or incomplete")
    record = predict_all(anchor, list(tail), calibration, config, model)
    issued = now()
    if issued >= anchor + timedelta(seconds=config["max_emission_delay_seconds"]):
        raise ValueError("inference missed the preregistered emission deadline; no eligible forecast published")
    record.update(mode="prospective", issued_at=stamp(issued), earliest_execution_ts=stamp(issued + timedelta(microseconds=1)),
                  total_seconds=time.perf_counter() - started, peak_rss_bytes=peak_rss_bytes(), data=audit,
                  protocol_sha256=digest_file(output / "protocol.json"), environment=environment(),
                  checkpoint_files_sha256=getattr(model, "checkpoint_files", {}))
    exclusive_json(output / "prospective-prediction.json", record)
    return record


def score_prospective(output, now=lambda: datetime.now(timezone.utc)):
    output = Path(output)
    protocol = load_protocol(output)
    prediction = json.loads((output / "prospective-prediction.json").read_text())
    config, anchor, end = dict(protocol["config"]), utc(prediction["anchor"]), utc(prediction["label_end"])
    if now() < end:
        raise ValueError("prospective outcome is not mature")
    if prediction["protocol_sha256"] != digest_file(output / "protocol.json") or not anchor <= utc(prediction["issued_at"]) < anchor + timedelta(seconds=config["max_emission_delay_seconds"]):
        raise ValueError("prediction is not eligible under this protocol")
    config.update(start=stamp(anchor), end=stamp(end))
    audit = {}
    target = list(read_minutes(config, audit))
    if len(target) != config["horizon"] or any(item.time != anchor + timedelta(minutes=i) for i, item in enumerate(target)):
        raise ValueError("prospective target has missing minutes")
    outcome = target[-1].log_close - prediction["reference_log_close"]
    result = {"sample_id": prediction["sample_id"], "return": outcome, "scored_at": stamp(now()),
              "prediction_sha256": digest_file(output / "prospective-prediction.json"), "data": audit,
              "metrics": metric_summary([(prediction, outcome)])}
    exclusive_json(output / "prospective-outcome.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("preregister")
    register.add_argument("--config", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)
    for name in ("backtest", "prospect", "score-prospective"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "prospect":
            command.add_argument("--anchor", required=True)
    args = parser.parse_args(argv)
    if args.command == "preregister":
        result = preregister(json.loads(args.config.read_text()), args.output)
        print(json.dumps({"registered": str(args.output), "config": result["config"]}, indent=2))
    elif args.command == "backtest":
        result = backtest(args.output)
        print(json.dumps({"metrics": result["metrics"], "total_seconds": result["total_seconds"], "peak_rss_bytes": result["peak_rss_bytes"]}, indent=2))
    elif args.command == "prospect":
        print(json.dumps(prospect(args.output, args.anchor), indent=2))
    else:
        print(json.dumps(score_prospective(args.output), indent=2))


if __name__ == "__main__":
    main()
