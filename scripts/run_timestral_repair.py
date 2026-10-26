#!/usr/bin/env python3
"""Register and execute a causal diagnostic replay on already observed history."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_experiment as shared  # noqa: E402
from research import timestral_repair as repair  # noqa: E402
from research.timestral_long_horizon import Hour  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("unchanged_static", "unchanged_dynamic", "timestral_price", "timestral_returns", "evidence_policy")


def read_json(path):
    return json.loads(Path(path).read_text())


def code_hashes():
    return {name: shared.digest_file(ROOT / name) for name in
            ("research/timestral_repair.py", "scripts/run_timestral_repair.py",
             "research/timestral_experiment.py", "research/timestral_long_horizon.py")}


def register(root):
    root = Path(root)
    provenance = read_json(root / "cache-provenance.json")
    if shared.digest_file(root / "hours.jsonl") != provenance["hours_sha256"]:
        raise ValueError("hour cache checksum mismatch")
    protocol = {"registered_at": shared.stamp(datetime.now(timezone.utc)),
                "status": "diagnostic replay of already observed data; not a fresh holdout or prospective test",
                "plan": read_json(root / "plan.json"), "plan_sha256": shared.digest_file(root / "plan.json"),
                "cache_provenance_sha256": shared.digest_file(root / "cache-provenance.json"),
                "hours_sha256": provenance["hours_sha256"], "code_sha256": code_hashes(),
                "fit_start": "2025-10-01T00:00:00Z", "evaluation_start": "2026-01-01T00:00:00Z",
                "evaluation_end": "2026-08-01T00:00:00Z", "horizons": [4, 24], "models": list(MODELS),
                "model_config": {"lookback": 256, "torch_threads": 2, "local_files_only": True,
                                 "model_cache": str(ROOT / "research/out/timestral-pilot-20261025/model-cache")},
                "model": {"id": shared.MODEL_ID, "code_revision": shared.CODE_REVISION,
                          "weight_revision": shared.WEIGHT_REVISION},
                "evaluation_policy": "include every aligned complete anchor, no post-replay tuning or winner replacement",
                "protection": "No source or target on or after August 1, 2026 is opened."}
    output = root / "run"
    output.mkdir(exist_ok=False)
    shared.exclusive_json(output / "protocol.json", protocol)
    shared.exclusive_json(output / "protocol.sha256.json", {"sha256": shared.digest_file(output / "protocol.json")})
    return protocol


def verified_protocol(root):
    root = Path(root)
    p = read_json(root / "run/protocol.json")
    if (shared.digest_file(root / "run/protocol.json") != read_json(root / "run/protocol.sha256.json")["sha256"]
            or p["code_sha256"] != code_hashes()
            or p["hours_sha256"] != shared.digest_file(root / "hours.jsonl")
            or p["cache_provenance_sha256"] != shared.digest_file(root / "cache-provenance.json")
            or p["plan_sha256"] != shared.digest_file(root / "plan.json")):
        raise ValueError("registered replay inputs or code changed")
    return p


def replay(hours, protocol, model, emit):
    """The emitter sees each prediction before this loop looks up its outcome."""
    start, evaluation, end = (shared.utc(protocol[k]) for k in ("fit_start", "evaluation_start", "evaluation_end"))
    if not start < evaluation < end <= shared.utc("2026-08-01"):
        raise ValueError("invalid or protected replay period")
    lookup = {row.time: i for i, row in enumerate(hours)}
    if len(lookup) != len(hours) or any(b.time - a.time != timedelta(hours=1) for a, b in zip(hours, hours[1:])):
        raise ValueError("hour cache has duplicates, disorder or gaps")
    history, decisions, evaluation_rows, static = [], {}, [], {}
    anchor = start
    while anchor < end:
        for horizon in (4, 24):
            if anchor.hour % horizon or anchor + timedelta(hours=horizon) > end:
                continue
            left, right = lookup[anchor - timedelta(hours=256)], lookup[anchor - timedelta(hours=1)] + 1
            context = hours[left:right]
            forecasts, scale = repair.forecast_candidates(model, context, anchor, horizon)
            context_hash = hashlib.sha256(shared.canonical([[shared.stamp(x.time), x.log_close, x.features] for x in context])).hexdigest()
            sample_id = hashlib.sha256(shared.canonical(["BTCUSDT", shared.stamp(anchor), horizon, context_hash])).hexdigest()
            record = {"sample_id": sample_id, "anchor": shared.stamp(anchor), "horizon_hours": horizon,
                      "label_end": shared.stamp(anchor + timedelta(hours=horizon)),
                      "input_start": shared.stamp(context[0].time), "input_end_exclusive": shared.stamp(anchor),
                      "input_sha256": context_hash, "scale": scale, "reference_price": float(np.exp(context[-1].log_close)),
                      "computed_at": shared.stamp(datetime.now(timezone.utc)), "predictions": forecasts,
                      "status": "historical_diagnostic", "independent_test": False}
            if anchor >= evaluation:
                key = f"{horizon}:{anchor:%Y-%m}"
                if key not in decisions:
                    decisions[key] = repair.choose_model(history, anchor, horizon)
                chosen = decisions[key]["selected"]
                point = 0. if chosen == "unchanged" else forecasts[chosen]["point"]
                interval = repair.calibrated_interval(history, anchor, horizon, chosen, point, scale)
                unchanged = repair.calibrated_interval(history, anchor, horizon, "unchanged", 0., scale)
                if interval["quantiles"] is None or unchanged["quantiles"] is None:
                    raise ValueError("insufficient past calibration on the evaluation period")
                if horizon not in static:
                    past = repair.matured(history, evaluation, horizon)
                    static[horizon] = np.quantile([x["actual_log_return"] for x in past], shared.LEVELS).tolist()
                record.update(selected=chosen, decision_key=key, interval_calibration=interval,
                              unchanged_calibration=unchanged)
                forecasts.update(unchanged_static={"point": 0., "quantiles": static[horizon]},
                                 unchanged_dynamic={"point": 0., "quantiles": unchanged["quantiles"]},
                                 evidence_policy={"point": point, "quantiles": interval["quantiles"]})
            emit("prediction", record)
            target = hours[lookup[anchor + timedelta(hours=horizon - 1)]]
            outcome = {"sample_id": sample_id, "actual_log_return": target.log_close - context[-1].log_close,
                       "actual_price": float(np.exp(target.log_close))}
            emit("outcome", outcome)
            combined = {**record, **outcome}
            history.append(combined)
            if anchor >= evaluation:
                evaluation_rows.append(combined)
        anchor += timedelta(hours=4)
    return history, evaluation_rows, decisions


def analyze(rows, decisions):
    result = {}
    for horizon in (4, 24):
        group = [row for row in rows if row["horizon_hours"] == horizon]
        result[str(horizon)] = {
            "samples": len(group), "selected_counts": dict(Counter(row["selected"] for row in group)),
            "metrics": {name: repair.score(group, name) for name in MODELS},
            "monthly": {month: {name: repair.score([row for row in group if row["anchor"][:7] == month], name)
                                for name in MODELS} for month in sorted({row["anchor"][:7] for row in group})},
        }
    return {"status": "exploratory replay on previously observed history", "independent_test": False,
            "prospective_test": "pending", "horizons": result, "decisions": decisions}


def export_csv(rows, output):
    records = []
    for row in rows:
        for name in MODELS:
            prediction = row["predictions"][name]
            records.append({"sample_id": row["sample_id"], "anchor_utc": row["anchor"],
                            "target_end_utc": row["label_end"], "horizon_hours": row["horizon_hours"], "model": name,
                            "reference_price_usdt": row["reference_price"],
                            "predicted_price_usdt": row["reference_price"] * np.exp(prediction["point"]),
                            "actual_price_usdt": row["actual_price"], "predicted_log_return": prediction["point"],
                            "actual_log_return": row["actual_log_return"],
                            "lower90_log_return": prediction["quantiles"][0], "upper90_log_return": prediction["quantiles"][2],
                            "selected_policy_model": row["selected"], "independent_test": False})
    with (output / "forecasts.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def run(root, model_factory=shared.Timestral):
    root = Path(root)
    protocol = verified_protocol(root)
    output = root / "run"
    started = time.perf_counter()
    hours = [Hour(shared.utc(x["time"]), x["log_close"], tuple(x["features"]))
             for x in (json.loads(line) for line in (root / "hours.jsonl").read_text().splitlines())]
    shared.exclusive_json(output / "opened.json", {"started_at": shared.stamp(datetime.now(timezone.utc)),
                                                  "protocol_sha256": shared.digest_file(output / "protocol.json")})
    model = model_factory(protocol["model_config"])
    with (output / "predictions.jsonl").open("x") as predictions, (output / "outcomes.jsonl").open("x") as outcomes:
        def emit(kind, record):
            shared.append_record(predictions if kind == "prediction" else outcomes, record)
        history, rows, decisions = replay(hours, protocol, model, emit)
    with (output / "evaluation.jsonl").open("x") as handle:
        for row in rows:
            handle.write(shared.canonical(row).decode() + "\n")
    analysis = analyze(rows, decisions)
    analysis.update(csv_rows=export_csv(rows, output), all_predictions=len(history),
                    wall_seconds=time.perf_counter() - started, peak_rss_bytes=shared.peak_rss_bytes(),
                    checkpoint_files_sha256=model.checkpoint_files, environment=shared.environment(),
                    protocol_sha256=shared.digest_file(output / "protocol.json"))
    analysis["output_sha256"] = {name: shared.digest_file(output / name) for name in
                                 ("predictions.jsonl", "outcomes.jsonl", "evaluation.jsonl", "forecasts.csv")}
    shared.exclusive_json(output / "analysis.json", analysis)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "run"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = register(args.root) if args.phase == "register" else run(args.root)
    print(json.dumps({key: result[key] for key in ("status", "csv_rows", "wall_seconds") if key in result}))


if __name__ == "__main__":
    main()
