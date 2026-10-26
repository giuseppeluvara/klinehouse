#!/usr/bin/env python3
"""Run the predeclared 2x2 capacity/context comparison without changing old runs."""

from datetime import datetime, timedelta, timezone
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_capacity as capacity  # noqa: E402
from research import timestral_experiment as shared  # noqa: E402
from research import timestral_repair as repair  # noqa: E402
from research.timestral_long_horizon import Hour  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/timestral-diagnosis-20261026"
CACHE = ROOT / "research/out/timestral-pilot-20261025/model-cache"
CODE = (
    "research/timestral_capacity.py",
    "scripts/run_timestral_capacity.py",
    "research/timestral_experiment.py",
    "research/timestral_repair.py",
    "research/timestral_long_horizon.py",
)


def hashes():
    return {n: shared.digest_file(ROOT / n) for n in CODE}


def checkpoints():
    return {
        size: {
            name: shared.digest_file(
                CACHE
                / ("models--" + spec["id"].replace("/", "--"))
                / "snapshots"
                / spec["revision"]
                / name
            )
            for name in ("config.json", "model.safetensors")
        }
        for size, spec in capacity.MODELS.items()
    }


def register(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    weights = checkpoints()
    if any(
        weights[size]["model.safetensors"] != spec["sha256"]
        or weights[size]["config.json"] != spec["config_sha256"]
        for size, spec in capacity.MODELS.items()
    ):
        raise ValueError("checkpoint checksum mismatch")
    p = {
        "registered_at": shared.stamp(datetime.now(timezone.utc)),
        "code_sha256": hashes(),
        "source": str(SOURCE),
        "inputs": {
            n: shared.digest_file(SOURCE / n)
            for n in ("hours.jsonl", "run/evaluation.jsonl")
        },
        "checkpoints": weights,
        "model_specifications": capacity.MODELS,
        "conditions": capacity.CONDITIONS,
        "context_hours": [256, 2048],
        "horizons": [4, 24],
        "status": "historical research screen on already observed data; not independent validation",
        "hypothesis": "The 5M size or 256-hour context might be insufficient. Cross both factors, keeping the target, samples, API, units and all other settings fixed. No covariate or fine-tuning changes in this experiment.",
        "gate": "At least1% MAE reduction versus unchanged, no worse RMSE,5/7 monthly wins, upper paired weekly bootstrap below0;4000 draws seed20261026 two-sided99.375% interval for8 comparisons. Passing only nominates for independent validation, not deployment or success.",
        "controls": "constant, linear and64-hour sinusoid through each size/context at4/24h; retain all results",
        "freeze": "No tuning of weights, context, threshold, dates or candidates after outcomes; protected Aug2026-Feb2027 data remain unopened",
        "runtime": {
            "device": "cpu",
            "threads": 2,
            "seed": 7,
            "new_checkpoint_bytes": 111296464,
        },
    }
    shared.exclusive_json(output / "protocol.json", p)
    shared.exclusive_json(
        output / "protocol.sha256.json",
        {"sha256": shared.digest_file(output / "protocol.json")},
    )
    return p


def verify(output):
    output = Path(output)
    p = json.loads((output / "protocol.json").read_text())
    if (
        shared.digest_file(output / "protocol.json")
        != json.loads((output / "protocol.sha256.json").read_text())["sha256"]
        or p["code_sha256"] != hashes()
        or p["checkpoints"] != checkpoints()
        or any(shared.digest_file(SOURCE / n) != v for n, v in p["inputs"].items())
    ):
        raise ValueError("registered inputs or code changed")
    return p


def run(output):
    output = Path(output)
    protocol = verify(output)
    shared.exclusive_json(
        output / "opened.json",
        {
            "at": shared.stamp(datetime.now(timezone.utc)),
            "protocol_sha256": shared.digest_file(output / "protocol.json"),
        },
    )
    started = time.perf_counter()
    hours = [
        Hour(shared.utc(r["time"]), r["log_close"], tuple(r["features"]))
        for r in (
            json.loads(line)
            for line in (SOURCE / "hours.jsonl").read_text().splitlines()
        )
    ]
    rows = [
        json.loads(line)
        for line in (SOURCE / "run/evaluation.jsonl").read_text().splitlines()
    ]
    lookup = {shared.stamp(r.time): i for i, r in enumerate(hours)}
    if (
        len(rows) != 1484
        or len(hours) != 8016
        or hours[-1].time >= shared.utc("2026-08-01")
    ):
        raise ValueError("unexpected or protected source")
    combined = [
        {**r, "predictions": {"unchanged": r["predictions"]["unchanged_dynamic"]}}
        for r in rows
    ]
    maximum_reproduction_error, controls, runtime = 0.0, [], {}
    with (output / "forecasts.jsonl").open("x") as handle:
        for size in capacity.MODELS:
            model = capacity.Forecast(size, CACHE)
            for length in (256, 2048):
                condition = f"{size}_{length}"
                begin = time.perf_counter()
                for i, row in enumerate(rows):
                    idx, h = lookup[row["anchor"]], row["horizon_hours"]
                    context = hours[idx - length : idx]
                    if len(context) != length or context[-1].time + timedelta(
                        hours=1
                    ) != shared.utc(row["anchor"]):
                        raise ValueError("missing past context")
                    prediction = model.predict(context, {"horizon": h}, False)
                    record = {
                        "condition": condition,
                        "sample_id": row["sample_id"],
                        "anchor": row["anchor"],
                        "horizon_hours": h,
                        "input_start": shared.stamp(context[0].time),
                        "prediction": prediction,
                    }
                    shared.append_record(handle, record)
                    actual = hours[idx + h - 1].log_close - context[-1].log_close
                    if abs(actual - row["actual_log_return"]) > 1e-14:
                        raise ValueError("target mismatch")
                    combined[i]["predictions"][condition] = prediction
                    if condition == "5m_256":
                        maximum_reproduction_error = max(
                            maximum_reproduction_error,
                            abs(
                                prediction["point"]
                                - row["predictions"]["timestral_price"]["point"]
                            ),
                        )
                    if (i + 1) % 400 == 0:
                        print(
                            f"{condition}: {i + 1}/{len(rows)}",
                            file=sys.stderr,
                            flush=True,
                        )
                runtime[condition] = time.perf_counter() - begin
                t = np.arange(length + 24)
                for name, values in (
                    ("constant", np.full(len(t), 10.0)),
                    ("linear", 10.0 + 0.0002 * t),
                    ("sine", 10.0 + 0.01 * np.sin(2 * np.pi * t / 64)),
                ):
                    context = [
                        Hour(
                            shared.utc("2000-01-01") + timedelta(hours=j),
                            float(v),
                            (0.0, 0.0, 0.0),
                        )
                        for j, v in enumerate(values[:length])
                    ]
                    for h in (4, 24):
                        controls.append(
                            {
                                "condition": condition,
                                "case": name,
                                "horizon": h,
                                "prediction": model.predict(
                                    context, {"horizon": h}, False
                                ),
                                "actual": float(
                                    values[length + h - 1] - values[length - 1]
                                ),
                            }
                        )
            del model
            gc.collect()
    result = {}
    for h in (4, 24):
        group = [r for r in combined if r["horizon_hours"] == h]
        result[str(h)] = {
            "metrics": {
                name: repair.score(group, name)
                for name in ("unchanged", *capacity.CONDITIONS)
            },
            "screens": {
                name: capacity.screen(group, name) for name in capacity.CONDITIONS
            },
            "monthly": {
                month: {
                    name: repair.score(
                        [r for r in group if r["anchor"][:7] == month], name
                    )
                    for name in ("unchanged", *capacity.CONDITIONS)
                }
                for month in sorted({r["anchor"][:7] for r in group})
            },
        }
    shared.exclusive_json(output / "controls.json", controls)
    results = {
        "status": protocol["status"],
        "horizons": result,
        "maximum_original_point_difference": maximum_reproduction_error,
        "inference_seconds": runtime,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": shared.peak_rss_bytes(),
        "environment": shared.environment(),
        "protocol_sha256": shared.digest_file(output / "protocol.json"),
        "output_sha256": {
            n: shared.digest_file(output / n)
            for n in ("forecasts.jsonl", "controls.json")
        },
    }
    shared.exclusive_json(output / "results.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = register(args.output) if args.phase == "register" else run(args.output)
    print(
        json.dumps(
            {k: r[k] for k in ("status", "wall_seconds", "peak_rss_bytes") if k in r}
        )
    )


if __name__ == "__main__":
    main()
