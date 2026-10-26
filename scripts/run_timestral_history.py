#!/usr/bin/env python3
"""Compare training history lengths on three predeclared historical cohorts."""

from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_experiment as shared  # noqa: E402
from research import timestral_history as history  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CODE = (
    "research/timestral_history.py",
    "scripts/run_timestral_history.py",
    "scripts/prepare_timestral_history.py",
    "research/timestral_funding.py",
    "research/timestral_experiment.py",
)


def hashes():
    return {n: shared.digest_file(ROOT / n) for n in CODE}


def register(output, source, plan):
    output, source, plan = Path(output), Path(source).resolve(), Path(plan).resolve()
    provenance = json.loads((source / "provenance.json").read_text())
    if shared.digest_file(source / "hours.jsonl") != provenance["hours_sha256"]:
        raise ValueError("source changed")
    output.mkdir(parents=True, exist_ok=False)
    p = {
        "registered_at": shared.stamp(datetime.now(timezone.utc)),
        "code_sha256": hashes(),
        "source": str(source),
        "inputs": {
            n: shared.digest_file(source / n)
            for n in ("hours.jsonl", "provenance.json")
        },
        "plan": json.loads(plan.read_text()),
        "plan_sha256": shared.digest_file(plan),
        "plan_path": str(plan),
        "status": "separate historical training-window ablation; design informed by later2026 results, not prospective evidence",
        "parameters": history.PARAMETERS,
        "windows_days": [180, 1095],
        "features": history.FEATURES,
        "cohorts": {
            "2024": ["2024-01-01", "2025-01-01"],
            "2025": ["2025-01-01", "2026-01-01"],
            "2026": ["2026-01-01", "2026-08-01"],
        },
        "sklearn_version": importlib.metadata.version("scikit-learn"),
        "numpy_version": np.__version__,
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
    source = Path(p["source"])
    if (
        p["code_sha256"] != hashes()
        or shared.digest_file(output / "protocol.json")
        != json.loads((output / "protocol.sha256.json").read_text())["sha256"]
        or any(shared.digest_file(source / n) != v for n, v in p["inputs"].items())
        or shared.digest_file(p["plan_path"]) != p["plan_sha256"]
        or p["sklearn_version"] != importlib.metadata.version("scikit-learn")
        or p["numpy_version"] != np.__version__
    ):
        raise ValueError("registered source, code, plan or environment changed")
    return p


def run(output):
    output = Path(output)
    p = verify(output)
    source = Path(p["source"])
    shared.exclusive_json(
        output / "opened.json", {"at": shared.stamp(datetime.now(timezone.utc))}
    )
    (output / "models").mkdir()
    started = time.perf_counter()
    hours = [
        json.loads(line) for line in (source / "hours.jsonl").read_text().splitlines()
    ]
    if any(shared.utc(r["time"]) >= shared.utc("2026-08-01") for r in hours):
        raise ValueError("protected dates")
    rows, audit = history.examples(hours)
    # Independent source reconciliation on the already observed overlap.
    old_source = ROOT / "reports/timestral-diagnosis-20261026/hours.jsonl"
    old = {
        r["time"]: r
        for r in (json.loads(line) for line in old_source.read_text().splitlines())
    }
    matches = [
        abs(np.log(r["close"]) - old[r["time"]]["log_close"])
        for r in hours
        if r["time"] in old
    ]
    if len(matches) != 8016 or max(matches) > 1e-12:
        raise ValueError("native and1s-derived hourly closes disagree")
    fitted = {}
    evaluated = []
    with (
        (output / "predictions.jsonl").open("x") as pf,
        (output / "outcomes.jsonl").open("x") as of,
    ):
        for year, (start, end) in p["cohorts"].items():
            group = [
                r
                for r in rows
                if shared.utc(start) <= shared.utc(r["anchor"]) < shared.utc(end)
            ]
            for month in sorted({r["anchor"][:7] for r in group}):
                cutoff = shared.utc(month + "-01")
                for h in (4, 24):
                    current = [
                        r
                        for r in group
                        if r["horizon_hours"] == h and r["anchor"].startswith(month)
                    ]
                    if not current:
                        raise ValueError("no evaluation rows in required month")
                    x = np.asarray([r["features"] for r in current])
                    scale = np.asarray([r["scale"] for r in current])
                    for days in p["windows_days"]:
                        past = [
                            r
                            for r in rows
                            if r["horizon_hours"] == h
                            and shared.utc(r["label_end"]) <= cutoff
                            and shared.utc(r["anchor"]) >= cutoff - timedelta(days=days)
                        ]
                        model = history.fit(past, cutoff, h)
                        points = model.predict(x) * scale
                        key = f"{month}-{h}-{days}"
                        file = output / "models" / f"{key}.pkl"
                        with file.open("xb") as handle:
                            pickle.dump(model, handle, protocol=5)
                        fitted[key] = {
                            "cutoff": shared.stamp(cutoff),
                            "samples": len(past),
                            "latest_label_end": max(r["label_end"] for r in past),
                            "training_anchors_sha256": hashlib.sha256(
                                shared.canonical([r["anchor"] for r in past])
                            ).hexdigest(),
                            "model_sha256": shared.digest_file(file),
                        }
                        for r, point in zip(current, points):
                            identity = f"{r['anchor']}:{h}:{days}"
                            prediction = {
                                "sample_id": identity,
                                "anchor": r["anchor"],
                                "horizon_hours": h,
                                "window_days": days,
                                "point": float(point),
                                "scale": r["scale"],
                                "features": r["features"],
                                "fit_key": key,
                                "cohort": year,
                            }
                            shared.append_record(pf, prediction)
                            shared.append_record(
                                of, {"sample_id": identity, "actual": r["actual"]}
                            )
                            evaluated.append({**prediction, "actual": r["actual"]})
                print(
                    f"Completed training-window comparison: {month}",
                    file=sys.stderr,
                    flush=True,
                )
    result = {}
    for year in p["cohorts"]:
        result[year] = {
            str(h): {
                str(days): history.metrics(
                    [
                        r
                        for r in evaluated
                        if r["cohort"] == year
                        and r["horizon_hours"] == h
                        and r["window_days"] == days
                    ]
                )
                for days in p["windows_days"]
            }
            for h in (4, 24)
        }
    shared.exclusive_json(output / "fits.json", fitted)
    result = {
        "status": p["status"],
        "cohorts": result,
        "data": audit,
        "matched_source_hours": len(matches),
        "maximum_close_log_difference": max(matches),
        "all_cohorts_pass": {
            str(h): {
                str(days): all(
                    result[year][str(h)][str(days)]["passes_historical_screen"]
                    for year in p["cohorts"]
                )
                for days in p["windows_days"]
            }
            for h in (4, 24)
        },
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": shared.peak_rss_bytes(),
        "environment": shared.environment(),
        "sklearn_version": p["sklearn_version"],
        "protocol_sha256": shared.digest_file(output / "protocol.json"),
        "output_sha256": {
            n: shared.digest_file(output / n)
            for n in ("predictions.jsonl", "outcomes.jsonl", "fits.json")
        },
    }
    shared.exclusive_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default="research/out/timestral-capacity-20261026/history-source",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default="research/out/timestral-capacity-20261026/history-plan.json",
    )
    args = parser.parse_args()
    r = (
        register(args.output, args.source, args.plan)
        if args.phase == "register"
        else run(args.output)
    )
    print(
        json.dumps(
            {k: r[k] for k in ("status", "wall_seconds", "peak_rss_bytes") if k in r}
        )
    )
