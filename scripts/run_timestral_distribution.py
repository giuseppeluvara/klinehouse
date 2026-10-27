#!/usr/bin/env python3
"""Register and run the conditional-distribution follow-up without refitting means."""

from datetime import datetime, timedelta, timezone
import argparse
import gzip
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_experiment as shared  # noqa: E402
from research import timestral_decomposition as decomposition  # noqa: E402
from research import timestral_distribution as distribution  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/timestral-capacity-20261026/history-source/hours.jsonl.gz"
PRIOR = ROOT / "research/out/timestral-decomposition-20261027"
CODE = [
    "research/timestral_distribution.py",
    "scripts/run_timestral_distribution.py",
    "research/timestral_decomposition.py",
    "research/timestral_history.py",
    "research/timestral_funding.py",
    "research/timestral_experiment.py",
]


def hashes():
    return {n: shared.digest_file(ROOT / n) for n in CODE}


def register(root):
    p = {
        "registered_at": shared.stamp(datetime.now(timezone.utc)),
        "plan": json.loads((root / "plan.json").read_text()),
        "plan_sha256": shared.digest_file(root / "plan.json"),
        "code_sha256": hashes(),
        "source_sha256": shared.digest_file(SOURCE),
        "prior_parameters_sha256": shared.digest_file(PRIOR / "parameters.json"),
        "scipy_version": importlib.metadata.version("scipy"),
    }
    shared.exclusive_json(root / "protocol.json", p)
    shared.exclusive_json(
        root / "protocol.sha256.json",
        {"sha256": shared.digest_file(root / "protocol.json")},
    )
    return p


def run(root):
    p = json.loads((root / "protocol.json").read_text())
    if (
        p["code_sha256"] != hashes()
        or p["plan_sha256"] != shared.digest_file(root / "plan.json")
        or p["source_sha256"] != shared.digest_file(SOURCE)
        or p["prior_parameters_sha256"] != shared.digest_file(PRIOR / "parameters.json")
        or p["scipy_version"] != importlib.metadata.version("scipy")
        or shared.digest_file(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
    ):
        raise ValueError("frozen experiment changed")
    shared.exclusive_json(
        root / "opened.json", {"at": shared.stamp(datetime.now(timezone.utc))}
    )
    started = time.perf_counter()
    rows, audit = decomposition.examples(
        [json.loads(line) for line in gzip.decompress(SOURCE.read_bytes()).splitlines()]
    )
    parameters = json.loads((PRIOR / "parameters.json").read_text())
    calibrations = {}
    evaluated = []
    with (
        (root / "predictions.jsonl").open("x") as pf,
        (root / "outcomes.jsonl").open("x") as of,
    ):
        for key, fit in sorted(parameters.items()):
            cutoff = shared.utc(fit["cutoff"])
            h = int(key.split(":")[1])
            month = key[:7]
            training = [
                r
                for r in rows
                if r["horizon_hours"] == h
                and shared.utc(r["label_end"]) <= cutoff
                and shared.utc(r["anchor"]) >= cutoff - timedelta(days=1095)
            ]
            import hashlib

            if (
                hashlib.sha256(
                    shared.canonical([r["sample_id"] for r in training])
                ).hexdigest()
                != fit["sample_ids_sha256"]
            ):
                raise ValueError("training membership changed")
            calibrations[key] = distribution.calibrate(training, fit, cutoff)
            for r in rows:
                if r["horizon_hours"] != h or not r["anchor"].startswith(month):
                    continue
                forecast = {
                    k: v for k, v in r.items() if k not in ("return", "variance")
                }
                forecast.update(
                    fit_key=key,
                    cohort=month[:4],
                    distribution=distribution.predict(calibrations[key], fit, r),
                )
                shared.append_record(pf, forecast)
                shared.append_record(
                    of, {"sample_id": r["sample_id"], "variance": r["variance"]}
                )
                evaluated.append({**forecast, "variance": r["variance"]})
    results = {
        year: {
            str(h): distribution.score(
                [
                    r
                    for r in evaluated
                    if r["cohort"] == year and r["horizon_hours"] == h
                ]
            )
            for h in (4, 24)
        }
        for year in ("2024", "2025", "2026")
    }
    shared.exclusive_json(root / "calibrations.json", calibrations)
    result = {
        "cohorts": results,
        "passes_all_cohorts": {
            str(h): all(results[y][str(h)]["passes_both_baselines"] for y in results)
            for h in (4, 24)
        },
        "direction_validated": False,
        "prospective_validation": False,
        "trading_validation": False,
        "source_audit": audit,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": shared.peak_rss_bytes(),
        "protocol_sha256": shared.digest_file(root / "protocol.json"),
        "output_sha256": {
            n: shared.digest_file(root / n)
            for n in ("predictions.jsonl", "outcomes.jsonl", "calibrations.json")
        },
    }
    shared.exclusive_json(root / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["register", "run"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    r = register(args.root) if args.phase == "register" else run(args.root)
    print(
        json.dumps(
            {
                k: r[k]
                for k in ("registered_at", "passes_all_cohorts", "wall_seconds")
                if k in r
            }
        )
    )
