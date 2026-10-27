#!/usr/bin/env python3
"""Register and run a separate sign/variance experiment on frozen hourly data."""

from datetime import datetime, timedelta, timezone
import argparse
import gzip
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_decomposition as model  # noqa: E402
from research import timestral_experiment as shared  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/timestral-capacity-20261026/history-source/hours.jsonl.gz"
CODE = [
    "research/timestral_decomposition.py",
    "scripts/run_timestral_decomposition.py",
    "research/timestral_history.py",
    "research/timestral_funding.py",
    "research/timestral_experiment.py",
]


def hashes():
    return {n: shared.digest_file(ROOT / n) for n in CODE}


def register(root):
    plan = json.loads((root / "plan.json").read_text())
    p = {
        "registered_at": shared.stamp(datetime.now(timezone.utc)),
        "plan": plan,
        "plan_sha256": shared.digest_file(root / "plan.json"),
        "code_sha256": hashes(),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": shared.digest_file(SOURCE),
        "direction_parameters": model.DIRECTION,
        "variance_parameters": model.VARIANCE,
        "sklearn_version": importlib.metadata.version("scikit-learn"),
        "numpy_version": np.__version__,
        "cohorts": {
            "2024": ["2024-01-01", "2025-01-01"],
            "2025": ["2025-01-01", "2026-01-01"],
            "2026": ["2026-01-01", "2026-08-01"],
        },
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
        hashes() != p["code_sha256"]
        or shared.digest_file(SOURCE) != p["source_sha256"]
        or shared.digest_file(root / "plan.json") != p["plan_sha256"]
        or shared.digest_file(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
        or p["sklearn_version"] != importlib.metadata.version("scikit-learn")
        or p["numpy_version"] != np.__version__
    ):
        raise ValueError("registered experiment changed")
    shared.exclusive_json(
        root / "opened.json", {"at": shared.stamp(datetime.now(timezone.utc))}
    )
    started = time.perf_counter()
    hours = [
        json.loads(line) for line in gzip.decompress(SOURCE.read_bytes()).splitlines()
    ]
    rows, audit = model.examples(hours)
    parameters = {}
    evaluated = []
    with (
        (root / "predictions.jsonl").open("x") as pf,
        (root / "outcomes.jsonl").open("x") as of,
    ):
        for year, (begin, end) in p["cohorts"].items():
            cohort = [
                r
                for r in rows
                if shared.utc(begin) <= shared.utc(r["anchor"]) < shared.utc(end)
            ]
            for month in sorted({r["anchor"][:7] for r in cohort}):
                cutoff = shared.utc(month + "-01")
                for h in (4, 24):
                    training = [
                        r
                        for r in rows
                        if r["horizon_hours"] == h
                        and shared.utc(r["label_end"]) <= cutoff
                        and shared.utc(r["anchor"]) >= cutoff - timedelta(days=1095)
                    ]
                    key = f"{month}:{h}"
                    parameters[key] = model.fit(training, cutoff, h)
                    for r in cohort:
                        if r["horizon_hours"] != h or not r["anchor"].startswith(month):
                            continue
                        forecast = {
                            k: v
                            for k, v in r.items()
                            if k not in ("return", "variance")
                        }
                        forecast.update(
                            cohort=year,
                            fit_key=key,
                            predictions=model.predict(parameters[key], r),
                        )
                        shared.append_record(pf, forecast)
                        shared.append_record(
                            of,
                            {
                                "sample_id": r["sample_id"],
                                "return": r["return"],
                                "variance": r["variance"],
                            },
                        )
                        evaluated.append(
                            {
                                **forecast,
                                "return": r["return"],
                                "variance": r["variance"],
                            }
                        )
                print("Completed " + month, file=sys.stderr, flush=True)
    result = {
        year: {
            str(h): {
                kind: model.score(
                    [
                        r
                        for r in evaluated
                        if r["cohort"] == year and r["horizon_hours"] == h
                    ],
                    kind,
                )
                for kind in ("direction", "variance")
            }
            for h in (4, 24)
        }
        for year in p["cohorts"]
    }
    diagnostics = {}
    for h in (4, 24):
        group = sorted(
            [r for r in evaluated if r["horizon_hours"] == h], key=lambda r: r["anchor"]
        )
        pairs = [
            (a, b)
            for a, b in zip(group, group[1:])
            if shared.utc(a["label_end"]) == shared.utc(b["anchor"])
        ]
        diagnostics[str(h)] = {"pairs": len(pairs)}
        for kind, fn in [
            ("return", lambda r: r["return"]),
            ("sign", lambda r: np.sign(r["return"])),
            ("absolute_return", lambda r: abs(r["return"])),
            ("realized_variance", lambda r: r["variance"]),
            ("log_realized_variance", lambda r: np.log(r["variance"])),
        ]:
            diagnostics[str(h)][kind + "_lag1_correlation"] = float(
                np.corrcoef([fn(a) for a, b in pairs], [fn(b) for a, b in pairs])[0, 1]
            )
    shared.exclusive_json(root / "parameters.json", parameters)
    answer = {
        "cohorts": result,
        "diagnostics": diagnostics,
        "source_audit": audit,
        "passes_all_cohorts": {
            str(h): {
                kind: all(
                    result[y][str(h)][kind]["passes_both_baselines"] for y in result
                )
                for kind in ("direction", "variance")
            }
            for h in (4, 24)
        },
        "prospective_validation": False,
        "trading_validation": False,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": shared.peak_rss_bytes(),
        "environment": shared.environment(),
        "protocol_sha256": shared.digest_file(root / "protocol.json"),
        "output_sha256": {
            n: shared.digest_file(root / n)
            for n in ("predictions.jsonl", "outcomes.jsonl", "parameters.json")
        },
    }
    shared.exclusive_json(root / "results.json", answer)
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["register", "run"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = register(args.root) if args.phase == "register" else run(args.root)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("registered_at", "passes_all_cohorts", "wall_seconds")
                if k in result
            }
        )
    )
