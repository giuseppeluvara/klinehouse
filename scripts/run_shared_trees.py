#!/usr/bin/env python3
"""Register a single nonlinear follow-up; execute all three historical cohorts."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_universal as shared  # noqa: E402
from research import timestral_shared_trees as trees  # noqa: E402
from scripts.run_universal_history import examples, epoch  # noqa: E402
from scripts.run_timestral_microdirection import ROOT, digest, write  # noqa: E402

REPORT = ROOT / "reports/timestral-universal-20261027"
PRIOR = ROOT / "reports/timestral-microdirection-20261027"
CODE = (
    "research/timestral_shared_trees.py",
    "research/timestral_universal.py",
    "research/timestral_reversal.py",
    "research/timestral_microdirection.py",
    "scripts/run_shared_trees.py",
    "scripts/run_universal_history.py",
    "scripts/run_timestral_microdirection.py",
)
PLAN = {
    "purpose": "One nonlinear shared model to test regime-dependent relationships; follow-up to linear failure on three of six older checks.",
    "history_status": "All cohort results are now explored. This is a newly registered retrospective candidate, not independent confirmation.",
    "model": trees.PARAMETERS,
    "features": "Exact23 frozen universal features; sign-reflected augmentation plus symmetric averaging of mirrored prediction probabilities. Shared weights across BTC/ETH/SOL, no asset IDs.",
    "cohorts": [2024, 2025, 2026],
    "training": "January-July cohorts, monthly90-day matured training, five-minute target after one-minute delay.",
    "gate": "Exact unchanged universal.evaluate criterion, threshold52%, all NINE asset/cohort checks must pass. No selecting a horizon, threshold, feature set, or model hyperparameter from results.",
    "artifacts": "Numeric tree nodes with values and thresholds, no pickle. Verify export against sklearn before publication. Independent audit reconstructs predictions; it does not rerun greedy tree fitting.",
}


def inputs():
    paths = [REPORT / "history" / f"{s}-minutes.npz" for s in shared.SYMBOLS]
    paths += [PRIOR / "micro/minutes.npz"] + [
        PRIOR / f"transfer/{s}-minutes.npz" for s in shared.SYMBOLS[1:]
    ]
    paths += [
        REPORT / "history" / f"{year}-{s}-predictions.npz"
        for year in (2024, 2025)
        for s in shared.SYMBOLS
    ]
    paths += [REPORT / f"{s}-predictions.npz" for s in shared.SYMBOLS]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def register(root):
    write(
        root / "protocol.json",
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "plan": PLAN,
            "inputs": inputs(),
            "code": {n: digest(ROOT / n) for n in CODE},
            "prior_results_sha256": digest(REPORT / "history/results.json"),
        },
    )
    write(root / "protocol.sha256.json", {"sha256": digest(root / "protocol.json")})


def run(root):
    p = json.loads((root / "protocol.json").read_text())
    if (
        p["plan"] != PLAN
        or p["inputs"] != inputs()
        or p["code"] != {n: digest(ROOT / n) for n in CODE}
        or digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
    ):
        raise ValueError("frozen nonlinear experiment changed")
    write(root / "opened.json", {"at": datetime.now(timezone.utc).isoformat()})
    old, _, _ = examples(
        {
            s: dict(np.load(REPORT / "history" / f"{s}-minutes.npz"))
            for s in shared.SYMBOLS
        }
    )
    recent = shared.examples(
        {
            s: dict(
                np.load(
                    PRIOR / "micro/minutes.npz"
                    if a == 0
                    else PRIOR / f"transfer/{s}-minutes.npz"
                )
            )
            for a, s in enumerate(shared.SYMBOLS)
        }
    )
    fits, results = {}, {}
    for year in (2024, 2025, 2026):
        sample = recent if year == 2026 else old
        ps = []
        ii = []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (sample["anchor"] >= cutoff - 90 * 86400) & (
                sample["end"] <= cutoff
            )
            test = (sample["anchor"] >= cutoff) & (
                sample["anchor"] < epoch(year, month + 1)
            )
            x = np.concatenate([sample["x"][a, train] for a in range(3)])
            y = np.concatenate([sample["y"][a, train] for a in range(3)])
            f = trees.fit(x, y, np.tile(sample["end"][train], 3), cutoff)
            ids = np.concatenate(
                [
                    np.column_stack((np.full(train.sum(), a), sample["anchor"][train]))
                    for a in range(3)
                ]
            ).astype("<i8")
            f["training_ids_sha256"] = hashlib.sha256(ids.tobytes()).hexdigest()
            f["training_labels_sha256"] = hashlib.sha256(
                y.astype("<i8").tobytes()
            ).hexdigest()
            fits[f"{year}_{month}"] = f
            ps.append(
                np.stack(
                    [
                        trees.predict(f, sample["x"][a, test], sample["anchor"][test])
                        for a in range(3)
                    ]
                )
            )
            ii.append(np.flatnonzero(test))
            print(f"fitted nonlinear {year}-{month:02d}", flush=True)
        pred = np.concatenate(ps, axis=1)
        ii = np.concatenate(ii)
        for a, s in enumerate(shared.SYMBOLS):
            path = (
                REPORT / f"{s}-predictions.npz"
                if year == 2026
                else REPORT / "history" / f"{year}-{s}-predictions.npz"
            )
            previous = dict(np.load(path))
            if not np.array_equal(previous["anchor"], sample["anchor"][ii]):
                raise ValueError("baseline or outcome membership mismatch")
            with (root / f"{year}-{s}-predictions.npz").open("xb") as f:
                np.savez_compressed(f, anchor=sample["anchor"][ii], probability=pred[a])
            results[f"{year}-{s}"] = shared.evaluate(
                sample["y"][a, ii],
                pred[a],
                previous["frequency"],
                previous["prior_reversal"],
                sample["anchor"][ii],
            )
            print(
                year,
                s,
                json.dumps(
                    {
                        k: results[f"{year}-{s}"].get(k)
                        for k in ("accuracy", "coverage", "gate")
                    }
                ),
                flush=True,
            )
    with (root / "fits.json.gz").open("xb") as f:
        f.write(
            gzip.compress(
                json.dumps(
                    fits, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode(),
                mtime=0,
            )
        )
    write(
        root / "results.json",
        {
            "cohorts": results,
            "all_cohorts_gate": all(v["gate"] for v in results.values()),
        },
    )
    write(
        root / "manifest.json",
        {
            p.name: digest(p)
            for p in sorted(root.iterdir())
            if p.suffix in (".json", ".npz", ".gz")
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {"register": register, "run": run}[args.action](args.output)
