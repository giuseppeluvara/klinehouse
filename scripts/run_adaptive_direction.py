#!/usr/bin/env python3
"""Register and replay a bounded adaptive calibration experiment."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_adaptive_direction as model  # noqa: E402
from research.timestral_universal import evaluate, SYMBOLS  # noqa: E402
from scripts.run_timestral_microdirection import ROOT, digest, write  # noqa: E402

SOURCE = ROOT / "reports/timestral-universal-20261027"
DEFAULT = ROOT / "reports/timestral-adaptive-direction-20261027"
CODE = (
    "research/timestral_adaptive_direction.py",
    "research/timestral_universal.py",
    "scripts/run_adaptive_direction.py",
    "scripts/run_timestral_microdirection.py",
)
PLAN = {
    "purpose": "Address overconfidence and temporal drift using only matured archived OOS forecasts; one primary adaptive method, one fixed calibration control, unchanged raw comparator.",
    "status": "Retrospective exploratory follow-up: all nine raw cohort results were already seen. This does not create an untouched validation set.",
    "cohorts": "BTC/ETH/SOL January-July of 2024, 2025, 2026, exact same anchors; reset calibration each January; seven-day identity warmup INCLUDED in all metrics.",
    "primary": "Daily per-asset calibration, trailing28 days, exponential weights half-life7 days based on outcome end; minimum1000 matured observations. Never use end>day cutoff. Same rule and hyperparameters for all assets/years.",
    "parameters": "Direction: sigmoid(alpha*log(p_up/p_down)), alpha in [0,1]. Flat: sigmoid(logit(p_flat)+offset), offset in [-3,3]. Joint multiclass logloss plus 1e-4*((alpha-1)^2+offset^2); separable convex roots. Recombine into three probabilities. Sign-symmetric, direction ordering preserved, no directional sign reversal.",
    "control": "Fixed control uses the first adaptive fit (January8) forever within that cohort; identical identity warmup.",
    "comparison": "All-sample paired Brier and logloss gains versus raw AND versus fixed control. 4000 seven-day moving-block bootstrap, seed271031, 99.444% intervals Bonferroni over9 cohorts for each metric/comparison separately. Conditional on saved fitted stream; no correction over historical experiments or all endpoints.",
    "gates": "Original universal.evaluate direction gate unchanged, threshold52%, all9 must pass. Probability improvement: both full-sample Brier and logloss gain lower bounds >0 on all9 versus raw. Adaptation-specific evidence separately applies the same gate versus fixed. No tuning after outcomes.",
    "protected_boundary": "2026-08-01T00:00:00Z, no new market sources or downloads; reuse only frozen pre-boundary outputs.",
    "reference": "https://scikit-learn.org/stable/modules/calibration.html",
}


def paths(year, symbol):
    parent = SOURCE if year == 2026 else SOURCE / "history"
    prefix = symbol if year == 2026 else f"{year}-{symbol}"
    return [parent / f"{prefix}-{suffix}.npz" for suffix in ("predictions", "outcomes")]


def inputs():
    return {
        str(p.relative_to(ROOT)): digest(p)
        for year in (2024, 2025, 2026)
        for s in SYMBOLS
        for p in paths(year, s)
    }


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    write(
        root / "protocol.json",
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "plan": PLAN,
            "inputs": inputs(),
            "code": {p: digest(ROOT / p) for p in CODE},
        },
    )
    write(root / "protocol.sha256.json", {"sha256": digest(root / "protocol.json")})


def run(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if (
        protocol["plan"] != PLAN
        or protocol["inputs"] != inputs()
        or protocol["code"] != {p: digest(ROOT / p) for p in CODE}
        or digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
    ):
        raise ValueError("registered experiment changed")
    write(root / "opened.json", {"at": datetime.now(timezone.utc).isoformat()})
    results, states = {}, {}
    for year in (2024, 2025, 2026):
        for s in SYMBOLS:
            key = f"{year}-{s}"
            pp, yy = paths(year, s)
            pred, outcome = dict(np.load(pp)), dict(np.load(yy))
            t, end, y = (outcome[k] for k in ("anchor", "end", "label"))
            if not np.array_equal(t, pred["anchor"]):
                raise ValueError("outcome membership mismatch")
            raw = pred["cross" if year == 2026 else "probability"]
            adaptive, fixed, states[key] = model.replay(raw, y, t, end)
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(f, anchor=t, adaptive=adaptive, fixed=fixed)
            results[key] = {
                name: evaluate(y, p, pred["frequency"], pred["prior_reversal"], t)
                for name, p in (("raw", raw), ("fixed", fixed), ("adaptive", adaptive))
            }
            results[key]["adaptive_vs_raw"] = model.compare(y, adaptive, raw, t)
            results[key]["adaptive_vs_fixed"] = model.compare(y, adaptive, fixed, t)
            results[key]["fixed_vs_raw"] = model.compare(y, fixed, raw, t)
            print(
                key,
                json.dumps(
                    {
                        name: {
                            k: results[key][name].get(k)
                            for k in ("accuracy", "coverage", "gate")
                        }
                        for name in ("raw", "fixed", "adaptive")
                    }
                ),
                flush=True,
            )
    write(root / "states.json", states)
    write(
        root / "results.json",
        {
            "cohorts": results,
            "all_direction_gates": all(r["adaptive"]["gate"] for r in results.values()),
            "all_probability_gates": all(
                r["adaptive_vs_raw"]["probability_gate"] for r in results.values()
            ),
            "all_adaptation_gates": all(
                r["adaptive_vs_fixed"]["probability_gate"] for r in results.values()
            ),
        },
    )
    write(
        root / "manifest.json",
        {p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"register": register, "run": run}[args.action](args.root)
