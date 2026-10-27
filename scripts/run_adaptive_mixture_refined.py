#!/usr/bin/env python3
"""Register one adaptive expert-mixture follow-up with fixed controls."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_adaptive_mixture_refined as mixture  # noqa: E402
from research.timestral_adaptive_direction import compare  # noqa: E402
from research.timestral_universal import SYMBOLS, evaluate  # noqa: E402
from scripts.run_adaptive_direction import SOURCE, paths, inputs as calibration_inputs  # noqa: E402
from scripts.run_timestral_microdirection import ROOT, digest, write  # noqa: E402

DEFAULT = ROOT / "reports/timestral-adaptive-direction-20261027/mixture-refined"
CODE = (
    "research/timestral_adaptive_mixture_refined.py",
    "research/timestral_adaptive_direction.py",
    "research/timestral_universal.py",
    "scripts/run_adaptive_mixture_refined.py",
    "scripts/run_adaptive_direction.py",
    "scripts/run_timestral_microdirection.py",
)
PLAN = {
    "status": "Exploratory follow-up AFTER adaptive calibration failed the combined gates. All underlying expert predictions and historical results already explored; no independent confirmation claimed.",
    "numerical_repair": "The original SLSQP stopped with KKT residual2.98e-6 above2e-6; no cohort metrics were produced. Objective, KKT tolerance and statistical protocol unchanged. Only tighter convergence and more iterations.",
    "motivation": "Calibration preserves directional ordering. A changing mixture can adapt to disagreement between different forecasters, not just soften confidence.",
    "experts": list(mixture.EXPERTS),
    "scope": "Same9 January-July asset/year cohorts, 2024-2026. Frozen cross-linear, shared-trees, prior-reversal, past-frequency probabilities on exactly identical anchors. No refitting underlying models.",
    "training": "Per asset, daily UTC, trailing28 days, half-life7 days weighted by matured outcome end, minimum1000, first7days original linear predictions. Reset every January.",
    "objective": "Weighted multiclass negative logloss + 1e-4*sum(w*log(4*w)); simplex4, eachweight>=1e-6. SLSQP starts uniform each day, maxiter500, ftol1e-15. Fail closed on solver failure or KKT residual>2e-6.",
    "controls": "Raw linear and fixed mixture learned on January8 with identical warmup. Adaptive versus raw, adaptive versus fixed, fixed versus raw on all same anchors.",
    "evaluation": "Exact frozen adaptive-calibration compare and universal.evaluate metrics/gates, unchanged52%selection, 4000 seven-day blocks seed271031. All9 direction and probability gates required; 99.444% intervals apply per comparison/metric across9, not all exploratory research.",
    "boundary": "Only frozen pre2026-08-01 outcomes; no downloads, no new source access, no hyperparameter or threshold sweep.",
}


def inputs():
    extra = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (
            SOURCE / "trees" / f"{year}-{s}-predictions.npz"
            for year in (2024, 2025, 2026)
            for s in SYMBOLS
        )
    }
    return {**calibration_inputs(), **extra}


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    write(
        root / "protocol.json",
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "plan": PLAN,
            "inputs": inputs(),
            "code": {p: digest(ROOT / p) for p in CODE},
            "previous_results_sha256": digest(root.parent / "results.json"),
        },
    )
    write(root / "protocol.sha256.json", {"sha256": digest(root / "protocol.json")})


def run(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if (
        protocol["plan"] != PLAN
        or protocol["inputs"] != inputs()
        or protocol["code"] != {p: digest(ROOT / p) for p in CODE}
        or protocol["previous_results_sha256"] != digest(root.parent / "results.json")
        or digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
    ):
        raise ValueError("registered mixture experiment changed")
    write(root / "opened.json", {"at": datetime.now(timezone.utc).isoformat()})
    results, states = {}, {}
    for year in (2024, 2025, 2026):
        for s in SYMBOLS:
            key = f"{year}-{s}"
            pp, yy = paths(year, s)
            pred, outcome = dict(np.load(pp)), dict(np.load(yy))
            trees = dict(np.load(SOURCE / "trees" / f"{key}-predictions.npz"))
            t, end, y = (outcome[k] for k in ("anchor", "end", "label"))
            if not np.array_equal(t, pred["anchor"]) or not np.array_equal(
                t, trees["anchor"]
            ):
                raise ValueError("expert outcome membership mismatch")
            raw = pred["cross" if year == 2026 else "probability"]
            experts = np.stack(
                (raw, trees["probability"], pred["prior_reversal"], pred["frequency"]),
                axis=1,
            )
            adaptive, fixed, states[key] = mixture.replay(experts, y, t, end)
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(f, anchor=t, adaptive=adaptive, fixed=fixed)
            results[key] = {
                name: evaluate(y, p, pred["frequency"], pred["prior_reversal"], t)
                for name, p in (("raw", raw), ("fixed", fixed), ("adaptive", adaptive))
            }
            for name, p, base in (
                ("adaptive_vs_raw", adaptive, raw),
                ("adaptive_vs_fixed", adaptive, fixed),
                ("fixed_vs_raw", fixed, raw),
            ):
                results[key][name] = compare(y, p, base, t)
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
