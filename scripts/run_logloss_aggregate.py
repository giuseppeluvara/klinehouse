#!/usr/bin/env python3
"""Registered delayed aggregation of four frozen probability experts."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_logloss_aggregate as model  # noqa: E402
from research import timestral_flat_repair as repair  # noqa: E402
from research import timestral_universal as shared  # noqa: E402
from research.timestral_futures_flow import comparison  # noqa: E402
from scripts import run_flat_repair as previous  # noqa: E402
from scripts.run_futures_flow import ROOT, digest, write, sources, prior_path  # noqa: E402

DEFAULT = ROOT / "reports/timestral-logloss-aggregate-20261028"
CENTER = previous.CENTER
COMMON = ROOT / "reports/timestral-uncertain-market-20261027"
CONTROLS = (*model.NAMES, "equal")
CODE = (
    "research/timestral_logloss_aggregate.py",
    "scripts/run_logloss_aggregate.py",
    "scripts/verify_logloss_aggregate.py",
    "tests/test_timestral_logloss_aggregate.py",
    *previous.CODE,
)
PLAN = {
    "purpose": "Test whether causal weighting of fixed direction and flat-probability alternatives reduces the opposing cohort regressions; distinguish observed universal dominance from a valid finite-sequence log-loss regret bound.",
    "status": "Retrospective sequential research on already explored Jan-Jul2024/2025/2026 BTC/ETH/SOL. Frozen experts chosen using previous results. New protocol before aggregate scores, not untouched confirmation.",
    "experts": "Ordered Cartesian four: common_old,common_mixed,uniform_old,uniform_mixed. common=uncertain-market adaptive_up,uniform=volume-center uniform_up,oldflat=universal cross/probability,mixedflat=(oldflat+existing90day frequencyflat)/2. All coefficients/probabilities reused exactly. No new source data or optimization.",
    "primary": "For each asset/year reset TWO separate log-weight vectors to equal1/4 priors. Forecast i uses lane i%2. Before it, update that lane by adding log(expert probability of label i-2), only after verifying end[i-2]<=anchor[i]. Normalize with log-space exponentials, arithmetic weighted mixture of four3class probabilities. End=anchor+360sec and spacing>=300sec. All coldstart forecasts included; ignore other-lane feedback even when available. No learning-rate, lookback,forgetting,prior or asset-exception sweep.",
    "complexity": "Same deterministic update for every asset/year, with separate causal states. This is adaptive extra flexibility versus fixed mixture, not zero adaptive complexity. No independence assumption or expert truth-probability interpretation.",
    "certificate": "For full sequence, cumulative aggregate logloss equals sum over lanes of negative log(mean_k exp(-expert cumulative lane loss)). Excess against best fixed one of4experts <=2ln4; directional and flat/moving component excess versus their2alternatives <=2ln2 each because Cartesian posterior factorizes. No guarantee of strict dominance, Brier,accuracy,selected-subset performance, switching experts,best convex mixture or profits. Evaluate certificate from all labels only after forecasting; last labels not fed into earlier forecasts.",
    "evaluation": "Same550362 anchors and five-minute target from anchor+60 to+360sec. Original direction gates for adaptive,4experts and equal-weight static ablation. Full3class and flat proper-score paired comparisons adaptive versus each of5controls on all anchors and exact frozen uniform_mixed52% selected subset. Conditional direction comparisons against common and uniform (also frequency). All cohorts retained.",
    "success": "all_original_gates=all9 adaptive original gates. all_paired_gates=all9 x5controls x2scopes paired improvement gates. all_direction_information_gates=all9 x2 conditional information gates. general_solution requires all3; do not substitute bounded regret for these empirical requirements.",
    "uncertainty": "Reuse paired4000seven-day blocks seed271032,99.444% intervals9cohort correction per endpoint; paired gate Brier/logloss LCB>0 and>=5positive months. Conditional old gates unchanged. Original52% thresholds and98.333% intervals unchanged. No joint correction across endpoints,controls,scopes or sequential research; no bootstrap refitting.",
    "protected": "No market source or outcome from2026-08-01 onward. Preserve old bundles, models, data, credentials and user files. Reuse existing small arrays and environment; no downloads,installations,orders or deployment.",
    "references": [
        "https://proceedings.mlr.press/v91/korotin18a/korotin18a.pdf",
        "https://proceedings.mlr.press/v76/orseau17a/orseau17a.pdf",
    ],
}


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    paths = [
        p
        for year in (2024, 2025, 2026)
        for p in [
            *sources(year).values(),
            *[prior_path(year, s) for s in shared.SYMBOLS],
            *[previous.outcome_path(year, s) for s in shared.SYMBOLS],
            *[
                folder / f"{year}-{s}-predictions.npz"
                for folder in (CENTER, COMMON)
                for s in shared.SYMBOLS
            ],
        ]
    ]
    write(
        root / "protocol.json",
        dict(
            at=datetime.now(timezone.utc).isoformat(),
            plan=PLAN,
            code={p: digest(ROOT / p) for p in CODE},
            inputs={str(p.relative_to(ROOT)): digest(p) for p in paths},
        ),
    )
    write(root / "protocol.sha256.json", dict(sha256=digest(root / "protocol.json")))


def run(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if (
        protocol["plan"] != PLAN
        or digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
        or any(
            digest(ROOT / p) != h
            for p, h in {**protocol["code"], **protocol["inputs"]}.items()
        )
    ):
        raise ValueError("frozen experiment changed")
    write(root / "opened.json", dict(at=datetime.now(timezone.utc).isoformat()))
    results, evidence = {}, {}
    for year in (2024, 2025, 2026):
        for s in shared.SYMBOLS:
            key = f"{year}-{s}"
            prior = dict(np.load(prior_path(year, s)))
            outcome = dict(np.load(previous.outcome_path(year, s)))
            center = dict(np.load(CENTER / f"{key}-predictions.npz"))
            common = dict(np.load(COMMON / f"{key}-predictions.npz"))
            t, y, end = outcome["anchor"], outcome["label"], outcome["end"]
            for d in (prior, center, common):
                if not np.array_equal(t, d["anchor"]):
                    raise ValueError("anchors changed")
            oldflat = prior["cross" if year == 2026 else "probability"][:, 1]
            experts = model.experts(
                common["adaptive_up"],
                center["uniform_up"],
                oldflat,
                prior["frequency"][:, 1],
            )
            adaptive, weights = model.replay(experts, y, t, end)
            e = model.evidence(experts, y, adaptive)
            if (
                e["regret"] > e["regret_bound"] + 1e-8
                or max(e["direction_regret"], e["flat_regret"])
                > e["component_bound"] + 1e-8
                or abs(e["telescoping_gap"]) > 1e-7
            ):
                raise ValueError("logloss certificate failed")
            evidence[key] = e
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(f, anchor=t, adaptive=adaptive, weights=weights)
            p = {name: experts[:, i] for i, name in enumerate(model.NAMES)}
            p.update(equal=experts.mean(1), adaptive=adaptive)
            fixed = (
                np.maximum(p["uniform_mixed"][:, 0], p["uniform_mixed"][:, 2]) >= 0.52
            )
            r = {
                name: shared.evaluate(
                    y, prob, prior["frequency"], prior["prior_reversal"], t
                )
                for name, prob in p.items()
            }
            r["paired"] = {
                scope: {
                    control: repair.paired(y, adaptive, p[control], t, mask)
                    for control in CONTROLS
                }
                for scope, mask in (
                    ("all", np.ones(len(y), dtype=bool)),
                    ("frozen_selected", fixed),
                )
            }
            aq = adaptive[:, 2] / (1 - adaptive[:, 1])
            fq = prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1])
            r["conditional"] = {
                name: comparison(y, aq, q, fq, t)
                for name, q in (
                    ("common", common["adaptive_up"]),
                    ("uniform", center["uniform_up"]),
                )
            }
            results[key] = r
            print(
                key,
                "original",
                r["adaptive"]["gate"],
                "all-case improvements",
                {c: r["paired"]["all"][c]["improvement_gate"] for c in CONTROLS},
                "regret",
                e["regret"],
                flush=True,
            )
    original = all(r["adaptive"]["gate"] for r in results.values())
    paired = all(
        r["paired"][scope][c]["improvement_gate"]
        for r in results.values()
        for scope in ("all", "frozen_selected")
        for c in CONTROLS
    )
    directional = all(
        r["conditional"][c]["information_gate"]
        for r in results.values()
        for c in ("common", "uniform")
    )
    write(root / "evidence.json", evidence)
    write(
        root / "results.json",
        dict(
            cohorts=results,
            all_original_gates=original,
            all_paired_gates=paired,
            all_direction_information_gates=directional,
            general_solution=original and paired and directional,
        ),
    )
    write(
        root / "manifest.json",
        {p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    (register if args.action == "register" else run)(args.root)
