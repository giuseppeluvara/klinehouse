#!/usr/bin/env python3
"""Registered zero-fit shrinkage of flat/moving probabilities."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_flat_repair as model  # noqa: E402
from research import timestral_universal as shared  # noqa: E402
from scripts import run_volume_center as previous  # noqa: E402
from scripts.run_futures_flow import ROOT, digest, write, sources, prior_path  # noqa: E402

DEFAULT = ROOT / "reports/timestral-flat-repair-20261028"
CENTER = previous.DEFAULT
UNIVERSAL = ROOT / "reports/timestral-universal-20261027"
CODE = (
    "research/timestral_flat_repair.py",
    "scripts/run_flat_repair.py",
    "scripts/verify_flat_repair.py",
    "tests/test_timestral_flat_repair.py",
    *previous.CODE,
)
PLAN = {
    "purpose": "Test whether fixed shrinkage of flat/moving probabilities repairs the remaining combined-score failure without changing up/down forecasts or fitting additional coefficients.",
    "status": "Retrospective sequential study on already explored Jan-Jul2024/2025/2026 BTC/ETH/SOL. Uniform-center arm chosen after previous study found it better than volume weighting; it remains a research candidate, not newly independent confirmation.",
    "direction": "Reuse frozen uniform_up probabilities exactly, all550362 original anchors and target anchor+60 toanchor+360seconds. No direction fitting, feature changes or threshold search. Prior conditional-direction failures versus common/flow remain unresolved by construction.",
    "primary": "mixed_flat=(frozen_flat+past_frequency_flat)/2. Fixed arithmetic half-weight, no optimized calibration. Frozen flat is cross/probability from universal study. Past frequency is existing asset-specific monthly90day matured counts with Laplace smoothing(count+1)/(n+3), independently reconstructed; no future labels.",
    "secondary": "frequency_flat=past_frequency_flat, replacing the complex flat model entirely. Report all comparisons and original gates, never promote secondary after seeing results. Old frozen flat is control. Recombine P(down)=(1-flat)*(1-up), P(up)=(1-flat)*up.",
    "paired": "For mixed and frequency versus old, compare full3class Brier/logloss and binary flat/moving Brier/logloss on all anchors AND the exact old-model52% selected subset. Same cases within every paired comparison.4000 paired7day moving blocks, seed271032,99.444% intervals (9cohort correction per endpoint). Improvement gate requires full3class Brier/logloss lower bounds>0 and>=5positive monthly Brier gains. Report all18 original evaluations plus9 old controls with unchanged criteria; selection can change under new flat probabilities.",
    "diagnosis": "On each arm's original-threshold selected cases, decompose logloss gain vs historical frequency into flat/moving and conditional-direction components. This is planned descriptive accounting, not an extra candidate or selection criterion.",
    "claims": "repair_success requires all9 mixed original gates. calibration_success requires all9 mixed paired improvement gates on BOTH all and frozen-selected scopes. general_solution requires BOTH and all9 prior uniform_vs_common conditional information gates, which calibration cannot change. Frequency secondary flags reported separately. Do not equate repair_success with universal superiority or prospective success.",
    "limits": "No joint correction across endpoints/scopes/arms/research sequence. No proof of no overfitting from zero new fitted parameters. No costs, deployment, orders or profitability claims. Preserve previous failed protocols and artifacts.",
    "protected": "No market source/outcome from2026-08-01 onward. Existing sources/environments reused with no downloads or installations; frozen experiments, credentials and user files unchanged.",
}


def outcome_path(year, s):
    return (
        UNIVERSAL / f"{s}-outcomes.npz"
        if year == 2026
        else UNIVERSAL / "history" / f"{year}-{s}-outcomes.npz"
    )


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    paths = [CENTER / "results.json"] + [
        p
        for year in (2024, 2025, 2026)
        for p in [
            *sources(year).values(),
            *[prior_path(year, s) for s in shared.SYMBOLS],
            *[outcome_path(year, s) for s in shared.SYMBOLS],
            *[CENTER / f"{year}-{s}-predictions.npz" for s in shared.SYMBOLS],
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
    results = {}
    previous_results = json.loads((CENTER / "results.json").read_text())["cohorts"]
    for year in (2024, 2025, 2026):
        for s in shared.SYMBOLS:
            key = f"{year}-{s}"
            prior = dict(np.load(prior_path(year, s)))
            outcome = dict(np.load(outcome_path(year, s)))
            old = dict(np.load(CENTER / f"{key}-predictions.npz"))
            for d in (prior, outcome):
                if not np.array_equal(d["anchor"], old["anchor"]):
                    raise ValueError("anchors changed")
            if outcome["end"].max() > shared.BOUNDARY:
                raise ValueError("protected outcomes")
            q = old["uniform_up"]
            flat = prior["cross" if year == 2026 else "probability"][:, 1]
            p = model.probabilities(q, flat, prior["frequency"][:, 1])
            y = outcome["label"]
            t = old["anchor"]
            fixed = np.maximum(p["old"][:, 0], p["old"][:, 2]) >= 0.52
            r = dict(
                paired={
                    scope: {
                        name: model.paired(y, p[name], p["old"], t, mask)
                        for name in ("mixed", "frequency")
                    }
                    for scope, mask in (
                        ("all", np.ones(len(y), dtype=bool)),
                        ("frozen_selected", fixed),
                    )
                },
                decomposition={},
                direction_changes=0,
            )
            for name, prob in p.items():
                r[name] = shared.evaluate(
                    y, prob, prior["frequency"], prior["prior_reversal"], t
                )
                select = np.maximum(prob[:, 0], prob[:, 2]) >= 0.52
                r["decomposition"][name] = model.decomposition(
                    y, prob, prior["frequency"], select
                )
            if r["old"] != previous_results[key]["uniform"]:
                raise ValueError("old scores changed")
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(
                    f,
                    anchor=t,
                    mixed_flat=p["mixed"][:, 1],
                    frequency_flat=p["frequency"][:, 1],
                )
            results[key] = r
            print(
                key,
                {a: r[a]["gate"] for a in ("old", "mixed", "frequency")},
                {
                    scope: {a: v["improvement_gate"] for a, v in comparisons.items()}
                    for scope, comparisons in r["paired"].items()
                },
                flush=True,
            )
    repair = {
        a: all(r[a]["gate"] for r in results.values()) for a in ("mixed", "frequency")
    }
    calibration = {
        a: all(
            r["paired"][scope][a]["improvement_gate"]
            for r in results.values()
            for scope in ("all", "frozen_selected")
        )
        for a in ("mixed", "frequency")
    }
    directional = all(
        r["comparisons"]["uniform_vs_common"]["information_gate"]
        for r in previous_results.values()
    )
    write(
        root / "results.json",
        dict(
            cohorts=results,
            repair_success=repair,
            calibration_success=calibration,
            prior_direction_information_success=directional,
            general_solution=repair["mixed"] and calibration["mixed"] and directional,
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
