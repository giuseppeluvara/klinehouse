#!/usr/bin/env python3
"""Registered simultaneous log-loss/Brier defensive forecast experiment."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_multiscore as model  # noqa: E402
from research import timestral_logloss_aggregate as fixed  # noqa: E402
from research import timestral_flat_repair as repair  # noqa: E402
from research import timestral_universal as shared  # noqa: E402
from research.timestral_futures_flow import comparison  # noqa: E402
from scripts import run_logloss_aggregate as previous  # noqa: E402
from scripts.run_futures_flow import ROOT, digest, write, sources, prior_path  # noqa: E402

DEFAULT = ROOT / "reports/timestral-multiscore-20261028"
CONTROLS = (*fixed.NAMES, "equal", "logloss_adaptive")
CODE = (
    "research/timestral_multiscore.py",
    "scripts/run_multiscore.py",
    "scripts/verify_multiscore.py",
    "tests/test_timestral_multiscore.py",
    *previous.CODE,
)
PLAN = {
    "purpose": "Test simultaneous protection of full three-class log loss and Brier after log-loss-only aggregation regressed on Brier. Distinguish a multi-score regret guarantee from empirical strict superiority.",
    "status": "Sequential retrospective research. All Jan-Jul2024/2025/2026 outcomes previously explored. Expert pool selected from those prior findings. No untouched confirmation.",
    "experts": "Six ordered causal forecasts: common_old,common_mixed,uniform_old,uniform_mixed from preceding Cartesian design; equal=static arithmetic mean of firstfour; logloss_adaptive=exact preceding frozen two-lane aggregate. No new information, source, model fit or per-asset exception.",
    "primary": "Two independent delayed lanes per asset/year, reset12 equal evaluator priors (6experts x2scores), eta1 for both natural logloss and FULL multiclass sum-squared Brier. Before forecasti update lanei%2 using maturei-2 realized outcome, adding learner loss minus each expert loss separately for bothscores. Learner loss CANNOT be discarded across different scores. All initial forecasts retained; no forgetting/window/tuned learningrate.",
    "prediction": "Normalize12 logweights. With A_y=sum log-evaluator-weight*expert_p_y and B_y=sum Brier-evaluator-weight*exp(-expert_Brier_y), solve for interior simplex p such that F_y=A_y/p_y+B_y*exp(sum(p^2)+1-2p_y)<=1 for ALL3possible outcomes. Deterministic Newton plus numerical fallback as frozen code, no scoring-based solver selection. Abort if maxF>1+1e-10. Save nonnegative slack=max(0,log(maxF)) eachround.",
    "certificate": "For each full cohort and every prefix, BOTH cumulative losses no more than best of6 expert losses plus2ln12+sum numerical slack. Under exact arithmetic slack0; safe eta1 established for full Brier by Hoeffding bound and logloss identity. Also independently verify ALL3outcome potential constraints eachforecast, strictprefix weight reconstruction, and lastlabels only used for postforecast certificate. No old direction/flat factorization guarantee.",
    "limits": "Relative approximate guarantees only: no dominance of all models, arbitrary outcomes, all metrics, switching experts, best convex mixture, selected subsets, months, calibration, accuracy or profits. Empirical multi-score improvement is a separate test. No claim that repeated explored data can establish universal future superiority.",
    "evaluation": "Same550362 anchors and target anchor+60 to+360sec. Seven original evaluations(cohort primary+6controls). Paired full3class and flat Brier/logloss vsall6controls, on allcases and exact frozen uniform_mixed52percent selectedsubset. Conditional direction vscommon anduniform plusfrequency. Everycohort retained.",
    "success": "all_certificates=9 verified bounds (technical property). all_original_gates=9 primary passes. all_paired_gates=9x6controls x2scopes improvementgates. all_direction_information_gates=9x2controls informationgates. general_solution is conjunction of last3empirical flags, never replaced by bound success.",
    "uncertainty": "Reuse4000 paired7day blocks seed271032;99.444percent intervals ninecohort correction perendpoint. Paired gate bothBrier andLLLCB>0 and>=5positive Brier months. Conditional and original gates unchanged. No joint correction across endpoints,controls,scopes or sequential studies; no bootstrap refitting.",
    "protected": "No market source or outcome from2026-08-01 onward. Preserve oldcode,protocols,models,data,credentials,userfiles. Reuse existing smallarrays/env; no downloads,installations,orders or deployment.",
    "references": [
        "https://arxiv.org/pdf/1003.2218",
        "https://jmlr.org/papers/volume10/vovk09a/vovk09a.pdf",
    ],
}


def certificate(p, y, prediction, slack):
    idx = np.arange(len(y))
    target = np.eye(3)[y]
    expert_losses = {
        "logloss": -np.log(p[idx[:, None], np.arange(p.shape[1])[None, :], y[:, None]]),
        "brier": np.square(p - target[:, None]).sum(2),
    }
    own_losses = {
        "logloss": -np.log(prediction[idx, y]),
        "brier": np.square(prediction - target).sum(1),
    }
    base = float(2 * np.log(2 * p.shape[1]))
    allowance = base + np.cumsum(slack)
    worst = max(
        float(
            np.max(
                np.cumsum(own_losses[m][:, None] - expert_losses[m], axis=0)
                - allowance[:, None]
            )
        )
        for m in own_losses
    )
    totals = {m: expert_losses[m].sum(0).tolist() for m in own_losses}
    own = {m: float(own_losses[m].sum()) for m in own_losses}
    return dict(
        expert_totals=totals,
        learner_totals=own,
        regrets={m: own[m] - min(totals[m]) for m in own},
        base_bound=base,
        numeric_slack=float(slack.sum()),
        effective_bound=base + float(slack.sum()),
        maximum_prefix_excess=worst,
        bound_pass=bool(worst <= 1e-8),
    )


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    paths = [
        p
        for year in (2024, 2025, 2026)
        for p in [
            *sources(year).values(),
            *[prior_path(year, s) for s in shared.SYMBOLS],
            *[previous.previous.outcome_path(year, s) for s in shared.SYMBOLS],
            *[
                folder / f"{year}-{s}-predictions.npz"
                for folder in (previous.CENTER, previous.COMMON, previous.DEFAULT)
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
            outcome = dict(np.load(previous.previous.outcome_path(year, s)))
            center = dict(np.load(previous.CENTER / f"{key}-predictions.npz"))
            common = dict(np.load(previous.COMMON / f"{key}-predictions.npz"))
            old_aggregate = dict(np.load(previous.DEFAULT / f"{key}-predictions.npz"))
            t, y, end = outcome["anchor"], outcome["label"], outcome["end"]
            for d in (prior, center, common, old_aggregate):
                if not np.array_equal(t, d["anchor"]):
                    raise ValueError("anchors changed")
            oldflat = prior["cross" if year == 2026 else "probability"][:, 1]
            four = fixed.experts(
                common["adaptive_up"],
                center["uniform_up"],
                oldflat,
                prior["frequency"][:, 1],
            )
            experts = np.concatenate(
                (four, four.mean(1)[:, None], old_aggregate["adaptive"][:, None]),
                axis=1,
            )
            prediction, slack, diagnostics = model.replay(experts, y, t, end)
            cert = certificate(experts, y, prediction, slack)
            if not cert["bound_pass"]:
                raise ValueError("multi-score certificate failed")
            evidence[key] = dict(certificate=cert, solver=diagnostics)
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(f, anchor=t, defensive=prediction, log_excess=slack)
            p = {name: experts[:, j] for j, name in enumerate(CONTROLS)}
            p["defensive"] = prediction
            selected = (
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
                    c: repair.paired(y, prediction, p[c], t, mask) for c in CONTROLS
                }
                for scope, mask in (
                    ("all", np.ones(len(y), dtype=bool)),
                    ("frozen_selected", selected),
                )
            }
            aq = prediction[:, 2] / (1 - prediction[:, 1])
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
                r["defensive"]["gate"],
                "regrets",
                cert["regrets"],
                "all-case improvements",
                {c: r["paired"]["all"][c]["improvement_gate"] for c in CONTROLS},
                flush=True,
            )
    original = all(r["defensive"]["gate"] for r in results.values())
    paired = all(
        r["paired"][scope][c]["improvement_gate"]
        for r in results.values()
        for scope in ("all", "frozen_selected")
        for c in CONTROLS
    )
    direction = all(
        r["conditional"][c]["information_gate"]
        for r in results.values()
        for c in ("common", "uniform")
    )
    write(root / "evidence.json", evidence)
    write(
        root / "results.json",
        dict(
            cohorts=results,
            all_certificates=all(
                e["certificate"]["bound_pass"] for e in evidence.values()
            ),
            all_original_gates=original,
            all_paired_gates=paired,
            all_direction_information_gates=direction,
            general_solution=original and paired and direction,
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
