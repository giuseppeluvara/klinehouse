#!/usr/bin/env python3
"""Preregistered same-expert temporal adaptation comparisons."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_temporal_adaptation as model  # noqa: E402
from research import timestral_logloss_aggregate as fixed  # noqa: E402
from research import timestral_universal as shared  # noqa: E402
from scripts import run_multiscore as previous  # noqa: E402
from scripts import run_logloss_aggregate as base  # noqa: E402
from scripts.run_flat_repair import outcome_path  # noqa: E402
from scripts.run_futures_flow import ROOT, digest, write, sources, prior_path  # noqa: E402

DEFAULT = ROOT / "reports/timestral-temporal-adaptation-20261028"
METHODS = ("share", "discount", "monthly")
CONTROLS = ("cumulative", "uniform_mixed")
CODE = (
    "research/timestral_temporal_adaptation.py",
    "scripts/run_temporal_adaptation.py",
    "scripts/verify_temporal_adaptation.py",
    "tests/test_timestral_temporal_adaptation.py",
    *previous.CODE,
)
PLAN = {
    "purpose": "Isolate temporal adaptation with exactly the same6frozen experts and defensive scoring solver. Seek stable improvement in direction, full Brier and logloss in all9asset/year cohorts.",
    "status": "Sequential retrospective research on already explored Jan-Jul2024/2025/2026 BTC/ETH/SOL. Protocol before newarm scores, not untouched confirmation. No newdata,features,fit or expertpoolchanges.",
    "experts": "Exactly common_old/common_mixed/uniform_old/uniform_mixed/equal/logloss_adaptive from frozen multiscore study. Same550362anchors, targetanchor+60to+360sec, allcoldstarts retained. Frozen cumulative_multiscore is exactsameexpert matchedpool control. uniform_mixed is secondstrongstatic control.",
    "common": "Threearms reuse identical frozen multiclass defensive forecast solver eta1 for natural logloss and full3class Brier,12equal evaluator priors perlane,2delayed lanes perasset/year. i uses onlymature feedbackj=i-2. Increment is ownissued loss minus each expert loss separately foreachscore; no outcome-dependent exceptions. Peroutcome potential<=1+1e-10 and numericalslack saved.",
    "share_primary": "After adding mature delta_j, mix weight with equalexpert distribution WITHIN each scoringrule, preserving scoremass: w_new=(1-alpha)w+alpha*rowsum(w)/6, alpha=1-exp(-dt/(30days)), dt=anchor_i-anchor_j. 30days is refresh timeconstant, actual switch probability5alpha/6. Logspace stable implementation.",
    "discount_secondary": "Beforeforecasti set logweights=rho*(logweights+delta_j),rho=2^(-dt/(30days)). Exacthalf-life30days fromforecastanchor age, includingnewlymatured increment. No optimizeddecay.",
    "monthly_secondary": "At firstforecastofnewUTC anchormonth independently foreachlane resetall12logs0 and skip pendingj if anchor_j inoldmonth. Oldforecaststillscored. Otherwise adddelta_j normally. Bothfirstmonthlyforecasts equalprior. No resettingbasedon observedperformance.",
    "selection": "Share is primary chosen beforeoutputs; discount/monthly are fixedsecondary alternatives. No interval/window/half-life sweep, winnerpromotion, cohort-specificmethod, or removaloffailingcohorts.",
    "evaluation": "For eacharm/cohort originalgateunchanged. Paired3class Brier/logloss against cumulative anduniform_mixed onallanchors andfixeduniform_mixed52percentselectedsubset. Conditionaldirection onallmovingoutcomes: binarysingle-square Brier, conditional logloss, andaccuracy(tiesupq>=0.5) againstsame2controls. Accuracygainisownminuscontrol, scoregainscontrolminusown. Samecasesforeachpair.",
    "uncertainty": "12000paired7daymovingblocksseed271032; quantiles1/1080and1079/1080 (99.8148148percent,Bonferroni27arm/cohort comparisonsperfixedendpoint/control/scope). No jointcorrectionacrossendpoints,controls,scopes,sequentialhistory. No bootstraprefitting. Originalgates retainold98.333percentintervalsand thresholds.",
    "success": "Perpairedcomparison ALLmetricgainlowerbounds>0 AND>=5positive monthsgains forEVERYmetric. Perarm stable_improvement requiresall9originalpasses,all36fullscorecomparisongates(9x2scopesx2controls),all18directiongates(9x2controls,all3metrics). Primarysuccessonlyshare. No redefinitionofoldergeneral_solution flags. Reportallscores,countsandfailedarms.",
    "certificates": "VerifylocalpotentialconstraintsforALL3possibleoutcomes andcausalweight recurrenceforeacharm. DO NOT apply oldconstantfullcohort2ln12bound toforgetting/sharing/resetting. No globalregretcertificateclaimed inthisstudy; empiricalfullperiodperformanceisscoredwithoutdiscounting.",
    "protected": "No market source oroutcomefrom2026-08-01 onward. Preservefrozenprotocols/code/data/models anduserfiles/credentials. Existingenvandsmallarraysonly; nodownloads,installations,orders ordeployment.",
    "references": [
        "https://mwarmuth.bitbucket.io/pubs/J39.pdf",
        "https://arxiv.org/pdf/1005.1918",
    ],
}


def paired_metrics(y, p, control, t, mask, direction=False):
    if direction:
        names = ("conditional_brier", "conditional_logloss", "accuracy")
        mask = mask & (y != 1)
        up = y == 2

        def values(prob):
            q = prob[:, 2] / (1 - prob[:, 1])
            return np.column_stack(
                (
                    (q - up) ** 2,
                    -np.log(np.where(up, q, 1 - q)),
                    ((q >= 0.5) == up).astype(float),
                )
            )
    else:
        names = ("brier", "logloss")
        target = np.eye(3)[y]

        def values(prob):
            return np.column_stack(
                (np.square(prob - target).sum(1), -np.log(prob[np.arange(len(y)), y]))
            )

    own, old = values(p), values(control)
    gains = old - own
    if direction:
        gains[:, 2] *= -1
    days, inv = np.unique(t // 86400, return_inverse=True)
    if np.any(np.diff(days) != 1) or not mask.any():
        raise ValueError("complete days and nonempty paired membership required")
    counts = np.bincount(inv, weights=mask)
    totals = np.column_stack(
        [np.bincount(inv, weights=mask * gains[:, j]) for j in range(len(names))]
    )
    starts = np.random.default_rng(271032).integers(
        0, len(days) - 6, (12000, (len(days) + 6) // 7)
    )
    ids = (starts[:, :, None] + np.arange(7)).reshape(12000, -1)[:, : len(days)]
    denom = counts[ids].sum(1)
    if np.any(denom == 0):
        raise ValueError("empty bootstrap sample")
    boot = totals[ids].sum(1) / denom[:, None]
    bounds = np.quantile(boot, [1 / 1080, 1079 / 1080], axis=0)
    months = t.astype("datetime64[s]").astype("datetime64[M]")
    monthly = {}
    for month in np.unique(months):
        part = mask & (months == month)
        if not part.any():
            raise ValueError("empty monthly comparison")
        monthly[str(month)] = {
            name: float(gains[part, j].mean()) for j, name in enumerate(names)
        }
    positive = {name: sum(v[name] > 0 for v in monthly.values()) for name in names}
    return dict(
        n=int(mask.sum()),
        own={name: float(own[mask, j].mean()) for j, name in enumerate(names)},
        control={name: float(old[mask, j].mean()) for j, name in enumerate(names)},
        gain={name: float(gains[mask, j].mean()) for j, name in enumerate(names)},
        interval={name: bounds[:, j].tolist() for j, name in enumerate(names)},
        monthly=monthly,
        positive_months=positive,
        improvement_gate=bool(
            np.all(bounds[0] > 0) and all(v >= 5 for v in positive.values())
        ),
    )


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    paths = [
        p
        for year in (2024, 2025, 2026)
        for p in [
            *sources(year).values(),
            *[prior_path(year, s) for s in shared.SYMBOLS],
            *[outcome_path(year, s) for s in shared.SYMBOLS],
            *[
                folder / f"{year}-{s}-predictions.npz"
                for folder in (base.CENTER, base.COMMON, base.DEFAULT, previous.DEFAULT)
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
            outcome = dict(np.load(outcome_path(year, s)))
            common = dict(np.load(base.COMMON / f"{key}-predictions.npz"))
            center = dict(np.load(base.CENTER / f"{key}-predictions.npz"))
            old = dict(np.load(base.DEFAULT / f"{key}-predictions.npz"))
            cumulative = dict(np.load(previous.DEFAULT / f"{key}-predictions.npz"))
            t, y, end = outcome["anchor"], outcome["label"], outcome["end"]
            for d in (prior, common, center, old, cumulative):
                if not np.array_equal(t, d["anchor"]):
                    raise ValueError("anchors changed")
            four = fixed.experts(
                common["adaptive_up"],
                center["uniform_up"],
                prior["cross" if year == 2026 else "probability"][:, 1],
                prior["frequency"][:, 1],
            )
            experts = np.concatenate(
                (four, four.mean(1)[:, None], old["adaptive"][:, None]), axis=1
            )
            controls = {
                "cumulative": cumulative["defensive"],
                "uniform_mixed": four[:, 3],
            }
            selected = np.maximum(four[:, 3, 0], four[:, 3, 2]) >= 0.52
            saved = {"anchor": t}
            results[key] = {}
            evidence[key] = {}
            for method in METHODS:
                p, slack, diag = model.replay(experts, y, t, end, method)
                saved[method] = p
                saved[method + "_log_excess"] = slack
                evidence[key][method] = dict(
                    maximum_potential_ratio=diag["maximum_potential_ratio"],
                    numeric_slack=float(slack.sum()),
                    local_valid=bool(diag["maximum_potential_ratio"] <= 1 + 1e-10),
                    diagnostics=diag,
                )
                if not evidence[key][method]["local_valid"]:
                    raise ValueError("local potential failed")
                r = dict(
                    original=shared.evaluate(
                        y, p, prior["frequency"], prior["prior_reversal"], t
                    ),
                    paired={
                        scope: {
                            c: paired_metrics(y, p, controls[c], t, mask)
                            for c in CONTROLS
                        }
                        for scope, mask in (
                            ("all", np.ones(len(y), dtype=bool)),
                            ("frozen_selected", selected),
                        )
                    },
                    direction={
                        c: paired_metrics(
                            y, p, controls[c], t, np.ones(len(y), dtype=bool), True
                        )
                        for c in CONTROLS
                    },
                )
                results[key][method] = r
                print(
                    key,
                    method,
                    "original",
                    r["original"]["gate"],
                    "full",
                    {c: r["paired"]["all"][c]["improvement_gate"] for c in CONTROLS},
                    "direction",
                    {c: r["direction"][c]["improvement_gate"] for c in CONTROLS},
                    flush=True,
                )
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(f, **saved)
    summary = {}
    for method in METHODS:
        cases = [v[method] for v in results.values()]
        orig = sum(r["original"]["gate"] for r in cases)
        paired = sum(
            r["paired"][scope][c]["improvement_gate"]
            for r in cases
            for scope in ("all", "frozen_selected")
            for c in CONTROLS
        )
        direction = sum(
            r["direction"][c]["improvement_gate"] for r in cases for c in CONTROLS
        )
        summary[method] = dict(
            original_passes=orig,
            paired_passes=paired,
            direction_passes=direction,
            stable_improvement=orig == 9 and paired == 36 and direction == 18,
        )
    write(
        root / "results.json",
        dict(
            cohorts=results,
            summary=summary,
            primary_method="share",
            primary_stable_improvement=summary["share"]["stable_improvement"],
        ),
    )
    write(root / "evidence.json", evidence)
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
