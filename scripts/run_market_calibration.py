#!/usr/bin/env python3
"""Registered prequential calibration of the frozen common-hour signal."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_market_calibration as model  # noqa: E402
from research import timestral_market_reversal as base  # noqa: E402
from research import timestral_universal as shared  # noqa: E402
from research.timestral_futures_flow import combine, comparison  # noqa: E402
from scripts import run_market_reversal as previous  # noqa: E402
from scripts.run_futures_flow import (
    ROOT,
    digest,
    write,
    epoch,
    sources,
    prior_path,
    examples,
)  # noqa: E402

DEFAULT = ROOT / "reports/timestral-market-calibration-20261027"
MARKET = previous.DEFAULT
FUTURES = previous.FUTURES
ARMS = ("raw", "adjusted")
CONTROLS = ("base", "unshrunk", "raw", "flow", "neutral")
CODE = (
    "research/timestral_market_calibration.py",
    "scripts/run_market_calibration.py",
    "scripts/verify_market_calibration.py",
    "tests/test_timestral_market_calibration.py",
    *previous.CODE,
)
PLAN = {
    "purpose": "Correct common-hour underconfidence using one prequential multiplicative calibration factor, with an uncertainty-based shrinkage ablation; no new directional feature or threshold.",
    "status": "Exploratory Jan-Jul2024/2025/2026 BTC/ETH/SOL on previously explored outcomes. New protocol before new scores, not untouched/prospective validation.",
    "base": "Exact frozen market60 penalty.25 coefficients reused Jan-Jul. Generate Nov/Dec preceding-year warmup fits by identical90day algorithm on available frozen history startingOctober; these warmup windows have about31/61days, explicitly shorter than90. Generate all Nov-Jul predictions only with models whose matured training outcomes end by the forecast month start. No downloads.",
    "calibration": "Monthly last56days matured nonflat labels and out-of-training base logits, pooled3assets, equal total weight per UTC day, minimum42 contiguous days. Include all finite issued forecasts, independently of futures availability at those forecast anchors. One factor a in[0,8] minimizes day-weighted logloss(sigmoid(a*base_logit)), no intercept. Brentq derivative root xtol1e-12 or boundary. All-zero logits fallback identity.21 calibration fits plus6 new warmup base fits;21 base fits reused.",
    "uncertainty": "For raw factor compute daily mean logistic score; center daily scores, Bartlett HAC lags1..6 (weights1-lag/7), variance of mean=nonnegative HAC sum/n_days^2. Divide by squared mean Hessian (floor1e-12). Shrink raw factor toward identity1 with weight max(0,1-variance/max((raw-1)^2,1e-24)); if Hessian<=1e-12 use identity. This is a fixed heuristic, not a guarantee against overfitting; no variance/window tuning after scores.",
    "arms": "raw calibration; primary adjusted uncertainty-shrunk calibration. Both preserve base directional ordering for positive factor; zero produces neutral. Use unchanged frozen flat probabilities for original gates. Same550362 anchors and five-minute spot outcomes anchor+60sec to anchor+360sec.",
    "comparisons": "Adjusted versus frozen base, frozen unshrunk, raw calibration, frozen flow and neutral.5. Each conditional comparison also tests historical frequency. Report original gates for both new arms. No selection of favorable cohorts or changing52% threshold.",
    "primary": "Calibration success requires all9 conditional gates vs base AND unshrunk AND neutral; general solution additionally requires all9 vs flow AND all9 original adjusted direction gates. Gate thresholds unchanged: accuracy lower99.444%>50%, Brier and logloss gain lower99.444%>0 versus named control and frequency,>=5 positive monthly Brier gains,>=100 each direction. Raw comparison is diagnostic.",
    "scoring": "Frozen4000 paired7day bootstrap seed271032,9cohort correction per endpoint, not joint across endpoints/comparisons/research history. No refitting bootstrap. Nonflat diagnostics do not identify future flats. Report mean/q95 confidence, gap, ties, fixed confidence bins[.5,.51,.52,.54,.58,.66,1], ECE and direction changes from base; descriptive not a calibration significance test.",
    "protected": "No market source/outcome from2026-08-01 onward. Older bundles, models and user files unchanged. All three years are January-July; no trading orders, costs or profitability/prospective claims.",
}


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    paths = [MARKET / "fits.json"] + [
        p
        for year in (2024, 2025, 2026)
        for p in [
            *sources(year).values(),
            *[prior_path(year, s) for s in shared.SYMBOLS],
            *[
                FUTURES / f"{year}-{s}-{suffix}.npz"
                for s in shared.SYMBOLS
                for suffix in ("futures", "predictions")
            ],
            *[MARKET / f"{year}-{s}-predictions.npz" for s in shared.SYMBOLS],
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


def validate(root):
    p = json.loads((root / "protocol.json").read_text())
    if (
        p["plan"] != PLAN
        or digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
        or any(digest(ROOT / k) != h for k, h in {**p["code"], **p["inputs"]}.items())
    ):
        raise ValueError("frozen calibration experiment changed")


def ids_for(t, mask):
    return np.concatenate(
        [np.column_stack((np.full(mask.sum(), a), t[mask])) for a in range(3)]
    ).astype("<i8")


def add_hashes(state, ids, labels):
    state["training_ids_sha256"] = hashlib.sha256(
        ids.astype("<i8").tobytes()
    ).hexdigest()
    state["training_labels_sha256"] = hashlib.sha256(
        labels.astype("<i8").tobytes()
    ).hexdigest()


def run(root):
    validate(root)
    write(root / "opened.json", dict(at=datetime.now(timezone.utc).isoformat()))
    old = json.loads((MARKET / "fits.json").read_text())
    base_fits, fits, results = {}, {}, {}
    for year in (2024, 2025, 2026):
        sample = examples(FUTURES, year)
        t, end, y = sample["anchor"], sample["end"], sample["y"]
        z = base.signal(sample["x"], "market60")
        logits, issued = np.full_like(z, np.nan), np.zeros(len(t), dtype=np.int64)
        for yy, month in [(year - 1, 11), (year - 1, 12)] + [
            (year, m) for m in range(1, 8)
        ]:
            cutoff = epoch(yy, month)
            train = (t >= cutoff - 90 * 86400) & (end <= cutoff) & sample["available"]
            moving = y[:, train].reshape(-1) != 1
            labels = (y[:, train].reshape(-1)[moving] == 2).astype(int)
            if yy == year:
                state = old[f"{year}-{month}-market60"].copy()
            else:
                state = base.fit(
                    z[:, train].reshape(-1)[moving],
                    labels,
                    np.tile(t[train], 3)[moving],
                    np.tile(end[train], 3)[moving],
                    cutoff,
                    0.25,
                )
                add_hashes(state, ids_for(t, train)[moving], labels)
            base_fits[f"{year}-{yy}-{month}"] = state
            test = (t >= cutoff) & (t < epoch(yy + (month == 12), month % 12 + 1))
            logits[:, test], issued[test] = state["beta"] * z[:, test], cutoff
        predictions = {name: [] for name in ARMS}
        indices = []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            cal = (
                (t >= cutoff - 56 * 86400)
                & (end <= cutoff)
                & np.isfinite(logits).all(axis=0)
            )
            moving = y[:, cal].reshape(-1) != 1
            labels = (y[:, cal].reshape(-1)[moving] == 2).astype(int)
            state = model.fit(
                logits[:, cal].reshape(-1)[moving],
                labels,
                np.tile(t[cal], 3)[moving],
                np.tile(end[cal], 3)[moving],
                np.tile(issued[cal], 3)[moving],
                cutoff,
            )
            add_hashes(state, ids_for(t, cal)[moving], labels)
            fits[f"{year}-{month}"] = state
            test = (t >= cutoff) & (t < epoch(year, month + 1))
            for name in ARMS:
                predictions[name].append(
                    np.stack(
                        [
                            model.predict(
                                state,
                                logits[a, test],
                                t[test],
                                adjusted=name == "adjusted",
                            )
                            for a in range(3)
                        ]
                    )
                )
            indices.append(np.flatnonzero(test))
            print("calibrated", year, month, flush=True)
        indices = np.concatenate(indices)
        q = {
            name: np.concatenate(pieces, axis=1) for name, pieces in predictions.items()
        }
        for a, s in enumerate(shared.SYMBOLS):
            key = f"{year}-{s}"
            prior = dict(np.load(prior_path(year, s)))
            market = dict(np.load(MARKET / f"{key}-predictions.npz"))
            flow = dict(np.load(FUTURES / f"{key}-predictions.npz"))
            for d in (prior, market, flow):
                if not np.array_equal(d["anchor"], t[indices]):
                    raise ValueError("evaluation anchors changed")
            np.testing.assert_allclose(
                1 / (1 + np.exp(-logits[a, indices])),
                market["market60_up"],
                atol=1e-12,
                rtol=1e-12,
            )
            controls = dict(
                base=market["market60_up"],
                unshrunk=market["market60_unshrunk_up"],
                flow=flow["flow_up"],
                neutral=np.full(len(indices), 0.5),
                **{name: q[name][a] for name in ARMS},
            )
            labels, anchors = y[a, indices], t[indices]
            flat = prior["cross" if year == 2026 else "probability"][:, 1]
            frequency = prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1])
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(
                    f, anchor=anchors, **{name + "_up": q[name][a] for name in ARMS}
                )
            result = dict(
                comparisons={
                    f"adjusted_vs_{c}": comparison(
                        labels, q["adjusted"][a], controls[c], frequency, anchors
                    )
                    for c in CONTROLS
                },
                calibration={
                    name: model.reliability(labels, p) for name, p in controls.items()
                },
                direction_changes={
                    name: int(np.sum((q[name][a] >= 0.5) != (controls["base"] >= 0.5)))
                    for name in ARMS
                },
            )
            for name in ARMS:
                result[name] = shared.evaluate(
                    labels,
                    combine(q[name][a], flat),
                    prior["frequency"],
                    prior["prior_reversal"],
                    anchors,
                )
            results[key] = result
            print(
                key,
                {c: v["information_gate"] for c, v in result["comparisons"].items()},
                "direction",
                result["adjusted"]["gate"],
                flush=True,
            )
    write(root / "base-fits.json", base_fits)
    write(root / "fits.json", fits)
    calibration = all(
        r["comparisons"][f"adjusted_vs_{c}"]["information_gate"]
        for r in results.values()
        for c in ("base", "unshrunk", "neutral")
    )
    information = calibration and all(
        r["comparisons"]["adjusted_vs_flow"]["information_gate"]
        for r in results.values()
    )
    direction = all(r["adjusted"]["gate"] for r in results.values())
    write(
        root / "results.json",
        dict(
            cohorts=results,
            all_calibration_gates=calibration,
            all_information_gates=information,
            all_direction_gates=direction,
            general_solution=information and direction,
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
    {"register": register, "run": run}[args.action](args.root)
