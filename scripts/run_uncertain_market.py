#!/usr/bin/env python3
"""Registered uncertainty adjustment of the frozen single market coefficient."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_uncertain_market as model
from research import timestral_market_reversal as base
from research.timestral_market_calibration import reliability  # noqa: E402
from research import timestral_universal as shared  # noqa: E402
from research.timestral_futures_flow import combine, comparison  # noqa: E402
from scripts import run_futures_flow as previous  # noqa: E402
from scripts.run_futures_flow import ROOT, digest, write, epoch, sources, prior_path  # noqa: E402

DEFAULT = ROOT / "reports/timestral-uncertain-market-20261027"
FUTURES = ROOT / "reports/timestral-futures-flow-20261027"
CODE = (
    "research/timestral_market_reversal.py",
    "research/timestral_uncertain_market.py",
    "research/timestral_market_calibration.py",
    "research/timestral_simple_direction.py",
    "scripts/run_uncertain_market.py",
    "scripts/verify_uncertain_market.py",
    "scripts/verify_futures_flow.py",
    "scripts/verify_premium_direction.py",
    "scripts/verify_universal_direction.py",
    "tests/test_timestral_market_reversal.py",
    "tests/test_timestral_uncertain_market.py",
    "scripts/verify_market_calibration.py",
    "scripts/verify_market_reversal.py",
    "tests/test_timestral_simple_direction.py",
    *previous.CODE,
)
MARKET = ROOT / "reports/timestral-market-reversal-20261027"
TEMPERATURE = ROOT / "reports/timestral-market-calibration-20261027"
CONTROLS = ("base", "unshrunk", "temperature", "flow", "neutral")
PLAN = {
    "purpose": "Correct excessive fixed shrinkage using measured uncertainty of the single common-hour coefficient; no second temperature factor.",
    "motivation": "Sequential follow-up after prequential temperature correction worsened Brier versus the unpenalized model in8/9 cohorts. Hypothesis: directly shrinking the raw common-hour coefficient avoids an extra time-varying calibration scale. Not proof that scale drift caused the previous loss.",
    "status": "Retrospective Jan-Jul2024/2025/2026 BTC/ETH/SOL. Outcomes and earlier fitted coefficients already explored; no pristine or prospective claim.",
    "base": "Reuse all21 frozen market60_unshrunk monthly scalar coefficients exactly. Same90day matured nonflat pooled3asset training membership, equal UTC-day weighting and futures availability as original. Same clipped common normalized hourly return; no refit or new feature.",
    "adjustment": "At raw beta compute daily mean logistic score and average Hessian. Center day scores, Bartlett HAC lags1..6 weights1-lag/7, variance=nonnegative score HAC sum/n_days^2/Hessian^2, Hessian floor1e-12. Weight=max(0,1-variance/max(beta^2,1e-24)); use0 if Hessian<=1e-12. Adjusted beta=weight*raw beta. This fixed heuristic estimates noise and shrinks toward neutrality; not a guarantee of calibrated probabilities or no overfitting. Minimum80 contiguous training days.21 adjustment states; no parameter/window sweep.",
    "scope": "All550362 original anchors, five-minute spot target anchor+60sec to anchor+360sec, unchanged flat probabilities. Positive beta preserves original up/down decisions, zero means neutral. No asset-specific calibration or new optimized scalar.",
    "comparisons": "Adaptive versus fixed-shrunk base, unshrunk, earlier adjusted temperature, frozen flow and neutral. All comparisons also test past frequency. Preserve original all-anchor52% direction gate.",
    "primary": "Calibration success requires all9 conditional information gates vs base AND unshrunk AND neutral. General solution also requires all9 vs flow and all9 original adaptive direction gates. Conditional thresholds unchanged: accuracy lower99.444%>50%, Brier/logloss gain lower99.444%>0 vs named control and frequency,>=5 positive monthly Brier gains,>=100 each direction. Earlier-temperature comparison diagnostic.",
    "uncertainty": "Frozen4000 paired7day blocks seed271032,9cohort correction per endpoint, no joint correction across endpoints/comparisons/sequential research history. Same fixed reliability bins as temperature study; ECE/gaps descriptive. No bootstrap refitting.",
    "protected": "No market sources/outcomes from2026-08-01 onward. Reuse frozen data and models; preserve every previous result. No downloads, installs, orders or profitability claim.",
}


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    paths = [MARKET / "fits.json"] + [
        p
        for year in (2024, 2025, 2026)
        for p in [
            *sources(year).values(),
            *[
                folder / f"{year}-{s}-predictions.npz"
                for folder in (MARKET, TEMPERATURE)
                for s in shared.SYMBOLS
            ],
            *[prior_path(year, s) for s in shared.SYMBOLS],
            *[
                FUTURES / f"{year}-{s}-{suffix}.npz"
                for s in shared.SYMBOLS
                for suffix in ("futures", "predictions")
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


def validate(root):
    p = json.loads((root / "protocol.json").read_text())
    if (
        p["plan"] != PLAN
        or digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
        or any(
            digest(ROOT / name) != h for name, h in {**p["code"], **p["inputs"]}.items()
        )
    ):
        raise ValueError("frozen experiment changed")


def run(root):
    validate(root)
    write(root / "opened.json", dict(at=datetime.now(timezone.utc).isoformat()))
    old = json.loads((MARKET / "fits.json").read_text())
    fits, results = {}, {}
    for year in (2024, 2025, 2026):
        sample = previous.examples(FUTURES, year)
        t, end, y = sample["anchor"], sample["end"], sample["y"]
        z = base.signal(sample["x"], "market60")
        pieces, indices = [], []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (t >= cutoff - 90 * 86400) & (end <= cutoff) & sample["available"]
            labels = y[:, train].reshape(-1)
            moving = labels != 1
            state = model.adjust(
                old[f"{year}-{month}-market60_unshrunk"],
                z[:, train].reshape(-1)[moving],
                (labels[moving] == 2).astype(int),
                np.tile(t[train], 3)[moving],
                np.tile(end[train], 3)[moving],
                cutoff,
            )
            ids = np.concatenate(
                [np.column_stack((np.full(train.sum(), a), t[train])) for a in range(3)]
            ).astype("<i8")
            state["training_ids_sha256"] = hashlib.sha256(
                ids[moving].tobytes()
            ).hexdigest()
            state["training_labels_sha256"] = hashlib.sha256(
                (labels[moving] == 2).astype("<i8").tobytes()
            ).hexdigest()
            fits[f"{year}-{month}"] = state
            test = (t >= cutoff) & (t < epoch(year, month + 1))
            pieces.append(
                np.stack([model.predict(state, z[a, test], t[test]) for a in range(3)])
            )
            indices.append(np.flatnonzero(test))
        indices = np.concatenate(indices)
        q = np.concatenate(pieces, axis=1)
        for a, s in enumerate(shared.SYMBOLS):
            key = f"{year}-{s}"
            prior = dict(np.load(prior_path(year, s)))
            market = dict(np.load(MARKET / f"{key}-predictions.npz"))
            temperature = dict(np.load(TEMPERATURE / f"{key}-predictions.npz"))
            flow = dict(np.load(FUTURES / f"{key}-predictions.npz"))
            for d in (prior, market, temperature, flow):
                if not np.array_equal(d["anchor"], t[indices]):
                    raise ValueError("changed evaluation anchors")
            controls = dict(
                base=market["market60_up"],
                unshrunk=market["market60_unshrunk_up"],
                temperature=temperature["adjusted_up"],
                flow=flow["flow_up"],
                neutral=np.full(len(indices), 0.5),
                adaptive=q[a],
            )
            labels, anchors = y[a, indices], t[indices]
            flat = prior["cross" if year == 2026 else "probability"][:, 1]
            frequency = prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1])
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(f, anchor=anchors, adaptive_up=q[a])
            results[key] = dict(
                comparisons={
                    f"adaptive_vs_{c}": comparison(
                        labels, q[a], controls[c], frequency, anchors
                    )
                    for c in CONTROLS
                },
                calibration={
                    name: reliability(labels, p) for name, p in controls.items()
                },
                adaptive=shared.evaluate(
                    labels,
                    combine(q[a], flat),
                    prior["frequency"],
                    prior["prior_reversal"],
                    anchors,
                ),
                direction_changes=int(
                    np.sum((q[a] >= 0.5) != (controls["base"] >= 0.5))
                ),
            )
            print(
                key,
                {
                    c: v["information_gate"]
                    for c, v in results[key]["comparisons"].items()
                },
                "direction",
                results[key]["adaptive"]["gate"],
                flush=True,
            )
    write(root / "fits.json", fits)
    calibration = all(
        r["comparisons"][f"adaptive_vs_{c}"]["information_gate"]
        for r in results.values()
        for c in ("base", "unshrunk", "neutral")
    )
    information = calibration and all(
        r["comparisons"]["adaptive_vs_flow"]["information_gate"]
        for r in results.values()
    )
    direction = all(r["adaptive"]["gate"] for r in results.values())
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
