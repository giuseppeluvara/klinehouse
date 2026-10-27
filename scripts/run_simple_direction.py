#!/usr/bin/env python3
"""Registered one-coefficient relative reversal experiment, no model sweep."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_simple_direction as model  # noqa: E402
from research import timestral_universal as shared  # noqa: E402
from research.timestral_futures_flow import combine, comparison  # noqa: E402
from scripts import run_futures_flow as previous  # noqa: E402
from scripts.run_futures_flow import ROOT, digest, write, epoch, sources, prior_path  # noqa: E402

DEFAULT = ROOT / "reports/timestral-simple-direction-20261027"
FUTURES = ROOT / "reports/timestral-futures-flow-20261027"
CODE = (
    "research/timestral_simple_direction.py",
    "scripts/run_simple_direction.py",
    "scripts/verify_simple_direction.py",
    "scripts/verify_futures_flow.py",
    "scripts/verify_premium_direction.py",
    "scripts/verify_universal_direction.py",
    "tests/test_timestral_simple_direction.py",
    *previous.CODE,
)
COMPARISONS = ("neutral", "flow", "reversal", "relative_unshrunk")
PLAN = {
    "purpose": "Reduce overfitting with one shared coefficient and fixed shrinkage; test whether relative rather than common one-minute moves carry stable reversal information.",
    "status": "Exploratory retrospective Jan-Jul2024/2025/2026 BTC/ETH/SOL; earlier outcomes motivated this hypothesis, not untouched validation.",
    "data": "Reuse frozen spot/futures arrays only. Same550362 evaluation anchors, target spot return from anchor+60sec to anchor+360sec. Futures read only for same training eligibility and frozen control; no futures features in new models. No new downloads.",
    "signals": "reversal=-clip(spot_return1,-3,3). relative=clip((other_return1-spot_return1)/sqrt(2),-3,3). Return1 is the frozen causal60min RMS-normalized one-minute log return; other_return1 is the mean of the other two normalized returns. Spot bars close through anchor. No asset ID, intercept, interactions, normalization fitting or regimes.",
    "models": "Three arms, each one nonnegative coefficient beta in[0,8]: reversal penalty.25; relative_unshrunk penalty0; primary relative penalty.25. Probability_up=sigmoid(beta*signal). No evidence for the predeclared orientation produces beta0 and probability.5, never a fitted sign flip.",
    "fit": "Monthly trailing90days, matured nonflat labels, pooled3assets and same available-futures membership as frozen flow. Every UTC training day has equal total weight across available nonflat asset samples. Minimize day-weighted logloss + penalty*beta^2/2. Exact one-dimensional convex derivative root or boundary, brentq xtol1e-13.63fits. Fixed penalty.25 chosen before new results to shrink toward neutrality; not selected by validation scores. The upper bound8 is a numerical guard, not confidence calibration.",
    "comparisons": "Primary relative vs neutral.5, frozen flow, simple reversal and relative_unshrunk. Flat probabilities unchanged for all original direction evaluations. No sample selection based on future flat outcomes for original gates.",
    "primary": "A general solution requires all9 original unconditional direction gates AND all9 conditional information gates versus neutral, flow and reversal. Comparison to unshrunk isolates regularization and is diagnostic, not another candidate-selection exercise. Preserve all prior thresholds. Conditional accuracy lower99.444%>50%; Brier/logloss gain lower99.444%>0 against named control AND past frequency; >=5 positive monthly Brier gains; >=100 each predicted direction.",
    "uncertainty": "Frozen scoring:4000 paired7day blocks seed271032; correction over9cohorts per endpoint, not jointly over endpoints, comparisons or research history. Original52% selection gate unchanged. No trading claim or prospective validation.",
    "diagnosis": "Record all63 coefficients, training logloss, mean/95th-percentile confidence, confidence minus observed accuracy and exact neutral count on nonflat outcomes. Compare shrinkage vs unshrunk without retuning. A fixed slope sign is an imposed constraint, not proof of signal stability. Calibration-in-the-large alone is not full calibration.",
    "protected": "No market source/outcome from2026-08-01 onward. Existing source arrays and frozen protocols/models/artifacts unchanged. All original anchors kept, including labels already admitted afterJul31 in2024/25 only.",
}


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    paths = [
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
    fits, results = {}, {}
    for year in (2024, 2025, 2026):
        sample = previous.examples(FUTURES, year)
        predictions = {name: [] for name, _ in model.ARMS}
        z = {
            name: model.signal(sample["x"], name != "reversal")
            for name, _ in model.ARMS
        }
        indices = []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (
                (sample["anchor"] >= cutoff - 90 * 86400)
                & (sample["end"] <= cutoff)
                & sample["available"]
            )
            test = (sample["anchor"] >= cutoff) & (
                sample["anchor"] < epoch(year, month + 1)
            )
            labels = sample["y"][:, train].reshape(-1)
            moving = labels != 1
            anchors = np.tile(sample["anchor"][train], 3)
            ends = np.tile(sample["end"][train], 3)
            ids = np.concatenate(
                [
                    np.column_stack((np.full(train.sum(), a), sample["anchor"][train]))
                    for a in range(3)
                ]
            ).astype("<i8")
            for name, penalty in model.ARMS:
                state = model.fit(
                    z[name][:, train].reshape(-1)[moving],
                    (labels[moving] == 2).astype(int),
                    anchors[moving],
                    ends[moving],
                    cutoff,
                    penalty,
                )
                state["training_ids_sha256"] = hashlib.sha256(
                    ids[moving].tobytes()
                ).hexdigest()
                state["training_labels_sha256"] = hashlib.sha256(
                    (labels[moving] == 2).astype("<i8").tobytes()
                ).hexdigest()
                fits[f"{year}-{month}-{name}"] = state
                predictions[name].append(
                    np.stack(
                        [
                            model.predict(
                                state, z[name][a, test], sample["anchor"][test]
                            )
                            for a in range(3)
                        ]
                    )
                )
            indices.append(np.flatnonzero(test))
            print("fitted", year, month, flush=True)
        indices = np.concatenate(indices)
        q = {k: np.concatenate(v, axis=1) for k, v in predictions.items()}
        for a, s in enumerate(shared.SYMBOLS):
            key = f"{year}-{s}"
            prior = dict(np.load(prior_path(year, s)))
            flow = dict(np.load(FUTURES / f"{key}-predictions.npz"))
            t, y = sample["anchor"][indices], sample["y"][a, indices]
            if not np.array_equal(t, prior["anchor"]) or not np.array_equal(
                t, flow["anchor"]
            ):
                raise ValueError("evaluation anchors changed")
            controls = {
                "flow": flow["flow_up"],
                "neutral": np.full(len(t), 0.5),
                **{name: q[name][a] for name, _ in model.ARMS},
            }
            flat = prior["cross" if year == 2026 else "probability"][:, 1]
            frequency = prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1])
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(
                    f, anchor=t, **{name + "_up": q[name][a] for name, _ in model.ARMS}
                )
            results[key] = dict(
                comparisons={
                    f"relative_vs_{control}": comparison(
                        y, q["relative"][a], controls[control], frequency, t
                    )
                    for control in COMPARISONS
                },
                calibration={
                    name: model.calibration(y, p) for name, p in controls.items()
                },
            )
            for name, _ in model.ARMS:
                results[key][name] = shared.evaluate(
                    y,
                    combine(q[name][a], flat),
                    prior["frequency"],
                    prior["prior_reversal"],
                    t,
                )
            print(
                key,
                {
                    k: v["information_gate"]
                    for k, v in results[key]["comparisons"].items()
                },
                "direction",
                results[key]["relative"]["gate"],
                flush=True,
            )
    write(root / "fits.json", fits)
    information = all(
        r["comparisons"][f"relative_vs_{c}"]["information_gate"]
        for r in results.values()
        for c in ("neutral", "flow", "reversal")
    )
    direction = all(r["relative"]["gate"] for r in results.values())
    write(
        root / "results.json",
        dict(
            cohorts=results,
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
