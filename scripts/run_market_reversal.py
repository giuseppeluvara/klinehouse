#!/usr/bin/env python3
"""Registered one-coefficient common-market reversal experiment, no model sweep."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_market_reversal as model  # noqa: E402
from research import timestral_universal as shared  # noqa: E402
from research.timestral_futures_flow import combine, comparison  # noqa: E402
from scripts import run_futures_flow as previous  # noqa: E402
from scripts.run_futures_flow import ROOT, digest, write, epoch, sources, prior_path  # noqa: E402

DEFAULT = ROOT / "reports/timestral-market-reversal-20261027"
FUTURES = ROOT / "reports/timestral-futures-flow-20261027"
CODE = (
    "research/timestral_market_reversal.py",
    "research/timestral_simple_direction.py",
    "scripts/run_market_reversal.py",
    "scripts/verify_market_reversal.py",
    "scripts/verify_futures_flow.py",
    "scripts/verify_premium_direction.py",
    "scripts/verify_universal_direction.py",
    "tests/test_timestral_market_reversal.py",
    "tests/test_timestral_simple_direction.py",
    *previous.CODE,
)
COMPARISONS = ("neutral", "flow", "market1", "market60_unshrunk")
PLAN = {
    "purpose": "Test one common-hour reversal coefficient after the one-minute relative hypothesis failed. This follow-up is explicitly informed by earlier results, not an independent confirmation.",
    "motivation": "The first simple experiment removed common one-minute movement and failed. Inspecting the already explored frozen flow coefficients showed negative average own/other60minute slopes in each year. Test that component alone rather than tune the failed residual model.",
    "status": "Retrospective Jan-Jul2024/2025/2026 BTC/ETH/SOL. Prior outcomes/model coefficients informed hypothesis; no pristine or prospective validation claim.",
    "data": "Same frozen sources,550362 anchors, five-minute spot target anchor+60sec to anchor+360sec. Futures only determine identical training eligibility and frozen control. No downloads, no protected data.",
    "signals": "market1=-clip((own_return1+2*other_return1)/3,-3,3); market60 uses own_return60/other_return60 likewise. These are frozen normalized log returns: own60=(logprice-now minus60minutes ago)/(past60min RMS*sqrt(60)), clipped+-8 before averaging; other is mean of other two normalized returns. All spot information closes through anchor. No fitted normalization, intercept, asset ID or regimes.",
    "models": "market1 penalty.25; market60_unshrunk penalty0; primary market60 penalty.25. Each has one pooled coefficient beta in[0,8] and probability_up=sigmoid(beta*signal). Beta0 means no support for predeclared reversal. A fixed orientation is an imposed constraint, not evidence of stability.",
    "fit": "Reuse unchanged scalar fitting code: monthly90day matured nonflat samples, same common futures availability, equal total weight per UTC training day pooled across3assets; mean day-weighted logloss + penalty*beta^2/2; convex derivative root xtol1e-13.63fits, no parameter/horizon search after scores. Upper bound8 is numerical guard.",
    "comparisons": "market60 vs neutral.5, frozen flow, market1 and market60_unshrunk. Fixed flat probabilities for original all-anchor direction gates. No ability to identify future flats is assumed.",
    "primary": "General solution requires all9 original unconditional market60 direction gates AND all9 conditional information gates versus neutral, flow and market1. Unshrunk comparison isolates regularization only. Same thresholds: accuracy lower99.444%>50%, Brier/logloss gain lower99.444%>0 against named control AND past frequency,>=5 positive monthly Brier gains,>=100 each direction. Original52% selection gate unchanged.",
    "uncertainty": "4000 paired7day blocks seed271032, nine-cohort correction per endpoint; not a joint correction over endpoints, comparisons or sequential research history. No costs/trading/prospective claim.",
    "diagnosis": "All63 coefficients and day-weighted training losses; nonflat mean/q95 confidence, confidence minus accuracy and neutral count. No refitting based on scores. Reduced confidence alone cannot establish directional discrimination.",
    "protected": "No market data/outcomes from2026-08-01 onward. Preserve all older frozen artifacts and user files. Previously admitted final targets afterJul31 only in2024/25 are unchanged.",
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
        z = {name: model.signal(sample["x"], name) for name, _ in model.ARMS}
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
                    f"market60_vs_{control}": comparison(
                        y, q["market60"][a], controls[control], frequency, t
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
                results[key]["market60"]["gate"],
                flush=True,
            )
    write(root / "fits.json", fits)
    information = all(
        r["comparisons"][f"market60_vs_{c}"]["information_gate"]
        for r in results.values()
        for c in ("neutral", "flow", "market1")
    )
    direction = all(r["market60"]["gate"] for r in results.values())
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
