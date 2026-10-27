#!/usr/bin/env python3
"""Independent audit of raw coefficients and uncertainty-adaptive shrinkage."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_universal_direction import (
    ROOT,
    ASSETS,
    check,
    same,
    digest,
    verify_metrics,
)  # noqa: E402
from scripts.verify_futures_flow import features, epoch, SOURCE  # noqa: E402
from scripts.verify_premium_direction import conditional  # noqa: E402
from scripts.verify_market_reversal import fit_check  # noqa: E402
from scripts.verify_market_calibration import reliability_check, membership  # noqa: E402

FUTURES = ROOT / "reports/timestral-futures-flow-20261027"
MARKET = ROOT / "reports/timestral-market-reversal-20261027"
TEMPERATURE = ROOT / "reports/timestral-market-calibration-20261027"


def adjustment_check(state, raw, z, y, ids, cutoff):
    error = fit_check(raw, z, y, ids, cutoff, 0)
    check(state["cutoff"] == cutoff and state["n"] == len(y), "adjustment scope")
    for key in ("training_ids_sha256", "training_labels_sha256"):
        check(state[key] == raw[key], "same raw training membership")
    days, inv = np.unique(ids[:, 1] // 86400, return_inverse=True)
    check(
        len(days) == state["days"] and len(days) >= 80 and np.all(np.diff(days) == 1),
        "ninety-day coverage",
    )
    beta = raw["beta"]
    p = 1 / (1 + np.exp(-beta * z))
    scores = np.array([np.mean((z * (p - y))[inv == k]) for k in range(len(days))])
    center = scores - scores.mean()
    kernel = np.maximum(
        0, 1 - abs(np.arange(len(days))[:, None] - np.arange(len(days))[None, :]) / 7
    )
    hessian = float(
        np.mean([np.mean((z * z * p * (1 - p))[inv == k]) for k in range(len(days))])
    )
    variance = (
        max(float(center @ kernel @ center), 0)
        / len(days) ** 2
        / max(hessian, 1e-12) ** 2
    )
    weight = max(0, 1 - variance / max(beta**2, 1e-24)) if hessian > 1e-12 else 0
    same(state["raw_beta"], beta)
    same(state["hessian"], hessian)
    same(state["variance"], variance)
    same(state["weight"], weight)
    same(state["beta"], weight * beta)
    return error


def verify(root):
    protocol = json.loads((root / "protocol.json").read_text())
    check(
        digest(root / "protocol.json")
        == json.loads((root / "protocol.sha256.json").read_text())["sha256"],
        "protocol hash",
    )
    for name, sha in json.loads((root / "manifest.json").read_text()).items():
        check(digest(root / name) == sha, f"artifact hash {name}")
    for name, sha in {**protocol["code"], **protocol["inputs"]}.items():
        check(digest(ROOT / name) == sha, f"code/input hash {name}")
    check(
        protocol["at"] < json.loads((root / "opened.json").read_text())["at"],
        "registration before scores",
    )
    arrays = {
        f"{year}-{s}": dict(np.load(FUTURES / f"{year}-{s}-futures.npz"))
        for year in (2024, 2025, 2026)
        for s in ASSETS
    }
    for d in arrays.values():
        check(d["time"].max() < 1785542400, "protected sources")
    fits = json.loads((root / "fits.json").read_text())
    old = json.loads((MARKET / "fits.json").read_text())
    results = json.loads((root / "results.json").read_text())
    maximum, count, cases = 0.0, 0, 0
    for year in (2024, 2025, 2026):
        t, end, x, y, available = features(FUTURES, year, arrays)
        z = np.maximum(-3, np.minimum(3, -x[:, :, 3] / 3 - 2 * x[:, :, 16] / 3))
        pieces, indices = [], []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (t >= cutoff - 90 * 86400) & (end <= cutoff) & available
            labels = y[:, train].reshape(-1)
            moving = labels != 1
            s = fits[f"{year}-{month}"]
            maximum = max(
                maximum,
                adjustment_check(
                    s,
                    old[f"{year}-{month}-market60_unshrunk"],
                    z[:, train].reshape(-1)[moving],
                    (labels[moving] == 2).astype(int),
                    membership(t, train)[moving],
                    cutoff,
                ),
            )
            test = (t >= cutoff) & (t < epoch(year, month + 1))
            pieces.append(1 / (1 + np.exp(-s["beta"] * z[:, test])))
            indices.append(np.flatnonzero(test))
            count += 1
        indices = np.concatenate(indices)
        q = np.concatenate(pieces, axis=1)
        for a, symbol in enumerate(ASSETS):
            key = f"{year}-{symbol}"
            prefix, folder = (
                (symbol, SOURCE) if year == 2026 else (key, SOURCE / "history")
            )
            prior = dict(np.load(folder / f"{prefix}-predictions.npz"))
            outcome = dict(np.load(folder / f"{prefix}-outcomes.npz"))
            market = dict(np.load(MARKET / f"{key}-predictions.npz"))
            temperature = dict(np.load(TEMPERATURE / f"{key}-predictions.npz"))
            flow = dict(np.load(FUTURES / f"{key}-predictions.npz"))
            saved = dict(np.load(root / f"{key}-predictions.npz"))
            for d in (prior, outcome, market, temperature, flow, saved):
                same(d["anchor"], t[indices])
            same(outcome["label"], y[a, indices])
            same(outcome["end"], end[indices])
            same(saved["adaptive_up"], q[a])
            check(end[indices].max() <= 1785542400, "protected outcomes")
            controls = dict(
                base=market["market60_up"],
                unshrunk=market["market60_unshrunk_up"],
                temperature=temperature["adjusted_up"],
                flow=flow["flow_up"],
                neutral=np.full(len(indices), 0.5),
                adaptive=q[a],
            )
            r = results["cohorts"][key]
            flat = prior["cross" if year == 2026 else "probability"][:, 1]
            check(
                r["direction_changes"]
                == int(np.sum((q[a] >= 0.5) != (controls["base"] >= 0.5))),
                "direction changes",
            )
            combined = np.column_stack(
                ((1 - flat) * (1 - q[a]), flat, (1 - flat) * q[a])
            )
            verify_metrics(
                r["adaptive"],
                y[a, indices],
                combined,
                prior["frequency"],
                prior["prior_reversal"],
                t[indices],
            )
            for c in ("base", "unshrunk", "temperature", "flow", "neutral"):
                conditional(
                    {
                        k.replace("vs_control", "vs_spot"): v
                        for k, v in r["comparisons"][f"adaptive_vs_{c}"].items()
                    },
                    y[a, indices],
                    q[a],
                    controls[c],
                    prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1]),
                    t[indices],
                )
            for name, p in controls.items():
                reliability_check(r["calibration"][name], y[a, indices], p)
            cases += len(indices)
    check(
        len(fits) == count == 21 and len(results["cohorts"]) == 9 and cases == 550362,
        "all states/cohorts/cases",
    )
    calibration = all(
        r["comparisons"][f"adaptive_vs_{c}"]["information_gate"]
        for r in results["cohorts"].values()
        for c in ("base", "unshrunk", "neutral")
    )
    information = calibration and all(
        r["comparisons"]["adaptive_vs_flow"]["information_gate"]
        for r in results["cohorts"].values()
    )
    direction = all(r["adaptive"]["gate"] for r in results["cohorts"].values())
    check(
        results["all_calibration_gates"] == calibration
        and results["all_information_gates"] == information
        and results["all_direction_gates"] == direction
        and results["general_solution"] == (information and direction),
        "combined gates",
    )
    return dict(
        verified=True,
        adjustments=count,
        cohorts=9,
        cases=cases,
        maximum_raw_fit_residual=maximum,
        protected_data_opened=False,
        general_solution=results["general_solution"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "reports/timestral-uncertain-market-20261027",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), sort_keys=True))
