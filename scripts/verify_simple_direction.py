#!/usr/bin/env python3
"""Independently reconstruct scalar signals and check convex optima; no fitting."""

import argparse
import hashlib
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

FUTURES = ROOT / "reports/timestral-futures-flow-20261027"
ARMS = (("reversal", 0.25), ("relative_unshrunk", 0.0), ("relative", 0.25))


def fit_check(state, z, y, ids, cutoff, penalty):
    check(state["n"] == len(y) and state["cutoff"] == cutoff, "training size/cutoff")
    check(
        state["penalty"] == penalty and 0 <= state["beta"] <= 8, "fixed penalty/bounds"
    )
    check(
        hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
        == state["training_ids_sha256"],
        "training membership",
    )
    check(
        hashlib.sha256(y.astype("<i8").tobytes()).hexdigest()
        == state["training_labels_sha256"],
        "mature training labels",
    )
    days, inv = np.unique(ids[:, 1] // 86400, return_inverse=True)
    count = np.bincount(inv)
    beta = state["beta"]
    p = 1 / (1 + np.exp(-beta * z))
    # Aggregate derivatives by UTC day, rather than using production weights.
    gradient = float(
        np.mean(np.bincount(inv, weights=z * (p - y)) / count) + penalty * beta
    )
    residual = (
        max(0, -gradient)
        if beta == 0
        else max(0, gradient)
        if beta == 8
        else abs(gradient)
    )
    check(residual < 1e-8, "scalar convex KKT condition")
    same(state["derivative"], gradient)
    check(state["days"] == len(days), "equal-day weighting")
    loss = np.logaddexp(0, beta * z) - y * beta * z
    same(state["training_logloss"], np.mean(np.bincount(inv, weights=loss) / count))
    return residual


def calibration_check(stored, y, p):
    keep = y != 1
    p, label = p[keep], y[keep] == 2
    confidence = 0.5 + abs(p - 0.5)
    correct = (p >= 0.5) == label
    check(
        stored["n"] == len(p) and stored["neutral_predictions"] == np.sum(p == 0.5),
        "calibration scope",
    )
    same(stored["mean_confidence"], confidence.mean())
    same(stored["confidence_q95"], np.quantile(confidence, 0.95))
    same(stored["confidence_minus_accuracy"], confidence.mean() - correct.mean())


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
        "registered before evaluation",
    )
    arrays = {
        f"{year}-{s}": dict(np.load(FUTURES / f"{year}-{s}-futures.npz"))
        for year in (2024, 2025, 2026)
        for s in ASSETS
    }
    for data in arrays.values():
        check(data["time"].max() < 1785542400, "protected futures data excluded")
    fits = json.loads((root / "fits.json").read_text())
    results = json.loads((root / "results.json").read_text())
    maximum, fit_count, cases = 0.0, 0, 0
    for year in (2024, 2025, 2026):
        t, end, x, y, available = features(FUTURES, year, arrays)
        z = {"reversal": np.maximum(-3, np.minimum(3, -x[:, :, 0]))}
        relative = (x[:, :, 13] - x[:, :, 0]) * (2**-0.5)
        z["relative"] = z["relative_unshrunk"] = np.maximum(-3, np.minimum(3, relative))
        predictions = {name: [] for name, _ in ARMS}
        indices = []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (t >= cutoff - 90 * 86400) & (end <= cutoff) & available
            test = (t >= cutoff) & (t < epoch(year, month + 1))
            labels = y[:, train].reshape(-1)
            moving = labels != 1
            ids = np.concatenate(
                [np.column_stack((np.full(train.sum(), a), t[train])) for a in range(3)]
            )
            for name, penalty in ARMS:
                state = fits[f"{year}-{month}-{name}"]
                error = fit_check(
                    state,
                    z[name][:, train].reshape(-1)[moving],
                    (labels[moving] == 2).astype(int),
                    ids[moving],
                    cutoff,
                    penalty,
                )
                maximum = max(maximum, error)
                predictions[name].append(
                    1 / (1 + np.exp(-state["beta"] * z[name][:, test]))
                )
                fit_count += 1
            check(
                fits[f"{year}-{month}-relative"]["beta"]
                <= fits[f"{year}-{month}-relative_unshrunk"]["beta"] + 1e-10,
                "penalty shrinks fitted magnitude",
            )
            indices.append(np.flatnonzero(test))
        indices = np.concatenate(indices)
        q = {
            name: np.concatenate(pieces, axis=1) for name, pieces in predictions.items()
        }
        for a, s in enumerate(ASSETS):
            key = f"{year}-{s}"
            prefix, folder = (s, SOURCE) if year == 2026 else (key, SOURCE / "history")
            prior = dict(np.load(folder / f"{prefix}-predictions.npz"))
            outcome = dict(np.load(folder / f"{prefix}-outcomes.npz"))
            flow = dict(np.load(FUTURES / f"{key}-predictions.npz"))
            saved = dict(np.load(root / f"{key}-predictions.npz"))
            for stored in (prior, outcome, flow, saved):
                same(stored["anchor"], t[indices])
            same(outcome["label"], y[a, indices])
            same(outcome["end"], end[indices])
            check(end[indices].max() <= 1785542400, "protected outcomes excluded")
            cases += len(indices)
            controls = {
                "flow": flow["flow_up"],
                "neutral": np.full(len(indices), 0.5),
                **{name: q[name][a] for name, _ in ARMS},
            }
            flat = prior["cross" if year == 2026 else "probability"][:, 1]
            for name, _ in ARMS:
                p = q[name][a]
                same(saved[name + "_up"], p)
                combined = np.column_stack(((1 - flat) * (1 - p), flat, (1 - flat) * p))
                verify_metrics(
                    results["cohorts"][key][name],
                    y[a, indices],
                    combined,
                    prior["frequency"],
                    prior["prior_reversal"],
                    t[indices],
                )
            for control in ("neutral", "flow", "reversal", "relative_unshrunk"):
                metrics = results["cohorts"][key]["comparisons"][
                    f"relative_vs_{control}"
                ]
                conditional(
                    {k.replace("vs_control", "vs_spot"): v for k, v in metrics.items()},
                    y[a, indices],
                    q["relative"][a],
                    controls[control],
                    prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1]),
                    t[indices],
                )
            for name, p in controls.items():
                calibration_check(
                    results["cohorts"][key]["calibration"][name], y[a, indices], p
                )
    check(
        len(fits) == fit_count == 63
        and len(results["cohorts"]) == 9
        and cases == 550362,
        "all registered fits/cohorts/anchors",
    )
    information = all(
        r["comparisons"][f"relative_vs_{c}"]["information_gate"]
        for r in results["cohorts"].values()
        for c in ("neutral", "flow", "reversal")
    )
    direction = all(r["relative"]["gate"] for r in results["cohorts"].values())
    check(
        results["all_information_gates"] == information
        and results["all_direction_gates"] == direction
        and results["general_solution"] == (information and direction),
        "combined gates",
    )
    return dict(
        verified=True,
        fits=fit_count,
        cohorts=9,
        cases=cases,
        maximum_kkt_residual=maximum,
        protected_data_opened=False,
        general_solution=results["general_solution"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "reports/timestral-simple-direction-20261027",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), sort_keys=True))
