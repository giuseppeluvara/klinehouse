#!/usr/bin/env python3
"""Independent prequential memberships, HAC uncertainty and calibration audit."""

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
from scripts.verify_market_reversal import fit_check, calibration_check  # noqa: E402

FUTURES = ROOT / "reports/timestral-futures-flow-20261027"
MARKET = ROOT / "reports/timestral-market-reversal-20261027"


def membership(t, mask):
    return np.concatenate(
        [np.column_stack((np.full(mask.sum(), a), t[mask])) for a in range(3)]
    )


def calibration_fit(state, x, y, ids, issued, cutoff):
    check(state["n"] == len(y) and state["cutoff"] == cutoff, "calibration scope")
    check(
        np.all(ids[:, 1] + 360 <= cutoff)
        and np.all(ids[:, 1] >= cutoff - 56 * 86400)
        and np.all(issued <= ids[:, 1]),
        "mature out-of-training scores",
    )
    for name, values in (("ids", ids), ("labels", y)):
        check(
            hashlib.sha256(values.astype("<i8").tobytes()).hexdigest()
            == state[f"training_{name}_sha256"],
            "calibration membership hash",
        )
    days, inv = np.unique(ids[:, 1] // 86400, return_inverse=True)
    check(
        len(days) == state["days"] and len(days) >= 42 and np.all(np.diff(days) == 1),
        "calibration day coverage",
    )
    a = state["raw_factor"]
    check(0 <= a <= 8, "factor bounds")
    p = 1 / (1 + np.exp(-a * x))
    score = np.array(
        [np.mean(x[inv == k] * (p[inv == k] - y[inv == k])) for k in range(len(days))]
    )
    gradient = float(score.mean())
    error = (
        max(0, -gradient) if a == 0 else max(0, gradient) if a == 8 else abs(gradient)
    )
    check(error < 1e-8, "calibration convex optimum")
    same(state["derivative"], gradient)
    centered = score - score.mean()
    bartlett = np.maximum(
        0, 1 - abs(np.arange(len(days))[:, None] - np.arange(len(days))[None, :]) / 7
    )
    omega = float(centered @ bartlett @ centered)
    hessian = float(
        np.mean([np.mean((x * x * p * (1 - p))[inv == k]) for k in range(len(days))])
    )
    variance = max(omega, 0) / len(days) ** 2 / max(hessian, 1e-12) ** 2
    shrink = max(0, 1 - variance / max((a - 1) ** 2, 1e-24)) if hessian > 1e-12 else 0
    same(state["hessian"], hessian)
    same(state["factor_variance"], variance)
    same(state["shrink"], shrink)
    same(state["adjusted_factor"], 1 + shrink * (a - 1))
    if np.all(x == 0):
        check(a == state["adjusted_factor"] == 1, "no-information identity fallback")
    losses = np.logaddexp(0, a * x) - y * a * x
    same(
        state["calibration_logloss"],
        np.mean([np.mean(losses[inv == k]) for k in range(len(days))]),
    )
    return error


def reliability_check(stored, y, p):
    calibration_check(stored, y, p)
    keep = y != 1
    confidence = np.maximum(p[keep], 1 - p[keep])
    correct = (p[keep] >= 0.5) == (y[keep] == 2)
    edges = [0.5, 0.51, 0.52, 0.54, 0.58, 0.66, 1]
    check(len(stored["bins"]) == 6, "fixed reliability bins")
    ece = 0.0
    for k, b in enumerate(stored["bins"]):
        take = (confidence >= edges[k]) & (
            (confidence < edges[k + 1]) if k < 5 else (confidence <= 1)
        )
        check(
            b["lower"] == edges[k]
            and b["upper"] == edges[k + 1]
            and b["n"] == int(take.sum()),
            "reliability membership",
        )
        if take.any():
            same(b["confidence"], confidence[take].mean())
            same(b["accuracy"], correct[take].mean())
            ece += abs(np.sum(confidence[take] - correct[take]))
        else:
            check(
                b["confidence"] is None and b["accuracy"] is None,
                "empty reliability bin",
            )
    same(stored["ece_fixed_bins"], ece / len(confidence))


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
        check(data["time"].max() < 1785542400, "protected sources")
    base_fits = json.loads((root / "base-fits.json").read_text())
    fits = json.loads((root / "fits.json").read_text())
    old = json.loads((MARKET / "fits.json").read_text())
    results = json.loads((root / "results.json").read_text())
    maximum, base_count, cal_count, cases = 0.0, 0, 0, 0
    for year in (2024, 2025, 2026):
        t, end, x, y, available = features(FUTURES, year, arrays)
        z = np.maximum(-3, np.minimum(3, -x[:, :, 3] / 3 - 2 * x[:, :, 16] / 3))
        logits, issued = np.full_like(z, np.nan), np.zeros(len(t), dtype=np.int64)
        for yy, month in [(year - 1, 11), (year - 1, 12)] + [
            (year, m) for m in range(1, 8)
        ]:
            cutoff = epoch(yy, month)
            train = (t >= cutoff - 90 * 86400) & (end <= cutoff) & available
            labels = y[:, train].reshape(-1)
            moving = labels != 1
            state = base_fits[f"{year}-{yy}-{month}"]
            maximum = max(
                maximum,
                fit_check(
                    state,
                    z[:, train].reshape(-1)[moving],
                    (labels[moving] == 2).astype(int),
                    membership(t, train)[moving],
                    cutoff,
                    0.25,
                ),
            )
            if yy == year:
                check(
                    state == old[f"{year}-{month}-market60"],
                    "base fits reused unchanged",
                )
            test = (t >= cutoff) & (t < epoch(yy + (month == 12), month % 12 + 1))
            logits[:, test], issued[test] = state["beta"] * z[:, test], cutoff
            base_count += 1
        predictions = {name: [] for name in ("raw", "adjusted")}
        indices = []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            cal = (
                (t >= cutoff - 56 * 86400)
                & (end <= cutoff)
                & np.isfinite(logits).all(axis=0)
            )
            labels = y[:, cal].reshape(-1)
            moving = labels != 1
            state = fits[f"{year}-{month}"]
            maximum = max(
                maximum,
                calibration_fit(
                    state,
                    logits[:, cal].reshape(-1)[moving],
                    (labels[moving] == 2).astype(int),
                    membership(t, cal)[moving],
                    np.tile(issued[cal], 3)[moving],
                    cutoff,
                ),
            )
            test = (t >= cutoff) & (t < epoch(year, month + 1))
            for name in predictions:
                predictions[name].append(
                    1 / (1 + np.exp(-state[name + "_factor"] * logits[:, test]))
                )
            indices.append(np.flatnonzero(test))
            cal_count += 1
        indices = np.concatenate(indices)
        q = {
            name: np.concatenate(pieces, axis=1) for name, pieces in predictions.items()
        }
        for a, s in enumerate(ASSETS):
            key = f"{year}-{s}"
            prefix, folder = (s, SOURCE) if year == 2026 else (key, SOURCE / "history")
            prior = dict(np.load(folder / f"{prefix}-predictions.npz"))
            outcome = dict(np.load(folder / f"{prefix}-outcomes.npz"))
            market = dict(np.load(MARKET / f"{key}-predictions.npz"))
            flow = dict(np.load(FUTURES / f"{key}-predictions.npz"))
            saved = dict(np.load(root / f"{key}-predictions.npz"))
            for d in (prior, outcome, market, flow, saved):
                same(d["anchor"], t[indices])
            same(outcome["label"], y[a, indices])
            same(outcome["end"], end[indices])
            check(end[indices].max() <= 1785542400, "protected outcomes")
            same(1 / (1 + np.exp(-logits[a, indices])), market["market60_up"])
            controls = dict(
                base=market["market60_up"],
                unshrunk=market["market60_unshrunk_up"],
                flow=flow["flow_up"],
                neutral=np.full(len(indices), 0.5),
                **{name: q[name][a] for name in q},
            )
            r = results["cohorts"][key]
            flat = prior["cross" if year == 2026 else "probability"][:, 1]
            for name in q:
                p = q[name][a]
                same(saved[name + "_up"], p)
                check(
                    r["direction_changes"][name]
                    == int(np.sum((p >= 0.5) != (controls["base"] >= 0.5))),
                    "direction preservation diagnostic",
                )
                combined = np.column_stack(((1 - flat) * (1 - p), flat, (1 - flat) * p))
                verify_metrics(
                    r[name],
                    y[a, indices],
                    combined,
                    prior["frequency"],
                    prior["prior_reversal"],
                    t[indices],
                )
            for c in ("base", "unshrunk", "raw", "flow", "neutral"):
                conditional(
                    {
                        k.replace("vs_control", "vs_spot"): v
                        for k, v in r["comparisons"][f"adjusted_vs_{c}"].items()
                    },
                    y[a, indices],
                    q["adjusted"][a],
                    controls[c],
                    prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1]),
                    t[indices],
                )
            for name, p in controls.items():
                reliability_check(r["calibration"][name], y[a, indices], p)
            cases += len(indices)
    check(
        len(base_fits) == base_count == 27
        and len(fits) == cal_count == 21
        and len(results["cohorts"]) == 9
        and cases == 550362,
        "all fit/cohort/anchor counts",
    )
    calibration = all(
        r["comparisons"][f"adjusted_vs_{c}"]["information_gate"]
        for r in results["cohorts"].values()
        for c in ("base", "unshrunk", "neutral")
    )
    information = calibration and all(
        r["comparisons"]["adjusted_vs_flow"]["information_gate"]
        for r in results["cohorts"].values()
    )
    direction = all(r["adjusted"]["gate"] for r in results["cohorts"].values())
    check(
        results["all_calibration_gates"] == calibration
        and results["all_information_gates"] == information
        and results["all_direction_gates"] == direction
        and results["general_solution"] == (information and direction),
        "combined registered gates",
    )
    return dict(
        verified=True,
        base_fits=base_count,
        calibration_fits=cal_count,
        cases=cases,
        cohorts=9,
        maximum_optimality_residual=maximum,
        protected_data_opened=False,
        general_solution=results["general_solution"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "reports/timestral-market-calibration-20261027",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), sort_keys=True))
