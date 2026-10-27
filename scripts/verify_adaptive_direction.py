#!/usr/bin/env python3
"""Independent reconstruction of daily calibration, optimality and paired scores.

No production model or runner imports, no optimization, no new source data.
"""

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

SOURCE = ROOT / "reports/timestral-universal-20261027"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calibrated(p, a, b):
    odds = (p[:, 2] / p[:, 0]) ** a
    flat_odds = p[:, 1] / (1 - p[:, 1]) * np.exp(b)
    flat = flat_odds / (1 + flat_odds)
    down = (1 - flat) / (1 + odds)
    return np.column_stack((down, flat, down * odds))


def kkt(gradient, parameter, lo, hi):
    check(lo <= parameter <= hi, "parameter bounds")
    residual = (
        max(0, -gradient)
        if parameter == lo
        else max(0, gradient)
        if parameter == hi
        else abs(gradient)
    )
    check(residual < 1e-9, "convex calibration optimality")
    return residual


def paired(stored, y, p, base, t):
    n = len(y)
    # Expanded squared norm is independent of the production one-hot formula.
    bp = (base * base).sum(1) - 2 * base[np.arange(n), y]
    pp = (p * p).sum(1) - 2 * p[np.arange(n), y]
    delta = np.column_stack(
        (bp - pp, np.log(p[np.arange(n), y] / base[np.arange(n), y]))
    )
    days, inv = np.unique(t // 86400, return_inverse=True)
    check(np.all(np.diff(days) == 1), "contiguous days")
    totals = np.zeros((len(days), 3))
    np.add.at(totals, inv, np.column_stack((np.ones(n), delta)))
    starts = np.random.default_rng(271031).integers(
        0, len(days) - 6, (4000, (len(days) + 6) // 7)
    )
    blocks = np.concatenate([starts[:, :, None] + j for j in range(7)], axis=2).reshape(
        4000, -1
    )[:, : len(days)]
    aggregate = totals[blocks].sum(1)
    ci = np.quantile(aggregate[:, 1:] / aggregate[:, :1], [1 / 360, 359 / 360], axis=0)
    same(stored["brier_gain"], delta[:, 0].mean())
    same(stored["logloss_gain"], delta[:, 1].mean())
    same(stored["brier_gain_ci99_444"], ci[:, 0])
    same(stored["logloss_gain_ci99_444"], ci[:, 1])
    selected = (p[:, 0] >= 0.52) | (p[:, 2] >= 0.52)
    if selected.any():
        same(
            stored["selected_confidence_minus_accuracy"],
            p.max(1)[selected].mean() - (p.argmax(1)[selected] == y[selected]).mean(),
        )
    else:
        check(stored["selected_confidence_minus_accuracy"] is None, "empty selection")
    check(
        stored["probability_gate"] == bool(np.all(ci[0] > 0)),
        "probability improvement gate",
    )


def verify(root):
    protocol = json.loads((root / "protocol.json").read_text())
    check(
        digest(root / "protocol.json")
        == json.loads((root / "protocol.sha256.json").read_text())["sha256"],
        "protocol hash",
    )
    check(
        protocol["at"] < json.loads((root / "opened.json").read_text())["at"],
        "registration chronology",
    )
    for name, sha in json.loads((root / "manifest.json").read_text()).items():
        check(digest(root / name) == sha, f"artifact hash {name}")
    for name, sha in {**protocol["code"], **protocol["inputs"]}.items():
        check(digest(ROOT / name) == sha, f"input or code hash {name}")
    states = json.loads((root / "states.json").read_text())
    results = json.loads((root / "results.json").read_text())
    diagnostics = json.loads((root / "diagnostics.json").read_text())["cohorts"]
    fits, max_residual = 0, 0.0
    expected_keys = {f"{year}-{s}" for year in (2024, 2025, 2026) for s in ASSETS}
    check(
        set(states) == set(results["cohorts"]) == expected_keys,
        "all nine cohorts required",
    )
    for year in (2024, 2025, 2026):
        for symbol in ASSETS:
            key = f"{year}-{symbol}"
            folder, prefix = (
                (SOURCE, symbol) if year == 2026 else (SOURCE / "history", key)
            )
            source = dict(np.load(folder / f"{prefix}-predictions.npz"))
            outcome = dict(np.load(folder / f"{prefix}-outcomes.npz"))
            stored = dict(np.load(root / f"{key}-predictions.npz"))
            t, end, y = (outcome[k] for k in ("anchor", "end", "label"))
            raw = source["cross" if year == 2026 else "probability"]
            same(t, stored["anchor"])
            same(t, source["anchor"])
            same(end, t + 360)
            check(
                end.max() <= 1785542400 and np.all(np.diff(t) == 300),
                "protected outcomes or gaps",
            )
            months = t.astype("datetime64[s]").astype("datetime64[M]").astype(str)
            check(
                all(f"{year}-01" <= m <= f"{year}-07" for m in np.unique(months)),
                "cohort scope",
            )
            days = np.unique(t // 86400) * 86400
            rows = states[key]
            same([f["cutoff"] for f in rows], days[7:])
            adaptive, fixed = raw.copy(), raw.copy()
            for f in rows:
                cutoff = f["cutoff"]
                mask = (end <= cutoff) & (t >= cutoff - 28 * 86400)
                check(
                    f["n"] == mask.sum() and f["n"] >= 1000, "past-only training count"
                )
                check(
                    f["max_end"] == int(end[mask].max()) <= cutoff,
                    "matured outcome bound",
                )
                for values, field in (
                    (t[mask], "ids_sha256"),
                    (y[mask], "labels_sha256"),
                ):
                    check(
                        hashlib.sha256(values.astype("<i8").tobytes()).hexdigest()
                        == f[field],
                        "calibration membership hash",
                    )
                weights = np.power(2.0, (end[mask] - cutoff) / 604800)
                weights /= weights.sum()
                p, labels = raw[mask], y[mask]
                x = np.log(p[:, 2]) - np.log(p[:, 0])
                a, b = f["alpha"], f["offset"]
                ga = np.dot(
                    weights, (labels != 1) * x * (sigmoid(a * x) - (labels == 2))
                ) + 0.0002 * (a - 1)
                gb = (
                    np.dot(
                        weights,
                        sigmoid(np.log(p[:, 1] / (1 - p[:, 1])) + b) - (labels == 1),
                    )
                    + 0.0002 * b
                )
                max_residual = max(max_residual, kkt(ga, a, 0, 1), kkt(gb, b, -3, 3))
                ix = (t >= cutoff) & (t < cutoff + 86400)
                adaptive[ix] = calibrated(raw[ix], a, b)
                fixed[ix] = calibrated(raw[ix], rows[0]["alpha"], rows[0]["offset"])
                fits += 1
            same(adaptive, stored["adaptive"])
            same(fixed, stored["fixed"])
            mixed = np.load(root / "mixture-refined" / f"{key}-predictions.npz")[
                "adaptive"
            ]
            for name, q in (
                ("raw", raw),
                ("calibration", adaptive),
                ("mixture", mixed),
            ):
                diag = diagnostics[key][name]
                flat = y == 1
                flat_loss = -np.log(np.where(flat, q[:, 1], 1 - q[:, 1])).mean()
                total_loss = -np.log(q[np.arange(len(y)), y]).mean()
                same(diag["flat_logloss"], flat_loss)
                same(diag["total_logloss"], total_loss)
                same(diag["direction_logloss_per_anchor"], total_loss - flat_loss)
                selected = (raw[:, 0] >= 0.52) | (raw[:, 2] >= 0.52)
                raw_class = raw.argmax(1)
                same(
                    diag["confidence_gap_on_raw_selected_class"],
                    q[np.arange(len(y)), raw_class][selected].mean()
                    - (raw_class == y)[selected].mean(),
                )
            r = results["cohorts"][key]
            for name, p in (("raw", raw), ("fixed", fixed), ("adaptive", adaptive)):
                verify_metrics(
                    r[name], y, p, source["frequency"], source["prior_reversal"], t
                )
            paired(r["adaptive_vs_raw"], y, adaptive, raw, t)
            paired(r["adaptive_vs_fixed"], y, adaptive, fixed, t)
            paired(r["fixed_vs_raw"], y, fixed, raw, t)
    cohorts = results["cohorts"].values()
    for name, method, flag in (
        ("all_direction_gates", "adaptive", "gate"),
        ("all_probability_gates", "adaptive_vs_raw", "probability_gate"),
        ("all_adaptation_gates", "adaptive_vs_fixed", "probability_gate"),
    ):
        check(results[name] == all(r[method][flag] for r in cohorts), "combined gate")
    return {
        "verified": True,
        "cohorts": 9,
        "daily_fits": fits,
        "max_kkt_residual": max_residual,
        "protected_data_opened": False,
        "optimization_rerun": False,
        "all_direction_gates": results["all_direction_gates"],
        "all_probability_gates": results["all_probability_gates"],
        "all_adaptation_gates": results["all_adaptation_gates"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "reports/timestral-adaptive-direction-20261027",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), sort_keys=True))
