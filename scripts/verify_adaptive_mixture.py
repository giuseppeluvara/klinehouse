#!/usr/bin/env python3
"""Independent convex-optimality and prediction audit for expert mixtures."""

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
from scripts.verify_adaptive_direction import paired, SOURCE  # noqa: E402


def verify(root):
    failed = root.parent / "mixture"
    if root.name == "mixture-refined":
        prior = json.loads((failed / "protocol.json").read_text())
        check(
            digest(failed / "protocol.json")
            == json.loads((failed / "protocol.sha256.json").read_text())["sha256"],
            "failed-attempt protocol",
        )
        for name, sha in {**prior["code"], **prior["inputs"]}.items():
            check(digest(ROOT / name) == sha, "failed-attempt frozen code and inputs")
        check(
            not (failed / "results.json").exists(),
            "no fabricated failed-attempt scores",
        )
        check(
            json.loads((failed / "failure.json").read_text())["status"]
            == "aborted_before_cohort_metrics",
            "numerical failure retained",
        )
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
    check(
        protocol["previous_results_sha256"] == digest(root.parent / "results.json"),
        "follow-up ancestry",
    )
    for name, sha in json.loads((root / "manifest.json").read_text()).items():
        check(digest(root / name) == sha, f"artifact hash {name}")
    for name, sha in {**protocol["code"], **protocol["inputs"]}.items():
        check(digest(ROOT / name) == sha, f"input or code hash {name}")
    states = json.loads((root / "states.json").read_text())
    results = json.loads((root / "results.json").read_text())
    keys = {f"{year}-{s}" for year in (2024, 2025, 2026) for s in ASSETS}
    check(set(states) == set(results["cohorts"]) == keys, "all9 cohorts")
    count, maximum = 0, 0.0
    for year in (2024, 2025, 2026):
        for s in ASSETS:
            key = f"{year}-{s}"
            folder, prefix = (SOURCE, s) if year == 2026 else (SOURCE / "history", key)
            source = dict(np.load(folder / f"{prefix}-predictions.npz"))
            outcome = dict(np.load(folder / f"{prefix}-outcomes.npz"))
            trees = dict(np.load(SOURCE / "trees" / f"{key}-predictions.npz"))
            saved = dict(np.load(root / f"{key}-predictions.npz"))
            t, end, y = (outcome[k] for k in ("anchor", "end", "label"))
            raw = source["cross" if year == 2026 else "probability"]
            for other in (source, trees, saved):
                same(t, other["anchor"])
            same(end, t + 360)
            check(
                end.max() <= 1785542400 and np.all(np.diff(t) == 300),
                "protected outcome or gap",
            )
            check(
                np.all(
                    t.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
                    == year
                ),
                "cohort year",
            )
            experts = [
                raw,
                trees["probability"],
                source["prior_reversal"],
                source["frequency"],
            ]
            rows = states[key]
            days = np.unique(t // 86400) * 86400
            same([r["cutoff"] for r in rows], days[7:])
            adaptive, fixed = raw.copy(), raw.copy()
            for row in rows:
                cutoff = row["cutoff"]
                train = (end <= cutoff) & (t >= cutoff - 28 * 86400)
                check(row["n"] == train.sum() and row["n"] >= 1000, "training scope")
                check(
                    row["max_end"] == int(end[train].max()) <= cutoff,
                    "outcome maturity",
                )
                for a, name in ((t[train], "ids_sha256"), (y[train], "labels_sha256")):
                    check(
                        hashlib.sha256(a.astype("<i8").tobytes()).hexdigest()
                        == row[name],
                        "training fingerprint",
                    )
                w = np.array(row["weights"])
                same(w.sum(), 1.0)
                check(w.shape == (4,) and np.all(w >= 1e-6 - 1e-12), "simplex bounds")
                decay = np.exp((end[train] - cutoff) * np.log(2) / 604800)
                decay /= decay.sum()
                p = np.column_stack(
                    [e[train][np.arange(train.sum()), y[train]] for e in experts]
                )
                mix = np.sum(p * w, axis=1)
                gradient = -np.sum(p * (decay / mix)[:, None], axis=0) + 0.0001 * (
                    np.log(4 * w) + 1
                )
                free = w > 1e-6 + 1e-8
                multiplier = gradient[free].mean()
                residuals = np.where(
                    free,
                    abs(gradient - multiplier),
                    np.maximum(multiplier - gradient, 0),
                )
                residual = float(residuals.max())
                check(residual <= 2e-6, "convex mixture KKT")
                same(residual, row["kkt_residual"])
                maximum = max(maximum, residual)
                current = (t >= cutoff) & (t < cutoff + 86400)
                adaptive[current] = sum(
                    w[j] * e[current] for j, e in enumerate(experts)
                )
                fixed[current] = sum(
                    rows[0]["weights"][j] * e[current] for j, e in enumerate(experts)
                )
                count += 1
            same(adaptive, saved["adaptive"])
            same(fixed, saved["fixed"])
            r = results["cohorts"][key]
            for name, p in (("raw", raw), ("fixed", fixed), ("adaptive", adaptive)):
                verify_metrics(
                    r[name], y, p, source["frequency"], source["prior_reversal"], t
                )
            for name, p, base in (
                ("adaptive_vs_raw", adaptive, raw),
                ("adaptive_vs_fixed", adaptive, fixed),
                ("fixed_vs_raw", fixed, raw),
            ):
                paired(r[name], y, p, base, t)
    for name, method, flag in (
        ("all_direction_gates", "adaptive", "gate"),
        ("all_probability_gates", "adaptive_vs_raw", "probability_gate"),
        ("all_adaptation_gates", "adaptive_vs_fixed", "probability_gate"),
    ):
        check(
            results[name] == all(r[method][flag] for r in results["cohorts"].values()),
            "combined criterion",
        )
    return {
        "verified": True,
        "daily_fits": count,
        "cohorts": 9,
        "max_kkt_residual": maximum,
        "protected_data_opened": False,
        "optimization_rerun": False,
        **{k: v for k, v in results.items() if k != "cohorts"},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "reports/timestral-adaptive-direction-20261027/mixture-refined",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), sort_keys=True))
