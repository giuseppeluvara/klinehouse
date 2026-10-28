#!/usr/bin/env python3
"""Post-hoc accounting of learning loss against a hindsight-only comparator."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_universal_direction import ROOT, ASSETS, check, same, digest  # noqa: E402
from scripts.verify_futures_flow import SOURCE, epoch  # noqa: E402

DEFAULT = ROOT / "reports/timestral-multiscore-20261028"
COMMON = ROOT / "reports/timestral-uncertain-market-20261027"
CENTER = ROOT / "reports/timestral-volume-center-20261027"
LOGLOSS = ROOT / "reports/timestral-logloss-aggregate-20261028"
EXPERTS = (
    "common_old",
    "common_mixed",
    "uniform_old",
    "uniform_mixed",
    "equal",
    "logloss_adaptive",
)
BOUNDARY = 1785542400
METHOD = {
    "status": "Explicitly post-hoc descriptive diagnosis after the frozen experiment; not a new candidate or registered evaluation.",
    "comparator": "For each cohort and score, select the fixed expert with the lowest total loss over the entire January-July period. This uses future outcomes and is a hindsight oracle for accounting only, not an admissible forecasting rule.",
    "excess": "Defensive loss minus oracle expert loss; positive is worse. Monthly excesses and January plus February-July excesses add to the whole period.",
    "windows": "First seven UTC calendar days are nested within January. February-July is the remaining period. Fractions of total excess may exceed one or be negative if later periods offset earlier losses; they are not probabilities.",
    "limits": "This decomposition cannot establish that a warm start, different prior, reset, or shorter memory improves future performance. No new candidate, parameter sweep or tuning is performed. The already-explored cohorts are not independent confirmation.",
    "protected": "Only previously admitted saved predictions and outcomes are read; no source or outcome from 2026-08-01 onward.",
}


def calculate(root):
    inputs = {}

    def read(path):
        inputs[str(path.relative_to(ROOT))] = digest(path)
        return dict(np.load(path))

    evidence_path = root / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    inputs[str(evidence_path.relative_to(ROOT))] = digest(evidence_path)
    cohorts = {}
    for year in (2024, 2025, 2026):
        for symbol in ASSETS:
            key = f"{year}-{symbol}"
            folder, prefix = (
                (SOURCE, symbol) if year == 2026 else (SOURCE / "history", key)
            )
            prior = read(folder / f"{prefix}-predictions.npz")
            outcome = read(folder / f"{prefix}-outcomes.npz")
            common = read(COMMON / f"{key}-predictions.npz")
            center = read(CENTER / f"{key}-predictions.npz")
            logloss = read(LOGLOSS / f"{key}-predictions.npz")
            saved = read(root / f"{key}-predictions.npz")
            t, y = outcome["anchor"], outcome["label"]
            for d in (prior, common, center, logloss, saved):
                same(t, d["anchor"])
            check(
                t.min() >= epoch(year, 1)
                and t.max() < epoch(year, 8)
                and np.all(np.diff(t) >= 300)
                and np.all(outcome["end"] == t + 360)
                and outcome["end"].max() <= BOUNDARY,
                "admitted ordered anchors and protected target boundary",
            )
            flat = prior["cross" if year == 2026 else "probability"][:, 1]
            forecasts = []
            for q in (common["adaptive_up"], center["uniform_up"]):
                for f in (flat, (flat + prior["frequency"][:, 1]) / 2):
                    forecasts.append(
                        np.column_stack(((1 - f) * (1 - q), f, (1 - f) * q))
                    )
            forecasts.extend((np.mean(forecasts, axis=0), logloss["adaptive"]))
            forecasts.append(saved["defensive"])
            p = np.stack(forecasts, axis=1)
            check(np.all(np.isfinite(p) & (p > 0) & (p < 1)), "interior forecasts")
            np.testing.assert_allclose(p.sum(axis=2), 1, atol=1e-12, rtol=0)
            losses = {
                "logloss": -np.log(
                    p[np.arange(len(y))[:, None], np.arange(7)[None, :], y[:, None]]
                ),
                "brier": np.square(p - np.eye(3)[y, None, :]).sum(axis=2),
            }
            periods = {
                "first_7_days": t < epoch(year, 1) + 7 * 86400,
                "january": t < epoch(year, 2),
                "february_to_july": t >= epoch(year, 2),
            }
            months = t.astype("datetime64[s]").astype("datetime64[M]")
            scores = {}
            for name, loss in losses.items():
                totals = loss.sum(axis=0)
                selected = int(np.argmin(totals[:6]))
                difference = loss[:, -1] - loss[:, selected]
                total_excess = float(difference.sum())
                cert = evidence[key]["certificate"]
                same(totals[:6], cert["expert_totals"][name])
                same(totals[-1], cert["learner_totals"][name])
                same(total_excess, cert["regrets"][name])

                def summarize(mask):
                    amount = float(difference[mask].sum())
                    return dict(
                        n=int(mask.sum()),
                        excess=amount,
                        mean_excess=float(difference[mask].mean()),
                        fraction_of_total=None
                        if total_excess == 0
                        else amount / total_excess,
                    )

                windows = {name: summarize(mask) for name, mask in periods.items()}
                monthly = {
                    str(month): summarize(months == month)
                    for month in np.unique(months)
                }
                same(sum(v["excess"] for v in monthly.values()), total_excess)
                same(
                    windows["january"]["excess"]
                    + windows["february_to_july"]["excess"],
                    total_excess,
                )
                scores[name] = dict(
                    oracle_expert=EXPERTS[selected],
                    oracle_total=float(totals[selected]),
                    learner_total=float(totals[-1]),
                    total_excess=total_excess,
                    windows=windows,
                    monthly=monthly,
                )
            cohorts[key] = dict(n=len(y), scores=scores)
    check(sum(v["n"] for v in cohorts.values()) == 550362, "all original cases")
    return dict(method=METHOD, inputs=inputs, cohorts=cohorts)


def compare(saved, expected, path="diagnosis"):
    if isinstance(expected, dict):
        check(isinstance(saved, dict) and set(saved) == set(expected), f"keys {path}")
        for key, value in expected.items():
            compare(saved[key], value, f"{path}.{key}")
    elif isinstance(expected, (float, int)):
        same(saved, expected)
    else:
        check(saved == expected, f"value {path}")


def main(root, verify):
    path = root / "learning-diagnosis.json"
    expected = calculate(root)
    if verify:
        saved = json.loads(path.read_text())
        when = saved.pop("at")
        check(
            when > json.loads((root / "opened.json").read_text())["at"],
            "explicitly post-hoc chronology",
        )
        compare(saved, expected)
        print(json.dumps(dict(verified=True, posthoc=True, cohorts=9, cases=550362)))
    else:
        expected["at"] = datetime.now(timezone.utc).isoformat()
        with path.open("x") as f:
            json.dump(expected, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        print(
            json.dumps(
                dict(
                    saved=str(path.relative_to(ROOT)),
                    posthoc=True,
                    cohorts=9,
                    cases=550362,
                )
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    main(args.root, args.verify)
