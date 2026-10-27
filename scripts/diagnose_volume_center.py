#!/usr/bin/env python3
"""Post-hoc loss decomposition of frozen predictions; no model selection."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/timestral-volume-center-20261027"
PRIOR = ROOT / "reports/timestral-universal-20261027"


def diagnose():
    scores = json.loads((REPORT / "results.json").read_text())["cohorts"]
    result = {}
    for key, cohort in scores.items():
        year, symbol = key.split("-")
        folder = PRIOR if year == "2026" else PRIOR / "history"
        name = symbol if year == "2026" else key
        prior = dict(np.load(folder / f"{name}-predictions.npz"))
        outcome = dict(np.load(folder / f"{name}-outcomes.npz"))
        saved = dict(np.load(REPORT / f"{key}-predictions.npz"))
        np.testing.assert_array_equal(prior["anchor"], saved["anchor"])
        np.testing.assert_array_equal(outcome["anchor"], saved["anchor"])
        assert outcome["end"].max() <= 1785542400
        y = outcome["label"]
        f = prior["cross" if year == "2026" else "probability"][:, 1]
        bf = prior["frequency"][:, 1]
        bq = prior["frequency"][:, 2] / (1 - bf)
        flat = y == 1
        result[key] = {}
        for arm in ("uniform", "volume"):
            q = saved[arm + "_up"]
            select = (1 - f) * np.maximum(q, 1 - q) >= 0.52
            binary = -np.log(np.where(y == 2, q, 1 - q))
            base_binary = -np.log(np.where(y == 2, bq, 1 - bq))
            flat_loss = -np.log(np.where(flat, f, 1 - f))
            base_flat_loss = -np.log(np.where(flat, bf, 1 - bf))
            directional = float(((~flat) * binary)[select].mean())
            base_directional = float(((~flat) * base_binary)[select].mean())
            flat_component = float(flat_loss[select].mean())
            base_flat_component = float(base_flat_loss[select].mean())
            np.testing.assert_allclose(
                directional + flat_component, cohort[arm]["log_loss"], atol=1e-12
            )
            np.testing.assert_allclose(
                base_directional + base_flat_component,
                cohort[arm]["baseline_log_loss"],
                atol=1e-12,
            )
            result[key][arm] = dict(
                selected=int(select.sum()),
                selected_flats=int(np.sum(select & flat)),
                direction_gain=base_directional - directional,
                flat_gain=base_flat_component - flat_component,
                total_gain=cohort[arm]["baseline_log_loss"] - cohort[arm]["log_loss"],
            )
    return dict(
        status="Post-hoc descriptive accounting after candidate outcomes, not used to alter candidates or gates; gains are baseline loss minus model loss on each candidate's selected rows.",
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        results_sha256=hashlib.sha256(
            (REPORT / "results.json").read_bytes()
        ).hexdigest(),
        cohorts=result,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = diagnose()
    if args.verify:
        saved = json.loads((REPORT / "loss-decomposition.json").read_text())
        for key in ("status", "code_sha256", "results_sha256"):
            assert saved[key] == result[key]
        assert saved["cohorts"].keys() == result["cohorts"].keys()
        for key, arms in result["cohorts"].items():
            assert saved["cohorts"][key].keys() == arms.keys()
            for arm, values in arms.items():
                assert saved["cohorts"][key][arm].keys() == values.keys()
                for name, value in values.items():
                    np.testing.assert_allclose(
                        saved["cohorts"][key][arm][name], value, rtol=1e-9, atol=1e-12
                    )
        print(json.dumps(dict(verified=True, cohorts=9, posthoc=True)))
    else:
        print(json.dumps(result, sort_keys=True, indent=2))
