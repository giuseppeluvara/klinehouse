#!/usr/bin/env python3
"""Independent asset-agreement, flat accounting, fit and score audit."""

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
from scripts.verify_market_calibration import membership, reliability_check  # noqa: E402
from scripts.verify_uncertain_market import adjustment_check  # noqa: E402

BASE = ROOT / "reports/timestral-uncertain-market-20261027"
FUTURES = ROOT / "reports/timestral-futures-flow-20261027"
PAIRS = (
    ("consensus", "common"),
    ("consensus", "blend"),
    ("consensus", "neutral"),
    ("consensus", "flow"),
    ("blend", "common"),
    ("blend", "neutral"),
    ("blend", "flow"),
)


def diagnostic_check(d, y, q, f, c, o):
    sel = ((1 - f) * q >= 0.52) | ((1 - f) * (1 - q) >= 0.52)
    moving = y != 1
    correct = (q >= 0.5) == (y == 2)
    own_correct = (o >= 0) == (y == 2)
    check(
        d["n"] == len(y) and d["selected"] == int(sel.sum()),
        "diagnostic all/selected counts",
    )
    values = dict(
        flat_rate=np.mean(~moving),
        flat_probability=np.mean(f),
        selected_flat_rate=np.sum(sel & ~moving) / sel.sum(),
        selected_binary_accuracy=np.sum(sel & moving & correct) / np.sum(sel & moving),
        selected_all_accuracy=np.sum(sel & moving & correct) / sel.sum(),
        selected_mean_flat_probability=np.mean(f[sel]),
    )
    for name, value in values.items():
        same(d[name], value)
    for name, group in (
        ("agree", (c * o) > 0),
        ("disagree", (c * o) < 0),
        ("zero", (c * o) == 0),
    ):
        s = d["groups"][name]
        use = group & moving
        chosen = use & sel
        check(
            s["all_n"] == int(group.sum())
            and s["nonflat_n"] == int(use.sum())
            and s["selected_nonflat_n"] == int(chosen.sum()),
            "diagnostic group counts",
        )
        for key, hits, mask in (
            ("common_accuracy", correct, use),
            ("own_accuracy", own_correct, use),
            ("selected_common_accuracy", correct, chosen),
        ):
            if mask.any():
                same(s[key], np.sum(mask & hits) / mask.sum())
            else:
                check(s[key] is None, "empty diagnostic cell")


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
        "registration before evaluation",
    )
    arrays = {
        f"{year}-{s}": dict(np.load(FUTURES / f"{year}-{s}-futures.npz"))
        for year in (2024, 2025, 2026)
        for s in ASSETS
    }
    for d in arrays.values():
        check(d["time"].max() < 1785542400, "protected sources")
    fits = json.loads((root / "fits.json").read_text())
    results = json.loads((root / "results.json").read_text())
    diagnosis = json.loads((root / "diagnosis.json").read_text())
    maximum, count, cases = 0.0, 0, 0
    for year in (2024, 2025, 2026):
        t, end, x, y, available = features(FUTURES, year, arrays)
        common = np.maximum(-3, np.minimum(3, -x[:, :, 3] / 3 - 2 * x[:, :, 16] / 3))
        own = np.maximum(-3, np.minimum(3, -x[:, :, 3]))
        z = 0.5 * common + 0.5 * own
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
                    s["adjusted"],
                    s["raw"],
                    z[:, train].reshape(-1)[moving],
                    (labels[moving] == 2).astype(int),
                    membership(t, train)[moving],
                    cutoff,
                ),
            )
            test = (t >= cutoff) & (t < epoch(year, month + 1))
            pieces.append(1 / (1 + np.exp(-s["adjusted"]["beta"] * z[:, test])))
            indices.append(np.flatnonzero(test))
            count += 1
        indices = np.concatenate(indices)
        blend = np.concatenate(pieces, axis=1)
        for a, symbol in enumerate(ASSETS):
            key = f"{year}-{symbol}"
            prefix, folder = (
                (symbol, SOURCE) if year == 2026 else (key, SOURCE / "history")
            )
            prior = dict(np.load(folder / f"{prefix}-predictions.npz"))
            outcome = dict(np.load(folder / f"{prefix}-outcomes.npz"))
            old = dict(np.load(BASE / f"{key}-predictions.npz"))
            flow = dict(np.load(FUTURES / f"{key}-predictions.npz"))
            saved = dict(np.load(root / f"{key}-predictions.npz"))
            for d in (prior, outcome, old, flow, saved):
                same(d["anchor"], t[indices])
            same(outcome["label"], y[a, indices])
            same(outcome["end"], end[indices])
            check(end[indices].max() <= 1785542400, "protected outcomes")
            c, o = common[a, indices], own[a, indices]
            agree = ((c > 0) & (o > 0)) | ((c < 0) & (o < 0))
            consensus = np.full(len(indices), 0.5)
            consensus[agree] = old["adaptive_up"][agree]
            q = dict(
                common=old["adaptive_up"],
                blend=blend[a],
                consensus=consensus,
                flow=flow["flow_up"],
                neutral=np.full(len(indices), 0.5),
            )
            r = results["cohorts"][key]
            flat = prior["cross" if year == 2026 else "probability"][:, 1]
            check(
                r["consensus_abstentions"] == int((~agree).sum()),
                "fixed input-only abstentions",
            )
            diagnostic_check(diagnosis[key], y[a, indices], q["common"], flat, c, o)
            for name in ("consensus", "blend"):
                p = q[name]
                same(saved[name + "_up"], p)
                check(
                    r["direction_changes"][name]
                    == int(np.sum((p >= 0.5) != (q["common"] >= 0.5))),
                    "direction changes",
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
            for candidate, control in PAIRS:
                conditional(
                    {
                        k.replace("vs_control", "vs_spot"): v
                        for k, v in r["comparisons"][
                            f"{candidate}_vs_{control}"
                        ].items()
                    },
                    y[a, indices],
                    q[candidate],
                    q[control],
                    prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1]),
                    t[indices],
                )
            for name, p in q.items():
                reliability_check(r["calibration"][name], y[a, indices], p)
            cases += len(indices)
    check(
        len(fits) == count == 21
        and len(results["cohorts"]) == len(diagnosis) == 9
        and cases == 550362,
        "all fits/cohorts/cases",
    )
    information = all(
        r["comparisons"][f"consensus_vs_{c}"]["information_gate"]
        for r in results["cohorts"].values()
        for c in ("common", "neutral", "flow")
    )
    direction = all(r["consensus"]["gate"] for r in results["cohorts"].values())
    check(
        results["all_information_gates"] == information
        and results["all_direction_gates"] == direction
        and results["general_solution"] == (information and direction),
        "combined registered gates",
    )
    return dict(
        verified=True,
        blend_fits=count,
        cohorts=9,
        cases=cases,
        maximum_raw_fit_residual=maximum,
        protected_data_opened=False,
        general_solution=results["general_solution"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "reports/timestral-market-specific-20261027"
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), sort_keys=True))
