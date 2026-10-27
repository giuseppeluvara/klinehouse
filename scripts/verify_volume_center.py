#!/usr/bin/env python3
"""Independent convolution-based center, fit membership and score audit."""

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
from scripts.verify_futures_flow import features, epoch, SOURCE, MICRO  # noqa: E402
from scripts.verify_premium_direction import conditional  # noqa: E402
from scripts.verify_market_calibration import membership, reliability_check  # noqa: E402
from scripts.verify_uncertain_market import adjustment_check  # noqa: E402

BASE = ROOT / "reports/timestral-uncertain-market-20261027"
FUTURES = ROOT / "reports/timestral-futures-flow-20261027"
PAIRS = (
    ("volume", "common"),
    ("volume", "uniform"),
    ("volume", "neutral"),
    ("volume", "flow"),
    ("uniform", "common"),
)


def reconstruct_centers(year, anchors):
    uniform, volume = [], []
    for a, symbol in enumerate(ASSETS):
        path = (
            SOURCE / "history" / f"{symbol}-minutes.npz"
            if year < 2026
            else MICRO
            / ("micro/minutes.npz" if a == 0 else f"transfer/{symbol}-minutes.npz")
        )
        source = dict(np.load(path))
        check(source["time"].max() < 1785542400, "protected spot source")
        take = (source["time"] >= epoch(year - 1, 10)) & (
            source["time"] < epoch(year, 8)
        )
        times = source["time"][take]
        log = np.log(source["close"][take])
        log -= log[0]
        v = source["volume"][take]
        i = np.searchsorted(times, anchors - 60)
        check(np.all(i >= 61) and np.all(i < len(times)), "full source history")
        same(times[i] + 60, anchors)
        check(np.all(np.diff(times) == 60), "complete spot context")
        r = np.r_[0.0, np.diff(log)]
        rms = np.sqrt(
            np.maximum(np.convolve(r * r, np.ones(60) / 60, "valid")[i - 60], 1e-16)
        )
        u = np.convolve(log, np.ones(60) / 60, "valid")[i - 59]
        total = np.convolve(v, np.ones(60), "valid")[i - 59]
        center = np.divide(
            np.convolve(v * log, np.ones(60), "valid")[i - 59],
            total,
            out=u.copy(),
            where=total > 0,
        )
        uniform.append(
            np.maximum(-8, np.minimum(8, 2 * (u - log[i]) / (rms * np.sqrt(60))))
        )
        volume.append(
            np.maximum(-8, np.minimum(8, 2 * (center - log[i]) / (rms * np.sqrt(60))))
        )
    return {
        "uniform": np.maximum(-3, np.minimum(3, np.sum(uniform, axis=0) / 3)),
        "volume": np.maximum(-3, np.minimum(3, np.sum(volume, axis=0) / 3)),
    }


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
        check(digest(ROOT / name) == sha, f"input/code hash {name}")
    check(
        protocol["at"] < json.loads((root / "opened.json").read_text())["at"],
        "registration chronology",
    )
    arrays = {
        f"{year}-{s}": dict(np.load(FUTURES / f"{year}-{s}-futures.npz"))
        for year in (2024, 2025, 2026)
        for s in ASSETS
    }
    for d in arrays.values():
        check(d["time"].max() < 1785542400, "protected futures source")
    fits = json.loads((root / "fits.json").read_text())
    results = json.loads((root / "results.json").read_text())
    maximum, count, cases, agreement_cases = 0.0, 0, 0, 0
    for year in (2024, 2025, 2026):
        t, end, x, y, available = features(FUTURES, year, arrays)
        center = reconstruct_centers(year, t)
        common = np.maximum(-3, np.minimum(3, -x[:, :, 3] / 3 - 2 * x[:, :, 16] / 3))
        own = -x[:, :, 3]
        agree = ((common > 0) & (own > 0)) | ((common < 0) & (own < 0))
        z = {}
        for name, c in center.items():
            z[name] = common.copy()
            z[name][agree] = np.broadcast_to(c, common.shape)[agree]
        pieces = {name: [] for name in center}
        indices = []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (t >= cutoff - 90 * 86400) & (end <= cutoff) & available
            labels = y[:, train].reshape(-1)
            moving = labels != 1
            test = (t >= cutoff) & (t < epoch(year, month + 1))
            for name in center:
                state = fits[f"{year}-{month}-{name}"]
                maximum = max(
                    maximum,
                    adjustment_check(
                        state["adjusted"],
                        state["raw"],
                        z[name][:, train].reshape(-1)[moving],
                        (labels[moving] == 2).astype(int),
                        membership(t, train)[moving],
                        cutoff,
                    ),
                )
                pieces[name].append(
                    1 / (1 + np.exp(-state["adjusted"]["beta"] * z[name][:, test]))
                )
                count += 1
            indices.append(np.flatnonzero(test))
        indices = np.concatenate(indices)
        predictions = {name: np.concatenate(p, axis=1) for name, p in pieces.items()}
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
            check(end[indices].max() <= 1785542400, "protected targets")
            q = {name: p[a] for name, p in predictions.items()}
            q.update(
                common=old["adaptive_up"],
                flow=flow["flow_up"],
                neutral=np.full(len(indices), 0.5),
            )
            agreement = agree[a, indices]
            same(saved["agreement"], agreement)
            r = results["cohorts"][key]
            check(r["agreement"]["n"] == int(agreement.sum()), "agreement membership")
            flat = prior["cross" if year == 2026 else "probability"][:, 1]
            frequency = prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1])
            for name in center:
                p = q[name]
                same(saved[name + "_up"], p)
                check(
                    r["direction_changes"][name]
                    == int(np.sum((p >= 0.5) != (q["common"] >= 0.5))),
                    "direction counts",
                )
                verify_metrics(
                    r[name],
                    y[a, indices],
                    np.column_stack(((1 - flat) * (1 - p), flat, (1 - flat) * p)),
                    prior["frequency"],
                    prior["prior_reversal"],
                    t[indices],
                )
            for name, p in q.items():
                reliability_check(r["calibration"][name], y[a, indices], p)
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
                    frequency,
                    t[indices],
                )
            for control in ("common", "uniform"):
                conditional(
                    {
                        k.replace("vs_control", "vs_spot"): v
                        for k, v in r["agreement"]["comparisons"][
                            f"volume_vs_{control}"
                        ].items()
                    },
                    y[a, indices][agreement],
                    q["volume"][agreement],
                    q[control][agreement],
                    frequency[agreement],
                    t[indices][agreement],
                )
            cases += len(indices)
            agreement_cases += int(agreement.sum())
    check(
        len(fits) == count == 42 and len(results["cohorts"]) == 9 and cases == 550362,
        "all fits/cohorts/cases",
    )
    information = all(
        r["comparisons"][f"volume_vs_{c}"]["information_gate"]
        and r["agreement"]["comparisons"][f"volume_vs_{c}"]["information_gate"]
        for r in results["cohorts"].values()
        for c in ("common", "uniform")
    )
    direction = all(r["volume"]["gate"] for r in results["cohorts"].values())
    check(
        results["all_information_gates"] == information
        and results["all_direction_gates"] == direction
        and results["general_solution"] == (information and direction),
        "combined gates",
    )
    return dict(
        verified=True,
        fits=count,
        cases=cases,
        cohorts=9,
        agreement_cases=agreement_cases,
        maximum_raw_fit_residual=maximum,
        protected_data_opened=False,
        general_solution=information and direction,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "reports/timestral-volume-center-20261027"
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), sort_keys=True))
