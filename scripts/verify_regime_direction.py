#!/usr/bin/env python3
"""Independent context/regime reconstruction, fit-gradient and drift audit."""

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
    ODD,
    check,
    same,
    digest,
    verify_metrics,
)  # noqa: E402
from scripts.verify_futures_flow import features as flow_features, epoch, SOURCE, MICRO  # noqa: E402
from scripts.verify_premium_direction import conditional  # noqa: E402

FUTURES = ROOT / "reports/timestral-futures-flow-20261027"


def features(year, arrays):
    t, end, x, y, available = flow_features(FUTURES, year, arrays)
    additions = np.zeros((3, len(t), 4))
    states = np.zeros((3, len(t)), dtype=np.int8)
    own = []
    for a, s in enumerate(ASSETS):
        path = (
            SOURCE / "history" / f"{s}-minutes.npz"
            if year < 2026
            else MICRO / "micro/minutes.npz"
            if a == 0
            else MICRO / f"transfer/{s}-minutes.npz"
        )
        d = dict(np.load(path))
        take = (d["time"] >= epoch(year - 1, 10)) & (d["time"] < epoch(year, 8) + 360)
        d = {k: v[take] for k, v in d.items()}
        j = np.searchsorted(d["time"], t - 60)
        valid = j >= 1500
        jj = j[valid]
        log = np.log(d["close"])
        r = np.r_[0.0, np.diff(log)]
        rs = []
        for h in (240, 1440):
            variance = np.convolve(r * r, np.ones(h), "valid")[jj - h + 1]
            rs.append(
                np.clip(
                    (log[jj] - log[jj - h]) / np.sqrt(np.maximum(variance, h * 1e-16)),
                    -8,
                    8,
                )
            )
        hour = np.convolve(d["trades"], np.ones(60), "valid")[jj - 59] / 60
        reference = np.convolve(d["trades"], np.ones(1440), "valid")[jj - 1499] / 1440
        states[a, valid] = (
            2 * (np.abs(rs[0]) >= 1) + (hour / np.maximum(reference, 1) >= 1)
        ).astype(np.int8)
        own.append(np.column_stack(rs))
    for a in range(3):
        b, c = [i for i in range(3) if i != a]
        additions[a, valid] = np.column_stack((own[a], (own[b] + own[c]) / 2))
    z = np.concatenate((x, additions), axis=2)
    interactions = np.empty((3, len(t), 32))
    for j, col in enumerate((0, 1, 4, 27, 28, 13, 35, 36)):
        for k in range(4):
            interactions[:, :, j * 4 + k] = z[:, :, col] * (
                (states == k).astype(float) - 0.25
            )
    return (
        t,
        end,
        np.concatenate((z, interactions), axis=2),
        y,
        available & valid,
        states,
    )


def binary_fit(state, x, y, ids, cutoff):
    check(state["n"] == len(y) and state["cutoff"] == cutoff, "binary fit scope")
    check(
        hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
        == state["training_ids_sha256"],
        "binary IDs",
    )
    check(
        hashlib.sha256(y.astype("<i8").tobytes()).hexdigest()
        == state["training_labels_sha256"],
        "binary labels",
    )
    odd = [j for j in ODD + list(range(23, x.shape[1])) if j < x.shape[1]]
    mirror = x.copy()
    mirror[:, odd] *= -1
    xx, yy = np.vstack((x, mirror)), np.r_[y, 1 - y]
    mean = xx.mean(0)
    mean[odd] = 0
    scale = np.maximum(xx.std(0), 1e-8)
    same(state["mean"], mean)
    same(state["scale"], scale)
    z = np.clip((xx - mean) / scale, -8, 8)
    coef = np.array(state["coef"])
    p = 1 / (1 + np.exp(-(z @ coef + state["intercept"])))
    gradient = z.T @ (p - yy) / len(yy) + coef / (0.1 * len(yy))
    maximum = float(max(np.max(abs(gradient)), abs((p - yy).mean())))
    check(maximum < 3e-6, "binary convex fit gradient")
    logits = z @ coef + state["intercept"]
    training_logloss = float(np.mean(np.logaddexp(0, logits) - yy * logits))
    return maximum, training_logloss


def verify_diagnosis(stored, y, t, state, x):
    for name, signal in (("spot_reversal", -x[:, 0]), ("futures_flow", x[:, 27])):
        d = stored[name]
        use = (y != 1) & (signal != 0)
        correct = (signal > 0) == (y == 2)
        n = np.bincount(state[use], minlength=4)
        hit = np.bincount(state[use], weights=correct[use], minlength=4)
        check(d["n"] == use.sum() and d["excluded"] == (~use).sum(), "diagnostic scope")
        same(d["accuracy"], hit.sum() / n.sum())
        days, inv = np.unique(t // 86400, return_inverse=True)
        totals = np.empty((len(days), 4, 2))
        for k in range(4):
            mask = use & (state == k)
            totals[:, k, 0] = np.bincount(inv, weights=mask, minlength=len(days))
            totals[:, k, 1] = np.bincount(
                inv, weights=mask & correct, minlength=len(days)
            )
        starts = np.random.default_rng(271033).integers(
            0, len(days) - 6, (4000, (len(days) + 6) // 7)
        )
        ids = np.concatenate(
            [starts[:, :, None] + k for k in range(7)], axis=2
        ).reshape(4000, -1)[:, : len(days)]
        totals = totals[ids].sum(axis=1)
        for k, c in enumerate(d["regimes"]):
            check(
                c["n"] == n[k] and c["correct"] == hit[k],
                "regime membership and correct count",
            )
            same(c["weight"], n[k] / n.sum())
            same(c["accuracy"], hit[k] / n[k])
            same(
                c["accuracy_ci95"],
                np.quantile(totals[:, k, 1] / totals[:, k, 0], [0.025, 0.975]),
            )
        alltotals = totals.sum(1)
        same(
            d["accuracy_ci95"],
            np.quantile(alltotals[:, 1] / alltotals[:, 0], [0.025, 0.975]),
        )


def verify_decomposition(d):
    for s in ASSETS:
        for year in (2025, 2026):
            for rule in ("spot_reversal", "futures_flow"):
                first = d["cohorts"][f"2024-{s}"][rule]
                last = d["cohorts"][f"{year}-{s}"][rule]
                v = d["decomposition"][f"{year}-{s}-{rule}"]
                mix = sum(
                    (b["weight"] - a["weight"]) * a["accuracy"]
                    for a, b in zip(first["regimes"], last["regimes"])
                )
                within = sum(
                    b["weight"] * (b["accuracy"] - a["accuracy"])
                    for a, b in zip(first["regimes"], last["regimes"])
                )
                same(v["composition"], mix)
                same(v["within"], within)
                same(v["total"], last["accuracy"] - first["accuracy"])
                same(v["total"], mix + within)
    check(len(d["decomposition"]) == 12, "all twelve decompositions")


def verify(root):
    protocol = json.loads((root / "protocol.json").read_text())
    for name, sha in json.loads((root / "manifest.json").read_text()).items():
        check(digest(root / name) == sha, f"artifact hash {name}")
    check(
        digest(root / "protocol.json")
        == json.loads((root / "protocol.sha256.json").read_text())["sha256"],
        "protocol hash",
    )
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
    fits = json.loads((root / "fits.json").read_text())
    results = json.loads((root / "results.json").read_text())
    diagnosis = json.loads((root / "diagnosis.json").read_text())
    verify_decomposition(diagnosis)
    maximum, count = 0.0, 0
    training_losses = {}
    for year in (2024, 2025, 2026):
        t, end, x, y, available, states = features(year, arrays)
        predictions = {name: [] for name in ("flow", "long", "regime")}
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
            for name, columns in (("flow", 35), ("long", 39), ("regime", 71)):
                state = fits[f"{year}-{month}-{name}"]
                xx = x[:, train, :columns].reshape(-1, columns)
                grad, train_loss = binary_fit(
                    state,
                    xx[moving],
                    (labels[moving] == 2).astype(int),
                    ids[moving],
                    cutoff,
                )
                maximum = max(maximum, grad)
                training_losses[f"{year}-{month}-{name}"] = train_loss
                z = np.clip(
                    (x[:, test, :columns] - state["mean"]) / state["scale"], -8, 8
                )
                predictions[name].append(
                    1
                    / (1 + np.exp(-(z @ np.array(state["coef"]) + state["intercept"])))
                )
                if name != "flow":
                    predictions[name][-1][:, ~available[test]] = predictions["flow"][
                        -1
                    ][:, ~available[test]]
                count += 1
            indices.append(np.flatnonzero(test))
        indices = np.concatenate(indices)
        predictions = {
            name: np.concatenate(pieces, axis=1) for name, pieces in predictions.items()
        }
        for a, s in enumerate(ASSETS):
            key = f"{year}-{s}"
            prefix, folder = (s, SOURCE) if year == 2026 else (key, SOURCE / "history")
            prior = dict(np.load(folder / f"{prefix}-predictions.npz"))
            outcome = dict(np.load(folder / f"{prefix}-outcomes.npz"))
            saved = dict(np.load(root / f"{key}-predictions.npz"))
            same(saved["available"], available[indices])
            same(saved["regime"], states[a, indices])
            same(
                saved["flow_up"], np.load(FUTURES / f"{key}-predictions.npz")["flow_up"]
            )
            verify_diagnosis(
                diagnosis["cohorts"][key],
                y[a, indices],
                t[indices],
                states[a, indices],
                x[a, indices],
            )
            check(
                results["cohorts"][key]["missing_contexts"]
                == (~available[indices]).sum(),
                "fallback count",
            )
            same(saved["anchor"], t[indices])
            same(saved["anchor"], prior["anchor"])
            same(outcome["anchor"], t[indices])
            same(outcome["label"], y[a, indices])
            same(outcome["end"], end[indices])
            check(end[indices].max() <= 1785542400, "protected outcomes")
            flat = prior["cross" if year == 2026 else "probability"][:, 1]
            for name in ("flow", "long", "regime"):
                p = predictions[name][a]
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
            for candidate, control in (
                ("long", "flow"),
                ("regime", "flow"),
                ("regime", "long"),
            ):
                metrics = results["cohorts"][key]["comparisons"][
                    f"{candidate}_vs_{control}"
                ]
                conditional(
                    {k.replace("vs_control", "vs_spot"): v for k, v in metrics.items()},
                    y[a, indices],
                    predictions[candidate][a],
                    predictions[control][a],
                    prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1]),
                    t[indices],
                )
    check(
        len(fits) == count == 63 and len(results["cohorts"]) == 9,
        "all models/cohorts required",
    )
    check(
        results["all_information_gates"]
        == all(
            r["comparisons"][c]["information_gate"]
            for r in results["cohorts"].values()
            for c in ("regime_vs_flow", "regime_vs_long")
        ),
        "combined information gate",
    )
    check(
        results["all_direction_gates"]
        == all(r["regime"]["gate"] for r in results["cohorts"].values()),
        "combined direction gate",
    )
    return {
        "verified": True,
        "cohorts": 9,
        "fits": count,
        "max_gradient": maximum,
        "posthoc_training_logloss": training_losses,
        "diagnostic_decompositions": 12,
        "frozen_flow_control_reproduced": True,
        "protected_data_opened": False,
        "all_information_gates": results["all_information_gates"],
        "all_direction_gates": results["all_direction_gates"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "reports/timestral-regime-direction-20261027",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), sort_keys=True))
