#!/usr/bin/env python3
"""Rebuild shared-market features, convex-fit gradients and saved evaluations.

Independent of experiment/model/reader modules; does not rerun optimization.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "reports/timestral-microdirection-20261027"
ASSETS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
ODD = list(range(7)) + list(range(13, 21))
FAMILIES = ("local", "shared", "cross", "leave_one_out")


def check(value, message):
    if not value:
        raise ValueError(message)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "biu" and b.dtype.kind in "biu":
        np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_allclose(a, b, atol=2e-9, rtol=1e-8)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def epoch(month):
    return int(datetime(2026, month, 1, tzinfo=timezone.utc).timestamp())


def softmax(x):
    ex = np.exp(x - x.max(axis=1)[:, None])
    return ex / ex.sum(axis=1)[:, None]


def reconstruct(datasets=None):
    if datasets is None:
        datasets = [dict(np.load(PRIOR / "micro/minutes.npz"))] + [
            dict(np.load(PRIOR / f"transfer/{name}-minutes.npz")) for name in ASSETS[1:]
        ]
    t = datasets[0]["time"]
    check(
        t.min() >= 1696118400 and t.max() < 1785542400 and np.all(np.diff(t) == 60),
        "protected boundary or gap",
    )
    i = np.flatnonzero(((t + 60) // 60) % 5 == 0)
    i = i[(i >= 61) & (i + 6 < len(t))]
    own, lag, y = [], [], []
    for data in datasets:
        same(data["time"], t)
        check(all(np.isfinite(a).all() for a in data.values()), "invalid inputs")
        log = np.log(data["close"])
        r = np.concatenate(([0.0], log[1:] - log[:-1]))
        ms = np.convolve(r * r, np.ones(60), mode="valid") / 60
        scale = np.sqrt(np.maximum(ms[i - 60], 1e-16))
        lagscale = np.sqrt(np.maximum(ms[i - 61], 1e-16))
        ret = np.column_stack(
            [
                np.clip((log[i] - log[i - h]) / (scale * np.sqrt(h)), -8, 8)
                for h in (1, 5, 15, 60)
            ]
        )
        lag.append(
            np.column_stack(
                [
                    np.clip(
                        (log[i - 1] - log[i - 1 - h]) / (lagscale * np.sqrt(h)), -8, 8
                    )
                    for h in (1, 5)
                ]
            )
        )
        trades = np.convolve(data["trades"], np.ones(60), mode="valid")[i - 60] / 60
        activity = np.log(np.maximum(data["trades"][i] / np.maximum(trades, 1), 1e-6))
        v = data["volume"][i]
        flow = np.zeros(len(i))
        np.divide(2 * data["taker"][i] - v, v, out=flow, where=v > 0)
        v5 = np.convolve(data["volume"], np.ones(5), mode="valid")[i - 4]
        buy5 = np.convolve(data["taker"], np.ones(5), mode="valid")[i - 4]
        flow5 = np.zeros(len(i))
        np.divide(2 * buy5 - v5, v5, out=flow5, where=v5 > 0)
        zero = np.convolve(r == 0, np.ones(60), mode="valid")[i - 59] / 60
        own.append(
            np.column_stack(
                (
                    ret,
                    flow,
                    flow5,
                    ret[:, 0] * np.clip(activity, -4, 4),
                    abs(ret[:, 0]),
                    np.log1p(data["trades"][i]),
                    zero,
                    activity,
                    np.log(scale),
                    abs(flow),
                )
            )
        )
        y.append(
            np.where(
                data["close"][i + 6] > data["close"][i + 1],
                2,
                np.where(data["close"][i + 6] < data["close"][i + 1], 0, 1),
            )
        )
    x = []
    for a in range(3):
        b, c = [j for j in range(3) if j != a]
        other = (own[b][:, :4] + own[c][:, :4]) / 2
        residual = np.clip(own[a][:, 0] - other[:, 0], -8, 8)
        dispersion = abs(own[b][:, 0] - own[c][:, 0]) / 2
        x.append(
            np.column_stack(
                (
                    own[a],
                    other,
                    (lag[b] + lag[c]) / 2,
                    residual * abs(residual) / 8,
                    own[a][:, 0] * dispersion,
                    (own[b][:, 9] + own[c][:, 9]) / 2,
                    abs(other[:, 0]),
                )
            )
        )
    return t[i] + 60, t[i + 6] + 60, np.stack(x), np.stack(y)


def verify_metrics(stored, y, p, base, prior, t):
    selected = np.maximum(p[:, 0], p[:, 2]) >= 0.52
    correct = p.argmax(1) == y
    loss = np.column_stack(
        [
            np.sum(q * q, axis=1) - 2 * q[np.arange(len(y)), y] + 1
            for q in (base, prior, p)
        ]
    )
    names = ("frequency", "prior_reversal", "shared_model")
    check(
        stored["n"] == len(y) and stored["selected"] == selected.sum(),
        "selected membership",
    )
    same(stored["coverage"], selected.mean())
    same([stored["all_brier"][k] for k in names], loss.mean(0))
    if not selected.any():
        check(not stored["gate"], "empty selection gate")
        return
    same(stored["accuracy"], correct[selected].mean())
    same([stored["selected_brier"][k] for k in names], loss[selected].mean(0))
    same(
        stored["log_loss"],
        -np.log(p[selected][np.arange(selected.sum()), y[selected]]).mean(),
    )
    same(
        stored["baseline_log_loss"],
        -np.log(base[selected][np.arange(selected.sum()), y[selected]]).mean(),
    )
    for k in (0, 2):
        check(
            stored["predicted_direction_counts"][str(k)]
            == np.sum(selected & (p.argmax(1) == k)),
            "predicted direction counts",
        )
    days, inv = np.unique(t // 86400, return_inverse=True)
    counts = np.zeros(len(days))
    np.add.at(counts, inv, selected)
    sums = np.zeros((len(days), 4))
    np.add.at(sums, inv, np.column_stack((loss, correct)) * selected[:, None])
    start = np.random.default_rng(271029).integers(
        0, len(days) - 6, size=(4000, (len(days) + 6) // 7)
    )
    ids = np.concatenate([start[:, :, None] + j for j in range(7)], axis=2).reshape(
        4000, -1
    )[:, : len(days)]
    den = counts[ids].sum(1)
    if np.any(den == 0):
        check(not stored["gate"], "empty bootstrap gate")
        return
    boot = sums[ids].sum(1) / den[:, None]
    q = [1 / 120, 119 / 120]
    same(stored["accuracy_ci98_333"], np.quantile(boot[:, 3], q))
    same(
        stored["gain_ci98_333_pct"],
        np.quantile(100 * (boot[:, 0] - boot[:, 2]) / boot[:, 0], q),
    )
    same(
        stored["gain_vs_prior_ci98_333_pct"],
        np.quantile(100 * (boot[:, 1] - boot[:, 2]) / boot[:, 1], q),
    )
    months = t.astype("datetime64[s]").astype("datetime64[M]")
    for m in np.unique(months):
        mask = selected & (months == m)
        v = stored["monthly"][str(m)]
        check(v["n"] == mask.sum(), "monthly selected count")
        if mask.any():
            same(v["accuracy"], correct[mask].mean())
            same(
                v["gain_pct"],
                100
                * (loss[mask, 0].mean() - loss[mask, 2].mean())
                / loss[mask, 0].mean(),
            )
        else:
            check(v["accuracy"] is None and v["gain_pct"] is None, "empty month")
    gate = (
        stored["coverage"] >= 0.05
        and stored["accuracy"] >= 0.52
        and stored["accuracy_ci98_333"][0] > 0.5
        and stored["gain_ci98_333_pct"][0] > 0
        and stored["log_loss"] <= stored["baseline_log_loss"]
        and min(stored["predicted_direction_counts"].values()) >= 100
        and sum(v["n"] >= 100 and v["gain_pct"] > 0 for v in stored["monthly"].values())
        >= 5
    )
    check(stored["gate"] == bool(gate), "generalization criterion")


def main(root):
    protocol = json.loads((root / "protocol.json").read_text())
    for name, sha in json.loads((root / "manifest.json").read_text()).items():
        check(digest(root / name) == sha, f"artifact hash {name}")
    check(
        digest(root / "protocol.json")
        == json.loads((root / "protocol.sha256.json").read_text())["sha256"],
        "protocol hash",
    )
    for name, sha in {**protocol["code"], **protocol["inputs"]}.items():
        check(digest(ROOT / name) == sha, f"input/code hash {name}")
    check(
        protocol["at"] < json.loads((root / "opened.json").read_text())["at"],
        "registration time",
    )
    t, ends, x, y = reconstruct()
    fits = json.loads((root / "fits.json").read_text())
    result = json.loads((root / "results.json").read_text())
    alltest = (t >= epoch(1)) & (t < epoch(8))
    saved = {
        a: dict(np.load(root / f"{name}-predictions.npz"))
        for a, name in enumerate(ASSETS)
    }
    max_gradient = 0.0
    checked = set()
    for month in range(1, 8):
        train = (t >= epoch(month) - 90 * 86400) & (ends <= epoch(month))
        test = (t >= epoch(month)) & (t < epoch(month + 1))
        for family in FAMILIES:
            for target in range(3) if family in ("local", "leave_one_out") else ["all"]:
                assets = (
                    [target]
                    if family == "local"
                    else [j for j in range(3) if j != target]
                    if family == "leave_one_out"
                    else [0, 1, 2]
                )
                ncol = 13 if family in ("local", "shared") else 23
                key = f"{family}_{month}_{target}"
                f = fits[key]
                checked.add(key)
                check(
                    f["assets"] == assets
                    and f["columns"] == ncol
                    and f["cutoff"] == epoch(month)
                    and f["n"] == train.sum() * len(assets),
                    "training scope",
                )
                ids = np.concatenate(
                    [
                        np.column_stack((np.full(train.sum(), a), t[train]))
                        for a in assets
                    ]
                ).astype("<i8")
                check(
                    hashlib.sha256(ids.tobytes()).hexdigest()
                    == f["training_ids_sha256"],
                    "training id hash",
                )
                raw = np.concatenate([x[a, train, :ncol] for a in assets])
                labels = np.concatenate([y[a, train] for a in assets])
                reflected = raw.copy()
                reflected[:, [j for j in ODD if j < ncol]] *= -1
                augmented = np.concatenate((raw, reflected))
                labels = np.concatenate((labels, 2 - labels))
                mean = augmented.mean(0)
                mean[[j for j in ODD if j < ncol]] = 0
                scale = np.maximum(augmented.std(0), 1e-8)
                same(f["mean"], mean)
                same(f["scale"], scale)
                z = np.clip((augmented - mean) / scale, -8, 8)
                w = np.asarray(f["coef"])
                b = np.asarray(f["intercept"])
                probability = softmax(z @ w.T + b)
                residual = probability - np.eye(3)[labels]
                gradient = max(
                    np.max(abs(residual.T @ z / len(z) + w / (0.1 * len(z)))),
                    np.max(abs(residual.mean(0))),
                )
                max_gradient = max(max_gradient, float(gradient))
                check(gradient < 1e-5, "fit not stationary")
                for a in range(3) if target == "all" else [target]:
                    pred = softmax(
                        np.clip((x[a, test, :ncol] - mean) / scale, -8, 8) @ w.T + b
                    )
                    mask = (saved[a]["anchor"] >= epoch(month)) & (
                        saved[a]["anchor"] < epoch(month + 1)
                    )
                    same(saved[a]["anchor"][mask], t[test])
                    same(saved[a][family][mask], pred)
        for a in range(3):
            count = np.bincount(y[a, train], minlength=3)
            expected = np.tile((count + 1) / (train.sum() + 3), (test.sum(), 1))
            mask = (saved[a]["anchor"] >= epoch(month)) & (
                saved[a]["anchor"] < epoch(month + 1)
            )
            same(saved[a]["frequency"][mask], expected)
    check(checked == set(fits), "unexpected or missing fits")
    for a, name in enumerate(ASSETS):
        out = dict(np.load(root / f"{name}-outcomes.npz"))
        same(out["anchor"], t[alltest])
        same(out["end"], ends[alltest])
        same(out["label"], y[a, alltest])
        prior = dict(
            np.load(
                PRIOR / "reversal/h5-predictions.npz"
                if a == 0
                else PRIOR / f"transfer/{name}-predictions.npz"
            )
        )
        same(saved[a]["anchor"], prior["anchor"])
        same(saved[a]["prior_reversal"], prior["probability"][:, 2])
        for family in FAMILIES:
            verify_metrics(
                result["assets"][name][family],
                y[a, alltest],
                saved[a][family],
                saved[a]["frequency"],
                saved[a]["prior_reversal"],
                t[alltest],
            )
    check(
        result["primary_gate"]
        == all(result["assets"][name]["cross"]["gate"] for name in ASSETS),
        "primary all-assets gate",
    )
    return {
        "verified": True,
        "fits": len(fits),
        "asset_family_comparisons": 12,
        "max_objective_gradient": max_gradient,
        "primary_gate": result["primary_gate"],
        "prospective_validation": False,
        "protected_data_opened": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "reports/timestral-universal-20261027"
    )
    args = parser.parse_args()
    print(json.dumps(main(args.root), sort_keys=True))
