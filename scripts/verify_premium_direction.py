#!/usr/bin/env python3
"""Independent premium features, binary fit gradients and saved-score audit."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_universal_direction import (
    ROOT,
    ASSETS,
    ODD,
    check,
    same,
    digest,
    reconstruct,
    verify_metrics,
)  # noqa: E402

SOURCE = ROOT / "reports/timestral-universal-20261027"
MICRO = ROOT / "reports/timestral-microdirection-20261027"


def epoch(year, month):
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())


def raw_audit(root, records, arrays):
    for receipt in records:
        name = receipt["url"].rsplit("/", 1)[-1]
        s, _, ys, ms = name.removesuffix(".zip").split("-")
        year, month = int(ys), int(ms)
        cohort = year + 1 if month >= 10 else year
        path = ROOT / "research/out/premium-direction-20261027/raw" / name
        check(digest(path) == receipt["sha256"], "raw checksum")
        check(path.stat().st_size == receipt["bytes"], "raw size")
        check(
            path.with_suffix(".zip.CHECKSUM").read_text().split()[0]
            == receipt["sha256"],
            "official checksum receipt",
        )
        rows = []
        with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
            for row in csv.reader(io.TextIOWrapper(f)):
                if row[0] == "open_time":
                    continue
                check(len(row) == 12, "raw kline shape")
                t, end = int(row[0]), int(row[6])
                o, h, low, c = map(float, row[1:5])
                check(t % 60000 == 0 and end == t + 59999, "millisecond minute units")
                check(
                    np.isfinite([o, h, low, c]).all()
                    and low <= o <= h
                    and low <= c <= h,
                    "premium OHLC",
                )
                rows.append((t // 1000, c))
        a = np.asarray(rows)
        expected = np.arange(
            epoch(year, month), epoch(year + (month == 12), month % 12 + 1), 60
        )
        check(
            np.isin(a[:, 0], expected).all() and np.all(np.diff(a[:, 0]) > 0),
            "raw gaps without duplicates",
        )
        check(
            receipt["missing_minutes"] == len(expected) - len(a), "raw missing minutes"
        )
        check(len(a) == receipt["rows"], "raw count")
        d = arrays[f"{cohort}-{s}"]
        indices = np.searchsorted(d["time"], a[:, 0].astype(np.int64))
        same(d["time"][indices], a[:, 0])
        same(d["close"][indices], a[:, 1])


def features(root, year, premium):
    datasets = []
    for index, symbol in enumerate(ASSETS):
        path = (
            SOURCE / "history" / f"{symbol}-minutes.npz"
            if year < 2026
            else MICRO / "micro/minutes.npz"
            if index == 0
            else MICRO / f"transfer/{symbol}-minutes.npz"
        )
        source = dict(np.load(path))
        take = (source["time"] >= epoch(year - 1, 10)) & (
            source["time"] < epoch(year, 8) + 360
        )
        datasets.append({k: v[take] for k, v in source.items()})
    t, end, x, y = reconstruct(datasets)
    grid = np.arange(epoch(year - 1, 10), epoch(year, 8), 60)
    j = np.searchsorted(grid, t - 120)
    in_range = (j >= 119) & (j < len(grid))
    own = []
    for symbol in ASSETS:
        source = premium[f"{year}-{symbol}"]
        series = np.full(len(grid), np.nan)
        series[np.searchsorted(grid, source["time"])] = np.clip(
            source["close"] * 1e4, -10, 10
        )
        jj = j[in_range]
        avg = [
            np.convolve(series, np.ones(h) / h, "valid")[jj - h + 1]
            for h in (5, 30, 120)
        ]
        values = np.full((len(t), 7), np.nan)
        values[in_range] = np.column_stack(
            (
                series[jj],
                *avg,
                series[jj] - series[jj - 5],
                series[jj] - series[jj - 30],
                series[jj] - avg[2],
            )
        )
        own.append(values)
    available = np.isfinite(np.stack(own)).all(axis=(0, 2))
    additions = np.zeros((3, len(t), 12))
    for a in range(3):
        b, c = [i for i in range(3) if i != a]
        other = (own[b][available][:, [0, 4]] + own[c][available][:, [0, 4]]) / 2
        own_a = own[a][available]
        additions[a, available] = np.column_stack(
            (
                own_a,
                other,
                own_a[:, 0] - other[:, 0],
                own_a[:, 0] * np.minimum(abs(x[a, available, 0]), 8),
                own_a[:, 0] * np.clip(x[a, available, 10], -4, 4),
            )
        )
    return t, end, np.concatenate((x, additions), axis=2), y, available


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
    odd = [j for j in ODD + list(range(23, 35)) if j < x.shape[1]]
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
    return maximum


def conditional(stored, y, p, spot, base, t):
    mask = y != 1
    label = (y[mask] == 2).astype(int)
    p, spot, base, t = p[mask], spot[mask], base[mask], t[mask]
    correct = (p >= 0.5) == label

    def ll(q):
        return -np.log(np.where(label == 1, q, 1 - q))

    def brier(q):
        return q * q - 2 * q * label + label

    values = np.column_stack(
        (
            correct,
            brier(spot) - brier(p),
            ll(spot) - ll(p),
            brier(base) - brier(p),
            ll(base) - ll(p),
        )
    )
    days, inv = np.unique(t // 86400, return_inverse=True)
    totals = np.zeros((len(days), 6))
    np.add.at(totals, inv, np.column_stack((np.ones(len(t)), values)))
    start = np.random.default_rng(271032).integers(
        0, len(days) - 6, (4000, (len(days) + 6) // 7)
    )
    ids = np.concatenate([start[:, :, None] + j for j in range(7)], axis=2).reshape(
        4000, -1
    )[:, : len(days)]
    sums = totals[ids].sum(1)
    ci = np.quantile(sums[:, 1:] / sums[:, :1], [1 / 360, 359 / 360], axis=0)
    names = (
        "accuracy",
        "brier_gain_vs_spot",
        "logloss_gain_vs_spot",
        "brier_gain_vs_frequency",
        "logloss_gain_vs_frequency",
    )
    for j, name in enumerate(names):
        same(stored[name], values[:, j].mean())
        same(stored[name + "_ci99_444"], ci[:, j])
    check(
        stored["n"] == mask.sum() and stored["flats"] == (~mask).sum(),
        "conditional denominator",
    )
    same(stored["brier"], brier(p).mean())
    same(stored["logloss"], ll(p).mean())
    same(
        stored["balanced_accuracy"],
        (correct[label == 0].mean() + correct[label == 1].mean()) / 2,
    )
    check(
        stored["predicted_up"] == (p >= 0.5).sum()
        and stored["predicted_down"] == (p < 0.5).sum(),
        "binary direction counts",
    )
    months = t.astype("datetime64[s]").astype("datetime64[M]")
    monthly = {str(m): float(values[months == m, 1].mean()) for m in np.unique(months)}
    for m, v in monthly.items():
        same(stored["monthly_brier_gain_vs_spot"][m], v)
    gate = (
        ci[0, 0] > 0.5
        and (ci[0, 1:] > 0).all()
        and sum(v > 0 for v in monthly.values()) >= 5
        and min(stored["predicted_up"], stored["predicted_down"]) >= 100
    )
    check(stored["information_gate"] == bool(gate), "new information gate")


def verify(root, raw=False):
    aborted = root.parent
    initial = json.loads((aborted / "protocol.json").read_text())
    check(
        digest(aborted / "protocol.json")
        == json.loads((aborted / "protocol.sha256.json").read_text())["sha256"],
        "aborted protocol preserved",
    )
    for name, sha in initial["code"].items():
        check(digest(ROOT / name) == sha, "aborted code preserved")
    check(not (aborted / "results.json").exists(), "no scores from failed preparation")
    check(
        json.loads((aborted / "failure.json").read_text())["status"]
        == "aborted_during_preparation_before_fitting",
        "preparation failure recorded",
    )
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
    provenance = json.loads((root / "sources.json").read_text())
    check(
        protocol["at"]
        < provenance["at"]
        < json.loads((root / "opened.json").read_text())["at"],
        "registration, preparation and evaluation chronology",
    )
    check(
        protocol["urls"] == [r["url"] for r in provenance["sources"]],
        "registered sources",
    )
    arrays = {}
    for year in (2024, 2025, 2026):
        for s in ASSETS:
            key = f"{year}-{s}"
            path = root / f"{key}-premium.npz"
            check(digest(path) == provenance["arrays"][path.name], "source array hash")
            arrays[key] = dict(np.load(path))
            check(
                np.isin(
                    arrays[key]["time"],
                    np.arange(epoch(year - 1, 10), epoch(year, 8), 60),
                ).all()
                and np.all(np.diff(arrays[key]["time"]) > 0),
                "native premium chronology",
            )
            check(
                np.isfinite(arrays[key]["close"]).all()
                and arrays[key]["time"][-1] < 1785542400,
                "premium values and protected boundary",
            )
    if raw:
        raw_audit(root, provenance["sources"], arrays)
    fits = json.loads((root / "fits.json").read_text())
    results = json.loads((root / "results.json").read_text())
    maximum, count = 0.0, 0
    for year in (2024, 2025, 2026):
        t, end, x, y, available = features(root, year, arrays)
        predictions = {name: [] for name in ("spot", "premium")}
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
            for name, columns in (("spot", 23), ("premium", 35)):
                state = fits[f"{year}-{month}-{name}"]
                xx = x[:, train, :columns].reshape(-1, columns)
                maximum = max(
                    maximum,
                    binary_fit(
                        state,
                        xx[moving],
                        (labels[moving] == 2).astype(int),
                        ids[moving],
                        cutoff,
                    ),
                )
                z = np.clip(
                    (x[:, test, :columns] - state["mean"]) / state["scale"], -8, 8
                )
                predictions[name].append(
                    1
                    / (1 + np.exp(-(z @ np.array(state["coef"]) + state["intercept"])))
                )
                if name == "premium":
                    predictions[name][-1][:, ~available[test]] = predictions["spot"][
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
            same(saved["premium_available"], available[indices])
            check(
                results["cohorts"][key]["premium_missing_contexts"]
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
            for name in ("spot", "premium"):
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
            conditional(
                results["cohorts"][key]["conditional"],
                y[a, indices],
                predictions["premium"][a],
                predictions["spot"][a],
                prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1]),
                t[indices],
            )
    check(
        len(fits) == count == 42 and len(results["cohorts"]) == 9,
        "all models/cohorts required",
    )
    check(
        results["all_information_gates"]
        == all(
            r["conditional"]["information_gate"] for r in results["cohorts"].values()
        ),
        "combined information gate",
    )
    check(
        results["all_direction_gates"]
        == all(r["premium"]["gate"] for r in results["cohorts"].values()),
        "combined direction gate",
    )
    return {
        "verified": True,
        "cohorts": 9,
        "fits": count,
        "max_gradient": maximum,
        "raw_sources_verified": raw,
        "protected_data_opened": False,
        "all_information_gates": results["all_information_gates"],
        "all_direction_gates": results["all_direction_gates"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "reports/timestral-premium-direction-20261027/gap-safe",
    )
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.raw), sort_keys=True))
