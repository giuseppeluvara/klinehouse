#!/usr/bin/env python3
"""Independent older-cohort and nonlinear shared-direction evidence audit."""

import argparse
import csv
from datetime import datetime, timezone
import gzip
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
    softmax,
    reconstruct,
    verify_metrics,
)  # noqa: E402


def epoch(year, month):
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())


def bundle(root):
    for name, sha in json.loads((root / "manifest.json").read_text()).items():
        check(digest(root / name) == sha, f"changed artifact {root / name}")
    p = json.loads((root / "protocol.json").read_text())
    check(
        digest(root / "protocol.json")
        == json.loads((root / "protocol.sha256.json").read_text())["sha256"],
        "protocol hash",
    )
    for name, sha in {**p["code"], **p.get("inputs", {})}.items():
        check(digest(ROOT / name) == sha, f"code or input hash {name}")
    check(
        p["at"] < json.loads((root / "opened.json").read_text())["at"],
        "registration chronology",
    )
    return p


def baseline_states(data, t, x):
    i = np.searchsorted(data["time"], t - 60)
    activity = data["trades"][i] / np.maximum(
        np.convolve(data["trades"], np.ones(60), mode="valid")[i - 60] / 60, 1.0
    )
    cells = np.digitize(abs(x[:, 0]), [0.5, 1, 2]) * 3 + np.digitize(
        activity, [0.5, 1.5]
    )
    return cells, np.sign(x[:, 0]).astype(int)


def check_baseline(f, cells, signs, y, train, test, cutoff):
    check(f["cutoff"] == cutoff and f["n"] == train.sum(), "baseline fit scope")
    labels = y[train]
    sgn = signs[train]
    valid = sgn != 0
    relative = np.where(sgn < 0, 2 - labels, labels)
    counts = np.bincount(relative[valid], minlength=3)
    prior = (counts + 1) / (valid.sum() + 3)
    count = np.bincount(
        3 * cells[train][valid] + relative[valid], minlength=36
    ).reshape(12, 3)
    table = (count + 100 * prior) / (count.sum(1)[:, None] + 100)
    base = (np.bincount(labels, minlength=3) + 1) / (len(labels) + 3)
    same(f["relative_counts"], counts)
    same(f["counts"], count)
    same(f["pooled"], prior)
    same(f["table"], table)
    same(f["frequency"], base)
    p = table[cells[test]].copy()
    p[signs[test] < 0] = p[signs[test] < 0, ::-1]
    p[signs[test] == 0] = base
    return np.tile(base, (test.sum(), 1)), p


def raw_sources(root, sources, datasets):
    for receipt in sources["sources"]:
        name = receipt["url"].rsplit("/", 1)[-1]
        path = ROOT / "research/out/universal-history-20261027/raw" / name
        check(digest(path) == receipt["sha256"], "native ZIP hash")
        check(
            path.with_name(name + ".CHECKSUM").read_text().split()[0]
            == receipt["sha256"],
            "official checksum receipt",
        )
        symbol, _, year, month = name.removesuffix(".zip").split("-")
        year, month = int(year), int(month)
        rows = []
        invalid = []
        with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
            for row in csv.reader(io.TextIOWrapper(f)):
                if row[0] == "open_time":
                    continue
                unit = 1 if int(row[0]) >= 10**15 else 1000
                t, ct = int(row[0]) * unit, int(row[6]) * unit
                when = datetime.fromtimestamp(t / 1e6, timezone.utc)
                check(
                    when.year == year
                    and when.month == month
                    and t % 60000000 == 0
                    and epoch(2023, 10) <= t // 1000000 < epoch(2025, 8),
                    "native timestamps",
                )
                o, h, low, c, v = map(float, row[1:6])
                tr, buy = float(row[8]), float(row[9])
                valid = (
                    t <= ct < t + 60000000
                    and np.isfinite([o, h, low, c, v, tr, buy]).all()
                    and 0 < low <= min(o, c) <= max(o, c) <= h
                    and min(v, tr, buy) >= 0
                    and buy <= v + 1e-8
                )
                if valid:
                    rows.append((t // 1000000, c, v, buy, tr))
                else:
                    invalid.append(row)
        check(
            invalid == receipt["invalid"] and len(rows) == receipt["rows"],
            "native row disposition",
        )
        a = np.asarray(rows)
        data = datasets[ASSETS.index(symbol)]
        ix = np.searchsorted(data["time"], a[:, 0].astype(np.int64))
        for j, k in enumerate(("time", "close", "volume", "taker", "trades")):
            same(a[:, j].astype(np.int64) if j == 0 else a[:, j], data[k][ix])


def audit_history(root, raw=False):
    folder = root / "history"
    p = bundle(folder)
    check(
        p["prior_results_sha256"] == digest(root / "results.json"),
        "replication follows frozen result",
    )
    sources = json.loads((folder / "sources.json").read_text())
    check(p["urls"] == [v["url"] for v in sources["sources"]], "source URL list")
    datasets = []
    for s in ASSETS:
        path = folder / f"{s}-minutes.npz"
        check(digest(path) == sources["derived"][s]["sha256"], "native derived hash")
        datasets.append(dict(np.load(path)))
    t, ends, x, y = reconstruct(datasets)
    fits = json.loads((folder / "fits.json").read_text())
    baselines = json.loads((folder / "baselines.json").read_text())
    results = json.loads((folder / "results.json").read_text())
    max_gradient = 0.0
    fitkeys = set()
    basekeys = set()
    states = [baseline_states(datasets[a], t, x[a]) for a in range(3)]
    for year in (2024, 2025):
        alltest = (t >= epoch(year, 1)) & (t < epoch(year, 8))
        saved = {
            a: dict(np.load(folder / f"{year}-{s}-predictions.npz"))
            for a, s in enumerate(ASSETS)
        }
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (t >= cutoff - 90 * 86400) & (ends <= cutoff)
            test = (t >= cutoff) & (t < epoch(year, month + 1))
            key = f"{year}_{month}"
            fitkeys.add(key)
            f = fits[key]
            check(
                f["cutoff"] == cutoff
                and f["assets"] == [0, 1, 2]
                and f["columns"] == 23
                and f["n"] == 3 * train.sum(),
                "shared fit scope",
            )
            ids = np.concatenate(
                [np.column_stack((np.full(train.sum(), a), t[train])) for a in range(3)]
            ).astype("<i8")
            check(
                hashlib.sha256(ids.tobytes()).hexdigest() == f["training_ids_sha256"],
                "shared training ids",
            )
            data = np.concatenate([x[a, train] for a in range(3)])
            labels = np.concatenate([y[a, train] for a in range(3)])
            mirror = data.copy()
            mirror[:, ODD] *= -1
            aug = np.concatenate((data, mirror))
            labels = np.concatenate((labels, 2 - labels))
            mean = aug.mean(0)
            mean[ODD] = 0
            scale = np.maximum(aug.std(0), 1e-8)
            same(f["mean"], mean)
            same(f["scale"], scale)
            z = np.clip((aug - mean) / scale, -8, 8)
            w = np.asarray(f["coef"])
            b = np.asarray(f["intercept"])
            residual = softmax(z @ w.T + b) - np.eye(3)[labels]
            gradient = max(
                np.max(abs(residual.T @ z / len(z) + w / (0.1 * len(z)))),
                np.max(abs(residual.mean(0))),
            )
            max_gradient = max(max_gradient, float(gradient))
            check(gradient < 1e-5, "linear fit stationarity")
            for a, s in enumerate(ASSETS):
                mask = (saved[a]["anchor"] >= cutoff) & (
                    saved[a]["anchor"] < epoch(year, month + 1)
                )
                same(saved[a]["anchor"][mask], t[test])
                same(
                    saved[a]["probability"][mask],
                    softmax(np.clip((x[a, test] - mean) / scale, -8, 8) @ w.T + b),
                )
                key = f"{year}_{month}_{s}"
                basekeys.add(key)
                base, prior = check_baseline(
                    baselines[key], *states[a], y[a], train, test, cutoff
                )
                same(saved[a]["frequency"][mask], base)
                same(saved[a]["prior_reversal"][mask], prior)
        for a, s in enumerate(ASSETS):
            actual = dict(np.load(folder / f"{year}-{s}-outcomes.npz"))
            same(actual["anchor"], t[alltest])
            same(actual["end"], ends[alltest])
            same(actual["label"], y[a, alltest])
            q = saved[a]
            verify_metrics(
                results["cohorts"][f"{year}-{s}"],
                y[a, alltest],
                q["probability"],
                q["frequency"],
                q["prior_reversal"],
                t[alltest],
            )
    check(
        fitkeys == set(fits) and basekeys == set(baselines),
        "all replication fits checked",
    )
    check(
        results["replication_gate"]
        == all(v["gate"] for v in results["cohorts"].values()),
        "replication gate",
    )
    if raw:
        raw_sources(folder, sources, datasets)
    return (t, ends, x, y), {
        "verified": True,
        "cohort_comparisons": 6,
        "fits": 14,
        "baseline_fits": 42,
        "max_objective_gradient": max_gradient,
        "raw_sources_verified": raw,
        "replication_gate": results["replication_gate"],
    }


def tree_raw(f, x):
    logits = np.broadcast_to(f["initial"], (len(x), 3)).copy()
    check(len(f["trees"]) == 120, "tree iteration count")
    for group in f["trees"]:
        check(len(group) == 3, "multiclass trees")
        for label, nodes in enumerate(group):
            check(len(nodes) <= 13 and np.isfinite(nodes).all(), "bounded finite trees")
            stack = [(0, np.arange(len(x)))]
            while stack:
                node, rows = stack.pop()
                value, feature, threshold, left, right, leaf = nodes[node]
                if leaf:
                    logits[rows, label] += value
                else:
                    feature, left, right = int(feature), int(left), int(right)
                    check(
                        0 <= feature < 23
                        and node < left < len(nodes)
                        and node < right < len(nodes),
                        "tree topology",
                    )
                    go_left = x[rows, feature] <= threshold
                    stack.extend(((left, rows[go_left]), (right, rows[~go_left])))
    return softmax(logits)


def audit_trees(root, old):
    folder = root / "trees"
    p = bundle(folder)
    check(
        p["prior_results_sha256"] == digest(root / "history/results.json"),
        "nonlinear chronology",
    )
    fits = json.loads(gzip.decompress((folder / "fits.json.gz").read_bytes()))
    results = json.loads((folder / "results.json").read_text())
    checked = set()
    for year in (2024, 2025, 2026):
        t, ends, x, y = reconstruct() if year == 2026 else old
        alltest = (t >= epoch(year, 1)) & (t < epoch(year, 8))
        saved = {
            a: dict(np.load(folder / f"{year}-{s}-predictions.npz"))
            for a, s in enumerate(ASSETS)
        }
        for month in range(1, 8):
            cutoff = epoch(year, month)
            key = f"{year}_{month}"
            checked.add(key)
            f = fits[key]
            train = (t >= cutoff - 90 * 86400) & (ends <= cutoff)
            test = (t >= cutoff) & (t < epoch(year, month + 1))
            check(
                f["cutoff"] == cutoff and f["n"] == train.sum() * 3,
                "nonlinear fit scope",
            )
            ids = np.concatenate(
                [np.column_stack((np.full(train.sum(), a), t[train])) for a in range(3)]
            ).astype("<i8")
            check(
                hashlib.sha256(ids.tobytes()).hexdigest() == f["training_ids_sha256"],
                "nonlinear training ids",
            )
            labels = np.concatenate([y[a, train] for a in range(3)])
            check(
                hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest()
                == f["training_labels_sha256"],
                "nonlinear training labels",
            )
            augmented = np.concatenate((labels, 2 - labels))
            freq = np.bincount(augmented, minlength=3) / len(augmented)
            same(softmax(np.asarray(f["initial"])[None, :])[0], freq)
            for a in range(3):
                data = x[a, test]
                mirror = data.copy()
                mirror[:, ODD] *= -1
                prediction = (tree_raw(f, data) + tree_raw(f, mirror)[:, ::-1]) / 2
                mask = (saved[a]["anchor"] >= cutoff) & (
                    saved[a]["anchor"] < epoch(year, month + 1)
                )
                same(saved[a]["anchor"][mask], t[test])
                same(saved[a]["probability"][mask], prediction)
        for a, s in enumerate(ASSETS):
            baseline = dict(
                np.load(
                    root / f"{s}-predictions.npz"
                    if year == 2026
                    else root / "history" / f"{year}-{s}-predictions.npz"
                )
            )
            same(saved[a]["anchor"], t[alltest])
            same(baseline["anchor"], t[alltest])
            verify_metrics(
                results["cohorts"][f"{year}-{s}"],
                y[a, alltest],
                saved[a]["probability"],
                baseline["frequency"],
                baseline["prior_reversal"],
                t[alltest],
            )
    check(checked == set(fits), "all nonlinear fits checked")
    check(
        results["all_cohorts_gate"]
        == all(v["gate"] for v in results["cohorts"].values()),
        "all-cohorts nonlinear gate",
    )
    return {
        "verified": True,
        "cohort_comparisons": 9,
        "fits": 21,
        "greedy_tree_fitting_rerun": False,
        "all_cohorts_gate": results["all_cohorts_gate"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "reports/timestral-universal-20261027"
    )
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--trees", action="store_true")
    args = parser.parse_args()
    old, history = audit_history(args.root, args.raw)
    result = {"history": history}
    if args.trees:
        result["trees"] = audit_trees(args.root, old)
    print(json.dumps(result, sort_keys=True))
