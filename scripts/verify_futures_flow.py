#!/usr/bin/env python3
"""Independent futures flow, fit-gradient and saved-score audit; no fitting."""

import argparse
import csv
from datetime import datetime, timezone
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
    check,
    same,
    digest,
    reconstruct,
    verify_metrics,
)  # noqa: E402

from scripts.verify_premium_direction import binary_fit, conditional  # noqa: E402

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
        path = ROOT / "research/out/futures-flow-20261027/raw" / name
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
                    and 0 < low <= o <= h
                    and low <= c <= h,
                    "futures OHLC",
                )
                volume, buys, trades = float(row[5]), float(row[9]), int(row[8])
                check(
                    np.isfinite([volume, buys, trades]).all()
                    and 0 <= buys <= volume
                    and trades >= 0,
                    "raw volume constraints",
                )
                rows.append((t // 1000, volume, buys, trades))
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
        for k, column in (("volume", 1), ("taker", 2), ("trades", 3)):
            same(d[k][indices], a[:, column])


def features(root, year, futures):
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
    in_range = (j >= 120) & (j < len(grid))
    jj = j[in_range]
    own = []
    for symbol in ASSETS:
        source = futures[f"{year}-{symbol}"]
        volume, buy, trades = [np.full(len(grid), np.nan) for _ in range(3)]
        for dest, k in ((volume, "volume"), (buy, "taker"), (trades, "trades")):
            dest[np.searchsorted(grid, source["time"])] = source[k]
        va, na = [
            np.clip(
                np.log(
                    np.maximum(
                        z[jj]
                        / np.maximum(
                            np.convolve(z, np.ones(120) / 120, "valid")[jj - 120], floor
                        ),
                        1e-6,
                    )
                ),
                -4,
                4,
            )
            for z, floor in ((volume, 1e-12), (trades, 1.0))
        ]
        flows = []
        for h in (1, 5, 30, 120):
            v = np.convolve(volume, np.ones(h), "valid")[jj - h + 1]
            b = np.convolve(buy, np.ones(h), "valid")[jj - h + 1]
            flows.append(np.divide(2 * b - v, v, out=np.zeros(len(jj)), where=v > 0))
        values = np.full((len(t), 6), np.nan)
        values[in_range] = np.column_stack((va, na, *flows))
        own.append(values)
    available = np.isfinite(np.stack(own)).all(axis=(0, 2))
    additions = np.zeros((3, len(t), 12))
    for a in range(3):
        va, na, f1, f5, f30, f120 = own[a][available].T
        b, c = [i for i in range(3) if i != a]
        other = (own[b][available, 2] + own[c][available, 2]) / 2
        spot = x[a, available]
        additions[a, available] = np.column_stack(
            (
                spot[:, 0] * va,
                spot[:, 1] * va,
                spot[:, 4] * na,
                spot[:, 0] * na,
                f1,
                f5,
                f30,
                f120,
                f1 - f30,
                f1 * va,
                other,
                f1 - other,
            )
        )
    return t, end, np.concatenate((x, additions), axis=2), y, available


def verify(root, raw=False):
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
            path = root / f"{key}-futures.npz"
            check(digest(path) == provenance["arrays"][path.name], "source array hash")
            arrays[key] = dict(np.load(path))
            source_rows = sum(
                r["rows"]
                for r in provenance["sources"]
                if f"/{s}/" in r["url"]
                and int(r["url"].rsplit("-", 2)[-2])
                + (int(r["url"].rsplit("-", 1)[-1][:2]) >= 10)
                == year
            )
            check(
                len(arrays[key]["time"]) == source_rows,
                "all raw records retained without synthesis",
            )
            check(
                np.isin(
                    arrays[key]["time"],
                    np.arange(epoch(year - 1, 10), epoch(year, 8), 60),
                ).all()
                and np.all(np.diff(arrays[key]["time"]) > 0),
                "native futures chronology",
            )
            check(
                all(
                    np.isfinite(arrays[key][k]).all()
                    for k in ("volume", "taker", "trades")
                )
                and (arrays[key]["volume"] >= arrays[key]["taker"]).all()
                and (arrays[key]["taker"] >= 0).all()
                and (arrays[key]["trades"] >= 0).all()
                and (arrays[key]["trades"] == np.floor(arrays[key]["trades"])).all()
                and arrays[key]["time"][-1] < 1785542400,
                "futures values and protected boundary",
            )
    if raw:
        raw_audit(root, provenance["sources"], arrays)
    fits = json.loads((root / "fits.json").read_text())
    results = json.loads((root / "results.json").read_text())
    maximum, count = 0.0, 0
    for year in (2024, 2025, 2026):
        t, end, x, y, available = features(root, year, arrays)
        predictions = {name: [] for name in ("spot", "volume", "flow")}
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
            for name, columns in (("spot", 23), ("volume", 27), ("flow", 35)):
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
                if name != "spot":
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
            same(saved["futures_available"], available[indices])
            check(
                results["cohorts"][key]["futures_missing_contexts"]
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
            for name in ("spot", "volume", "flow"):
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
                ("volume", "spot"),
                ("flow", "spot"),
                ("flow", "volume"),
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
            for c in ("flow_vs_spot", "flow_vs_volume")
        ),
        "combined information gate",
    )
    check(
        results["all_direction_gates"]
        == all(r["flow"]["gate"] for r in results["cohorts"].values()),
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
        default=ROOT / "reports/timestral-futures-flow-20261027",
    )
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.raw), sort_keys=True))
