#!/usr/bin/env python3
"""Independent reconstruction of minute directions, counts and saved scores.

Imports no production readers, feature builders, classifiers or metric code.
Optional --raw also verifies original Parquet/ZIP source bytes and aggregation.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
END = 1785542400
START = 1767225600
NAMES = ("frequency", "price", "flow", "seconds")


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "biu" and b.dtype.kind in "biu":
        np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=2e-11)


def epoch(month):
    return int(datetime(2026, month, 1, tzinfo=timezone.utc).timestamp())


def bundle(root):
    for name, sha in json.loads((root / "manifest.json").read_text()).items():
        check(digest(root / name) == sha, f"changed artifact: {root / name}")
    p = json.loads((root / "protocol.json").read_text())
    check(
        digest(root / "protocol.json")
        == json.loads((root / "protocol.sha256.json").read_text())["sha256"],
        "protocol checksum",
    )
    for name, sha in p["code"].items():
        check(digest(ROOT / name) == sha, f"changed frozen code: {name}")
    check(
        p["at"] < json.loads((root / "opened.json").read_text())["at"],
        "registration must precede run",
    )
    return p


def reconstruct(data, h, d):
    t = data["time"]
    check(
        np.all(np.diff(t) > 0)
        and np.all(t % 60 == 0)
        and t.min() >= 1756684800
        and t.max() < END,
        "source time boundary/order",
    )
    check(all(np.isfinite(a).all() for a in data.values()), "nonfinite data")
    close = data["close"]
    check(np.all(close > 0), "invalid close")
    i = np.arange(61, len(t) - d - h)
    keep = (
        (t[i] - t[i - 61] == 3660)
        & (t[i + d + h] - t[i] == (d + h) * 60)
        & (((t[i] + 60) // 60) % h == 0)
    )
    i = i[keep]
    r = np.zeros(len(t))
    r[1:] = np.log(close[1:]) - np.log(close[:-1])
    mean_square = np.convolve(r * r, np.ones(60) / 60, mode="valid")[i - 60]
    z = r[i] / np.sqrt(np.maximum(mean_square, 1e-16))
    # Sum integral trade counts before dividing: a dot product with 1/60
    # weights can move an exact 1.5 activity ratio across the registered cut.
    mean_trades = np.convolve(data["trades"], np.ones(60), mode="valid")[i - 60] / 60
    activity = data["trades"][i] / np.maximum(mean_trades, 1.0)
    b = np.digitize(activity, [0.5, 1.5])
    price = np.digitize(z, [-1.0, -0.25, 0.25, 1.0]) * 3 + b
    v = data["volume"][i]
    flow = np.zeros(len(i))
    np.divide(2 * data["taker"][i] - v, v, out=flow, where=v > 0)
    f = price * 5 + np.digitize(flow, [-0.6, -0.2, 0.2, 0.6])
    seconds = f * 3 + np.digitize(data["tail_flow"][i], [-0.3, 0.3])
    result = {
        "anchor": t[i] + 60,
        "end": t[i + d + h] + 60,
        "cells": np.column_stack((price, f, seconds)),
        "reversal_cells": np.digitize(abs(z), [0.5, 1.0, 2.0]) * 3 + b,
        "signs": np.sign(r[i]).astype(int),
        "returns": np.log(close[i + d + h] / close[i + d]),
        "label": np.sign(close[i + d + h] - close[i + d]).astype(int) + 1,
    }
    check(result["end"].max() <= END, "protected outcome")
    return result


def predictions(s, fits, key, reversal):
    arrays = []
    selections = []
    for month in range(1, 8):
        cutoff = epoch(month)
        train = (s["anchor"] >= cutoff - 90 * 86400) & (s["end"] <= cutoff)
        test = (s["anchor"] >= cutoff) & (s["anchor"] < epoch(month + 1))
        f = fits[f"{key}_{month}"]
        y = s["label"][train]
        check(f["cutoff"] == cutoff and f["n"] == len(y), "fit cutoff/count")
        check(
            hashlib.sha256(s["anchor"][train].astype("<i8").tobytes()).hexdigest()
            == f["training_anchor_sha256"],
            "training ids",
        )
        base = (np.bincount(y, minlength=3) + 1) / (len(y) + 3)
        same(f["frequency"], base)
        if not reversal:
            same(f["counts"], np.bincount(y, minlength=3))
            tables = []
            prior = np.broadcast_to(base, (15, 3))
            for j, (name, size, weight) in enumerate(
                zip(NAMES[1:], [15, 75, 225], [100, 100, 150])
            ):
                count = np.bincount(
                    3 * s["cells"][train, j] + y, minlength=3 * size
                ).reshape(size, 3)
                same(f[name]["counts"], count)
                table = (count + weight * prior) / (count.sum(axis=1)[:, None] + weight)
                same(f[name]["probabilities"], table)
                tables.append(table[s["cells"][test, j]])
                prior = np.repeat(table, 5 if j == 0 else 3, axis=0)
            arrays.append(
                np.stack([np.broadcast_to(base, (test.sum(), 3))] + tables, axis=1)
            )
        else:
            signs = s["signs"][train]
            valid = signs != 0
            relative = np.where(signs < 0, 2 - y, y)
            counts = np.bincount(relative[valid], minlength=3)
            same(f["relative_counts"], counts)
            pooled = (counts + 1) / (valid.sum() + 3)
            same(f["pooled"], pooled)
            counts = np.bincount(
                3 * s["reversal_cells"][train][valid] + relative[valid], minlength=36
            ).reshape(12, 3)
            same(f["counts"], counts)
            table = (counts + 100 * pooled) / (counts.sum(axis=1)[:, None] + 100)
            same(f["table"], table)
            base = np.broadcast_to(base, (test.sum(), 3)).copy()
            prior = np.broadcast_to(pooled, base.shape).copy()
            condition = table[s["reversal_cells"][test]].copy()
            signs = s["signs"][test]
            for value in (prior, condition):
                value[signs < 0] = value[signs < 0][:, [2, 1, 0]]
                value[signs == 0] = base[signs == 0]
            arrays.append(np.stack([base, prior, condition], axis=1))
            selections.append(
                (signs != 0) & (np.maximum(condition[:, 0], condition[:, 2]) >= 0.52)
            )
    return np.concatenate(arrays), np.concatenate(selections) if reversal else None


def bootstrap(anchors, values, selected, seed):
    days, inv = np.unique(anchors // 86400, return_inverse=True)
    check(np.all(np.diff(days) == 1), "missing evaluation day")
    den = np.bincount(inv, weights=selected)
    sums = np.zeros((len(days), values.shape[1]))
    np.add.at(sums, inv, values * selected[:, None])
    starts = np.random.default_rng(seed).integers(
        0, len(days) - 6, size=(2000, (len(days) + 6) // 7)
    )
    ids = np.concatenate([starts[:, :, None] + j for j in range(7)], axis=2).reshape(
        2000, -1
    )[:, : len(days)]
    totals = den[ids].sum(axis=1)
    check(np.all(totals > 0), "empty bootstrap replicate")
    return sums[ids].sum(axis=1) / totals[:, None]


def metrics(y, p, t, stored, selected=None):
    # Brier via algebraic expansion, independently of one-hot implementation.
    loss = (
        (p * p).sum(axis=2)
        - 2 * np.take_along_axis(p, y[:, None, None], axis=2)[:, :, 0]
        + 1
    )
    correct = p.argmax(axis=2) == y[:, None]
    month = t.astype("datetime64[s]").astype("datetime64[M]")
    check(stored["n"] == len(y), "evaluation count")
    if selected is None:
        same(stored["classes"], np.bincount(y, minlength=3))
        for j, name in enumerate(NAMES):
            v = stored["metrics"][name]
            same(v["brier"], loss[:, j].mean())
            same(v["log_loss"], -np.log(p[np.arange(len(y)), j, y]).mean())
            same(v["accuracy"], correct[:, j].mean())
            for label, labelname in ((0, "down"), (2, "up")):
                same(v[labelname + "_recall"], correct[y == label, j].mean())
        boot = bootstrap(t, loss, np.ones(len(t), dtype=bool), 271027)
        for j, name in enumerate(NAMES[1:], 1):
            for k, base in enumerate(NAMES[:j]):
                v = stored["comparisons"][f"{name}_vs_{base}"]
                same(
                    v["gain_pct"],
                    100 * (loss[:, k].mean() - loss[:, j].mean()) / loss[:, k].mean(),
                )
                same(
                    v["ci95_pct"],
                    np.quantile(
                        100 * (boot[:, k] - boot[:, j]) / boot[:, k], [0.025, 0.975]
                    ),
                )
        for m in np.unique(month):
            for j, name in enumerate(NAMES):
                same(stored["monthly"][str(m)][name], loss[month == m, j].mean())
        gate = all(
            stored["comparisons"][f"seconds_vs_{base}"]["gain_pct"] >= 1
            and stored["comparisons"][f"seconds_vs_{base}"]["ci95_pct"][0] > 0
            and stored["metrics"]["seconds"]["log_loss"]
            <= stored["metrics"][base]["log_loss"]
            and sum(v["seconds"] < v[base] for v in stored["monthly"].values()) >= 5
            for base in ("frequency", "price")
        )
    else:
        check(stored["selected"] == selected.sum(), "selection count")
        same(stored["coverage"], selected.mean())
        for j, name in enumerate(("frequency", "pooled", "conditional")):
            v = stored["metrics"][name]
            same(v["all_brier"], loss[:, j].mean())
            same(v["selected_brier"], loss[selected, j].mean())
            same(v["selected_accuracy"], correct[selected, j].mean())
            same(
                v["selected_log_loss"],
                -np.log(p[selected, j][np.arange(selected.sum()), y[selected]]).mean(),
            )
        for m in np.unique(month):
            mask = selected & (month == m)
            v = stored["monthly"][str(m)]
            check(v["n"] == mask.sum(), "monthly selection")
            if mask.any():
                same(v["accuracy"], correct[mask, 2].mean())
                same(
                    v["gain_pct"],
                    100
                    * (loss[mask, 0].mean() - loss[mask, 2].mean())
                    / loss[mask, 0].mean(),
                )
            else:
                check(v["accuracy"] is None and v["gain_pct"] is None, "empty month")
        boot = bootstrap(t, np.column_stack((loss, correct[:, 2])), selected, 271028)
        same(stored["accuracy_ci95"], np.quantile(boot[:, 3], [0.025, 0.975]))
        same(
            stored["gain_ci95_pct"],
            np.quantile(100 * (boot[:, 0] - boot[:, 2]) / boot[:, 0], [0.025, 0.975]),
        )
        for label in (0, 2):
            check(
                stored["selected_direction_counts"][str(label)]
                == np.sum(selected & (p[:, 2].argmax(axis=1) == label)),
                "selected predicted direction",
            )
        c, b = stored["metrics"]["conditional"], stored["metrics"]["frequency"]
        gate = (
            stored["coverage"] >= 0.05
            and c["selected_accuracy"] >= 0.52
            and stored["accuracy_ci95"][0] > 0.5
            and stored["gain_ci95_pct"][0] > 0
            and c["selected_log_loss"] <= b["selected_log_loss"]
            and sum(
                v["n"] >= 100 and v["gain_pct"] > 0 for v in stored["monthly"].values()
            )
            >= 5
            and min(stored["selected_direction_counts"].values()) >= 100
        )
    check(stored["gate"] == bool(gate), "gate mismatch")


def verify_run(data, root, key, h, d, fits, stored, reversal, outcomes=None):
    s = reconstruct(data, h, d)
    test = (s["anchor"] >= START) & (s["anchor"] < END)
    saved = dict(np.load(root / f"{key}-predictions.npz"))
    pred, selected = predictions(s, fits, key, reversal)
    same(saved["anchor"], s["anchor"][test])
    same(saved["probability"], pred)
    if reversal:
        same(saved["cells"], s["reversal_cells"][test])
        same(saved["signs"], s["signs"][test])
        same(saved["selected"], selected)
    else:
        same(saved["cells"], s["cells"][test])
    if outcomes is None:
        outcomes = root / f"{key}-outcomes.npz"
    actual = dict(np.load(outcomes))
    for name in ("anchor", "end", "label", "returns"):
        same(actual[name], s[name][test])
    metrics(s["label"][test], pred, s["anchor"][test], stored, selected)


def raw_btc(protocol, data):
    import pyarrow.parquet as pq

    position = 0
    for entry in protocol["files"]:
        path = ROOT / entry["path"]
        check(digest(path) == entry["sha256"], "raw Parquet hash")
        cols = {
            k: pq.read_table(path, columns=[k])[k].to_numpy()
            for k in (
                "open_time",
                "close_time",
                "close",
                "volume",
                "taker_buy_base",
                "trades",
            )
        }
        t = cols["open_time"].astype("datetime64[us]").astype(np.int64)
        ct = cols["close_time"].astype("datetime64[us]").astype(np.int64)
        check(
            len(t) == 86400
            and np.all(np.diff(t) == 1000000)
            and np.all((ct >= t) & (ct < t + 1000000)),
            "raw seconds",
        )
        ids = np.arange(0, len(t), 60)
        sl = slice(position, position + len(ids))
        same(data["time"][sl], t[ids] // 1000000)
        same(data["close"][sl], cols["close"][ids + 59])
        for k, col in (
            ("volume", "volume"),
            ("taker", "taker_buy_base"),
            ("trades", "trades"),
        ):
            same(data[k][sl], np.add.reduceat(cols[col], ids))
        v = np.array([cols["volume"][i + 50 : i + 60].sum() for i in ids])
        b = np.array([cols["taker_buy_base"][i + 50 : i + 60].sum() for i in ids])
        same(
            data["tail_flow"][sl],
            np.divide(2 * b - v, v, out=np.zeros(len(ids)), where=v > 0),
        )
        position += len(ids)
    check(position == len(data["time"]), "raw BTC row count")


def raw_transfer(sources, datasets):
    grouped = {s: [] for s in datasets}
    for item in sources["sources"]:
        name = item["url"].rsplit("/", 1)[-1]
        path = ROOT / "research/out/direction-transfer-20261027/raw" / name
        check(digest(path) == item["sha256"], "raw native ZIP hash")
        check(
            path.with_name(name + ".CHECKSUM").read_text().split()[0] == item["sha256"],
            "native checksum receipt",
        )
        symbol, _, year, month = name.removesuffix(".zip").split("-")
        invalid = []
        with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
            for row in csv.reader(io.TextIOWrapper(f)):
                if row[0] == "open_time":
                    continue
                t, ct = int(row[0]), int(row[6])
                dt = datetime.fromtimestamp(t / 1e6, timezone.utc)
                check(
                    dt.year == int(year)
                    and dt.month == int(month)
                    and t % 60000000 == 0
                    and t < END * 1000000,
                    "raw minute time",
                )
                o, hi, lo, c, v = map(float, row[1:6])
                trades, buy = float(row[8]), float(row[9])
                valid = (
                    t <= ct < t + 60000000
                    and np.isfinite([o, hi, lo, c, v, trades, buy]).all()
                    and 0 < lo <= min(o, c) <= max(o, c) <= hi
                    and min(v, trades, buy) >= 0
                    and buy <= v + 1e-8
                )
                if valid:
                    grouped[symbol].append((t // 1000000, c, v, buy, trades))
                else:
                    invalid.append(row)
        check(invalid == item["invalid"], "invalid native rows")
    for symbol, rows in grouped.items():
        a = np.asarray(rows)
        for j, k in enumerate(("time", "close", "volume", "taker", "trades")):
            same(
                a[:, j].astype(np.int64) if k == "time" else a[:, j],
                datasets[symbol][k],
            )
        check(
            np.all(datasets[symbol]["tail_flow"] == 0),
            "transfer must not invent seconds",
        )


def main(root, raw=False):
    micro, rev, transfer = root / "micro", root / "reversal", root / "transfer"
    mp, rp, tp = [bundle(r) for r in (micro, rev, transfer)]
    data = dict(np.load(micro / "minutes.npz"))
    check(
        digest(micro / "minutes.npz")
        == json.loads((micro / "source.json").read_text())["sha256"]
        == rp["source_sha256"],
        "minute source hashes",
    )
    check(
        rp["prior_results_sha256"] == digest(micro / "results.json"),
        "follow-up context",
    )
    check(
        tp["prior_results_sha256"] == digest(rev / "results.json"), "transfer context"
    )
    fits = json.loads((micro / "fits.json").read_text())
    results = json.loads((micro / "results.json").read_text())
    for h in (1, 5, 15):
        for d in (0, 1):
            key = f"h{h}_d{d}"
            verify_run(data, micro, key, h, d, fits, results["runs"][key], False)
    check(results["primary_gate"] == results["runs"]["h5_d1"]["gate"], "micro primary")
    fits = json.loads((rev / "fits.json").read_text())
    results = json.loads((rev / "results.json").read_text())
    for h in (1, 5, 15):
        verify_run(
            data,
            rev,
            f"h{h}",
            h,
            1,
            fits,
            results["runs"][f"h{h}"],
            True,
            micro / f"h{h}_d1-outcomes.npz",
        )
    check(results["primary_gate"] == results["runs"]["h5"]["gate"], "reversal primary")
    sources = json.loads((transfer / "sources.json").read_text())
    fits = json.loads((transfer / "fits.json").read_text())
    results = json.loads((transfer / "results.json").read_text())
    datasets = {}
    check(tp["urls"] == [s["url"] for s in sources["sources"]], "source URL set")
    for symbol in ("ETHUSDT", "SOLUSDT"):
        path = transfer / f"{symbol}-minutes.npz"
        check(
            digest(path) == sources["derived"][symbol]["sha256"], "native derived hash"
        )
        datasets[symbol] = dict(np.load(path))
        verify_run(
            datasets[symbol],
            transfer,
            symbol,
            5,
            1,
            fits,
            results["runs"][symbol],
            True,
        )
    check(
        results["primary_gate"] == all(v["gate"] for v in results["runs"].values()),
        "transfer gate",
    )
    diagnostics = json.loads((root / "direction-diagnostics.json").read_text())
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        pf = (
            rev / "h5-predictions.npz"
            if symbol == "BTCUSDT"
            else transfer / f"{symbol}-predictions.npz"
        )
        of = (
            micro / "h5_d1-outcomes.npz"
            if symbol == "BTCUSDT"
            else transfer / f"{symbol}-outcomes.npz"
        )
        p, actual = dict(np.load(pf)), dict(np.load(of))
        selected, probability = p["selected"], p["probability"][:, 2]
        direction, y = probability.argmax(axis=1), actual["label"]
        matrix = np.bincount(
            3 * y[selected] + direction[selected], minlength=9
        ).reshape(3, 3)
        evidence = diagnostics[symbol]
        same(evidence["confusion_rows_actual_columns_predicted_down_flat_up"], matrix)
        same(
            evidence["mean_probability_of_selected_direction"],
            probability[selected, direction[selected]].mean(),
        )
        for label, name in ((0, "down"), (2, "up")):
            same(
                evidence[name + "_precision"],
                np.mean(y[selected & (direction == label)] == label),
            )
    preview = json.loads((root / "preview-eth.json").read_text())
    saved = dict(np.load(transfer / "ETHUSDT-predictions.npz"))
    anchor = int(datetime.fromisoformat(preview["anchor"]).timestamp())
    idx = np.flatnonzero(saved["anchor"] == anchor)
    check(len(idx) == 1, "preview anchor")
    probability = saved["probability"][idx[0], 2]
    same([preview["probabilities"][k] for k in ("down", "flat", "up")], probability)
    expected_signal = (
        ("down", "flat", "up")[probability.argmax()]
        if saved["selected"][idx[0]]
        else "abstain"
    )
    check(preview["signal"] == expected_signal, "preview signal")
    if raw:
        raw_btc(mp, data)
        raw_transfer(sources, datasets)
    return {
        "verified": True,
        "runs": 11,
        "monthly_fits": 77,
        "protected_boundary": "2026-08-01T00:00:00Z",
        "raw_sources_verified": raw,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "reports/timestral-microdirection-20261027"
    )
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    print(json.dumps(main(args.root, args.raw), sort_keys=True))
