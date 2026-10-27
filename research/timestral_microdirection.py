"""Frozen short-horizon direction experiment; no Timestral weights are used.

Hierarchical categorical frequencies expose the contribution of minute flow and
the last ten seconds of flow without a neural model or hyperparameter search.
"""

import numpy as np

NAMES = ("frequency", "price", "flow", "seconds")
BOUNDARY = int(np.datetime64("2026-08-01", "s").astype(int))


def features(data):
    """Index i is the final observed minute; anchor is its end, i + 1 minute."""
    t, close, volume, taker, trades, tail = (
        data[k] for k in ("time", "close", "volume", "taker", "trades", "tail_flow")
    )
    if len(t) < 62 or np.any(np.diff(t) <= 0) or t[-1] >= BOUNDARY:
        raise ValueError("ordered, pre-protected minute data required")
    if not all(np.isfinite(data[k]).all() for k in data):
        raise ValueError("nonfinite source")
    if (
        np.any(close <= 0)
        or np.any(volume < 0)
        or np.any(taker < 0)
        or np.any(taker > volume + 1e-8)
    ):
        raise ValueError("invalid prices or flow")
    r = np.r_[0.0, np.diff(np.log(close))]
    # At i use sixty returns ending at i-1: no current-return self-normalization.
    sq = np.r_[0.0, np.cumsum(r * r)]
    activity = np.r_[0.0, np.cumsum(trades)]
    i = np.arange(61, len(t))
    scale = np.sqrt(np.maximum((sq[i] - sq[i - 60]) / 60, 1e-16))
    intensity = trades[i] / np.maximum((activity[i] - activity[i - 60]) / 60, 1.0)
    imbalance = np.divide(
        2 * taker[i] - volume[i], volume[i], out=np.zeros(len(i)), where=volume[i] > 0
    )
    price = np.searchsorted([-1.0, -0.25, 0.25, 1.0], r[i] / scale, side="right") * 3
    price += np.searchsorted([0.5, 1.5], intensity, side="right")
    flow = price * 5 + np.searchsorted([-0.6, -0.2, 0.2, 0.6], imbalance, side="right")
    seconds = flow * 3 + np.searchsorted([-0.3, 0.3], tail[i], side="right")
    # Missing minutes invalidate the entire context, not just the current bar.
    valid = t[i] - t[i - 61] == 61 * 60
    return i[valid], np.stack((price, flow, seconds), axis=1)[valid]


def samples(data, horizon, delay):
    if horizon not in (1, 5, 15) or delay not in (0, 1):
        raise ValueError("unregistered horizon or delay")
    i, cells = features(data)
    t = data["time"]
    anchor = t[i] + 60
    ok = (i + delay + horizon < len(t)) & ((anchor // 60) % horizon == 0)
    i, cells, anchor = i[ok], cells[ok], anchor[ok]
    end = i + delay + horizon
    ok = (t[end] - t[i] == (delay + horizon) * 60) & (t[end] + 60 <= BOUNDARY)
    i, cells, anchor, end = i[ok], cells[ok], anchor[ok], end[ok]
    # Delay 1 excludes the complete first minute after the prediction anchor.
    delta = data["close"][end] - data["close"][i + delay]
    label = np.sign(delta).astype(int) + 1  # down, exactly flat, up; no dropped ties
    return {
        "index": i,
        "anchor": anchor,
        "end": t[end] + 60,
        "cells": cells,
        "label": label,
        "return": np.log(data["close"][end] / data["close"][i + delay]),
    }


def fit(cells, labels, ends, cutoff):
    if len(labels) < 100 or np.any(ends > cutoff):
        raise ValueError("fit requires matured labels and sufficient history")
    counts = np.bincount(labels, minlength=3)
    base = (counts + 1.0) / (len(labels) + 3.0)
    result = {
        "cutoff": int(cutoff),
        "n": len(labels),
        "counts": counts.tolist(),
        "frequency": base.tolist(),
    }
    prior = np.tile(base, (15, 1))
    for column, name, size, weight in (
        (0, "price", 15, 100.0),
        (1, "flow", 75, 100.0),
        (2, "seconds", 225, 150.0),
    ):
        counts = np.zeros((size, 3), dtype=int)
        np.add.at(counts, (cells[:, column], labels), 1)
        table = (counts + weight * prior) / (counts.sum(axis=1)[:, None] + weight)
        result[name] = {"counts": counts.tolist(), "probabilities": table.tolist()}
        prior = np.repeat(table, 5 if name == "price" else 3, axis=0)
    return result


def predict(model, cells, anchors):
    if np.any(anchors < model["cutoff"]) or np.any(anchors >= BOUNDARY):
        raise ValueError("future fit or protected anchor")
    return np.stack(
        [np.tile(model["frequency"], (len(cells), 1))]
        + [
            np.asarray(model[name]["probabilities"])[cells[:, j]]
            for j, name in enumerate(NAMES[1:])
        ],
        axis=1,
    )


def scores(labels, probabilities, anchors):
    target = np.eye(3)[labels]
    losses = np.square(probabilities - target[:, None, :]).sum(axis=2)
    result = {}
    for j, name in enumerate(NAMES):
        p = probabilities[:, j]
        pred = np.argmax(p, axis=1)
        result[name] = {
            "brier": float(losses[:, j].mean()),
            "log_loss": float(-np.log(p[np.arange(len(labels)), labels]).mean()),
            "accuracy": float(np.mean(pred == labels)),
            "up_recall": float(np.mean(pred[labels == 2] == 2)),
            "down_recall": float(np.mean(pred[labels == 0] == 0)),
        }
    # Paired moving-block bootstrap of daily sums, 7 consecutive days per block.
    day = anchors // 86400
    unique, inverse = np.unique(day, return_inverse=True)
    count = np.bincount(inverse)
    sums = np.stack(
        [np.bincount(inverse, weights=losses[:, j]) for j in range(4)], axis=1
    )
    if len(unique) < 14 or np.any(np.diff(unique) != 1):
        raise ValueError("bootstrap requires contiguous evaluation days")
    rng = np.random.default_rng(271027)
    starts = rng.integers(
        0, len(unique) - 6, size=(2000, int(np.ceil(len(unique) / 7)))
    )
    ids = (starts[:, :, None] + np.arange(7)).reshape(2000, -1)[:, : len(unique)]
    boot = sums[ids].sum(axis=1) / count[ids].sum(axis=1)[:, None]
    comparisons = {}
    for j, name in enumerate(NAMES[1:], 1):
        for k, base in enumerate(NAMES[:j]):
            gains = 100 * (boot[:, k] - boot[:, j]) / boot[:, k]
            comparisons[f"{name}_vs_{base}"] = {
                "gain_pct": 100
                * (result[base]["brier"] - result[name]["brier"])
                / result[base]["brier"],
                "ci95_pct": np.quantile(gains, [0.025, 0.975]).tolist(),
            }
    months = anchors.astype("datetime64[s]").astype("datetime64[M]")
    monthly = {}
    for month in np.unique(months):
        mask = months == month
        monthly[str(month)] = {
            name: float(losses[mask, j].mean()) for j, name in enumerate(NAMES)
        }
    # Primary: seconds, 5-minute target, one-minute delay; other runs diagnostics.
    gate = all(
        comparisons[f"seconds_vs_{base}"]["gain_pct"] >= 1.0
        and comparisons[f"seconds_vs_{base}"]["ci95_pct"][0] > 0
        and result["seconds"]["log_loss"] <= result[base]["log_loss"]
        and sum(m["seconds"] < m[base] for m in monthly.values()) >= 5
        for base in ("frequency", "price")
    )
    return {
        "n": len(labels),
        "classes": np.bincount(labels, minlength=3).tolist(),
        "metrics": result,
        "comparisons": comparisons,
        "monthly": monthly,
        "gate": bool(gate),
    }
