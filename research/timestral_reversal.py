"""Sign-symmetric short-term reversal probabilities with explicit abstention."""

import numpy as np

from . import timestral_microdirection as micro


def states(data, indices):
    r = np.r_[0.0, np.diff(np.log(data["close"]))]
    sq = np.r_[0.0, np.cumsum(r * r)]
    trades = np.r_[0.0, np.cumsum(data["trades"])]
    scale = np.sqrt(np.maximum((sq[indices] - sq[indices - 60]) / 60, 1e-16))
    z = r[indices] / scale
    activity = data["trades"][indices] / np.maximum(
        (trades[indices] - trades[indices - 60]) / 60, 1.0
    )
    cell = np.searchsorted([0.5, 1.0, 2.0], np.abs(z), side="right") * 3
    cell += np.searchsorted([0.5, 1.5], activity, side="right")
    return cell, np.sign(r[indices]).astype(int)


def fit(cells, signs, labels, ends, cutoff):
    if len(labels) < 100 or np.any(ends > cutoff):
        raise ValueError("immature labels")
    # Relative classes are reversal, flat, continuation. Zero observed moves
    # receive the unconditional forecast and never trigger an actionable signal.
    y = np.where(signs < 0, 2 - labels, labels)
    valid = signs != 0
    count = np.bincount(y[valid], minlength=3)
    prior = (count + 1.0) / (valid.sum() + 3.0)
    global_count = np.bincount(labels, minlength=3)
    table = np.zeros((12, 3), dtype=int)
    np.add.at(table, (cells[valid], y[valid]), 1)
    return {
        "cutoff": int(cutoff),
        "n": len(labels),
        "relative_counts": count.tolist(),
        "frequency": ((global_count + 1.0) / (len(labels) + 3.0)).tolist(),
        "pooled": prior.tolist(),
        "counts": table.tolist(),
        "table": ((table + 100 * prior) / (table.sum(axis=1)[:, None] + 100)).tolist(),
    }


def predict(fit, cells, signs, anchors):
    if np.any(anchors < fit["cutoff"]) or np.any(anchors >= micro.BOUNDARY):
        raise ValueError("future fit or protected anchor")
    base = np.tile(fit["frequency"], (len(cells), 1))
    pooled = np.tile(fit["pooled"], (len(cells), 1))
    conditional = np.asarray(fit["table"])[cells].copy()
    for a in (pooled, conditional):
        a[signs < 0] = a[signs < 0, ::-1]
        a[signs == 0] = base[signs == 0]
    selected = (signs != 0) & (conditional[:, [0, 2]].max(axis=1) >= 0.52)
    return np.stack((base, pooled, conditional), axis=1), selected


def evaluate(y, p, anchors, selected):
    names = ("frequency", "pooled", "conditional")
    brier = ((p - np.eye(3)[y, None, :]) ** 2).sum(axis=2)
    correct = p.argmax(axis=2) == y[:, None]
    result = {
        "n": len(y),
        "selected": int(selected.sum()),
        "coverage": float(selected.mean()),
    }
    months = anchors.astype("datetime64[s]").astype("datetime64[M]")
    result["monthly"] = {}
    for month in np.unique(months):
        mask = (months == month) & selected
        result["monthly"][str(month)] = {
            "n": int(mask.sum()),
            "accuracy": float(correct[mask, 2].mean()) if mask.any() else None,
            "gain_pct": float(
                100
                * (brier[mask, 0].mean() - brier[mask, 2].mean())
                / brier[mask, 0].mean()
            )
            if mask.any()
            else None,
        }
    result["metrics"] = {}
    for j, name in enumerate(names):
        result["metrics"][name] = {
            "all_brier": float(brier[:, j].mean()),
            "selected_brier": float(brier[selected, j].mean())
            if selected.any()
            else None,
            "selected_accuracy": float(correct[selected, j].mean())
            if selected.any()
            else None,
            "selected_log_loss": float(
                -np.log(p[selected, j][np.arange(selected.sum()), y[selected]]).mean()
            )
            if selected.any()
            else None,
        }
    if not selected.any():
        result["gate"] = False
        return result
    unique, inverse = np.unique(anchors // 86400, return_inverse=True)
    counts = np.bincount(inverse, weights=selected)
    sums = np.stack(
        [np.bincount(inverse, weights=brier[:, j] * selected) for j in range(3)]
        + [np.bincount(inverse, weights=correct[:, 2] * selected)],
        axis=1,
    )
    rng = np.random.default_rng(271028)
    starts = rng.integers(
        0, len(unique) - 6, size=(2000, int(np.ceil(len(unique) / 7)))
    )
    ids = (starts[:, :, None] + np.arange(7)).reshape(2000, -1)[:, : len(unique)]
    den = counts[ids].sum(axis=1)
    if np.any(den == 0):
        raise ValueError("insufficient selected blocks")
    boot = sums[ids].sum(axis=1) / den[:, None]
    result["accuracy_ci95"] = np.quantile(boot[:, 3], [0.025, 0.975]).tolist()
    result["gain_ci95_pct"] = np.quantile(
        100 * (boot[:, 0] - boot[:, 2]) / boot[:, 0], [0.025, 0.975]
    ).tolist()
    result["selected_direction_counts"] = {
        str(k): int(np.sum(selected & (p[:, 2].argmax(axis=1) == k))) for k in (0, 2)
    }
    c, b = result["metrics"]["conditional"], result["metrics"]["frequency"]
    result["gate"] = bool(
        result["coverage"] >= 0.05
        and c["selected_accuracy"] >= 0.52
        and result["accuracy_ci95"][0] > 0.5
        and result["gain_ci95_pct"][0] > 0
        and c["selected_log_loss"] <= b["selected_log_loss"]
        and sum(v["n"] >= 100 and v["gain_pct"] > 0 for v in result["monthly"].values())
        >= 5
        and min(result["selected_direction_counts"].values()) >= 100
    )
    return result
