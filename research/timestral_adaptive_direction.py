"""Causal daily calibration of archived out-of-sample direction forecasts."""

import hashlib

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

BOUNDARY = 1785542400
DAY = 86400
PENALTY = 1e-4


def validate(p, anchor, end):
    if (
        p.shape != (len(anchor), 3)
        or end.shape != anchor.shape
        or not len(anchor)
        or not np.isfinite(p).all()
        or np.any(p <= 0)
        or np.any(p >= 1)
        or not np.allclose(p.sum(1), 1, atol=1e-12, rtol=0)
        or np.any(np.diff(anchor) <= 0)
        or np.any(anchor % 300)
        or np.any(end != anchor + 360)
        or np.any(end > BOUNDARY)
    ):
        raise ValueError("invalid probabilities, chronology or protected outcome")


def transform(p, alpha, offset):
    if not 0 <= alpha <= 1 or not -3 <= offset <= 3:
        raise ValueError("calibration parameters outside registered bounds")
    up = expit(alpha * np.log(p[:, 2] / p[:, 0]))
    flat = expit(logit(p[:, 1]) + offset)
    return np.column_stack(((1 - flat) * (1 - up), flat, (1 - flat) * up))


def fit(p, y, anchor, end, cutoff):
    validate(p, anchor, end)
    if (
        y.shape != anchor.shape
        or not np.isin(y, [0, 1, 2]).all()
        or len(y) < 1000
        or np.any(end > cutoff)
        or np.any(anchor < cutoff - 28 * DAY)
        or cutoff % DAY
    ):
        raise ValueError("calibration needs matured past-only observations")
    w = np.exp2(-(cutoff - end) / (7 * DAY))
    w /= w.sum()
    direction = y != 1
    x = np.log(p[:, 2] / p[:, 0])
    z = logit(p[:, 1])

    def da(a):
        return np.sum(w * direction * x * (expit(a * x) - (y == 2))) + 2 * PENALTY * (
            a - 1
        )

    def db(b):
        return np.sum(w * (expit(z + b) - (y == 1))) + 2 * PENALTY * b

    def solve(gradient, low, high):
        if gradient(low) >= 0:
            return low
        if gradient(high) <= 0:
            return high
        return brentq(gradient, low, high, xtol=1e-12)

    a, b = solve(da, 0, 1), solve(db, -3, 3)
    return {
        "cutoff": int(cutoff),
        "n": len(y),
        "alpha": float(a),
        "offset": float(b),
        "max_end": int(end.max()),
        "ids_sha256": hashlib.sha256(anchor.astype("<i8").tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(y.astype("<i8").tobytes()).hexdigest(),
    }


def replay(p, y, anchor, end):
    validate(p, anchor, end)
    if y.shape != anchor.shape or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("invalid outcome labels")
    adaptive, fixed = p.copy(), p.copy()
    states, first = [], None
    first_day = int(anchor[0] // DAY * DAY)
    for day in np.unique(anchor // DAY) * DAY:
        if day < first_day + 7 * DAY:
            continue
        train = (anchor >= day - 28 * DAY) & (end <= day)
        state = fit(p[train], y[train], anchor[train], end[train], int(day))
        states.append(state)
        if first is None:
            first = state
        current = anchor // DAY == day // DAY
        adaptive[current] = transform(p[current], state["alpha"], state["offset"])
        fixed[current] = transform(p[current], first["alpha"], first["offset"])
    return adaptive, fixed, states


def compare(y, p, raw, anchor):
    """Paired full-sample gains; family-wise intervals across nine cohorts."""
    ix = np.arange(len(y))
    target = np.eye(3)[y]
    delta = np.column_stack(
        (
            np.square(raw - target).sum(1) - np.square(p - target).sum(1),
            np.log(p[ix, y]) - np.log(raw[ix, y]),
        )
    )
    days, inv = np.unique(anchor // DAY, return_inverse=True)
    if np.any(np.diff(days) != 1):
        raise ValueError("contiguous evaluation days required")
    counts = np.bincount(inv)
    sums = np.column_stack([np.bincount(inv, weights=delta[:, j]) for j in range(2)])
    starts = np.random.default_rng(271031).integers(
        0, len(days) - 6, size=(4000, int(np.ceil(len(days) / 7)))
    )
    ids = (starts[:, :, None] + np.arange(7)).reshape(4000, -1)[:, : len(days)]
    boot = sums[ids].sum(1) / counts[ids].sum(1)[:, None]
    interval = np.quantile(boot, [0.05 / 18, 1 - 0.05 / 18], axis=0)
    selected = np.maximum(p[:, 0], p[:, 2]) >= 0.52
    return {
        "brier_gain": float(delta[:, 0].mean()),
        "logloss_gain": float(delta[:, 1].mean()),
        "brier_gain_ci99_444": interval[:, 0].tolist(),
        "logloss_gain_ci99_444": interval[:, 1].tolist(),
        "selected_confidence_minus_accuracy": float(
            (p.max(1)[selected] - (p.argmax(1) == y)[selected]).mean()
        )
        if selected.any()
        else None,
        "probability_gate": bool((interval[0] > 0).all()),
    }
