"""One-coefficient depth-information hypothesis for a separate future test.

Only the newly collected first 24 hours may fit this coefficient. This module
does not collect data, load a historical reader, or alter the running pilot.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

RIDGE = 0.25
LIMIT = 8.0
SCHEMA = "klinehouse-depth-information-one-coefficient-v1"


def epoch(value):
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or value < 0
        or value != int(value)
        or value % 300
    ):
        raise ValueError("finite integral five-minute timestamp required")
    return int(value)


def fit_first_day(anchors, baseline, imbalance, labels, received, *, first_anchor):
    """Fit a shared signed QI20 correction to issued baseline conditional logits.

    Inputs contain only complete three-asset book/forecast anchors, chosen by
    availability before observing outcomes. Missing anchors are documented by
    the caller, never backfilled. Equal total weight per moving anchor accounts
    for flat outcomes without class balancing. Insufficient data returns beta0.
    """
    first_anchor = epoch(first_anchor)
    anchors = np.asarray(anchors)
    q, x, y, at = (np.asarray(v) for v in (baseline, imbalance, labels, received))
    cutoff = first_anchor + 86400
    n = len(anchors)
    if (
        anchors.shape != (n,)
        or not np.isfinite(anchors).all()
        or np.any(anchors != np.floor(anchors))
        or any(v.shape != (n, 3) for v in (q, x, y, at))
        or np.any(np.diff(anchors) <= 0)
        or np.any(anchors % 300)
        or np.any(anchors < first_anchor)
        or np.any(anchors + 360 > cutoff)
        or not all(np.isfinite(v).all() for v in (q, x, y, at))
        or np.any((q <= 0) | (q >= 1))
        or np.any(abs(x) > 1)
        or not np.isin(y, [0, 1, 2]).all()
        or np.any(at < anchors[:, None] + 360)
        or np.any(at > cutoff)
    ):
        raise ValueError(
            "only complete, causal first-day training observations allowed"
        )
    moving = y != 1
    counts = moving.sum(0)
    result = {
        "schema": SCHEMA,
        "cutoff": int(cutoff),
        "first_anchor": int(first_anchor),
        "ridge": RIDGE,
        "coefficient": 0.0,
        "training_anchors": n,
        "moving_labels_per_asset": counts.tolist(),
        "status": "insufficient_training",
    }
    if counts.sum() < 300 or np.any(counts < 100):
        return result
    total = moving.sum(1)
    weight = np.divide(1.0, total, out=np.zeros(n), where=total > 0)
    w = np.broadcast_to(weight[:, None], moving.shape)[moving]
    w /= w.sum()
    xx, yy, offset = x[moving], y[moving] == 2, logit(q[moving])

    def gradient(beta):
        return float(np.sum(w * xx * (expit(offset + beta * xx) - yy)) + RIDGE * beta)

    beta = (
        -LIMIT
        if gradient(-LIMIT) >= 0
        else LIMIT
        if gradient(LIMIT) <= 0
        else brentq(gradient, -LIMIT, LIMIT, xtol=1e-13)
    )
    result.update(
        coefficient=float(beta),
        status="frozen_after_first_day",
        gradient=float(gradient(beta)),
    )
    return result


def predict(state, probability, imbalance, *, anchor):
    anchor = epoch(anchor)
    first = epoch(state["first_anchor"])
    cutoff = epoch(state["cutoff"])
    p, x = np.asarray(probability, dtype=float), np.asarray(imbalance, dtype=float)
    beta = state["coefficient"]
    if (
        state.get("schema") != SCHEMA
        or state.get("status")
        not in ("frozen_after_first_day", "insufficient_training")
        or cutoff != first + 86400
        or anchor < cutoff
        or p.ndim != 2
        or p.shape[1] != 3
        or x.shape != (len(p),)
        or not np.isfinite(p).all()
        or not np.isfinite(x).all()
        or np.any((p <= 0) | (p >= 1))
        or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0)
        or np.any(abs(x) > 1)
        or not np.isfinite(beta)
        or abs(beta) > LIMIT
        or state["ridge"] != RIDGE
    ):
        raise ValueError(
            "frozen causal model, interior baseline and bounded QI20 required"
        )
    q = p[:, 2] / (p[:, 0] + p[:, 2])
    adjusted = expit(logit(q) + beta * x)
    return np.column_stack(
        ((1 - p[:, 1]) * (1 - adjusted), p[:, 1], (1 - p[:, 1]) * adjusted)
    )
