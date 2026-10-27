"""Uncertainty-adaptive shrinkage of a single frozen common-hour coefficient."""

import numpy as np
from scipy.special import expit
from research.timestral_universal import BOUNDARY


def adjust(raw, z, y, anchors, ends, cutoff):
    if (
        len(y) != raw["n"]
        or len(y) < 100
        or raw["cutoff"] != cutoff
        or raw["penalty"] != 0
        or not 0 <= raw["beta"] <= 8
        or any(a.shape != y.shape for a in (z, anchors, ends))
        or not np.isfinite(z).all()
        or np.any(abs(z) > 3)
        or not np.isin(y, [0, 1]).all()
        or np.any(ends != anchors + 360)
        or np.any(ends > cutoff)
        or np.any(anchors < cutoff - 90 * 86400)
        or cutoff >= BOUNDARY
    ):
        raise ValueError("frozen raw fit and matured ninety-day samples required")
    days, inv, counts = np.unique(
        anchors // 86400, return_inverse=True, return_counts=True
    )
    if len(days) < 80 or np.any(np.diff(days) != 1):
        raise ValueError("at least80 consecutive training days required")
    beta = raw["beta"]
    p = expit(beta * z)
    scores = np.bincount(inv, weights=z * (p - y)) / counts
    scores -= scores.mean()
    omega = scores @ scores
    for lag in range(1, 7):
        omega += 2 * (1 - lag / 7) * (scores[lag:] @ scores[:-lag])
    hessian = float(np.mean(np.bincount(inv, weights=z * z * p * (1 - p)) / counts))
    variance = float(max(omega, 0) / len(days) ** 2 / max(hessian, 1e-12) ** 2)
    weight = (
        max(0.0, 1 - variance / max(beta * beta, 1e-24)) if hessian > 1e-12 else 0.0
    )
    return dict(
        cutoff=int(cutoff),
        n=len(y),
        days=len(days),
        raw_beta=beta,
        beta=float(weight * beta),
        variance=variance,
        hessian=hessian,
        weight=float(weight),
    )


def predict(state, z, anchors):
    if (
        z.shape != anchors.shape
        or not np.isfinite(z).all()
        or np.any(abs(z) > 3)
        or np.any(anchors < state["cutoff"])
        or np.any(anchors + 360 > BOUNDARY)
        or not 0 <= state["beta"] <= 8
    ):
        raise ValueError("invalid or protected prediction")
    return expit(state["beta"] * z)
