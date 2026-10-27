"""One-factor prequential calibration with uncertainty-based identity shrinkage."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from research.timestral_universal import BOUNDARY
from research.timestral_simple_direction import day_weights, calibration

EDGES = np.array([0.5, 0.51, 0.52, 0.54, 0.58, 0.66, 1.0])


def fit(logits, y, anchors, ends, issued_cutoffs, cutoff):
    if (
        len(y) < 100
        or any(a.shape != y.shape for a in (logits, anchors, ends, issued_cutoffs))
        or not np.isfinite(logits).all()
        or not np.isin(y, [0, 1]).all()
        or np.any(ends != anchors + 360)
        or np.any(ends > cutoff)
        or np.any(issued_cutoffs > anchors)
        or np.any(anchors < cutoff - 56 * 86400)
        or cutoff >= BOUNDARY
    ):
        raise ValueError("mature, prequential calibration samples required")
    days, inverse = np.unique(anchors // 86400, return_inverse=True)
    if len(days) < 42 or np.any(np.diff(days) != 1):
        raise ValueError("at least42 consecutive calibration days required")
    weights = day_weights(anchors)

    def derivative(factor):
        return float(weights @ (logits * (expit(factor * logits) - y)))

    if np.all(logits == 0):
        raw = 1.0
    elif derivative(0) >= 0:
        raw = 0.0
    elif derivative(8) <= 0:
        raw = 8.0
    else:
        raw = float(brentq(derivative, 0, 8, xtol=1e-12))
    p = expit(raw * logits)
    count = np.bincount(inverse)
    scores = np.bincount(inverse, weights=logits * (p - y)) / count
    centered = scores - scores.mean()
    omega = centered @ centered
    for lag in range(1, 7):
        omega += 2 * (1 - lag / 7) * (centered[lag:] @ centered[:-lag])
    hessian = float(weights @ (logits * logits * p * (1 - p)))
    variance = float(max(omega, 0) / len(days) ** 2 / max(hessian, 1e-12) ** 2)
    shrink = max(0.0, 1 - variance / max((raw - 1) ** 2, 1e-24))
    if hessian <= 1e-12:
        shrink = 0.0
    adjusted = 1 + shrink * (raw - 1)
    return dict(
        cutoff=int(cutoff),
        n=len(y),
        days=len(days),
        raw_factor=raw,
        adjusted_factor=float(adjusted),
        factor_variance=variance,
        shrink=float(shrink),
        hessian=hessian,
        derivative=derivative(raw),
        calibration_logloss=float(
            weights @ (np.logaddexp(0, raw * logits) - y * raw * logits)
        ),
    )


def predict(state, logits, anchors, adjusted=True):
    factor = state["adjusted_factor" if adjusted else "raw_factor"]
    if (
        logits.shape != anchors.shape
        or not np.isfinite(logits).all()
        or np.any(anchors < state["cutoff"])
        or np.any(anchors + 360 > BOUNDARY)
        or not 0 <= factor <= 8
    ):
        raise ValueError("invalid calibration prediction or protected target")
    return expit(factor * logits)


def reliability(y, p):
    result = calibration(y, p)
    keep = y != 1
    q, label = p[keep], y[keep] == 2
    confidence = np.maximum(q, 1 - q)
    correct = (q >= 0.5) == label
    ids = np.minimum(
        np.searchsorted(EDGES, confidence, side="right") - 1, len(EDGES) - 2
    )
    bins = []
    for k in range(len(EDGES) - 1):
        take = ids == k
        bins.append(
            dict(
                lower=float(EDGES[k]),
                upper=float(EDGES[k + 1]),
                n=int(take.sum()),
                confidence=float(confidence[take].mean()) if take.any() else None,
                accuracy=float(correct[take].mean()) if take.any() else None,
            )
        )
    result["bins"] = bins
    result["ece_fixed_bins"] = float(
        sum(b["n"] * abs(b["confidence"] - b["accuracy"]) for b in bins if b["n"])
        / len(q)
    )
    return result
