"""One-coefficient, sign-constrained relative reversal with fixed shrinkage."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from research.timestral_universal import BOUNDARY

ARMS = (("reversal", 0.25), ("relative_unshrunk", 0.0), ("relative", 0.25))


def signal(x, relative):
    """Dimensionless spot returns are already normalized using causal past RMS."""
    if x.ndim < 2 or x.shape[-1] < 14 or not np.isfinite(x).all():
        raise ValueError("finite spot features required")
    z = (x[..., 13] - x[..., 0]) / np.sqrt(2) if relative else -x[..., 0]
    return np.clip(z, -3, 3)


def day_weights(anchors):
    _, inverse, counts = np.unique(
        anchors // 86400, return_inverse=True, return_counts=True
    )
    return 1 / (len(counts) * counts[inverse])


def fit(z, y, anchors, ends, cutoff, penalty):
    if (
        len(y) < 100
        or any(a.shape != y.shape for a in (z, anchors, ends))
        or not np.isfinite(z).all()
        or np.any(abs(z) > 3)
        or not np.isin(y, [0, 1]).all()
        or np.any(ends > cutoff)
        or np.any(ends != anchors + 360)
        or np.any(anchors < cutoff - 90 * 86400)
        or cutoff >= BOUNDARY
        or penalty not in (0.0, 0.25)
    ):
        raise ValueError("registered bounded, matured binary training samples required")
    w = day_weights(anchors)

    def derivative(beta):
        return float(w @ (z * (expit(beta * z) - y)) + penalty * beta)

    if derivative(0) >= 0:
        beta = 0.0
    elif derivative(8) <= 0:
        beta = 8.0
    else:
        beta = float(brentq(derivative, 0, 8, xtol=1e-13))
    logits = beta * z
    return dict(
        cutoff=int(cutoff),
        n=len(y),
        days=len(np.unique(anchors // 86400)),
        beta=beta,
        penalty=penalty,
        training_logloss=float(w @ (np.logaddexp(0, logits) - y * logits)),
        derivative=derivative(beta),
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
        raise ValueError("invalid prediction scope or protected target")
    return expit(state["beta"] * z)


def calibration(y, p):
    moving = y != 1
    label, q = y[moving] == 2, p[moving]
    correct = (q >= 0.5) == label
    confidence = np.maximum(q, 1 - q)
    return dict(
        n=int(moving.sum()),
        mean_confidence=float(confidence.mean()),
        confidence_q95=float(np.quantile(confidence, 0.95)),
        confidence_minus_accuracy=float((confidence - correct).mean()),
        neutral_predictions=int((q == 0.5).sum()),
    )
