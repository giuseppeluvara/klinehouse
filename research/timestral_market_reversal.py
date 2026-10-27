"""One-coefficient common-market reversal; unchanged scalar shrinkage fitter."""

import numpy as np
from research.timestral_simple_direction import fit, predict, calibration  # noqa: F401

ARMS = (("market1", 0.25), ("market60_unshrunk", 0.0), ("market60", 0.25))


def signal(x, name):
    if x.ndim < 2 or x.shape[-1] < 17 or not np.isfinite(x).all():
        raise ValueError("finite causal spot features required")
    if name not in dict(ARMS):
        raise ValueError("unregistered market signal")
    own, others = (0, 13) if name == "market1" else (3, 16)
    return np.clip(-(x[..., own] + 2 * x[..., others]) / 3, -3, 3)
