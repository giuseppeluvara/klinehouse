"""Premium contexts with explicit missing-data fallback, never forward filling."""

import numpy as np

from research.timestral_premium_direction import (  # noqa: F401
    FEATURES,
    ODD,
    combine,
    conditional_metrics,
    fit,
    predict,
    shared,
)


def premium_features(sources, anchors, spot_x):
    if tuple(sources) != shared.SYMBOLS or spot_x.shape != (3, len(anchors), 23):
        raise ValueError("three aligned registered assets required")
    eligible, indices = np.ones(len(anchors), dtype=bool), []
    for source in sources.values():
        t, p = source["time"], source["close"]
        if (
            len(t) < 121
            or t[-1] >= shared.BOUNDARY
            or np.any(np.diff(t) <= 0)
            or np.any(t % 60)
            or not np.isfinite(p).all()
        ):
            raise ValueError("invalid or protected premium source")
        j = np.searchsorted(t, anchors - 120)
        valid = (j >= 119) & (j < len(t))
        valid[valid] &= (t[j[valid]] == anchors[valid] - 120) & (
            t[j[valid]] - t[j[valid] - 119] == 119 * 60
        )
        eligible &= valid
        indices.append(j)
    own = []
    for source, j in zip(sources.values(), indices):
        j = j[eligible]
        p = np.clip(source["close"] * 10000, -10, 10)
        total = np.r_[0.0, np.cumsum(p)]
        means = [(total[j + 1] - total[j + 1 - h]) / h for h in (5, 30, 120)]
        own.append(
            np.column_stack(
                (p[j], *means, p[j] - p[j - 5], p[j] - p[j - 30], p[j] - means[2])
            )
        )
    out = []
    for a in range(3):
        other = np.mean([own[b][:, [0, 4]] for b in range(3) if b != a], axis=0)
        out.append(
            np.column_stack(
                (
                    own[a],
                    other,
                    own[a][:, 0] - other[:, 0],
                    own[a][:, 0] * np.minimum(abs(spot_x[a, eligible, 0]), 8),
                    own[a][:, 0] * np.clip(spot_x[a, eligible, 10], -4, 4),
                )
            )
        )
    return eligible, np.stack(out)
