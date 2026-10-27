"""Causal perpetual-futures volume intensity and aggressor-flow features."""

import numpy as np

from research.timestral_premium_direction import (  # noqa: F401
    combine,
    conditional_metrics,
    fit,
    predict,
    shared,
)

FEATURES = (
    "spot_return1_futures_volume_activity",
    "spot_return5_futures_volume_activity",
    "spot_flow1_futures_trade_activity",
    "spot_return1_futures_trade_activity",
    "futures_flow1",
    "futures_flow5",
    "futures_flow30",
    "futures_flow120",
    "futures_flow1_minus_flow30",
    "futures_flow1_volume_activity",
    "other_futures_flow1",
    "futures_flow1_minus_other",
)
ARMS = (("spot", 23), ("volume", 27), ("flow", 35))
COMPARISONS = (("volume", "spot"), ("flow", "spot"), ("flow", "volume"))


def futures_features(sources, anchors, spot_x):
    if tuple(sources) != shared.SYMBOLS or spot_x.shape != (3, len(anchors), 23):
        raise ValueError("three aligned registered assets required")
    eligible, indices = np.ones(len(anchors), dtype=bool), []
    for source in sources.values():
        t, v, b, n = (source[k] for k in ("time", "volume", "taker", "trades"))
        if (
            len(t) < 121
            or t[-1] >= shared.BOUNDARY
            or np.any(np.diff(t) <= 0)
            or np.any(t % 60)
            or not all(np.isfinite(z).all() and z.shape == t.shape for z in (v, b, n))
            or np.any(v < 0)
            or np.any(b < 0)
            or np.any(b > v)
            or np.any(n < 0)
            or np.any(n != np.floor(n))
        ):
            raise ValueError("invalid or protected futures source")
        j = np.searchsorted(t, anchors - 120)
        valid = (j >= 120) & (j < len(t))
        valid[valid] &= (t[j[valid]] == anchors[valid] - 120) & (
            t[j[valid]] - t[j[valid] - 120] == 120 * 60
        )
        eligible &= valid
        indices.append(j)
    own = []
    for source, j in zip(sources.values(), indices):
        j = j[eligible]
        v, b, n = (source[k] for k in ("volume", "taker", "trades"))
        vs, bs, ns = [np.r_[0.0, np.cumsum(z)] for z in (v, b, n)]
        va, na = [
            np.clip(
                np.log(
                    np.maximum(
                        z[j] / np.maximum((total[j] - total[j - 120]) / 120, floor),
                        1e-6,
                    )
                ),
                -4,
                4,
            )
            for z, total, floor in ((v, vs, 1e-12), (n, ns, 1.0))
        ]
        flows = []
        for h in (1, 5, 30, 120):
            volume = v[j] if h == 1 else vs[j + 1] - vs[j + 1 - h]
            buys = b[j] if h == 1 else bs[j + 1] - bs[j + 1 - h]
            flows.append(
                np.divide(
                    2 * buys - volume, volume, out=np.zeros(len(j)), where=volume > 0
                )
            )
        own.append((va, na, flows))
    out = []
    for a, (va, na, f) in enumerate(own):
        other = np.mean([own[b][2][0] for b in range(3) if b != a], axis=0)
        spot = spot_x[a, eligible]
        out.append(
            np.column_stack(
                (
                    spot[:, 0] * va,
                    spot[:, 1] * va,
                    spot[:, 4] * na,
                    spot[:, 0] * na,
                    *f,
                    f[0] - f[2],
                    f[0] * va,
                    other,
                    f[0] - other,
                )
            )
        )
    return eligible, np.stack(out)


def comparison(y, p, control, frequency, anchors):
    return {
        k.replace("vs_spot", "vs_control"): v
        for k, v in conditional_metrics(y, p, control, frequency, anchors).items()
    }
