"""One-hour price-path centers with fixed volume and uniform weighting."""

import numpy as np

from research.timestral_universal import BOUNDARY, SYMBOLS


def centers(datasets, anchors):
    """Use closed minute close prices, not trade-level execution VWAP."""
    if tuple(datasets) != SYMBOLS:
        raise ValueError("three ordered assets required")
    own_uniform, own_volume = [], []
    for d in datasets.values():
        t, price, volume = (d[k] for k in ("time", "close", "volume"))
        if (
            len(t) < 62
            or t[-1] >= BOUNDARY
            or np.any(np.diff(t) != 60)
            or not all(np.isfinite(z).all() for z in (t, price, volume))
            or np.any(price <= 0)
            or np.any(volume < 0)
        ):
            raise ValueError("complete finite pre-protected source required")
        i = np.searchsorted(t, anchors - 60)
        if np.any(i < 61) or np.any(i >= len(t)) or np.any(t[i] != anchors - 60):
            raise ValueError("closed source and full trailing context required")
        log = np.log(price)
        log -= log[0]
        r = np.r_[0.0, np.diff(log)]

        def window(z, ends=i):
            views = np.lib.stride_tricks.sliding_window_view(z, 60)
            out = np.empty(len(ends))
            for start in range(0, len(ends), 1024):
                stop = start + 1024
                out[start:stop] = views[ends[start:stop] - 59].sum(axis=1)
            return out

        scale = np.sqrt(np.maximum(window(r * r, i - 1) / 60, 1e-16))

        uniform = window(log) / 60
        total_volume = window(volume)
        weighted = np.divide(
            window(volume * log),
            total_volume,
            out=uniform.copy(),
            where=total_volume > 0,
        )
        for result, center in ((own_uniform, uniform), (own_volume, weighted)):
            result.append(np.clip(-2 * (log[i] - center) / scale / np.sqrt(60), -8, 8))
    return {
        "uniform": np.clip(np.mean(own_uniform, axis=0), -3, 3),
        "volume": np.clip(np.mean(own_volume, axis=0), -3, 3),
    }


def signals(x, center):
    common = np.clip(-(x[..., 3] + 2 * x[..., 16]) / 3, -3, 3)
    own = np.clip(-x[..., 3], -3, 3)
    agree = common * own > 0
    return {
        name: np.where(agree, value[None, :], common) for name, value in center.items()
    }, agree
