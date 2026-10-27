"""Conditional up/down models with lagged perpetual-premium information."""

import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from research import timestral_universal as shared

FEATURES = (
    "premium",
    "premium_mean5",
    "premium_mean30",
    "premium_mean120",
    "premium_change5",
    "premium_change30",
    "premium_deviation120",
    "other_premium",
    "other_premium_change5",
    "premium_residual",
    "premium_abs_return1",
    "premium_activity",
)
ODD = shared.ODD + tuple(range(23, 35))


def premium_features(sources, anchors, spot_x):
    """Use a premium candle closed at least one minute before each anchor."""
    if tuple(sources) != shared.SYMBOLS:
        raise ValueError("three registered premium assets required")
    t = sources[shared.SYMBOLS[0]]["time"]
    if len(t) < 121 or np.any(np.diff(t) != 60) or t[-1] >= shared.BOUNDARY:
        raise ValueError("complete pre-protected premium minutes required")
    j = np.searchsorted(t, anchors - 120)
    valid = (j >= 119) & (j < len(t))
    valid[valid] &= t[j[valid]] == anchors[valid] - 120
    if spot_x.shape != (3, len(anchors), 23):
        raise ValueError("spot membership mismatch")
    j = j[valid]
    own = []
    for source in sources.values():
        p = source["close"]
        if not np.array_equal(source["time"], t) or not np.isfinite(p).all():
            raise ValueError("premium source misalignment")
        p = np.clip(p * 10000, -10, 10)
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
                    own[a][:, 0] * np.minimum(abs(spot_x[a, valid, 0]), 8),
                    own[a][:, 0] * np.clip(spot_x[a, valid, 10], -4, 4),
                )
            )
        )
    return valid, np.stack(out)


def fit(x, y, end, cutoff):
    if (
        len(y) < 100
        or x.shape[0] != len(y)
        or not np.isin(y, [0, 1]).all()
        or np.any(end > cutoff)
        or not np.isfinite(x).all()
    ):
        raise ValueError("matured binary direction samples required")
    odd = [j for j in ODD if j < x.shape[1]]
    mirror = x.copy()
    mirror[:, odd] *= -1
    a, b = np.concatenate((x, mirror)), np.concatenate((y, 1 - y))
    mean, scale = a.mean(0), np.maximum(a.std(0), 1e-8)
    mean[odd] = 0
    z = np.clip((a - mean) / scale, -8, 8)
    estimator = LogisticRegression(C=0.1, max_iter=500, tol=1e-8, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(z, b)
    return {
        "n": len(y),
        "cutoff": int(cutoff),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": estimator.coef_[0].tolist(),
        "intercept": float(estimator.intercept_[0]),
        "iterations": int(estimator.n_iter_[0]),
    }


def predict(state, x, anchor):
    if np.any(anchor < state["cutoff"]) or np.any(anchor + 360 > shared.BOUNDARY):
        raise ValueError("future fit or protected target")
    z = np.clip((x - state["mean"]) / state["scale"], -8, 8)
    return expit(z @ state["coef"] + state["intercept"])


def combine(up, flat):
    return np.column_stack(((1 - flat) * (1 - up), flat, (1 - flat) * up))


def conditional_metrics(y, p, spot, frequency, anchors):
    """New-information test on moving prices, separate from unconditional gates."""
    moving = y != 1
    label = y[moving] == 2
    p, spot, frequency = (q[moving] for q in (p, spot, frequency))
    t = anchors[moving]
    correct = (p >= 0.5) == label

    def loss(q):
        return np.square(q - label), -np.log(np.where(label, q, 1 - q))

    brier, ll = loss(p)
    sb, sl = loss(spot)
    fb, fl = loss(frequency)
    values = np.column_stack((correct, sb - brier, sl - ll, fb - brier, fl - ll))
    days, inv = np.unique(t // 86400, return_inverse=True)
    if np.any(np.diff(days) != 1):
        raise ValueError("complete evaluation days required")
    counts = np.bincount(inv)
    totals = np.column_stack([np.bincount(inv, weights=values[:, j]) for j in range(5)])
    starts = np.random.default_rng(271032).integers(
        0, len(days) - 6, (4000, (len(days) + 6) // 7)
    )
    ids = (starts[:, :, None] + np.arange(7)).reshape(4000, -1)[:, : len(days)]
    boot = totals[ids].sum(1) / counts[ids].sum(1)[:, None]
    intervals = np.quantile(boot, [1 / 360, 359 / 360], axis=0)
    months = t.astype("datetime64[s]").astype("datetime64[M]")
    monthly = {
        str(m): float((sb - brier)[months == m].mean()) for m in np.unique(months)
    }
    names = (
        "accuracy",
        "brier_gain_vs_spot",
        "logloss_gain_vs_spot",
        "brier_gain_vs_frequency",
        "logloss_gain_vs_frequency",
    )
    result = {name: float(value) for name, value in zip(names, values.mean(0))}
    result.update(
        {name + "_ci99_444": intervals[:, j].tolist() for j, name in enumerate(names)}
    )
    result.update(
        n=int(moving.sum()),
        flats=int((~moving).sum()),
        brier=float(brier.mean()),
        logloss=float(ll.mean()),
        monthly_brier_gain_vs_spot=monthly,
        balanced_accuracy=float((correct[label].mean() + correct[~label].mean()) / 2),
        predicted_up=int((p >= 0.5).sum()),
        predicted_down=int((p < 0.5).sum()),
    )
    result["information_gate"] = bool(
        intervals[0, 0] > 0.5
        and (intervals[0, 1:] > 0).all()
        and sum(v > 0 for v in monthly.values()) >= 5
        and min(result["predicted_up"], result["predicted_down"]) >= 100
    )
    return result
