"""Shared directional models with causal long context and recurring regimes."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from research.timestral_futures_flow import combine, comparison, predict  # noqa: F401
from research import timestral_universal as shared

ARMS = (("flow", 35), ("long", 39), ("regime", 71))
COMPARISONS = (("long", "flow"), ("regime", "flow"), ("regime", "long"))
CORE = (0, 1, 4, 27, 28, 13, 35, 36)
REGIMES = ("range_quiet", "range_active", "trend_quiet", "trend_active")


def context(data, anchors):
    """Four-hour trend strength and last-hour activity vs the preceding day."""
    if tuple(data) != shared.SYMBOLS:
        raise ValueError("ordered registered assets required")
    t = data[shared.SYMBOLS[0]]["time"]
    if (
        len(t) < 1501
        or t[-1] >= shared.BOUNDARY
        or np.any(np.diff(t) != 60)
        or np.any(t % 60)
    ):
        raise ValueError("complete pre-protected spot minutes required")
    j = np.searchsorted(t, anchors - 60)
    valid = (j >= 1500) & (j < len(t))
    valid[valid] &= t[j[valid]] == anchors[valid] - 60
    j = j[valid]
    own, states = [], []
    for d in data.values():
        if (
            not np.array_equal(d["time"], t)
            or not np.isfinite(d["close"]).all()
            or not np.isfinite(d["trades"]).all()
            or np.any(d["close"] <= 0)
            or np.any(d["trades"] < 0)
        ):
            raise ValueError("invalid or misaligned context source")
        log = np.log(d["close"])
        r = np.r_[0.0, np.diff(log)]
        squares = np.r_[0.0, np.cumsum(r * r)]
        trades = np.r_[0.0, np.cumsum(d["trades"])]
        returns = [
            np.clip(
                (log[j] - log[j - h])
                / np.sqrt(np.maximum(squares[j + 1] - squares[j + 1 - h], h * 1e-16)),
                -8,
                8,
            )
            for h in (240, 1440)
        ]
        activity = ((trades[j + 1] - trades[j - 59]) / 60) / np.maximum(
            (trades[j - 59] - trades[j - 1499]) / 1440, 1.0
        )
        own.append(np.column_stack(returns))
        states.append((2 * (abs(returns[0]) >= 1) + (activity >= 1)).astype(np.int8))
    extended = np.stack(
        [
            np.column_stack(
                (own[a], np.mean([own[b] for b in range(3) if b != a], axis=0))
            )
            for a in range(3)
        ]
    )
    return valid, extended, np.stack(states)


def interact(x, states):
    if (
        x.ndim != 3
        or x.shape[2] != 39
        or states.shape != x.shape[:2]
        or not np.isin(states, range(4)).all()
    ):
        raise ValueError("39 aligned features and four valid regimes required")
    contrasts = np.eye(4)[states] - 0.25
    deviations = (x[:, :, CORE, None] * contrasts[:, :, None, :]).reshape(
        *x.shape[:2], 32
    )
    return np.concatenate((x, deviations), axis=2)


def fit(x, y, end, cutoff):
    if (
        len(y) < 100
        or x.shape[0] != len(y)
        or not np.isin(y, [0, 1]).all()
        or np.any(end > cutoff)
        or not np.isfinite(x).all()
        or x.shape[1] not in (35, 39, 71)
    ):
        raise ValueError("registered finite matured training examples required")
    odd = list(shared.ODD) + list(range(23, x.shape[1]))
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
    return dict(
        n=len(y),
        cutoff=int(cutoff),
        mean=mean.tolist(),
        scale=scale.tolist(),
        coef=estimator.coef_[0].tolist(),
        intercept=float(estimator.intercept_[0]),
        iterations=int(estimator.n_iter_[0]),
    )


def diagnose(y, t, state, x):
    """Descriptive fixed-rule rates; these results do not select any models."""
    out = {}
    for name, signal in (("spot_reversal", -x[:, 0]), ("futures_flow", x[:, 27])):
        keep = (y != 1) & (signal != 0)
        correct = (signal > 0) == (y == 2)
        cells = []
        for k in range(4):
            take = keep & (state == k)
            n = int(take.sum())
            cells.append(
                dict(
                    n=n,
                    correct=int(correct[take].sum()),
                    accuracy=float(correct[take].mean()) if n else None,
                    weight=float(n / keep.sum()),
                )
            )
        days, inv = np.unique(t // 86400, return_inverse=True)
        totals = np.zeros((len(days), 4, 2))
        np.add.at(totals, (inv, state), np.column_stack((keep, keep & correct)))
        start = np.random.default_rng(271033).integers(
            0, len(days) - 6, (4000, (len(days) + 6) // 7)
        )
        ids = (start[:, :, None] + np.arange(7)).reshape(4000, -1)[:, : len(days)]
        sums = totals[ids].sum(1)
        for k, c in enumerate(cells):
            c["accuracy_ci95"] = (
                np.quantile(sums[:, k, 1] / sums[:, k, 0], [0.025, 0.975]).tolist()
                if c["n"] and (sums[:, k, 0] > 0).all()
                else None
            )
        aggregate = sums.sum(1)
        aggregate_ci = np.quantile(
            aggregate[:, 1] / aggregate[:, 0], [0.025, 0.975]
        ).tolist()
        out[name] = dict(
            n=int(keep.sum()),
            excluded=int((~keep).sum()),
            accuracy=float(correct[keep].mean()),
            accuracy_ci95=aggregate_ci,
            regimes=cells,
        )
    return out


def drift(diagnostics):
    result = {}
    for s in shared.SYMBOLS:
        for year in (2025, 2026):
            for signal in ("spot_reversal", "futures_flow"):
                first = diagnostics[f"2024-{s}"][signal]
                last = diagnostics[f"{year}-{s}"][signal]
                w0, g0 = np.array(
                    [[c["weight"], c["accuracy"]] for c in first["regimes"]]
                ).T
                w1, g1 = np.array(
                    [[c["weight"], c["accuracy"]] for c in last["regimes"]]
                ).T
                composition = float((w1 - w0) @ g0)
                within = float(w1 @ (g1 - g0))
                result[f"{year}-{s}-{signal}"] = dict(
                    total=last["accuracy"] - first["accuracy"],
                    composition=composition,
                    within=within,
                )
    return result
