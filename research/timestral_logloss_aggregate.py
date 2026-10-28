"""Two delayed log-loss aggregators over four fixed joint forecasts."""

import numpy as np
from scipy.special import logsumexp

BOUNDARY = 1785542400
NAMES = ("common_old", "common_mixed", "uniform_old", "uniform_mixed")


def experts(common, uniform, flat, frequency):
    if not (common.shape == uniform.shape == flat.shape == frequency.shape) or any(
        not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1))
        for p in (common, uniform, flat, frequency)
    ):
        raise ValueError("aligned interior probabilities required")
    result = []
    for up in (common, uniform):
        for f in (flat, (flat + frequency) / 2):
            result.append(np.column_stack(((1 - f) * (1 - up), f, (1 - f) * up)))
    return np.stack(result, axis=1)


def replay(p, y, anchor, end):
    n = len(anchor)
    if (
        p.shape != (n, 4, 3)
        or y.shape != anchor.shape
        or end.shape != anchor.shape
        or not n
        or not np.isfinite(p).all()
        or np.any((p <= 0) | (p >= 1))
        or not np.allclose(p.sum(2), 1, atol=1e-12, rtol=0)
        or not np.isin(y, [0, 1, 2]).all()
        or np.any(np.diff(anchor) < 300)
        or np.any(anchor % 300)
        or np.any(end != anchor + 360)
        or np.any(end > BOUNDARY)
    ):
        raise ValueError("invalid delayed forecast sequence or protected target")
    logs = np.zeros((2, 4))
    weights = np.empty((n, 4))
    adaptive = np.empty((n, 3))
    for i in range(n):
        lane = i % 2
        if i >= 2:
            j = i - 2
            if end[j] > anchor[i]:
                raise ValueError("immature feedback")
            logs[lane] += np.log(p[j, :, y[j]])
        w = np.exp(logs[lane] - logs[lane].max())
        w /= w.sum()
        weights[i] = w
        adaptive[i] = w @ p[i]
    return adaptive, weights


def evidence(p, y, adaptive):
    n = len(y)
    losses = -np.log(p[np.arange(n)[:, None], np.arange(4)[None, :], y[:, None]])
    total = float(-np.log(adaptive[np.arange(n), y]).sum())
    expert = losses.sum(0)
    exact = -sum(logsumexp(-losses[lane::2].sum(0)) - np.log(4) for lane in (0, 1))
    moving = y != 1
    qs = p[:, (0, 2), 2] / (1 - p[:, (0, 2), 1])
    fs = p[:, (0, 1), 1]
    direction_loss = -np.log(np.where(y[:, None] == 2, qs, 1 - qs)) * moving[:, None]
    flat_loss = -np.log(np.where(moving[:, None], 1 - fs, fs))
    aq = adaptive[:, 2] / (1 - adaptive[:, 1])
    al_direction = float((-np.log(np.where(y == 2, aq, 1 - aq)) * moving).sum())
    al_flat = float(-np.log(np.where(moving, 1 - adaptive[:, 1], adaptive[:, 1])).sum())
    return dict(
        expert_logloss=expert.tolist(),
        adaptive_logloss=total,
        best_expert_logloss=float(expert.min()),
        regret=total - float(expert.min()),
        regret_bound=float(2 * np.log(4)),
        direction_regret=al_direction - float(direction_loss.sum(0).min()),
        flat_regret=al_flat - float(flat_loss.sum(0).min()),
        component_bound=float(2 * np.log(2)),
        telescoping_gap=float(total - exact),
    )
