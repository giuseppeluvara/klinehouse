"""Delayed defensive forecasts for simultaneous log-loss and Brier regret."""

import numpy as np
from scipy.optimize import minimize

BOUNDARY = 1785542400
TOLERANCE = 2e-12


def potential(probability, a, b):
    """Potential ratios for each possible outcome, using normalized weights."""
    return a / probability + b * np.exp(probability @ probability + 1 - 2 * probability)


def _fallback(a, b, start):
    """Convex epigraph minimization; acceptance always checks all outcomes."""

    def values(x):
        p = x[:3]
        term = b * np.exp(p @ p + 1 - 2 * p)
        f = a / p + term
        jac = 2 * term[:, None] * (p[None, :] - np.eye(3))
        jac[np.diag_indices(3)] -= a / p**2
        return f, jac

    result = minimize(
        lambda x: x[3],
        np.r_[start, potential(start, a, b).max()],
        jac=lambda x: np.array([0.0, 0.0, 0.0, 1.0]),
        method="SLSQP",
        bounds=[(1e-15, 1)] * 3 + [(0, None)],
        constraints=[
            {
                "type": "eq",
                "fun": lambda x: x[:3].sum() - 1,
                "jac": lambda x: np.array([1.0, 1.0, 1.0, 0.0]),
            },
            {
                "type": "ineq",
                "fun": lambda x: x[3] - values(x)[0],
                "jac": lambda x: np.column_stack((-values(x)[1], np.ones(3))),
            },
        ],
        options={"maxiter": 200, "ftol": 1e-13},
    )
    p = result.x[:3]
    p = p / p.sum()
    if (
        not np.isfinite(p).all()
        or np.any(p <= 0)
        or potential(p, a, b).max() > 1 + 1e-10
    ):
        raise ArithmeticError("no numerically certified defensive forecast")
    return p


def forecast(expert, weights):
    """Return probability, log slack, Newton steps, fallback flag and ratio."""
    a = weights[0] @ expert
    b = weights[1] @ np.exp(-(expert**2).sum(1)[:, None] - 1 + 2 * expert)
    p = weights.sum(0) @ expert
    p /= p.sum()
    steps = 0
    for steps in range(40):
        term = b * np.exp(p @ p + 1 - 2 * p)
        f = a / p + term
        if f.max() <= 1 + TOLERANCE:
            ratio = float(f.max())
            return p, max(0.0, float(np.log(ratio))), steps, False, ratio
        residual = f[:2] - f[2]
        jac = 2 * term[:, None] * (p[None, :] - np.eye(3))
        jac[np.diag_indices(3)] -= a / p**2
        reduced = jac[:2, :2] - jac[2, :2] - jac[:2, 2, None] + jac[2, 2]
        try:
            delta = np.linalg.solve(reduced, -residual)
        except np.linalg.LinAlgError:
            break
        direction = np.r_[delta, -delta.sum()]
        if not np.isfinite(direction).all():
            break
        negative = direction < 0
        if not np.any(negative):
            break
        step = min(1.0, float(np.min(-0.99 * p[negative] / direction[negative])))
        old_norm = float(np.ptp(f))
        accepted = False
        for _ in range(30):
            trial = p + step * direction
            trial /= trial.sum()
            tf = potential(trial, a, b)
            if tf.max() <= 1 + TOLERANCE or np.ptp(tf) < (1 - 1e-4 * step) * old_norm:
                p = trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    p = _fallback(a, b, p)
    ratio = float(potential(p, a, b).max())
    return p, max(0.0, float(np.log(ratio))), steps + 1, True, ratio


def replay(expert, y, anchor, end):
    """Use only mature same-lane labels; retain all initial forecasts."""
    n = len(anchor)
    if (
        expert.ndim != 3
        or expert.shape[0] != n
        or expert.shape[1] < 1
        or expert.shape[2] != 3
        or y.shape != anchor.shape
        or end.shape != anchor.shape
        or not n
        or not np.isfinite(expert).all()
        or np.any((expert <= 0) | (expert >= 1))
        or not np.allclose(expert.sum(2), 1, atol=1e-12, rtol=0)
        or not np.isin(y, [0, 1, 2]).all()
        or np.any(np.diff(anchor) < 300)
        or np.any(anchor % 300)
        or np.any(end != anchor + 360)
        or np.any(end > BOUNDARY)
    ):
        raise ValueError("invalid delayed forecast sequence or protected target")
    k = expert.shape[1]
    log_weights = np.zeros((2, 2, k))
    predictions = np.empty((n, 3))
    slack = np.empty(n)
    fallback_count = 0
    newton_steps = 0
    maximum_ratio = 0.0
    for i in range(n):
        lane = i % 2
        if i >= 2:
            j = i - 2
            if end[j] > anchor[i]:
                raise ValueError("immature feedback")
            own = predictions[j]
            target = y[j]
            log_weights[lane, 0] += np.log(expert[j, :, target]) - np.log(own[target])
            own_brier = own @ own + 1 - 2 * own[target]
            expert_brier = (expert[j] ** 2).sum(1) + 1 - 2 * expert[j, :, target]
            log_weights[lane, 1] += own_brier - expert_brier
        weights = np.exp(log_weights[lane] - log_weights[lane].max())
        weights /= weights.sum()
        predictions[i], slack[i], steps, fallback, ratio = forecast(expert[i], weights)
        newton_steps += steps
        fallback_count += fallback
        maximum_ratio = max(maximum_ratio, ratio)
    return (
        predictions,
        slack,
        {
            "maximum_potential_ratio": maximum_ratio,
            "per_lane_slack": [float(slack[lane::2].sum()) for lane in (0, 1)],
            "newton_steps": newton_steps,
            "fallback_count": fallback_count,
        },
    )
