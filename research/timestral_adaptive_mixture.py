"""Past-only convex mixtures of frozen directional experts."""

import hashlib

import numpy as np
from scipy.optimize import minimize

from research.timestral_adaptive_direction import DAY, validate

EXPERTS = ("linear", "trees", "reversal", "frequency")
PENALTY = 1e-4
FLOOR = 1e-6


def fit(experts, y, t, end, cutoff):
    if experts.shape != (len(t), 4, 3):
        raise ValueError("four aligned experts required")
    for j in range(4):
        validate(experts[:, j], t, end)
    if (
        y.shape != t.shape
        or not np.isin(y, [0, 1, 2]).all()
        or len(y) < 1000
        or np.any(end > cutoff)
        or np.any(t < cutoff - 28 * DAY)
        or cutoff % DAY
    ):
        raise ValueError("matured trailing calibration data required")
    target = experts[np.arange(len(y)), :, y]
    decay = np.exp2(-(cutoff - end) / (7 * DAY))
    decay /= decay.sum()

    def objective(w):
        probability = target @ w
        loss = -np.dot(decay, np.log(probability)) + PENALTY * np.sum(w * np.log(4 * w))
        grad = -target.T @ (decay / probability) + PENALTY * (np.log(4 * w) + 1)
        return loss, grad

    result = minimize(
        objective,
        np.full(4, 0.25),
        jac=True,
        method="SLSQP",
        bounds=[(FLOOR, 1)] * 4,
        constraints={
            "type": "eq",
            "fun": lambda w: w.sum() - 1,
            "jac": lambda w: np.ones(4),
        },
        options={"ftol": 1e-12, "maxiter": 200},
    )
    if not result.success or abs(result.x.sum() - 1) > 1e-10:
        raise ValueError(f"mixture optimization failed: {result.message}")
    gradient = objective(result.x)[1]
    free = result.x > FLOOR + 1e-8
    lagrange = gradient[free].mean()
    residual = max(
        np.max(abs(gradient[free] - lagrange)),
        np.max(np.maximum(lagrange - gradient[~free], 0), initial=0),
    )
    if residual > 2e-6:
        raise ValueError("mixture optimality residual too large")
    return {
        "cutoff": int(cutoff),
        "n": len(y),
        "max_end": int(end.max()),
        "weights": result.x.tolist(),
        "kkt_residual": float(residual),
        "ids_sha256": hashlib.sha256(t.astype("<i8").tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(y.astype("<i8").tobytes()).hexdigest(),
    }


def replay(experts, y, t, end):
    if (
        experts.shape != (len(t), 4, 3)
        or y.shape != t.shape
        or not np.isin(y, [0, 1, 2]).all()
    ):
        raise ValueError("four experts and aligned labels required")
    for j in range(4):
        validate(experts[:, j], t, end)
    adaptive, fixed = experts[:, 0].copy(), experts[:, 0].copy()
    states = []
    days = np.unique(t // DAY) * DAY
    for cutoff in days[7:]:
        train = (end <= cutoff) & (t >= cutoff - 28 * DAY)
        state = fit(experts[train], y[train], t[train], end[train], int(cutoff))
        states.append(state)
        current = t // DAY == cutoff // DAY
        adaptive[current] = np.einsum("nec,e->nc", experts[current], state["weights"])
        fixed[current] = np.einsum("nec,e->nc", experts[current], states[0]["weights"])
    return adaptive, fixed, states
