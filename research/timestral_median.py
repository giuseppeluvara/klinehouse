"""Past-only, regularized conditional-median calibration and state ablation."""

from datetime import timedelta

import numpy as np

from .timestral_experiment import stamp, utc

FEATURES = (
    "return4_scaled",
    "return24_scaled",
    "return168_scaled",
    "imbalance24",
    "trade_intensity_change",
    "timestral_price_scaled",
    "timestral_return_scaled",
)
PENALTY = 0.025


def feature_vector(hours, anchor, scale, predictions):
    anchor = utc(anchor)
    if len(hours) != 256 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid context or scale")
    if any(r.time != anchor - timedelta(hours=256 - i) for i, r in enumerate(hours)):
        raise ValueError("context must precede anchor with consecutive hours")
    prices = np.asarray([r.log_close for r in hours])
    state = np.asarray([r.features for r in hours])
    values = [
        *((prices[-1] - prices[-1 - lag]) / scale for lag in (4, 24, 168)),
        state[-24:, 2].mean(),
        state[-24:, 1].mean() - state[-168:, 1].mean(),
        predictions["timestral_price"]["point"] / scale,
        predictions["timestral_returns"]["point"] / scale,
    ]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite past feature")
    return np.asarray(values)


def fit(rows, cutoff, horizon, with_timestral):
    """Exact LP: weighted MAE plus L1 shrinkage of all coefficients toward zero."""
    from scipy.optimize import linprog

    cutoff = utc(cutoff)
    if not rows or any(
        utc(r["label_end"]) > cutoff or r["horizon_hours"] != horizon for r in rows
    ):
        raise ValueError("fit only accepts matured labels of one horizon")
    if any(
        utc(r["label_end"]) != utc(r["anchor"]) + timedelta(hours=horizon) for r in rows
    ):
        raise ValueError("invalid target interval")
    ordered = sorted(rows, key=lambda r: r["anchor"])
    if len({r["sample_id"] for r in rows}) != len(rows) or any(
        utc(b["anchor"]) < utc(a["label_end"]) for a, b in zip(ordered, ordered[1:])
    ):
        raise ValueError("duplicate or overlapping training labels")
    if len(rows) < (60 if horizon == 24 else 360):
        raise ValueError("insufficient training history")
    columns = 7 if with_timestral else 5
    x = np.asarray([r["features"][:columns] for r in rows])
    scale = np.asarray([r["scale"] for r in rows])
    y = np.asarray([r["actual"] for r in rows])
    if (
        not np.isfinite(x).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(y).all()
        or (scale <= 0).any()
    ):
        raise ValueError("invalid training values")
    decay = np.asarray(
        [
            2 ** (-(cutoff - utc(r["label_end"])).total_seconds() / (60 * 86400))
            for r in rows
        ]
    )
    weights = decay / decay.sum()
    mean = np.average(x, axis=0, weights=weights)
    std = np.maximum(
        np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights)), 1e-8
    )
    design = np.column_stack([np.ones(len(rows)), np.clip((x - mean) / std, -8, 8)])
    normalized_target = y / scale
    # Weighting normalized residuals by their origin scale preserves log-return
    # MAE (up to a positive constant), not volatility-normalized MAE.
    loss_weights = decay * scale
    loss_weights /= loss_weights.sum()
    n, d = design.shape
    # Variables: unconstrained beta, positive |residual| and positive |beta|.
    objective = np.r_[np.zeros(d), loss_weights, np.full(d, PENALTY)]
    constraints = np.block(
        [
            [design, -np.eye(n), np.zeros((n, d))],
            [-design, -np.eye(n), np.zeros((n, d))],
            [np.eye(d), np.zeros((d, n)), -np.eye(d)],
            [-np.eye(d), np.zeros((d, n)), -np.eye(d)],
        ]
    )
    bounds = [(None, None)] * d + [(0, None)] * (n + d)
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=np.r_[normalized_target, -normalized_target, np.zeros(2 * d)],
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise ValueError(f"median optimizer failed: {result.message}")
    beta = result.x[:d]
    recomputed = float(
        loss_weights @ abs(design @ beta - normalized_target)
        + PENALTY * abs(beta).sum()
    )
    zero = float(loss_weights @ abs(normalized_target))
    if abs(recomputed - result.fun) > 1e-7 or recomputed > zero + 1e-7:
        raise ValueError("optimizer solution failed independent objective check")
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "beta": beta.tolist(),
        "feature_order": list(FEATURES[:columns]),
        "cutoff": stamp(cutoff),
        "latest_label_end": max(r["label_end"] for r in rows),
        "samples": n,
        "penalty": PENALTY,
        "objective": recomputed,
        "zero_objective": zero,
        "optimizer": "scipy.optimize.linprog highs",
    }


def predict(parameters, features, scale):
    x = np.asarray(features[: len(parameters["mean"])])
    z = np.r_[1.0, np.clip((x - parameters["mean"]) / parameters["std"], -8, 8)]
    if not np.isfinite(z).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid present input")
    return float(scale * (z @ parameters["beta"]))
