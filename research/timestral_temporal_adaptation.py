"""Predeclared time adaptation for delayed, simultaneous-score forecasting."""

import numpy as np
from scipy.special import logsumexp

from research.timestral_multiscore import BOUNDARY, forecast

METHODS = ("share", "discount", "monthly")
PERIOD_SECONDS = 30 * 86400


def update(log_weights, delta, elapsed, method, month_changed=False):
    """Apply one mature feedback update and the fixed temporal transformation."""
    if (
        method not in METHODS
        or log_weights.ndim != 2
        or log_weights.shape[0] != 2
        or log_weights.shape[1] < 1
        or delta.shape != log_weights.shape
        or not np.isfinite(log_weights).all()
        or not np.isfinite(delta).all()
        or not np.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise ValueError("invalid temporal update")
    if method == "monthly" and month_changed:
        return np.zeros_like(log_weights)
    updated = log_weights + delta
    if method == "share":
        log_retention = -elapsed / PERIOD_SECONDS
        log_replacement = np.log(-np.expm1(log_retention))
        metric_mean = logsumexp(updated, axis=1, keepdims=True) - np.log(
            updated.shape[1]
        )
        return np.logaddexp(log_retention + updated, log_replacement + metric_mean)
    if method == "discount":
        return np.exp2(-elapsed / PERIOD_SECONDS) * updated
    return updated


def replay(expert, y, anchor, end, method):
    """Reuse the frozen forecast solver with two independent temporal states."""
    n = len(anchor)
    if (
        method not in METHODS
        or expert.ndim != 3
        or expert.shape[0] != n
        or expert.shape[1] < 1
        or expert.shape[2] != 3
        or anchor.ndim != 1
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
        raise ValueError("invalid temporal forecast sequence or protected target")
    months = anchor.astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
    k = expert.shape[1]
    logs = np.zeros((2, 2, k))
    predictions = np.empty((n, 3))
    slack = np.empty(n)
    diagnostics = {
        "method": method,
        "maximum_potential_ratio": 0.0,
        "updates": 0,
        "skipped_feedback": 0,
        "resets": 0,
        "newton_steps": 0,
        "fallback_count": 0,
    }
    for i in range(n):
        lane = i % 2
        if i >= 2:
            j = i - 2
            if end[j] > anchor[i]:
                raise ValueError("immature feedback")
            reset = method == "monthly" and months[i] != months[j]
            if reset:
                # The old-month outcome is deliberately discarded, not applied.
                logs[lane] = update(
                    logs[lane], np.zeros((2, k)), anchor[i] - anchor[j], method, True
                )
                diagnostics["resets"] += 1
                diagnostics["skipped_feedback"] += 1
            else:
                own = predictions[j]
                target = y[j]
                own_brier = own @ own + 1 - 2 * own[target]
                expert_brier = (expert[j] ** 2).sum(1) + 1 - 2 * expert[j, :, target]
                delta = np.stack(
                    (
                        np.log(expert[j, :, target]) - np.log(own[target]),
                        own_brier - expert_brier,
                    )
                )
                logs[lane] = update(logs[lane], delta, anchor[i] - anchor[j], method)
                diagnostics["updates"] += 1
        weights = np.exp(logs[lane] - logs[lane].max())
        weights /= weights.sum()
        predictions[i], slack[i], steps, fallback, ratio = forecast(expert[i], weights)
        if not np.isfinite(ratio) or ratio > 1 + 1e-10:
            raise ArithmeticError("temporal potential exceeded numerical tolerance")
        diagnostics["newton_steps"] += steps
        diagnostics["fallback_count"] += fallback
        diagnostics["maximum_potential_ratio"] = max(
            diagnostics["maximum_potential_ratio"], ratio
        )
    diagnostics["per_lane_slack"] = [float(slack[lane::2].sum()) for lane in (0, 1)]
    return predictions, slack, diagnostics
