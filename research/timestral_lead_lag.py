"""Conservative futures/spot lead-lag features and fixed ridge-offset fits."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

BOUNDARY = 1785542400
WINDOW_SECONDS = 90 * 86400
MINIMUM_DAYS = 80
PENALTY = 0.25
SCALE_FLOOR = 1e-8
GRADIENT_TOLERANCE = 1e-7
HORIZONS = (5, 60)


def _timestamps(values, name):
    t = np.asarray(values)
    if (
        t.ndim != 1
        or not len(t)
        or not np.issubdtype(t.dtype, np.number)
        or not np.isfinite(t).all()
        or np.any(t != np.floor(t))
        or np.any((t < 0) | (t >= BOUNDARY))
    ):
        raise ValueError(f"invalid or protected {name} timestamps")
    return t.astype(np.int64)


def _source(data):
    time = _timestamps(data["time"], "source")
    close = np.asarray(data["close"], dtype=float)
    if (
        close.shape != time.shape
        or not np.isfinite(close).all()
        or np.any(close <= 0)
        or np.any(np.diff(time) <= 0)
        or np.any(time % 60)
    ):
        raise ValueError(
            "finite positive closes and sorted unique minute opens required"
        )
    return time, np.log(close)


def _anchors(values):
    anchor = _timestamps(values, "forecast")
    if np.any(anchor + 360 > BOUNDARY) or np.any(anchor % 60):
        raise ValueError("invalid or protected forecast target")
    return anchor


def features(spot, futures, anchors):
    """Use exact past opens; incomplete windows return zero features and false."""
    anchor = _anchors(anchors)
    st, slog = _source(spot)
    ft, flog = _source(futures)
    latest = anchor - 120
    si = np.searchsorted(st, latest)
    fi = np.searchsorted(ft, latest)
    sj = np.minimum(si, len(st) - 1)
    fj = np.minimum(fi, len(ft) - 1)
    valid = (
        (si < len(st))
        & (fi < len(ft))
        & (st[sj] == latest)
        & (ft[fj] == latest)
        & (sj >= 61)
        & (fj >= 60)
    )
    sgaps = np.r_[0, np.cumsum(np.diff(st) != 60)]
    fgaps = np.r_[0, np.cumsum(np.diff(ft) != 60)]
    valid &= sgaps[sj] == sgaps[np.maximum(sj - 61, 0)]
    valid &= fgaps[fj] == fgaps[np.maximum(fj - 60, 0)]
    price = np.zeros((len(anchor), 2))
    lead = np.zeros_like(price)
    if valid.any():
        # r[k] uses close[k] - close[k-1]; the latest r[j] is excluded from RMS.
        returns = np.r_[0.0, np.diff(slog)]
        squares = np.r_[0.0, np.cumsum(returns**2)]
        j, k = sj[valid], fj[valid]
        rms = np.sqrt(np.maximum((squares[j] - squares[j - 60]) / 60, 0))
        rms = np.maximum(rms, SCALE_FLOOR)
        for column, horizon in enumerate(HORIZONS):
            spot_return = slog[j] - slog[j - horizon]
            future_return = flog[k] - flog[k - horizon]
            denominator = rms * np.sqrt(horizon)
            price[valid, column] = np.clip(spot_return / denominator, -8, 8)
            lead[valid, column] = np.clip(
                (future_return - spot_return) / denominator, -8, 8
            )
    return price, lead, valid


def _design(x, offset):
    x = np.asarray(x, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if (
        x.ndim != 2
        or x.shape[1] != 2
        or offset.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(offset).all()
    ):
        raise ValueError("two finite features and aligned logit offsets required")
    return x, offset


def offsetfit(x, y, offset, anchors, ends, cutoff):
    """Fit two coefficients on an already-filtered, mature 90-day window."""
    x, offset = _design(x, offset)
    anchor = _anchors(anchors)
    end = np.asarray(ends)
    y = np.asarray(y)
    if (
        not np.isscalar(cutoff)
        or not np.isfinite(cutoff)
        or cutoff != int(cutoff)
        or cutoff > BOUNDARY
        or cutoff < WINDOW_SECONDS
        or anchor.shape != (len(x),)
        or end.shape != anchor.shape
        or y.shape != anchor.shape
        or not np.isin(y, [0, 1]).all()
        or np.any(end != anchor + 360)
        or np.any(end > cutoff)
        or np.any(anchor < cutoff - WINDOW_SECONDS)
    ):
        raise ValueError("binary mature outcomes inside the past 90 days required")
    days, inv, counts = np.unique(
        anchor // 86400, return_inverse=True, return_counts=True
    )
    if len(days) < MINIMUM_DAYS:
        raise ValueError("at least 80 training UTC days required")
    weights = 1 / (len(days) * counts[inv])
    scale = np.maximum(np.sqrt(weights @ np.square(x)), SCALE_FLOOR)
    z = np.clip(x / scale, -3, 3)

    def objective(beta):
        eta = offset + z @ beta
        loss = weights @ (np.logaddexp(0, eta) - y * eta)
        gradient = z.T @ (weights * (expit(eta) - y)) + PENALTY * beta
        return float(loss + PENALTY / 2 * (beta @ beta)), gradient

    result = minimize(
        objective,
        np.zeros(2),
        method="BFGS",
        jac=True,
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    value, gradient = objective(result.x)
    if (
        not np.isfinite(result.x).all()
        or not np.isfinite(value)
        or not np.isfinite(gradient).all()
        or np.max(np.abs(gradient)) > GRADIENT_TOLERANCE
    ):
        raise ArithmeticError("ridge-offset optimum failed its gradient check")
    return {
        "coefficients": result.x.tolist(),
        "scale": scale.tolist(),
        "gradient": gradient.tolist(),
        "max_abs_gradient": float(np.max(np.abs(gradient))),
        "n": len(y),
        "cutoff": int(cutoff),
        "days": len(days),
        "training_loss": float(value - PENALTY / 2 * (result.x @ result.x)),
        "objective": value,
        "penalty": PENALTY,
        "iterations": int(result.nit),
        "optimizer_success": bool(result.success),
        "first_anchor": int(anchor.min()),
        "last_anchor": int(anchor.max()),
        "max_end": int(end.max()),
    }


def predict(state, x, offset, anchors):
    """Keep the issued baseline offset fixed and add only the fitted correction."""
    x, offset = _design(x, offset)
    anchor = _anchors(anchors)
    beta = np.asarray(state["coefficients"], dtype=float)
    scale = np.asarray(state["scale"], dtype=float)
    cutoff = state["cutoff"]
    if (
        anchor.shape != (len(x),)
        or beta.shape != (2,)
        or scale.shape != (2,)
        or not np.isfinite(beta).all()
        or not np.isfinite(scale).all()
        or np.any(scale < SCALE_FLOOR)
        or not np.isfinite(cutoff)
        or cutoff != int(cutoff)
        or cutoff > BOUNDARY
        or np.any(anchor < cutoff)
    ):
        raise ValueError("invalid fit state or prediction preceding its cutoff")
    return expit(offset + np.clip(x / scale, -3, 3) @ beta)
