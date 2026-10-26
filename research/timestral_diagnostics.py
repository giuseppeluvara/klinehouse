"""Controlled diagnostics; no model selection or fresh-test claims."""

from __future__ import annotations

import numpy as np

from .timestral_experiment import LEVELS


def interval_scores(actual, quantiles):
    """Separate interval quality from the median's contribution to pinball loss."""
    actual, quantiles = np.asarray(actual, dtype=float), np.asarray(quantiles, dtype=float)
    if (actual.ndim != 1 or not len(actual) or quantiles.shape != (len(actual), 3)
            or not np.isfinite(actual).all() or not np.isfinite(quantiles).all()
            or (np.diff(quantiles, axis=1) < 0).any()):
        raise ValueError("expected finite outcomes and ordered 5/50/95 percent quantiles")
    residual = actual[:, None] - quantiles
    losses = np.maximum(np.asarray(LEVELS) * residual, (np.asarray(LEVELS) - 1) * residual)
    lower, upper = quantiles[:, 0], quantiles[:, 2]
    width = upper - lower
    interval = width + 20 * np.maximum(lower - actual, 0) + 20 * np.maximum(actual - upper, 0)
    return {"samples": len(actual), "coverage90": float(((lower <= actual) & (actual <= upper)).mean()),
            "width90": float(width.mean()), "interval_score90": float(interval.mean()),
            "tail_pinball": float(losses[:, [0, 2]].mean()), "median_pinball": float(losses[:, 1].mean()),
            "all_quantile_pinball": float(losses.mean())}


def compare_intervals(rows):
    """Keep saved predictions intact and expose the mismatched median explicitly."""
    y = np.asarray([r["actual_log_return"] for r in rows])
    static = np.asarray([r["predictions"]["unchanged_static"]["quantiles"] for r in rows])
    dynamic = np.asarray([r["predictions"]["unchanged_dynamic"]["quantiles"] for r in rows])
    point = np.asarray([r["predictions"]["unchanged_static"]["point"] for r in rows])
    if not np.array_equal(point, [r["predictions"]["unchanged_dynamic"]["point"] for r in rows]):
        raise ValueError("point forecasts differ; this does not isolate intervals")
    matched = static.copy()
    matched[:, 1] = point
    return {"saved_static": interval_scores(y, static), "matched_median_static": interval_scores(y, matched),
            "dynamic": interval_scores(y, dynamic),
            "static_median_minus_point": sorted(set((static[:, 1] - point).tolist())),
            "dynamic_median_matches_point": bool(np.array_equal(dynamic[:, 1], point))}


def centered_forecast(model, log_closes, horizon):
    """An affine-equivalence probe, not a newly selected forecasting model.

    Center in float64 before casting to float32 and express values in bps.
    Convert predicted levels back to horizon returns, not return increments.
    """
    values = np.asarray(log_closes, dtype=float)
    if values.shape != (256,) or not np.isfinite(values).all() or horizon not in (4, 24):
        raise ValueError("expected 256 finite log closes and a supported horizon")
    context = ((values - values[-1]) * 10000).astype(np.float32)[None, None, :]
    result = model.forecaster.forecast(context, prediction_length=horizon,
                                       symmetric_averaging=True, clip_nonnegative=False)
    levels = np.asarray(result.quantile_levels)
    indices = []
    for level in LEVELS:
        found = np.flatnonzero(np.isclose(levels, level))
        if len(found) != 1:
            raise ValueError("missing or ambiguous quantile")
        indices.append(int(found[0]))
    quantiles = result.quantiles[0, 0, -1, :].detach().cpu().numpy().astype(float)[indices] / 10000
    if not np.isfinite(quantiles).all() or (np.diff(quantiles) < 0).any():
        raise ValueError("invalid transformed output")
    return {"point": float(quantiles[1]), "quantiles": quantiles.tolist()}


def synthetic_cases(seed=20261026, random_walks=128):
    """Fixed positive and negative controls, independent of the market outcomes."""
    t = np.arange(280)
    cases = [("constant", 10. + np.zeros_like(t)),
             ("linear_up", 10. + .0002 * t), ("linear_down", 10. - .0002 * t)]
    for phase in range(4):
        cases.append((f"sine_phase_{phase}", 10. + .01 * np.sin(2 * np.pi * t / 64 + phase * np.pi / 2)))
    rng = np.random.default_rng(seed)
    for i in range(random_walks):
        cases.append((f"random_walk_{i}", 10. + np.cumsum(rng.normal(0, .002, len(t)))))
    return cases


def correlation(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def overlap_statistics(log_closes, horizon):
    """Lag one of rolling h-step returns overlaps; lag h does not."""
    closes = np.asarray(log_closes, dtype=float)
    if horizon < 1 or len(closes) <= 2 * horizon + 1 or not np.isfinite(closes).all():
        raise ValueError("insufficient or invalid series")
    rolling = closes[horizon:] - closes[:-horizon]
    return {"rolling_lag1_correlation": correlation(rolling[:-1], rolling[1:]),
            "rolling_lag_h_correlation": correlation(rolling[:-horizon], rolling[horizon:]),
            "iid_increment_lag1_expectation": 1 - 1 / horizon,
            "iid_increment_lag_h_expectation": 0., "rolling_samples": len(rolling)}
