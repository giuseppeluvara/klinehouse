"""One controlled overlap ablation, not a selected or validated trading model."""

from datetime import timedelta

import numpy as np

from .timestral_experiment import LEVELS, utc


def horizon_inputs(hours, anchor, horizon):
    """Build dense and disjoint returns over identical 64*h past price changes.

    The disjoint series contains 64 h-hour returns; the dense series samples
    the same trailing h-hour sum every hour. Its last h-step forecast and the
    disjoint series' one-step forecast refer to the exact same future interval.
    """
    anchor = utc(anchor)
    if horizon not in (4, 24) or len(hours) != 64 * horizon + 1:
        raise ValueError("expected exactly 64*h+1 past hourly closes")
    if any(row.time != anchor - timedelta(hours=len(hours) - i) for i, row in enumerate(hours)):
        raise ValueError("hours must be consecutive and end before anchor")
    closes = np.asarray([row.log_close for row in hours], dtype=float)
    if not np.isfinite(closes).all():
        raise ValueError("nonfinite close")
    dense = (closes[horizon:] - closes[:-horizon]) * 10000
    disjoint = np.diff(closes[::horizon]) * 10000
    np.testing.assert_allclose(disjoint, dense[::horizon], rtol=0, atol=0)
    return {"dense": (dense, horizon), "disjoint": (disjoint, 1)}


def forecast(model, hours, anchor, horizon):
    predictions = {}
    for name, (series, steps) in horizon_inputs(hours, anchor, horizon).items():
        output = model.forecaster.forecast(series.astype(np.float32)[None, None, :], prediction_length=steps,
                                          symmetric_averaging=True, clip_nonnegative=False)
        levels = np.asarray(output.quantile_levels)
        indices = []
        for level in LEVELS:
            match = np.flatnonzero(np.isclose(levels, level))
            if len(match) != 1:
                raise ValueError("missing or ambiguous quantile")
            indices.append(int(match[0]))
        values = output.quantiles[0, 0, -1, :].detach().cpu().numpy().astype(float)[indices] / 10000
        if not np.isfinite(values).all() or (np.diff(values) < 0).any():
            raise ValueError("invalid forecast quantiles")
        predictions[name] = {"point": float(values[1]), "quantiles": values.tolist(),
                             "input_points": len(series), "forecast_steps": steps}
    return predictions


def paired_mae_interval(rows, name, reference):
    """Fixed weekly block bootstrap, descriptive on already inspected history."""
    y = np.asarray([r["actual"] for r in rows])
    p = np.asarray([r["predictions"][name]["point"] for r in rows])
    base = np.zeros(len(rows)) if reference == "unchanged" else np.asarray([r["predictions"][reference]["point"] for r in rows])
    differences = abs(p - y) - abs(base - y)
    blocks = np.asarray([(utc(r["anchor"]) - utc("2026-01-01")).days // 7 for r in rows])
    n = int(blocks.max() + 1)
    sizes, sums = np.bincount(blocks, minlength=n), np.bincount(blocks, weights=differences, minlength=n)
    draws = np.random.default_rng(20261026).integers(0, n, size=(2000, n))
    interval = np.quantile(sums[draws].sum(axis=1) / sizes[draws].sum(axis=1), [.00625, .99375])
    return {"mae_difference": float(differences.mean()), "interval98_75": interval.tolist(),
            "interpretation": "exploratory; four comparisons within this ablation, not all previous research or future decisions"}
