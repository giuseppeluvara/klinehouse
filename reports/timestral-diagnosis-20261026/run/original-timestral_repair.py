"""Causal forecast diagnostics and a baseline-aware policy for a separate replay.

This module never modifies the earlier frozen experiments. Historical replay of
already inspected data cannot certify a newly independent or prospective result.
"""

from __future__ import annotations

from datetime import timedelta
import math

import numpy as np

from .timestral_experiment import LEVELS, stamp, utc

CANDIDATES = ("timestral_price", "timestral_returns")
MINIMUM = {4: 360, 24: 60}


def past_inputs(hours, anchor, horizon):
    """Use the same 256 past hourly closes as the price model, with no extra history."""
    anchor = utc(anchor)
    if horizon not in MINIMUM or len(hours) != 256:
        raise ValueError("expected 256 complete past hours")
    start = anchor - timedelta(hours=len(hours))
    if any(utc(row.time) != start + timedelta(hours=i) for i, row in enumerate(hours)):
        raise ValueError("context must be consecutive and end strictly before anchor")
    closes = np.asarray([row.log_close for row in hours], dtype=float)
    features = np.asarray([row.features for row in hours], dtype=float)
    if (features.shape != (len(hours), 3) or not np.isfinite(closes).all()
            or not np.isfinite(features).all() or (features[:, 0] < 0).any()):
        raise ValueError("invalid past inputs")
    # The last forecast step predicts log(P[t+h-1]/P[t-1]) directly. Summing
    # marginal one-hour quantiles would not give a horizon-return quantile.
    returns = (closes[horizon:] - closes[:-horizon]) * 10000
    scale = max(float(np.linalg.norm(features[-24:, 0]) * math.sqrt(horizon / 24)), 1e-8)
    return returns, scale


def forecast_candidates(model, hours, anchor, horizon):
    series, scale = past_inputs(hours, anchor, horizon)
    price = model.predict(hours[-256:], {"horizon": horizon}, False)
    result = model.forecaster.forecast(
        series.astype(np.float32)[None, None, :], prediction_length=horizon,
        symmetric_averaging=True, clip_nonnegative=False,
    )
    levels = np.asarray(result.quantile_levels)
    values = result.quantiles[0, 0, -1, :].detach().cpu().numpy().astype(float) / 10000
    indices = []
    for level in LEVELS:
        matches = np.flatnonzero(np.isclose(levels, level))
        if len(matches) != 1:
            raise ValueError("missing or ambiguous quantile level")
        indices.append(int(matches[0]))
    if not np.isfinite(values).all():
        raise ValueError("nonfinite horizon-return forecast")
    selected = values[indices]
    return {"timestral_price": price,
            "timestral_returns": {"point": float(selected[1]), "quantiles": selected.tolist(),
                                  "all_quantile_levels": levels.tolist(),
                                  "all_return_quantiles": values.tolist()}}, scale


def month_start(value, offset=0):
    value = utc(value)
    number = value.year * 12 + value.month - 1 + offset
    return value.replace(year=number // 12, month=number % 12 + 1,
                         day=1, hour=0, minute=0, second=0, microsecond=0)


def matured(history, anchor, horizon):
    """Never admit a future label, even if a caller has already materialized it."""
    anchor = utc(anchor)
    result = []
    identities = set()
    for row in history:
        if row["horizon_hours"] != horizon:
            continue
        left, right = utc(row["anchor"]), utc(row["label_end"])
        if right != left + timedelta(hours=horizon):
            raise ValueError("invalid label interval")
        if right > anchor:
            continue
        if row["sample_id"] in identities:
            raise ValueError("duplicate matured sample")
        identities.add(row["sample_id"])
        if (not np.isfinite(row["actual_log_return"]) or not np.isfinite(row["scale"])
                or row["scale"] <= 0):
            raise ValueError("invalid matured outcome or scale")
        result.append(row)
    result.sort(key=lambda row: row["anchor"])
    if any(utc(b["anchor"]) < utc(a["label_end"]) for a, b in zip(result, result[1:])):
        raise ValueError("overlapping matured labels")
    return result


def choose_model(history, anchor, horizon):
    """Select using three completed months; unchanged price is always admissible."""
    if horizon not in MINIMUM:
        raise ValueError("unsupported horizon")
    cutoff, start = month_start(anchor), month_start(anchor, -3)
    rows = [row for row in matured(history, cutoff, horizon) if utc(row["anchor"]) >= start]
    decision = {"selected": "unchanged", "status": "insufficient_history", "samples": len(rows),
                "fit_start": stamp(start), "fit_cutoff": stamp(cutoff),
                "latest_label_end": max((row["label_end"] for row in rows), default=None),
                "candidate_checks": {}, "baseline_always_eligible": True}
    months = [stamp(month_start(cutoff, offset))[:7] for offset in (-3, -2, -1)]
    groups = {month: [i for i, row in enumerate(rows) if row["anchor"][:7] == month] for month in months}
    if len(rows) < MINIMUM[horizon] or any(len(indices) < MINIMUM[horizon] // 3 for indices in groups.values()):
        return decision
    y = np.asarray([row["actual_log_return"] for row in rows])
    blocks = np.asarray([(utc(row["anchor"]) - start).days // 7 for row in rows])
    count = math.ceil((cutoff - start).days / 7)
    sizes = np.bincount(blocks, minlength=count)
    choices = np.random.default_rng(17).integers(0, count, size=(2000, count))
    denominators = sizes[choices].sum(axis=1)
    if (denominators == 0).any():
        return decision
    eligible = []
    for name in CANDIDATES:
        point = np.asarray([row["predictions"][name]["point"] for row in rows])
        if not np.isfinite(point).all():
            raise ValueError("nonfinite past candidate")
        error = point - y
        difference = np.abs(error) - np.abs(y)
        sums = np.bincount(blocks, weights=difference, minlength=count)
        interval = np.quantile(sums[choices].sum(axis=1) / denominators, [.00625, .99375])
        checks = {"lower_mae": float(difference.mean()) < 0,
                  "rmse_no_worse": float(np.mean(error ** 2)) <= float(np.mean(y ** 2)),
                  "two_monthly_wins": sum(difference[indices].mean() < 0 for indices in groups.values()) >= 2,
                  "upper_interval_below_zero": float(interval[1]) < 0}
        admitted = all(checks.values())
        decision["candidate_checks"][name] = {
            "admitted": admitted, "checks": checks, "mae_difference": float(difference.mean()),
            "interval98_75": interval.tolist(), "mae": float(np.abs(error).mean()),
        }
        if admitted:
            eligible.append((float(np.abs(error).mean()), CANDIDATES.index(name), name))
    if eligible:
        decision.update(selected=min(eligible)[2], status="historical_gate_passed")
    else:
        decision["status"] = "baseline_no_evidence"
    return decision


def calibrated_interval(history, anchor, horizon, model, point, scale):
    """Finite-rank radius using only prior, volatility-normalized residuals.

No exchangeability or coverage guarantee is asserted for time-series data or
model selection. All normalizing scales are those known at each forecast origin.
"""
    if horizon not in MINIMUM or model not in (*CANDIDATES, "unchanged"):
        raise ValueError("unsupported horizon or candidate")
    if not np.isfinite([point, scale]).all() or scale <= 0:
        raise ValueError("invalid present forecast or scale")
    anchor = utc(anchor)
    rows = [row for row in matured(history, anchor, horizon)
            if utc(row["label_end"]) > anchor - timedelta(days=90)]
    if len(rows) < MINIMUM[horizon]:
        return {"status": "insufficient_calibration", "samples": len(rows), "quantiles": None}
    residuals = [abs(row["actual_log_return"] - (0 if model == "unchanged" else row["predictions"][model]["point"]))
                 / row["scale"] for row in rows]
    if not np.isfinite(residuals).all():
        raise ValueError("invalid calibration residual")
    rank = math.ceil((len(rows) + 1) * .90)
    radius = float(np.sort(residuals)[rank - 1] * scale)
    return {"status": "available", "samples": len(rows), "rank": rank,
            "latest_label_end": max(row["label_end"] for row in rows),
            "quantiles": [point - radius, point, point + radius],
            "nominal_coverage": .9, "coverage_guaranteed": False}


def score(rows, model):
    actual = np.asarray([row["actual_log_return"] for row in rows])
    point = np.asarray([row["predictions"][model]["point"] for row in rows])
    quantiles = np.asarray([row["predictions"][model]["quantiles"] for row in rows])
    error = point - actual
    residuals = actual[:, None] - quantiles
    called = point != 0
    correlation = float(np.corrcoef(point, actual)[0, 1]) if point.std() > 0 and actual.std() > 0 else None
    return {"samples": len(rows), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae_skill_vs_unchanged": float(1 - np.abs(error).mean() / np.abs(actual).mean()) if np.abs(actual).mean() else None,
            "correlation": correlation, "prediction_std": float(point.std()), "actual_std": float(actual.std()),
            "bias": float(error.mean()), "direction_calls": int(called.sum()),
            "direction_accuracy_when_called": float((np.sign(point[called]) == np.sign(actual[called])).mean()) if called.any() else None,
            "coverage90": float(((quantiles[:, 0] <= actual) & (actual <= quantiles[:, 2])).mean()),
            "width90": float((quantiles[:, 2] - quantiles[:, 0]).mean()),
            "pinball": float(np.maximum(np.array(LEVELS) * residuals, (np.array(LEVELS) - 1) * residuals).mean()),
            "quantile_crossing": float((np.diff(quantiles, axis=1) < 0).any(axis=1).mean())}
