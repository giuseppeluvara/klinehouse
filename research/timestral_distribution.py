"""Separate conditional mean, expected log and quantiles of future variance."""

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma
from scipy.stats import gamma

from .timestral_experiment import utc

NAMES = ("gamma", "persistence", "ewma")


def raw_mean(parameters, row):
    p = parameters["variance"]
    x = np.asarray(row["variance_features"])
    z = np.clip((x - p["mean"]) / p["scale"], -8, 8)
    return p["unit"] * math.exp(float(z @ p["coef"] + p["intercept"]))


def calibrate(rows, parameters, cutoff):
    cutoff = utc(cutoff)
    if (
        cutoff != utc(parameters["cutoff"])
        or len(rows) != parameters["samples"]
        or any(utc(r["label_end"]) > cutoff for r in rows)
    ):
        raise ValueError("calibration requires exact matured fit sample")
    actual = np.asarray([r["variance"] for r in rows])
    result = {}
    for name in NAMES:
        predicted = np.asarray(
            [
                raw_mean(parameters, r) if name == "gamma" else r["variance_" + name]
                for r in rows
            ]
        )
        if (
            not np.isfinite(actual).all()
            or not np.isfinite(predicted).all()
            or (actual <= 0).any()
            or (predicted <= 0).any()
        ):
            raise ValueError("positive finite variances required")
        bias = float(np.mean(actual / predicted))
        discrepancy = float(-np.log(actual / (bias * predicted)).mean())
        if discrepancy <= 0 or discrepancy >= math.log(0.001) - digamma(0.001):
            raise ValueError("unidentifiable Gamma shape")
        k = float(
            brentq(
                lambda k: math.log(k) - digamma(k) - discrepancy, 0.001, 1e8, xtol=1e-10
            )
        )
        result[name] = {
            "bias": bias,
            "shape": k,
            "log_shift": float(digamma(k) - math.log(k)),
            "lower_ratio": float(gamma.ppf(0.05, k) / k),
            "upper_ratio": float(gamma.ppf(0.95, k) / k),
        }
    return {"cutoff": parameters["cutoff"], "samples": len(rows), "parameters": result}


def predict(calibration, parameters, row):
    if utc(calibration["cutoff"]) > utc(row["anchor"]):
        raise ValueError("future calibration")
    result = {}
    for name, c in calibration["parameters"].items():
        raw = raw_mean(parameters, row) if name == "gamma" else row["variance_" + name]
        mean = raw * c["bias"]
        result[name] = {
            "mean": mean,
            "expected_log": math.log(mean) + c["log_shift"],
            "lower90": mean * c["lower_ratio"],
            "upper90": mean * c["upper_ratio"],
            "probability_above_recent": float(
                gamma.sf(
                    row["variance_persistence"], c["shape"], scale=mean / c["shape"]
                )
            ),
        }
    return result


def score(rows):
    actual = np.asarray([r["variance"] for r in rows])
    metrics = {}
    losses = {}
    for name in NAMES:
        forecasts = [r["distribution"][name] for r in rows]
        mean = np.asarray([r["mean"] for r in forecasts])
        log = np.asarray([r["expected_log"] for r in forecasts])
        q = actual / mean - np.log(actual / mean) - 1
        lower = np.asarray([r["lower90"] for r in forecasts])
        upper = np.asarray([r["upper90"] for r in forecasts])
        interval = (
            upper
            - lower
            + 20 * np.maximum(lower - actual, 0)
            + 20 * np.maximum(actual - upper, 0)
        )
        threshold = np.asarray([r["variance_persistence"] for r in rows])
        prob = np.asarray([r["probability_above_recent"] for r in forecasts])
        metrics[name] = {
            "qlike": float(q.mean()),
            "expected_log_mse": float(np.mean((np.log(actual) - log) ** 2)),
            "coverage90": float(np.mean((actual >= lower) & (actual <= upper))),
            "interval_score_scaled": float(np.mean(interval / threshold)),
            "event_brier": float(np.mean((prob - (actual > threshold)) ** 2)),
            "samples": len(rows),
        }
        losses[name] = q
    months = sorted({r["anchor"][:7] for r in rows})
    start = utc(rows[0]["anchor"]).replace(day=1, hour=0)
    blocks = np.asarray([(utc(r["anchor"]) - start).days // 7 for r in rows])
    n = int(max(blocks) + 1)
    sizes = np.bincount(blocks, minlength=n)
    draws = np.random.default_rng(20261027).integers(0, n, (4000, n))
    screens = {}
    for name in NAMES[1:]:
        diff = losses["gamma"] - losses[name]
        sums = np.bincount(blocks, weights=diff, minlength=n)
        ci = np.quantile(
            sums[draws].sum(axis=1) / sizes[draws].sum(axis=1),
            [0.05 / 24, 1 - 0.05 / 24],
        )
        gain = float(1 - metrics["gamma"]["qlike"] / metrics[name]["qlike"])
        wins = sum(
            float(diff[[i for i, r in enumerate(rows) if r["anchor"][:7] == m]].mean())
            < 0
            for m in months
        )
        checks = {
            "gain_at_least_one_percent": gain >= 0.01,
            "expected_log_mse_no_worse": metrics["gamma"]["expected_log_mse"]
            <= metrics[name]["expected_log_mse"],
            "two_thirds_months": wins >= math.ceil(2 * len(months) / 3),
            "adjusted_upper_below_zero": float(ci[1]) < 0,
        }
        screens[name] = {
            "gain": gain,
            "monthly_wins": wins,
            "months": len(months),
            "interval99_583333": ci.tolist(),
            "checks": checks,
            "passes": all(checks.values()),
        }
    return {
        "metrics": metrics,
        "screens": screens,
        "passes_both_baselines": all(s["passes"] for s in screens.values()),
    }
