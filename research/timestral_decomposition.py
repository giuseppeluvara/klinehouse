"""Causal sign/variance decomposition; volatility success is not price success."""

from datetime import timedelta
import hashlib
import math

import numpy as np
from sklearn.linear_model import GammaRegressor, LogisticRegression

from .timestral_experiment import canonical, stamp, utc
from .timestral_history import features as state_features

DIRECTION = dict(C=0.1, solver="lbfgs", max_iter=1000, tol=1e-8)
VARIANCE = dict(alpha=0.01, max_iter=1000, tol=1e-8)
FLOOR = 1e-12


def origin(context, anchor, horizon):
    anchor = utc(anchor)
    if (
        horizon not in (4, 24)
        or len(context) != 721
        or any(
            utc(r["time"]) != anchor - timedelta(hours=721 - i)
            for i, r in enumerate(context)
        )
    ):
        raise ValueError("require 721 consecutive closed past hours")
    closes = np.asarray([r["close"] for r in context])
    if not np.isfinite(closes).all() or (closes <= 0).any():
        raise ValueError("invalid past price")
    squares = np.diff(np.log(closes)) ** 2
    state, _ = state_features(context[-169:], anchor, horizon)
    rv = [max(float(squares[-n:].mean()), FLOOR) for n in (24, 168, 720)]
    weights = 2.0 ** (-np.arange(719, -1, -1) / 24)
    return {
        "state": state,
        "variance_features": np.log(rv).tolist(),
        "variance_persistence": horizon * rv[0],
        "variance_ewma": horizon
        * max(float(np.average(squares, weights=weights)), FLOOR),
    }


def examples(hours):
    times = [utc(r["time"]) for r in hours]
    if any(b <= a for a, b in zip(times, times[1:])) or any(
        t >= utc("2026-08-01") for t in times
    ):
        raise ValueError("unordered or protected hours")
    gaps = np.r_[
        0, np.cumsum([b - a != timedelta(hours=1) for a, b in zip(times, times[1:])])
    ]
    prices = np.log([r["close"] for r in hours])
    rows, audit = (
        [],
        {
            "missing_context": 0,
            "missing_target": 0,
            "zero_return": 0,
            "variance_floor": 0,
        },
    )
    for i in range(721, len(hours)):
        for h in (4, 24):
            if times[i].hour % h:
                continue
            if gaps[i] != gaps[i - 721]:
                audit["missing_context"] += 1
                continue
            if i + h > len(hours) or gaps[i + h - 1] != gaps[i]:
                audit["missing_target"] += 1
                continue
            features = origin(hours[i - 721 : i], times[i], h)
            future = np.diff(prices[i - 1 : i + h])
            terminal = float(prices[i + h - 1] - prices[i - 1])
            variance = float(future @ future)
            audit["zero_return"] += int(terminal == 0)
            audit["variance_floor"] += int(variance < FLOOR)
            rows.append(
                {
                    "sample_id": f"{stamp(times[i])}:{h}",
                    "anchor": stamp(times[i]),
                    "label_end": stamp(times[i] + timedelta(hours=h)),
                    "horizon_hours": h,
                    **features,
                    "return": terminal,
                    "variance": max(variance, FLOOR),
                }
            )
    return rows, audit


def fit(rows, cutoff, horizon):
    cutoff = utc(cutoff)
    ordered = sorted(rows, key=lambda r: r["anchor"])
    if (
        len(rows) < (365 if horizon == 24 else 2190)
        or any(
            r["horizon_hours"] != horizon
            or utc(r["label_end"]) > cutoff
            or utc(r["label_end"]) != utc(r["anchor"]) + timedelta(hours=horizon)
            for r in rows
        )
        or any(
            utc(b["anchor"]) < utc(a["label_end"]) for a, b in zip(ordered, ordered[1:])
        )
    ):
        raise ValueError("insufficient, overlapping or unmatured training")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate training")
    result = {
        "cutoff": stamp(cutoff),
        "samples": len(rows),
        "latest_label_end": max(r["label_end"] for r in rows),
        "sample_ids_sha256": hashlib.sha256(
            canonical([r["sample_id"] for r in rows])
        ).hexdigest(),
    }
    for kind, columns in (("direction", "state"), ("variance", "variance_features")):
        train = [r for r in rows if r["return"] != 0] if kind == "direction" else rows
        x = np.asarray([r[columns] for r in train])
        y = np.asarray(
            [
                int(r["return"] > 0) if kind == "direction" else r["variance"]
                for r in train
            ]
        )
        if (
            not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or (kind == "variance" and (y <= 0).any())
        ):
            raise ValueError("invalid training values")
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
        z = np.clip((x - mean) / scale, -8, 8)
        if kind == "direction":
            if len(np.unique(y)) != 2:
                raise ValueError("both direction classes required")
            model = LogisticRegression(**DIRECTION).fit(z, y)
            unit = 1.0
            baseline = float((y.sum() + 1) / (len(y) + 2))
        else:
            unit = float(y.mean())
            model = GammaRegressor(**VARIANCE).fit(z, y / unit)
            baseline = unit
        iterations = int(np.max(np.atleast_1d(model.n_iter_)))
        if iterations >= 1000:
            raise ValueError("optimizer did not converge")
        result[kind] = {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "coef": np.ravel(model.coef_).tolist(),
            "intercept": float(np.ravel(model.intercept_)[0]),
            "unit": unit,
            "baseline": baseline,
            "samples": len(train),
            "iterations": iterations,
        }
    return result


def predict(parameters, row):
    if utc(parameters["cutoff"]) > utc(row["anchor"]):
        raise ValueError("fit comes from future")
    result = {}
    for kind, column in (("direction", "state"), ("variance", "variance_features")):
        p = parameters[kind]
        x = np.asarray(row[column], dtype=float)
        if x.shape != np.asarray(p["mean"]).shape or not np.isfinite(x).all():
            raise ValueError("invalid origin features")
        z = np.clip((x - p["mean"]) / p["scale"], -8, 8)
        eta = float(z @ p["coef"] + p["intercept"])
        value = (
            1 / (1 + math.exp(-max(-700, min(700, eta))))
            if kind == "direction"
            else p["unit"] * math.exp(eta)
        )
        if not np.isfinite(value) or (kind == "variance" and value <= 0):
            raise ValueError("invalid forecast")
        result[kind] = value
    result.update(
        direction_prior=parameters["direction"]["baseline"],
        direction_chance=0.5,
        variance_mean=parameters["variance"]["baseline"],
        variance_persistence=row["variance_persistence"],
        variance_ewma=row["variance_ewma"],
    )
    return result


def losses(y, p, kind):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("invalid scoring arrays")
    if kind == "direction":
        if (p < 0).any() or (p > 1).any() or not np.isin(y, [0, 1]).all():
            raise ValueError("invalid probability or label")
        safe = np.clip(p, 1e-12, 1 - 1e-12)
        return (p - y) ** 2, -(y * np.log(safe) + (1 - y) * np.log1p(-safe))
    if kind != "variance" or (y <= 0).any() or (p <= 0).any():
        raise ValueError("positive variance required")
    ratio = y / p
    return ratio - np.log(ratio) - 1, (np.log(y) - np.log(p)) ** 2


def score(rows, kind):
    rows = [r for r in rows if kind != "direction" or r["return"] != 0]
    if not rows:
        raise ValueError("empty evaluation")
    y = np.asarray(
        [int(r["return"] > 0) if kind == "direction" else r["variance"] for r in rows]
    )
    names = (
        [kind, "direction_chance", "direction_prior"]
        if kind == "direction"
        else [kind, "variance_persistence", "variance_ewma", "variance_mean"]
    )
    scores, values = {}, {}
    for name in names:
        p = np.asarray([r["predictions"][name] for r in rows])
        a, b = losses(y, p, kind)
        values[name] = (a, b)
        scores[name] = {
            "primary": float(a.mean()),
            "secondary": float(b.mean()),
            "samples": len(rows),
        }
        if kind == "direction":
            scores[name]["accuracy"] = float(((p >= 0.5) == y).mean())
            scores[name]["mean_probability"] = float(p.mean())
    months = sorted({r["anchor"][:7] for r in rows})
    start = utc(rows[0]["anchor"]).replace(day=1, hour=0)
    blocks = np.asarray([(utc(r["anchor"]) - start).days // 7 for r in rows])
    n = int(blocks.max() + 1)
    sizes = np.bincount(blocks, minlength=n)
    draws = np.random.default_rng(20261027).integers(0, n, (4000, n))
    screens = {}
    for name in names[1:3]:
        diff = values[kind][0] - values[name][0]
        sums = np.bincount(blocks, weights=diff, minlength=n)
        interval = np.quantile(
            sums[draws].sum(axis=1) / sizes[draws].sum(axis=1),
            [0.05 / 48, 1 - 0.05 / 48],
        )
        gain = (
            1 - scores[kind]["primary"] / scores[name]["primary"]
            if scores[name]["primary"] > 0
            else None
        )
        wins = sum(
            float(
                diff[[i for i, r in enumerate(rows) if r["anchor"][:7] == month]].mean()
            )
            < 0
            for month in months
        )
        checks = {
            "gain_at_least_one_percent": gain is not None and gain >= 0.01,
            "secondary_no_worse": scores[kind]["secondary"]
            <= scores[name]["secondary"],
            "two_thirds_months": wins >= math.ceil(2 * len(months) / 3),
            "adjusted_upper_below_zero": float(interval[1]) < 0,
        }
        screens[name] = {
            "gain": gain,
            "monthly_wins": wins,
            "months": len(months),
            "interval99_791667": interval.tolist(),
            "checks": checks,
            "passes": all(checks.values()),
        }
    return {
        "metrics": scores,
        "screens": screens,
        "passes_both_baselines": all(s["passes"] for s in screens.values()),
        "target": kind,
        "trading_validation": False,
    }
