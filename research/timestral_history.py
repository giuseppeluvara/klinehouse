"""Controlled training-history experiment with native-hourly causal state."""

from datetime import timedelta
import math
import numpy as np

from .timestral_experiment import utc
from .timestral_funding import PARAMETERS

FEATURES = (
    "return4_scaled",
    "return24_scaled",
    "return168_scaled",
    "volatility24_over168",
    "last_range_scaled",
    "volume24_vs168",
    "imbalance24",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)


def features(context, anchor, horizon):
    anchor = utc(anchor)
    if (
        horizon not in (4, 24)
        or len(context) != 169
        or any(
            utc(r["time"]) != anchor - timedelta(hours=169 - i)
            for i, r in enumerate(context)
        )
    ):
        raise ValueError("expected169 consecutive past hourly candles")
    logs = np.log([r["close"] for r in context])
    returns = np.diff(logs)
    vol24 = float(np.linalg.norm(returns[-24:]))
    vol168 = float(np.linalg.norm(returns))
    scale = max(vol24 * math.sqrt(horizon / 24), 1e-6)
    volume = np.asarray([r["volume"] for r in context])
    taker = np.asarray([r["taker"] for r in context])
    values = [
        *((logs[-1] - logs[-1 - lag]) / scale for lag in (4, 24, 168)),
        vol24 / max(vol168 / math.sqrt(7), 1e-6),
        math.log(context[-1]["high"] / context[-1]["low"]) / scale,
        float(np.log1p(volume[-24:].mean()) - np.log1p(volume[-168:].mean())),
        float(2 * taker[-24:].sum() / volume[-24:].sum() - 1)
        if volume[-24:].sum() > 0
        else 0.0,
        math.sin(2 * math.pi * anchor.hour / 24),
        math.cos(2 * math.pi * anchor.hour / 24),
        math.sin(2 * math.pi * anchor.weekday() / 7),
        math.cos(2 * math.pi * anchor.weekday() / 7),
    ]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite state")
    return values, scale


def examples(hours):
    times = [utc(r["time"]) for r in hours]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("unordered hours")
    gaps = np.r_[
        0, np.cumsum([b - a != timedelta(hours=1) for a, b in zip(times, times[1:])])
    ]
    rows = []
    audit = {"missing_context": 0, "missing_target": 0}
    for i in range(169, len(hours)):
        anchor = times[i]
        for h in (4, 24):
            if anchor.hour % h:
                continue
            if (
                times[i - 1] + timedelta(hours=1) != anchor
                or gaps[i - 1] != gaps[i - 169]
            ):
                audit["missing_context"] += 1
                continue
            if i + h > len(hours) or gaps[i + h - 1] != gaps[i]:
                audit["missing_target"] += 1
                continue
            values, scale = features(hours[i - 169 : i], anchor, h)
            rows.append(
                {
                    "anchor": hours[i]["time"],
                    "label_end": (anchor + timedelta(hours=h)).isoformat(),
                    "horizon_hours": h,
                    "features": values,
                    "scale": scale,
                    "actual": math.log(
                        hours[i + h - 1]["close"] / hours[i - 1]["close"]
                    ),
                }
            )
    return rows, audit


def fit(rows, cutoff, horizon):
    from sklearn.ensemble import HistGradientBoostingRegressor

    cutoff = utc(cutoff)
    if len(rows) < (60 if horizon == 24 else 360) or any(
        utc(r["label_end"]) > cutoff or r["horizon_hours"] != horizon for r in rows
    ):
        raise ValueError("insufficient or unmatured training")
    x = np.asarray([r["features"] for r in rows])
    scale = np.asarray([r["scale"] for r in rows])
    y = np.asarray([r["actual"] for r in rows])
    weights = (
        np.asarray(
            [
                2 ** (-(cutoff - utc(r["label_end"])).total_seconds() / (365 * 86400))
                for r in rows
            ]
        )
        * scale
    )
    weights *= len(rows) / weights.sum()
    model = HistGradientBoostingRegressor(**PARAMETERS)
    model.fit(x, y / scale, sample_weight=weights)
    return model


def metrics(rows):
    y = np.asarray([r["actual"] for r in rows])
    p = np.asarray([r["point"] for r in rows])
    diff = abs(p - y) - abs(y)
    months = sorted({r["anchor"][:7] for r in rows})
    start = utc(rows[0]["anchor"]).replace(day=1, hour=0)
    blocks = np.asarray([(utc(r["anchor"]) - start).days // 7 for r in rows])
    n = int(blocks.max() + 1)
    sizes = np.bincount(blocks, minlength=n)
    sums = np.bincount(blocks, weights=diff, minlength=n)
    choices = np.random.default_rng(20261026).integers(0, n, (4000, n))
    interval = np.quantile(
        sums[choices].sum(axis=1) / sizes[choices].sum(axis=1),
        [0.05 / 24, 1 - 0.05 / 24],
    )
    wins = sum(
        float(diff[[i for i, r in enumerate(rows) if r["anchor"][:7] == m]].mean()) < 0
        for m in months
    )
    gain = float(1 - abs(p - y).mean() / abs(y).mean())
    checks = {
        "gain_at_least_one_percent": gain >= 0.01,
        "rmse_no_worse": bool(np.mean((p - y) ** 2) <= np.mean(y * y)),
        "two_thirds_months": wins >= math.ceil(2 * len(months) / 3),
        "adjusted_upper_below_zero": float(interval[1]) < 0,
    }
    return {
        "samples": len(rows),
        "mae": float(abs(p - y).mean()),
        "unchanged_mae": float(abs(y).mean()),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "unchanged_rmse": float(np.sqrt(np.mean(y * y))),
        "gain": gain,
        "monthly_wins": wins,
        "months": len(months),
        "interval99_5833": interval.tolist(),
        "checks": checks,
        "passes_historical_screen": all(checks.values()),
        "solution_found": False,
    }
