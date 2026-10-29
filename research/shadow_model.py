"""Frozen July direction models for a separate prospective shadow stream.

This module does not read historical sources or relax their protected-date guards.
The caller supplies already-received observations and records receipt provenance.
"""

from decimal import Decimal, InvalidOperation

import numpy as np
from scipy.special import expit, softmax

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
ARMS = ("baseline", "price_control", "positions")
FROZEN_CUTOFF = 1782864000  # 2026-07-01 UTC, the last allowed fitted snapshot.
PROTECTED_BOUNDARY = 1785542400
SPOT_MINUTES = 63
METRICS_LAG_SECONDS = 900
SCHEMA = "klinehouse-shadow-july2026-v1"


def validate_snapshot(snapshot):
    """Fail closed on malformed, future-trained or incompatible frozen states."""
    if (
        snapshot.get("schema") != SCHEMA
        or snapshot.get("cutoff") != FROZEN_CUTOFF
        or tuple(snapshot.get("symbols", [])) != SYMBOLS
        or snapshot.get("metrics_lag_seconds") != METRICS_LAG_SECONDS
        or not np.isfinite(snapshot.get("baseline_beta", np.nan))
        or not 0 <= snapshot["baseline_beta"] <= 8
    ):
        raise ValueError("invalid frozen shadow snapshot")
    flat = snapshot["flat_model"]
    for field, shape in (
        ("mean", (23,)),
        ("scale", (23,)),
        ("coef", (3, 23)),
        ("intercept", (3,)),
    ):
        value = np.asarray(flat[field], dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("invalid frozen flat model")
    if flat["cutoff"] != FROZEN_CUTOFF or np.any(np.asarray(flat["scale"]) < 1e-8):
        raise ValueError("invalid frozen flat model cutoff/scale")
    for symbol in SYMBOLS:
        value = np.asarray(snapshot["frequency"][symbol], dtype=float)
        if (
            value.shape != (3,)
            or not np.isfinite(value).all()
            or np.any(value <= 0)
            or abs(value.sum() - 1) > 1e-12
        ):
            raise ValueError("invalid frozen class frequency")
    for arm in ARMS[1:]:
        state = snapshot["corrections"][arm]
        for key in ("coefficients", "scale"):
            value = np.asarray(state[key], dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError("invalid frozen correction")
        if state["cutoff"] != FROZEN_CUTOFF or np.any(
            np.asarray(state["scale"]) < 1e-8
        ):
            raise ValueError("invalid frozen correction cutoff/scale")
        if state["baseline_beta"] != snapshot["baseline_beta"]:
            raise ValueError("correction offset disagrees with frozen baseline")


def _anchor(anchor):
    if (
        not np.isscalar(anchor)
        or not np.isfinite(anchor)
        or anchor != int(anchor)
        or anchor % 300
        or anchor < FROZEN_CUTOFF
    ):
        raise ValueError("aligned forecast anchor after the frozen fit required")
    return int(anchor)


def _spot_context(data, anchor):
    fields = ("time", "close", "volume", "trades", "taker")
    arrays = {key: np.asarray(data[key], dtype=float) for key in fields}
    t = arrays["time"]
    if (
        t.ndim != 1
        or len(t) < SPOT_MINUTES
        or not np.isfinite(t).all()
        or np.any(t != np.floor(t))
        or np.any(t % 60)
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError(
            "sorted unique minute timestamps and 63 closed minutes required"
        )
    if any(a.shape != t.shape for a in arrays.values()) or np.any(t >= anchor):
        raise ValueError("aligned fields and no unclosed/future candle required")
    mask = t >= anchor - SPOT_MINUTES * 60
    arrays = {key: value[mask] for key, value in arrays.items()}
    if not np.array_equal(arrays["time"], np.arange(anchor - 3780, anchor, 60)):
        raise ValueError("complete 63-minute context required; no gap filling")
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("finite minute inputs required")
    if (
        np.any(arrays["close"] <= 0)
        or np.any(arrays["volume"] < 0)
        or np.any(arrays["trades"] < 0)
        or np.any(arrays["trades"] != np.floor(arrays["trades"]))
        or np.any(arrays["taker"] < 0)
        or np.any(arrays["taker"] > arrays["volume"] + 1e-8)
    ):
        raise ValueError("invalid price or minute activity")
    return arrays


def spot_features(spot, anchor):
    """Reproduce the 23 flat-model features, uniform center and price control."""
    anchor = _anchor(anchor)
    if set(spot) != set(SYMBOLS):
        raise ValueError("all three registered assets required")
    own, lagged, centers, price = [], [], [], []
    for symbol in SYMBOLS:
        d = _spot_context(spot[symbol], anchor)
        log = np.log(d["close"])
        r = np.r_[0.0, np.diff(log)]
        i = len(log) - 1
        # Exact historical windows: the newest return is excluded from RMS.
        scale = np.sqrt(max(np.sum(r[i - 60 : i] ** 2) / 60, 1e-16))
        prev_scale = np.sqrt(max(np.sum(r[i - 61 : i - 1] ** 2) / 60, 1e-16))
        returns = np.clip(
            [(log[i] - log[i - h]) / scale / np.sqrt(h) for h in (1, 5, 15, 60)], -8, 8
        )
        lag = np.clip(
            [(log[i - 1] - log[i - 1 - h]) / prev_scale / np.sqrt(h) for h in (1, 5)],
            -8,
            8,
        )
        activity = np.log(
            max(d["trades"][i] / max(np.mean(d["trades"][i - 60 : i]), 1.0), 1e-6)
        )
        v, v5 = d["volume"][i], np.sum(d["volume"][i - 4 : i + 1])
        f1 = (2 * d["taker"][i] - v) / v if v > 0 else 0.0
        f5 = (2 * np.sum(d["taker"][i - 4 : i + 1]) - v5) / v5 if v5 > 0 else 0.0
        own.append(
            np.r_[
                returns,
                f1,
                f5,
                returns[0] * np.clip(activity, -4, 4),
                abs(returns[0]),
                np.log1p(d["trades"][i]),
                np.mean(r[i - 59 : i + 1] == 0),
                activity,
                np.log(scale),
                abs(f1),
            ]
        )
        lagged.append(lag)
        # Subtracting log[0] matches the frozen center's numerical convention.
        centered = log - log[0]
        centers.append(
            np.clip(
                -2
                * (centered[i] - np.mean(centered[i - 59 : i + 1]))
                / scale
                / np.sqrt(60),
                -8,
                8,
            )
        )
        j = i - 1
        price.append(
            np.clip(
                [(log[j] - log[j - h]) / prev_scale / np.sqrt(h) for h in (5, 60)],
                -8,
                8,
            )
        )
    own, lagged = np.asarray(own), np.asarray(lagged)
    x = []
    for a in range(3):
        others = [b for b in range(3) if b != a]
        market = own[others, :4].mean(axis=0)
        residual = np.clip(own[a, 0] - market[0], -8, 8)
        x.append(
            np.r_[
                own[a],
                market,
                lagged[others].mean(axis=0),
                residual * abs(residual) / 8,
                own[a, 0] * np.std(own[others, 0]),
                np.mean(own[others, 9]),
                abs(market[0]),
            ]
        )
    x = np.asarray(x)
    common = np.clip(-(x[:, 3] + 2 * x[:, 16]) / 3, -3, 3)
    own_sign = np.clip(-x[:, 3], -3, 3)
    signal = np.where(common * own_sign > 0, np.clip(np.mean(centers), -3, 3), common)
    return x, signal, np.asarray(price)


def position_features(metrics, anchor, spot_hour):
    """Use thirteen exact stock labels; missing/duplicate/invalid rows fail closed."""
    anchor = _anchor(anchor)
    expected = np.arange(anchor - 4500, anchor - 899, 300)
    result, valid, reasons = [], [], []
    for a, symbol in enumerate(SYMBOLS):
        rows = metrics.get(symbol, [])
        selected = []
        for label in expected:
            matches = [row for row in rows if row.get("label") == int(label)]
            if len(matches) != 1:
                break
            row = matches[0]
            try:
                values = np.array([float(row["oi"]), float(row["position_ratio"])])
            except (KeyError, TypeError, ValueError):
                break
            if not np.isfinite(values).all() or np.any(values <= 0):
                break
            selected.append(values)
        okay = len(selected) == 13
        valid.append(okay)
        reasons.append(None if okay else "missing_duplicate_or_invalid_metric_context")
        if okay:
            old, latest = selected[0], selected[-1]
            change = np.log(latest[0]) - np.log(old[0])
            ratio = (latest[1] - 1) / (latest[1] + 1) - (old[1] - 1) / (old[1] + 1)
            result.append(
                np.clip([np.clip(spot_hour[a], -3, 3) * change, ratio], -8, 8)
            )
        else:
            result.append([0.0, 0.0])
    return np.asarray(result), np.asarray(valid), reasons


def predict(snapshot, spot, metrics, anchor):
    """Produce three fixed arms; all corrections share the same missing-data fallback."""
    validate_snapshot(snapshot)
    anchor = _anchor(anchor)
    x, signal, price = spot_features(spot, anchor)
    positions, valid, reasons = position_features(metrics, anchor, price[:, 1])
    flat_state = snapshot["flat_model"]
    z = np.clip((x - flat_state["mean"]) / flat_state["scale"], -8, 8)
    raw_flat = softmax(
        z @ np.asarray(flat_state["coef"]).T + flat_state["intercept"], axis=1
    )[:, 1]
    flat = (raw_flat + np.array([snapshot["frequency"][s][1] for s in SYMBOLS])) / 2
    offset = snapshot["baseline_beta"] * signal
    q = {"baseline": expit(offset)}
    for name, features in (("price_control", price), ("positions", positions)):
        state = snapshot["corrections"][name]
        q[name] = (
            expit(
                offset
                + np.clip(features / state["scale"], -3, 3)
                @ np.asarray(state["coefficients"])
            )
            if valid.all()
            else q["baseline"].copy()
        )
    out = {}
    for a, symbol in enumerate(SYMBOLS):
        models = {
            name: [
                float((1 - flat[a]) * (1 - up[a])),
                float(flat[a]),
                float((1 - flat[a]) * up[a]),
            ]
            for name, up in q.items()
        }
        out[symbol] = {
            "models": models,
            "diagnostics": {
                "correction_available": bool(valid.all()),
                "metric_context_valid": bool(valid[a]),
                "fallback_assets": [s for b, s in enumerate(SYMBOLS) if not valid[b]],
                "metric_context_reason": reasons[a],
                "uniform_signal": float(signal[a]),
                "price_features": price[a].tolist(),
                "position_features": positions[a].tolist(),
                "flat_probability": float(flat[a]),
                "fit_cutoff": FROZEN_CUTOFF,
                "spot_first_open": anchor - 3780,
                "spot_last_open": anchor - 60,
                "metric_first_label": anchor - 4500,
                "metric_last_label": anchor - 900,
            },
        }
    return out


def outcome_label(entry_close, exit_close):
    """Classify the exact five-minute close-to-close target, with no deadband."""
    try:
        values = [Decimal(str(value)) for value in (entry_close, exit_close)]
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("positive finite target closes required") from exc
    if not all(value.is_finite() and value > 0 for value in values):
        raise ValueError("positive finite target closes required")
    return 2 if values[1] > values[0] else 0 if values[1] < values[0] else 1
