"""Fixed nonlinear state/foundation-model/funding information ablation."""

from datetime import timedelta
import numpy as np

from .timestral_experiment import utc

CONDITIONS = {"state": 5, "state_timestral": 7, "state_timestral_funding": 9}
PARAMETERS = {
    "loss": "quantile",
    "quantile": 0.5,
    "max_iter": 100,
    "learning_rate": 0.03,
    "max_leaf_nodes": 3,
    "max_depth": 2,
    "min_samples_leaf": 30,
    "l2_regularization": 10.0,
    "early_stopping": False,
    "random_state": 7,
}


def funding_features(records, anchor):
    """Conservative assumed availability: one hour after final settlement."""
    anchor = utc(anchor)
    known = [r for r in records if utc(r["time"]) + timedelta(hours=1) <= anchor]
    if len(known) < 3:
        raise ValueError("insufficient settled funding history")
    tail = known[-3:]
    if any(
        utc(b["time"]) - utc(a["time"])
        > timedelta(hours=max(a["interval_hours"], b["interval_hours"]), minutes=5)
        for a, b in zip(tail, tail[1:])
    ):
        raise ValueError("gap in funding history")
    if anchor - utc(tail[-1]["time"]) > timedelta(
        hours=tail[-1]["interval_hours"] + 1, minutes=5
    ):
        raise ValueError("stale funding history")
    if any(
        not np.isfinite([r["rate"], r["interval_hours"]]).all()
        or r["interval_hours"] <= 0
        for r in tail
    ):
        raise ValueError("invalid funding")
    return [
        tail[-1]["rate"] / tail[-1]["interval_hours"] * 10000,
        sum(r["rate"] for r in tail) / sum(r["interval_hours"] for r in tail) * 10000,
    ], tail[-1]["time"]


def fit(rows, cutoff, horizon, condition):
    from sklearn.ensemble import HistGradientBoostingRegressor

    cutoff = utc(cutoff)
    if not rows or any(
        utc(r["label_end"]) > cutoff or r["horizon_hours"] != horizon for r in rows
    ):
        raise ValueError("only matured labels of one horizon may be fitted")
    if len({r["sample_id"] for r in rows}) != len(rows) or len(rows) < (
        60 if horizon == 24 else 360
    ):
        raise ValueError("duplicate or insufficient training samples")
    ordered = sorted(rows, key=lambda r: r["anchor"])
    if any(
        utc(r["label_end"]) != utc(r["anchor"]) + timedelta(hours=horizon) for r in rows
    ) or any(
        utc(b["anchor"]) < utc(a["label_end"]) for a, b in zip(ordered, ordered[1:])
    ):
        raise ValueError("invalid or overlapping target intervals")
    columns = CONDITIONS[condition]
    x = np.asarray([r["features"][:columns] for r in rows])
    scale = np.asarray([r["scale"] for r in rows])
    y = np.asarray([r["actual"] for r in rows])
    if (
        not np.isfinite(x).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.isfinite(y).all()
    ):
        raise ValueError("invalid training values")
    decay = np.asarray(
        [
            2 ** (-(cutoff - utc(r["label_end"])).total_seconds() / (60 * 86400))
            for r in rows
        ]
    )
    weights = decay * scale
    weights *= len(rows) / weights.sum()
    model = HistGradientBoostingRegressor(**PARAMETERS)
    model.fit(x, y / scale, sample_weight=weights)
    return model


def predict(model, features, scale):
    values = np.asarray(features[: model.n_features_in_], dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid forecast features")
    return float(model.predict(values[None, :])[0] * scale)
