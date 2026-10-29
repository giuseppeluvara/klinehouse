"""Conditional position features; the fixed lag is not certified availability."""

import numpy as np

from research.timestral_lead_lag import (  # noqa: F401
    BOUNDARY,
    offsetfit,
    predict,
)

ASSUMED_LAG_SECONDS = 900
TIMING_CERTIFIED = False


def metrics_features(data, anchors, spot_hour):
    """Return two conditional features and validity, without retiming raw rows."""
    label = np.asarray(data["label"])
    values = np.asarray(data["values"], dtype=float)
    anchor = np.asarray(anchors)
    hour = np.asarray(spot_hour, dtype=float)
    if (
        label.ndim != 1
        or values.shape != (len(label), 6)
        or np.asarray(data["file_day"]).shape != label.shape
        or np.asarray(data["row"]).shape != label.shape
        or not np.issubdtype(label.dtype, np.number)
        or not np.isfinite(label).all()
        or np.any(label != np.floor(label))
        or np.any(label < 0)
        or np.any(label >= BOUNDARY - 300)
    ):
        raise ValueError("invalid metrics schema or plausibly protected content time")
    if (
        anchor.ndim != 1
        or hour.shape != anchor.shape
        or not np.issubdtype(anchor.dtype, np.number)
        or not np.isfinite(anchor).all()
        or not np.isfinite(hour).all()
        or np.any(anchor != np.floor(anchor))
        or np.any(anchor < 0)
        or np.any(anchor % 300)
        or np.any(anchor > BOUNDARY - 360)
    ):
        raise ValueError("invalid forecast anchors, spot returns or protected targets")
    label = label.astype(np.int64)
    anchor = anchor.astype(np.int64)
    features = np.zeros((len(anchor), 2))
    valid = np.zeros(len(anchor), dtype=bool)
    if not len(label) or not len(anchor):
        return features, valid
    _, inverse, counts = np.unique(label, return_inverse=True, return_counts=True)
    usable = (
        (counts[inverse] == 1)
        & (label % 300 == 0)
        & np.isfinite(values[:, [0, 3]]).all(axis=1)
        & (values[:, 0] > 0)
        & (values[:, 3] > 0)
    )
    order = np.argsort(label[usable], kind="stable")
    observed = label[usable][order]
    values = values[usable][order]
    if not len(observed):
        return features, valid
    latest = anchor - ASSUMED_LAG_SECONDS
    index = np.searchsorted(observed, latest)
    j = np.minimum(index, len(observed) - 1)
    old = np.maximum(j - 12, 0)
    valid = (
        (index < len(observed))
        & (observed[j] == latest)
        & (j >= 12)
        & (observed[j] - observed[old] == 3600)
    )
    if valid.any():
        last, first = j[valid], old[valid]
        oi_change = np.log(values[last, 0]) - np.log(values[first, 0])
        ratio_now, ratio_old = values[last, 3], values[first, 3]
        imbalance_change = (ratio_now - 1) / (ratio_now + 1) - (ratio_old - 1) / (
            ratio_old + 1
        )
        features[valid, 0] = np.clip(np.clip(hour[valid], -3, 3) * oi_change, -8, 8)
        features[valid, 1] = np.clip(imbalance_change, -8, 8)
    return features, valid
