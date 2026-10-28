import numpy as np
import pytest
from scipy.special import expit

from research.timestral_lead_lag import (
    BOUNDARY,
    PENALTY,
    SCALE_FLOOR,
    features,
    offsetfit,
    predict,
)


def sources(n=180):
    time = 1704067200 + 60 * np.arange(n)
    returns = np.full(n, 1e-4)
    returns[90] = 4e-4
    spot = dict(time=time, close=np.exp(np.cumsum(returns)))
    futures = dict(time=time, close=np.exp(np.cumsum(returns * 1.5)))
    return spot, futures


def training():
    rng = np.random.default_rng(281)
    cutoff = int(np.datetime64("2024-04-01T00:00:00", "s").astype(np.int64))
    days = cutoff - 90 * 86400 + np.arange(90) * 86400
    count = 1 + np.arange(90) % 5
    anchor = np.repeat(days, count)
    x = rng.normal(size=(len(anchor), 2))
    offset = rng.normal(0, 0.3, len(anchor))
    y = (rng.uniform(size=len(anchor)) < expit(offset + x @ [0.3, -0.2])).astype(int)
    return x, y, offset, anchor, anchor + 360, cutoff


def test_exact_endpoints_and_rms_exclude_the_latest_return():
    spot, futures = sources()
    anchor = spot["time"][90:92] + 120
    price, lead, valid = features(spot, futures, anchor)
    assert valid.all()
    slog, flog = np.log(spot["close"]), np.log(futures["close"])
    for row, j in enumerate((90, 91)):
        rms = max(np.sqrt(np.mean(np.diff(slog[j - 61 : j]) ** 2)), SCALE_FLOOR)
        for column, h in enumerate((5, 60)):
            raw = slog[j] - slog[j - h]
            denom = rms * np.sqrt(h)
            assert price[row, column] == pytest.approx(np.clip(raw / denom, -8, 8))
            assert lead[row, column] == pytest.approx(
                np.clip((flog[j] - flog[j - h] - raw) / denom, -8, 8)
            )
    # The large latest return affects the feature but cannot inflate its own scale.
    assert price[0, 0] > price[1, 0]


def test_gaps_missing_endpoint_and_insufficient_history_are_not_filled():
    spot, futures = sources()
    anchors = spot["time"][[60, 90, 170]] + 120
    missing = {k: np.delete(v, 70) for k, v in futures.items()}
    price, lead, valid = features(spot, missing, anchors)
    np.testing.assert_array_equal(valid, [False, False, True])
    np.testing.assert_array_equal(price[:2], np.zeros((2, 2)))
    np.testing.assert_array_equal(lead[:2], np.zeros((2, 2)))
    latest_missing = {k: np.delete(v, 170) for k, v in futures.items()}
    assert not features(spot, latest_missing, anchors[-1:])[2][0]
    spot_missing = {k: np.delete(v, 70) for k, v in spot.items()}
    assert not features(spot_missing, futures, anchors[1:2])[2][0]


def test_future_appends_and_future_changes_cannot_affect_features():
    spot, futures = sources()
    anchors = spot["time"][[65, 80, 90]] + 120
    short_spot = {k: v[:91] for k, v in spot.items()}
    short_futures = {k: v[:91] for k, v in futures.items()}
    first = features(short_spot, short_futures, anchors)
    for a, b in zip(first, features(spot, futures, anchors)):
        np.testing.assert_array_equal(a, b)
    spot["close"][91:] *= 2
    futures["close"][91:] *= 0.5
    for a, b in zip(first, features(spot, futures, anchors)):
        np.testing.assert_array_equal(a, b)


def test_day_balanced_offset_fit_and_independent_gradient():
    x, y, offset, anchor, end, cutoff = training()
    state = offsetfit(x, y, offset, anchor, end, cutoff)
    _, inv, counts = np.unique(anchor // 86400, return_inverse=True, return_counts=True)
    w = 1 / (90 * counts[inv])
    scale = np.maximum(np.sqrt((w[:, None] * x**2).sum(0)), SCALE_FLOOR)
    np.testing.assert_allclose(state["scale"], scale, atol=1e-14)
    z = np.clip(x / scale, -3, 3)
    beta = np.array(state["coefficients"])
    prob = expit(offset + z @ beta)
    gradient = (z * (w * (prob - y))[:, None]).sum(0) + PENALTY * beta
    assert np.max(np.abs(gradient)) < 1e-7
    np.testing.assert_allclose(state["gradient"], gradient, atol=1e-14)
    loss = -sum(w * (y * np.log(prob) + (1 - y) * np.log1p(-prob)))
    assert state["training_loss"] == pytest.approx(loss, abs=1e-14)
    assert state["objective"] == pytest.approx(
        loss + PENALTY / 2 * sum(beta**2), abs=1e-14
    )
    zero = predict(
        state, np.zeros((3, 2)), np.array([-0.3, 0, 0.3]), cutoff + np.arange(3) * 300
    )
    np.testing.assert_array_equal(zero, expit([-0.3, 0, 0.3]))


def test_duplicating_one_days_rows_does_not_change_the_fit():
    x, y, offset, anchor, end, cutoff = training()
    state = offsetfit(x, y, offset, anchor, end, cutoff)
    duplicate = np.flatnonzero(anchor // 86400 == anchor[20] // 86400)
    indices = np.r_[np.arange(len(y)), duplicate, duplicate]
    copied = offsetfit(
        x[indices], y[indices], offset[indices], anchor[indices], end[indices], cutoff
    )
    np.testing.assert_allclose(copied["scale"], state["scale"], atol=1e-14)
    np.testing.assert_allclose(
        copied["coefficients"], state["coefficients"], atol=1e-10
    )
    reflected = offsetfit(-x, 1 - y, -offset, anchor, end, cutoff)
    np.testing.assert_allclose(
        reflected["coefficients"], state["coefficients"], atol=1e-10
    )


def test_invalid_sources_training_windows_and_protected_predictions_rejected():
    spot, futures = sources()
    anchor = spot["time"][90:91] + 120
    bad = {k: v.copy() for k, v in spot.items()}
    bad["close"][0] = 0
    with pytest.raises(ValueError):
        features(bad, futures, anchor)
    bad = {k: np.r_[v, v[-1]] for k, v in spot.items()}
    with pytest.raises(ValueError):
        features(bad, futures, anchor)
    protected = dict(time=np.r_[spot["time"], BOUNDARY], close=np.r_[spot["close"], 1])
    with pytest.raises(ValueError):
        features(protected, futures, anchor)
    with pytest.raises(ValueError):
        features(spot, futures, np.array([BOUNDARY - 300]))
    x, y, offset, t, end, cutoff = training()
    for bad_t, bad_end in ((t - 86400, end - 86400), (t, end + 1)):
        with pytest.raises(ValueError):
            offsetfit(x, y, offset, bad_t, bad_end, cutoff)
    with pytest.raises(ValueError):
        offsetfit(x, y, offset, t, end, cutoff - 86400)
    with pytest.raises(ValueError):
        offsetfit(x[:20], y[:20], offset[:20], t[:20], end[:20], cutoff)
    state = offsetfit(x, y, offset, t, end, cutoff)
    for prediction_time in (cutoff - 60, BOUNDARY - 300):
        with pytest.raises(ValueError):
            predict(state, x[:1], offset[:1], np.array([prediction_time]))
