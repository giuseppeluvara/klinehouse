"""Prequential scope, factor recovery, uncertainty shrinkage and reliability."""

import numpy as np
import pytest
import hashlib
from research import timestral_market_calibration as model


def batch():
    rng = np.random.default_rng(1008)
    cutoff = 1704067200
    t = (
        cutoff
        - 56 * 86400
        + np.arange(56 * 200) // 200 * 86400
        + np.arange(56 * 200) % 200 * 300
    )
    logits = rng.normal(0, 0.15, len(t))
    y = (rng.random(len(t)) < 1 / (1 + np.exp(-3 * logits))).astype(int)
    issued = t // 86400 * 86400
    return logits, y, t, t + 360, issued, cutoff


def test_recovers_miscalibration_with_scalar_and_preserves_orientation():
    x, y, t, end, issued, cutoff = batch()
    s = model.fit(x, y, t, end, issued, cutoff)
    assert 2.5 < s["raw_factor"] < 3.5
    assert 1 < s["adjusted_factor"] <= s["raw_factor"]
    a = np.full(len(x), cutoff)
    p = model.predict(s, x, a)
    np.testing.assert_array_equal(p > 0.5, x > 0)
    np.testing.assert_allclose(model.predict(s, -x, a), 1 - p)
    mirrored = model.fit(-x, 1 - y, t, end, issued, cutoff)
    assert mirrored["adjusted_factor"] == pytest.approx(s["adjusted_factor"], abs=1e-10)


def test_uncertain_and_zero_information_calibration_stays_at_identity():
    x, y, t, end, issued, cutoff = batch()
    s = model.fit(np.zeros_like(x), y, t, end, issued, cutoff)
    assert s["raw_factor"] == s["adjusted_factor"] == 1
    # A long constant-outcome run must enter the daily serial-correlation estimate.
    blocked = ((t // 86400) % 14 < 7).astype(int)
    s = model.fit(x, blocked, t, end, issued, cutoff)
    assert min(1, s["raw_factor"]) <= s["adjusted_factor"] <= max(1, s["raw_factor"])


def test_rejects_training_predictions_immature_labels_and_protected_targets():
    x, y, t, end, issued, cutoff = batch()
    with pytest.raises(ValueError):
        model.fit(x, y, t, end, t + 1, cutoff)
    with pytest.raises(ValueError):
        model.fit(x, y, t, np.full(len(t), cutoff + 1), issued, cutoff)
    with pytest.raises(ValueError):
        model.fit(x[:100], y[:100], t[:100], end[:100], issued[:100], cutoff)
    s = model.fit(x, y, t, end, issued, cutoff)
    with pytest.raises(ValueError):
        model.predict(s, x, np.full(len(x), model.BOUNDARY))


def test_reliability_keeps_flats_separate_and_reports_empty_bins():
    r = model.reliability(np.array([0, 1, 2, 0]), np.array([0.49, 0.8, 0.52, 0.48]))
    assert r["n"] == 3
    assert sum(b["n"] for b in r["bins"]) == 3
    assert any(b["accuracy"] is None for b in r["bins"])
    assert r["ece_fixed_bins"] == pytest.approx(1 - (0.51 + 0.52 + 0.52) / 3)


def test_independent_auditor_detects_changed_uncertainty_and_training_overlap():
    from scripts.verify_market_calibration import calibration_fit

    x, y, t, end, issued, cutoff = batch()
    s = model.fit(x, y, t, end, issued, cutoff)
    ids = np.column_stack((np.zeros(len(t), dtype=int), t))
    for name, values in (("ids", ids), ("labels", y)):
        s[f"training_{name}_sha256"] = hashlib.sha256(
            values.astype("<i8").tobytes()
        ).hexdigest()
    assert calibration_fit(s, x, y, ids, issued, cutoff) < 1e-8
    changed = {**s, "factor_variance": s["factor_variance"] + 0.01}
    with pytest.raises(AssertionError):
        calibration_fit(changed, x, y, ids, issued, cutoff)
    with pytest.raises(ValueError):
        calibration_fit(s, x, y, ids, t + 1, cutoff)
