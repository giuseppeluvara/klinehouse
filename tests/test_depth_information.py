import numpy as np
import pytest

from research.depth_information import fit_first_day, predict


def data(n=240):
    first = 1793374200
    t = first + np.arange(n) * 300
    x = np.tile(np.where(np.arange(n) % 2, 0.7, -0.7)[:, None], (1, 3))
    y = np.where(x > 0, 2, 0)
    return first, t, np.full((n, 3), 0.5), x, y, t[:, None] + np.full((n, 3), 370.0)


def test_fixed_convex_fit_recovers_planted_information_and_reflection():
    first, t, q, x, y, at = data()
    s = fit_first_day(t, q, x, y, at, first_anchor=first)
    assert s["status"] == "frozen_after_first_day" and s["coefficient"] > 0
    assert abs(s["gradient"]) < 1e-10
    reflected = fit_first_day(t, q, -x, y, at, first_anchor=first)
    assert reflected["coefficient"] == pytest.approx(-s["coefficient"])


def test_rejects_immature_and_late_received_training_outcomes():
    first, t, q, x, y, at = data()
    at[0, 0] = first + 86401
    with pytest.raises(ValueError):
        fit_first_day(t, q, x, y, at, first_anchor=first)
    first, t, q, x, y, at = data()
    t[-1] = first + 86100
    at[-1] = t[-1] + 370
    with pytest.raises(ValueError):
        fit_first_day(t, q, x, y, at, first_anchor=first)


def test_insufficient_data_stays_baseline_without_extending_window():
    first, t, q, x, y, at = data(50)
    s = fit_first_day(t, q, x, y, at, first_anchor=first)
    assert s["coefficient"] == 0 and s["status"] == "insufficient_training"
    p = np.array([[0.48, 0.04, 0.48]])
    np.testing.assert_allclose(predict(s, p, [0.7], anchor=first + 86400), p)


def test_flat_labels_do_not_enter_direction_fit_but_flat_probability_is_preserved():
    first, t, q, x, y, at = data()
    y[:30, 0] = 1
    s = fit_first_day(t, q, x, y, at, first_anchor=first)
    assert s["moving_labels_per_asset"] == [210, 240, 240]
    p = np.array([[0.43, 0.07, 0.50], [0.45, 0.09, 0.46]])
    out = predict(s, p, [0.7, -0.7], anchor=s["cutoff"])
    np.testing.assert_array_equal(p[:, 1], out[:, 1])
    np.testing.assert_allclose(out.sum(1), 1)
    assert out[0, 2] > p[0, 2] and out[1, 2] < p[1, 2]


def test_no_prediction_before_frozen_training_cutoff():
    first, t, q, x, y, at = data()
    s = fit_first_day(t, q, x, y, at, first_anchor=first)
    with pytest.raises(ValueError):
        predict(s, [[0.49, 0.02, 0.49]], [0], anchor=s["cutoff"] - 300)


def test_fractional_anchor_is_rejected_before_integer_conversion():
    first, t, q, x, y, at = data()
    t = t.astype(float)
    t[0] += 0.5
    with pytest.raises(ValueError):
        fit_first_day(t, q, x, y, at, first_anchor=first)


@pytest.mark.parametrize("field", ["cutoff", "first_anchor"])
def test_nonfinite_model_timestamps_cannot_bypass_chronology(field):
    first, t, q, x, y, at = data()
    s = fit_first_day(t, q, x, y, at, first_anchor=first)
    s[field] = np.nan
    with pytest.raises(ValueError):
        predict(s, [[0.49, 0.02, 0.49]], [0], anchor=first + 86400)


def test_nonfinite_issue_and_wrong_schema_are_rejected():
    first, t, q, x, y, at = data()
    s = fit_first_day(t, q, x, y, at, first_anchor=first)
    with pytest.raises(ValueError):
        predict(s, [[0.49, 0.02, 0.49]], [0], anchor=np.nan)
    s["schema"] = "unregistered"
    with pytest.raises(ValueError):
        predict(s, [[0.49, 0.02, 0.49]], [0], anchor=first + 86400)
