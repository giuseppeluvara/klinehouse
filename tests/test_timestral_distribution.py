"""Known Gamma truth, functional mismatch and fair baseline calibration."""

from datetime import timedelta
import math
import numpy as np
import pytest
from scipy.special import digamma
from research import timestral_distribution as d
from research.timestral_experiment import stamp, utc


def sample():
    rng = np.random.default_rng(10)
    n = 3000
    start = utc("2020-01-01")
    rows = [
        {
            "sample_id": str(i),
            "anchor": stamp(start + timedelta(hours=i)),
            "label_end": stamp(start + timedelta(hours=i + 1)),
            "variance": float(y),
            "variance_features": [0.0, 0.0, 0.0],
            "variance_persistence": 0.5,
            "variance_ewma": 2.0,
        }
        for i, y in enumerate(rng.gamma(2, 0.5, n))
    ]
    parameters = {
        "cutoff": rows[-1]["label_end"],
        "samples": n,
        "variance": {
            "mean": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "coef": [0.0, 0.0, 0.0],
            "intercept": 0.0,
            "unit": 1.0,
        },
    }
    return rows, parameters


def test_calibration_recovers_known_gamma_and_treats_baselines_equally():
    rows, p = sample()
    c = d.calibrate(rows, p, p["cutoff"])
    assert c["parameters"]["gamma"]["shape"] == pytest.approx(2, rel=0.08)
    row = {**rows[-1], "anchor": p["cutoff"]}
    f = d.predict(c, p, row)
    for name in d.NAMES:
        assert f[name]["mean"] == pytest.approx(np.mean([r["variance"] for r in rows]))
        assert f[name]["expected_log"] == pytest.approx(
            np.mean(np.log([r["variance"] for r in rows]))
        )
        assert f[name]["lower90"] < f[name]["mean"] < f[name]["upper90"]
        assert 0 < f[name]["probability_above_recent"] < 1
    assert c["parameters"]["persistence"]["bias"] == pytest.approx(
        4 * c["parameters"]["ewma"]["bias"]
    )


def test_mean_of_log_and_log_of_mean_have_distinct_optima():
    rng = np.random.default_rng(42)
    y = rng.gamma(2, 0.5, 100000)
    expected_log = digamma(2) - math.log(2)
    mean = 1.0

    def q(forecast):
        return np.mean(y / forecast - np.log(y / forecast) - 1)

    assert q(mean) < q(math.exp(expected_log))
    assert np.mean((np.log(y) - expected_log) ** 2) < np.mean(
        (np.log(y) - math.log(mean)) ** 2
    )
    assert abs(expected_log - math.log(mean)) > 0.25


def test_future_and_invalid_calibration_rejected():
    rows, p = sample()
    with pytest.raises(ValueError, match="matured"):
        d.calibrate(rows, p, utc(p["cutoff"]) - timedelta(hours=1))
    c = d.calibrate(rows, p, p["cutoff"])
    with pytest.raises(ValueError, match="future"):
        d.predict(c, p, rows[0])
    bad = [{**rows[0], "variance": -1}, *rows[1:]]
    with pytest.raises(ValueError, match="positive"):
        d.calibrate(bad, p, p["cutoff"])


def test_identical_distributions_do_not_pass():
    rows, p = sample()
    c = d.calibrate(rows, p, p["cutoff"])
    start = utc("2021-01-01")
    evaluated = []
    for i, row in enumerate(rows[:300]):
        r = {**row, "anchor": stamp(start + timedelta(days=i))}
        r["distribution"] = d.predict(c, p, r)
        for name in d.NAMES:
            r["distribution"][name] = r["distribution"]["gamma"].copy()
        evaluated.append(r)
    result = d.score(evaluated)
    assert not result["passes_both_baselines"]
    assert all(s["gain"] == 0 for s in result["screens"].values())
