from pathlib import Path

import numpy as np
import pytest

from scripts.preview_shared_direction import forecast, point_features


@pytest.mark.parametrize(
    "anchor", ["2026-08-01T00:00:00Z", "2026-07-31T23:55:00Z", "2026-07-31T23:50:00"]
)
def test_shared_preview_refuses_protected_or_ambiguous_targets(anchor):
    with pytest.raises(ValueError):
        forecast(Path("/nonexistent"), anchor)


def test_point_forecast_matches_saved_outputs_without_future_context():
    root = Path(__file__).resolve().parents[1] / "reports/timestral-universal-20261027"
    result = forecast(root, "2026-07-31T23:50:00Z")
    assert result["shared_fit_key"] == "cross_7_all"
    for symbol, item in result["assets"].items():
        saved = np.load(root / f"{symbol}-predictions.npz")
        stamp = int(np.datetime64("2026-07-31T23:50:00", "s").astype(int))
        expected = saved["cross"][saved["anchor"] == stamp][0]
        np.testing.assert_allclose(
            list(item["probabilities"].values()), expected, atol=1e-9
        )


def test_point_features_require_all_assets():
    with pytest.raises(ValueError, match="all registered"):
        point_features({})
