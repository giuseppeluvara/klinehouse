from pathlib import Path

import pytest

from scripts.preview_direction_forecast import forecast


@pytest.mark.parametrize(
    "anchor",
    [
        "2026-08-01T00:00:00Z",
        "2026-07-31T23:55:00Z",
        "2026-07-31T23:54:00Z",
        "2026-07-31T23:50:00",
    ],
)
def test_preview_refuses_protected_targets_or_ambiguous_time(anchor):
    with pytest.raises(ValueError):
        forecast(Path("/nonexistent"), "ETHUSDT", anchor)


def test_frozen_eth_forecast_uses_only_past_context():
    root = (
        Path(__file__).resolve().parents[1]
        / "reports/timestral-microdirection-20261027"
    )
    result = forecast(root, "ETHUSDT", "2026-07-31T23:50:00Z")
    assert result["fit_cutoff"] == "2026-07-01T00:00:00+00:00"
    assert result["return_start"] == "2026-07-31T23:51:00+00:00"
    assert result["return_end"] == "2026-07-31T23:56:00+00:00"
    assert sum(result["probabilities"].values()) == pytest.approx(1)
