"""The research screen must not manufacture a successful solution."""

from datetime import timedelta
import json

import pytest

from research import timestral_capacity as capacity
from research.timestral_experiment import utc


def examples(factor):
    start = utc("2026-01-01")
    return [
        {
            "anchor": (start + timedelta(days=i)).isoformat(),
            "actual_log_return": 0.01 if i % 2 else -0.012,
            "predictions": {
                "candidate": {"point": factor * (0.01 if i % 2 else -0.012)}
            },
        }
        for i in range(212)
    ]


@pytest.mark.parametrize(
    "factor,expected", [(0.0, False), (-1.0, False), (0.005, False), (0.5, True)]
)
def test_ties_losses_tiny_gains_and_clear_signals_are_distinguished(factor, expected):
    result = capacity.screen(examples(factor), "candidate")
    assert result["passes_replay_screen"] is expected
    assert result["solution_found"] is False
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_improvement_concentrated_in_one_month_is_not_stable():
    rows = examples(0.0)
    for row in rows:
        if row["anchor"].startswith("2026-01"):
            row["predictions"]["candidate"]["point"] = row["actual_log_return"]
    result = capacity.screen(rows, "candidate")
    assert result["mae_improvement"] > 0.01
    assert result["monthly_wins"] == 1
    assert result["passes_replay_screen"] is False


def test_perfect_zero_baseline_cannot_be_beaten_or_cause_nonfinite_json():
    rows = examples(0.0)
    for row in rows:
        row["actual_log_return"] = 0.0
    result = capacity.screen(rows, "candidate")
    assert result["mae_improvement"] is None
    assert result["passes_replay_screen"] is False
    json.dumps(result, allow_nan=False)
    rows[0]["actual_log_return"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        capacity.screen(rows, "candidate")
