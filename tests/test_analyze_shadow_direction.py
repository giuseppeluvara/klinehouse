"""Synthetic diagnoses with known direction, ranking and capture boundaries."""

import copy
import json
import math

import pytest

from scripts.analyze_shadow_direction import (
    arithmetic,
    auc,
    collect_cases,
    diagnostics,
    summarize,
    write_result,
)


def case(label, probability, *, anchor=0, symbol="BTCUSDT", other=None):
    return {
        "anchor": anchor,
        "symbol": symbol,
        "label": label,
        "models": {
            "baseline": probability,
            "price_control": probability if other is None else other,
            "positions": probability if other is None else other,
        },
        "diagnostics": {"price_features": [0, -1], "uniform_signal": 1},
    }


def forecast_events(anchor=1000, *, eligible=True, committed=None):
    probability = [0.3, 0.1, 0.6]
    forecast = {
        "anchor": anchor,
        "symbol": "BTCUSDT",
        "issued_at": anchor + 10,
        "models": case(2, probability)["models"],
    }
    committed = anchor + 11 if committed is None else committed
    return [
        {
            "kind": "forecast",
            "seq": 1,
            "digest": "saved-hash",
            "received_at": anchor + 10,
            "payload": forecast,
        },
        {
            "kind": "forecast_commit",
            "seq": 2,
            "received_at": committed + 0.1,
            "payload": {
                "anchor": anchor,
                "symbol": "BTCUSDT",
                "forecast_seq": 1,
                "forecast_hash": "saved-hash",
                "eligible": eligible,
                "clock_healthy": True,
                "committed_at": committed,
            },
        },
        {
            "kind": "outcome",
            "seq": 3,
            "received_at": anchor + 361,
            "payload": {
                "anchor": anchor,
                "symbol": "BTCUSDT",
                "label": 2,
                "return_end": anchor + 360,
            },
        },
    ]


def test_flat_denominators_and_exact_direction_decomposition():
    cases = [
        case(0, [0.7, 0.1, 0.2]),
        case(1, [0.2, 0.6, 0.2]),
        case(2, [0.2, 0.1, 0.7]),
    ]
    result = summarize(cases, "baseline")
    assert result["n"] == 3 and result["direction_n"] == 2
    assert result["class_correct"] == 3 and result["direction_correct"] == 2
    assert result["realized_class_counts"] == {"0": 1, "1": 1, "2": 1}
    assert result["confusion_actual_down_up_by_predicted_down_up"] == [[1, 0], [0, 1]]
    assert result["balanced_accuracy"] == 1 and result["auc"] == 1
    assert result["decomposition"]["class_imbalance_component"] == 0
    assert result["decomposition"]["association_component"] == 0.5
    assert result["decomposition"]["identity_error"] == 0
    assert result["brier"] == pytest.approx((0.14 + 0.24 + 0.14) / 3)


def test_class_bias_can_raise_accuracy_without_ranking_information():
    cases = [case(y, [0.1, 0.1, 0.8]) for y in [2, 2, 2, 0]]
    result = summarize(cases, "baseline")
    assert result["direction_accuracy"] == 0.75
    assert result["balanced_accuracy"] == 0.5 and result["auc"] == 0.5
    assert result["decomposition"]["class_imbalance_component"] == 0.25
    assert result["decomposition"]["association_component"] == 0


def test_swapped_ranking_is_not_only_class_bias():
    result = summarize([case(0, [0.1, 0.1, 0.8]), case(2, [0.8, 0.1, 0.1])], "baseline")
    assert result["direction_accuracy"] == 0
    assert result["balanced_accuracy"] == 0 and result["auc"] == 0
    assert result["decomposition"]["class_imbalance_component"] == 0
    assert result["decomposition"]["association_component"] == -0.5
    assert auc([(0.5, 0), (0.5, 1)]) == 0.5


def test_threshold_uses_unconditional_probability_and_fixed_comparison_subset():
    # Conditional confidence .6 is insufficient when flat mass leaves p_up=.3.
    assert arithmetic([0.2, 0.5, 0.3], 2)["q"] == 0.6
    assert not arithmetic([0.2, 0.5, 0.3], 2)["selected"]
    assert arithmetic([0.38, 0.1, 0.52], 2)["selected"]
    cases = [
        case(2, [0.3, 0.1, 0.6], other=[0.46, 0.1, 0.44]),
        case(0, [0.44, 0.1, 0.46], anchor=300, other=[0.6, 0.1, 0.3]),
    ]
    result = diagnostics(cases)["assets"]["BTCUSDT"]
    assert result["own_threshold_subset"]["baseline"]["direction_correct"] == 1
    assert result["own_threshold_subset"]["positions"]["direction_correct"] == 1
    fixed = result["fixed_baseline_threshold_subset"]["models"]
    assert fixed["positions"]["direction_n"] == 1
    assert fixed["positions"]["direction_correct"] == 0
    assert result["flips_vs_baseline"]["positions"] == {
        "changed": 2,
        "helped": 1,
        "hurt": 1,
        "flat": 0,
    }


def test_neutral_control_preserves_flat_mass_and_does_not_invent_information():
    cases = [
        case(0, [0.2, 0.1, 0.7]),
        case(1, [0.3, 0.4, 0.3]),
        case(2, [0.7, 0.1, 0.2]),
    ]
    result = summarize(cases, "baseline", neutral=True)
    assert result["direction_n"] == 2 and result["direction_accuracy"] == 0.5
    assert result["auc"] == 0.5 and result["mean_q"] == 0.5
    assert result["log_loss"] == pytest.approx(
        -(math.log(0.45) * 2 + math.log(0.4)) / 3
    )


def test_future_outcome_and_ack_cannot_enter_an_earlier_prefix():
    events = forecast_events()
    cases, counts = collect_cases(events, now=1360.5)
    assert not cases and counts["pending_count"] == 1
    assert counts["mature_pending_count"] == 1
    cases, counts = collect_cases(events, now=1362)
    assert len(cases) == 1 and counts["scored_count"] == 1
    future_ack = copy.deepcopy(events)
    future_ack[1]["received_at"] = 1400
    assert not collect_cases(future_ack, now=1362)[0]
    with pytest.raises(ValueError, match="precedes its saved acknowledgement"):
        collect_cases(future_ack, now=1401)


@pytest.mark.parametrize("change", ["late", "unhealthy", "ineligible", "wrong_hash"])
def test_invalid_acknowledgements_are_excluded(change):
    events = forecast_events()
    ack = events[1]["payload"]
    if change == "late":
        ack["committed_at"] = 1031
        events[1]["received_at"] = 1032
    elif change == "unhealthy":
        ack["clock_healthy"] = False
    elif change == "ineligible":
        ack["eligible"] = False
    else:
        ack["forecast_hash"] = "different"
    cases, counts = collect_cases(events, now=1362)
    assert not cases and counts["ineligible_count"] == 1


def test_premature_outcome_is_rejected():
    events = forecast_events()
    events[2]["received_at"] = 1359
    with pytest.raises(ValueError, match="before maturity"):
        collect_cases(events, now=1362)


def test_utc_blocks_and_stratified_auc_keep_all_cases():
    cases = [
        case(0, [0.7, 0.1, 0.2], anchor=0),
        case(2, [0.2, 0.1, 0.7], anchor=300),
        case(2, [0.1, 0.1, 0.8], anchor=21600),
        case(0, [0.8, 0.1, 0.1], anchor=21900),
    ]
    result = diagnostics(cases)["assets"]["BTCUSDT"]["temporal"]
    assert len(result["blocks_utc_six_hours_by_anchor"]) == 2
    assert result["stratified_auc"]["baseline"] == {
        "comparable_pairs": 2,
        "pair_weighted_within_block_auc": 1,
    }


def test_output_is_new_and_never_written_to_live_pilot(tmp_path):
    result = {"source_run": str(tmp_path / "live"), "value": 1}
    output = tmp_path / "diagnosis.json"
    write_result(result, output)
    assert json.loads(output.read_text())["value"] == 1
    with pytest.raises(FileExistsError):
        write_result(result, output)
    live = tmp_path / "live"
    live.mkdir()
    with pytest.raises(ValueError, match="outside the live pilot"):
        write_result(result, live / "diagnosis.json")
