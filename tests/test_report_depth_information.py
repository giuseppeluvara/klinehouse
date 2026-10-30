"""Registered criteria and safe JSON output, without fitting or collecting."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import pytest

from scripts import report_depth_information as report

CUTOFF = 1793460900
END = 1793877000


def audit_fixture(*, moving=288, flats=0, one_date=False):
    first = (
        int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()) - 300
        if one_date
        else CUTOFF
    )
    cases, metrics = [], {}
    for symbol in report.SYMBOLS:
        for i in range(moving + flats):
            anchor = first + i * 300
            cases.append(
                dict(
                    symbol=symbol,
                    anchor=anchor,
                    label=(2 if i < moving else 1),
                    target_end=anchor + 360,
                    source_received_at=anchor + 361,
                    forecast_sha256="a" * 64,
                )
            )
        models = {}
        for arm, correct, brier, loss in (
            ("baseline", 190, 0.51, 0.81),
            ("price_control", 180, 0.52, 0.82),
            ("positions", 210, 0.40, 0.70),
            ("depth", 200, 0.50, 0.80),
        ):
            correct = min(moving, correct)
            models[arm] = dict(
                n=moving + flats,
                moving_n=moving,
                direction_correct=correct,
                direction_accuracy=correct / moving if moving else None,
                brier=brier if moving + flats else None,
                log_loss=loss if moving + flats else None,
            )
        metrics[symbol] = models
    return dict(
        verified=True,
        schema="independent-depth-information-audit-v1",
        as_of=CUTOFF + 2 * 86400,
        training_cutoff=CUTOFF,
        end_at=END,
        phase="model_frozen",
        model_fitted=True,
        scored_outcomes=len(cases),
        scored_cases=sorted(cases, key=lambda v: (v["anchor"], v["symbol"])),
        metrics=metrics,
        source_snapshots=dict(
            depth=dict(
                root="/new/depth-research-source", event_count=30, chain_tip="d" * 64
            ),
            pilot=dict(
                root="/new/shadow-pilot-20261029-v2", event_count=90, chain_tip="e" * 64
            ),
        ),
        candidate_forecasts=len(cases) + 3,
        pending_outcomes=3,
        mature_pending_outcomes=[[CUTOFF + 86400, "BTCUSDT"]],
        unconfirmed_forecasts=[[CUTOFF, "BTCUSDT"]],
        ineligible_forecasts=[],
        late_confirmations=[[CUTOFF, "BTCUSDT"]],
        clock_unhealthy_forecasts=[],
        missing_emissions=1,
        missing_reasons={"explicit_gap": 1},
        fallback_missing_book=12,
        fallback_zero_coefficient=0,
        unaccounted_anchor_symbols=[[CUTOFF + 300, "ETHUSDT"]],
    )


def test_exact_288_and_two_utc_dates_pass_jointly_without_requiring_positions_win():
    audit = audit_fixture()
    result = report.build_report(audit)
    assert result["criterion_status"] == "preliminary_criteria_met"
    assert result["evaluation_stage"] == "interim"
    assert result["completion_verified"] is False
    btc = result["assets"]["BTCUSDT"]
    assert btc["moving_n"] == 288 and len(btc["moving_target_end_utc_dates"]) == 2
    assert (
        btc["comparisons"]["baseline"]["delta_positive_is_better"]["direction_correct"]
        == 10
    )
    assert btc["comparisons"]["baseline"]["delta_positive_is_better"]["brier"] > 0
    assert btc["comparisons"]["positions"]["jointly_improved"] is False
    assert btc["comparisons"]["positions"]["role"] == "descriptive_only"
    assert result["validated_audit"] == audit


@pytest.mark.parametrize("moving,one_date", [(287, False), (288, True)])
def test_sample_boundaries_require_288_and_two_target_end_dates(moving, one_date):
    result = report.build_report(audit_fixture(moving=moving, one_date=one_date))
    assert result["criterion_status"] == "insufficient_evaluation"


def test_flat_labels_do_not_satisfy_moving_count_or_add_moving_dates():
    audit = audit_fixture(moving=287, flats=2, one_date=True)
    audit["as_of"] = CUTOFF + 3 * 86400
    for case in audit["scored_cases"]:
        if case["label"] == 1:
            case["anchor"] += 86400
            case["target_end"] += 86400
            case["source_received_at"] += 86400
    result = report.build_report(audit)
    assert result["criterion_status"] == "insufficient_evaluation"
    for asset in result["assets"].values():
        assert asset["n"] == 289 and asset["moving_n"] == 287 and asset["flat_n"] == 2
        assert len(asset["moving_target_end_utc_dates"]) == 1
        assert all(model["n"] == 289 for model in asset["models"].values())


@pytest.mark.parametrize(
    "key,value", [("direction_correct", 189), ("brier", 0.60), ("log_loss", 0.90)]
)
def test_one_asset_losing_any_joint_metric_fails_all_asset_criterion(key, value):
    audit = audit_fixture()
    candidate = audit["metrics"]["SOLUSDT"]["depth"]
    candidate[key] = value
    candidate["direction_accuracy"] = candidate["direction_correct"] / 288
    result = report.build_report(audit)
    assert result["criterion_status"] == "criteria_not_met"
    assert result["assets"]["BTCUSDT"]["comparisons"]["baseline"]["jointly_improved"]


def test_beating_baseline_but_losing_price_control_does_not_pass():
    audit = audit_fixture()
    audit["metrics"]["ETHUSDT"]["price_control"]["brier"] = 0.49
    result = report.build_report(audit)
    asset = result["assets"]["ETHUSDT"]
    assert asset["comparisons"]["baseline"]["jointly_improved"]
    assert asset["comparisons"]["price_control"]["jointly_improved"] is False
    assert result["criterion_status"] == "criteria_not_met"


@pytest.mark.parametrize("key", ["direction_correct", "brier", "log_loss"])
def test_ties_in_each_registered_metric_are_not_strict_improvement(key):
    audit = audit_fixture()
    metrics = audit["metrics"]["BTCUSDT"]
    metrics["depth"][key] = metrics["baseline"][key]
    metrics["depth"]["direction_accuracy"] = metrics["depth"]["direction_correct"] / 288
    assert report.build_report(audit)["criterion_status"] == "criteria_not_met"


@pytest.mark.parametrize("value", [None, math.nan, math.inf])
def test_missing_or_nonfinite_metric_is_insufficient_and_json_remains_strict(value):
    audit = audit_fixture()
    audit["metrics"]["BTCUSDT"]["depth"]["log_loss"] = value
    result = report.build_report(audit)
    assert result["criterion_status"] == "insufficient_evaluation"
    json.dumps(result, allow_nan=False)
    if value is not None:
        assert result["nonfinite_audit_values_represented_as_null"]


def test_zero_samples_missing_primary_models_and_flat_only_are_insufficient():
    for audit in (audit_fixture(moving=0), audit_fixture(moving=0, flats=300)):
        assert (
            report.build_report(audit)["criterion_status"] == "insufficient_evaluation"
        )
    audit = audit_fixture()
    del audit["metrics"]["BTCUSDT"]["price_control"]
    assert report.build_report(audit)["criterion_status"] == "insufficient_evaluation"


@pytest.mark.parametrize("missing", [True, False])
def test_positions_missing_or_nonfinite_is_descriptive_and_never_changes_joint_gate(
    missing,
):
    audit = audit_fixture()
    if missing:
        del audit["metrics"]["BTCUSDT"]["positions"]
    else:
        audit["metrics"]["BTCUSDT"]["positions"]["brier"] = math.nan
    result = report.build_report(audit)
    assert result["criterion_status"] == "preliminary_criteria_met"
    assert result["descriptive_warnings"]
    asset = result["assets"]["BTCUSDT"]
    assert asset["metrics_available"]
    assert asset["metrics_available_by_arm"]["positions"] is False
    assert asset["comparisons"]["positions"]["delta_positive_is_better"] is None
    assert asset["comparisons"]["baseline"]["jointly_improved"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("key", ["n", "moving_n", "direction_accuracy"])
def test_inconsistent_denominators_or_accuracy_are_rejected(key):
    audit = audit_fixture()
    audit["metrics"]["ETHUSDT"]["baseline"][key] -= 1
    with pytest.raises(ValueError, match="denominator|numerator"):
        report.build_report(audit)


def test_duplicate_cases_or_unverified_audit_are_rejected():
    audit = audit_fixture()
    audit["scored_cases"].append(deepcopy(audit["scored_cases"][0]))
    audit["scored_outcomes"] += 1
    with pytest.raises(ValueError, match="duplicate"):
        report.build_report(audit)
    audit = audit_fixture()
    audit["verified"] = False
    with pytest.raises(ValueError, match="successful independent"):
        report.build_report(audit)


def test_time_window_elapsed_does_not_claim_process_completion():
    audit = audit_fixture()
    audit["as_of"] = END + 119
    assert report.build_report(audit)["evaluation_stage"] == "interim"
    audit["as_of"] = END + 120
    result = report.build_report(audit)
    assert result["evaluation_stage"] == "scheduled_window_elapsed"
    assert result["completion_verified"] is False
    assert any(
        "arresto effettivo resta da verificare" in text
        for text in result["interpretation"]
    )


def test_warmup_draft_uses_only_fresh_audit_and_preserves_all_coverage(
    tmp_path, monkeypatch, capsys
):
    audit = audit_fixture(moving=0)
    audit.update(as_of=CUTOFF - 100, phase="warmup", model_fitted=False)
    calls = []

    def fresh_verify(root):
        calls.append(root)
        return audit

    monkeypatch.setattr(report.verifier, "verify", fresh_verify)
    monkeypatch.setattr(
        Path, "read_text", lambda *args, **kwargs: pytest.fail("read after audit")
    )
    assert report.main(["--root", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert calls == [tmp_path.resolve()]
    assert result["criterion_status"] == "insufficient_evaluation"
    assert result["validated_audit"] == audit
    assert result["validated_audit"]["fallback_missing_book"] == 12
    assert not list(tmp_path.iterdir())


def test_output_is_new_json_in_dedicated_reports_only_and_never_overwrites(tmp_path):
    root = tmp_path / "depth-information-run"
    root.mkdir()
    result = report.build_report(audit_fixture())
    output = root / "reports" / "interim-001.json"
    report.write_report(root, output, result)
    assert json.loads(output.read_text()) == result
    before = output.read_bytes()
    with pytest.raises(ValueError, match="new run/reports"):
        report.write_report(root, output, result)
    assert output.read_bytes() == before
    for directory in (
        "forecasts",
        "acks",
        "eligibility",
        "missing",
        "outcomes",
        "runtime",
    ):
        with pytest.raises(ValueError, match="new run/reports"):
            report.write_report(root, root / directory / "new.json", result)
    for name in ("model.json", "config.json", ".private.json", "output.txt"):
        with pytest.raises(ValueError, match="new run/reports"):
            report.write_report(root, root / "reports" / name, result)


def test_symlink_source_ancestor_and_protected_output_guards(tmp_path):
    root, outside = tmp_path / "depth-information-run", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "reports").symlink_to(outside, target_is_directory=True)
    result = report.build_report(audit_fixture())
    with pytest.raises(ValueError, match="real dedicated"):
        report.write_report(root, root / "reports" / "new.json", result)
    (root / "reports").unlink()
    result["validated_audit"]["source_snapshots"]["depth"]["root"] = str(
        root / "reports" / "depth-research-source"
    )
    with pytest.raises(ValueError, match="overlaps"):
        report.write_report(root, root / "reports" / "new.json", result)
    for protected in (
        "archive",
        "holdout",
        "shadow-pilot-20261029-v2",
        "depth-research-source",
    ):
        target = tmp_path / protected
        target.mkdir()
        with pytest.raises(ValueError, match="protected or source"):
            report.write_report(target, target / "reports" / "new.json", result)


def test_cli_verification_failure_returns_json_without_report(
    tmp_path, monkeypatch, capsys
):
    def fail(root):
        raise ValueError("independent verification failed")

    monkeypatch.setattr(report.verifier, "verify", fail)
    assert report.main(["--root", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert not captured.out and json.loads(captured.err)["report_generated"] is False
