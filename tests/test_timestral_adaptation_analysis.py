"""Integrity, calendar resampling and fixed historical gate regressions."""

import copy
import csv
from datetime import timedelta
import json

import numpy as np
import pytest

from scripts import analyze_timestral_adaptation as analysis


def protocol():
    return {"test_start": "2026-01-01T00:00:00Z", "test_end": "2026-04-01T00:00:00Z",
            "selected_candidates": {"4": "fixed_candidate", "24": "fixed_candidate"},
            "code_sha256": {analysis.ANALYSIS_KEY: analysis.digest(analysis.__file__)}}


def record(day=0, horizon=24, offset=0):
    anchor = analysis.utc(protocol()["test_start"]) + timedelta(days=day, hours=offset)
    outcome = .01 if day % 2 else -.02
    forecasts = {}
    for name, scale in zip(analysis.MODELS, (0, .1, .2, .5)):
        point = outcome * scale
        forecasts[name] = {"point": point, "quantiles": [point - .03, point, point + .03]}
    return {"sample_id": f"{horizon}-{day}-{offset}", "anchor": anchor.isoformat().replace("+00:00", "Z"),
            "label_end": (anchor + timedelta(hours=horizon)).isoformat().replace("+00:00", "Z"),
            "horizon_hours": horizon, "reference_price": 100.0, "actual_log_return": outcome,
            "actual_price": float(100 * np.exp(outcome)), "predictions": forecasts}


def all_records():
    return [record(day, horizon, hour) for horizon in (4, 24) for day in range(90) for hour in range(0, 24, horizon)]


def saved_run(tmp_path, rows=None):
    rows = all_records() if rows is None else rows
    (tmp_path / "protocol.json").write_text(json.dumps(protocol()))
    predictions = [{key: value for key, value in row.items() if key not in ("actual_log_return", "actual_price")} for row in rows]
    outcomes = [{key: value for key, value in row.items() if key not in ("predictions", "reference_price")} | {"status": "scored"} for row in rows]
    for name, records in (("predictions", predictions), ("outcomes", outcomes), ("evaluation", rows)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    rehash(tmp_path)


def rehash(tmp_path):
    results = {"protocol_sha256": analysis.digest(tmp_path / "protocol.json"),
               "output_sha256": {name: analysis.digest(tmp_path / name) for name in ("predictions.jsonl", "outcomes.jsonl", "evaluation.jsonl")}}
    (tmp_path / "results.json").write_text(json.dumps(results))


def test_full_calendar_gate_and_neutral_direction():
    groups = analysis.validate_records(all_records(), protocol())
    for horizon, group in groups.items():
        summary = analysis.summarize(group, horizon, protocol())
        assert summary["samples"] == {4: 540, 24: 90}[horizon]
        assert summary["excluded_samples"] == 0
        assert summary["paired_comparison"]["blocks"] == 13
        assert summary["paired_comparison"]["block_sample_counts"] == [7 * 24 // horizon] * 12 + [6 * 24 // horizon]
        assert summary["gate"]["passed"]
        assert summary["monthly_mae_wins"] == 3
        assert summary["metrics"]["unchanged"]["direction_accuracy_when_called"] is None
        assert summary["metrics"]["unchanged"]["direction_call_fraction"] == 0
        assert summary["metrics"]["adapted"]["mae_skill_vs_unchanged"] == pytest.approx(.5)


def test_calendar_bootstrap_preserves_pairs_and_weights_partial_blocks():
    rows = [record(day) for day in (0, 1, 2, 7, 15)]
    for i, row in enumerate(rows):
        row["predictions"]["adapted"]["point"] = row["actual_log_return"] * (i / 4)
    result = analysis.paired_comparison(rows, "2026-01-01T00:00:00Z", "2026-01-22T00:00:00Z")
    assert result["block_sample_counts"] == [3, 1, 1]
    delta = np.array([abs(row["predictions"]["adapted"]["point"] - row["actual_log_return"]) - abs(row["actual_log_return"]) for row in rows])
    blocks = [np.array([0, 1, 2]), np.array([3]), np.array([4])]
    choices = np.random.default_rng(17).integers(0, 3, size=(10000, 3))
    explicit = np.array([delta[np.concatenate([blocks[i] for i in choice])].mean() for choice in choices])
    primary = result["contrasts"]["adapted_minus_unchanged"]
    assert primary["mean_mae_difference"] == pytest.approx(delta.mean())
    assert primary["percentile97_5"] == pytest.approx(np.quantile(explicit, [.0125, .9875]))
    assert result == analysis.paired_comparison(rows, "2026-01-01T00:00:00Z", "2026-01-22T00:00:00Z")


def test_equality_and_insufficient_coverage_cannot_pass():
    rows = [record(day) for day in range(90)]
    for row in rows:
        row["predictions"]["adapted"] = copy.deepcopy(row["predictions"]["unchanged"])
    gate = analysis.summarize(rows, 24, protocol())["gate"]
    assert not gate["passed"]
    assert not gate["criteria"]["strict_mae_improvement"]
    assert not gate["criteria"]["adjusted_upper_ci_below_zero"]
    assert not analysis.summarize([record(day) for day in range(79)], 24, protocol())["gate"]["criteria"]["minimum_samples"]
    assert analysis.summarize([record(day) for day in range(80)], 24, protocol())["gate"]["criteria"]["minimum_samples"]


@pytest.mark.parametrize("mutation,match", [
    (lambda row: row.update(actual_log_return=float("nan")), "non-finite"),
    (lambda row: row.update(actual_price=101), "disagree"),
    (lambda row: row.update(label_end="2026-01-02T01:00:00Z"), "endpoint"),
    (lambda row: row.update(anchor="2026-01-01T01:00:00Z"), "endpoint"),
    (lambda row: row.update(anchor="2026-01-01T00:00:00"), "UTC"),
    (lambda row: row["predictions"]["adapted"].update(point=float("inf")), "non-finite"),
    (lambda row: row["predictions"]["adapted"].update(quantiles=[0, 1]), "quantiles"),
    (lambda row: row["predictions"]["unchanged"].update(point=.001), "zero"),
    (lambda row: row["predictions"].pop("adapted"), "identical"),
])
def test_invalid_evaluation_values_fail(mutation, match):
    rows = [record(horizon=24), record(horizon=4)]
    mutation(rows[0])
    with pytest.raises(ValueError, match=match):
        analysis.validate_records(rows, protocol())


def test_duplicates_and_out_of_order_anchors_fail():
    with pytest.raises(ValueError, match="duplicate"):
        analysis.validate_records([record(), record(), record(horizon=4)], protocol())
    with pytest.raises(ValueError, match="unordered"):
        analysis.validate_records([record(1), record(), record(horizon=4)], protocol())


def test_saved_output_checksum_checked_before_parsing(tmp_path):
    saved_run(tmp_path)
    assert len(analysis.verified_records(tmp_path)[2][24]) == 90
    with (tmp_path / "evaluation.jsonl").open("a") as handle:
        handle.write("invalid json\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        analysis.verified_records(tmp_path)


@pytest.mark.parametrize("change", ["code", "target", "forecast", "omission", "duplicate"])
def test_rehashed_inconsistencies_are_still_rejected(tmp_path, change):
    saved_run(tmp_path)
    if change == "code":
        value = protocol()
        value["code_sha256"][analysis.ANALYSIS_KEY] = "changed"
        (tmp_path / "protocol.json").write_text(json.dumps(value))
    else:
        rows = analysis.read_records(tmp_path / "evaluation.jsonl")
        if change == "target":
            rows[0]["actual_log_return"] = .1
        elif change == "forecast":
            rows[0]["predictions"]["adapted"]["point"] = .1
        elif change == "omission":
            rows.pop()
        else:
            rows.append(copy.deepcopy(rows[0]))
        (tmp_path / "evaluation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    rehash(tmp_path)
    with pytest.raises(ValueError):
        analysis.verified_records(tmp_path)


def test_analysis_exports_actual_evidence_and_refuses_overwrite(tmp_path):
    run, output = tmp_path / "run", tmp_path / "analysis"
    run.mkdir()
    saved_run(run)
    result = analysis.analyze(run, output)
    assert result["historical"] is True
    assert result["csv_rows"] == 2520
    assert result["all_horizons_pass"] is True
    assert b"\r" not in (output / "forecasts.csv").read_bytes()
    with (output / "forecasts.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    adapted = next(row for row in rows if row["model"] == "adapted")
    assert float(adapted["predicted_price_usdt"]) == pytest.approx(100 * np.exp(-.01))
    assert float(adapted["actual_price_usdt"]) == pytest.approx(100 * np.exp(-.02))
    assert (output / "forecast-comparison.png").read_bytes().startswith(b"\x89PNG")
    assert all(analysis.digest(output / name) == value for name, value in result["artifact_sha256"].items())
    with pytest.raises(FileExistsError):
        analysis.analyze(run, output)
