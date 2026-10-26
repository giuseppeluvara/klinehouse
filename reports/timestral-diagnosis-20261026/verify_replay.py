"""Audit saved replay arithmetic without importing the replay or policy modules."""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def verify(root):
    root = Path(root)
    run = root / "run"
    report = read(run / "analysis-recovered.json")
    for name, digest in report["output_sha256"].items():
        assert sha(run / name) == digest
    cache = read(root / "cache-provenance.json")
    assert sha(root / "hours.jsonl") == cache["hours_sha256"]
    hours = {utc(row["time"]): row for row in lines(root / "hours.jsonl")}
    predictions, outcomes = lines(run / "predictions.jsonl"), lines(run / "outcomes.jsonl")
    assert len(predictions) == len(outcomes) == 2128
    history = []
    for row, outcome in zip(predictions, outcomes):
        assert row["sample_id"] == outcome["sample_id"]
        h, anchor = row["horizon_hours"], utc(row["anchor"])
        context = [hours[anchor - timedelta(hours=256 - i)] for i in range(256)]
        assert sha_context(context) == row["input_sha256"]
        actual = hours[anchor + timedelta(hours=h - 1)]["log_close"] - context[-1]["log_close"]
        assert actual == outcome["actual_log_return"]
        scale = max(math.sqrt(math.fsum(x["features"][0] ** 2 for x in context[-24:])) * math.sqrt(h / 24), 1e-8)
        assert abs(scale - row["scale"]) < 1e-12
        if "selected" in row:
            past = [x for x in history if x["horizon_hours"] == h and anchor - timedelta(days=90) < utc(x["label_end"]) <= anchor]
            for model, key, metadata in [("unchanged", "unchanged_dynamic", "unchanged_calibration"),
                                         (row["selected"], "evidence_policy", "interval_calibration")]:
                point = 0. if model == "unchanged" else row["predictions"][model]["point"]
                residuals = sorted(abs(x["actual_log_return"] - (0. if model == "unchanged" else x["predictions"][model]["point"])) / x["scale"] for x in past)
                rank = math.ceil((len(past) + 1) * .9)
                radius = residuals[rank - 1] * scale
                assert row[metadata]["samples"] == len(past) and row[metadata]["rank"] == rank
                assert utc(row[metadata]["latest_label_end"]) <= anchor
                np.testing.assert_allclose(row["predictions"][key]["quantiles"], [point - radius, point, point + radius], atol=1e-12, rtol=1e-12)
                assert row["predictions"][key]["point"] == point
        history.append({**row, **outcome})
    evaluation = [x for x in history if x["anchor"] >= "2026-01-01T00:00:00Z"]
    assert evaluation == lines(run / "evaluation.jsonl") and len(evaluation) == 1484
    for h, summary in report["horizons"].items():
        group = [x for x in evaluation if x["horizon_hours"] == int(h)]
        for model, metrics in summary["metrics"].items():
            errors = [x["predictions"][model]["point"] - x["actual_log_return"] for x in group]
            expected = {"mae": math.fsum(abs(x) for x in errors) / len(group),
                        "rmse": math.sqrt(math.fsum(x * x for x in errors) / len(group)),
                        "coverage90": sum(x["predictions"][model]["quantiles"][0] <= x["actual_log_return"] <= x["predictions"][model]["quantiles"][2] for x in group) / len(group)}
            for key, value in expected.items():
                assert abs(value - metrics[key]) < 1e-12
    # Independently expand weekly bootstrap blocks to their individual observations.
    for key, decision in report["decisions"].items():
        h = int(key.split(":")[0]);start, end = utc(decision["fit_start"]), utc(decision["fit_cutoff"])
        past = [x for x in history if x["horizon_hours"] == h and start <= utc(x["anchor"]) and utc(x["label_end"]) <= end]
        assert len(past) == decision["samples"]
        count = math.ceil((end - start).days / 7)
        blocks = [np.array([i for i, x in enumerate(past) if (utc(x["anchor"]) - start).days // 7 == b], dtype=int) for b in range(count)]
        draws = np.random.default_rng(17).integers(count, size=(2000, count))
        for model, check in decision["candidate_checks"].items():
            delta = np.array([abs(x["actual_log_return"] - x["predictions"][model]["point"]) - abs(x["actual_log_return"]) for x in past])
            means = [delta[np.concatenate([blocks[b] for b in sample])].mean() for sample in draws]
            interval = np.quantile(means, [.00625, .99375])
            np.testing.assert_allclose(interval, check["interval98_75"], atol=1e-12, rtol=1e-12)
            assert check["admitted"] == all(check["checks"].values())
        admissible = [(c["mae"], ["timestral_price", "timestral_returns"].index(m), m) for m, c in decision["candidate_checks"].items() if c["admitted"]]
        assert decision["selected"] == (min(admissible)[2] if admissible else "unchanged")
    old = {}
    for path in (Path("reports/timestral-adaptation-20261025/test/predictions.jsonl"), Path("reports/timestral-long-horizon-20261025/predictions.jsonl")):
        for row in lines(path):
            old[(row["anchor"], row["horizon_hours"])] = row["predictions"]["timestral_price"]["point"]
    compared, maximum = 0, 0.
    for row in evaluation:
        key = row["anchor"], row["horizon_hours"]
        if key in old:
            compared += 1
            maximum = max(maximum, abs(row["predictions"]["timestral_price"]["point"] - old[key]))
    assert compared == 1274 and maximum < 1e-12
    result = {"passed": True, "predictions_audited": len(history), "evaluation_rows": len(evaluation),
              "monthly_bootstraps_recomputed": len(report["decisions"]), "prior_price_forecasts_compared": compared,
              "maximum_prior_price_difference": maximum, "analysis_sha256": sha(run / "analysis-recovered.json"),
              "auditor_sha256": sha(__file__), "limits": "Does not rerun Timestral or establish prospective validity. Hours are the previously verified cache; raw provenance is retained separately."}
    with (root / "replay-verification.json").open("x") as handle:
        json.dump(result, handle, indent=2);handle.write("\n")
    print(json.dumps(result))


def sha_context(context):
    return hashlib.sha256(canonical([[x["time"], x["log_close"], x["features"]] for x in context])).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    verify(parser.parse_args().root)
