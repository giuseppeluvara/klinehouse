#!/usr/bin/env python3
"""Register and run bounded root-cause probes on already inspected history."""

from datetime import datetime, timedelta, timezone
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_diagnostics as diagnosis  # noqa: E402
from research import timestral_experiment as shared  # noqa: E402
from research.timestral_long_horizon import Hour  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/timestral-diagnosis-20261026"
INPUTS = ("hours.jsonl", "run/evaluation.jsonl")
CODE = ("research/timestral_diagnostics.py", "scripts/diagnose_timestral.py",
        "research/timestral_experiment.py", "research/timestral_long_horizon.py")


def hashes():
    return {name: shared.digest_file(ROOT / name) for name in CODE}


def register(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "registered_at": shared.stamp(datetime.now(timezone.utc)), "code_sha256": hashes(),
        "source": str(SOURCE), "input_sha256": {name: shared.digest_file(SOURCE / name) for name in INPUTS},
        "status": "diagnostic on previously observed history; no independent validation or model selection",
        "probes": {
            "scoring": "Separate 5/95 percent tail loss and interval score from 50 percent median loss; also match static median to the existing zero point. No original forecasts modified.",
            "affine": "All 1484 Jan-Jul anchors, 256 identical closes: center on last log close in float64 then multiply by10000 before float32 inference. Compare with saved uncentered forecasts. Absolute point changes above0.1bps are flagged, not used to select a model.",
            "positive": "Constant, positive/negative linear trend (0.0002/hour), four phase64-hour sinusoids amplitude0.01, each at4/24h; judge constant within0.1bps and trend/sine aggregate MAE against last price, with all individual outcomes retained.",
            "negative": "128 independent Gaussian random walks sigma0.002/hour seed20261026; zero is the known population conditional mean/median. A finite sample can favor either model. 256past+24future points, horizons4/24.",
            "overlap": "Describe lag1 versus lag-h autocorrelation of rolling h-hour returns, and associations of saved forecasts with past returns and future outcomes. Overlap autocorrelation is not forecast skill.",
        },
        "model": {"id": shared.MODEL_ID, "code_revision": shared.CODE_REVISION, "weight_revision": shared.WEIGHT_REVISION},
        "model_config": {"lookback": 256, "torch_threads": 2, "local_files_only": True,
                         "model_cache": str(ROOT / "research/out/timestral-pilot-20261025/model-cache")},
        "preservation": "No new source data, downloads, old-file edits, or protected dates opened. No tuning after results.",
    }
    shared.exclusive_json(output / "protocol.json", protocol)
    shared.exclusive_json(output / "protocol.sha256.json", {"sha256": shared.digest_file(output / "protocol.json")})
    return protocol


def verify(output):
    output = Path(output)
    p = json.loads((output / "protocol.json").read_text())
    if (shared.digest_file(output / "protocol.json") != json.loads((output / "protocol.sha256.json").read_text())["sha256"]
            or hashes() != p["code_sha256"] or str(SOURCE) != p["source"]
            or any(shared.digest_file(SOURCE / name) != value for name, value in p["input_sha256"].items())):
        raise ValueError("registered code or inputs changed")
    return p


def json_lines(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle]


def control_summary(records, horizon, prefix):
    rows = [r for r in records if r["horizon"] == horizon and r["case"].startswith(prefix)]
    y, p = np.asarray([r["actual"] for r in rows]), np.asarray([r["prediction"]["point"] for r in rows])
    baseline, mae = float(np.abs(y).mean()), float(np.abs(p - y).mean())
    return {"n": len(rows), "mae": mae, "unchanged_mae": baseline,
            "relative_mae_change": mae / baseline - 1 if baseline else None,
            "direction_correct": float((np.sign(p) == np.sign(y)).mean()),
            "maximum_absolute_error": float(np.abs(p - y).max())}


def run(output):
    output = Path(output)
    protocol = verify(output)
    shared.exclusive_json(output / "opened.json", {"at": shared.stamp(datetime.now(timezone.utc)),
                                                  "protocol_sha256": shared.digest_file(output / "protocol.json")})
    started = time.perf_counter()
    rows, hours = json_lines(SOURCE / "run/evaluation.jsonl"), json_lines(SOURCE / "hours.jsonl")
    closes = np.asarray([r["log_close"] for r in hours])
    lookup = {r["time"]: i for i, r in enumerate(hours)}
    if len(rows) != 1484 or len(hours) != 8016 or hours[-1]["time"] != "2026-07-31T23:00:00Z":
        raise ValueError("unexpected diagnostic sample")
    model = shared.Timestral(protocol["model_config"])
    probes, controls = [], []
    with (output / "affine-probes.jsonl").open("x") as handle:
        for i, row in enumerate(rows):
            idx, h = lookup[row["anchor"]], row["horizon_hours"]
            context = closes[idx - 256:idx]
            # No future values are passed to the model.
            transformed = diagnosis.centered_forecast(model, context, h)
            target = closes[idx + h - 1] - closes[idx - 1]
            if abs(target - row["actual_log_return"]) > 1e-14:
                raise ValueError("saved target does not match cache endpoint")
            probe = {"sample_id": row["sample_id"], "horizon": h, "anchor": row["anchor"],
                     "saved": row["predictions"]["timestral_price"], "transformed": transformed,
                     "actual": float(target), "past_h_return": float(closes[idx - 1] - closes[idx - h - 1]),
                     "input_cast_max_error": float(np.abs(context.astype(np.float32).astype(float) - context).max())}
            shared.append_record(handle, probe)
            probes.append(probe)
            if (i + 1) % 400 == 0:
                print(f"Affine probes: {i + 1}/{len(rows)}", file=sys.stderr, flush=True)
    origin = datetime(2000, 1, 1, tzinfo=timezone.utc)
    with (output / "controls.jsonl").open("x") as handle:
        for name, values in diagnosis.synthetic_cases():
            context = [Hour(origin + timedelta(hours=i), float(v), (0., 0., 0.)) for i, v in enumerate(values[:256])]
            for h in (4, 24):
                prediction = model.predict(context, {"horizon": h}, False)
                record = {"case": name, "horizon": h, "prediction": prediction,
                          "actual": float(values[255 + h] - values[255]), "synthetic": True}
                shared.append_record(handle, record)
                controls.append(record)
    results = {}
    for h in (4, 24):
        selected = [r for r in rows if r["horizon_hours"] == h]
        group = [r for r in probes if r["horizon"] == h]
        p, transformed, y, past = (np.asarray([r[k]["point"] if k in ("saved", "transformed") else r[k] for r in group])
                                   for k in ("saved", "transformed", "actual", "past_h_return"))
        difference = np.abs(p - transformed)
        overlap = diagnosis.overlap_statistics(closes, h)
        assoc = {}
        for name in ("timestral_price", "timestral_returns"):
            forecast = np.asarray([r["predictions"][name]["point"] for r in selected])
            assoc[name] = {"prediction_past_return_correlation": diagnosis.correlation(forecast, past),
                           "prediction_future_return_correlation": diagnosis.correlation(forecast, y),
                           "prediction_std": float(forecast.std())}
        results[str(h)] = {
            "intervals": {"jan_mar": diagnosis.compare_intervals([r for r in selected if r["anchor"] < "2026-04-01"]),
                          "jan_jul": diagnosis.compare_intervals(selected)},
            "affine": {"samples": len(group), "max_point_difference_bps": float(difference.max() * 10000),
                       "median_point_difference_bps": float(np.median(difference) * 10000),
                       "flagged_above_0_1bps": int((difference > .00001).sum()),
                       "max_input_rounding_bps": max(r["input_cast_max_error"] for r in group) * 10000,
                       "saved_mae": float(np.abs(p - y).mean()), "transformed_mae": float(np.abs(transformed - y).mean()),
                       "unchanged_mae": float(np.abs(y).mean())},
            "controls": {name: control_summary(controls, h, name) for name in ("constant", "linear", "sine", "random_walk")},
            "overlap": overlap, "associations": assoc,
            "past_future_return_correlation": diagnosis.correlation(past, y),
        }
    result = {"status": protocol["status"], "horizons": results,
              "wall_seconds": time.perf_counter() - started, "peak_rss_bytes": shared.peak_rss_bytes(),
              "environment": shared.environment(), "checkpoint_files_sha256": model.checkpoint_files,
              "protocol_sha256": shared.digest_file(output / "protocol.json"),
              "output_sha256": {n: shared.digest_file(output / n) for n in ("affine-probes.jsonl", "controls.jsonl")}}
    shared.exclusive_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = register(args.output) if args.phase == "register" else run(args.output)
    print(json.dumps({k: result[k] for k in ("status", "wall_seconds", "peak_rss_bytes") if k in result}))


if __name__ == "__main__":
    main()
