#!/usr/bin/env python3
"""A fixed loss-aligned online calibration; state-only is the paired ablation."""

from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_experiment as shared  # noqa: E402
from research import timestral_median as median  # noqa: E402
from research import timestral_capacity as capacity  # noqa: E402
from research.timestral_long_horizon import Hour  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/timestral-diagnosis-20261026"
INPUTS = ("hours.jsonl", "run/predictions.jsonl", "run/outcomes.jsonl")
CODE = (
    "research/timestral_median.py",
    "scripts/run_timestral_median.py",
    "research/timestral_capacity.py",
    "research/timestral_experiment.py",
    "research/timestral_long_horizon.py",
)


def hashes():
    return {n: shared.digest_file(ROOT / n) for n in CODE}


def register(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    p = {
        "registered_at": shared.stamp(datetime.now(timezone.utc)),
        "code_sha256": hashes(),
        "inputs": {n: shared.digest_file(SOURCE / n) for n in INPUTS},
        "source": str(SOURCE),
        "status": "diagnostic historical replay; no fresh test or claimed functioning predictive solution",
        "hypothesis": "Prior ridge adaptation minimized squared error although the primary score is absolute error. Fit a sparse conditional median to the original causal state and frozen Timestral forecasts; compare state-only on identical samples.",
        "fit": "Refit first anchor of each UTC month, only labels already mature; trailing180days, initial history October-December2025, minimum60daily/360four-hour samples; exponential time weight half-life60days; penalty0.025 on all standardized coefficients including intercept; z clip8. All choices fixed.",
        "loss": "Exact linear-program solution: log-return weighted absolute error, expressed in units of past volatility, plus L1 coefficient penalty. Volatility and features are known at each origin; no future normalization.",
        "features": median.FEATURES,
        "candidates": ["state_only", "state_timestral"],
        "screen": "Same conservative screen as capacity experiment: >=1%MAE gain,no worse RMSE,5/7monthly wins,upper weekly bootstrap99.375%below0;4000draws seed20261026. No familywise guarantee across sequential research; independent validation still required.",
        "scipy_version": importlib.metadata.version("scipy"),
        "protected_dates": "No source at or after2026-08-01 opened",
    }
    shared.exclusive_json(output / "protocol.json", p)
    shared.exclusive_json(
        output / "protocol.sha256.json",
        {"sha256": shared.digest_file(output / "protocol.json")},
    )
    return p


def verify(output):
    output = Path(output)
    p = json.loads((output / "protocol.json").read_text())
    if (
        p["code_sha256"] != hashes()
        or shared.digest_file(output / "protocol.json")
        != json.loads((output / "protocol.sha256.json").read_text())["sha256"]
        or any(shared.digest_file(SOURCE / n) != v for n, v in p["inputs"].items())
        or importlib.metadata.version("scipy") != p["scipy_version"]
    ):
        raise ValueError("registered code, inputs or optimizer changed")
    return p


def run(output):
    output = Path(output)
    p = verify(output)
    shared.exclusive_json(
        output / "opened.json", {"at": shared.stamp(datetime.now(timezone.utc))}
    )
    started = time.perf_counter()

    def records(name):
        return [json.loads(line) for line in (SOURCE / name).read_text().splitlines()]

    hours = [
        Hour(shared.utc(r["time"]), r["log_close"], tuple(r["features"]))
        for r in records("hours.jsonl")
    ]
    lookup = {shared.stamp(r.time): i for i, r in enumerate(hours)}
    predictions = records("run/predictions.jsonl")
    outcomes = {r["sample_id"]: r for r in records("run/outcomes.jsonl")}
    if (
        len(predictions) != 2128
        or len(hours) != 8016
        or hours[-1].time >= shared.utc("2026-08-01")
    ):
        raise ValueError("unexpected or protected source")
    history = []
    evaluation = []
    parameters = {}
    with (
        (output / "predictions.jsonl").open("x") as pf,
        (output / "outcomes.jsonl").open("x") as of,
    ):
        for row in predictions:
            anchor, h = shared.utc(row["anchor"]), row["horizon_hours"]
            idx = lookup[row["anchor"]]
            features = median.feature_vector(
                hours[idx - 256 : idx], anchor, row["scale"], row["predictions"]
            ).tolist()
            current = {
                k: row[k]
                for k in ("sample_id", "anchor", "label_end", "horizon_hours", "scale")
            }
            current["features"] = features
            if anchor >= shared.utc("2026-01-01"):
                forecasts = {}
                for name, enabled in [("state_only", False), ("state_timestral", True)]:
                    key = f"{h}:{anchor:%Y-%m}:{name}"
                    if key not in parameters:
                        past = [
                            r
                            for r in history
                            if r["horizon_hours"] == h
                            and shared.utc(r["label_end"]) <= anchor
                            and shared.utc(r["anchor"]) >= anchor - timedelta(days=180)
                        ]
                        parameters[key] = median.fit(past, anchor, h, enabled)
                        parameters[key]["sample_ids_sha256"] = hashlib.sha256(
                            shared.canonical([r["sample_id"] for r in past])
                        ).hexdigest()
                        print(
                            f"Fitted {key}: {len(past)} past labels",
                            file=sys.stderr,
                            flush=True,
                        )
                    forecasts[name] = {
                        "point": median.predict(parameters[key], features, row["scale"])
                    }
                current["predictions"] = forecasts
                shared.append_record(pf, current)
            actual = hours[idx + h - 1].log_close - hours[idx - 1].log_close
            if abs(actual - outcomes[row["sample_id"]]["actual_log_return"]) > 1e-14:
                raise ValueError("target mismatch")
            current["actual"] = actual
            history.append(current)
            if anchor >= shared.utc("2026-01-01"):
                shared.append_record(
                    of, {"sample_id": row["sample_id"], "actual": actual}
                )
                evaluation.append({**current, "actual_log_return": actual})
    result = {}
    for h in (4, 24):
        group = [r for r in evaluation if r["horizon_hours"] == h]
        scores = {}
        for name in ("unchanged", "state_only", "state_timestral"):
            y = np.asarray([r["actual"] for r in group])
            v = (
                np.zeros(len(group))
                if name == "unchanged"
                else np.asarray([r["predictions"][name]["point"] for r in group])
            )
            scores[name] = {
                "samples": len(group),
                "mae": float(abs(v - y).mean()),
                "rmse": float(np.sqrt(np.mean((v - y) ** 2))),
                "nonzero_predictions": int((v != 0).sum()),
                "prediction_std": float(v.std()),
            }
        result[str(h)] = {
            "metrics": scores,
            "screens": {
                n: capacity.screen(group, n) for n in ("state_only", "state_timestral")
            },
        }
    shared.exclusive_json(output / "parameters.json", parameters)
    result = {
        "status": p["status"],
        "horizons": result,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": shared.peak_rss_bytes(),
        "environment": shared.environment(),
        "scipy_version": p["scipy_version"],
        "protocol_sha256": shared.digest_file(output / "protocol.json"),
        "output_sha256": {
            n: shared.digest_file(output / n)
            for n in ("predictions.jsonl", "outcomes.jsonl", "parameters.json")
        },
    }
    shared.exclusive_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = register(args.output) if args.phase == "register" else run(args.output)
    print(
        json.dumps(
            {k: r[k] for k in ("status", "wall_seconds", "peak_rss_bytes") if k in r}
        )
    )


if __name__ == "__main__":
    main()
