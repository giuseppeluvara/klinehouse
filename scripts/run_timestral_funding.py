#!/usr/bin/env python3
"""Test one nonlinear learner with controlled additions of information."""

from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_experiment as shared  # noqa: E402
from research import timestral_median as median  # noqa: E402
from research import timestral_funding as funding  # noqa: E402
from research import timestral_capacity as capacity  # noqa: E402
from research.timestral_long_horizon import Hour  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/timestral-diagnosis-20261026"
INPUTS = ("hours.jsonl", "run/predictions.jsonl", "run/outcomes.jsonl")
CODE = (
    "research/timestral_funding.py",
    "scripts/run_timestral_funding.py",
    "scripts/prepare_timestral_funding.py",
    "research/timestral_median.py",
    "research/timestral_capacity.py",
    "research/timestral_experiment.py",
    "research/timestral_long_horizon.py",
)


def hashes():
    return {n: shared.digest_file(ROOT / n) for n in CODE}


def register(output, funding_root):
    output = Path(output)
    funding_root = Path(funding_root).resolve()
    provenance = json.loads((funding_root / "provenance.json").read_text())
    if (
        shared.digest_file(funding_root / "funding.json")
        != provenance["funding_sha256"]
    ):
        raise ValueError("funding checksum mismatch")
    output.mkdir(parents=True, exist_ok=False)
    p = {
        "registered_at": shared.stamp(datetime.now(timezone.utc)),
        "code_sha256": hashes(),
        "inputs": {n: shared.digest_file(SOURCE / n) for n in INPUTS},
        "funding_root": str(funding_root),
        "funding_inputs": {
            n: shared.digest_file(funding_root / n)
            for n in ("funding.json", "provenance.json")
        },
        "status": "historical exploratory information ablation; source timestamps do not establish publication latency or prospective validity",
        "hypothesis": "A linear correction may miss conditional patterns; settled futures funding may add information absent from past spot OHLCV. Compare identical nonlinear learner on state, state+Timestral, state+Timestral+funding. Do not select by inspecting results.",
        "features": list(median.FEATURES)
        + [
            "last_settled_funding_bps_per_hour",
            "last3_settlements_weighted_bps_per_hour",
        ],
        "funding_availability": "Final settlement timestamp plus1hour, never the next funding settlement. This is a conservative assumption, not an audited publication timestamp. Reject stale data.",
        "parameters": funding.PARAMETERS,
        "candidates": list(funding.CONDITIONS),
        "fit": "Monthly refit on trailing180days of matured labels, minimum60daily/360four-hour; initial history Oct-Dec2025; half-life60days; normalize target by causal volatility and weight loss by origin volatility to preserve return-MAE objective.",
        "evaluation": "All1484 Jan-Jul2026 anchors. Same conservative screen: >=1%MAE gain, no worse RMSE,5/7monthly wins, upper weekly bootstrap99.375%below0 with4000draws seed20261026. No correction over all sequential research, no deployment or independent-validation claim.",
        "sklearn_version": importlib.metadata.version("scikit-learn"),
        "numpy_version": np.__version__,
        "protection": "No source or target on or after2026-08-01. BTCUSDT spot remains target; no derivative orders or PnL.",
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
    root = Path(p["funding_root"])
    if (
        shared.digest_file(output / "protocol.json")
        != json.loads((output / "protocol.sha256.json").read_text())["sha256"]
        or p["code_sha256"] != hashes()
        or p["sklearn_version"] != importlib.metadata.version("scikit-learn")
        or p["numpy_version"] != np.__version__
        or any(shared.digest_file(SOURCE / n) != v for n, v in p["inputs"].items())
        or any(
            shared.digest_file(root / n) != v for n, v in p["funding_inputs"].items()
        )
    ):
        raise ValueError("registered inputs, code or environment changed")
    return p


def run(output):
    output = Path(output)
    p = verify(output)
    shared.exclusive_json(
        output / "opened.json", {"at": shared.stamp(datetime.now(timezone.utc))}
    )
    (output / "models").mkdir()
    started = time.perf_counter()

    def records(name):
        return [json.loads(line) for line in (SOURCE / name).read_text().splitlines()]

    hours = [
        Hour(shared.utc(r["time"]), r["log_close"], tuple(r["features"]))
        for r in records("hours.jsonl")
    ]
    lookup = {shared.stamp(r.time): i for i, r in enumerate(hours)}
    old = records("run/predictions.jsonl")
    outcomes = {r["sample_id"]: r for r in records("run/outcomes.jsonl")}
    settlements = json.loads((Path(p["funding_root"]) / "funding.json").read_text())
    if (
        len(old) != 2128
        or len(hours) != 8016
        or any(shared.utc(r["time"]) >= shared.utc("2026-08-01") for r in settlements)
    ):
        raise ValueError("unexpected or protected source")
    history = []
    evaluation = []
    models = {}
    fits = {}
    with (
        (output / "predictions.jsonl").open("x") as pf,
        (output / "outcomes.jsonl").open("x") as of,
    ):
        for row in old:
            anchor, h = shared.utc(row["anchor"]), row["horizon_hours"]
            idx = lookup[row["anchor"]]
            extra, latest = funding.funding_features(settlements, anchor)
            features = (
                median.feature_vector(
                    hours[idx - 256 : idx], anchor, row["scale"], row["predictions"]
                ).tolist()
                + extra
            )
            current = {
                k: row[k]
                for k in ("sample_id", "anchor", "label_end", "horizon_hours", "scale")
            }
            current.update(features=features, latest_funding_settlement=latest)
            if anchor >= shared.utc("2026-01-01"):
                forecasts = {}
                for name in funding.CONDITIONS:
                    key = f"{h}-{anchor:%Y-%m}-{name}"
                    if key not in models:
                        past = [
                            r
                            for r in history
                            if r["horizon_hours"] == h
                            and shared.utc(r["label_end"]) <= anchor
                            and shared.utc(r["anchor"]) >= anchor - timedelta(days=180)
                        ]
                        models[key] = funding.fit(past, anchor, h, name)
                        file = output / "models" / f"{key}.pkl"
                        with file.open("xb") as handle:
                            pickle.dump(models[key], handle, protocol=5)
                        fits[key] = {
                            "samples": len(past),
                            "cutoff": shared.stamp(anchor),
                            "latest_label_end": max(r["label_end"] for r in past),
                            "sample_ids_sha256": hashlib.sha256(
                                shared.canonical([r["sample_id"] for r in past])
                            ).hexdigest(),
                            "model_sha256": shared.digest_file(file),
                        }
                        print(f"Fit {key}: {len(past)}", file=sys.stderr, flush=True)
                    forecasts[name] = {
                        "point": funding.predict(models[key], features, row["scale"])
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
        for name in ("unchanged", *funding.CONDITIONS):
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
                "prediction_std": float(v.std()),
            }
        result[str(h)] = {
            "metrics": scores,
            "screens": {n: capacity.screen(group, n) for n in funding.CONDITIONS},
        }
    shared.exclusive_json(output / "fits.json", fits)
    result = {
        "status": p["status"],
        "horizons": result,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": shared.peak_rss_bytes(),
        "environment": shared.environment(),
        "sklearn_version": p["sklearn_version"],
        "protocol_sha256": shared.digest_file(output / "protocol.json"),
        "output_sha256": {
            n: shared.digest_file(output / n)
            for n in ("predictions.jsonl", "outcomes.jsonl", "fits.json")
        },
    }
    shared.exclusive_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--funding-root",
        type=Path,
        default="research/out/timestral-capacity-20261026/funding-source",
    )
    args = parser.parse_args()
    r = (
        register(args.output, args.funding_root)
        if args.phase == "register"
        else run(args.output)
    )
    print(
        json.dumps(
            {k: r[k] for k in ("status", "wall_seconds", "peak_rss_bytes") if k in r}
        )
    )
