#!/usr/bin/env python3
"""Replay one historical risk forecast using only closed candles before its origin.

This does not fetch current data or place orders. The saved monthly model must
match the requested UTC month. Direction and trading usefulness are not validated.
"""

from collections import deque
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_decomposition as decomposition  # noqa: E402
from research import timestral_distribution as distribution  # noqa: E402
from research import timestral_experiment as shared  # noqa: E402


def past_context(path, anchor):
    anchor = shared.utc(anchor)
    context = deque(maxlen=721)
    previous = None
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            t = shared.utc(row["time"])
            if previous is not None and t <= previous:
                raise ValueError("unordered candle source")
            if t >= anchor:
                break
            context.append(row)
            previous = t
    return list(context)


def preview(hours, parameters, calibrations, symbol, anchor):
    anchor = shared.utc(anchor)
    if anchor.minute or anchor.second or anchor.microsecond or anchor.hour % 4:
        raise ValueError("four-hour UTC origin required")
    # Preserve this research task's protected dates, including when invoked manually.
    if anchor >= shared.utc("2026-08-01"):
        raise ValueError(
            "protected dates; prospective use requires a separate protocol"
        )
    key = f"{anchor:%Y-%m}:4" if symbol == "BTCUSDT" else f"{symbol}:{anchor:%Y-%m}"
    if key not in parameters or key not in calibrations:
        raise ValueError("missing current-month fit; do not reuse stale models")
    context = past_context(hours, anchor)
    features = decomposition.origin(context, anchor, 4)
    row = {**features, "anchor": shared.stamp(anchor)}
    forecasts = distribution.predict(calibrations[key], parameters[key], row)
    gamma = forecasts["gamma"]
    return {
        "status": "historical risk forecast; not live or direction/trading validation",
        "symbol": symbol,
        "anchor": shared.stamp(anchor),
        "horizon_hours": 4,
        "fit_key": key,
        "fit_cutoff": parameters[key]["cutoff"],
        "context_start": context[0]["time"],
        "context_end": context[-1]["time"],
        "input_sha256": hashlib.sha256(shared.canonical(context)).hexdigest(),
        "distribution": gamma,
        "root_mean_variance_percent": 100 * math.sqrt(gamma["mean"]),
        "realized_volatility_interval90_percent": [
            100 * math.sqrt(gamma["lower90"]),
            100 * math.sqrt(gamma["upper90"]),
        ],
        "interval_is_not_price_range": True,
        "direction_forecast": None,
        "probability_definition": "future sum of four squared hourly log returns exceeds past24-hour mean square scaled to4hours",
        "caution": "Gamma interval coverage was conservative in the historical evaluation; nominal probabilities are not guaranteed.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--calibrations", type=Path, required=True)
    parser.add_argument(
        "--symbol", choices=["BTCUSDT", "ETHUSDT", "SOLUSDT"], required=True
    )
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preview(
        args.hours,
        json.loads(args.parameters.read_text()),
        json.loads(args.calibrations.read_text()),
        args.symbol,
        args.anchor,
    )
    shared.exclusive_json(args.output, result)
    print(json.dumps(result, indent=2))
