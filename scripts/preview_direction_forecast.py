#!/usr/bin/env python3
"""Emit a reproducible historical five-minute direction forecast or abstention."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_microdirection as micro  # noqa: E402
from research import timestral_reversal as reversal  # noqa: E402
from scripts.run_timestral_microdirection import ROOT, digest, write  # noqa: E402


def forecast(root, symbol, anchor):
    when = datetime.fromisoformat(anchor.replace("Z", "+00:00"))
    if when.tzinfo is None:
        raise ValueError("explicit UTC offset required")
    when = when.astimezone(timezone.utc)
    stamp = int(when.timestamp())
    if (
        when.second
        or when.microsecond
        or when.minute % 5
        or not 1767225600 <= stamp
        or stamp + 360 > micro.BOUNDARY
    ):
        raise ValueError(
            "aligned historical anchor and target before protected boundary required"
        )
    if symbol == "BTCUSDT":
        source = root / "micro/minutes.npz"
        folder = root / "reversal"
        fitkey = f"h5_{when.month}"
    elif symbol in ("ETHUSDT", "SOLUSDT"):
        source = root / "transfer" / f"{symbol}-minutes.npz"
        folder = root / "transfer"
        fitkey = f"{symbol}_{when.month}"
    else:
        raise ValueError("unregistered symbol")
    manifest = json.loads((folder / "manifest.json").read_text())
    for name in ("fits.json", "protocol.json"):
        if digest(folder / name) != manifest[name]:
            raise ValueError("changed frozen artifact")
    p = json.loads((folder / "protocol.json").read_text())
    for name, sha in p["code"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("changed frozen code")
    expected = (
        p["source_sha256"]
        if symbol == "BTCUSDT"
        else json.loads((folder / "sources.json").read_text())["derived"][symbol][
            "sha256"
        ]
    )
    if (
        symbol != "BTCUSDT"
        and digest(folder / "sources.json") != manifest["sources.json"]
    ):
        raise ValueError("changed source receipt")
    if digest(source) != expected:
        raise ValueError("changed source")
    data = dict(np.load(source))
    # The inference path uses no outcome file and excludes every unclosed minute.
    mask = (data["time"] < stamp) & (data["time"] >= stamp - 62 * 60)
    context = {k: v[mask] for k, v in data.items()}
    if (
        len(context["time"]) != 62
        or context["time"][-1] + 60 != stamp
        or np.any(np.diff(context["time"]) != 60)
    ):
        raise ValueError("complete 62-minute context required")
    fit = json.loads((folder / "fits.json").read_text())[fitkey]
    month_start = int(when.replace(day=1, hour=0, minute=0).timestamp())
    if fit["cutoff"] != month_start:
        raise ValueError("wrong monthly fit")
    begin = time.perf_counter()
    cells, signs = reversal.states(context, np.array([61]))
    pred, selected = reversal.predict(fit, cells, signs, np.array([stamp]))
    probabilities = pred[0, 2]
    return {
        "symbol": symbol,
        "anchor": when.isoformat(),
        "context_end_exclusive": when.isoformat(),
        "return_start": (when + timedelta(minutes=1)).isoformat(),
        "return_end": (when + timedelta(minutes=6)).isoformat(),
        "fit_cutoff": datetime.fromtimestamp(fit["cutoff"], timezone.utc).isoformat(),
        "probabilities": dict(zip(("down", "flat", "up"), map(float, probabilities))),
        "signal": ("down", "flat", "up")[probabilities.argmax()]
        if selected[0]
        else "abstain",
        "inference_seconds": time.perf_counter() - begin,
        "status": "Historical research preview; no prospective or executable-price validation.",
        "source_sha256": expected,
        "fit_sha256": digest(folder / "fits.json"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "reports/timestral-microdirection-20261027"
    )
    parser.add_argument(
        "--symbol", choices=("BTCUSDT", "ETHUSDT", "SOLUSDT"), default="ETHUSDT"
    )
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = forecast(args.root, args.symbol, args.anchor)
    if args.output:
        write(args.output, result)
    print(json.dumps(result, sort_keys=True))
