#!/usr/bin/env python3
"""Forecast all three assets from one shared historical fit and closed context."""

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_universal as model  # noqa: E402
from scripts.run_timestral_microdirection import ROOT, digest, write  # noqa: E402


def point_features(contexts):
    own, lag = [], []
    if tuple(contexts) != model.SYMBOLS:
        raise ValueError("all registered assets required")
    times = contexts[model.SYMBOLS[0]]["time"]
    if len(times) != 63 or np.any(np.diff(times) != 60):
        raise ValueError("63 consecutive closed minutes required")
    for d in contexts.values():
        if (
            not np.array_equal(d["time"], times)
            or not all(np.isfinite(v).all() for v in d.values())
            or np.any(d["close"] <= 0)
        ):
            raise ValueError("aligned finite positive context required")
        log = np.log(d["close"])
        r = np.diff(log)
        scale = np.sqrt(max(np.mean(r[-61:-1] ** 2), 1e-16))
        lagscale = np.sqrt(max(np.mean(r[:60] ** 2), 1e-16))
        ret = np.clip(
            [(log[-1] - log[-1 - h]) / scale / np.sqrt(h) for h in (1, 5, 15, 60)],
            -8,
            8,
        )
        lag.append(
            np.clip(
                [(log[-2] - log[-2 - h]) / lagscale / np.sqrt(h) for h in (1, 5)], -8, 8
            )
        )
        activity = np.log(
            max(d["trades"][-1] / max(d["trades"][-61:-1].mean(), 1), 1e-6)
        )
        v = d["volume"][-1]
        v5 = d["volume"][-5:].sum()
        flow = (2 * d["taker"][-1] - v) / v if v else 0
        flow5 = (2 * d["taker"][-5:].sum() - v5) / v5 if v5 else 0
        own.append(
            np.r_[
                ret,
                flow,
                flow5,
                ret[0] * np.clip(activity, -4, 4),
                abs(ret[0]),
                np.log1p(d["trades"][-1]),
                np.mean(r[-60:] == 0),
                activity,
                np.log(scale),
                abs(flow),
            ]
        )
    features = []
    for a in range(3):
        others = [b for b in range(3) if b != a]
        other = np.mean([own[b][:4] for b in others], axis=0)
        residual = np.clip(own[a][0] - other[0], -8, 8)
        features.append(
            np.r_[
                own[a],
                other,
                np.mean([lag[b] for b in others], axis=0),
                residual * abs(residual) / 8,
                own[a][0] * np.std([own[b][0] for b in others]),
                np.mean([own[b][9] for b in others]),
                abs(other[0]),
            ]
        )
    return np.asarray(features)


def forecast(root, anchor):
    when = datetime.fromisoformat(anchor.replace("Z", "+00:00"))
    if when.tzinfo is None:
        raise ValueError("explicit timezone required")
    when = when.astimezone(timezone.utc)
    stamp = int(when.timestamp())
    if (
        when.second
        or when.microsecond
        or when.minute % 5
        or not 1767225600 <= stamp
        or stamp + 360 > model.BOUNDARY
    ):
        raise ValueError("aligned historical target before protected boundary required")
    manifest = json.loads((root / "manifest.json").read_text())
    for name in ("fits.json", "protocol.json", "results.json"):
        if digest(root / name) != manifest[name]:
            raise ValueError("changed frozen bundle")
    protocol = json.loads((root / "protocol.json").read_text())
    for name, sha in protocol["code"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("changed frozen code")
    prior = ROOT / "reports/timestral-microdirection-20261027"
    context = {}
    for a, s in enumerate(model.SYMBOLS):
        path = (
            prior / "micro/minutes.npz"
            if a == 0
            else prior / f"transfer/{s}-minutes.npz"
        )
        if digest(path) != protocol["inputs"][str(path.relative_to(ROOT))]:
            raise ValueError("changed source")
        d = dict(np.load(path))
        mask = (d["time"] >= stamp - 63 * 60) & (d["time"] < stamp)
        context[s] = {k: v[mask] for k, v in d.items()}
        if len(context[s]["time"]) != 63 or context[s]["time"][-1] + 60 != stamp:
            raise ValueError("context must end exactly at anchor")
    key = f"cross_{when.month}_all"
    fit = json.loads((root / "fits.json").read_text())[key]
    if fit["assets"] != [0, 1, 2] or fit["cutoff"] != int(
        when.replace(day=1, hour=0, minute=0).timestamp()
    ):
        raise ValueError("incorrect shared monthly fit")
    probabilities = model.predict(fit, point_features(context), np.full(3, stamp))
    result = {
        "anchor": when.isoformat(),
        "return_start": (when + timedelta(minutes=1)).isoformat(),
        "return_end": (when + timedelta(minutes=6)).isoformat(),
        "shared_fit_key": key,
        "fit_sha256": digest(root / "fits.json"),
        "status": "Historical research preview; older-cohort generalization criterion failed.",
        "assets": {},
    }
    for s, p in zip(model.SYMBOLS, probabilities):
        result["assets"][s] = {
            "probabilities": dict(zip(("down", "flat", "up"), map(float, p))),
            "signal": ("down", "flat", "up")[p.argmax()]
            if max(p[0], p[2]) >= 0.52
            else "abstain",
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "reports/timestral-universal-20261027"
    )
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = forecast(args.root, args.anchor)
    if args.output:
        write(args.output, result)
    print(json.dumps(result, sort_keys=True))
