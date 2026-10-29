#!/usr/bin/env python3
"""Freeze the last allowed July states for a separate prospective shadow test."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import shadow_model as model  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(output_path, root=ROOT):
    """Read only enumerated frozen pre-August artifacts; never refit or overwrite."""
    root, output_path = Path(root), Path(output_path)
    paths = {
        "center": root / "reports/timestral-volume-center-20261027/fits.json",
        "positions": root / "reports/timestral-position-information-20261029/fits.json",
        "universal": root / "reports/timestral-universal-20261027/fits.json",
    }
    states = {key: json.loads(path.read_text()) for key, path in paths.items()}
    frequency = {}
    for symbol in model.SYMBOLS:
        path = root / f"reports/timestral-universal-20261027/{symbol}-predictions.npz"
        with np.load(path, allow_pickle=False) as data:
            anchor = data["anchor"]
            if not np.isfinite(anchor).all() or np.any(
                anchor + 360 > model.PROTECTED_BOUNDARY
            ):
                raise ValueError("unexpected protected prediction rows")
            mask = anchor >= model.FROZEN_CUTOFF
            values = data["frequency"][mask]
            if not len(values) or not np.all(values == values[0]):
                raise ValueError("one fixed July frequency required")
            frequency[symbol] = values[0].tolist()
        paths[f"frequency_{symbol}"] = path
    snapshot = {
        "schema": model.SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cutoff": model.FROZEN_CUTOFF,
        "symbols": list(model.SYMBOLS),
        "metrics_lag_seconds": model.METRICS_LAG_SECONDS,
        "baseline_beta": states["center"]["2026-7-uniform"]["adjusted"]["beta"],
        "flat_model": states["universal"]["cross_7_all"],
        "frequency": frequency,
        "corrections": {
            arm: states["positions"][f"2026-7-{arm}"] for arm in model.ARMS[1:]
        },
        "sources": {
            str(path.relative_to(root)): digest(path) for path in paths.values()
        },
        "code": {
            name: digest(root / name)
            for name in ("research/shadow_model.py", "scripts/freeze_shadow_model.py")
        },
        "target": "sign(close(open=anchor+300)-close(open=anchor))+1; exact equality is flat; target closes anchor+60 to anchor+360",
        "limitations": [
            "July coefficients, normalizations and class frequencies are frozen, not current rolling fits.",
            "Historical OI/position timing and archive-to-REST equivalence are not certified by this snapshot.",
            "The caller must enforce receipt-before-issuance, prospective start, no protected backfill, and forecast deadlines.",
            "Missing any asset's metric context makes both correction arms fall back to baseline for every asset.",
        ],
    }
    model.validate_snapshot(snapshot)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(snapshot, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return snapshot


def audit_frozen_parity(snapshot, root=ROOT):
    """Compare daily July emissions with saved predictions, without fitting.

    This explicit offline check is never called by the prospective predictor.
    Every source is enumerated and checked against the protected boundary before
    loading any price, metric value or saved probability array.
    """
    root = Path(root)
    model.validate_snapshot(snapshot)
    anchors = model.FROZEN_CUTOFF + np.arange(31) * 86400 + 12 * 3600
    spot, metrics, expected = {}, {}, {}
    receipts = {}
    for a, symbol in enumerate(model.SYMBOLS):
        spot_path = (
            root
            / "reports/timestral-microdirection-20261027"
            / ("micro/minutes.npz" if a == 0 else f"transfer/{symbol}-minutes.npz")
        )
        metrics_path = (
            root
            / f"reports/timestral-position-information-20261029/2026-{symbol}-metrics.npz"
        )
        prediction_path = (
            root
            / f"reports/timestral-position-information-20261029/2026-{symbol}-predictions.npz"
        )
        with np.load(spot_path, allow_pickle=False) as source:
            time = source["time"]
            if np.any(time >= model.PROTECTED_BOUNDARY):
                raise ValueError("protected source in offline parity audit")
            spot[symbol] = {
                name: source[name]
                for name in ("time", "close", "volume", "trades", "taker")
            }
        with np.load(metrics_path, allow_pickle=False) as source:
            label = source["label"]
            if np.any(label + 300 >= model.PROTECTED_BOUNDARY):
                raise ValueError("protected metrics in offline parity audit")
            metrics[symbol] = dict(label=label, values=source["values"])
        with np.load(prediction_path, allow_pickle=False) as source:
            time = source["anchor"]
            if np.any(time + 360 > model.PROTECTED_BOUNDARY):
                raise ValueError("protected predictions in offline parity audit")
            ids = np.searchsorted(time, anchors)
            if np.any(ids >= len(time)) or not np.array_equal(time[ids], anchors):
                raise ValueError("missing July audit anchors")
            expected[symbol] = {
                name: source[name][ids] for name in (*model.ARMS, "available")
            }
        for path in (spot_path, metrics_path, prediction_path):
            receipts[str(path.relative_to(root))] = digest(path)
    maximum_error = 0.0
    available_emissions = 0
    for row, anchor in enumerate(anchors):
        context, rows = {}, {}
        for symbol in model.SYMBOLS:
            source = spot[symbol]
            mask = (source["time"] >= anchor - 3780) & (source["time"] < anchor)
            context[symbol] = {key: value[mask] for key, value in source.items()}
            metric = metrics[symbol]
            mask = (metric["label"] >= anchor - 4500) & (
                metric["label"] <= anchor - 900
            )
            rows[symbol] = [
                dict(
                    label=int(label), oi=float(value[0]), position_ratio=float(value[3])
                )
                for label, value in zip(metric["label"][mask], metric["values"][mask])
            ]
        actual = model.predict(snapshot, context, rows, int(anchor))
        for symbol in model.SYMBOLS:
            valid = actual[symbol]["diagnostics"]["correction_available"]
            if valid != bool(expected[symbol]["available"][row]):
                raise ValueError("saved and prospective fallback availability differs")
            available_emissions += int(valid)
            for arm in model.ARMS:
                probability = np.array(actual[symbol]["models"][arm])
                error = float(np.max(np.abs(probability - expected[symbol][arm][row])))
                maximum_error = max(maximum_error, error)
                np.testing.assert_allclose(
                    probability, expected[symbol][arm][row], atol=2e-10, rtol=0
                )
    return {
        "status": "passed",
        "anchors": len(anchors),
        "asset_emissions": len(anchors) * 3,
        "model_probability_vectors": len(anchors) * 9,
        "corrections_available_asset_emissions": available_emissions,
        "maximum_absolute_probability_error": maximum_error,
        "first_anchor": int(anchors[0]),
        "last_anchor": int(anchors[-1]),
        "sources": receipts,
        "protected_data_opened": False,
        "refitting_performed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-parity",
        action="store_true",
        help="also compare 31 daily July anchors with frozen saved probabilities",
    )
    args = parser.parse_args()
    snapshot = freeze(args.output)
    result = {
        "snapshot": str(args.output),
        "sha256": digest(args.output),
        "cutoff": model.FROZEN_CUTOFF,
    }
    if args.verify_parity:
        result["parity"] = audit_frozen_parity(snapshot)
    print(json.dumps(result, sort_keys=True))
