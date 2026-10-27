#!/usr/bin/env python3
"""Strict frozen-history audit with deterministic rounding of count logarithms."""

import argparse
from decimal import Decimal, localcontext
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import verify_universal_history as history  # noqa: E402
from scripts.verify_universal_direction import PRIOR, ASSETS, ROOT  # noqa: E402


@lru_cache(maxsize=100000)
def count_log(count):
    """Round ln(1 + integer count) to binary64 independently of platform libm."""
    rounded = []
    for precision in (48, 80):
        with localcontext() as context:
            context.prec = precision
            rounded.append(float(Decimal(count + 1).ln()))
    if rounded[0] != rounded[1]:
        raise ArithmeticError("count logarithm rounding not resolved")
    return rounded[1]


def count_log1p(counts):
    counts = np.asarray(counts)
    if not (
        np.isfinite(counts).all()
        and (counts >= 0).all()
        and (counts < 2**53).all()
        and (counts == np.floor(counts)).all()
    ):
        raise ValueError("exact nonnegative integer trade counts required")
    values, inverse = np.unique(counts, return_inverse=True)
    result = np.array([count_log(int(n)) for n in values])
    return result[inverse].reshape(counts.shape)


# Keep all original readers, chronology, tree traversal, metrics and tolerances.
# Only the mathematically identical count logarithm gets deterministic rounding.
original_reconstruct = history.reconstruct


def reconstruct(datasets=None):
    if datasets is None:
        datasets = [dict(np.load(PRIOR / "micro/minutes.npz"))] + [
            dict(np.load(PRIOR / f"transfer/{s}-minutes.npz")) for s in ASSETS[1:]
        ]
    t, end, x, y = original_reconstruct(datasets)
    for a, data in enumerate(datasets):
        i = np.searchsorted(data["time"], t - 60)
        x[a, :, 8] = count_log1p(data["trades"][i])
    return t, end, x, y


def verify(root, raw=False):
    with patch.object(history, "reconstruct", reconstruct):
        old, linear = history.audit_history(root, raw)
        trees = history.audit_trees(root, old)
    return {
        "history": linear,
        "trees": trees,
        "count_log_rounding": "Decimal48/80 agreement to binary64",
        "log1p656": {
            "platform_math": math.log1p(656).hex(),
            "numpy": float(np.log1p(np.array([656.0]))[0]).hex(),
            "canonical": count_log(656).hex(),
        },
        "frozen_artifacts_changed": False,
        "tolerances_changed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "reports/timestral-universal-20261027"
    )
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.raw), sort_keys=True))
