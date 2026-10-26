#!/usr/bin/env python3
"""Independent bootstrap audit. Execute only after the parent authorizes opening test results.

This standalone script imports no pipeline module. It expands sampled blocks to
individual row indices, rather than aggregating weighted block sums as the main
analysis does. It creates only bootstrap-verification.json and never replaces it.
"""

from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

MODELS = ("unchanged", "timestral_price", "timestral_covariates", "adapted")
CONTRASTS = (("adapted", "unchanged"), ("timestral_price", "unchanged"),
             ("timestral_covariates", "unchanged"), ("adapted", "timestral_price"))
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 4, 1, tzinfo=timezone.utc)
BLOCK_DAYS, BLOCKS, DRAWS, SEED = 7, 13, 10000, 17
ATOL, RTOL = 1e-14, 1e-12


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("all timestamps must declare UTC")
    return parsed


def audit(root):
    root = Path(root).resolve()
    output = root / "bootstrap-verification.json"
    if output.exists():
        raise FileExistsError(output)
    paths = {"evaluation.jsonl": root / "test-run/evaluation.jsonl",
             "analysis.json": root / "analysis/analysis.json",
             "frozen/protocol.json": root / "frozen/protocol.json"}
    hashes = {name: sha(path) for name, path in paths.items()}
    analysis = json.loads(paths["analysis.json"].read_text())
    protocol = json.loads(paths["frozen/protocol.json"].read_text())
    with paths["evaluation.jsonl"].open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if analysis["input_sha256"]["evaluation.jsonl"] != hashes["evaluation.jsonl"]:
        raise ValueError("analysis refers to different evaluation bytes")
    uncertainty = protocol["uncertainty"]
    if (utc(protocol["test_start"]) != START or utc(protocol["test_end"]) != END
            or uncertainty["calendar_block_days"] != BLOCK_DAYS
            or uncertainty["bootstrap_draws"] != DRAWS or uncertainty["seed"] != SEED
            or uncertainty["primary_interval_quantiles"] != [.0125, .9875]):
        raise ValueError("frozen bootstrap specification differs from this independent auditor")
    if analysis["analysis_code_sha256"] != protocol["code_sha256"]["scripts/analyze_timestral_adaptation.py"]:
        raise ValueError("analysis reports a code hash different from the frozen protocol")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate evaluation identities")
    if {row["horizon_hours"] for row in rows} != {4, 24}:
        raise ValueError("unexpected horizon set")
    findings, errors = {}, []
    deviations = {"mean_mae_difference": 0., "percentile95": 0., "percentile97_5": 0.,
                  "fraction_nonempty_blocks_lower_mae": 0.}

    def compare(label, observed, expected, category):
        left, right = np.asarray(observed, dtype=float), np.asarray(expected, dtype=float)
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"invalid values or shape: {label}")
        maximum = float(np.max(np.abs(left - right)))
        deviations[category] = max(deviations[category], maximum)
        if not np.allclose(left, right, atol=ATOL, rtol=RTOL):
            errors.append(label)

    # The identical random integer stream is required by the frozen seed.
    # Loss resampling below is independent: each draw expands actual row indices.
    choices = np.random.default_rng(SEED).integers(BLOCKS, size=(DRAWS, BLOCKS))
    for horizon in (4, 24):
        group = sorted((row for row in rows if row["horizon_hours"] == horizon), key=lambda row: utc(row["anchor"]))
        summary = analysis["horizons"][str(horizon)]
        reported = summary["paired_comparison"]
        if [row["sample_id"] for row in group] != summary["sample_ids"]:
            raise ValueError("analysis and evaluation sample identities differ")
        planned = (END - START).days * 24 // horizon
        if (summary["samples"] != len(group) or summary["expected_samples"] != planned
                or summary["excluded_samples"] != planned - len(group)):
            raise ValueError("analysis sample counts differ")
        anchors = [utc(row["anchor"]) for row in group]
        if any(not START <= anchor < END or anchor.minute or anchor.second or anchor.microsecond
               or anchor.hour % horizon for anchor in anchors):
            raise ValueError("incorrect anchor calendar")
        if any(right < left + timedelta(hours=horizon) for left, right in zip(anchors, anchors[1:])):
            raise ValueError("overlapping horizon labels")
        if any(utc(row["label_end"]) != anchor + timedelta(hours=horizon) or utc(row["label_end"]) > END
               or row.get("status") != "scored" or set(row["predictions"]) != set(MODELS)
               for row, anchor in zip(group, anchors)):
            raise ValueError("incorrect target endpoint or model coverage")
        blocks = [np.array([i for i, anchor in enumerate(anchors)
                            if START + timedelta(days=BLOCK_DAYS * block) <= anchor
                            < min(END, START + timedelta(days=BLOCK_DAYS * (block + 1)))], dtype=int)
                  for block in range(BLOCKS)]
        counts = [len(block) for block in blocks]
        if (reported["seed"] != SEED or reported["draws"] != DRAWS or reported["blocks"] != BLOCKS
                or reported["block_sample_counts"] != counts or utc(reported["block_origin_utc"]) != START):
            raise ValueError("reported bootstrap seed, calendar, or counts differ")
        actual = np.array([row["actual_log_return"] for row in group], dtype=float)
        losses = {name: np.abs(np.array([row["predictions"][name]["point"] for row in group]) - actual)
                  for name in MODELS}
        differences = np.column_stack([losses[candidate] - losses[reference] for candidate, reference in CONTRASTS])
        if not np.isfinite(differences).all():
            raise ValueError("non-finite prediction or outcome")
        draws = np.empty((DRAWS, len(CONTRASTS)))
        for draw, sampled_blocks in enumerate(choices):
            expanded_rows = np.concatenate([blocks[index] for index in sampled_blocks])
            if not len(expanded_rows):
                raise ValueError("empty expanded calendar sample")
            draws[draw] = differences[expanded_rows].mean(axis=0)
        contrasts = {}
        for column, (candidate, reference) in enumerate(CONTRASTS):
            name = f"{candidate}_minus_{reference}"
            computed = {"mean_mae_difference": float(differences[:, column].mean()),
                        "percentile95": np.quantile(draws[:, column], [.025, .975]).tolist(),
                        "fraction_nonempty_blocks_lower_mae": float(np.mean([
                            differences[block, column].mean() < 0 for block in blocks if len(block)]))}
            if column == 0:
                computed["percentile97_5"] = np.quantile(draws[:, column], [.0125, .9875]).tolist()
                if (computed["percentile97_5"][1] < 0) != summary["gate"]["criteria"]["adjusted_upper_ci_below_zero"]:
                    errors.append(f"{horizon}h.primary_upper_endpoint_gate")
            for key, value in computed.items():
                compare(f"{horizon}h.{name}.{key}", value, reported["contrasts"][name][key], key)
            contrasts[name] = computed
        findings[str(horizon)] = {"samples": len(group), "expected_samples": planned,
                                  "block_sample_counts": counts, "contrasts": contrasts}
    report = {
        "passed": not errors, "checked_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": hashes, "auditor_sha256": sha(__file__), "numpy_version": np.__version__,
        "implementation": "standalone auditor; sampled calendar blocks expanded to individual row indices; direct mean of paired per-anchor absolute-error differences; no pipeline imports",
        "method": {"seed": SEED, "draws": DRAWS, "blocks": BLOCKS, "block_days": BLOCK_DAYS,
                   "last_block_days": 6, "origin": START.isoformat(), "primary_quantiles": [.0125, .9875],
                   "secondary_quantiles": [.025, .975], "weighting": "equal weight per sampled anchor"},
        "criterion": {"counts_and_identities": "exact match", "numeric_absolute_tolerance": ATOL,
                      "numeric_relative_tolerance": RTOL, "all_contrasts_match": not errors},
        "maximum_absolute_deviations": deviations, "horizons": findings, "failures": errors,
        "limitations": "This checks the arithmetic and reported endpoints; it does not establish independence, calibrated coverage, prospective accuracy, or profitability.",
    }
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    output.chmod(0o444)
    print(json.dumps({"passed": report["passed"], "output": str(output),
                      "maximum_absolute_deviations": deviations}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    audit(parser.parse_args().root)
