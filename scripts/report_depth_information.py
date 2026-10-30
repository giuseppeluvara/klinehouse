#!/usr/bin/env python3
"""Report the registered QI20 criteria from one fresh independent audit only."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import verify_depth_information as verifier  # noqa: E402

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
ARMS = ("baseline", "price_control", "positions", "depth")
COMPARATORS = ("baseline", "price_control")
PROTECTED = {"archive", "archives", "holdout", "holdouts", "runtime"}
RESERVED = {
    "model.json",
    "config.json",
    "config-sha256.json",
    "protocol-plan.json",
    "source-identities.json",
    "training-data.json",
    "status.json",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def json_safe(value, *, path="audit", nonfinite=None):
    """Keep missing numeric evidence explicit while emitting strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite.append(path)
        return None
    if isinstance(value, dict):
        return {
            key: json_safe(item, path=f"{path}.{key}", nonfinite=nonfinite)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            json_safe(item, path=f"{path}[{i}]", nonfinite=nonfinite)
            for i, item in enumerate(value)
        ]
    return value


def build_report(audit):
    """Never reopen forecasts, outcomes, model files or sources after auditing."""
    require(
        audit.get("verified") is True
        and audit.get("schema") == "independent-depth-information-audit-v1",
        "a fresh successful independent depth audit is required",
    )
    require(
        all(finite(audit.get(key)) for key in ("as_of", "training_cutoff", "end_at")),
        "audit evaluation timestamps absent",
    )
    nonfinite = []
    audit = json_safe(audit, nonfinite=nonfinite)
    cases = audit.get("scored_cases")
    require(isinstance(cases, list), "audit scored_cases absent")
    require(
        audit.get("scored_outcomes") == len(cases),
        "audit scored-outcome denominator differs from scored_cases",
    )
    total, moving, dates, seen = Counter(), Counter(), defaultdict(set), set()
    for case in cases:
        symbol, anchor = case.get("symbol"), case.get("anchor")
        require(
            symbol in SYMBOLS and count(anchor) and anchor % 300 == 0,
            "invalid audited case membership",
        )
        require((anchor, symbol) not in seen, "duplicate audited case")
        seen.add((anchor, symbol))
        label, end, received = (
            case.get("label"),
            case.get("target_end"),
            case.get("source_received_at"),
        )
        require(
            count(label)
            and label in (0, 1, 2)
            and end == anchor + 360
            and finite(received)
            and end <= received <= audit["as_of"]
            and anchor >= audit["training_cutoff"]
            and end <= audit["end_at"],
            "invalid audited outcome chronology or label",
        )
        digest = case.get("forecast_sha256")
        require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            "audited forecast identity absent",
        )
        total[symbol] += 1
        if label != 1:
            moving[symbol] += 1
            dates[symbol].add(
                datetime.fromtimestamp(end, timezone.utc).date().isoformat()
            )

    assets, insufficient, descriptive_warnings = {}, [], []
    for symbol in SYMBOLS:
        n, moving_n = total[symbol], moving[symbol]
        available, issues, models = {arm: True for arm in ARMS}, [], {}
        raw_metrics = audit.get("metrics")
        raw_models = raw_metrics.get(symbol) if isinstance(raw_metrics, dict) else None
        raw_models = raw_models if isinstance(raw_models, dict) else {}
        for arm in ARMS:
            value = raw_models.get(arm)
            arm_issues = []
            if not isinstance(value, dict):
                models[arm] = None
                available[arm] = False
                issues.append((arm, f"{symbol}.{arm}: missing metrics"))
                continue
            models[arm] = value
            for key, expected in (("n", n), ("moving_n", moving_n)):
                if value.get(key) is None:
                    arm_issues.append(f"{symbol}.{arm}: missing {key}")
                else:
                    require(
                        count(value[key]) and value[key] == expected,
                        f"{symbol}.{arm} {key} denominator differs from scored_cases",
                    )
            correct, accuracy = (
                value.get("direction_correct"),
                value.get("direction_accuracy"),
            )
            if correct is None or (moving_n and accuracy is None):
                arm_issues.append(f"{symbol}.{arm}: missing direction metric")
            else:
                require(
                    count(correct) and correct <= moving_n,
                    f"{symbol}.{arm} invalid direction numerator",
                )
                require(
                    (not moving_n and accuracy is None)
                    or (
                        moving_n
                        and finite(accuracy)
                        and math.isclose(
                            accuracy, correct / moving_n, abs_tol=1e-12, rel_tol=0
                        )
                    ),
                    f"{symbol}.{arm} direction accuracy differs from numerator/denominator",
                )
            for key in ("brier", "log_loss"):
                if not finite(value.get(key)):
                    arm_issues.append(f"{symbol}.{arm}: missing or nonfinite {key}")
                else:
                    require(value[key] >= 0, f"{symbol}.{arm} negative proper score")
            available[arm] = not arm_issues
            issues.extend((arm, issue) for issue in arm_issues)
        insufficient.extend(issue for arm, issue in issues if arm != "positions")
        descriptive_warnings.extend(
            issue for arm, issue in issues if arm == "positions"
        )
        sample_ready = moving_n >= 288 and len(dates[symbol]) >= 2 and n > 0
        if not sample_ready:
            insufficient.append(
                f"{symbol}: requires 288 moving outcomes across at least two UTC target-end dates"
            )
        comparisons = {}
        for arm in (*COMPARATORS, "positions"):
            delta = None
            if available["depth"] and available[arm] and moving_n:
                candidate, control = models["depth"], models[arm]
                delta = dict(
                    direction_correct=candidate["direction_correct"]
                    - control["direction_correct"],
                    direction_accuracy=candidate["direction_accuracy"]
                    - control["direction_accuracy"],
                    brier=control["brier"] - candidate["brier"],
                    log_loss=control["log_loss"] - candidate["log_loss"],
                )
            comparisons[arm] = dict(
                role="registered_joint_comparator"
                if arm in COMPARATORS
                else "descriptive_only",
                delta_positive_is_better=delta,
                jointly_improved=(
                    all(
                        delta[key] > 0
                        for key in ("direction_correct", "brier", "log_loss")
                    )
                    if delta is not None
                    else None
                ),
            )
        assets[symbol] = dict(
            n=n,
            moving_n=moving_n,
            flat_n=n - moving_n,
            moving_target_end_utc_dates=sorted(dates[symbol]),
            sample_requirement_met=sample_ready,
            metrics_available=all(available[arm] for arm in (*COMPARATORS, "depth")),
            metrics_available_by_arm=available,
            models=models,
            comparisons=comparisons,
        )
    status = (
        "insufficient_evaluation"
        if insufficient
        else "preliminary_criteria_met"
        if all(
            assets[s]["comparisons"][a]["jointly_improved"]
            for s in SYMBOLS
            for a in COMPARATORS
        )
        else "criteria_not_met"
    )
    interim = audit["as_of"] < audit["end_at"] + 120
    return dict(
        schema="depth-information-registered-criteria-report-v1",
        as_of=audit["as_of"],
        as_of_utc=datetime.fromtimestamp(audit["as_of"], timezone.utc).isoformat(),
        evaluation_stage="interim" if interim else "scheduled_window_elapsed",
        completion_verified=False,
        criterion_status=status,
        registered_requirements=dict(
            minimum_moving_per_asset=288,
            minimum_distinct_utc_target_end_dates=2,
            all_assets_required=list(SYMBOLS),
            joint_comparators=list(COMPARATORS),
            strict_improvements=[
                "direction_correct",
                "three_class_sum_brier",
                "natural_log_loss",
            ],
            direction_excludes_flat=True,
            proper_scores_include_flat=True,
        ),
        assets=assets,
        insufficient_evidence=insufficient,
        descriptive_warnings=descriptive_warnings,
        nonfinite_audit_values_represented_as_null=nonfinite,
        validated_audit=audit,
        validated_audit_sha256=hashlib.sha256(
            json.dumps(
                audit, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
        interpretation=[
            "Confronto preliminare dei valori osservati; non prova superiorità statistica, redditività o validità universale.",
            "Valutazione intermedia: il periodo e il modello registrati restano invariati."
            if interim
            else "Scadenza temporale registrata trascorsa, inclusi i due minuti finali per gli esiti; l'arresto effettivo resta da verificare esternamente.",
            "Copertura, fallback, conferme tardive ed esiti mancanti restano integralmente visibili nell'audit; non costituiscono nuovi criteri di successo.",
        ],
    )


def report_root(path):
    root = Path(path).resolve()
    require(
        root.is_dir()
        and not any(
            part.casefold() in PROTECTED
            or part.casefold().startswith("shadow-pilot")
            or part.casefold().startswith("depth-research-")
            for part in root.parts
        ),
        "protected or source root refused",
    )
    return root


def write_report(root, output, result):
    root = report_root(root)
    output = Path(output).absolute()
    reports = root / "reports"
    require(
        not reports.is_symlink() and (not reports.exists() or reports.is_dir()),
        "report directory must be a real dedicated directory",
    )
    require(
        output.parent.resolve() == reports
        and output.suffix == ".json"
        and not output.name.startswith(".")
        and output.name.casefold() not in RESERVED
        and not output.exists()
        and not output.is_symlink(),
        "output must be a new run/reports/<name>.json file",
    )
    for source in result["validated_audit"].get("source_snapshots", {}).values():
        source = Path(source["root"]).resolve()
        require(
            source != reports
            and source not in reports.parents
            and reports not in source.parents,
            "output overlaps an audited source",
        )
    reports.mkdir(exist_ok=True)
    directory = os.open(reports, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    pending = f".{output.name}.pending-{os.getpid()}-{time.time_ns()}"
    try:
        handle = os.open(
            pending,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(handle, "w") as stream:
            stream.write(
                json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(
            pending,
            output.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        os.fsync(directory)
    finally:
        os.close(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        root = report_root(args.root)
        result = build_report(verifier.verify(root))
        if args.output:
            write_report(root, args.output, result)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(
            json.dumps(dict(error=str(error), report_generated=False)), file=sys.stderr
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
