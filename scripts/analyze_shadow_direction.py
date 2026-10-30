#!/usr/bin/env python3
"""Describe direction failure on an independently verified shadow checkpoint.

This post-hoc analysis reads saved probabilities and outcomes only. It never
fits a model, reruns inference, accesses historical market readers, or changes
the live collector. Its neutral and continuation controls are diagnostics, not
new prospective forecasts or evidence that reversing the signal will work.
"""

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from itertools import combinations, groupby
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_shadow_run import verify  # noqa: E402


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MODELS = ("baseline", "price_control", "positions")
THRESHOLD = 0.52
BLOCK_SECONDS = 6 * 3600


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def mean(values):
    return math.fsum(values) / len(values) if values else None


def utc(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def probabilities(value):
    require(
        len(value) == 3 and all(finite(p) and 0 < p < 1 for p in value),
        "three interior probabilities required",
    )
    require(
        math.isclose(math.fsum(value), 1, rel_tol=0, abs_tol=1e-9),
        "probabilities do not sum to one",
    )
    return tuple(value)


def arithmetic(probability, label):
    """Independent arithmetic; class order down/flat/up, direction ties up."""
    p = probabilities(probability)
    require(type(label) is int and label in (0, 1, 2), "invalid realized class")
    q = p[2] / (p[0] + p[2])
    return {
        "q": q,
        "predicted_up": int(q >= 0.5),
        "class_prediction": max(range(3), key=lambda i: p[i]),
        "brier": math.fsum((p[i] - (i == label)) ** 2 for i in range(3)),
        "log_loss": -math.log(p[label]),
        "selected": max(p[0], p[2]) >= THRESHOLD,
    }


def auc(observations):
    """P(score_up > score_down), with half credit for score ties."""
    up = sum(label for _, label in observations)
    down = len(observations) - up
    if not up or not down:
        return None
    concordant, previous_down = 0.0, 0
    for _, group in groupby(sorted(observations), key=lambda row: row[0]):
        labels = [label for _, label in group]
        group_up = sum(labels)
        group_down = len(labels) - group_up
        concordant += group_up * (previous_down + group_down / 2)
        previous_down += group_down
    return concordant / (up * down)


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {}
    result = {}
    for name, fraction in (
        ("min", 0),
        ("p10", 0.1),
        ("p25", 0.25),
        ("p50", 0.5),
        ("p75", 0.75),
        ("p90", 0.9),
        ("max", 1),
    ):
        position = fraction * (len(ordered) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        result[name] = ordered[lower] + (ordered[upper] - ordered[lower]) * (
            position - lower
        )
    return result


def collect_cases(events, *, now, deadline=30):
    """Join only same-asset/anchor forecasts, valid timely acks and mature labels.

    The caller verifies ledger integrity separately. This function also applies
    the captured-time boundary, so later acknowledgements/outcomes cannot enter
    an earlier descriptive prefix.
    """
    require(finite(now), "finite capture time required")
    forecasts, acknowledgements, outcomes = {}, {}, {}
    for event in events:
        if event["received_at"] > now:
            continue
        kind = event["kind"]
        if kind not in ("forecast", "forecast_commit", "outcome"):
            continue
        payload = event["payload"]
        payload = json.loads(payload) if isinstance(payload, str) else payload
        key = (payload["anchor"], payload["symbol"])
        target = {
            "forecast": forecasts,
            "forecast_commit": acknowledgements,
            "outcome": outcomes,
        }[kind]
        require(key not in target, "duplicate saved observation")
        target[key] = (event, payload)
    eligible = {}
    for key, (event, forecast) in forecasts.items():
        if key not in acknowledgements:
            continue
        ack_event, ack = acknowledgements[key]
        anchor = key[0]
        committed = ack.get("committed_at")
        if (
            ack.get("eligible") is True
            and ack.get("clock_healthy") is True
            and finite(committed)
            and forecast["issued_at"] <= event["received_at"] <= committed
            and committed <= ack_event["received_at"] <= now
            and anchor <= forecast["issued_at"] <= anchor + deadline
            and committed <= anchor + deadline
            and event["seq"] < ack_event["seq"]
            and ack.get("forecast_seq") == event["seq"]
            and ack.get("forecast_hash") == event["digest"]
        ):
            eligible[key] = forecast
    cases = []
    for key in sorted(eligible.keys() & outcomes.keys()):
        outcome_event, outcome = outcomes[key]
        ack_event, _ = acknowledgements[key]
        anchor, symbol = key
        require(
            anchor + 360 <= outcome_event["received_at"] <= now,
            "outcome recorded before maturity",
        )
        require(
            outcome.get("return_end") == anchor + 360,
            "outcome target end differs",
        )
        require(
            ack_event["seq"] < outcome_event["seq"]
            and ack_event["received_at"] <= outcome_event["received_at"],
            "outcome precedes its saved acknowledgement",
        )
        forecast = eligible[key]
        require(set(forecast["models"]) == set(MODELS), "forecast arms differ")
        for model in MODELS:
            arithmetic(forecast["models"][model], outcome["label"])
        cases.append(
            {
                "anchor": anchor,
                "symbol": symbol,
                "label": outcome["label"],
                "models": forecast["models"],
                "diagnostics": forecast.get("diagnostics", {}),
            }
        )
    return cases, {
        "forecast_count": len(forecasts),
        "eligible_forecast_count": len(eligible),
        "scored_count": len(cases),
        "pending_count": len(eligible) - len(cases),
        "mature_pending_count": sum(
            key not in outcomes and key[0] + 360 <= now for key in eligible
        ),
        "unacknowledged_count": len(forecasts.keys() - acknowledgements.keys()),
        "ineligible_count": len(forecasts) - len(eligible),
    }


def summarize(cases, model, *, neutral=False):
    rows = []
    for case in cases:
        p = case["models"][model]
        if neutral:
            p = ((1 - p[1]) / 2, p[1], (1 - p[1]) / 2)
        rows.append((case, arithmetic(p, case["label"])))
    moving = [(case, score) for case, score in rows if case["label"] != 1]
    y = [int(case["label"] == 2) for case, _ in moving]
    h = [score["predicted_up"] for _, score in moving]
    q = [score["q"] for _, score in moving]
    confusion = [[0, 0], [0, 0]]
    for actual, predicted in zip(y, h):
        confusion[actual][predicted] += 1
    up, down = sum(y), len(y) - sum(y)
    hits = sum(actual == predicted for actual, predicted in zip(y, h))
    actual_up, predicted_up = mean(y), mean(h)
    if y:
        imbalance = 2 * (actual_up - 0.5) * (predicted_up - 0.5)
        association = 2 * (
            mean([a * b for a, b in zip(y, h)]) - actual_up * predicted_up
        )
        accuracy = hits / len(y)
        decomposition = {
            "accuracy_minus_half": accuracy - 0.5,
            "class_imbalance_component": imbalance,
            "association_component": association,
            "identity_error": accuracy - 0.5 - imbalance - association,
        }
    else:
        accuracy, decomposition = None, {}
    classes = Counter(case["label"] for case in cases)
    bins = []
    for lower, upper in ((0, 0.48), (0.48, 0.5), (0.5, 0.52), (0.52, 1)):
        selected = [
            (case, score) for case, score in moving if lower <= score["q"] < upper
        ]
        bins.append(
            {
                "lower_inclusive": lower,
                "upper_exclusive": upper,
                "n": len(selected),
                "mean_q": mean([score["q"] for _, score in selected]),
                "observed_up_fraction": mean(
                    [int(case["label"] == 2) for case, _ in selected]
                ),
            }
        )
    return {
        "n": len(cases),
        "realized_class_counts": {str(k): classes[k] for k in (0, 1, 2)},
        "class_correct": sum(
            score["class_prediction"] == case["label"] for case, score in rows
        ),
        "three_class_accuracy": mean(
            [int(score["class_prediction"] == case["label"]) for case, score in rows]
        ),
        "brier": mean([score["brier"] for _, score in rows]),
        "log_loss": mean([score["log_loss"] for _, score in rows]),
        "direction_n": len(moving),
        "direction_correct": hits,
        "direction_accuracy": accuracy,
        "predicted_up_count_all": sum(score["predicted_up"] for _, score in rows),
        "predicted_up_fraction_all": mean([score["predicted_up"] for _, score in rows]),
        "actual_up_fraction_moving": actual_up,
        "predicted_up_fraction_moving": predicted_up,
        "balanced_accuracy": (
            (confusion[1][1] / up + confusion[0][0] / down) / 2 if up and down else None
        ),
        "auc": auc(list(zip(q, y))),
        "confusion_actual_down_up_by_predicted_down_up": confusion,
        "mean_q": mean(q),
        "mean_direction_confidence": mean([max(p, 1 - p) for p in q]),
        "q_quantiles": quantiles(q),
        "decomposition": decomposition,
        "fixed_q_calibration_bins": bins,
    }


def flip_counts(cases, model):
    changed, helped, hurt, flat = 0, 0, 0, 0
    for case in cases:
        old = arithmetic(case["models"]["baseline"], case["label"])
        new = arithmetic(case["models"][model], case["label"])
        if old["predicted_up"] == new["predicted_up"]:
            continue
        changed += 1
        if case["label"] == 1:
            flat += 1
        elif new["predicted_up"] == (case["label"] == 2):
            helped += 1
        else:
            hurt += 1
    return {"changed": changed, "helped": helped, "hurt": hurt, "flat": flat}


def hour_diagnostic(cases):
    signal_hour, prediction_matches, continuation = [], [], []
    for case in cases:
        d = case["diagnostics"]
        hour = d.get("price_features", [None, None])[1]
        signal = d.get("uniform_signal")
        if not finite(hour) or not finite(signal):
            continue
        signal_hour.append((signal, hour))
        predicted = arithmetic(case["models"]["baseline"], case["label"])
        prediction_matches.append(int(predicted["predicted_up"] == (hour >= 0)))
        if case["label"] != 1:
            continuation.append(int((hour >= 0) == (case["label"] == 2)))
    if signal_hour:
        x, y = zip(*signal_hour)
        mx, my = mean(x), mean(y)
        variance = math.fsum((v - mx) ** 2 for v in x) * math.fsum(
            (v - my) ** 2 for v in y
        )
        correlation = (
            math.fsum((a - mx) * (b - my) for a, b in signal_hour) / math.sqrt(variance)
            if variance
            else None
        )
    else:
        correlation = None
    return {
        "n": len(prediction_matches),
        "signal_vs_lagged_hour_correlation": correlation,
        "baseline_follows_lagged_hour_fraction": mean(prediction_matches),
        "moving_n": len(continuation),
        "outcome_continues_lagged_hour_fraction": mean(continuation),
        "note": "Descriptive continuation benchmark; not a saved forecast arm.",
    }


def temporal_diagnostics(cases):
    blocks = {}
    for block in sorted({case["anchor"] // BLOCK_SECONDS for case in cases}):
        selected = [case for case in cases if case["anchor"] // BLOCK_SECONDS == block]
        blocks[utc(block * BLOCK_SECONDS)] = {
            "anchor_start": block * BLOCK_SECONDS,
            "models": {model: summarize(selected, model) for model in MODELS},
        }
    stratified = {}
    for model in MODELS:
        components = [block["models"][model] for block in blocks.values()]
        pairs = [
            row["realized_class_counts"]["0"] * row["realized_class_counts"]["2"]
            for row in components
        ]
        count = sum(pairs)
        stratified[model] = {
            "comparable_pairs": count,
            "pair_weighted_within_block_auc": (
                math.fsum(
                    weight * row["auc"]
                    for weight, row in zip(pairs, components)
                    if weight
                )
                / count
                if count
                else None
            ),
        }
    return {"blocks_utc_six_hours_by_anchor": blocks, "stratified_auc": stratified}


def diagnostics(cases):
    assets = {}
    for symbol in SYMBOLS:
        own = [case for case in cases if case["symbol"] == symbol]
        baseline_subset = [
            case
            for case in own
            if arithmetic(case["models"]["baseline"], case["label"])["selected"]
        ]
        own_subsets = {}
        for model in MODELS:
            selected = [
                case
                for case in own
                if arithmetic(case["models"][model], case["label"])["selected"]
            ]
            own_subsets[model] = {
                "coverage": len(selected) / len(own) if own else None,
                **summarize(selected, model),
            }
        assets[symbol] = {
            "all_saved_matured": {model: summarize(own, model) for model in MODELS},
            "own_threshold_subset": own_subsets,
            "fixed_baseline_threshold_subset": {
                "coverage": len(baseline_subset) / len(own) if own else None,
                "models": {
                    model: summarize(baseline_subset, model) for model in MODELS
                },
            },
            "neutral_direction_preserving_baseline_flat": summarize(
                own, "baseline", neutral=True
            ),
            "flips_vs_baseline": {
                model: flip_counts(own, model) for model in MODELS[1:]
            },
            "lagged_hour": hour_diagnostic(own),
            "temporal": temporal_diagnostics(own),
        }
    by_anchor = {}
    for case in cases:
        by_anchor.setdefault(case["anchor"], {})[case["symbol"]] = case
    common = [row for row in by_anchor.values() if set(row) == set(SYMBOLS)]
    same_prediction = []
    pair_agreements = {}
    for row in common:
        predictions = [
            arithmetic(row[s]["models"]["baseline"], row[s]["label"])["predicted_up"]
            for s in SYMBOLS
        ]
        same_prediction.append(int(len(set(predictions)) == 1))
    for first, second in combinations(SYMBOLS, 2):
        selected = [
            row
            for row in common
            if row[first]["label"] != 1 and row[second]["label"] != 1
        ]
        pair_agreements[f"{first}:{second}"] = {
            "moving_n": len(selected),
            "realized_direction_agreement": mean(
                [int(row[first]["label"] == row[second]["label"]) for row in selected]
            ),
        }
    return {
        "assets": assets,
        "cross_asset_dependence": {
            "common_anchors": len(common),
            "baseline_same_direction_all_three_fraction": mean(same_prediction),
            "pairs": pair_agreements,
        },
    }


def analyze_checkpoint(checkpoint):
    root = Path(checkpoint).resolve()
    require(
        "shadow-pilot-20261029-v2" not in root.parts,
        "use an isolated checkpoint, not the live pilot",
    )
    receipt_path = root.parent / "checkpoint-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    captured = receipt["captured_at"]
    require(receipt.get("verified") is True and finite(captured), "invalid receipt")
    for name in ("events.sqlite", "config.json", "model.json", "general-audit.json"):
        require(
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            == receipt["files"][name],
            f"checkpoint file hash differs: {name}",
        )
    audit = verify(
        root / "events.sqlite",
        snapshot=root / "model.json",
        protocol=root / "config.json",
        now=captured,
    )
    require(
        audit["event_count"] == receipt["event_count"]
        and audit["chain_tip"] == receipt["chain_tip"],
        "checkpoint prefix differs from captured receipt",
    )
    with closing(
        sqlite3.connect((root / "events.sqlite").as_uri() + "?mode=ro", uri=True)
    ) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        events = [
            dict(row) for row in connection.execute("SELECT * FROM events ORDER BY seq")
        ]
        connection.rollback()
    require(
        all(event["received_at"] <= captured for event in events),
        "checkpoint includes observations after capture",
    )
    config = json.loads((root / "config.json").read_text())
    cases, accounting = collect_cases(
        events, now=captured, deadline=config["issuance_deadline_seconds"]
    )
    require(
        accounting["scored_count"] == audit["outcome_count"]
        and accounting["eligible_forecast_count"] == audit["eligible_forecast_count"],
        "independent joined counts differ from integrity audit",
    )
    return {
        "schema": "shadow-direction-diagnosis-v1",
        "captured_at": captured,
        "captured_utc": utc(captured),
        "checkpoint": str(root),
        "source_run": receipt["source_run"],
        "integrity_audit_verified": audit["verified"],
        "event_count": audit["event_count"],
        "chain_tip": audit["chain_tip"],
        "accounting": accounting,
        "methods": {
            "threshold": THRESHOLD,
            "selection": "max(unconditional p_down, unconditional p_up) >= 0.52",
            "direction": "Moving labels only; q=p_up/(p_down+p_up), ties up.",
            "three_class": "All labels; argmax, ties choose the lowest class index.",
            "scores": "Three-class sum Brier; natural logarithm log-loss.",
            "confusion": "Rows actual down/up; columns predicted down/up.",
            "temporal_blocks": "Fixed six-hour UTC blocks assigned by anchor.",
        },
        "limitations": [
            "Post-hoc descriptive diagnosis of a short, correlated sample.",
            "No fitting, inference, threshold search or modification of the live pilot.",
            "Neutral and continuation controls are diagnostics, not saved forecast arms.",
            "No statistical, stable-regime, profitability or signal-inversion claim.",
            "Positive probability rescaling cannot change direction signs or AUC.",
        ],
        **diagnostics(cases),
    }


def write_result(result, output):
    path = Path(output).resolve()
    source = Path(result["source_run"])
    if not source.is_absolute():
        source = Path(__file__).resolve().parents[1] / source
    require(
        path != source.resolve()
        and source.resolve() not in path.parents
        and "shadow-pilot-20261029-v2" not in path.parts,
        "diagnosis output must remain outside the live pilot",
    )
    with path.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_checkpoint(args.checkpoint)
    write_result(result, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "captured_utc": result["captured_utc"],
                "integrity_audit_verified": result["integrity_audit_verified"],
                **result["accounting"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
