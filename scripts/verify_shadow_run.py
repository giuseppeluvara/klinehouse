#!/usr/bin/env python3
"""Independently audit a read-only shadow ledger and its matured prediction scores.

This module deliberately imports neither the collector, ledger implementation,
nor prediction code. It verifies recorded causality and arithmetic; local hashes
are integrity checks, not an external timestamp authority or proof of skill.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
ARMS = ("baseline", "price_control", "positions")
SOURCES = ("spot_klines", "open_interest", "position_ratio")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def price(value):
    require(isinstance(value, str), "outcome prices must retain original strings")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal price") from exc
    require(number.is_finite() and number > 0, "invalid positive decimal price")
    return number


def source_rows(payload, received, config):
    """Decode source timestamps independently from the declared content_times."""
    source = payload["source"]
    require(source in SOURCES, "unregistered market source")
    require(payload.get("symbol") in SYMBOLS, "unregistered receipt symbol")
    body = payload["body"]
    require(isinstance(body, list), "market response must retain its JSON array")
    require(payload.get("body_sha256") == sha(body), "raw response body hash mismatch")
    times = []
    for row in body:
        if source == "spot_klines":
            require(isinstance(row, list) and len(row) == 12, "invalid raw spot candle")
            timestamp = row[0] / 1000
            require(row[6] == row[0] + 59999, "spot candle is not one minute")
            require(
                (row[6] + 1) / 1000
                <= received + config.get("max_clock_skew_seconds", 2),
                "received unclosed spot candle",
            )
            require(
                (row[6] + 1) / 1000 <= config["end_at"],
                "spot candle closes after prospective end",
            )
            require(timestamp % 60 == 0, "spot minute is off grid")
            for index in (1, 2, 3, 4):
                price(row[index])
            for index in (5, 7, 8, 9, 10):
                require(
                    math.isfinite(float(row[index])) and float(row[index]) >= 0,
                    "invalid spot activity",
                )
            require(float(row[9]) <= float(row[5]), "spot buy volume exceeds total")
        else:
            require(isinstance(row, dict), "invalid raw metrics row")
            timestamp = row["timestamp"] / 1000
            require(timestamp % 300 == 0, "metric label is off grid")
            field = "sumOpenInterest" if source == "open_interest" else "longShortRatio"
            price(row[field])
            if "symbol" in row:
                require(
                    row["symbol"] == payload["symbol"], "source row symbol mismatch"
                )
        require(
            finite(timestamp) and config["start_at"] <= timestamp < config["end_at"],
            "market row outside prospective window",
        )
        require(
            timestamp <= received + config.get("max_clock_skew_seconds", 2),
            "market row dated after receipt",
        )
        times.append(timestamp)
    require(
        times == payload["content_times"],
        "declared source times differ from raw response",
    )
    require(
        len(times) == len(set(times)), "duplicate source labels within one response"
    )
    params = payload.get("request_params")
    require(
        isinstance(params, dict), "market response lacks bounded request parameters"
    )
    require(params.get("symbol") == payload["symbol"], "request symbol mismatch")
    require(
        params.get("startTime", 0) >= math.ceil(config["start_at"] * 1000),
        "request reaches before prospective start",
    )
    require(
        params.get("endTime", float("inf")) < config["end_at"] * 1000,
        "request reaches beyond prospective end",
    )
    require(
        all(
            params["startTime"] <= stamp * 1000 <= params["endTime"] for stamp in times
        ),
        "response row outside request interval",
    )
    requested = payload.get("requested_at")
    if requested is not None:
        require(
            finite(requested) and requested <= received, "request follows its receipt"
        )
    return times


def losses(probabilities, label):
    """Natural log-loss, sum-of-squares 3-class Brier, conditional direction."""
    result = {
        "brier": sum(
            (p - int(label == index)) ** 2 for index, p in enumerate(probabilities)
        ),
        "log_loss": -math.log(probabilities[label]),
    }
    if label != 1:
        up = probabilities[2] / (probabilities[0] + probabilities[2])
        actual = int(label == 2)
        result.update(
            direction_brier=(up - actual) ** 2,
            direction_log_loss=-math.log(up if actual else 1 - up),
            direction_accuracy=float(int(up >= 0.5) == actual),
        )
    return result


def aggregate(values):
    keys = sorted({key for row in values for key in row})
    return {
        key: {"n": len(items), "mean": math.fsum(items) / len(items)}
        for key in keys
        if (items := [row[key] for row in values if key in row])
    }


def delay_summary(values):
    ordered = sorted(values)

    def percentile(q):
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def verify(database, *, snapshot=None, protocol=None, now=None):
    """Verify a consistent SQLite read transaction without modifying the run."""
    database = Path(database).resolve()
    require(database.is_file(), "shadow ledger is absent")
    with closing(
        sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    ) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        require(
            connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok",
            "SQLite integrity check failed",
        )
        metadata = dict(connection.execute("SELECT key,value FROM meta"))
        events = connection.execute("SELECT * FROM events ORDER BY seq")
        return _verify_events(
            metadata, events, snapshot=snapshot, protocol=protocol, now=now
        )


def _verify_events(metadata, events, *, snapshot, protocol, now):
    """Stream the consistent transaction instead of copying the entire event log."""
    config = json.loads(metadata["config"])
    require(sha(config) == metadata["config_sha256"], "configuration hash mismatch")
    require(
        all(finite(config[key]) for key in ("start_at", "end_at")),
        "invalid prospective bounds",
    )
    require(config["end_at"] > config["start_at"], "invalid prospective interval")
    deadline = config.get("issuance_deadline_seconds", 30)
    require(
        finite(deadline) and 0 < deadline < 60, "issuance deadline must precede target"
    )
    if protocol is not None:
        require(
            json.loads(Path(protocol).read_text()) == config,
            "external protocol differs from ledger configuration",
        )
    if snapshot is not None:
        require(
            hashlib.sha256(Path(snapshot).read_bytes()).hexdigest()
            == config["snapshot_sha256"],
            "external model snapshot hash mismatch",
        )
    receipts, forecasts, outcomes, missing = {}, {}, {}, {}
    forecast_events, commit_acks, active_corrections = {}, {}, {}
    latest = {(source, symbol): {} for source in SOURCES for symbol in SYMBOLS}
    source_lags = defaultdict(list)
    first_receipt_delays = defaultdict(lambda: defaultdict(list))
    scores = defaultdict(lambda: defaultdict(list))
    pairs = defaultdict(lambda: defaultdict(list))
    prev_hash = metadata["config_sha256"]
    sequence = 0
    for sequence, event in enumerate(events, 1):
        require(event["seq"] == sequence, "event sequence is not contiguous")
        payload = json.loads(event["payload"])
        require(
            canonical(payload) == event["payload"], "event payload is not canonical"
        )
        require(event["prev_hash"] == prev_hash, "event chain predecessor mismatch")
        envelope = {
            "seq": sequence,
            "kind": event["kind"],
            "received_ns": event["received_ns"],
            "payload": payload,
            "prev_hash": prev_hash,
        }
        require(sha(envelope) == event["digest"], "event chain digest mismatch")
        prev_hash = event["digest"]
        received = event["received_ns"] / 1_000_000_000
        require(
            abs(received - event["received_at"]) <= 1e-6, "event receive times disagree"
        )
        require(received >= config["start_at"], "event precedes prospective start")
        require(
            event["anchor"] == payload.get("anchor"),
            "indexed anchor differs from payload",
        )
        require(
            event["symbol"] == payload.get("symbol"),
            "indexed symbol differs from payload",
        )
        kind = event["kind"]
        if kind == "receipt":
            times = source_rows(payload, received, config)
            receipts[sequence] = (payload, received, times)
            source_lags[payload["source"]].extend(received - t for t in times)
            for stamp, row in zip(times, payload["body"]):
                if stamp not in latest[payload["source"], payload["symbol"]]:
                    content_end = stamp + (
                        60 if payload["source"] == "spot_klines" else 0
                    )
                    first_receipt_delays[payload["source"]][payload["symbol"]].append(
                        received - content_end
                    )
                latest[payload["source"], payload["symbol"]][stamp] = (sequence, row)
        elif kind in ("forecast", "forecast_commit", "missing", "outcome"):
            anchor, symbol = payload["anchor"], payload["symbol"]
            key = (anchor, symbol)
            require(
                isinstance(anchor, int) and anchor % 300 == 0, "invalid forecast grid"
            )
            require(symbol in SYMBOLS, "invalid forecast symbol")
            require(
                config["start_at"] <= anchor and anchor + 360 <= config["end_at"],
                "target outside prospective window",
            )
            if kind == "forecast_commit":
                require(
                    key in forecasts and key not in commit_acks,
                    "commit acknowledgment has no unique earlier forecast",
                )
                original_seq, original_hash, original_received = forecast_events[key]
                require(
                    payload.get("forecast_seq") == original_seq,
                    "commit acknowledgment references wrong forecast sequence",
                )
                require(
                    payload.get("forecast_hash") == original_hash,
                    "commit acknowledgment references wrong forecast hash",
                )
                committed = payload.get("committed_at")
                require(
                    finite(committed) and original_received <= committed <= received,
                    "invalid post-commit acknowledgment time",
                )
                require(
                    type(payload.get("eligible")) is bool,
                    "commit acknowledgment needs explicit eligibility",
                )
                require(
                    type(payload.get("clock_healthy")) is bool,
                    "commit acknowledgment needs explicit clock health",
                )
                require(
                    payload["eligible"]
                    is (committed <= anchor + deadline and payload["clock_healthy"]),
                    "commit acknowledgment eligibility disagrees with deadline or clock health",
                )
                commit_acks[key] = payload
                continue
            if kind == "missing":
                require(
                    key not in forecasts and key not in missing,
                    "duplicate or contradictory missing forecast",
                )
                require(bool(payload.get("reason")), "missing forecast has no reason")
                missing[key] = payload
                continue
            if kind == "forecast":
                require(
                    key not in forecasts and key not in missing,
                    "duplicate or contradictory forecast",
                )
                issued = payload["issued_at"]
                require(
                    finite(issued) and anchor <= issued <= anchor + deadline,
                    "forecast issued outside its deadline",
                )
                require(
                    issued <= received < anchor + 60,
                    "forecast recorded after target begins",
                )
                require(
                    payload["snapshot_sha256"] == config["snapshot_sha256"],
                    "forecast snapshot changed",
                )
                models = payload["models"]
                require(set(models) == set(ARMS), "forecast arms changed")
                for probabilities in models.values():
                    require(
                        isinstance(probabilities, list) and len(probabilities) == 3,
                        "invalid probability vector",
                    )
                    require(
                        all(finite(p) and 0 < p < 1 for p in probabilities),
                        "non-interior probability",
                    )
                    require(
                        abs(math.fsum(probabilities) - 1) <= 1e-9,
                        "probabilities do not sum to one",
                    )
                inputs = payload["input_ids"]
                require(
                    isinstance(inputs, list)
                    and inputs
                    and len(inputs) == len(set(inputs)),
                    "forecast needs unique source receipt references",
                )
                seen_symbols = set()
                for source_id in inputs:
                    require(
                        source_id in receipts,
                        "forecast source receipt does not precede it",
                    )
                    source, source_received, times = receipts[source_id]
                    require(
                        source_received <= issued,
                        "forecast used data received after issuance",
                    )
                    require(
                        all(t <= anchor for t in times),
                        "forecast referenced future source labels",
                    )
                    if source["source"] == "spot_klines":
                        require(
                            all(t + 60 <= anchor for t in times),
                            "forecast referenced unclosed anchor candle",
                        )
                        seen_symbols.add(source["symbol"])
                require(
                    seen_symbols == set(SYMBOLS),
                    "cross-asset forecast lacks spot source references",
                )
                expected_ids = set()
                complete_metrics = True
                for asset in SYMBOLS:
                    candles = latest["spot_klines", asset]
                    for stamp in range(anchor - 3780, anchor, 60):
                        require(
                            stamp in candles,
                            "forecast lacks complete 63-minute spot context",
                        )
                        expected_ids.add(candles[stamp][0])
                    oi = latest["open_interest", asset]
                    ratios = latest["position_ratio", asset]
                    for stamp in range(anchor - 4500, anchor - 899, 300):
                        if stamp not in oi or stamp not in ratios:
                            complete_metrics = False
                        else:
                            expected_ids.update((oi[stamp][0], ratios[stamp][0]))
                require(
                    set(inputs) == expected_ids,
                    "forecast did not cite exact latest-received feature context",
                )
                if not complete_metrics:
                    require(
                        models["baseline"]
                        == models["price_control"]
                        == models["positions"],
                        "incomplete metric context did not trigger common fallback",
                    )
                diagnostics = payload.get("diagnostics", {})
                if "correction_available" in diagnostics:
                    require(
                        diagnostics["correction_available"] is complete_metrics,
                        "correction availability disagrees with observed context",
                    )
                forecasts[key] = payload
                forecast_events[key] = (sequence, event["digest"], received)
                active_corrections[key] = complete_metrics
                continue
            require(
                key in forecasts and key not in outcomes,
                "outcome has no unique earlier forecast",
            )
            require(
                key in commit_acks and commit_acks[key]["eligible"],
                "outcome lacks an eligible post-commit forecast acknowledgment",
            )
            require(received >= anchor + 360, "outcome recorded before target matures")
            require(
                payload["return_start"] == anchor + 60
                and payload["return_end"] == anchor + 360,
                "outcome horizon changed",
            )
            entry, exit_price = (
                price(payload["entry_price"]),
                price(payload["exit_price"]),
            )
            expected_label = 2 if exit_price > entry else 0 if exit_price < entry else 1
            require(
                payload["label"] == expected_label,
                "outcome direction differs from exact price strings",
            )
            expected_return = math.log(float(exit_price / entry))
            require(
                finite(payload["return"])
                and math.isclose(
                    payload["return"], expected_return, abs_tol=1e-12, rel_tol=1e-10
                ),
                "outcome log return mismatch",
            )
            input_ids = payload.get(
                "input_ids",
                [payload.get("entry_receipt_id"), payload.get("exit_receipt_id")],
            )
            require(
                isinstance(input_ids, list) and input_ids,
                "outcome lacks source receipts",
            )
            matching = {anchor: [], anchor + 300: []}
            for source_id in input_ids:
                require(source_id in receipts, "outcome receipt does not precede it")
                source, source_received, _ = receipts[source_id]
                require(source_received <= received, "outcome used future receipt")
                if source["source"] == "spot_klines" and source["symbol"] == symbol:
                    for row in source["body"]:
                        minute = row[0] // 1000
                        if minute in matching:
                            matching[minute].append((source_id, row[4]))
            require(
                any(Decimal(p) == entry for _, p in matching[anchor]),
                "entry price absent from cited candle",
            )
            require(
                any(Decimal(p) == exit_price for _, p in matching[anchor + 300]),
                "exit price absent from cited candle",
            )
            for field, minute, expected in (
                ("entry_receipt_id", anchor, entry),
                ("exit_receipt_id", anchor + 300, exit_price),
            ):
                if field in payload:
                    require(
                        any(
                            identifier == payload[field] and Decimal(p) == expected
                            for identifier, p in matching[minute]
                        ),
                        "explicit outcome receipt is wrong",
                    )
            outcomes[key] = payload
            row_losses = {
                arm: losses(forecasts[key]["models"][arm], expected_label)
                for arm in ARMS
            }
            for arm in ARMS:
                scores[symbol][arm].append(row_losses[arm])
            for candidate, control in (
                ("positions", "baseline"),
                ("positions", "price_control"),
                ("price_control", "baseline"),
            ):
                pairs[symbol][f"{candidate}_vs_{control}"].append(
                    {
                        name: (row_losses[candidate][name] - row_losses[control][name])
                        if name == "direction_accuracy"
                        else (row_losses[control][name] - row_losses[candidate][name])
                        for name in row_losses[candidate]
                    }
                )
    now = time.time() if now is None else now
    require(finite(now), "invalid audit time")
    last_due = min(
        math.floor((now - deadline) / 300) * 300,
        math.floor((config["end_at"] - 360) / 300) * 300,
    )
    first_anchor = math.ceil(config["start_at"] / 300) * 300
    expected_keys = {
        (anchor, symbol)
        for anchor in range(first_anchor, last_due + 1, 300)
        for symbol in SYMBOLS
    }
    unaccounted = sorted(expected_keys - forecasts.keys() - missing.keys())
    eligible = {key for key, ack in commit_acks.items() if ack["eligible"]}
    unacknowledged = sorted(forecasts.keys() - commit_acks.keys())
    late_commits = sorted(
        key
        for key, ack in commit_acks.items()
        if ack["committed_at"] > key[0] + deadline
    )
    unhealthy_commits = sorted(
        key for key, ack in commit_acks.items() if not ack["clock_healthy"]
    )
    ineligible_commits = sorted(
        key for key, ack in commit_acks.items() if not ack["eligible"]
    )
    matured_pending = sorted(
        key for key in eligible - outcomes.keys() if key[0] + 360 <= now
    )
    accepted_active_outcomes = {
        symbol: sum(key[1] == symbol and active_corrections[key] for key in outcomes)
        for symbol in SYMBOLS
    }
    return {
        "verified": True,
        "audit_scope": "Independent read-only ledger integrity, receipt/issuance/outcome causality and score arithmetic; not a replay of model inference or external timestamp attestation.",
        "as_of": now,
        "event_count": sequence,
        "chain_tip": prev_hash,
        "protocol_sha256": metadata["config_sha256"],
        "snapshot_sha256": config["snapshot_sha256"],
        "external_protocol_verified": protocol is not None,
        "external_snapshot_verified": snapshot is not None,
        "receipt_count": len(receipts),
        "forecast_count": len(forecasts),
        "eligible_forecast_count": len(eligible),
        "commit_ack_count": len(commit_acks),
        "require_commit_ack": config.get("require_commit_ack", False),
        "unacknowledged_forecasts": [list(key) for key in unacknowledged],
        "late_commit_forecasts": [list(key) for key in late_commits],
        "clock_unhealthy_forecasts": [list(key) for key in unhealthy_commits],
        "ineligible_commit_forecasts": [list(key) for key in ineligible_commits],
        "outcome_count": len(outcomes),
        "missing_count": len(missing),
        "missing_reasons": dict(Counter(row["reason"] for row in missing.values())),
        "unaccounted_anchor_symbols": [list(key) for key in unaccounted],
        "matured_pending_outcomes": [list(key) for key in matured_pending],
        "accepted_active_outcomes_by_symbol": accepted_active_outcomes,
        "technical_pipeline_observed": all(accepted_active_outcomes.values()),
        "all_market_receipts_inside_prospective_window": True,
        "statistical_superiority_verified": False,
        "source_age_seconds": {
            source: {
                "rows": len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for source, values in source_lags.items()
            if values
        },
        "source_first_receipt_delay_note": "First observation by this client relative to spot candle close or metrics label. Includes polling/outage delays; does not identify the exchange's exact publication time. Repeated responses and revisions are excluded.",
        "source_first_receipt_delay_seconds": {
            source: {
                symbol: delay_summary(values)
                for symbol, values in symbols.items()
                if values
            }
            for source, symbols in first_receipt_delays.items()
        },
        "metrics": {
            symbol: {arm: aggregate(rows) for arm, rows in arms.items()}
            for symbol, arms in scores.items()
        },
        "paired_mean_gains": {
            symbol: {pair: aggregate(rows) for pair, rows in comparisons.items()}
            for symbol, comparisons in pairs.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.database, snapshot=args.snapshot, protocol=args.protocol)
    output = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
