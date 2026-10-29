#!/usr/bin/env python3
"""Independently check the earliest complete three-asset prospective cycle.

Only the selected run directory is read. No historical readers, production
inference functions, ledger writer or general auditor are imported.
"""

import argparse
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MODELS = ("baseline", "price_control", "positions")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def arithmetic(probabilities, label):
    require(
        len(probabilities) == 3
        and all(
            isinstance(p, (int, float))
            and not isinstance(p, bool)
            and math.isfinite(p)
            and 0 < p < 1
            for p in probabilities
        ),
        "invalid probabilities",
    )
    require(
        math.isclose(math.fsum(probabilities), 1, abs_tol=1e-10, rel_tol=0),
        "probabilities do not sum to one",
    )
    result = {
        "brier": math.fsum((probabilities[i] - (i == label)) ** 2 for i in range(3)),
        "log_loss": -math.log(probabilities[label]),
    }
    if label != 1:
        q = probabilities[2] / (probabilities[0] + probabilities[2])
        actual = int(label == 2)
        result.update(
            direction_brier=(q - actual) ** 2,
            direction_log_loss=-math.log(q if actual else 1 - q),
            direction_accuracy=float((q >= 0.5) == bool(actual)),
        )
    return result


def raw_rows(payload, received, config):
    source, symbol = payload["source"], payload["symbol"]
    require(
        source in ("spot_klines", "open_interest", "position_ratio")
        and symbol in SYMBOLS,
        "unregistered source",
    )
    require(payload["body_sha256"] == sha(payload["body"]), "response hash differs")
    params = payload["request_params"]
    require(
        params["symbol"] == symbol
        and params["startTime"] >= config["start_at"] * 1000
        and params["endTime"] < config["end_at"] * 1000,
        "request crosses prospective boundary",
    )
    require(payload["requested_at"] <= received, "request after receipt")
    stamps = []
    for row in payload["body"]:
        stamp = row[0] / 1000 if source == "spot_klines" else row["timestamp"] / 1000
        require(
            config["start_at"] <= stamp < config["end_at"]
            and params["startTime"] <= stamp * 1000 <= params["endTime"],
            "raw timestamp outside registered interval",
        )
        require(stamp <= received, "future raw timestamp")
        if source == "spot_klines":
            require(
                len(row) == 12
                and row[6] == row[0] + 59999
                and stamp % 60 == 0
                and stamp + 60 <= received,
                "unclosed or malformed candle",
            )
        else:
            require(
                stamp % 300 == 0 and row["symbol"] == symbol, "malformed metric label"
            )
            value = Decimal(
                row[
                    "sumOpenInterest" if source == "open_interest" else "longShortRatio"
                ]
            )
            require(value.is_finite() and value > 0, "invalid stock value")
        stamps.append(int(stamp))
    require(
        stamps == payload["content_times"] and len(stamps) == len(set(stamps)),
        "declared content timestamps differ",
    )
    return stamps


def verify(
    run_dir, *, database=None, general_audit=None, pre_target_proof=None, now=None
):
    root = Path(run_dir).resolve()
    database = Path(database).resolve() if database else root / "events.sqlite"
    config = json.loads((root / "config.json").read_text())
    require(
        config.get("require_commit_ack") is True,
        "post-commit acknowledgement must be required",
    )
    require(
        hashlib.sha256((root / "model.json").read_bytes()).hexdigest()
        == config["snapshot_sha256"],
        "snapshot hash differs",
    )
    general = json.loads(Path(general_audit).read_text()) if general_audit else None
    if general:
        require(
            general.get("verified") is True
            and general["protocol_sha256"] == sha(config)
            and general["snapshot_sha256"] == config["snapshot_sha256"],
            "general audit belongs to another or unverified run",
        )
    maximum = general["event_count"] if general else 2**63 - 1
    events = []
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        metadata = dict(conn.execute("SELECT key,value FROM meta"))
        require(
            json.loads(metadata["config"]) == config
            and metadata["config_sha256"] == sha(config),
            "external configuration differs",
        )
        for raw in conn.execute(
            "SELECT * FROM events WHERE seq<=? ORDER BY seq", (maximum,)
        ):
            event = dict(raw)
            event["payload"] = json.loads(event["payload"])
            events.append(event)
        conn.rollback()
    now = time.time() if now is None else now
    if general:
        require(len(events) == maximum, "general audit prefix is unavailable")
    receipts, forecasts, acknowledgements, outcomes = {}, {}, {}, {}
    latest = defaultdict(dict)
    complete = {}
    previous = sha(config)
    for seq, event in enumerate(events, 1):
        p, received = event["payload"], event["received_ns"] / 1e9
        require(
            event["seq"] == seq and event["prev_hash"] == previous,
            "broken sequence or hash chain",
        )
        envelope = {
            "seq": seq,
            "kind": event["kind"],
            "received_ns": event["received_ns"],
            "payload": p,
            "prev_hash": previous,
        }
        require(event["digest"] == sha(envelope), "event digest differs")
        require(
            abs(received - event["received_at"]) <= 1e-6
            and config["start_at"] <= received <= now,
            "invalid observed event time",
        )
        require(
            event["anchor"] == p.get("anchor") and event["symbol"] == p.get("symbol"),
            "event index differs",
        )
        previous = event["digest"]
        if event["kind"] == "receipt":
            stamps = raw_rows(p, received, config)
            receipts[seq] = event
            for stamp, row in zip(stamps, p["body"]):
                latest[p["source"], p["symbol"]][stamp] = (seq, row)
        elif event["kind"] in ("forecast", "forecast_commit", "outcome"):
            key = (p["anchor"], p["symbol"])
            anchor, symbol = key
            require(
                symbol in SYMBOLS
                and anchor % 300 == 0
                and config["start_at"] <= anchor
                and anchor + 360 <= config["end_at"],
                "unregistered forecast target",
            )
            if event["kind"] == "forecast":
                require(
                    key not in forecasts
                    and anchor
                    <= p["issued_at"]
                    <= received
                    <= anchor + config["issuance_deadline_seconds"]
                    < anchor + 60,
                    "late or duplicate forecast",
                )
                require(
                    p["snapshot_sha256"] == config["snapshot_sha256"]
                    and set(p["models"]) == set(MODELS),
                    "forecast snapshot or arms differ",
                )
                expected, all_metrics = set(), True
                for asset in SYMBOLS:
                    for stamp in range(anchor - 3780, anchor, 60):
                        require(
                            stamp in latest["spot_klines", asset],
                            "incomplete minute context",
                        )
                        expected.add(latest["spot_klines", asset][stamp][0])
                    for stamp in range(anchor - 4500, anchor - 899, 300):
                        if (
                            stamp in latest["open_interest", asset]
                            and stamp in latest["position_ratio", asset]
                        ):
                            expected.update(
                                (
                                    latest["open_interest", asset][stamp][0],
                                    latest["position_ratio", asset][stamp][0],
                                )
                            )
                        else:
                            all_metrics = False
                require(
                    len(p["input_ids"]) == len(set(p["input_ids"]))
                    and set(p["input_ids"]) == expected,
                    "input receipt versions differ",
                )
                for identifier in p["input_ids"]:
                    source = receipts[identifier]
                    require(
                        source["received_at"] <= p["issued_at"],
                        "input received after forecast",
                    )
                    require(
                        all(
                            t
                            + (
                                60
                                if source["payload"]["source"] == "spot_klines"
                                else 0
                            )
                            <= anchor
                            for t in source["payload"]["content_times"]
                        ),
                        "forecast cites future content",
                    )
                require(
                    p["diagnostics"]["correction_available"] is all_metrics,
                    "correction availability differs",
                )
                if not all_metrics:
                    require(
                        p["models"]["baseline"]
                        == p["models"]["price_control"]
                        == p["models"]["positions"],
                        "missing common fallback",
                    )
                for probabilities in p["models"].values():
                    arithmetic(probabilities, 1)
                complete[key] = all_metrics
                forecasts[key] = event
            elif event["kind"] == "forecast_commit":
                require(
                    key in forecasts and key not in acknowledgements,
                    "orphan or duplicate acknowledgement",
                )
                forecast = forecasts[key]
                require(
                    p["forecast_seq"] == forecast["seq"]
                    and p["forecast_hash"] == forecast["digest"],
                    "acknowledgement forecast differs",
                )
                require(
                    forecast["received_at"] <= p["committed_at"] <= received,
                    "acknowledgement before durable forecast",
                )
                require(
                    type(p["clock_healthy"]) is bool
                    and p["eligible"]
                    is (
                        p["clock_healthy"]
                        and p["committed_at"]
                        <= anchor + config["issuance_deadline_seconds"]
                    ),
                    "acknowledgement eligibility differs",
                )
                acknowledgements[key] = event
            else:
                require(
                    key in forecasts
                    and key in acknowledgements
                    and acknowledgements[key]["payload"]["eligible"] is True
                    and key not in outcomes,
                    "outcome lacks eligible prior forecast",
                )
                require(
                    received >= anchor + 360
                    and p["return_start"] == anchor + 60
                    and p["return_end"] == anchor + 360,
                    "premature or changed target",
                )
                values = []
                for field, stamp, price_field in (
                    ("entry_receipt_id", anchor, "entry_price"),
                    ("exit_receipt_id", anchor + 300, "exit_price"),
                ):
                    source = receipts[p[field]]
                    require(
                        p[field] in p["input_ids"]
                        and source["seq"] < seq
                        and source["received_at"] <= received
                        and source["payload"]["source"] == "spot_klines"
                        and source["payload"]["symbol"] == symbol,
                        "outcome source receipt differs",
                    )
                    raw = [
                        row
                        for row in source["payload"]["body"]
                        if row[0] == stamp * 1000
                    ]
                    require(
                        len(raw) == 1
                        and isinstance(p[price_field], str)
                        and p[price_field] == raw[0][4],
                        "outcome does not retain cited raw price",
                    )
                    value = Decimal(p[price_field])
                    require(
                        value.is_finite() and value > 0, "invalid decimal outcome price"
                    )
                    values.append(value)
                entry, exit_price = values
                label = 2 if exit_price > entry else 0 if exit_price < entry else 1
                require(
                    type(p["label"]) is int and p["label"] == label,
                    "incorrect exact price direction",
                )
                require(
                    math.isclose(
                        p["return"],
                        math.log(float(exit_price / entry)),
                        abs_tol=1e-12,
                        rel_tol=1e-10,
                    ),
                    "incorrect log return",
                )
                outcomes[key] = event
    if general:
        require(previous == general["chain_tip"], "general audit chain prefix differs")
    aggregates = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key, event in outcomes.items():
        for arm in MODELS:
            for metric, value in arithmetic(
                forecasts[key]["payload"]["models"][arm], event["payload"]["label"]
            ).items():
                aggregates[key[1]][arm][metric].append(value)
    aggregate_rows = {
        symbol: {
            arm: {
                metric: {"n": len(values), "mean": math.fsum(values) / len(values)}
                for metric, values in metrics.items()
            }
            for arm, metrics in arms.items()
        }
        for symbol, arms in aggregates.items()
    }
    if general:
        require(
            set(aggregate_rows) == set(general["metrics"]),
            "general metric symbols differ",
        )
        for symbol, arms in aggregate_rows.items():
            require(
                set(arms) == set(general["metrics"][symbol]),
                "general metric arms differ",
            )
            for arm, metrics in arms.items():
                other = general["metrics"][symbol][arm]
                require(set(metrics) == set(other), "general metric fields differ")
                for metric, value in metrics.items():
                    require(
                        value["n"] == other[metric]["n"]
                        and math.isclose(
                            value["mean"],
                            other[metric]["mean"],
                            abs_tol=1e-12,
                            rel_tol=1e-10,
                        ),
                        "general metric arithmetic differs",
                    )
    candidates = sorted(
        {
            anchor
            for anchor, _ in forecasts
            if all(complete.get((anchor, symbol), False) for symbol in SYMBOLS)
        }
    )
    ready = [
        anchor
        for anchor in candidates
        if all((anchor, symbol) in outcomes for symbol in SYMBOLS)
    ]
    result = {
        "status": "verified" if ready else "pending",
        "scope": "First complete real cycle chronology and independent score arithmetic; no inference replay or skill claim",
        "as_of": now,
        "event_count": len(events),
        "chain_tip": previous,
        "snapshot_sha256": config["snapshot_sha256"],
        "protocol_sha256": sha(config),
        "general_audit_compared": general is not None,
        "forecast_count": len(forecasts),
        "outcome_count": len(outcomes),
        "full_context_anchors": candidates,
        "aggregate_metrics": aggregate_rows,
        "statistical_superiority_verified": False,
        "pre_target_proof_verified": False,
        "database": str(database),
    }
    if not ready:
        result["reason"] = (
            "No common three-asset anchor has full correction context, eligible post-commit acknowledgements and all matured outcomes"
        )
        return result
    anchor = ready[0]
    if pre_target_proof:
        proof = json.loads(Path(pre_target_proof).read_text())
        captured = proof["captured_at"]
        require(
            isinstance(captured, (int, float))
            and math.isfinite(captured)
            and anchor <= captured < anchor + 60,
            "proof was not captured before target entry",
        )
        require(
            proof["anchor"] == anchor
            and proof["protocol_sha256"] == sha(config)
            and proof["snapshot_sha256"] == config["snapshot_sha256"],
            "proof run or anchor differs",
        )
        for field, collection in (
            ("forecasts", forecasts),
            ("commit_acks", acknowledgements),
        ):
            require(set(proof[field]) == set(SYMBOLS), "proof lacks all three assets")
            for symbol in SYMBOLS:
                actual = collection[anchor, symbol]
                require(
                    actual["received_at"] <= captured,
                    "proof precedes its captured event",
                )
                require(
                    proof[field][symbol]
                    == {key: actual[key] for key in ("seq", "digest", "payload")},
                    "pre-target event changed after capture",
                )
        if "ledger_head" in proof:
            head = proof["ledger_head"]
            require(
                1 <= head["seq"] <= len(events)
                and events[head["seq"] - 1]["digest"] == head["digest"],
                "pre-target ledger head differs",
            )
            require(
                all(
                    acknowledgements[anchor, symbol]["seq"]
                    <= head["seq"]
                    < outcomes[anchor, symbol]["seq"]
                    for symbol in SYMBOLS
                ),
                "proof ledger head does not precede the outcomes",
            )
        require(
            proof.get("outcome_count", 0) == 0,
            "proof already contained target outcomes",
        )
        result["pre_target_proof_verified"] = True
        result["pre_target_proof_captured_at"] = captured
        result["pre_target_proof_sha256"] = hashlib.sha256(
            Path(pre_target_proof).read_bytes()
        ).hexdigest()
    symbols = {}
    for symbol in SYMBOLS:
        key = (anchor, symbol)
        forecast, ack, outcome = forecasts[key], acknowledgements[key], outcomes[key]
        fp, ap, op = forecast["payload"], ack["payload"], outcome["payload"]
        symbols[symbol] = {
            "forecast_seq": forecast["seq"],
            "forecast_hash": forecast["digest"],
            "forecast_issued_at": fp["issued_at"],
            "forecast_issued_utc": utc(fp["issued_at"]),
            "commit_observed_at": ap["committed_at"],
            "commit_observed_utc": utc(ap["committed_at"]),
            "commit_ack_seq": ack["seq"],
            "input_last_received_at": max(
                receipts[i]["received_at"] for i in fp["input_ids"]
            ),
            "seconds_before_target_entry": anchor + 60 - ap["committed_at"],
            "outcome_seq": outcome["seq"],
            "outcome_received_at": outcome["received_at"],
            "outcome_received_utc": utc(outcome["received_at"]),
            "entry_price": op["entry_price"],
            "exit_price": op["exit_price"],
            "label": op["label"],
            "models": {
                arm: {
                    "probabilities_down_flat_up": fp["models"][arm],
                    "metrics": arithmetic(fp["models"][arm], op["label"]),
                }
                for arm in MODELS
            },
        }
    result["first_cycle"] = {
        "anchor": anchor,
        "anchor_utc": utc(anchor),
        "return_start": anchor + 60,
        "return_start_utc": utc(anchor + 60),
        "return_end": anchor + 360,
        "return_end_utc": utc(anchor + 360),
        "symbols": symbols,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--general-audit", type=Path)
    parser.add_argument("--pre-target-proof", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.run_dir,
        database=args.database,
        general_audit=args.general_audit,
        pre_target_proof=args.pre_target_proof,
    )
    with args.output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "event_count",
                    "forecast_count",
                    "outcome_count",
                    "general_audit_compared",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
