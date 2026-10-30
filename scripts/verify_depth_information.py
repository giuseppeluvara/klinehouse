#!/usr/bin/env python3
"""Independently audit the separate prospective QI20 information experiment.

No runner, model core, production scoring, network client or historical market
reader is imported. Both source journals are read in coherent read transactions;
saved prefix hashes bind earlier model/forecast references to those snapshots.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
CONTROLS = ("baseline", "price_control", "positions")
ARMS = (*CONTROLS, "depth")
PILOT_START = 1793272200
RIDGE = 0.25
LIMITS = dict(
    deadline=30,
    clock_age=600,
    terminal_clock_age=1800,
    clock_tolerance=0.25,
    min_free_bytes=40 * 1024**3,
    max_bytes=128 * 1024**2,
    reserve_bytes=4 * 1024**2,
)


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def interior(probability):
    require(
        isinstance(probability, list)
        and len(probability) == 3
        and all(finite(p) and 0 < p < 1 for p in probability)
        and math.isclose(math.fsum(probability), 1, rel_tol=0, abs_tol=1e-9),
        "invalid three-class probabilities",
    )
    return probability


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def decimal_price(value):
    require(isinstance(value, str), "source price must retain its string")
    number = Decimal(value)
    require(number.is_finite() and number > 0, "invalid source price")
    return number


def source_snapshot(root, kind):
    """Verify complete source hash chains; never open a historical store."""
    root = Path(root).resolve()
    require(
        (kind == "pilot" and root.name == "shadow-pilot-20261029-v2")
        or (kind == "depth" and root.name.startswith("depth-research-")),
        "only named prospective sources are allowed",
    )
    require(
        not any(part in ("archive", "holdout", "holdouts") for part in root.parts),
        "historical source directory refused",
    )
    config = load(root / "config.json")
    if kind == "pilot":
        require(
            config.get("schema") == "klinehouse-shadow-pilot-v2"
            and config.get("start_at") == PILOT_START
            and config.get("require_commit_ack") is True
            and config.get("orders") is False,
            "unauthorized original pilot configuration",
        )
        require(
            file_sha(root / "model.json") == config["snapshot_sha256"],
            "original snapshot hash differs",
        )
    else:
        require(
            config.get("schema") == "klinehouse-depth-research-v1"
            and config.get("activation_at", 0) >= PILOT_START
            and config.get("capture_only") is True
            and config.get("candidate_predictions") is False
            and config.get("orders") is False,
            "unauthorized prospective depth configuration",
        )
        require(
            load(root / "config-sha256.json")["sha256"] == sha(config),
            "depth configuration hash differs",
        )
    require(config["symbols"] == list(SYMBOLS), "source symbols differ")
    path = root / ("events.sqlite" if kind == "pilot" else "depth-events.sqlite")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        require(
            connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok",
            "source SQLite integrity failed",
        )
        require(
            dict(connection.execute("SELECT key,value FROM meta"))
            == {"config": canonical(config), "config_sha256": sha(config)},
            "source configuration differs from ledger seed",
        )
        previous = sha(config)
        events, kinds = {}, defaultdict(dict)
        for number, raw in enumerate(
            connection.execute("SELECT * FROM events ORDER BY seq"), 1
        ):
            row = dict(raw)
            payload = json.loads(row["payload"])
            require(
                row["seq"] == number
                and row["prev_hash"] == previous
                and canonical(payload) == row["payload"],
                "source sequence or canonical payload differs",
            )
            if kind == "pilot":
                stamp = row["received_ns"] / 1e9
                require(
                    abs(stamp - row["received_at"]) <= 1e-6 and stamp >= PILOT_START,
                    "original receipt timestamp differs",
                )
                envelope = dict(
                    seq=number,
                    kind=row["kind"],
                    received_ns=row["received_ns"],
                    payload=payload,
                    prev_hash=previous,
                )
            else:
                stamp = row["recorded_at"]
                require(
                    stamp >= config["activation_at"], "depth event predates activation"
                )
                envelope = {
                    name: row[name]
                    for name in (
                        "seq",
                        "kind",
                        "recorded_at",
                        "anchor",
                        "symbol",
                        "prev_hash",
                    )
                }
                envelope["payload"] = payload
            require(
                finite(stamp) and sha(envelope) == row["digest"],
                "source event hash differs",
            )
            require(
                row["anchor"] == payload.get("anchor", row["anchor"])
                and row["symbol"] == payload.get("symbol", row["symbol"]),
                "source indexed membership differs",
            )
            event = dict(
                seq=number,
                digest=row["digest"],
                kind=row["kind"],
                received_at=stamp,
                anchor=row["anchor"],
                symbol=row["symbol"],
                payload=payload,
            )
            events[number] = event
            unique_kinds = (
                "forecast",
                "forecast_commit",
                "outcome",
                "missing",
                "book_receipt",
            )
            key = (
                (row["anchor"], row["symbol"])
                if row["kind"] in unique_kinds
                else (number, None)
            )
            require(key not in kinds[row["kind"]], "duplicate source key")
            kinds[row["kind"]][key] = event
            previous = row["digest"]
        connection.rollback()
    return dict(
        root=root,
        kind=kind,
        config=config,
        events=events,
        kinds=kinds,
        event_count=len(events),
        chain_tip=previous,
    )


def get(source, kind, anchor, symbol):
    return source["kinds"].get(kind, {}).get((anchor, symbol))


def ref(row):
    return {name: row[name] for name in ("seq", "digest", "received_at")}


def prefix(source, proof, known_at=None):
    require(
        Path(proof["root"]).resolve() == source["root"], "source prefix root differs"
    )
    count = proof["event_count"]
    require(
        type(count) is int and 0 <= count <= source["event_count"],
        "source prefix unavailable",
    )
    expected = source["events"][count]["digest"] if count else sha(source["config"])
    require(proof["chain_tip"] == expected, "source prefix hash differs")
    if known_at is not None:
        require(
            all(
                e["received_at"] <= known_at
                for seq, e in source["events"].items()
                if seq <= count
            ),
            "source prefix contains later observations",
        )
    return count


def validate_spot_receipt(source, event):
    payload = event["payload"]
    require(
        event["kind"] == "receipt"
        and payload["source"] == "spot_klines"
        and payload["symbol"] in SYMBOLS,
        "wrong outcome source receipt",
    )
    body = payload["body"]
    require(
        payload["body_sha256"] == sha(body), "original raw candle body hash differs"
    )
    params = payload["request_params"]
    require(
        params["symbol"] == payload["symbol"]
        and params["startTime"] >= PILOT_START * 1000
        and params["endTime"] < source["config"]["end_at"] * 1000,
        "original candle request crosses prospective boundary",
    )
    times = []
    for row in body:
        require(
            isinstance(row, list)
            and len(row) == 12
            and row[0] % 60000 == 0
            and row[6] == row[0] + 59999,
            "invalid original candle shape",
        )
        stamp = row[0] // 1000
        require(
            PILOT_START <= stamp
            and stamp + 60 <= source["config"]["end_at"]
            and stamp + 60 <= event["received_at"] + 2
            and params["startTime"] <= row[0] <= params["endTime"],
            "original candle time is not prospective/closed",
        )
        for i in (1, 2, 3, 4):
            decimal_price(row[i])
        times.append(stamp)
    require(
        times == payload["content_times"] and len(times) == len(set(times)),
        "original candle labels differ",
    )
    return body


def original(source, anchor, symbol, *, limit=None):
    event, acknowledgement = (
        get(source, "forecast", anchor, symbol),
        get(source, "forecast_commit", anchor, symbol),
    )
    if event is None or acknowledgement is None:
        return None
    if limit is not None and max(event["seq"], acknowledgement["seq"]) > limit:
        return None
    p, a = event["payload"], acknowledgement["payload"]
    require(
        a["forecast_seq"] == event["seq"] and a["forecast_hash"] == event["digest"],
        "original acknowledgement references another forecast",
    )
    if not (
        a["eligible"] is True
        and a["clock_healthy"] is True
        and anchor
        <= p["issued_at"]
        <= event["received_at"]
        <= a["committed_at"]
        <= acknowledgement["received_at"]
        <= anchor + 30
        and event["seq"] < acknowledgement["seq"]
    ):
        return None
    require(
        p["snapshot_sha256"] == source["config"]["snapshot_sha256"]
        and set(p["models"]) == set(CONTROLS),
        "original model identity differs",
    )
    for arm in CONTROLS:
        interior(p["models"][arm])
    for identifier in p.get("input_ids", []):
        require(
            identifier in source["events"]
            and identifier < event["seq"]
            and source["events"][identifier]["received_at"] <= p["issued_at"],
            "original forecast cites a later source receipt",
        )
    return dict(
        models=p["models"],
        forecast=ref(event),
        acknowledgement=ref(acknowledgement),
        available_at=acknowledgement["received_at"],
    )


def observed(source, anchor, symbol, *, limit=None):
    event = get(source, "outcome", anchor, symbol)
    if event is None or (limit is not None and event["seq"] > limit):
        return None
    p = event["payload"]
    require(
        p["return_start"] == anchor + 60
        and p["return_end"] == anchor + 360
        and event["received_at"] >= anchor + 360,
        "original outcome precedes maturity",
    )
    entry, exit_price = decimal_price(p["entry_price"]), decimal_price(p["exit_price"])
    label = 2 if exit_price > entry else 0 if exit_price < entry else 1
    require(
        type(p["label"]) is int and p["label"] == label, "exact outcome label differs"
    )
    for field, minute, value in (
        ("entry_receipt_id", anchor, entry),
        ("exit_receipt_id", anchor + 300, exit_price),
    ):
        identifier = p[field]
        require(
            identifier in source["events"] and identifier < event["seq"],
            "outcome cites missing/later receipt",
        )
        receipt = source["events"][identifier]
        require(
            receipt["received_at"] <= event["received_at"]
            and receipt["payload"]["symbol"] == symbol,
            "outcome source asset or receipt time differs",
        )
        rows = validate_spot_receipt(source, receipt)
        require(
            any(row[0] // 1000 == minute and Decimal(row[4]) == value for row in rows),
            "outcome price differs from original closed candle",
        )
    return dict(
        label=label,
        entry_price=p["entry_price"],
        exit_price=p["exit_price"],
        source=ref(event),
        received_at=event["received_at"],
    )


def clock_probe(source, row):
    p = row["payload"]
    if p.get("healthy") is not True:
        return None
    r = p["receipt"]
    require(
        r["source"] == "time"
        and r["url"] == source["config"]["urls"]["time"]
        and r["http_status"] == 200
        and json.loads(r["raw_body"]) == r["body"]
        and hashlib.sha256(r["raw_body"].encode()).hexdigest() == r["body_sha256"],
        "clock raw response differs",
    )
    server = r["body"]["serverTime"] / 1000
    bound = max(abs(server - r["requested_at"]), abs(server - r["received_at"]))
    require(
        r["requested_at"] <= r["received_at"] <= row["received_at"]
        and abs(bound - p["absolute_offset_bound_seconds"]) <= 1e-6
        and bound <= 2,
        "source clock bound differs",
    )
    return dict(
        utc=r["received_at"],
        monotonic=r["received_monotonic"],
        absolute_offset_bound_seconds=p["absolute_offset_bound_seconds"],
    )


def depth_book(source, anchor, symbol, *, limit=None):
    event = get(source, "book_receipt", anchor, symbol)
    if event is None or (limit is not None and event["seq"] > limit):
        return None
    acknowledgements = [
        e
        for e in source["kinds"].get("commit_ack", {}).values()
        if e["payload"]["record_seq"] == event["seq"]
        and (limit is None or e["seq"] <= limit)
    ]
    invalidated = any(
        e["payload"]["record_seq"] == event["seq"]
        and (limit is None or e["seq"] <= limit)
        for e in source["kinds"].get("invalidated", {}).values()
    )
    if len(acknowledgements) != 1 or invalidated:
        return None
    acknowledgement = acknowledgements[0]
    p, a = event["payload"], acknowledgement["payload"]
    r = p["receipt"]
    if not (
        a["eligible"] is True
        and a["clock_healthy"] is True
        and p["clock_healthy"] is True
        and not p["validation_errors"]
        and event["received_at"] <= a["committed_at"] <= acknowledgement["received_at"]
        and a["committed_at"] <= anchor + 30
        and source["config"]["activation_at"]
        <= r["requested_at"]
        <= r["received_at"]
        < anchor
    ):
        return None
    require(
        a["record_hash"] == event["digest"] and event["seq"] < acknowledgement["seq"],
        "depth acknowledgement hash differs",
    )
    source_reference = a.get("clock_reference")
    require(
        isinstance(source_reference, list) and len(source_reference) == 2,
        "depth commit lacks source clock reference",
    )
    probes = [
        e
        for e in source["kinds"].get("clock_probe", {}).values()
        if e["seq"] < event["seq"]
        and e["payload"].get("healthy") is True
        and [
            e["payload"]["receipt"]["received_at"],
            e["payload"]["receipt"]["received_monotonic"],
        ]
        == source_reference
    ]
    require(len(probes) == 1, "depth commit clock reference is unavailable")
    reference = clock_probe(source, probes[0])
    require(
        a.get("commit_elapsed_clock_healthy") is True
        and a.get("clock_reference_healthy") is True
        and elapsed(
            (event["received_at"], a["record_started_monotonic"]),
            (a["committed_at"], a["committed_monotonic"]),
        )
        and healthy((a["committed_at"], a["committed_monotonic"]), reference),
        "depth post-commit clock arithmetic differs",
    )
    raw, body = r["raw_body"].encode(), r["body"]
    require(
        r["source"] == "depth"
        and r["symbol"] == symbol
        and r["http_status"] == 200
        and r["url"] == source["config"]["urls"]["depth"]
        and r["params"] == {"limit": 20, "symbol": symbol}
        and hashlib.sha256(raw).hexdigest() == r["body_sha256"]
        and len(raw) == r["body_bytes"]
        and json.loads(raw) == body,
        "depth raw body or public request differs",
    )
    require(
        elapsed(
            (r["requested_at"], r["requested_monotonic"]),
            (r["received_at"], r["received_monotonic"]),
        )
        and r["received_at"] <= event["received_at"],
        "depth request clock differs",
    )
    require(
        p.get("source_event_timestamp") is None
        and len(body["bids"]) == len(body["asks"]) == 20,
        "depth source semantics or level count differs",
    )
    with localcontext() as context:
        context.prec = 60
        bid = [
            (decimal_price(price), decimal_price(qty)) for price, qty in body["bids"]
        ]
        ask = [
            (decimal_price(price), decimal_price(qty)) for price, qty in body["asks"]
        ]
        require(
            all(
                bid[i][0] > bid[i + 1][0] and ask[i][0] < ask[i + 1][0]
                for i in range(19)
            )
            and bid[0][0] < ask[0][0],
            "invalid sorted/crossed depth",
        )
        b20, a20 = sum(q for _, q in bid), sum(q for _, q in ask)
        qi = (b20 - a20) / (b20 + a20)
        require(
            abs(Decimal(p["features"]["qi20"]) - qi) < Decimal("1e-45"),
            "QI20 differs from same depth body",
        )
    require(finite(float(qi)) and abs(float(qi)) <= 1, "invalid QI20")
    return dict(
        qi20=float(p["features"]["qi20"]),
        receipt=ref(event),
        acknowledgement=ref(acknowledgement),
        available_at=acknowledgement["received_at"],
    )


def elapsed(start, end):
    return (
        all(finite(v) for v in (*start, *end))
        and end[0] >= start[0]
        and end[1] >= start[1]
        and abs((end[0] - start[0]) - (end[1] - start[1])) <= 0.25
    )


def healthy(moment, reference):
    return (
        finite(reference.get("absolute_offset_bound_seconds"))
        and reference["absolute_offset_bound_seconds"] <= 2
        and 0 <= moment[0] - reference["utc"] <= 600
        and elapsed((reference["utc"], reference["monotonic"]), moment)
    )


def verify_clock_reference(source, reference, known_at):
    matches = [
        e
        for e in source["kinds"].get("clock_probe", {}).values()
        if e["received_at"] <= known_at
        and e["payload"].get("healthy") is True
        and e["payload"]["receipt"]["received_at"] == reference.get("utc")
        and e["payload"]["receipt"]["received_monotonic"] == reference.get("monotonic")
    ]
    require(
        len(matches) == 1 and clock_probe(source, matches[0]) == reference,
        "clock reference lacks an earlier verified source probe",
    )


def expected_training(config, depth, pilot):
    data = dict(anchors=[], baseline=[], imbalance=[], labels=[], received=[])
    membership = []
    cutoff = config["training_cutoff"]
    for anchor in range(config["first_anchor"], cutoff, 300):
        if anchor + 360 > cutoff:
            continue
        entries = [
            (
                original(pilot, anchor, s),
                depth_book(depth, anchor, s),
                observed(pilot, anchor, s),
            )
            for s in SYMBOLS
        ]
        if any(
            a is None
            or b is None
            or y is None
            or max(a["available_at"], b["available_at"], y["received_at"]) > cutoff
            for a, b, y in entries
        ):
            continue
        data["anchors"].append(anchor)
        data["baseline"].append(
            [
                a["models"]["baseline"][2]
                / (a["models"]["baseline"][0] + a["models"]["baseline"][2])
                for a, _, _ in entries
            ]
        )
        data["imbalance"].append([b["qi20"] for _, b, _ in entries])
        data["labels"].append([y["label"] for _, _, y in entries])
        data["received"].append([y["received_at"] for _, _, y in entries])
        membership.append(
            dict(
                anchor=anchor,
                assets={
                    s: dict(original=a, book=b, outcome=y)
                    for s, (a, b, y) in zip(SYMBOLS, entries)
                },
            )
        )
    return data, membership


def coefficient_certificate(data, state):
    counts = [sum(row[i] != 1 for row in data["labels"]) for i in range(3)]
    require(
        state.get("schema") == "klinehouse-depth-information-one-coefficient-v1"
        and state["ridge"] == RIDGE
        and finite(state["coefficient"])
        and abs(state["coefficient"]) <= 8
        and state["training_anchors"] == len(data["anchors"])
        and state["moving_labels_per_asset"] == counts,
        "frozen model state or training counts differ",
    )
    sufficient = sum(counts) >= 300 and all(c >= 100 for c in counts)
    if not sufficient:
        require(
            state["status"] == "insufficient_training" and state["coefficient"] == 0,
            "insufficient training must freeze the neutral coefficient",
        )
        return dict(
            sufficient_training=False,
            moving_labels_per_asset=counts,
            optimal_coefficient=0,
            coefficient_error=0,
            derivative=None,
        )
    require(state["status"] == "frozen_after_first_day", "wrong fitted status")
    observations = []
    for qs, xs, ys in zip(data["baseline"], data["imbalance"], data["labels"]):
        moving = sum(y != 1 for y in ys)
        if moving:
            for q, x, y in zip(qs, xs, ys):
                if y != 1:
                    observations.append(
                        (math.log(q / (1 - q)), x, int(y == 2), 1 / moving)
                    )
    mass = math.fsum(weight for _, _, _, weight in observations)

    def gradient(beta):
        return (
            math.fsum(
                weight * x * (sigmoid(offset + beta * x) - y)
                for offset, x, y, weight in observations
            )
            / mass
            + RIDGE * beta
        )

    if gradient(-8) >= 0:
        optimum = -8.0
    elif gradient(8) <= 0:
        optimum = 8.0
    else:
        lower, upper = -8.0, 8.0
        for _ in range(90):
            midpoint = (lower + upper) / 2
            if gradient(midpoint) < 0:
                lower = midpoint
            else:
                upper = midpoint
        optimum = (lower + upper) / 2
    require(
        abs(optimum - state["coefficient"]) <= 1e-9,
        "coefficient is not the independent ridge optimum",
    )
    require(
        finite(state.get("gradient"))
        and abs(state["gradient"] - gradient(state["coefficient"])) <= 1e-10,
        "stored fit derivative differs",
    )
    return dict(
        sufficient_training=True,
        moving_labels_per_asset=counts,
        optimal_coefficient=optimum,
        coefficient_error=abs(optimum - state["coefficient"]),
        derivative=gradient(state["coefficient"]),
    )


def verify_config(root):
    config = load(root / "config.json")
    require(
        config.get("schema") == "klinehouse-depth-information-run-v1"
        and config.get("orders") is False
        and config.get("network") is False
        and config.get("original_pilot_read_only") is True
        and config["symbols"] == list(SYMBOLS)
        and config["limits"] == LIMITS
        and sha(config) == load(root / "config-sha256.json")["sha256"],
        "frozen run configuration differs",
    )
    require(
        config["code"]
        == {
            "script": file_sha(root / "runtime/scripts/run_depth_information.py"),
            "core": file_sha(root / "runtime/research/depth_information.py"),
        }
        and config["plan_sha256"] == file_sha(root / "protocol-plan.json")
        and config["source_identities_sha256"]
        == file_sha(root / "source-identities.json"),
        "frozen code, plan or source identity hash differs",
    )
    plan = load(root / "protocol-plan.json")
    require(
        plan.get("schema") == "depth-information-research-plan-v1"
        and plan["core_code"]["research/depth_information.py"] == config["code"]["core"]
        and plan["registered_at"] <= config["created_at"],
        "registered plan identity differs",
    )
    return config


def score(probability, label):
    interior(probability)
    with localcontext() as context:
        context.prec = 50
        p = [Decimal(str(v)) for v in probability]
        brier = sum((v - Decimal(i == label)) ** 2 for i, v in enumerate(p))
        loss = -p[label].ln()
    direction = (
        None
        if label == 1
        else int(
            (probability[2] / (probability[0] + probability[2]) >= 0.5) == (label == 2)
        )
    )
    return float(brier), float(loss), direction


def verify(root, *, now=None):
    root = Path(root).resolve()
    require(
        not any(
            part.startswith("shadow-pilot")
            or part in ("archive", "holdout", "holdouts")
            for part in root.parts
        ),
        "audit root must be separate from protected sources",
    )
    now = time.time() if now is None else now
    require(finite(now), "invalid audit time")
    config = verify_config(root)
    depth = source_snapshot(config["capture_root"], "depth")
    pilot = source_snapshot(config["pilot_root"], "pilot")
    identities = load(root / "source-identities.json")
    for name, source, identity_name in (
        ("capture", depth, "capture"),
        ("pilot", pilot, "pilot"),
    ):
        identity = identities[identity_name]
        require(
            Path(identity["root"]).resolve() == source["root"]
            and identity["config"] == source["config"]
            and identity["config_sha256"] == sha(source["config"]),
            "frozen source identity differs",
        )
    first, cutoff = config["first_anchor"], config["training_cutoff"]
    require(
        first == math.ceil(depth["config"]["activation_at"] / 300) * 300
        and cutoff == first + 86400
        and identities["first_anchor"] == first
        and identities["training_cutoff"] == cutoff
        and config["end_at"] == depth["config"]["end_at"] <= pilot["config"]["end_at"]
        and config["capture_config_sha256"] == sha(depth["config"])
        and config["pilot_config_sha256"] == sha(pilot["config"])
        and config["original_snapshot_sha256"] == pilot["config"]["snapshot_sha256"],
        "fixed source window differs",
    )
    validated_books = sum(
        depth_book(depth, e["anchor"], e["symbol"]) is not None
        for e in depth["kinds"].get("book_receipt", {}).values()
    )
    validated_original_outcomes = sum(
        observed(pilot, e["anchor"], e["symbol"]) is not None
        for e in pilot["kinds"].get("outcome", {}).values()
        if e["anchor"] >= first
    )
    directories = (
        "forecasts",
        "acks",
        "eligibility",
        "missing",
        "outcomes",
        "model-receipts",
    )
    paths = {name: sorted((root / name).glob("*.json")) for name in directories}
    model_path, training_path = root / "model.json", root / "training-data.json"
    if now < cutoff:
        require(
            not model_path.exists()
            and not training_path.exists()
            and not any(paths.values()),
            "model or candidate artifacts exist before training cutoff",
        )
    model = None
    model_certificate = None
    if training_path.exists():
        training = load(training_path)
        data, membership = expected_training(config, depth, pilot)
        require(
            training["fixed_cutoff"] == cutoff
            and training["config_sha256"] == sha(config)
            and training["membership"] == membership
            and training["data"] == data,
            "frozen first-day membership or data is incomplete/different",
        )
        for name, source in (("depth", depth), ("pilot", pilot)):
            bound = prefix(source, training["source_prefixes"][name])
            for entry in membership:
                for values in entry["assets"].values():
                    refs = (
                        [values["book"]["receipt"], values["book"]["acknowledgement"]]
                        if name == "depth"
                        else [
                            values["original"]["forecast"],
                            values["original"]["acknowledgement"],
                            values["outcome"]["source"],
                        ]
                    )
                    require(
                        all(reference["seq"] <= bound for reference in refs),
                        "training used observations outside frozen prefix",
                    )
        if model_path.exists():
            model = load(model_path)
            require(
                model["training_data_sha256"] == file_sha(training_path)
                and model["config_sha256"] == sha(config)
                and model["fixed_cutoff"] == cutoff
                and model["training_membership"] == membership
                and model["training_membership_sha256"] == sha(membership)
                and model["source_prefixes"] == training["source_prefixes"]
                and cutoff <= model["fitted_at"] <= now
                and model["state"]["first_anchor"] == first
                and model["state"]["cutoff"] == cutoff,
                "immutable model identity or chronology differs",
            )
            for name, source in (("depth", depth), ("pilot", pilot)):
                prefix(source, model["source_prefixes"][name], model["fitted_at"])
            model_certificate = coefficient_certificate(data, model["state"])
    else:
        require(not model_path.exists(), "model lacks frozen training-data bytes")
    require(
        model is not None or not paths["forecasts"],
        "candidate forecast exists without frozen model",
    )
    emissions, unconfirmed, ineligible, late, clock_unhealthy = {}, [], [], [], []
    fallback_book, fallback_zero = 0, 0
    for path in paths["forecasts"]:
        forecast = load(path)
        anchor, symbol = forecast["anchor"], forecast["symbol"]
        key = (anchor, symbol)
        require(
            key not in emissions
            and symbol in SYMBOLS
            and anchor % 300 == 0
            and cutoff <= anchor
            and anchor + 360 <= config["end_at"]
            and path.name == f"{anchor}-{symbol}.json",
            "candidate forecast key/window differs",
        )
        ack_path, seal_path = (
            root / "acks" / path.name,
            root / "eligibility" / path.name,
        )
        if not ack_path.exists() or not seal_path.exists():
            unconfirmed.append(list(key))
            continue
        acknowledgement, seal = load(ack_path), load(seal_path)
        digest = file_sha(path)
        require(
            acknowledgement["forecast_sha256"] == seal["forecast_sha256"] == digest
            and seal["ack_sha256"] == file_sha(ack_path),
            "candidate confirmation hash differs",
        )
        issued = (forecast["issued_at"], forecast["issued_monotonic"])
        committed = (
            acknowledgement["forecast_committed_at"],
            acknowledgement["forecast_committed_monotonic"],
        )
        confirmed = (seal["ack_committed_at"], seal["ack_committed_monotonic"])
        reference = forecast["clock_reference"]
        verify_clock_reference(depth, reference, issued[0])
        require(
            acknowledgement["clock_reference"] == reference
            and acknowledgement["deadline"] == anchor + 30,
            "candidate clock/deadline differs",
        )
        primary_clock = (
            elapsed(issued, committed)
            and healthy(issued, reference)
            and healthy(committed, reference)
        )
        primary = primary_clock and anchor <= issued[0] <= committed[0] <= anchor + 30
        confirmation_clock = elapsed(committed, confirmed) and healthy(
            confirmed, reference
        )
        eligible = primary and confirmation_clock and confirmed[0] <= anchor + 30
        require(
            type(seal["eligible"]) is bool
            and seal["eligible"] == eligible
            and acknowledgement["primary_clock_healthy"] is primary
            and seal["clock_healthy"] is confirmation_clock,
            "candidate eligibility/clock arithmetic differs",
        )
        if committed[0] > anchor + 30 or confirmed[0] > anchor + 30:
            late.append(list(key))
        if not primary_clock or not confirmation_clock:
            clock_unhealthy.append(list(key))
        if not eligible:
            ineligible.append(list(key))
            continue
        require(
            committed[0]
            <= acknowledgement["recorded_at"]
            <= confirmed[0]
            <= seal["recorded_at"]
            <= now,
            "candidate acknowledgement/seal chronology differs",
        )
        require(
            forecast["schema"] == "depth-information-forecast-v1"
            and forecast["config_sha256"] == sha(config)
            and forecast["target_start"] == anchor + 60
            and forecast["target_end"] == anchor + 360
            and set(forecast["models"]) == set(ARMS),
            "candidate target/configuration differs",
        )
        limits = {
            name: prefix(source, forecast["source_prefixes"][name], issued[0])
            for name, source in (("depth", depth), ("pilot", pilot))
        }
        controls = original(pilot, anchor, symbol, limit=limits["pilot"])
        require(
            controls is not None
            and controls == forecast["original"]
            and controls["available_at"] <= issued[0],
            "candidate original controls were not available before issue",
        )
        book = depth_book(depth, anchor, symbol, limit=limits["depth"])
        book = book if book is not None and book["available_at"] <= issued[0] else None
        require(
            forecast["depth_book"] == book
            and forecast["fallback_missing_book"] is (book is None),
            "candidate book/fallback membership differs",
        )
        beta = model["state"]["coefficient"]
        require(
            forecast["fallback_zero_coefficient"] is (beta == 0)
            and forecast["qi20"] == (None if book is None else book["qi20"]),
            "candidate QI20 or neutral fallback differs",
        )
        baseline = controls["models"]["baseline"]
        for arm in ARMS:
            interior(forecast["models"][arm])
        require(
            all(forecast["models"][arm] == controls["models"][arm] for arm in CONTROLS),
            "candidate changed a saved control",
        )
        expected = baseline
        if book is not None and beta != 0:
            q = baseline[2] / (baseline[0] + baseline[2])
            adjusted = sigmoid(math.log(q / (1 - q)) + beta * book["qi20"])
            expected = [
                (1 - baseline[1]) * (1 - adjusted),
                baseline[1],
                (1 - baseline[1]) * adjusted,
            ]
        require(
            forecast["models"]["depth"][1] == baseline[1]
            and all(
                abs(a - b) <= 1e-12
                for a, b in zip(forecast["models"]["depth"], expected)
            ),
            "candidate formula or preserved flat probability differs",
        )
        if book is None or beta == 0:
            require(
                forecast["models"]["depth"] == baseline,
                "fallback must equal original baseline exactly",
            )
        receipt_ref = forecast["model_receipt"]
        relative = Path(receipt_ref["path"])
        require(
            not relative.is_absolute()
            and len(relative.parts) == 2
            and relative.parts[0] == "model-receipts"
            and ".." not in relative.parts,
            "model receipt path escapes the run",
        )
        receipt_path = root / relative
        receipt = load(receipt_path)
        require(
            file_sha(receipt_path) == receipt_ref["sha256"]
            and receipt["model_sha256"]
            == receipt_ref["model_sha256"]
            == file_sha(model_path)
            and model["fitted_at"]
            <= receipt["observed_durable_at"]
            <= receipt_ref["durable_receipt_observed_at"]
            <= issued[0],
            "model was not durably observed before candidate emission",
        )
        verify_clock_reference(
            depth, receipt["clock_reference"], receipt["observed_durable_at"]
        )
        require(
            healthy(
                (receipt["observed_durable_at"], receipt["observed_monotonic"]),
                receipt["clock_reference"],
            ),
            "model durability clock unhealthy",
        )
        emissions[key] = (forecast, digest)
        fallback_book += book is None
        fallback_zero += beta == 0
    metrics_rows = {s: {arm: [] for arm in ARMS} for s in SYMBOLS}
    outcomes = set()
    for path in paths["outcomes"]:
        value = load(path)
        key = (value["anchor"], value["symbol"])
        require(
            key in emissions
            and key not in outcomes
            and path.name == f"{key[0]}-{key[1]}.json",
            "outcome lacks unique eligible candidate",
        )
        forecast, digest = emissions[key]
        observed_value = observed(pilot, *key)
        require(
            observed_value is not None
            and all(value[name] == item for name, item in observed_value.items())
            and value["forecast_sha256"] == digest
            and value["received_at"] <= value["copied_at"] <= now,
            "copied outcome differs from original price/receipt or chronology",
        )
        outcomes.add(key)
        for arm in ARMS:
            metrics_rows[key[1]][arm].append(
                score(forecast["models"][arm], value["label"])
            )
    metrics = {}
    for symbol, models in metrics_rows.items():
        metrics[symbol] = {}
        for arm, rows in models.items():
            moving = [row[2] for row in rows if row[2] is not None]
            metrics[symbol][arm] = dict(
                n=len(rows),
                moving_n=len(moving),
                direction_correct=sum(moving),
                direction_accuracy=math.fsum(moving) / len(moving) if moving else None,
                brier=math.fsum(row[0] for row in rows) / len(rows) if rows else None,
                log_loss=math.fsum(row[1] for row in rows) / len(rows)
                if rows
                else None,
            )
    missing, reasons = set(), Counter()
    for path in paths["missing"]:
        value = load(path)
        key = (value["anchor"], value["symbol"])
        require(
            key[0] % 300 == 0
            and cutoff <= key[0]
            and key[0] + 360 <= config["end_at"]
            and key[1] in SYMBOLS
            and key not in missing
            and key not in emissions
            and value.get("reason")
            and key[0] <= value["recorded_at"] <= now,
            "invalid or contradictory missing record",
        )
        missing.add(key)
        reasons[value["reason"]] += 1
    last = min(
        math.floor((now - 30) / 300) * 300,
        math.floor((config["end_at"] - 360) / 300) * 300,
    )
    expected_keys = (
        {(anchor, s) for anchor in range(cutoff, last + 1, 300) for s in SYMBOLS}
        if now >= cutoff
        else set()
    )
    forecast_keys = {
        tuple(item) for item in unconfirmed + ineligible
    } | emissions.keys()
    return {
        "verified": True,
        "schema": "independent-depth-information-audit-v1",
        "as_of": now,
        "as_of_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "config_sha256": sha(config),
        "training_cutoff": cutoff,
        "phase": "warmup"
        if now < cutoff
        else "model_frozen"
        if model
        else "awaiting_model",
        "model_fitted": model is not None,
        "coefficient_certificate": model_certificate,
        "source_snapshots": {
            name: dict(
                root=str(source["root"]),
                event_count=source["event_count"],
                chain_tip=source["chain_tip"],
            )
            for name, source in (("depth", depth), ("pilot", pilot))
        },
        "validated_source_books": validated_books,
        "validated_original_outcomes": validated_original_outcomes,
        "candidate_forecasts": len(emissions),
        "all_forecast_files": len(paths["forecasts"]),
        "unconfirmed_forecasts": unconfirmed,
        "ineligible_forecasts": ineligible,
        "late_confirmations": late,
        "clock_unhealthy_forecasts": clock_unhealthy,
        "missing_emissions": len(missing),
        "missing_reasons": dict(reasons),
        "fallback_missing_book": fallback_book,
        "fallback_zero_coefficient": fallback_zero,
        "scored_outcomes": len(outcomes),
        "pending_outcomes": len(emissions) - len(outcomes),
        "mature_pending_outcomes": [
            list(k) for k in sorted(emissions.keys() - outcomes) if k[0] + 360 <= now
        ],
        "unaccounted_anchor_symbols": [
            list(k) for k in sorted(expected_keys - forecast_keys - missing)
        ],
        "metrics": metrics,
        "statistical_superiority_verified": False,
        "scope": [
            "Independent source-chain, membership, scalar-optimum, formula, timing and score checks; no inference replay.",
            "Only post-acknowledgement-fsync observations sealed as eligible are scored; missing seals stay unconfirmed.",
            "Seal records do not independently attest their own fsync completion; timestamps remain local evidence.",
            "REST depth provides client availability, not a certified source event time or freshness guarantee.",
            "Original pilot model internals are not refitted or replayed; saved controls and original source prices are checked.",
            "No historical data, protected holdouts, credentials, orders or network access are used.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = args.output.resolve()
        require(
            not any(
                part.startswith("shadow-pilot")
                or part in ("archive", "holdout", "holdouts")
                for part in output.parts
            ),
            "audit output cannot modify protected sources",
        )
        for name in ("capture_root", "pilot_root"):
            source = Path(load(args.root / "config.json")[name]).resolve()
            require(
                source != output and source not in output.parents,
                "audit output cannot modify source journals",
            )
        with output.open("x") as handle:
            handle.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
