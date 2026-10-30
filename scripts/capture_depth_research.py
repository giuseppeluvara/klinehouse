#!/usr/bin/env python3
"""Capture new public depth snapshots for a separate prospective research stream.

This capture-only program never imports a model, the shadow pilot or a market
archive. REST depth has no exchange event timestamp: receipt time establishes
availability to this client, not the age of the exchange's in-memory book.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation, localcontext
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time
from datetime import datetime, timezone

import requests


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
URLS = {
    "depth": "https://api.binance.com/api/v3/depth",
    "time": "https://api.binance.com/api/v3/time",
}
LIMITS = {
    "levels": 20,
    "grid_seconds": 300,
    "lead_seconds": 10,
    "target_start_seconds": 60,
    "target_end_seconds": 360,
    "commit_deadline_seconds": 30,
    "clock_skew_bound_seconds": 2,
    "wall_monotonic_tolerance_seconds": 0.25,
    "response_bytes": 100_000,
    "max_storage_bytes": 64 * 1024**2,
    # A complete failed trio can retain raw and decoded bodies plus SQLite
    # pages. Leave room for that evidence and a terminal stop without pruning.
    "storage_reserve_bytes": 4 * 1024**2,
    "min_free_bytes": 40 * 1024**3,
    "max_run_seconds": 8 * 86400,
}
DEFAULT_END = "2026-11-05T11:10:00+00:00"
LEGACY_CODE_SHA256 = "97e72f387e0f50b9c286a740142e009a6d2f489c10772098a4bbddb5d879c623"
LEGACY_CONFIG_SHA256 = (
    "dd575129fe3a5be8029d506de585824754c1570202d15d70c27594356e12dbd7"
)
LEGACY_LIMITS = {**LIMITS, "max_storage_bytes": 32 * 1024**2}
AMENDMENT_SCHEMA = "klinehouse-depth-budget-amendment-v1"
AMENDMENT_REASON = (
    "SQLite measured 24-28 KiB per five-minute anchor including raw and decoded "
    "bodies, acknowledgements and page/index overhead; 1674 anchors project "
    "39-46 MiB, exceeding the original 28 MiB usable budget. Only raise storage "
    "cap to 64 MiB; preserve all data, timing, model and selection rules."
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def code_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def parse_utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("an explicit UTC end time is required")
    return parsed.timestamp()


def root_path(value):
    result = Path(value).resolve()
    if any(part.startswith("shadow-pilot") for part in result.parts):
        raise ValueError("the shadow pilot is outside this collector's scope")
    return result


def atomic_json(path, value, *, exclusive=False):
    """Sync an independent receipt, or atomically replace mutable health status."""
    encoded = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if exclusive:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def initialize(root, *, end_at=None, clock=time.time):
    root = root_path(root)
    start = clock()
    end = parse_utc(DEFAULT_END) if end_at is None else float(end_at)
    if not math.isfinite(end) or not 360 < end - start <= LIMITS["max_run_seconds"]:
        raise ValueError("end must allow a new target and be within eight days")
    if shutil.disk_usage(root.parent).free < LIMITS["min_free_bytes"]:
        raise ValueError("insufficient free space before activation")
    root.mkdir(parents=False, exist_ok=False)
    config = {
        "schema": "klinehouse-depth-research-v1",
        "activation_at": start,
        "activation_utc": utc(start),
        "end_at": end,
        "end_utc": utc(end),
        "symbols": list(SYMBOLS),
        "urls": URLS,
        "limits": LIMITS,
        "code_sha256": code_sha256(),
        "capture_only": True,
        "candidate_predictions": False,
        "orders": False,
        "source_semantics": "Aggregated visible resting depth, not trades, OI, positions or event-level OFI. REST lastUpdateId is a sequence, not an event timestamp. HTTP receipt time does not certify source freshness.",
        "availability": "Request no earlier than activation. Start at anchor-10 seconds; full body must arrive before anchor. Durable primary receipt commit must complete by anchor+30 seconds. Preserve late, invalid, missing and unacknowledged observations; never backfill books.",
        "features": "QI1, QI20 and spread from the SAME depth body; Decimal arithmetic. QI20 is the only proposed new predictor. No mixing asynchronous REST bookTicker fields.",
        "future_hypothesis_not_executed": {
            "feature": "QI20",
            "model": "One shared offset-logistic coefficient on the frozen baseline, ridge 0.25, beta in [-8,8].",
            "training": "First 24 hours of newly captured books and only subsequently matured authorized outcomes; then freeze before future evaluation through this capture's end. No retuning.",
            "target": "Existing five-minute spot return anchor+60 to anchor+360 seconds; this collector neither reads outcomes nor emits predictions.",
        },
    }
    atomic_json(root / "config.json", config, exclusive=True)
    atomic_json(root / "config-sha256.json", {"sha256": sha(config)}, exclusive=True)
    with Journal(root, config, clock=clock) as journal:
        journal.append("activated", {"capture_only": True})
        journal.status("initialized")
    return config


def load_config(root):
    root = root_path(root)
    config = json.loads((root / "config.json").read_text())
    if (
        sha(config) != json.loads((root / "config-sha256.json").read_text())["sha256"]
        or config.get("urls") != URLS
        or config.get("symbols") != list(SYMBOLS)
        or config.get("capture_only") is not True
        or config.get("candidate_predictions") is not False
        or config.get("orders") is not False
    ):
        raise ValueError("frozen depth collector configuration/code changed")
    if config.get("code_sha256") == code_sha256() and config.get("limits") == LIMITS:
        if (root / "budget-amendment.json").exists():
            raise ValueError("native 64 MiB runs cannot inherit a legacy amendment")
    else:
        load_budget_amendment(root, config, require_runtime=True)
    return config


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_bytes(path, encoded):
    """Publish complete new bytes atomically; retain private crash evidence."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    sync_directory(path.parent)


def sync_directory(path):
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def legacy_identity(root, config):
    """Only the already activated, exactly identified 32 MiB stream may migrate."""
    root = root_path(root)
    original = root / "runtime/capture_depth_research.py"
    if (
        root.name != "depth-research-20261030"
        or sha(config) != LEGACY_CONFIG_SHA256
        or config.get("code_sha256") != LEGACY_CODE_SHA256
        or config.get("limits") != LEGACY_LIMITS
        or config.get("urls") != URLS
        or config.get("symbols") != list(SYMBOLS)
        or config.get("capture_only") is not True
        or config.get("candidate_predictions") is not False
        or config.get("orders") is not False
        or json.loads((root / "config-sha256.json").read_text())
        != {"sha256": LEGACY_CONFIG_SHA256}
        or (root / "runtime").is_symlink()
        or original.is_symlink()
        or file_sha256(original) != LEGACY_CODE_SHA256
    ):
        raise ValueError("only the known unchanged legacy capture can be amended")
    return root


@contextmanager
def read_only_journal(root, config):
    database = root / "depth-events.sqlite"
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        if dict(connection.execute("SELECT key,value FROM meta")) != {
            "config": canonical(config),
            "config_sha256": sha(config),
        }:
            raise ValueError("amendment source journal identity differs")
        yield connection
    finally:
        connection.close()


def clock_reference_from_row(row):
    if row is None:
        raise ValueError("a recent committed healthy clock probe is required")
    payload = json.loads(row["payload"])
    receipt = payload["receipt"]
    evidence = clock_evidence(receipt)
    if (
        payload.get("healthy") is not True
        or evidence["healthy"] is not True
        or evidence["absolute_offset_bound_seconds"]
        != payload.get("absolute_offset_bound_seconds")
        or receipt.get("source") != "time"
        or receipt.get("url") != URLS["time"]
        or receipt.get("params") != {}
        or receipt.get("http_status") != 200
    ):
        raise ValueError("amendment clock probe is not healthy")
    return {
        "probe_seq": row["seq"],
        "probe_digest": row["digest"],
        "utc": receipt["received_at"],
        "monotonic": receipt["received_monotonic"],
        "absolute_offset_bound_seconds": evidence["absolute_offset_bound_seconds"],
    }


def amendment_clock_healthy(reference, now, mono):
    values = (now, mono, reference["utc"], reference["monotonic"])
    return all(type(v) in (int, float) and math.isfinite(v) for v in values) and (
        0 <= now - reference["utc"] <= 600
        and elapsed_healthy(
            {
                "requested_at": reference["utc"],
                "requested_monotonic": reference["monotonic"],
                "received_at": now,
                "received_monotonic": mono,
            }
        )
    )


def load_budget_amendment(root, config, *, require_runtime=False):
    root = legacy_identity(root, config)
    path = root / "budget-amendment.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("legacy 32 MiB configuration requires its budget amendment")
    document = json.loads(path.read_text())
    body = {key: value for key, value in document.items() if key != "amendment_sha256"}
    first = math.ceil(config["activation_at"] / 300) * 300
    expected = {
        "schema": AMENDMENT_SCHEMA,
        "root": str(root),
        "config_sha256": LEGACY_CONFIG_SHA256,
        "original_code_sha256": LEGACY_CODE_SHA256,
        "amended_code_sha256": code_sha256(),
        "original_runtime_path": "runtime/capture_depth_research.py",
        "amended_runtime_path": f"runtime-amendments/budget-64MiB-{code_sha256()}/capture_depth_research.py",
        "old_limits": LEGACY_LIMITS,
        "new_limits": LIMITS,
        "first_anchor": first,
        "training_cutoff": first + 86400,
        "activation_at": config["activation_at"],
        "end_at": config["end_at"],
        "symbols": config["symbols"],
        "urls": config["urls"],
        "reason": AMENDMENT_REASON,
        "data_selection_and_model_changes": False,
    }
    variable = {
        "registered_at",
        "registered_monotonic",
        "registration_clock_reference",
        "prepared_source_prefix",
    }
    if (
        set(body) != set(expected) | variable
        or any(body.get(key) != value for key, value in expected.items())
        or document.get("amendment_sha256") != sha(body)
        or not amendment_clock_healthy(
            body["registration_clock_reference"],
            body["registered_at"],
            body["registered_monotonic"],
        )
        or not config["activation_at"] <= body["registered_at"] < first + 86400
    ):
        raise ValueError("budget amendment document/hash/limits/registration differs")
    relative = Path(body["amended_runtime_path"])
    runtime = root / relative
    if (
        any(
            (root / Path(*relative.parts[:i])).is_symlink()
            for i in range(1, len(relative.parts) + 1)
        )
        or file_sha256(runtime) != code_sha256()
        or (require_runtime and Path(__file__).resolve() != runtime.resolve())
    ):
        raise ValueError("budget amendment runtime bytes/path differ")
    with read_only_journal(root, config) as connection:
        prefix = body["prepared_source_prefix"]
        if (
            set(prefix) != {"event_count", "chain_tip"}
            or type(prefix["event_count"]) is not int
            or prefix["event_count"] < 1
        ):
            raise ValueError("budget amendment source prefix differs")
        row = connection.execute(
            "SELECT digest FROM events WHERE seq=?", (prefix["event_count"],)
        ).fetchone()
        if row is None or row["digest"] != prefix["chain_tip"]:
            raise ValueError("budget amendment source prefix differs")
        reference = body["registration_clock_reference"]
        probe = connection.execute(
            "SELECT * FROM events WHERE seq=? AND kind='clock_probe'",
            (reference["probe_seq"],),
        ).fetchone()
        if (
            reference != clock_reference_from_row(probe)
            or reference["probe_seq"] > prefix["event_count"]
        ):
            raise ValueError("budget amendment registration clock differs")
    return document


def prepare_budget_amendment(root, *, clock=time.time, monotonic=time.monotonic):
    """Prepare new runtime/manifest only, while the unchanged legacy writer runs."""
    root = root_path(root)
    config = json.loads((root / "config.json").read_text())
    legacy_identity(root, config)
    if (root / "budget-amendment.json").exists() or (
        root / "budget-amendment.json"
    ).is_symlink():
        raise FileExistsError("the immutable budget amendment already exists")
    with read_only_journal(root, config) as connection:
        head = connection.execute(
            "SELECT seq,digest FROM events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        probe = connection.execute(
            "SELECT * FROM events WHERE kind='clock_probe' ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        reference = clock_reference_from_row(probe)
        prefix = {"event_count": head["seq"], "chain_tip": head["digest"]}
    first = math.ceil(config["activation_at"] / 300) * 300
    now, mono = clock(), monotonic()
    if not now < first + 86400 or not amendment_clock_healthy(reference, now, mono):
        raise ValueError(
            "budget amendment must be registered before cutoff with a healthy clock"
        )
    if shutil.disk_usage(root).free < LIMITS["min_free_bytes"]:
        raise ValueError("insufficient free space before budget amendment")
    relative = Path(
        f"runtime-amendments/budget-64MiB-{code_sha256()}/capture_depth_research.py"
    )
    for parent in (root / "runtime-amendments", root / relative.parent):
        if parent.is_symlink():
            raise ValueError("budget amendment runtime directory must not be a symlink")
        parent.mkdir(exist_ok=True)
        sync_directory(parent.parent)
    runtime = root / relative
    if runtime.exists():
        if runtime.is_symlink() or file_sha256(runtime) != code_sha256():
            raise ValueError("prepared runtime bytes differ")
    else:
        immutable_bytes(runtime, Path(__file__).read_bytes())
    with runtime.open("rb") as handle:
        os.fsync(handle.fileno())
    for parent in (runtime.parent, runtime.parent.parent, root):
        sync_directory(parent)
    now, mono = clock(), monotonic()
    if not config[
        "activation_at"
    ] <= now < first + 86400 or not amendment_clock_healthy(reference, now, mono):
        raise ValueError(
            "budget amendment preparation exceeded its causal registration window"
        )
    body = {
        "schema": AMENDMENT_SCHEMA,
        "root": str(root),
        "registered_at": now,
        "registered_monotonic": mono,
        "registration_clock_reference": reference,
        "config_sha256": LEGACY_CONFIG_SHA256,
        "original_code_sha256": LEGACY_CODE_SHA256,
        "amended_code_sha256": code_sha256(),
        "original_runtime_path": "runtime/capture_depth_research.py",
        "amended_runtime_path": str(relative),
        "old_limits": LEGACY_LIMITS,
        "new_limits": LIMITS,
        "first_anchor": first,
        "training_cutoff": first + 86400,
        "activation_at": config["activation_at"],
        "end_at": config["end_at"],
        "symbols": config["symbols"],
        "urls": config["urls"],
        "prepared_source_prefix": prefix,
        "reason": AMENDMENT_REASON,
        "data_selection_and_model_changes": False,
    }
    document = {**body, "amendment_sha256": sha(body)}
    immutable_bytes(
        root / "budget-amendment.json",
        (
            json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode(),
    )
    return load_budget_amendment(root, config)


def activate_budget_amendment(journal):
    """Observe one immutable amendment before any new producer/capture events."""
    if (
        journal.config.get("limits") == LIMITS
        and journal.config.get("code_sha256") == code_sha256()
    ):
        return None
    document = load_budget_amendment(journal.root, journal.config, require_runtime=True)
    payload = {
        "document": document,
        "amendment_sha256": document["amendment_sha256"],
        "document_sha256": file_sha256(journal.root / "budget-amendment.json"),
    }
    existing = list(
        journal.connection.execute(
            "SELECT * FROM events WHERE kind='resource_budget_amended' ORDER BY seq"
        )
    )
    if len(existing) > 1:
        raise ValueError("budget amendment event must occur once")
    cutoff = document["training_cutoff"]
    if not existing:
        probe = journal.connection.execute(
            "SELECT * FROM events WHERE kind='clock_probe' ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        reference = clock_reference_from_row(probe)
        if not document[
            "registered_at"
        ] <= journal.clock() < cutoff or not amendment_clock_healthy(
            reference, journal.clock(), journal.monotonic()
        ):
            raise ValueError(
                "budget amendment activation must precede cutoff with a healthy clock"
            )
        record = journal.append(
            "resource_budget_amended",
            payload,
            clock_healthy=True,
            clock_reference=[reference["utc"], reference["monotonic"]],
        )
        if (
            not record["clock_healthy"]
            or not record["ack_clock_healthy"]
            or record["ack_commit_observed_at"] >= cutoff
        ):
            journal.append(
                "invalidated",
                {
                    "record_seq": record["seq"],
                    "ack_seq": record["ack_seq"],
                    "reason": "resource_budget_amendment_confirmation_failed",
                },
            )
            raise ValueError("budget amendment confirmation failed before acquisition")
        existing = list(
            journal.connection.execute(
                "SELECT * FROM events WHERE kind='resource_budget_amended'"
            )
        )
    event = existing[0]
    matches = [
        row
        for row in journal.connection.execute(
            "SELECT * FROM events WHERE kind='commit_ack'"
        )
        if json.loads(row["payload"])["record_seq"] == event["seq"]
    ]
    bad = any(
        json.loads(row[0])["record_seq"] == event["seq"]
        for row in journal.connection.execute(
            "SELECT payload FROM events WHERE kind='invalidated'"
        )
    )
    if (
        len(matches) != 1
        or bad
        or json.loads(event["payload"]) != payload
        or event["anchor"] is not None
        or event["symbol"] is not None
    ):
        raise ValueError("budget amendment event/confirmation differs")
    ack_row = matches[0]
    ack = json.loads(ack_row["payload"])
    probe = journal.connection.execute(
        "SELECT * FROM events WHERE kind='clock_probe' AND seq<? ORDER BY seq DESC LIMIT 1",
        (event["seq"],),
    ).fetchone()
    reference = clock_reference_from_row(probe)
    if not (
        document["registered_at"]
        <= event["recorded_at"]
        <= ack["committed_at"]
        <= ack_row["recorded_at"]
        < cutoff
        and ack["clock_healthy"] is True
        and ack["clock_reference"] == [reference["utc"], reference["monotonic"]]
        and amendment_clock_healthy(
            reference, ack["committed_at"], ack["committed_monotonic"]
        )
    ):
        raise ValueError("budget amendment confirmation clock/cutoff differs")
    journal.effective_limits = LIMITS
    journal.budget_amendment = document["amendment_sha256"]
    return document


def request_spec(source, symbol=None):
    if source == "time" and symbol is None:
        return URLS[source], {}
    if source == "depth" and symbol in SYMBOLS:
        return URLS[source], {"symbol": symbol, "limit": LIMITS["levels"]}
    raise ValueError("only registered public depth/time requests are allowed")


def fetch(source, symbol=None, *, clock=time.time, monotonic=time.monotonic):
    url, params = request_spec(source, symbol)
    requested, requested_mono = clock(), monotonic()
    receipt = {
        "source": source,
        "symbol": symbol,
        "url": url,
        "params": params,
        "requested_at": requested,
        "requested_monotonic": requested_mono,
    }
    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.get(
                url, params=params, timeout=(2, 5), allow_redirects=False, stream=True
            ) as response:
                chunks, size = [], 0
                for chunk in response.iter_content(chunk_size=8192):
                    size += len(chunk)
                    if size > LIMITS["response_bytes"]:
                        raise ValueError("public response exceeds frozen body limit")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                receipt.update(
                    http_status=response.status_code,
                    http_date=response.headers.get("Date"),
                    used_weight_1m=response.headers.get("X-MBX-USED-WEIGHT-1M"),
                    body_bytes=len(raw),
                    raw_body=raw.decode("utf-8"),
                    body_sha256=hashlib.sha256(raw).hexdigest(),
                )
        receipt.update(received_at=clock(), received_monotonic=monotonic())
        if receipt["http_status"] != 200:
            raise ValueError(f"public HTTP status {receipt['http_status']}")
        receipt["body"] = json.loads(receipt["raw_body"])
    except (requests.RequestException, ValueError, UnicodeError) as error:
        receipt.setdefault("received_at", clock())
        receipt.setdefault("received_monotonic", monotonic())
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)
    return receipt


def elapsed_healthy(receipt):
    wall = receipt["received_at"] - receipt["requested_at"]
    mono = receipt["received_monotonic"] - receipt["requested_monotonic"]
    return (
        wall >= 0
        and mono >= 0
        and abs(wall - mono) <= LIMITS["wall_monotonic_tolerance_seconds"]
    )


def clock_evidence(receipt):
    healthy, bound = False, None
    body = receipt.get("body", {})
    server = (
        body.get("serverTime")
        if isinstance(body, dict) and set(body) == {"serverTime"}
        else None
    )
    if (
        not receipt.get("error")
        and elapsed_healthy(receipt)
        and type(server) is int
        and 0 < server < 10**16
    ):
        server /= 1000
        bound = max(
            abs(server - receipt["requested_at"]), abs(server - receipt["received_at"])
        )
        healthy = bound <= LIMITS["clock_skew_bound_seconds"]
    return {
        "healthy": healthy,
        "absolute_offset_bound_seconds": bound,
        "note": "Bound includes the complete observed request round trip; not an exchange-book timestamp.",
    }


def decimal_positive(value):
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("book price/quantity must be bounded decimal strings")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal book value") from exc
    if not result.is_finite() or result <= 0 or abs(result.adjusted()) > 100:
        raise ValueError("book price/quantity must be finite and positive")
    return result


def book_features(body):
    if not isinstance(body, dict) or set(body) != {"lastUpdateId", "bids", "asks"}:
        raise ValueError("unexpected depth schema")
    if type(body["lastUpdateId"]) is not int or body["lastUpdateId"] < 0:
        raise ValueError("invalid depth update identifier")
    sides = []
    for side in ("bids", "asks"):
        rows = body[side]
        if not isinstance(rows, list) or len(rows) != LIMITS["levels"]:
            raise ValueError("exactly twenty depth levels per side required")
        parsed = []
        for row in rows:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError("depth rows must have price and quantity")
            parsed.append(tuple(decimal_positive(value) for value in row))
        prices = [row[0] for row in parsed]
        if any(
            (a <= b if side == "bids" else a >= b) for a, b in zip(prices, prices[1:])
        ):
            raise ValueError("depth prices must be unique and strictly sorted")
        sides.append(parsed)
    bids, asks = sides
    if bids[0][0] >= asks[0][0]:
        raise ValueError("crossed or locked depth snapshot")
    with localcontext() as context:
        context.prec = 50
        bid, ask = bids[0][0], asks[0][0]
        b1, a1 = bids[0][1], asks[0][1]
        b20, a20 = sum(row[1] for row in bids), sum(row[1] for row in asks)
        mid = (bid + ask) / 2
        return {
            "qi1": str((b1 - a1) / (b1 + a1)),
            "qi20": str((b20 - a20) / (b20 + a20)),
            "spread_price": str(ask - bid),
            "spread_bps": str(10000 * (ask - bid) / mid),
            "mid_price": str(mid),
            "best_bid": str(bid),
            "best_ask": str(ask),
            "best_bid_quantity": str(b1),
            "best_ask_quantity": str(a1),
        }


SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE events(seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, recorded_at REAL NOT NULL,
                    anchor INTEGER, symbol TEXT, payload TEXT NOT NULL,
                    prev_hash TEXT NOT NULL, digest TEXT UNIQUE NOT NULL);
CREATE UNIQUE INDEX unique_book ON events(anchor,symbol) WHERE kind='book_receipt';
CREATE UNIQUE INDEX unique_missing ON events(anchor,symbol) WHERE kind='missing';
CREATE TRIGGER events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'append-only'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'append-only'); END;
CREATE TRIGGER meta_no_update BEFORE UPDATE ON meta BEGIN SELECT RAISE(ABORT,'immutable config'); END;
CREATE TRIGGER meta_no_delete BEFORE DELETE ON meta BEGIN SELECT RAISE(ABORT,'immutable config'); END;
"""


class Journal:
    def __init__(self, root, config, *, clock=time.time, monotonic=time.monotonic):
        self.root, self.config = root_path(root), config
        self.clock, self.monotonic = clock, monotonic
        self.effective_limits = config["limits"]
        self.budget_amendment = None
        path = self.root / "depth-events.sqlite"
        exists = path.exists()
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        if not exists:
            self.connection.executescript(SCHEMA)
            with self.connection:
                self.connection.execute(
                    "INSERT INTO meta VALUES ('config',?)", (canonical(config),)
                )
                self.connection.execute(
                    "INSERT INTO meta VALUES ('config_sha256',?)", (sha(config),)
                )
        found = dict(self.connection.execute("SELECT key,value FROM meta"))
        if found != {"config": canonical(config), "config_sha256": sha(config)}:
            self.connection.close()
            raise ValueError("journal configuration differs from frozen config")
        self.verify()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.close()

    def _commit(self):
        self.connection.commit()

    def _write(self, kind, payload, anchor=None, symbol=None):
        row = self.connection.execute(
            "SELECT seq,digest FROM events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        seq, previous = (
            (row["seq"] + 1, row["digest"]) if row else (1, sha(self.config))
        )
        recorded, recorded_mono = self.clock(), self.monotonic()
        envelope = {
            "seq": seq,
            "kind": kind,
            "recorded_at": recorded,
            "anchor": anchor,
            "symbol": symbol,
            "payload": payload,
            "prev_hash": previous,
        }
        digest = sha(envelope)
        self.connection.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?,?)",
            (seq, kind, recorded, anchor, symbol, canonical(payload), previous, digest),
        )
        self._commit()
        committed, committed_mono = self.clock(), self.monotonic()
        return {
            "seq": seq,
            "digest": digest,
            "recorded_at": recorded,
            "recorded_monotonic": recorded_mono,
            "committed_at": committed,
            "committed_monotonic": committed_mono,
            "commit_clock_healthy": elapsed_healthy(
                {
                    "requested_at": recorded,
                    "requested_monotonic": recorded_mono,
                    "received_at": committed,
                    "received_monotonic": committed_mono,
                }
            ),
        }

    def append(
        self,
        kind,
        payload,
        *,
        anchor=None,
        symbol=None,
        eligible=False,
        clock_healthy=False,
        clock_reference=None,
    ):
        """Only this call can acknowledge the FULL commit it actually observed."""
        primary = self._write(kind, payload, anchor, symbol)
        deadline = (
            anchor + LIMITS["commit_deadline_seconds"] if anchor is not None else None
        )
        reference_ok = self.reference_healthy(primary, clock_reference)
        post_commit_healthy = bool(
            clock_healthy and primary["commit_clock_healthy"] and reference_ok
        )
        accepted = bool(
            eligible and post_commit_healthy and primary["committed_at"] <= deadline
        )
        ack_payload = {
            "record_seq": primary["seq"],
            "record_hash": primary["digest"],
            "record_started_monotonic": primary["recorded_monotonic"],
            "committed_at": primary["committed_at"],
            "committed_monotonic": primary["committed_monotonic"],
            "deadline": deadline,
            "eligible": accepted,
            "clock_healthy": post_commit_healthy,
            "commit_elapsed_clock_healthy": primary["commit_clock_healthy"],
            "clock_reference": clock_reference,
            "clock_reference_healthy": reference_ok,
            "durable_before_anchor": bool(
                anchor is not None and primary["committed_at"] < anchor
            ),
        }
        ack = self._write("commit_ack", ack_payload, anchor, symbol)
        ack_clock_ok = ack["commit_clock_healthy"] and self.reference_healthy(
            ack, clock_reference
        )
        if accepted and not ack_clock_ok:
            # The acknowledgement's own fsync can straddle a clock jump. Preserve
            # it and record a terminal invalidation rather than count it as data.
            self.append(
                "invalidated",
                {
                    "record_seq": primary["seq"],
                    "ack_seq": ack["seq"],
                    "reason": "clock_unhealthy_after_acknowledgment_commit",
                },
                anchor=anchor,
                symbol=symbol,
            )
            accepted = False
        return {
            **primary,
            "ack_seq": ack["seq"],
            "ack_hash": ack["digest"],
            "eligible": accepted,
            "ack_commit_observed_at": ack["committed_at"],
            "ack_clock_healthy": ack_clock_ok,
            "clock_healthy": bool(post_commit_healthy and ack_clock_ok),
        }

    @staticmethod
    def reference_healthy(record, reference):
        if reference is None:
            return True
        return elapsed_healthy(
            {
                "requested_at": reference[0],
                "requested_monotonic": reference[1],
                "received_at": record["committed_at"],
                "received_monotonic": record["committed_monotonic"],
            }
        )

    def verify(self):
        previous, records = sha(self.config), {}
        for expected, row in enumerate(
            self.connection.execute("SELECT * FROM events ORDER BY seq"), 1
        ):
            payload = json.loads(row["payload"])
            envelope = {
                k: row[k]
                for k in ("seq", "kind", "recorded_at", "anchor", "symbol", "prev_hash")
            }
            envelope["payload"] = payload
            if (
                row["seq"] != expected
                or row["prev_hash"] != previous
                or row["digest"] != sha(envelope)
            ):
                raise ValueError("depth journal hash chain mismatch")
            if canonical(payload) != row["payload"]:
                raise ValueError("noncanonical depth journal payload")
            if row["kind"] == "commit_ack":
                primary = records.get(payload["record_seq"])
                if (
                    primary is None
                    or primary["kind"] == "commit_ack"
                    or primary["digest"] != payload["record_hash"]
                ):
                    raise ValueError("invalid depth commit acknowledgment")
                if (
                    payload["clock_healthy"]
                    and not primary["recorded_at"]
                    <= payload["committed_at"]
                    <= row["recorded_at"]
                ):
                    raise ValueError("invalid observed post-commit time")
                elapsed_ok = elapsed_healthy(
                    {
                        "requested_at": primary["recorded_at"],
                        "requested_monotonic": payload["record_started_monotonic"],
                        "received_at": payload["committed_at"],
                        "received_monotonic": payload["committed_monotonic"],
                    }
                )
                if elapsed_ok != payload["commit_elapsed_clock_healthy"]:
                    raise ValueError(
                        "commit clock evidence disagrees with elapsed time"
                    )
                reference_ok = self.reference_healthy(
                    payload, payload["clock_reference"]
                )
                if reference_ok != payload["clock_reference_healthy"]:
                    raise ValueError("post-commit reference clock evidence disagrees")
                if payload["clock_healthy"] and (
                    not elapsed_ok or not payload["clock_reference_healthy"]
                ):
                    raise ValueError("post-commit clock health was not established")
                if payload["eligible"] and (
                    not payload["clock_healthy"]
                    or payload["committed_at"] > payload["deadline"]
                ):
                    raise ValueError("ineligible depth acknowledgment")
            records[row["seq"]] = dict(row)
            previous = row["digest"]
        return {"events": len(records), "chain_tip": previous}

    def completed(self, anchor):
        rows = list(
            self.connection.execute(
                "SELECT kind,symbol,payload FROM events WHERE anchor=? ORDER BY seq",
                (anchor,),
            )
        )
        invalidated = {
            json.loads(row["payload"])["record_seq"]
            for row in rows
            if row["kind"] == "invalidated"
        }
        completed = set()
        for row in rows:
            payload = json.loads(row["payload"])
            if row["kind"] == "missing" or (
                row["kind"] == "commit_ack"
                and payload["eligible"]
                and payload["record_seq"] not in invalidated
            ):
                completed.add(row["symbol"])
        return completed

    def status(self, phase, **extra):
        counts = dict(
            self.connection.execute("SELECT kind,COUNT(*) FROM events GROUP BY kind")
        )
        head = self.connection.execute(
            "SELECT seq,digest FROM events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        accepted = self.connection.execute("""SELECT COUNT(*) FROM events AS ack
            WHERE ack.kind='commit_ack' AND json_extract(ack.payload,'$.eligible')=1
            AND NOT EXISTS(SELECT 1 FROM events AS bad WHERE bad.kind='invalidated'
                AND json_extract(bad.payload,'$.record_seq')=json_extract(ack.payload,'$.record_seq'))""").fetchone()[
            0
        ]
        value = {
            "phase": phase,
            "updated_at": self.clock(),
            "updated_utc": utc(self.clock()),
            "config_sha256": sha(self.config),
            "pid": os.getpid(),
            "counts": counts,
            "accepted_books": accepted,
            "last_seq": head["seq"] if head else 0,
            "chain_tip": head["digest"] if head else sha(self.config),
            "free_bytes": shutil.disk_usage(self.root).free,
            "directory_bytes": directory_bytes(self.root),
            "capture_only": True,
            "candidate_predictions": False,
            "effective_limits": self.effective_limits,
            "budget_amendment_sha256": self.budget_amendment,
            **extra,
        }
        atomic_json(self.root / "status.json", value)
        return value


def directory_bytes(root):
    return sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())


def resource_reason(root, *, disk_usage=shutil.disk_usage, limits=None):
    limits = LIMITS if limits is None else limits
    if disk_usage(root).free < limits["min_free_bytes"]:
        return "free_space_below_40_GiB"
    if (
        directory_bytes(root) + limits["storage_reserve_bytes"]
        >= limits["max_storage_bytes"]
    ):
        return f"{limits['max_storage_bytes'] // 1024**2}_MiB_budget_reserve_reached"
    return None


@contextmanager
def writer_lock(root):
    with (root / "writer.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another depth research writer is running") from exc
        yield


def capture_cycle(
    journal, anchor, *, fetcher=fetch, clock=time.time, monotonic=time.monotonic
):
    config = journal.config
    if anchor % 300 or not config["activation_at"] <= anchor <= config["end_at"] - 360:
        raise ValueError("anchor outside the registered prospective five-minute grid")
    if clock() < max(config["activation_at"], anchor - 10):
        raise ValueError("capture window has not opened")
    pending = set(SYMBOLS) - journal.completed(anchor)
    # A process that did not observe a primary commit returning cannot certify
    # it later or replace it with a second request for the same anchor.
    existing = {
        row[0]
        for row in journal.connection.execute(
            "SELECT symbol FROM events WHERE kind='book_receipt' AND anchor=?",
            (anchor,),
        )
    }
    for symbol in sorted(pending & existing):
        journal.append(
            "missing",
            {"reason": "prior_book_unacknowledged_or_ineligible"},
            anchor=anchor,
            symbol=symbol,
        )
    pending -= existing
    if not pending:
        return []
    if clock() >= anchor:
        for symbol in sorted(pending):
            journal.append(
                "missing",
                {"reason": "missed_pre_anchor_capture_window"},
                anchor=anchor,
                symbol=symbol,
            )
        return []
    probe = fetcher("time", clock=clock, monotonic=monotonic)
    evidence = clock_evidence(probe)
    journal.append("clock_probe", {"receipt": probe, **evidence}, anchor=anchor)
    if not evidence["healthy"] or clock() >= anchor:
        for symbol in sorted(pending):
            journal.append(
                "missing",
                {"reason": "clock_unhealthy_or_capture_window_elapsed"},
                anchor=anchor,
                symbol=symbol,
            )
        return []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            symbol: pool.submit(
                fetcher, "depth", symbol, clock=clock, monotonic=monotonic
            )
            for symbol in SYMBOLS
            if symbol in pending
        }
        receipts = {symbol: future.result() for symbol, future in futures.items()}
    accepted = []
    for symbol, receipt in receipts.items():
        now, mono = clock(), monotonic()
        healthy = (
            elapsed_healthy(receipt)
            and abs((now - probe["received_at"]) - (mono - probe["received_monotonic"]))
            <= LIMITS["wall_monotonic_tolerance_seconds"]
        )
        reasons = []
        if receipt.get("error"):
            reasons.append(receipt["error_type"])
        if receipt["requested_at"] < config["activation_at"]:
            reasons.append("request_before_activation")
        if receipt["received_at"] >= anchor:
            reasons.append("body_received_at_or_after_anchor")
        if not healthy:
            reasons.append("wall_monotonic_clock_unhealthy")
        try:
            features = book_features(receipt.get("body"))
        except (ValueError, InvalidOperation) as error:
            features = None
            reasons.append(str(error))
        payload = {
            "receipt": receipt,
            "features": features,
            "validation_errors": reasons,
            "source_event_timestamp": None,
            "clock_healthy": healthy,
            "availability_note": "Actual full-body client receipt; source age is unknown.",
        }
        record = journal.append(
            "book_receipt",
            payload,
            anchor=anchor,
            symbol=symbol,
            eligible=not reasons,
            clock_healthy=healthy,
            clock_reference=[probe["received_at"], probe["received_monotonic"]],
        )
        if record["eligible"]:
            accepted.append(
                {"symbol": symbol, "received_at": receipt["received_at"], **record}
            )
        else:
            fallback_reason = (
                "late_durable_commit"
                if record["clock_healthy"]
                else "post_commit_clock_unhealthy"
            )
            journal.append(
                "missing",
                {
                    "reason": "; ".join(reasons) if reasons else fallback_reason,
                    "book_seq": record["seq"],
                },
                anchor=anchor,
                symbol=symbol,
            )
    if len(accepted) == 3 and not (journal.root / "first-cycle.json").exists():
        proof = {
            "anchor": anchor,
            "anchor_utc": utc(anchor),
            "captured_at": clock(),
            "books": accepted,
            "all_body_receipts_before_anchor": all(
                row["received_at"] < anchor for row in accepted
            ),
            "all_primary_commits_before_anchor": all(
                row["committed_at"] < anchor for row in accepted
            ),
            "all_ack_commits_before_anchor": all(
                row["ack_commit_observed_at"] < anchor for row in accepted
            ),
            "capture_only": True,
            "candidate_predictions": False,
            "journal_head": journal.verify(),
        }
        atomic_json(journal.root / "first-cycle.json", proof, exclusive=True)
        atomic_json(
            journal.root / "first-cycle-write-receipt.json",
            {
                "durable_write_observed_at": clock(),
                "sha256": hashlib.sha256(
                    (journal.root / "first-cycle.json").read_bytes()
                ).hexdigest(),
            },
            exclusive=True,
        )
    journal.status("running", last_anchor=anchor, latest_clock_ok=evidence["healthy"])
    return accepted


def run(
    root,
    *,
    once=False,
    clock=time.time,
    monotonic=time.monotonic,
    sleep=time.sleep,
    fetcher=fetch,
):
    root = root_path(root)
    config = load_config(root)
    with (
        writer_lock(root),
        Journal(root, config, clock=clock, monotonic=monotonic) as journal,
    ):
        amendment = activate_budget_amendment(journal)
        journal.append(
            "process_started",
            {
                "capture_only": True,
                "at": clock(),
                "producer_code_sha256": code_sha256(),
                "amendment_sha256": None
                if amendment is None
                else amendment["amendment_sha256"],
            },
        )
        anchor = math.ceil(config["activation_at"] / 300) * 300
        last_anchor = math.floor((config["end_at"] - 360) / 300) * 300
        while True:
            reason = resource_reason(root, limits=journal.effective_limits)
            if reason or clock() >= config["end_at"]:
                phase = "resource_stopped" if reason else "finished"
                journal.append(
                    "stopped", {"reason": reason or "registered_end_reached"}
                )
                journal.status(phase)
                return 2 if reason else 0
            if anchor > last_anchor:
                journal.status("awaiting_registered_end")
                sleep(max(0, min(30, config["end_at"] - clock())))
                continue
            if journal.completed(anchor) == set(SYMBOLS):
                anchor += 300
                continue
            if clock() < anchor - 10:
                journal.status(
                    "waiting", next_anchor=anchor, next_anchor_utc=utc(anchor)
                )
                sleep(max(0, min(30, anchor - 10 - clock())))
                continue
            capture_cycle(
                journal, anchor, fetcher=fetcher, clock=clock, monotonic=monotonic
            )
            if once:
                journal.status("paused_after_one_anchor", last_anchor=anchor)
                return 0
            anchor += 300


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    initialize_parser = sub.add_parser("init")
    initialize_parser.add_argument("--root", type=Path, required=True)
    initialize_parser.add_argument("--end-utc", default=DEFAULT_END)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--once", action="store_true")
    amendment_parser = sub.add_parser("prepare-budget-amendment")
    amendment_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-budget-amendment":
        print(json.dumps(prepare_budget_amendment(args.root), sort_keys=True))
        return 0
    if args.command == "init":
        config = initialize(args.root, end_at=parse_utc(args.end_utc))
        print(
            json.dumps(
                {
                    "root": str(root_path(args.root)),
                    "config_sha256": sha(config),
                    "capture_only": True,
                }
            )
        )
        return 0
    return run(args.root, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
