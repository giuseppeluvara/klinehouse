#!/usr/bin/env python3
"""Collect and score a bounded seven-day prospective shadow trial, without orders."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.shadow_ledger import Ledger  # noqa: E402
from research.shadow_model import predict, outcome_label, validate_snapshot  # noqa: E402
from research.shadow_sources import SYMBOLS, URLS, clock_probe, fetch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CODE = (
    "research/shadow_ledger.py",
    "research/shadow_model.py",
    "research/shadow_sources.py",
    "scripts/run_shadow.py",
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_atomic(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(obj, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def utc(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def initialize(run_dir, snapshot_path, *, start_at=None):
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise ValueError("new trial requires a new directory")
    snapshot = json.loads(Path(snapshot_path).read_text())
    validate_snapshot(snapshot)
    start = int(start_at or math.ceil((time.time() + 5) / 60) * 60)
    if start < time.time() or start % 60:
        raise ValueError("activation must be a future UTC minute")
    run_dir.mkdir(parents=True)
    shutil.copyfile(snapshot_path, run_dir / "model.json")
    config = {
        "schema": "klinehouse-shadow-pilot-v2",
        "start_at": start,
        "end_at": start + 7 * 86400,
        "start_utc": utc(start),
        "end_utc": utc(start + 7 * 86400),
        "symbols": list(SYMBOLS),
        "snapshot_sha256": file_hash(run_dir / "model.json"),
        "issuance_deadline_seconds": 30,
        "require_commit_ack": True,
        "decision_delay_seconds": 10,
        "poll_seconds": 30,
        "clock_probe_seconds": 300,
        "max_clock_age_seconds": 600,
        "max_storage_bytes": 512 * 1024**2,
        "min_free_bytes": 40 * 1024**3,
        "metrics_lag_seconds": 900,
        "spot_warmup_minutes": 63,
        "target": "sign(spot close at anchor+360 minus spot close at anchor+60); flat iff exactly equal",
        "models": ["baseline", "price_control", "positions"],
        "training": "Frozen July2026 coefficients, normalization and prior class frequency; no online fitting",
        "availability": "Only received responses with content times >=activation; no preactivation backfill. Latest observed revision before issue is allowed. Forecasts are never regenerated.",
        "positions_mapping": "Live REST sumOpenInterest and longShortRatio; archive-equivalent timing/semantics NOT certified. Fixed 15minute label lag plus actual receipt cutoff.",
        "failures": "Missing spot context/clock outage/late issue => explicit missing. Missing any asset position context => both corrections equal baseline on all assets. Outcomes can arrive later, preserving receipt provenance.",
        "evaluation": "Seven-day engineering pilot only. All scheduled anchors accounted for. Descriptive three-class Brier, natural logloss and moving direction accuracy. No significance, superiority, profitability or universal claim.",
        "orders": False,
        "historical_holdouts_opened": False,
        "stop": "No anchors after end_at-360. Collect final outcomes through end_at+120, then exit successfully. Stop early on disk budget exhaustion; never delete evidence.",
        "code": {name: file_hash(ROOT / name) for name in CODE},
    }
    with (run_dir / "config.json").open("x") as handle:
        json.dump(config, handle, sort_keys=True, indent=2)
        handle.write("\n")
    # Runtime files are isolated from subsequent research edits in the checkout.
    for name in CODE:
        target = run_dir / "runtime" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    for name in ("research", "scripts"):
        (run_dir / "runtime" / name / "__init__.py").touch()
    with Ledger(run_dir / "events.sqlite", config):
        pass
    return config


class Cache:
    """Latest received versions; every value retains its original receipt id."""

    def __init__(self):
        self.data = {(source, symbol): {} for source in URLS for symbol in SYMBOLS}
        self.last_body = {}
        self.revisions = 0

    def add(self, seq, payload):
        source, symbol = payload["source"], payload["symbol"]
        self.last_body[source, symbol] = payload["body_sha256"]
        target = self.data[source, symbol]
        for row, stamp in zip(payload["body"], payload["content_times"]):
            old = target.get(int(stamp))
            if old is not None and old[0] != row:
                self.revisions += 1
            target[int(stamp)] = (row, seq)

    def context(self, anchor):
        spot, metrics, ids = {}, {}, set()
        for symbol in SYMBOLS:
            candles = self.data["spot_klines", symbol]
            selected = [
                (t, candles[t])
                for t in range(anchor - 3780, anchor, 60)
                if t in candles
            ]
            spot[symbol] = {
                "time": np.array([t for t, _ in selected], dtype=np.int64),
                **{
                    name: np.array([float(record[0][index]) for _, record in selected])
                    for name, index in (
                        ("close", 4),
                        ("volume", 5),
                        ("trades", 8),
                        ("taker", 9),
                    )
                },
            }
            ids.update(record[1] for _, record in selected)
            metrics[symbol] = []
            oi, ratio = (
                self.data["open_interest", symbol],
                self.data["position_ratio", symbol],
            )
            for stamp in range(anchor - 4500, anchor - 899, 300):
                if stamp in oi and stamp in ratio:
                    metrics[symbol].append(
                        {
                            "label": stamp,
                            "oi": oi[stamp][0]["sumOpenInterest"],
                            "position_ratio": ratio[stamp][0]["longShortRatio"],
                        }
                    )
                    ids.update((oi[stamp][1], ratio[stamp][1]))
        return spot, metrics, sorted(ids)


class Runner:
    def __init__(
        self,
        run_dir,
        *,
        clock=time.time,
        monotonic=time.monotonic,
        fetcher=fetch,
        probe=clock_probe,
    ):
        self.root = Path(run_dir).resolve()
        self.clock, self.fetcher, self.probe = clock, fetcher, probe
        self.monotonic = monotonic
        self.config = json.loads((self.root / "config.json").read_text())
        self.snapshot = json.loads((self.root / "model.json").read_text())
        if file_hash(self.root / "model.json") != self.config["snapshot_sha256"]:
            raise ValueError("frozen model changed")
        validate_snapshot(self.snapshot)
        for name, expected in self.config["code"].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError(f"runtime code changed: {name}")
        self.ledger = Ledger(self.root / "events.sqlite", self.config, clock=clock)
        self.cache = Cache()
        self.decided, self.scored, self.forecasts = set(), set(), {}
        recorded = {}
        self.unacknowledged = set()
        self.ineligible = set()
        for event in self.ledger.rows():
            p, kind = event["payload"], event["kind"]
            if kind == "receipt":
                self.cache.add(event["seq"], p)
            elif kind in ("forecast", "missing"):
                key = (p["anchor"], p["symbol"])
                self.decided.add(key)
                if kind == "forecast":
                    recorded[key] = p
                    self.unacknowledged.add(key)
            elif kind == "forecast_commit":
                key = (p["anchor"], p["symbol"])
                self.unacknowledged.discard(key)
                if p["eligible"]:
                    self.forecasts[key] = recorded[key]
                else:
                    self.ineligible.add(key)
            elif kind == "outcome":
                self.scored.add((p["anchor"], p["symbol"]))
        self.clock_checked = 0
        self.clock_checked_mono = 0
        self.clock_ok = False
        self.errors = {}
        self.next_poll = 0
        self.recovery = True
        self.stopped = False
        self.started = clock()
        self.prefetched_anchor = None
        self.last_status = 0
        self.next_clock_retry = 0

    def check_clock(self):
        try:
            result = self.probe()
            self.clock_ok = bool(result["healthy"])
            if self.clock_ok:
                self.errors.pop("clock", None)
            self.clock_checked = self.clock()
            self.clock_checked_mono = self.monotonic()
            self.ledger.append("clock_check", result)
        except Exception as exc:
            self.clock_ok = False
            self.errors["clock"] = type(exc).__name__
            self.clock_checked = self.clock()
            self.clock_checked_mono = self.monotonic()
            self.ledger.append("clock_error", {"error": type(exc).__name__})

    def valid_clock(self, now=None):
        now = self.clock() if now is None else now
        age = now - self.clock_checked
        elapsed = self.monotonic() - self.clock_checked_mono
        return (
            self.clock_ok
            and 0 <= age <= self.config["max_clock_age_seconds"]
            and elapsed >= 0
            and abs(age - elapsed) <= 2
        )

    def collect(self):
        if not self.valid_clock():
            return
        jobs = [(source, symbol) for source in URLS for symbol in SYMBOLS]
        receipts, failures = [], []
        with ThreadPoolExecutor(max_workers=9) as pool:
            futures = {
                pool.submit(
                    self.fetcher,
                    source,
                    symbol,
                    self.config["start_at"],
                    self.config["end_at"],
                    recovery=self.recovery,
                ): (source, symbol)
                for source, symbol in jobs
            }
            for future in as_completed(futures):
                source, symbol = futures[future]
                key = source + ":" + symbol
                try:
                    result = future.result()
                    if result is not None:
                        receipts.append(result)
                    self.errors.pop(key, None)
                except Exception as exc:
                    message = str(exc)[:160]
                    self.errors[key] = message
                    failures.append(
                        {
                            "source": source,
                            "symbol": symbol,
                            "error_type": type(exc).__name__,
                            "message": message,
                        }
                    )
        if not self.valid_clock():
            self.ledger.append(
                "clock_rejected_batch", {"responses_discarded": len(receipts)}
            )
            self.recovery = True
            self.next_poll = self.clock() + self.config["poll_seconds"]
            return
        for p in sorted(receipts, key=lambda x: x["received_at"]):
            key = (p["source"], p["symbol"])
            if not p["body"] or self.cache.last_body.get(key) == p["body_sha256"]:
                continue
            received_at = p.pop("received_at")
            identifier = self.ledger.append("receipt", p, received_at=received_at)
            self.cache.add(identifier, p)
        for failure in failures:
            self.ledger.append("source_error", failure)
        self.recovery = any(":" in key for key in self.errors)
        self.next_poll = self.clock() + self.config["poll_seconds"]

    def decide(self):
        now = self.clock()
        first = math.ceil(self.config["start_at"] / 300) * 300
        last = min(
            math.floor(now / 300) * 300,
            math.floor((self.config["end_at"] - 360) / 300) * 300,
        )
        for anchor in range(first, last + 1, 300):
            pending = [s for s in SYMBOLS if (anchor, s) not in self.decided]
            if not pending or now < anchor + self.config["decision_delay_seconds"]:
                continue
            reason, prediction = None, None
            ids = []
            if now > anchor + self.config["issuance_deadline_seconds"]:
                reason = "missed_deadline_or_process_downtime"
            elif not self.valid_clock(now):
                reason = "clock_unverified_or_stale"
            else:
                try:
                    spot, metrics, ids = self.cache.context(anchor)
                    prediction = predict(self.snapshot, spot, metrics, anchor)
                except (ValueError, KeyError) as exc:
                    reason = "incomplete_or_invalid_context: " + str(exc)[:160]
            for symbol in pending:
                if self.clock() > anchor + self.config["issuance_deadline_seconds"]:
                    reason = "missed_deadline_or_process_downtime"
                if reason:
                    self.ledger.append(
                        "missing",
                        {"anchor": anchor, "symbol": symbol, "reason": reason},
                    )
                else:
                    p = {
                        "anchor": anchor,
                        "symbol": symbol,
                        "issued_at": self.clock(),
                        "snapshot_sha256": self.config["snapshot_sha256"],
                        "input_ids": ids,
                        **prediction[symbol],
                    }
                    forecast_seq = self.ledger.append("forecast", p)
                    # Observe time only after the FULL-synchronous COMMIT returns.
                    committed_at = self.clock()
                    clock_healthy = self.valid_clock(committed_at)
                    eligible = (
                        committed_at
                        <= anchor + self.config["issuance_deadline_seconds"]
                        and clock_healthy
                    )
                    forecast_hash = self.ledger.connection.execute(
                        "SELECT digest FROM events WHERE seq=?", (forecast_seq,)
                    ).fetchone()[0]
                    self.ledger.append(
                        "forecast_commit",
                        {
                            "anchor": anchor,
                            "symbol": symbol,
                            "forecast_seq": forecast_seq,
                            "forecast_hash": forecast_hash,
                            "committed_at": committed_at,
                            "clock_healthy": clock_healthy,
                            "eligible": eligible,
                        },
                    )
                    if eligible:
                        self.forecasts[anchor, symbol] = p
                    else:
                        self.ineligible.add((anchor, symbol))
                self.decided.add((anchor, symbol))

    def resolve(self):
        if not self.valid_clock():
            return
        now = self.clock()
        for key, forecast in self.forecasts.items():
            anchor, symbol = key
            if key in self.scored or now < anchor + 360:
                continue
            source = self.cache.data["spot_klines", symbol]
            if anchor not in source or anchor + 300 not in source:
                continue
            entry, exit_ = source[anchor], source[anchor + 300]
            price_a, price_b = entry[0][4], exit_[0][4]
            label = outcome_label(price_a, price_b)
            self.ledger.append(
                "outcome",
                {
                    "anchor": anchor,
                    "symbol": symbol,
                    "label": label,
                    "return_start": anchor + 60,
                    "return_end": anchor + 360,
                    "return": math.log(float(Decimal(price_b) / Decimal(price_a))),
                    "entry_price": price_a,
                    "exit_price": price_b,
                    "entry_receipt_id": entry[1],
                    "exit_receipt_id": exit_[1],
                    "input_ids": sorted({entry[1], exit_[1]}),
                },
            )
            self.scored.add(key)

    def status(self, phase="running"):
        now = self.clock()
        status = {
            **self.ledger.status(),
            "phase": phase,
            "pid": os.getpid(),
            "updated_at": now,
            "updated_utc": utc(now),
            "start_at": self.config["start_at"],
            "end_at": self.config["end_at"],
            "end_utc": self.config["end_utc"],
            "clock_ok": self.clock_ok,
            "source_errors": self.errors,
            "revisions": self.cache.revisions,
            "forecasts": len(self.forecasts),
            "outcomes": len(self.scored),
            "pending_outcomes": len(self.forecasts) - len(self.scored),
            "unacknowledged_forecasts": len(self.unacknowledged),
            "ineligible_forecasts": len(self.ineligible),
            "latest_content": {
                source + ":" + symbol: max(
                    self.cache.data[source, symbol], default=None
                )
                for source in URLS
                for symbol in SYMBOLS
            },
            "disk_bytes": sum(
                p.stat().st_size for p in self.root.rglob("*") if p.is_file()
            ),
            "free_bytes": shutil.disk_usage(self.root).free,
            "status_note": "Engineering pilot in progress; no validated predictive edge or trading results",
        }
        write_atomic(self.root / "status.json", status)
        self.last_status = now
        return status

    def run(self, *, max_cycles=None):
        self.ledger.append(
            "process_started", {"pid": os.getpid(), "code": self.config["code"]}
        )
        count = 0
        try:
            while not self.stopped:
                now = self.clock()
                if now >= self.config["end_at"] + 120:
                    self.decide()
                    self.resolve()
                    self.ledger.append(
                        "completed",
                        {"pending_outcomes": len(self.forecasts) - len(self.scored)},
                    )
                    self.status("completed")
                    break
                if now - self.clock_checked >= self.config["clock_probe_seconds"] or (
                    not self.valid_clock(now)
                    and self.monotonic() >= self.next_clock_retry
                ):
                    self.check_clock()
                    self.next_clock_retry = self.monotonic() + 30
                # Fetch fresh closed candles before the decision, with a real deadline.
                anchor = math.floor(now / 300) * 300
                decision_due = (
                    anchor + 5 <= now <= anchor + 20
                    and self.prefetched_anchor != anchor
                    and any((anchor, s) not in self.decided for s in SYMBOLS)
                )
                if now >= self.next_poll or decision_due:
                    self.collect()
                    self.resolve()
                    if decision_due:
                        self.prefetched_anchor = anchor
                self.decide()
                if self.clock() - self.last_status >= 30:
                    status = self.status()
                    if (
                        status["disk_bytes"] >= self.config["max_storage_bytes"]
                        or status["free_bytes"] < self.config["min_free_bytes"]
                    ):
                        self.ledger.append(
                            "resource_stop",
                            {
                                "disk_bytes": status["disk_bytes"],
                                "free_bytes": status["free_bytes"],
                            },
                        )
                        self.status("resource_stopped")
                        break
                count += 1
                if max_cycles is not None and count >= max_cycles:
                    break
                time.sleep(1)
        finally:
            if self.stopped:
                self.ledger.append(
                    "process_stopped",
                    {"pid": os.getpid(), "reason": "termination_signal"},
                )
                self.status("stopped")
            self.ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--run-dir", required=True, type=Path)
    init.add_argument("--snapshot", required=True, type=Path)
    run = sub.add_parser("run")
    run.add_argument("--run-dir", required=True, type=Path)
    run.add_argument("--max-cycles", type=int)
    status = sub.add_parser("status")
    status.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "init":
        print(json.dumps(initialize(args.run_dir, args.snapshot), indent=2))
    elif args.command == "status":
        print((args.run_dir / "status.json").read_text())
    else:
        with (args.run_dir / "runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            config = json.loads((args.run_dir / "config.json").read_text())
            if (
                time.time() >= config["end_at"] + 120
                and (args.run_dir / "status.json").exists()
            ):
                previous = json.loads((args.run_dir / "status.json").read_text())
                if previous.get("phase") == "completed":
                    return
            while time.time() < config["start_at"]:
                time.sleep(min(1, config["start_at"] - time.time()))
            runner = Runner(args.run_dir)

            def stop(*_):
                runner.stopped = True

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            runner.run(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
