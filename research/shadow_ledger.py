"""Durable append-only evidence for a separately authorized prospective trial.

No historical reader, exchange client or trading endpoint is imported here.
The caller supplies public data and its original receipt time. A hash chain
detects modification relative to the committed head; external head receipts
are still needed to detect replacement of an entire database or tail removal.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MODELS = ("baseline", "price_control", "positions")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _event_hash(seq: int, kind: str, received_ns: int, payload: dict, previous: str):
    return sha256(
        canonical(
            dict(
                seq=seq,
                kind=kind,
                received_ns=received_ns,
                payload=payload,
                prev_hash=previous,
            )
        )
    )


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    received_at REAL NOT NULL,
    received_ns INTEGER NOT NULL,
    anchor INTEGER,
    symbol TEXT,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    digest TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS events_kind_seq ON events(kind, seq);
CREATE INDEX IF NOT EXISTS events_anchor_symbol ON events(anchor, symbol, kind);
CREATE UNIQUE INDEX IF NOT EXISTS events_unique_prediction
    ON events(anchor, symbol) WHERE kind IN ('forecast', 'missing');
CREATE UNIQUE INDEX IF NOT EXISTS events_unique_outcome
    ON events(anchor, symbol) WHERE kind = 'outcome';
CREATE UNIQUE INDEX IF NOT EXISTS events_unique_commit_ack
    ON events(anchor, symbol) WHERE kind = 'forecast_commit';
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
    BEGIN SELECT RAISE(ABORT, 'append-only events'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
    BEGIN SELECT RAISE(ABORT, 'append-only events'); END;
CREATE TRIGGER IF NOT EXISTS meta_no_update BEFORE UPDATE ON meta
    BEGIN SELECT RAISE(ABORT, 'immutable configuration'); END;
CREATE TRIGGER IF NOT EXISTS meta_no_delete BEFORE DELETE ON meta
    BEGIN SELECT RAISE(ABORT, 'immutable configuration'); END;
"""


class Ledger:
    """A synchronous, bounded-memory SQLite event journal.

    ``clock`` is injectable for deterministic tests; production uses time.time.
    Explicit receipt times describe the completed HTTP response, while forecast
    deadlines additionally use the actual append clock so a later process cannot
    retroactively emit a forecast. Runs with ``require_commit_ack`` also require
    a separate acknowledgement observed after the forecast's durable commit.
    ``read_only`` never creates or checkpoints.
    """

    def __init__(self, path, config=None, *, clock=time.time, read_only=False):
        self.path = Path(path).resolve()
        self.clock = clock
        self.read_only = read_only
        # Only this writer can attest a commit it actually observed returning.
        # A restarted writer must leave old unacknowledged forecasts unscored.
        self._pending_commits = {}
        if read_only:
            self.connection = sqlite3.connect(
                f"{self.path.as_uri()}?mode=ro", uri=True, timeout=10
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, timeout=10)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout=10000")
        if not read_only:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA wal_autocheckpoint=256")
            self.connection.executescript(SCHEMA)
        try:
            found = self.connection.execute(
                "SELECT value FROM meta WHERE key='config'"
            ).fetchone()
            if found is None:
                if config is None or read_only:
                    raise ValueError("new ledger requires prospective configuration")
                normalized = json.loads(canonical(config))
                self._validate_config(normalized)
                encoded = canonical(normalized)
                with self.connection:
                    self.connection.execute(
                        "INSERT INTO meta VALUES ('config', ?)", (encoded,)
                    )
                    self.connection.execute(
                        "INSERT INTO meta VALUES ('config_sha256', ?)",
                        (sha256(encoded),),
                    )
                self.config = normalized
            else:
                self.config = json.loads(found[0])
                if config is not None and canonical(config) != canonical(self.config):
                    raise ValueError("existing ledger configuration differs")
            self._validate_config(self.config)
            self.config_sha256 = sha256(canonical(self.config))
            self.verify()
        except BaseException:
            self.connection.close()
            raise

    @staticmethod
    def _validate_config(config):
        start = _finite(config.get("start_at"), "start_at")
        end = _finite(config.get("end_at"), "end_at")
        deadline = _finite(config.get("issuance_deadline_seconds", 30), "deadline")
        if start < 0 or end <= start or end - start > 8 * 86400:
            raise ValueError("prospective trial must last at most eight days")
        if not 0 < deadline < 60:
            raise ValueError("forecast deadline must precede target entry")
        if type(config.get("require_commit_ack", False)) is not bool:
            raise ValueError("require_commit_ack must be boolean")
        digest = config.get("snapshot_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("frozen snapshot SHA256 required")

    def _prediction_key(self, payload):
        anchor = _finite(payload.get("anchor"), "anchor")
        if anchor != int(anchor) or anchor % 300:
            raise ValueError("forecast anchor must be aligned to five minutes")
        anchor = int(anchor)
        if not self.config["start_at"] <= anchor <= self.config["end_at"] - 360:
            raise ValueError("anchor outside prospective trial")
        symbol = payload.get("symbol")
        if symbol not in SYMBOLS:
            raise ValueError("unregistered symbol")
        return anchor, symbol

    def _inputs(self, payload, cutoff, *, required=True):
        identifiers = payload.get("input_ids", [])
        if not isinstance(identifiers, list) or (required and not identifiers):
            raise ValueError("input receipt identifiers required")
        if any(type(identifier) is not int for identifier in identifiers):
            raise ValueError("invalid input receipt identifier")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate input receipt identifier")
        for identifier in identifiers:
            row = self.connection.execute(
                "SELECT kind,received_at,payload FROM events WHERE seq=?",
                (identifier,),
            ).fetchone()
            if row is None or row["kind"] != "receipt" or row["received_at"] > cutoff:
                raise ValueError("input was not received before decision")
            content_times = json.loads(row["payload"]).get("content_times", [])
            if any(t > cutoff for t in content_times):
                raise ValueError("future content used in decision")

    def _validate_event(self, kind, payload, received_at, now):
        if not isinstance(kind, str) or not kind or len(kind) > 80:
            raise ValueError("invalid event kind")
        if not isinstance(payload, dict):
            raise ValueError("event payload must be an object")
        if received_at < self.config["start_at"] or received_at > now + 1:
            raise ValueError("receipt time outside observed prospective time")
        anchor, symbol = payload.get("anchor"), payload.get("symbol")
        if kind == "receipt":
            if received_at > self.config["end_at"] + 120:
                raise ValueError("receipt after trial collection deadline")
            times = payload.get("content_times")
            if not isinstance(times, list):
                raise ValueError("declared source content times required")
            for stamp in times:
                stamp = _finite(stamp, "source content timestamp")
                if (
                    stamp < self.config["start_at"]
                    or stamp > self.config["end_at"]
                    or stamp > received_at + 1
                ):
                    raise ValueError(
                        "source content outside prospective receipt bounds"
                    )
            if not isinstance(payload.get("source"), str) or "body" not in payload:
                raise ValueError("public source and original response required")
            if payload.get("body_sha256") != sha256(canonical(payload["body"])):
                raise ValueError("public response digest differs")
            symbol = payload.get("symbol")
            if symbol is not None and symbol not in SYMBOLS:
                raise ValueError("unregistered symbol")
        elif kind in ("forecast", "forecast_commit", "missing", "outcome"):
            anchor, symbol = self._prediction_key(payload)
            if kind == "forecast":
                issued = _finite(payload.get("issued_at"), "issued_at")
                deadline = anchor + self.config.get("issuance_deadline_seconds", 30)
                if not anchor <= issued <= received_at <= now <= deadline:
                    raise ValueError("forecast is late or backdated")
                if payload.get("snapshot_sha256") != self.config["snapshot_sha256"]:
                    raise ValueError("forecast frozen snapshot differs")
                models = payload.get("models")
                if not isinstance(models, dict) or set(models) != set(MODELS):
                    raise ValueError("all three registered models required")
                for probabilities in models.values():
                    if not isinstance(probabilities, list) or len(probabilities) != 3:
                        raise ValueError("down/flat/up probability vector required")
                    numbers = [_finite(v, "probability") for v in probabilities]
                    if any(p <= 0 or p >= 1 for p in numbers) or not math.isclose(
                        sum(numbers), 1.0, abs_tol=1e-10, rel_tol=0
                    ):
                        raise ValueError(
                            "probabilities must be positive and sum to one"
                        )
                self._inputs(payload, issued)
            elif kind == "forecast_commit":
                forecast_seq = payload.get("forecast_seq")
                if type(forecast_seq) is not int:
                    raise ValueError("forecast commit requires an event identifier")
                forecast = self.connection.execute(
                    "SELECT kind,anchor,symbol,received_at,digest FROM events WHERE seq=?",
                    (forecast_seq,),
                ).fetchone()
                if (
                    forecast is None
                    or forecast["kind"] != "forecast"
                    or forecast["anchor"] != anchor
                    or forecast["symbol"] != symbol
                    or forecast["digest"] != payload.get("forecast_hash")
                ):
                    raise ValueError("forecast commit provenance differs")
                committed = _finite(payload.get("committed_at"), "committed_at")
                if not forecast["received_at"] <= committed <= received_at <= now:
                    raise ValueError("invalid post-commit acknowledgement timing")
                eligible = payload.get("eligible")
                healthy = payload.get("clock_healthy")
                deadline = anchor + self.config.get("issuance_deadline_seconds", 30)
                if (
                    type(eligible) is not bool
                    or type(healthy) is not bool
                    or eligible != (committed <= deadline and healthy)
                ):
                    raise ValueError("commit eligibility differs from time or clock gate")
                observed = self._pending_commits.get(forecast_seq)
                if observed is not None and committed < observed:
                    raise ValueError("acknowledgement predates observed durable commit")
                if eligible and observed is None:
                    raise ValueError("cannot certify a commit not observed by this writer")
            elif kind == "missing":
                if now < anchor:
                    raise ValueError("cannot mark a future forecast missing")
                if not isinstance(payload.get("reason"), str) or not payload["reason"]:
                    raise ValueError("missing prediction reason required")
            else:
                if now < anchor + 360 or received_at < anchor + 360:
                    raise ValueError("outcome has not matured")
                if (
                    payload.get("return_start") != anchor + 60
                    or payload.get("return_end") != anchor + 360
                ):
                    raise ValueError("outcome target interval differs")
                if type(payload.get("label")) is not int or payload["label"] not in (
                    0,
                    1,
                    2,
                ):
                    raise ValueError("outcome label must be down=0, flat=1 or up=2")
                forecast = self.connection.execute(
                    "SELECT seq FROM events WHERE kind='forecast' AND anchor=? AND symbol=?",
                    (anchor, symbol),
                ).fetchone()
                if forecast is None:
                    raise ValueError("outcome requires an earlier forecast")
                if self.config.get("require_commit_ack", False):
                    ack = self.connection.execute(
                        "SELECT payload FROM events WHERE kind='forecast_commit' "
                        "AND anchor=? AND symbol=?",
                        (anchor, symbol),
                    ).fetchone()
                    if ack is None or json.loads(ack["payload"])["eligible"] is not True:
                        raise ValueError("outcome requires an eligible post-commit acknowledgement")
                self._inputs(payload, received_at)
                if "return" in payload:
                    _finite(payload["return"], "realized return")
        return anchor, symbol

    def append(self, kind, payload, received_at=None):
        """Atomically validate and append one event; return its sequence number."""
        if self.read_only:
            raise ValueError("read-only ledger")
        now = _finite(self.clock(), "current clock")
        stamp = now if received_at is None else _finite(received_at, "received_at")
        # Round trip freezes the caller's object and rejects NaN and infinity.
        body = json.loads(canonical(payload))
        if kind == "receipt" and "body" in body:
            body.setdefault("body_sha256", sha256(canonical(body["body"])))
        received_ns = round(stamp * 1_000_000_000)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            anchor, symbol = self._validate_event(kind, body, stamp, now)
            previous = self.connection.execute(
                "SELECT seq,digest,received_ns FROM events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            seq = 1 if previous is None else previous["seq"] + 1
            prior_hash = self.config_sha256 if previous is None else previous["digest"]
            # HTTP calls may complete in a different order, so receipt timestamps
            # need not be monotonic; sequence and hashes preserve append order.
            digest = _event_hash(seq, kind, received_ns, body, prior_hash)
            self.connection.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    seq,
                    kind,
                    stamp,
                    received_ns,
                    anchor,
                    symbol,
                    canonical(body),
                    prior_hash,
                    digest,
                ),
            )
        # The SQLite context manager above has returned from COMMIT and its FULL
        # durability synchronization. Never use the pre-transaction timestamp to
        # certify that this write reached durable storage before the deadline.
        if kind == "forecast":
            self._pending_commits[seq] = _finite(self.clock(), "post-commit clock")
        elif kind == "forecast_commit":
            self._pending_commits.pop(body["forecast_seq"], None)
        return seq

    def rows(self, kind=None, *, after_id=0):
        """Stream a consistent SQLite statement snapshot without loading the log."""
        query = "SELECT * FROM events WHERE seq>?"
        arguments = [after_id]
        if kind is not None:
            query += " AND kind=?"
            arguments.append(kind)
        query += " ORDER BY seq"
        cursor = self.connection.execute(query, arguments)
        try:
            for row in cursor:
                event = dict(row)
                event["payload"] = json.loads(event["payload"])
                yield event
        finally:
            cursor.close()

    def last(self, kind=None):
        query = "SELECT * FROM events"
        args = ()
        if kind is not None:
            query += " WHERE kind=?"
            args = (kind,)
        row = self.connection.execute(
            query + " ORDER BY seq DESC LIMIT 1", args
        ).fetchone()
        if row is None:
            return None
        event = dict(row)
        event["payload"] = json.loads(event["payload"])
        return event

    def verify(self):
        """Check storage, immutable configuration and every event's hash chain."""
        integrity = self.connection.execute("PRAGMA quick_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError("SQLite integrity verification failed")
        config_row = self.connection.execute(
            "SELECT value FROM meta WHERE key='config_sha256'"
        ).fetchone()
        if config_row is None or config_row[0] != self.config_sha256:
            raise ValueError("configuration digest differs")
        previous, count = self.config_sha256, 0
        for event in self.rows():
            count += 1
            payload = event["payload"]
            if (
                event["seq"] != count
                or event["prev_hash"] != previous
                or round(event["received_at"] * 1_000_000_000) != event["received_ns"]
                or event["anchor"] != payload.get("anchor")
                or event["symbol"] != payload.get("symbol")
                or event["digest"]
                != _event_hash(
                    count, event["kind"], event["received_ns"], payload, previous
                )
            ):
                raise ValueError(f"event hash chain differs at sequence {count}")
            previous = event["digest"]
        return dict(
            events=count, head_sha256=previous, config_sha256=self.config_sha256
        )

    def status(self):
        counts = dict(
            self.connection.execute(
                "SELECT kind,count(*) FROM events GROUP BY kind"
            ).fetchall()
        )
        last = self.last()
        return dict(
            counts=counts,
            last_seq=last["seq"] if last else 0,
            last_received_at=last["received_at"] if last else None,
            head_sha256=last["digest"] if last else self.config_sha256,
            config_sha256=self.config_sha256,
        )

    def checkpoint(self):
        if self.read_only:
            return None
        return tuple(
            self.connection.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        )

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
