#!/usr/bin/env python3
"""Separate, delayed QI20 forecasts from authorized new prospective journals.

No HTTP client, archive reader, credential loader or original-pilot writer is
used. The fixed first-day fit precedes all new emissions. Immutable forecasts,
post-fsync acknowledgements and final eligibility seals live in a NEW directory.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "research/depth_information.py"
PLAN = ROOT / "reports/depth-information-20261030/protocol-plan.json"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
CONTROLS = ("baseline", "price_control", "positions")
ARMS = (*CONTROLS, "depth")
PILOT_START = 1793272200
LIMITS = dict(
    deadline=30,
    clock_age=600,
    terminal_clock_age=1800,
    clock_tolerance=0.25,
    min_free_bytes=40 * 1024**3,
    max_bytes=128 * 1024**2,
    reserve_bytes=4 * 1024**2,
)
spec = importlib.util.spec_from_file_location("depth_information_frozen", CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def durable_write(path, value, *, exclusive=True):
    """The caller observes UTC and monotonic time only AFTER this returns."""
    path = Path(path)
    encoded = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    target = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    with target.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    if exclusive:
        # The final name appears atomically, without replacing an existing file.
        # Preserve the private temporary as crash/publication evidence.
        os.link(target, path)
    else:
        os.replace(target, path)
    fsync_directory(path.parent)


def fsync_directory(path):
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def safe_root(path):
    path = Path(path).resolve()
    require(
        not any(
            part.startswith("shadow-pilot")
            or part in ("archive", "holdout", "holdouts")
            for part in path.parts
        ),
        "original pilot and historical stores cannot be output roots",
    )
    return path


def validate_sources(capture_root, pilot_root):
    capture, pilot = Path(capture_root).resolve(), Path(pilot_root).resolve()
    require(
        capture.name.startswith("depth-research-")
        and pilot.name == "shadow-pilot-20261029-v2",
        "only named new prospective source roots are allowed",
    )
    c, p = load(capture / "config.json"), load(pilot / "config.json")
    require(
        c.get("schema") == "klinehouse-depth-research-v1"
        and c.get("capture_only") is True
        and c.get("candidate_predictions") is False
        and c.get("orders") is False,
        "not an authorized capture-only depth journal",
    )
    require(
        p.get("schema") == "klinehouse-shadow-pilot-v2"
        and p.get("start_at") == PILOT_START
        and p.get("orders") is False
        and p.get("require_commit_ack") is True,
        "not the authorized new shadow-pilot activation",
    )
    require(
        c["activation_at"] >= p["start_at"]
        and c["end_at"] <= p["end_at"]
        and c["symbols"] == p["symbols"] == list(SYMBOLS),
        "source activation, end or assets differ",
    )
    require(
        sha(c) == load(capture / "config-sha256.json")["sha256"]
        and file_sha(pilot / "model.json") == p["snapshot_sha256"],
        "source configuration/model hash mismatch",
    )
    require(
        (capture / "depth-events.sqlite").is_file()
        and (pilot / "events.sqlite").is_file(),
        "prospective journal absent",
    )
    for directory, database, config in (
        (capture, "depth-events.sqlite", c),
        (pilot, "events.sqlite", p),
    ):
        with closing(
            sqlite3.connect((directory / database).as_uri() + "?mode=ro", uri=True)
        ) as connection:
            require(
                dict(connection.execute("SELECT key,value FROM meta"))
                == {"config": canonical(config), "config_sha256": sha(config)},
                "source SQLite identity differs from its frozen configuration",
            )
    return capture, pilot, c, p


def initialize(root, *, capture_root, pilot_root, plan_path=PLAN, clock=time.time):
    root = safe_root(root)
    capture, pilot, c, p = validate_sources(capture_root, pilot_root)
    require(
        not any(
            root == source or root in source.parents or source in root.parents
            for source in (capture, pilot)
        ),
        "output must be separate from both source directories",
    )
    require(
        shutil.disk_usage(root.parent).free >= LIMITS["min_free_bytes"],
        "insufficient free space",
    )
    first = math.ceil(c["activation_at"] / 300) * 300
    cutoff = first + 86400
    require(
        cutoff + 360 <= c["end_at"],
        "capture does not permit future evaluation after fixed training",
    )
    plan = load(plan_path)
    require(
        plan.get("schema") == "depth-information-research-plan-v1"
        and plan["core_code"]["research/depth_information.py"] == file_sha(CORE),
        "registered depth plan/core differs",
    )
    root.mkdir(exist_ok=False)
    for name in (
        "forecasts",
        "acks",
        "eligibility",
        "missing",
        "outcomes",
        "model-receipts",
        "runtime/scripts",
        "runtime/research",
    ):
        (root / name).mkdir(parents=True)
    shutil.copyfile(Path(__file__), root / "runtime/scripts/run_depth_information.py")
    shutil.copyfile(CORE, root / "runtime/research/depth_information.py")
    shutil.copyfile(plan_path, root / "protocol-plan.json")
    identities = dict(
        capture=dict(root=str(capture), config=c, config_sha256=sha(c)),
        pilot=dict(root=str(pilot), config=p, config_sha256=sha(p)),
        first_anchor=first,
        training_cutoff=cutoff,
    )
    durable_write(root / "source-identities.json", identities)
    config = dict(
        schema="klinehouse-depth-information-run-v1",
        created_at=clock(),
        capture_root=str(capture),
        pilot_root=str(pilot),
        capture_config_sha256=sha(c),
        pilot_config_sha256=sha(p),
        original_snapshot_sha256=p["snapshot_sha256"],
        first_anchor=first,
        training_cutoff=cutoff,
        training_cutoff_utc=utc(cutoff),
        end_at=c["end_at"],
        end_utc=utc(c["end_at"]),
        symbols=list(SYMBOLS),
        limits=LIMITS,
        code={"script": file_sha(Path(__file__)), "core": file_sha(CORE)},
        plan_sha256=file_sha(plan_path),
        source_identities_sha256=file_sha(root / "source-identities.json"),
        orders=False,
        network=False,
        original_pilot_read_only=True,
        rule="One shared QI20 coefficient, fixed first 24h, one immutable fit; missing books fall back exactly to original baseline; no retro-emission or retro-ack.",
    )
    durable_write(root / "config.json", config)
    durable_write(root / "config-sha256.json", {"sha256": sha(config)})
    durable_write(
        root / "status.json",
        dict(
            phase="initialized",
            updated_at=clock(),
            pid=None,
            training_cutoff=cutoff,
            candidate_forecasts=0,
        ),
    )
    return config


def read_config(root):
    root = safe_root(root)
    c = load(root / "config.json")
    require(
        c.get("schema") == "klinehouse-depth-information-run-v1"
        and sha(c) == load(root / "config-sha256.json")["sha256"]
        and c["code"] == {"script": file_sha(Path(__file__)), "core": file_sha(CORE)}
        and c["plan_sha256"] == file_sha(root / "protocol-plan.json")
        and c["source_identities_sha256"] == file_sha(root / "source-identities.json")
        and c["limits"] == LIMITS
        and c.get("orders") is False
        and c.get("network") is False,
        "frozen pipeline code/configuration changed",
    )
    _, _, depth, pilot = validate_sources(c["capture_root"], c["pilot_root"])
    require(
        sha(depth) == c["capture_config_sha256"]
        and sha(pilot) == c["pilot_config_sha256"]
        and c["first_anchor"] == math.ceil(depth["activation_at"] / 300) * 300
        and c["training_cutoff"] == c["first_anchor"] + 86400,
        "source identity or fixed training interval changed",
    )
    return c


class Source:
    """Incrementally verify coherent read-only prefixes; retain only needed rows."""

    def __init__(self, root, kind, *, first_anchor):
        self.root, self.kind = Path(root), kind
        self.config = load(self.root / "config.json")
        self.path = self.root / (
            "events.sqlite" if kind == "pilot" else "depth-events.sqlite"
        )
        self.seq, self.tip = 0, sha(self.config)
        self.rows = {}
        self.by_kind = {}
        self.first_anchor = first_anchor

    def update(self):
        with closing(
            sqlite3.connect(
                self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5
            )
        ) as c:
            c.row_factory = sqlite3.Row
            c.execute("BEGIN")
            meta = dict(c.execute("SELECT key,value FROM meta"))
            require(
                meta
                == {
                    "config": canonical(self.config),
                    "config_sha256": sha(self.config),
                },
                "source metadata changed",
            )
            head = c.execute("SELECT MAX(seq) FROM events").fetchone()[0] or 0
            require(head >= self.seq, "source prefix was truncated")
            if self.seq:
                require(
                    c.execute(
                        "SELECT digest FROM events WHERE seq=?", (self.seq,)
                    ).fetchone()[0]
                    == self.tip,
                    "retained source prefix changed",
                )
            for row in c.execute(
                "SELECT * FROM events WHERE seq>? ORDER BY seq", (self.seq,)
            ):
                payload = json.loads(row["payload"])
                if self.kind == "pilot":
                    envelope = dict(
                        seq=row["seq"],
                        kind=row["kind"],
                        received_ns=row["received_ns"],
                        payload=payload,
                        prev_hash=row["prev_hash"],
                    )
                    received = row["received_ns"] / 1e9
                    require(
                        abs(received - row["received_at"]) <= 1e-6
                        and received >= self.config["start_at"],
                        "pilot event receipt differs",
                    )
                else:
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
                    received = row["recorded_at"]
                require(
                    row["seq"] == self.seq + 1
                    and row["prev_hash"] == self.tip
                    and sha(envelope) == row["digest"]
                    and canonical(payload) == row["payload"],
                    "source event hash chain differs",
                )
                require(
                    row["anchor"] == payload.get("anchor", row["anchor"])
                    and row["symbol"] == payload.get("symbol", row["symbol"]),
                    "source indexed membership differs",
                )
                self.seq, self.tip = row["seq"], row["digest"]
                relevant = row["kind"] in (
                    "forecast",
                    "forecast_commit",
                    "outcome",
                    "missing",
                    "book_receipt",
                    "commit_ack",
                    "invalidated",
                    "clock_probe",
                )
                if relevant and (
                    row["anchor"] is None or row["anchor"] >= self.first_anchor
                ):
                    item = dict(
                        seq=row["seq"],
                        digest=row["digest"],
                        received_at=received,
                        anchor=row["anchor"],
                        symbol=row["symbol"],
                        payload=payload,
                    )
                    self.rows[item["seq"]] = item
                    key = (
                        (row["seq"], None)
                        if row["kind"] == "commit_ack"
                        else (row["anchor"], row["symbol"])
                    )
                    self.by_kind.setdefault(row["kind"], {})[key] = item
            require(self.seq == head, "source snapshot prefix ended unexpectedly")

    def get(self, kind, anchor, symbol):
        return self.by_kind.get(kind, {}).get((anchor, symbol))

    def proof(self):
        return dict(root=str(self.root), event_count=self.seq, chain_tip=self.tip)


def ref(row):
    return dict(seq=row["seq"], digest=row["digest"], received_at=row["received_at"])


def original(source, anchor, symbol):
    row, ack = (
        source.get("forecast", anchor, symbol),
        source.get("forecast_commit", anchor, symbol),
    )
    if row is None or ack is None:
        return None
    p, a = row["payload"], ack["payload"]
    require(
        a["forecast_seq"] == row["seq"] and a["forecast_hash"] == row["digest"],
        "original acknowledgement references another forecast",
    )
    valid = (
        a["eligible"] is True
        and a["clock_healthy"] is True
        and row["received_at"] <= a["committed_at"] <= ack["received_at"] <= anchor + 30
        and anchor <= p["issued_at"] <= anchor + 30
        and p["snapshot_sha256"] == source.config["snapshot_sha256"]
    )
    if not valid:
        return None
    values = np.array([p["models"][arm] for arm in CONTROLS], dtype=float)
    require(
        values.shape == (3, 3)
        and np.isfinite(values).all()
        and np.all((values > 0) & (values < 1))
        and np.allclose(values.sum(1), 1, rtol=0, atol=1e-9),
        "invalid issued original probability vectors",
    )
    return dict(
        models=p["models"],
        forecast=ref(row),
        acknowledgement=ref(ack),
        available_at=ack["received_at"],
    )


def outcome(source, anchor, symbol):
    row = source.get("outcome", anchor, symbol)
    if row is None:
        return None
    p = row["payload"]
    require(
        p["return_start"] == anchor + 60
        and p["return_end"] == anchor + 360
        and row["received_at"] >= anchor + 360,
        "original outcome precedes maturity or target differs",
    )
    entry, exit_ = Decimal(p["entry_price"]), Decimal(p["exit_price"])
    require(
        entry.is_finite()
        and exit_.is_finite()
        and entry > 0
        and exit_ > 0
        and p["label"] == (2 if exit_ > entry else 0 if exit_ < entry else 1),
        "original outcome label differs from exact prices",
    )
    return dict(
        label=p["label"],
        entry_price=p["entry_price"],
        exit_price=p["exit_price"],
        source=ref(row),
        received_at=row["received_at"],
    )


def book(source, anchor, symbol):
    row = source.get("book_receipt", anchor, symbol)
    if row is None:
        return None
    acknowledgements = [
        r
        for r in source.by_kind.get("commit_ack", {}).values()
        if r["payload"]["record_seq"] == row["seq"]
    ]
    invalid = any(
        r["payload"]["record_seq"] == row["seq"]
        for r in source.by_kind.get("invalidated", {}).values()
    )
    if len(acknowledgements) != 1 or invalid:
        return None
    ack = acknowledgements[0]
    a, p = ack["payload"], row["payload"]
    receipt = p["receipt"]
    if not (
        a["record_hash"] == row["digest"]
        and a["eligible"] is True
        and a["clock_healthy"] is True
        and row["received_at"] <= a["committed_at"] <= ack["received_at"]
        and a["committed_at"] <= anchor + 30
        and p["clock_healthy"] is True
        and not p["validation_errors"]
        and source.config["activation_at"]
        <= receipt["requested_at"]
        <= receipt["received_at"]
        < anchor
    ):
        return None
    body = receipt["body"]
    require(
        len(body["bids"]) == len(body["asks"]) == 20, "QI20 source lacks twenty levels"
    )
    bid = sum(Decimal(level[1]) for level in body["bids"])
    ask = sum(Decimal(level[1]) for level in body["asks"])
    qi = float(p["features"]["qi20"])
    require(
        bid.is_finite()
        and ask.is_finite()
        and bid > 0
        and ask > 0
        and math.isfinite(qi)
        and abs(qi) <= 1
        and math.isclose(
            qi, float((bid - ask) / (bid + ask)), abs_tol=1e-14, rel_tol=0
        ),
        "QI20 differs from original depth body",
    )
    return dict(
        qi20=qi,
        receipt=ref(row),
        acknowledgement=ref(ack),
        available_at=ack["received_at"],
    )


def elapsed_ok(start, stop):
    wall, mono = stop[0] - start[0], stop[1] - start[1]
    return (
        math.isfinite(wall)
        and math.isfinite(mono)
        and wall >= 0
        and mono >= 0
        and abs(wall - mono) <= LIMITS["clock_tolerance"]
    )


class Runner:
    def __init__(
        self, root, *, clock=time.time, monotonic=time.monotonic, writer=durable_write
    ):
        self.root, self.config = safe_root(root), read_config(root)
        self.clock, self.monotonic, self.writer = clock, monotonic, writer
        self.capture = Source(
            self.config["capture_root"],
            "depth",
            first_anchor=self.config["first_anchor"],
        )
        self.pilot = Source(
            self.config["pilot_root"], "pilot", first_anchor=self.config["first_anchor"]
        )
        self.model, self.model_receipt = None, None
        self.done, self.pending = set(), {}
        self.changed, self.cached_summary, self.last_status = True, None, None

    def moment(self):
        return self.clock(), self.monotonic()

    def clock_reference(self):
        rows = self.capture.by_kind.get("clock_probe", {})
        if not rows:
            return None
        p = max(rows.values(), key=lambda r: r["seq"])["payload"]
        r = p["receipt"]
        if (
            p.get("healthy") is not True
            or p.get("absolute_offset_bound_seconds", math.inf) > 2
        ):
            return None
        return dict(
            utc=r["received_at"],
            monotonic=r["received_monotonic"],
            absolute_offset_bound_seconds=p["absolute_offset_bound_seconds"],
        )

    def healthy(self, moment, reference, *, lifecycle=False):
        # After the final emission deadline no more HTTP probes are scheduled.
        # A slightly older probe can attest continuous time for final outcomes
        # and shutdown only; emission and fitting always retain the 600s bound.
        last_anchor = math.floor((self.config["end_at"] - 360) / 300) * 300
        terminal = lifecycle and moment[0] > last_anchor + LIMITS["deadline"]
        age_limit = LIMITS["terminal_clock_age"] if terminal else LIMITS["clock_age"]
        return (
            reference is not None
            and 0 <= moment[0] - reference["utc"] <= age_limit
            and elapsed_ok((reference["utc"], reference["monotonic"]), moment)
        )

    def training_data(self):
        times, q, x, y, received, membership = [], [], [], [], [], []
        cutoff = self.config["training_cutoff"]
        for anchor in range(self.config["first_anchor"], cutoff, 300):
            if anchor + 360 > cutoff:
                continue
            rows = [
                (
                    original(self.pilot, anchor, s),
                    book(self.capture, anchor, s),
                    outcome(self.pilot, anchor, s),
                )
                for s in SYMBOLS
            ]
            if any(
                a is None
                or b is None
                or c is None
                or max(a["available_at"], b["available_at"], c["received_at"]) > cutoff
                for a, b, c in rows
            ):
                continue
            times.append(anchor)
            q.append(
                [
                    a["models"]["baseline"][2]
                    / (a["models"]["baseline"][0] + a["models"]["baseline"][2])
                    for a, _, _ in rows
                ]
            )
            x.append([b["qi20"] for _, b, _ in rows])
            y.append([c["label"] for _, _, c in rows])
            received.append([c["received_at"] for _, _, c in rows])
            membership.append(
                dict(
                    anchor=anchor,
                    assets={
                        s: dict(original=a, book=b, outcome=c)
                        for s, (a, b, c) in zip(SYMBOLS, rows)
                    },
                )
            )
        return dict(
            anchors=np.array(times, dtype=np.int64),
            baseline=np.array(q, dtype=float).reshape(-1, 3),
            imbalance=np.array(x, dtype=float).reshape(-1, 3),
            labels=np.array(y, dtype=np.int64).reshape(-1, 3),
            received=np.array(received, dtype=float).reshape(-1, 3),
        ), membership

    def freeze(self):
        require(
            self.clock() >= self.config["training_cutoff"],
            "training cutoff has not arrived",
        )
        reference = self.clock_reference()
        if not self.healthy(self.moment(), reference):
            return False
        path = self.root / "model.json"
        if not path.exists():
            training_path = self.root / "training-data.json"
            if not training_path.exists():
                data, membership = self.training_data()
                self.writer(
                    training_path,
                    dict(
                        data={key: value.tolist() for key, value in data.items()},
                        membership=membership,
                        fixed_cutoff=self.config["training_cutoff"],
                        config_sha256=sha(self.config),
                        source_prefixes=dict(
                            depth=self.capture.proof(), pilot=self.pilot.proof()
                        ),
                    ),
                )
            training = load(training_path)
            require(
                training["fixed_cutoff"] == self.config["training_cutoff"]
                and training["config_sha256"] == sha(self.config),
                "fixed training snapshot changed",
            )
            data = {
                key: np.asarray(
                    value, dtype=np.int64 if key in ("anchors", "labels") else float
                )
                for key, value in training["data"].items()
            }
            for key in ("baseline", "imbalance", "labels", "received"):
                data[key] = data[key].reshape(-1, 3)
            membership = training["membership"]
            state = core.fit_first_day(**data, first_anchor=self.config["first_anchor"])
            value = dict(
                state=state,
                fitted_at=self.clock(),
                fixed_cutoff=self.config["training_cutoff"],
                training_membership=membership,
                training_membership_sha256=sha(membership),
                source_prefixes=training["source_prefixes"],
                config_sha256=sha(self.config),
                training_data_sha256=file_sha(training_path),
            )
            self.writer(path, value)
        value = load(path)
        require(
            value["config_sha256"] == sha(self.config)
            and value["fixed_cutoff"] == self.config["training_cutoff"]
            and value["training_membership_sha256"] == sha(value["training_membership"])
            and value["state"]["cutoff"] == self.config["training_cutoff"],
            "immutable frozen model differs",
        )
        require(
            value["training_data_sha256"] == file_sha(self.root / "training-data.json"),
            "frozen training snapshot bytes changed",
        )
        # Recovery certifies these immutable bytes NOW; it never certifies an old forecast.
        start = self.moment()
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        with (self.root / "training-data.json").open("rb") as handle:
            os.fsync(handle.fileno())
        fsync_directory(self.root)
        observed = self.moment()
        if not elapsed_ok(start, observed) or not self.healthy(observed, reference):
            return False
        receipt = dict(
            model_sha256=file_sha(path),
            observed_durable_at=observed[0],
            observed_monotonic=observed[1],
            clock_reference=reference,
            recovery_of_existing_bytes=True,
        )
        receipt_path = self.root / "model-receipts" / f"{time.time_ns()}.json"
        self.writer(receipt_path, receipt)
        committed = self.moment()
        if not elapsed_ok(observed, committed) or not self.healthy(
            committed, reference
        ):
            return False
        self.model = value["state"]
        self.model_receipt = dict(
            path=str(receipt_path.relative_to(self.root)),
            sha256=file_sha(receipt_path),
            model_sha256=receipt["model_sha256"],
            durable_receipt_observed_at=committed[0],
        )
        self.changed = True
        return True

    def path(self, kind, anchor, symbol):
        return self.root / kind / f"{anchor}-{symbol}.json"

    def missing(self, anchor, symbol, reason):
        path = self.path("missing", anchor, symbol)
        if not path.exists():
            self.writer(
                path,
                dict(
                    anchor=anchor,
                    symbol=symbol,
                    reason=reason,
                    recorded_at=self.clock(),
                ),
            )
            self.changed = True
        self.done.add((anchor, symbol))

    def emit(self, anchor, symbol, control, depth):
        path = self.path("forecasts", anchor, symbol)
        require(not path.exists(), "forecast already exists; no overwrite")
        reference = self.clock_reference()
        issued = self.moment()
        require(self.healthy(issued, reference), "current capture clock unverified")
        require(
            control["available_at"] <= issued[0]
            and (depth is None or depth["available_at"] <= issued[0])
            and self.model_receipt["durable_receipt_observed_at"] <= issued[0],
            "source or model confirmation was not available before candidate issue",
        )
        baseline = control["models"]["baseline"]
        candidate = (
            baseline[:]
            if depth is None or self.model["coefficient"] == 0
            else core.predict(
                self.model,
                np.array([baseline]),
                np.array([depth["qi20"]]),
                anchor=anchor,
            )[0].tolist()
        )
        value = dict(
            schema="depth-information-forecast-v1",
            anchor=anchor,
            symbol=symbol,
            issued_at=issued[0],
            issued_monotonic=issued[1],
            target_start=anchor + 60,
            target_end=anchor + 360,
            models={
                **{arm: control["models"][arm] for arm in CONTROLS},
                "depth": candidate,
            },
            fallback_missing_book=depth is None,
            fallback_zero_coefficient=self.model["coefficient"] == 0,
            qi20=None if depth is None else depth["qi20"],
            original=control,
            depth_book=depth,
            source_prefixes=dict(depth=self.capture.proof(), pilot=self.pilot.proof()),
            model_receipt=self.model_receipt,
            config_sha256=sha(self.config),
            clock_reference=reference,
        )
        self.writer(path, value)
        committed = self.moment()
        primary_ok = (
            elapsed_ok(issued, committed)
            and self.healthy(committed, reference)
            and anchor <= issued[0] <= committed[0] <= anchor + 30
        )
        acknowledgement = dict(
            forecast_sha256=file_sha(path),
            forecast_committed_at=committed[0],
            forecast_committed_monotonic=committed[1],
            primary_clock_healthy=bool(primary_ok),
            clock_reference=reference,
            deadline=anchor + 30,
            recorded_at=self.clock(),
        )
        ack_path = self.path("acks", anchor, symbol)
        self.writer(ack_path, acknowledgement)
        ack_committed = self.moment()
        accepted = bool(
            primary_ok
            and elapsed_ok(committed, ack_committed)
            and self.healthy(ack_committed, reference)
            and ack_committed[0] <= anchor + 30
        )
        seal = dict(
            forecast_sha256=acknowledgement["forecast_sha256"],
            ack_sha256=file_sha(ack_path),
            ack_committed_at=ack_committed[0],
            ack_committed_monotonic=ack_committed[1],
            eligible=accepted,
            clock_healthy=bool(
                self.healthy(ack_committed, reference)
                and elapsed_ok(committed, ack_committed)
            ),
            recorded_at=self.clock(),
            note="Post-ack-fsync observation; absent seals are never regenerated after a crash.",
        )
        self.writer(self.path("eligibility", anchor, symbol), seal)
        self.changed = True
        self.done.add((anchor, symbol))
        if not accepted:
            self.missing(
                anchor, symbol, "late_or_clock_unhealthy_postcommit_confirmation"
            )
        else:
            self.pending[(anchor, symbol)] = (value, path)

    def step(self):
        now = self.clock()
        if (
            directory_bytes(self.root) + LIMITS["reserve_bytes"] >= LIMITS["max_bytes"]
            or shutil.disk_usage(self.root).free < LIMITS["min_free_bytes"]
        ):
            return self.status("resource_stopped")
        if now < self.config["training_cutoff"]:
            return self.status(
                "warmup",
                seconds_until_training_cutoff=self.config["training_cutoff"] - now,
            )
        self.capture.update()
        self.pilot.update()
        if not self.healthy(self.moment(), self.clock_reference(), lifecycle=True):
            return self.status("awaiting_clock")
        if self.model is None:
            self.freeze()
        if not self.healthy(self.moment(), self.clock_reference(), lifecycle=True):
            return self.status("awaiting_clock")
        now = self.clock()
        last = min(
            math.floor(now / 300) * 300,
            math.floor((self.config["end_at"] - 360) / 300) * 300,
        )
        for anchor in range(self.config["training_cutoff"], last + 1, 300):
            for symbol in SYMBOLS:
                if not self.healthy(
                    self.moment(), self.clock_reference(), lifecycle=True
                ):
                    return self.status("awaiting_clock")
                if (anchor, symbol) in self.done:
                    continue
                if self.path("missing", anchor, symbol).exists():
                    self.done.add((anchor, symbol))
                    continue
                if self.path("forecasts", anchor, symbol).exists():
                    if not valid_emission(self.root, anchor, symbol):
                        self.missing(
                            anchor,
                            symbol,
                            "orphan_or_ineligible_forecast_never_reacknowledged",
                        )
                    else:
                        path = self.path("forecasts", anchor, symbol)
                        self.pending[(anchor, symbol)] = (load(path), path)
                        self.done.add((anchor, symbol))
                    continue
                current = self.clock()
                if current > anchor + 30:
                    self.missing(anchor, symbol, "missed_deadline_no_retroemission")
                    continue
                controls = original(self.pilot, anchor, symbol)
                if controls is not None and controls["available_at"] > current:
                    controls = None
                if (
                    controls is None
                    or self.model is None
                    or not self.healthy(self.moment(), self.clock_reference())
                ):
                    if current >= anchor + 30:
                        self.missing(
                            anchor,
                            symbol,
                            "original_missing_ineligible_model_unfrozen_or_clock_unhealthy",
                        )
                    continue
                depth = book(self.capture, anchor, symbol)
                if depth is not None and depth["available_at"] > self.clock():
                    depth = None
                self.emit(anchor, symbol, controls, depth)
        if self.healthy(self.moment(), self.clock_reference(), lifecycle=True):
            for (a, s), (p, path) in list(self.pending.items()):
                if not self.healthy(
                    self.moment(), self.clock_reference(), lifecycle=True
                ):
                    return self.status("awaiting_clock")
                target = self.path("outcomes", a, s)
                if target.exists():
                    del self.pending[(a, s)]
                    continue
                if self.clock() < a + 360:
                    continue
                observation = outcome(self.pilot, a, s)
                if (
                    observation is not None
                    and observation["received_at"] <= self.clock()
                ):
                    self.writer(
                        target,
                        dict(
                            anchor=a,
                            symbol=s,
                            **observation,
                            copied_at=self.clock(),
                            forecast_sha256=file_sha(path),
                        ),
                    )
                    del self.pending[(a, s)]
                    self.changed = True
        if not self.healthy(self.moment(), self.clock_reference(), lifecycle=True):
            return self.status("awaiting_clock")
        return self.status(
            "finished"
            if self.clock() >= self.config["end_at"] + 120
            else "running"
            if self.model is not None
            else "awaiting_model_clock"
        )

    def status(self, phase, **extra):
        if (
            self.last_status is not None
            and not self.changed
            and phase == self.last_status["phase"]
            and self.clock() - self.last_status["updated_at"] < 30
        ):
            return self.last_status
        if self.changed or self.cached_summary is None:
            self.cached_summary = summary(self.root, now=self.clock())
        self.cached_summary["matured_pending_outcomes"] = sum(
            row["anchor"] + 360 <= self.clock()
            for row in self.cached_summary["pending_targets"]
        )
        self.cached_summary["future_pending_outcomes"] = (
            self.cached_summary["pending_outcomes"]
            - self.cached_summary["matured_pending_outcomes"]
        )
        value = dict(
            phase=phase,
            updated_at=self.clock(),
            updated_utc=utc(self.clock()),
            pid=os.getpid(),
            training_cutoff=self.config["training_cutoff"],
            training_cutoff_utc=utc(self.config["training_cutoff"]),
            model_fitted=(self.root / "model.json").exists(),
            config_sha256=sha(self.config),
            **self.cached_summary,
            **extra,
        )
        durable_write(self.root / "status.json", value, exclusive=False)
        self.changed, self.last_status = False, value
        return value


def directory_bytes(root):
    return sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())


def valid_emission(root, anchor, symbol):
    stem = f"{anchor}-{symbol}.json"
    try:
        f, a, e = [
            Path(root) / name / stem for name in ("forecasts", "acks", "eligibility")
        ]
        forecast, ack, seal = load(f), load(a), load(e)
        return bool(
            seal["eligible"] is True
            and seal["clock_healthy"] is True
            and ack["primary_clock_healthy"] is True
            and seal["forecast_sha256"] == ack["forecast_sha256"] == file_sha(f)
            and seal["ack_sha256"] == file_sha(a)
            and anchor
            <= forecast["issued_at"]
            <= ack["forecast_committed_at"]
            <= seal["ack_committed_at"]
            <= anchor + 30
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def summary(root, *, now=None):
    root = Path(root)
    now = time.time() if now is None else now
    records, invalid = [], 0
    for path in sorted((root / "forecasts").glob("*.json")):
        try:
            p = load(path)
            if valid_emission(root, p["anchor"], p["symbol"]):
                records.append((p, path))
            else:
                invalid += 1
        except (ValueError, KeyError):
            invalid += 1
    result = dict(
        candidate_forecasts=len(records),
        ineligible_or_unconfirmed_forecasts=invalid,
        missing_emissions=len(list((root / "missing").glob("*.json"))),
        fallback_missing_book=sum(p["fallback_missing_book"] for p, _ in records),
        pending_outcomes=0,
        matured_pending_outcomes=0,
        future_pending_outcomes=0,
        pending_targets=[],
        scored_outcomes=0,
        metrics={},
    )
    for symbol in SYMBOLS:
        pairs = []
        for p, path in records:
            if p["symbol"] != symbol:
                continue
            observation_path = root / "outcomes" / path.name
            if not observation_path.exists():
                result["pending_outcomes"] += 1
                result["pending_targets"].append(
                    dict(anchor=p["anchor"], symbol=symbol)
                )
                result[
                    "matured_pending_outcomes"
                    if p["anchor"] + 360 <= now
                    else "future_pending_outcomes"
                ] += 1
                continue
            observation = load(observation_path)
            require(
                observation["anchor"] == p["anchor"]
                and observation["symbol"] == symbol
                and observation["forecast_sha256"] == file_sha(path)
                and observation["received_at"] >= p["anchor"] + 360
                and observation["label"] in (0, 1, 2),
                "copied outcome membership differs",
            )
            pairs.append((p, observation))
        result["scored_outcomes"] += len(pairs)
        models = {}
        for arm in ARMS:
            moving, brier, loss = [], [], []
            for p, observation in pairs:
                prob, y = np.asarray(p["models"][arm]), observation["label"]
                brier.append(float(np.sum((prob - (np.arange(3) == y)) ** 2)))
                loss.append(-math.log(prob[y]))
                if y != 1:
                    moving.append(
                        int((prob[2] / (prob[0] + prob[2]) >= 0.5) == (y == 2))
                    )
            models[arm] = dict(
                n=len(pairs),
                moving_n=len(moving),
                direction_correct=sum(moving),
                direction_accuracy=sum(moving) / len(moving) if moving else None,
                brier=math.fsum(brier) / len(brier) if brier else None,
                log_loss=math.fsum(loss) / len(loss) if loss else None,
            )
        result["metrics"][symbol] = models
    return result


def run(root, *, once=False):
    root = safe_root(root)
    with (root / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Runner(root)
        while True:
            status = runner.step()
            if once or status["phase"] in ("finished", "resource_stopped"):
                return 2 if status["phase"] == "resource_stopped" else 0
            now = time.time()
            cutoff = runner.config["training_cutoff"]
            if now < cutoff:
                delay = min(30, cutoff - now)
            else:
                delay = 1 if now % 300 <= 31 else min(30, 300 - now % 300)
            time.sleep(max(0, delay))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--capture-root", type=Path, required=True)
    init.add_argument("--pilot-root", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--root", type=Path, required=True)
    execute.add_argument("--once", action="store_true")
    report = sub.add_parser("status")
    report.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        print(
            json.dumps(
                initialize(
                    args.root,
                    capture_root=args.capture_root,
                    pilot_root=args.pilot_root,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "status":
        print(
            json.dumps(
                {
                    "status": load(args.root / "status.json"),
                    "summary": summary(args.root),
                },
                sort_keys=True,
            )
        )
        return 0
    return run(args.root, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
