"""Synthetic new journals and clocks; no network, original data or real fitting."""

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from scripts import run_depth_information as run

FIRST = 1793374500
CUTOFF = FIRST + 86400
END = FIRST + 4 * 86400


class Clock:
    def __init__(self, value=FIRST - 100):
        self.value = float(value)
        self.mono = 10000.0 + self.value - FIRST

    def time(self):
        return self.value

    def monotonic(self):
        return self.mono

    def advance(self, seconds):
        self.value += seconds
        self.mono += seconds

    def set(self, value):
        self.advance(value - self.value)


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


class Ledger:
    def __init__(self, directory, config, kind):
        self.kind, self.config, self.seq, self.tip = kind, config, 0, run.sha(config)
        directory.mkdir()
        save(directory / "config.json", config)
        save(directory / "config-sha256.json", {"sha256": self.tip})
        self.path = directory / (
            "events.sqlite" if kind == "pilot" else "depth-events.sqlite"
        )
        self.c = sqlite3.connect(self.path)
        self.c.execute("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT)")
        self.c.executemany(
            "INSERT INTO meta VALUES (?,?)",
            [("config", run.canonical(config)), ("config_sha256", self.tip)],
        )
        if kind == "pilot":
            self.c.execute(
                "CREATE TABLE events(seq INTEGER PRIMARY KEY, kind TEXT,received_at REAL,received_ns INTEGER,anchor INTEGER,symbol TEXT,payload TEXT,prev_hash TEXT,digest TEXT)"
            )
        else:
            self.c.execute(
                "CREATE TABLE events(seq INTEGER PRIMARY KEY,kind TEXT,recorded_at REAL,anchor INTEGER,symbol TEXT,payload TEXT,prev_hash TEXT,digest TEXT)"
            )
        self.c.commit()

    def add(self, kind, payload, received, *, anchor=None, symbol=None):
        received = float(received)
        self.seq += 1
        anchor = payload.get("anchor", anchor)
        symbol = payload.get("symbol", symbol)
        if self.kind == "pilot":
            ns = int(received * 1e9)
            envelope = dict(
                seq=self.seq,
                kind=kind,
                received_ns=ns,
                payload=payload,
                prev_hash=self.tip,
            )
            values = (
                self.seq,
                kind,
                received,
                ns,
                anchor,
                symbol,
                run.canonical(payload),
                self.tip,
                run.sha(envelope),
            )
        else:
            envelope = dict(
                seq=self.seq,
                kind=kind,
                recorded_at=received,
                anchor=anchor,
                symbol=symbol,
                payload=payload,
                prev_hash=self.tip,
            )
            values = (
                self.seq,
                kind,
                received,
                anchor,
                symbol,
                run.canonical(payload),
                self.tip,
                run.sha(envelope),
            )
        self.c.execute(
            f"INSERT INTO events VALUES ({','.join('?' for _ in values)})", values
        )
        self.c.commit()
        self.tip = run.sha(envelope)
        return self.seq, self.tip


@pytest.fixture
def sources(tmp_path):
    model = b"{}\n"
    p = dict(
        schema="klinehouse-shadow-pilot-v2",
        start_at=run.PILOT_START,
        end_at=END,
        symbols=list(run.SYMBOLS),
        require_commit_ack=True,
        orders=False,
        snapshot_sha256=hashlib.sha256(model).hexdigest(),
    )
    d = dict(
        schema="klinehouse-depth-research-v1",
        activation_at=FIRST - 100,
        end_at=END,
        symbols=list(run.SYMBOLS),
        capture_only=True,
        candidate_predictions=False,
        orders=False,
    )
    pilot = Ledger(tmp_path / "shadow-pilot-20261029-v2", p, "pilot")
    (pilot.path.parent / "model.json").write_bytes(model)
    capture = Ledger(tmp_path / "depth-research-synthetic", d, "depth")
    yield capture, pilot
    capture.c.close()
    pilot.c.close()


def original(ledger, anchor, symbol, *, eligible=True, ack_delay=0.2):
    models = dict(
        baseline=[0.45, 0.10, 0.45],
        price_control=[0.44, 0.10, 0.46],
        positions=[0.46, 0.10, 0.44],
    )
    p = dict(
        anchor=anchor,
        symbol=symbol,
        issued_at=anchor + 10,
        models=models,
        snapshot_sha256=ledger.config["snapshot_sha256"],
    )
    seq, digest = ledger.add("forecast", p, anchor + 10)
    ledger.add(
        "forecast_commit",
        dict(
            anchor=anchor,
            symbol=symbol,
            forecast_seq=seq,
            forecast_hash=digest,
            committed_at=anchor + 10.1,
            clock_healthy=True,
            eligible=eligible,
        ),
        anchor + 10 + ack_delay,
    )
    return models


def observed(ledger, anchor, symbol, *, received=None, label=2):
    ledger.add(
        "outcome",
        dict(
            anchor=anchor,
            symbol=symbol,
            return_start=anchor + 60,
            return_end=anchor + 360,
            entry_price="100",
            exit_price="101" if label == 2 else "99" if label == 0 else "100",
            label=label,
        ),
        anchor + 370 if received is None else received,
    )


def depth(ledger, anchor, symbol):
    body = dict(
        bids=[[str(100 - i / 100), "3"] for i in range(20)],
        asks=[[str(101 + i / 100), "1"] for i in range(20)],
    )
    p = dict(
        receipt=dict(requested_at=anchor - 10, received_at=anchor - 9, body=body),
        features=dict(qi20="0.5"),
        validation_errors=[],
        clock_healthy=True,
    )
    seq, digest = ledger.add(
        "book_receipt", p, anchor - 8, anchor=anchor, symbol=symbol
    )
    ledger.add(
        "commit_ack",
        dict(
            record_seq=seq,
            record_hash=digest,
            eligible=True,
            clock_healthy=True,
            committed_at=anchor - 7,
        ),
        anchor - 6,
        anchor=anchor,
        symbol=symbol,
    )


def probe(ledger, anchor):
    at = anchor - 9
    ledger.add(
        "clock_probe",
        dict(
            healthy=True,
            absolute_offset_bound_seconds=0.1,
            receipt=dict(received_at=at, received_monotonic=10000 + at - FIRST),
        ),
        at,
        anchor=anchor,
    )


def prepare(
    tmp_path, sources, clock, *, training=True, missing_book=None, invalid_original=None
):
    capture, pilot = sources
    if training:
        for anchor in (FIRST, FIRST + 300):
            for symbol in run.SYMBOLS:
                depth(capture, anchor, symbol)
                original(pilot, anchor, symbol)
                observed(pilot, anchor, symbol)
    probe(capture, CUTOFF)
    for symbol in run.SYMBOLS:
        if symbol != missing_book:
            depth(capture, CUTOFF, symbol)
        original(pilot, CUTOFF, symbol, eligible=symbol != invalid_original)
    root = tmp_path / "depth-information-run"
    run.initialize(
        root,
        capture_root=capture.path.parent,
        pilot_root=pilot.path.parent,
        clock=clock.time,
    )
    return root


def runner(root, clock, **kwargs):
    return run.Runner(root, clock=clock.time, monotonic=clock.monotonic, **kwargs)


def test_warmup_cannot_fit_or_emit_and_fixed_cutoff_and_runtime_are_registered(
    tmp_path, sources, monkeypatch
):
    clock = Clock()
    root = prepare(tmp_path, sources, clock)
    r = runner(root, clock)
    monkeypatch.setattr(
        run.core, "fit_first_day", lambda **kwargs: pytest.fail("fit during warmup")
    )
    monkeypatch.setattr(
        r.capture, "update", lambda: pytest.fail("read market rows during warmup")
    )
    state = r.step()
    assert state["phase"] == "warmup" and state["pid"] > 0
    assert state["training_cutoff"] == CUTOFF
    assert not (root / "model.json").exists()
    assert not list((root / "forecasts").iterdir())
    assert run.file_sha(
        root / "runtime/scripts/run_depth_information.py"
    ) == run.file_sha(Path(run.__file__))
    result = subprocess.run(
        [
            sys.executable,
            str(root / "runtime/scripts/run_depth_information.py"),
            "status",
            "--root",
            str(root),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["status"]["phase"] == "warmup"


def test_first_day_membership_excludes_late_receipts_and_incomplete_three_asset_anchors(
    tmp_path, sources
):
    clock = Clock()
    root = prepare(tmp_path, sources, clock)
    capture, pilot = sources
    # This otherwise complete anchor has one outcome received after fixed cutoff.
    for symbol in run.SYMBOLS:
        depth(capture, FIRST + 600, symbol)
        original(pilot, FIRST + 600, symbol)
        observed(
            pilot,
            FIRST + 600,
            symbol,
            received=CUTOFF + (1 if symbol == "SOLUSDT" else -1),
        )
    # This anchor ends beyond cutoff and must not enter training either.
    for symbol in run.SYMBOLS:
        depth(capture, CUTOFF - 300, symbol)
        original(pilot, CUTOFF - 300, symbol)
        observed(pilot, CUTOFF - 300, symbol)
    r = runner(root, clock)
    r.capture.update()
    r.pilot.update()
    data, membership = r.training_data()
    assert data["anchors"].tolist() == [FIRST, FIRST + 300]
    assert len(membership) == 2
    with pytest.raises(ValueError, match="cutoff has not arrived"):
        r.freeze()


def test_freeze_once_and_same_case_controls_missing_book_falls_back_exactly(
    tmp_path, sources, monkeypatch
):
    clock = Clock()
    root = prepare(tmp_path, sources, clock, missing_book="SOLUSDT")
    clock.set(CUTOFF + 11)
    r = runner(root, clock)
    status = r.step()
    assert status["candidate_forecasts"] == 3 and status["fallback_missing_book"] == 1
    model_hash = run.file_sha(root / "model.json")
    state = run.load(root / "model.json")
    assert (
        state["state"]["coefficient"] == 0
        and state["state"]["status"] == "insufficient_training"
    )
    assert state["state"]["training_anchors"] == 2
    forecast = run.load(r.path("forecasts", CUTOFF, "SOLUSDT"))
    assert forecast["models"]["depth"] == forecast["models"]["baseline"]
    assert set(forecast["models"]) == set(run.ARMS)
    assert run.valid_emission(root, CUTOFF, "SOLUSDT")
    monkeypatch.setattr(
        run.core, "fit_first_day", lambda **kwargs: pytest.fail("second fit")
    )
    restarted = runner(root, clock)
    restarted.step()
    assert run.file_sha(root / "model.json") == model_hash


@pytest.mark.parametrize(
    "folder,wall,mono",
    [("forecasts", 25, 25), ("acks", 40, 40), ("forecasts", -50, 1), ("acks", -50, 1)],
)
def test_actual_primary_ack_fsync_deadline_and_clock_jump_are_ineligible(
    tmp_path, sources, folder, wall, mono
):
    clock = Clock()
    root = prepare(tmp_path, sources, clock)
    changed = False

    def slow_writer(path, value, **kwargs):
        nonlocal changed
        run.durable_write(path, value, **kwargs)
        if path.parent.name == folder and not changed:
            clock.value += wall
            clock.mono += mono
            changed = True

    clock.set(CUTOFF + 11)
    status = runner(root, clock, writer=slow_writer).step()
    assert status["candidate_forecasts"] == 0
    seals = list((root / "eligibility").glob("*.json"))
    assert seals and all(run.load(path)["eligible"] is False for path in seals)
    assert list((root / "missing").glob("*.json"))


def test_orphan_forecast_is_preserved_never_reacknowledged_or_overwritten(
    tmp_path, sources
):
    clock = Clock()
    root = prepare(tmp_path, sources, clock)
    clock.set(CUTOFF + 11)
    r = runner(root, clock)
    r.step()
    path = r.path("forecasts", CUTOFF, "BTCUSDT")
    before = path.read_bytes()
    r.path(
        "eligibility", CUTOFF, "BTCUSDT"
    ).unlink()  # Synthetic crash before final seal.
    restarted = runner(root, clock)
    state = restarted.step()
    assert path.read_bytes() == before
    assert not restarted.path("eligibility", CUTOFF, "BTCUSDT").exists()
    assert state["candidate_forecasts"] == 2 and state["missing_emissions"] == 1
    with pytest.raises(ValueError, match="already exists"):
        restarted.emit(CUTOFF, "BTCUSDT", {}, None)


def test_original_missing_or_late_ack_is_explicit_missing_without_candidate(
    tmp_path, sources
):
    clock = Clock()
    root = prepare(tmp_path, sources, clock, invalid_original="SOLUSDT")
    clock.set(CUTOFF + 11)
    r = runner(root, clock)
    assert r.step()["candidate_forecasts"] == 2
    clock.advance(25)
    status = r.step()
    assert status["missing_emissions"] == 1
    assert not r.path("forecasts", CUTOFF, "SOLUSDT").exists()
    _, pilot = sources
    a = CUTOFF + 300
    original(pilot, a, "SOLUSDT", ack_delay=25)
    r.pilot.update()
    assert run.original(r.pilot, a, "SOLUSDT") is None


def test_no_future_source_availability_at_issue(tmp_path, sources):
    clock = Clock()
    root = prepare(tmp_path, sources, clock)
    clock.set(CUTOFF + 11)
    r = runner(root, clock)
    r.capture.update()
    r.pilot.update()
    assert r.freeze()
    control = run.original(r.pilot, CUTOFF, "BTCUSDT")
    control["available_at"] = clock.time() + 1
    with pytest.raises(ValueError, match="not available before"):
        r.emit(CUTOFF, "BTCUSDT", control, None)
    assert not list((root / "forecasts").glob("*.json"))


def test_outcomes_only_after_maturity_actual_receipt_same_case_scores_and_pending(
    tmp_path, sources
):
    clock = Clock()
    root = prepare(tmp_path, sources, clock)
    clock.set(CUTOFF + 11)
    r = runner(root, clock)
    assert r.step()["future_pending_outcomes"] == 3
    _, pilot = sources
    for symbol in run.SYMBOLS:
        observed(
            pilot,
            CUTOFF,
            symbol,
            received=CUTOFF + 370,
            label=1 if symbol == "SOLUSDT" else 2,
        )
    clock.set(CUTOFF + 365)
    status = r.step()
    assert status["scored_outcomes"] == 0 and status["matured_pending_outcomes"] >= 3
    clock.advance(6)
    status = r.step()
    assert status["scored_outcomes"] == 3
    for symbol in run.SYMBOLS:
        for values in status["metrics"][symbol].values():
            assert values["n"] == 1 and values["moving_n"] == (
                0 if symbol == "SOLUSDT" else 1
            )
            assert values["brier"] is not None and values["log_loss"] is not None


def test_protected_source_output_aliases_and_changed_frozen_hashes_are_rejected(
    tmp_path, sources
):
    clock = Clock()
    capture, pilot = sources
    with pytest.raises(ValueError, match="source roots"):
        run.initialize(
            tmp_path / "new",
            capture_root=tmp_path / "archive",
            pilot_root=pilot.path.parent,
            clock=clock.time,
        )
    with pytest.raises(ValueError, match="output roots"):
        run.initialize(
            pilot.path.parent / "new",
            capture_root=capture.path.parent,
            pilot_root=pilot.path.parent,
            clock=clock.time,
        )
    root = prepare(tmp_path, sources, clock)
    (root / "protocol-plan.json").write_text("{}")
    with pytest.raises(ValueError, match="configuration changed"):
        runner(root, clock)


def test_incremental_read_only_reader_detects_replaced_prefix_and_never_changes_source(
    tmp_path, sources
):
    capture, _ = sources
    probe(capture, FIRST)
    reader = run.Source(capture.path.parent, "depth", first_anchor=FIRST)
    before = run.file_sha(capture.path)
    reader.update()
    assert reader.seq == 1 and run.file_sha(capture.path) == before
    capture.c.execute("UPDATE events SET digest='changed' WHERE seq=1")
    capture.c.commit()
    with pytest.raises(ValueError, match="retained source prefix changed"):
        reader.update()


def test_atomic_publication_preserves_existing_bytes_and_private_partial(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.json"
    original_link = run.os.link

    def crash_before_publish(source, target):
        raise RuntimeError("synthetic crash before atomic publication")

    monkeypatch.setattr(run.os, "link", crash_before_publish)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        run.durable_write(path, {"complete": True})
    assert not path.exists()
    evidence = list(tmp_path.glob(".model.json.pending-*"))
    assert len(evidence) == 1 and run.load(evidence[0]) == {"complete": True}
    monkeypatch.setattr(run.os, "link", original_link)
    run.durable_write(path, {"complete": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        run.durable_write(path, {"complete": False})
    assert path.read_bytes() == before


def test_model_crash_reuses_fixed_training_snapshot_without_membership_expansion(
    tmp_path, sources, monkeypatch
):
    clock = Clock()
    root = prepare(tmp_path, sources, clock)
    clock.set(CUTOFF + 11)
    original_link = run.os.link

    def crash_model_publish(source, target):
        if Path(target).name == "model.json":
            raise RuntimeError("synthetic model publication crash")
        return original_link(source, target)

    monkeypatch.setattr(run.os, "link", crash_model_publish)
    with pytest.raises(RuntimeError, match="model publication crash"):
        runner(root, clock).step()
    assert not (root / "model.json").exists()
    snapshot_hash = run.file_sha(root / "training-data.json")
    assert list(root.glob(".model.json.pending-*"))
    capture, pilot = sources
    for symbol in run.SYMBOLS:
        depth(capture, FIRST + 600, symbol)
        original(pilot, FIRST + 600, symbol)
        observed(pilot, FIRST + 600, symbol)
    monkeypatch.setattr(run.os, "link", original_link)
    state = runner(root, clock).step()
    assert state["candidate_forecasts"] == 3
    model = run.load(root / "model.json")
    assert model["state"]["training_anchors"] == 2
    assert model["training_data_sha256"] == snapshot_hash
    assert run.file_sha(root / "training-data.json") == snapshot_hash


def test_outcome_crash_leaves_no_partial_final_and_restart_recovers_complete_copy(
    tmp_path, sources, monkeypatch
):
    clock = Clock()
    root = prepare(tmp_path, sources, clock)
    clock.set(CUTOFF + 11)
    r = runner(root, clock)
    r.step()
    _, pilot = sources
    for symbol in run.SYMBOLS:
        observed(pilot, CUTOFF, symbol)
    clock.set(CUTOFF + 371)
    original_link = run.os.link

    def crash_outcome_publish(source, target):
        if Path(target).parent.name == "outcomes":
            raise RuntimeError("synthetic outcome publication crash")
        return original_link(source, target)

    monkeypatch.setattr(run.os, "link", crash_outcome_publish)
    with pytest.raises(RuntimeError, match="outcome publication crash"):
        r.step()
    assert not list((root / "outcomes").glob("*.json"))
    assert list((root / "outcomes").glob(".*.pending-*"))
    monkeypatch.setattr(run.os, "link", original_link)
    state = runner(root, clock).step()
    assert state["scored_outcomes"] == 3 and state["pending_outcomes"] == 0


@pytest.mark.parametrize("jump", [3 * 3600, END - CUTOFF + 3600])
def test_forward_clock_jump_does_not_consume_anchors_or_finish_and_recovers(
    tmp_path, sources, jump
):
    clock = Clock()
    root = prepare(tmp_path, sources, clock)
    clock.set(CUTOFF + 11)
    r = runner(root, clock)
    r.step()
    done, missing = set(r.done), list((root / "missing").glob("*.json"))
    clock.value += jump  # UTC jumps; monotonic time remains continuous.
    assert r.step()["phase"] == "awaiting_clock"
    assert r.done == done and list((root / "missing").glob("*.json")) == missing
    clock.value -= jump
    assert r.step()["phase"] == "running"
    assert r.done == done


def test_warmup_forward_clock_jump_cannot_trigger_fit_or_missing(tmp_path, sources):
    clock = Clock()
    root = prepare(tmp_path, sources, clock)
    r = runner(root, clock)
    assert r.step()["phase"] == "warmup"
    clock.value = CUTOFF + 11
    assert r.step()["phase"] == "awaiting_clock"
    assert not (root / "model.json").exists()
    assert not list((root / "missing").glob("*.json"))


def test_terminal_outcomes_and_stop_allow_older_continuous_probe_without_emission(
    tmp_path, sources
):
    clock = Clock()
    root = prepare(tmp_path, sources, clock)
    clock.set(CUTOFF + 11)
    r = runner(root, clock)
    r.step()
    last_anchor = (END - 360) // 300 * 300
    # Represent all intermediate cases as already accounted for; isolate final
    # collection without generating thousands of unrelated fixture files.
    r.done.update(
        (anchor, symbol)
        for anchor in range(CUTOFF, last_anchor, 300)
        for symbol in run.SYMBOLS
    )
    capture, pilot = sources
    probe(capture, last_anchor)
    for symbol in run.SYMBOLS:
        depth(capture, last_anchor, symbol)
        original(pilot, last_anchor, symbol)
        observed(pilot, last_anchor, symbol)
    clock.set(last_anchor + 11)
    assert r.step()["candidate_forecasts"] == 6
    clock.set(END + 120)
    reference = r.clock_reference()
    assert not r.healthy(r.moment(), reference)
    assert r.healthy(r.moment(), reference, lifecycle=True)
    status = r.step()
    assert status["phase"] == "finished" and status["scored_outcomes"] == 3
