"""Hand-built prospective evidence, independent of the depth runner/model core."""

import copy
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import pytest

from scripts.verify_depth_information import verify


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
ARMS = ("baseline", "price_control", "positions")
START = 1793272200
FIRST = 1793374500
CUTOFF = FIRST + 86400
MONO_OFFSET = 1793000000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(event):
    return {k: event[k] for k in ("seq", "digest", "received_at")}


class Journal:
    def __init__(self, root, config, kind):
        self.root, self.config, self.kind = root, config, kind
        self.events = []
        self.tip = sha(config)

    def add(self, kind, stamp, payload, anchor=None, symbol=None):
        stamp = float(stamp)
        anchor = payload.get("anchor", anchor)
        symbol = payload.get("symbol", symbol)
        seq = len(self.events) + 1
        if self.kind == "pilot":
            received_ns = round(stamp * 1e9)
            stamp = received_ns / 1e9
            envelope = dict(
                seq=seq,
                kind=kind,
                received_ns=received_ns,
                payload=payload,
                prev_hash=self.tip,
            )
        else:
            envelope = dict(
                seq=seq,
                kind=kind,
                recorded_at=stamp,
                anchor=anchor,
                symbol=symbol,
                payload=payload,
                prev_hash=self.tip,
            )
        event = dict(
            seq=seq,
            kind=kind,
            received_at=stamp,
            anchor=anchor,
            symbol=symbol,
            payload=copy.deepcopy(payload),
            prev_hash=self.tip,
            digest=sha(envelope),
        )
        if self.kind == "pilot":
            event["received_ns"] = received_ns
        self.events.append(event)
        self.tip = event["digest"]
        return event

    def proof(self):
        return dict(
            root=str(self.root), event_count=len(self.events), chain_tip=self.tip
        )

    def publish(self):
        name = "events.sqlite" if self.kind == "pilot" else "depth-events.sqlite"
        self.root.mkdir(exist_ok=True)
        write(self.root / "config.json", self.config)
        if self.kind == "depth":
            write(self.root / "config-sha256.json", {"sha256": sha(self.config)})
        connection = sqlite3.connect(self.root / name)
        connection.execute(
            "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO meta VALUES(?,?)",
            [("config", canonical(self.config)), ("config_sha256", sha(self.config))],
        )
        if self.kind == "pilot":
            connection.execute(
                "CREATE TABLE events(seq INTEGER PRIMARY KEY,kind TEXT,received_at REAL,received_ns INTEGER,anchor INTEGER,symbol TEXT,payload TEXT,prev_hash TEXT,digest TEXT)"
            )
            for e in self.events:
                connection.execute(
                    "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        e["seq"],
                        e["kind"],
                        e["received_at"],
                        e["received_ns"],
                        e["anchor"],
                        e["symbol"],
                        canonical(e["payload"]),
                        e["prev_hash"],
                        e["digest"],
                    ),
                )
        else:
            connection.execute(
                "CREATE TABLE events(seq INTEGER PRIMARY KEY,kind TEXT,recorded_at REAL,anchor INTEGER,symbol TEXT,payload TEXT,prev_hash TEXT,digest TEXT)"
            )
            for e in self.events:
                connection.execute(
                    "INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",
                    (
                        e["seq"],
                        e["kind"],
                        e["received_at"],
                        e["anchor"],
                        e["symbol"],
                        canonical(e["payload"]),
                        e["prev_hash"],
                        e["digest"],
                    ),
                )
        connection.commit()
        connection.close()


def models():
    # For x=.5 and balanced labels this has the known ridge optimum beta=.5:
    # sigmoid(logit(q)+.5*.5)=.25, so .5*(.25-.5)+.25*.5=0.
    odds = math.exp(math.log(1 / 3) - 0.25)
    q = odds / (1 + odds)
    p = [0.9 * (1 - q), 0.1, 0.9 * q]
    return dict(
        baseline=p,
        price_control=[p[0] - 0.01, 0.1, p[2] + 0.01],
        positions=[p[0] + 0.01, 0.1, p[2] - 0.01],
    )


def original_forecast(journal, anchor, symbol):
    payload = dict(
        anchor=anchor,
        symbol=symbol,
        issued_at=anchor + 10,
        snapshot_sha256=journal.config["snapshot_sha256"],
        models=models(),
        input_ids=[],
    )
    event = journal.add("forecast", anchor + 10, payload, anchor, symbol)
    ack = journal.add(
        "forecast_commit",
        anchor + 11,
        dict(
            anchor=anchor,
            symbol=symbol,
            forecast_seq=event["seq"],
            forecast_hash=event["digest"],
            committed_at=anchor + 10.5,
            eligible=True,
            clock_healthy=True,
        ),
        anchor,
        symbol,
    )
    return dict(
        models=payload["models"],
        forecast=reference(event),
        acknowledgement=reference(ack),
        available_at=ack["received_at"],
    )


def candle(stamp, close):
    return [
        stamp * 1000,
        close,
        close,
        close,
        close,
        "10",
        stamp * 1000 + 59999,
        "1000",
        4,
        "5",
        "500",
        "0",
    ]


def original_outcome(journal, anchor, symbol, label):
    entry = "100.00"
    exit_price = {0: "99.00", 1: "100.00", 2: "101.00"}[label]
    body = [candle(anchor, entry), candle(anchor + 300, exit_price)]
    receipt = journal.add(
        "receipt",
        anchor + 360.5,
        dict(
            source="spot_klines",
            symbol=symbol,
            body=body,
            body_sha256=sha(body),
            content_times=[anchor, anchor + 300],
            requested_at=anchor + 360,
            request_params=dict(
                symbol=symbol,
                startTime=START * 1000,
                endTime=math.floor(journal.config["end_at"] * 1000) - 1,
            ),
        ),
    )
    payload = dict(
        anchor=anchor,
        symbol=symbol,
        return_start=anchor + 60,
        return_end=anchor + 360,
        entry_price=entry,
        exit_price=exit_price,
        label=label,
        entry_receipt_id=receipt["seq"],
        exit_receipt_id=receipt["seq"],
        input_ids=[receipt["seq"]],
        **{"return": math.log(float(exit_price) / 100)},
    )
    event = journal.add("outcome", anchor + 361, payload, anchor, symbol)
    return dict(
        label=label,
        entry_price=entry,
        exit_price=exit_price,
        source=reference(event),
        received_at=event["received_at"],
    )


def depth_clock(journal, anchor):
    requested, received = anchor - 9.5, anchor - 9
    body = {"serverTime": (anchor - 9) * 1000}
    raw = canonical(body)
    receipt = dict(
        source="time",
        url=journal.config["urls"]["time"],
        symbol=None,
        http_status=200,
        body=body,
        raw_body=raw,
        body_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        body_bytes=len(raw),
        requested_at=requested,
        received_at=received,
        requested_monotonic=requested - MONO_OFFSET,
        received_monotonic=received - MONO_OFFSET,
        params={},
    )
    journal.add(
        "clock_probe",
        anchor - 8.9,
        dict(healthy=True, absolute_offset_bound_seconds=0.5, receipt=receipt),
        anchor,
    )
    return dict(
        utc=received,
        monotonic=received - MONO_OFFSET,
        absolute_offset_bound_seconds=0.5,
    )


def depth_observation(journal, anchor, symbol, clock):
    with localcontext() as context:
        context.prec = 50
        body = dict(
            lastUpdateId=anchor,
            bids=[[str(Decimal("100") - Decimal(i) / 100), "3"] for i in range(20)],
            asks=[[str(Decimal("100.01") + Decimal(i) / 100), "1"] for i in range(20)],
        )
    raw = canonical(body)
    receipt = dict(
        source="depth",
        symbol=symbol,
        url=journal.config["urls"]["depth"],
        http_status=200,
        body=body,
        raw_body=raw,
        body_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        body_bytes=len(raw),
        requested_at=anchor - 8.5,
        received_at=anchor - 8,
        requested_monotonic=anchor - 8.5 - MONO_OFFSET,
        received_monotonic=anchor - 8 - MONO_OFFSET,
        params=dict(limit=20, symbol=symbol),
    )
    payload = dict(
        receipt=receipt,
        features=dict(qi20="0.5"),
        clock_healthy=True,
        source_event_timestamp=None,
        validation_errors=[],
    )
    event = journal.add("book_receipt", anchor - 7, payload, anchor, symbol)
    ack = journal.add(
        "commit_ack",
        anchor - 6.8,
        dict(
            record_seq=event["seq"],
            record_hash=event["digest"],
            committed_at=anchor - 6.9,
            committed_monotonic=anchor - 6.9 - MONO_OFFSET,
            record_started_monotonic=anchor - 7 - MONO_OFFSET,
            eligible=True,
            clock_healthy=True,
            clock_reference_healthy=True,
            commit_elapsed_clock_healthy=True,
            deadline=anchor + 30,
            clock_reference=[clock["utc"], clock["monotonic"]],
        ),
        anchor,
        symbol,
    )
    return dict(
        qi20=0.5,
        receipt=reference(event),
        acknowledgement=reference(ack),
        available_at=ack["received_at"],
    )


def fixture(tmp_path, *, fitted=False, missing_book=False, omit_seal=False, late=False):
    root = tmp_path / "depth-information-fixture"
    capture = tmp_path / "depth-research-fixture"
    pilot = tmp_path / "shadow-pilot-20261029-v2"
    root.mkdir()
    capture.mkdir()
    pilot.mkdir()
    write(
        pilot / "model.json", {"frozen": "original model identity only; no inference"}
    )
    pc = dict(
        schema="klinehouse-shadow-pilot-v2",
        start_at=START,
        end_at=CUTOFF + 3600,
        symbols=list(SYMBOLS),
        orders=False,
        require_commit_ack=True,
        snapshot_sha256=file_sha(pilot / "model.json"),
    )
    dc = dict(
        schema="klinehouse-depth-research-v1",
        activation_at=FIRST - 20,
        end_at=CUTOFF + 3600,
        symbols=list(SYMBOLS),
        capture_only=True,
        candidate_predictions=False,
        orders=False,
        urls=dict(
            depth="https://api.binance.com/api/v3/depth",
            time="https://api.binance.com/api/v3/time",
        ),
    )
    pj, dj = Journal(pilot, pc, "pilot"), Journal(capture, dc, "depth")
    # Several events deliberately share (anchor=None,symbol=None).
    for journal, stamp in ((pj, START + 1), (dj, FIRST - 19)):
        journal.add("process_started", stamp, {"at": stamp})
        journal.add("clock_check", stamp + 0.1, {"healthy": True})
        journal.add("clock_check", stamp + 0.2, {"healthy": True})
    for name in (
        "forecasts",
        "acks",
        "eligibility",
        "missing",
        "outcomes",
        "model-receipts",
    ):
        (root / name).mkdir()
    script, core = (
        root / "runtime/scripts/run_depth_information.py",
        root / "runtime/research/depth_information.py",
    )
    script.parent.mkdir(parents=True)
    core.parent.mkdir(parents=True)
    script.write_text("frozen runner identity\n")
    core.write_text("frozen scalar core identity\n")
    plan = dict(
        schema="depth-information-research-plan-v1",
        registered_at=FIRST - 30,
        core_code={"research/depth_information.py": file_sha(core)},
    )
    write(root / "protocol-plan.json", plan)
    identity = dict(
        capture=dict(root=str(capture), config=dc, config_sha256=sha(dc)),
        pilot=dict(root=str(pilot), config=pc, config_sha256=sha(pc)),
        first_anchor=FIRST,
        training_cutoff=CUTOFF,
    )
    write(root / "source-identities.json", identity)
    config = dict(
        schema="klinehouse-depth-information-run-v1",
        created_at=FIRST - 10,
        capture_root=str(capture),
        pilot_root=str(pilot),
        capture_config_sha256=sha(dc),
        pilot_config_sha256=sha(pc),
        original_snapshot_sha256=pc["snapshot_sha256"],
        first_anchor=FIRST,
        training_cutoff=CUTOFF,
        end_at=dc["end_at"],
        symbols=list(SYMBOLS),
        code=dict(script=file_sha(script), core=file_sha(core)),
        plan_sha256=file_sha(root / "protocol-plan.json"),
        source_identities_sha256=file_sha(root / "source-identities.json"),
        orders=False,
        network=False,
        original_pilot_read_only=True,
        limits=dict(
            deadline=30,
            clock_age=600,
            terminal_clock_age=1800,
            clock_tolerance=0.25,
            min_free_bytes=40 * 1024**3,
            max_bytes=128 * 1024**2,
            reserve_bytes=4 * 1024**2,
        ),
    )
    write(root / "config.json", config)
    write(root / "config-sha256.json", {"sha256": sha(config)})
    if not fitted:
        dj.publish()
        pj.publish()
        return dict(root=root, now=FIRST + 1, capture=capture, pilot=pilot)
    data = dict(anchors=[], baseline=[], imbalance=[], labels=[], received=[])
    membership = []
    for i in range(120):
        anchor = FIRST + i * 300
        clock = depth_clock(dj, anchor)
        assets = {}
        for symbol in SYMBOLS:
            control = original_forecast(pj, anchor, symbol)
            book = depth_observation(dj, anchor, symbol, clock)
            outcome = original_outcome(pj, anchor, symbol, 2 if i % 2 else 0)
            assets[symbol] = dict(original=control, book=book, outcome=outcome)
        membership.append(dict(anchor=anchor, assets=assets))
        data["anchors"].append(anchor)
        p = models()["baseline"]
        data["baseline"].append([p[2] / (p[0] + p[2])] * 3)
        data["imbalance"].append([0.5] * 3)
        data["labels"].append([2 if i % 2 else 0] * 3)
        data["received"].append([assets[s]["outcome"]["received_at"] for s in SYMBOLS])
    clock = depth_clock(dj, CUTOFF)
    books = {
        s: None
        if missing_book and s == "SOLUSDT"
        else depth_observation(dj, CUTOFF, s, clock)
        for s in SYMBOLS
    }
    frozen_prefixes = dict(depth=dj.proof(), pilot=pj.proof())
    training = dict(
        data=data,
        membership=membership,
        source_prefixes=frozen_prefixes,
        fixed_cutoff=CUTOFF,
        config_sha256=sha(config),
    )
    write(root / "training-data.json", training)
    model = dict(
        state=dict(
            schema="klinehouse-depth-information-one-coefficient-v1",
            cutoff=CUTOFF,
            first_anchor=FIRST,
            ridge=0.25,
            coefficient=0.5,
            training_anchors=120,
            moving_labels_per_asset=[120] * 3,
            status="frozen_after_first_day",
            gradient=0,
        ),
        fitted_at=CUTOFF + 1,
        fixed_cutoff=CUTOFF,
        training_membership=membership,
        training_membership_sha256=sha(membership),
        source_prefixes=frozen_prefixes,
        training_data_sha256=file_sha(root / "training-data.json"),
        config_sha256=sha(config),
    )
    write(root / "model.json", model)
    receipt = dict(
        model_sha256=file_sha(root / "model.json"),
        observed_durable_at=CUTOFF + 2,
        observed_monotonic=CUTOFF + 2 - MONO_OFFSET,
        clock_reference=clock,
        recovery_of_existing_bytes=True,
    )
    receipt_path = root / "model-receipts/first.json"
    write(receipt_path, receipt)
    receipt_ref = dict(
        path="model-receipts/first.json",
        sha256=file_sha(receipt_path),
        model_sha256=receipt["model_sha256"],
        durable_receipt_observed_at=CUTOFF + 2.1,
    )
    controls = {s: original_forecast(pj, CUTOFF, s) for s in SYMBOLS}
    emission_prefixes = dict(depth=dj.proof(), pilot=pj.proof())
    for i, symbol in enumerate(SYMBOLS):
        baseline = controls[symbol]["models"]["baseline"]
        book = books[symbol]
        candidate = baseline[:] if book is None else [0.9 * 0.75, 0.1, 0.9 * 0.25]
        forecast = dict(
            schema="depth-information-forecast-v1",
            anchor=CUTOFF,
            symbol=symbol,
            issued_at=CUTOFF + 12,
            issued_monotonic=CUTOFF + 12 - MONO_OFFSET,
            target_start=CUTOFF + 60,
            target_end=CUTOFF + 360,
            models={**controls[symbol]["models"], "depth": candidate},
            fallback_missing_book=book is None,
            fallback_zero_coefficient=False,
            qi20=None if book is None else 0.5,
            original=controls[symbol],
            depth_book=book,
            source_prefixes=emission_prefixes,
            model_receipt=receipt_ref,
            config_sha256=sha(config),
            clock_reference=clock,
        )
        stem = f"{CUTOFF}-{symbol}.json"
        write(root / "forecasts" / stem, forecast)
        acknowledgement = dict(
            forecast_sha256=file_sha(root / "forecasts" / stem),
            forecast_committed_at=CUTOFF + 13,
            forecast_committed_monotonic=CUTOFF + 13 - MONO_OFFSET,
            primary_clock_healthy=True,
            clock_reference=clock,
            deadline=CUTOFF + 30,
            recorded_at=CUTOFF + 13.1,
        )
        write(root / "acks" / stem, acknowledgement)
        seal = dict(
            forecast_sha256=acknowledgement["forecast_sha256"],
            ack_sha256=file_sha(root / "acks" / stem),
            ack_committed_at=CUTOFF + (31 if late else 14),
            ack_committed_monotonic=CUTOFF + (31 if late else 14) - MONO_OFFSET,
            eligible=not late,
            clock_healthy=True,
            recorded_at=CUTOFF + (31.1 if late else 14.1),
        )
        if not omit_seal:
            write(root / "eligibility" / stem, seal)
        value = original_outcome(pj, CUTOFF, symbol, (2, 0, 1)[i])
        if not omit_seal and not late:
            write(
                root / "outcomes" / stem,
                dict(
                    anchor=CUTOFF,
                    symbol=symbol,
                    **value,
                    copied_at=CUTOFF + 362,
                    forecast_sha256=acknowledgement["forecast_sha256"],
                ),
            )
        else:
            write(
                root / "missing" / stem,
                dict(
                    anchor=CUTOFF,
                    symbol=symbol,
                    reason="late_or_unconfirmed",
                    recorded_at=CUTOFF + 31.2,
                ),
            )
        write(
            root / "missing" / f"{CUTOFF + 300}-{symbol}.json",
            dict(
                anchor=CUTOFF + 300,
                symbol=symbol,
                reason="missed_deadline_no_retroemission",
                recorded_at=CUTOFF + 331,
            ),
        )
    dj.publish()
    pj.publish()
    return dict(root=root, now=CUTOFF + 370, capture=capture, pilot=pilot)


def change_forecast(root, symbol, mutate):
    stem = f"{CUTOFF}-{symbol}.json"
    path = root / "forecasts" / stem
    forecast = json.loads(path.read_text())
    mutate(forecast)
    write(path, forecast)
    ack_path = root / "acks" / stem
    ack = json.loads(ack_path.read_text())
    ack["forecast_sha256"] = file_sha(path)
    write(ack_path, ack)
    seal_path = root / "eligibility" / stem
    seal = json.loads(seal_path.read_text())
    seal.update(forecast_sha256=file_sha(path), ack_sha256=file_sha(ack_path))
    write(seal_path, seal)
    out_path = root / "outcomes" / stem
    if out_path.exists():
        out = json.loads(out_path.read_text())
        out["forecast_sha256"] = file_sha(path)
        write(out_path, out)


def test_warmup_verifies_sources_without_fitting_or_emitting(tmp_path):
    f = fixture(tmp_path)
    result = verify(f["root"], now=f["now"])
    assert result["verified"] and result["phase"] == "warmup"
    assert not result["model_fitted"] and result["candidate_forecasts"] == 0
    assert result["scored_cases"] == [] and result["end_at"] == CUTOFF + 3600
    assert (
        result["scored_outcomes"] == 0
        and result["metrics"]["BTCUSDT"]["depth"]["n"] == 0
    )


def test_model_or_emission_during_warmup_is_rejected(tmp_path):
    f = fixture(tmp_path)
    write(f["root"] / "model.json", {"not": "allowed yet"})
    with pytest.raises(ValueError, match="before training cutoff"):
        verify(f["root"], now=f["now"])


def test_full_known_optimum_emission_and_exact_outcome_cycle(tmp_path):
    f = fixture(tmp_path, fitted=True)
    result = verify(f["root"], now=f["now"])
    assert result["verified"] and result["candidate_forecasts"] == 3
    assert result["coefficient_certificate"]["optimal_coefficient"] == pytest.approx(
        0.5, abs=1e-12
    )
    assert result["coefficient_certificate"]["moving_labels_per_asset"] == [120] * 3
    assert result["scored_outcomes"] == 3 and result["pending_outcomes"] == 0
    assert result["missing_emissions"] == 3 and not result["unaccounted_anchor_symbols"]
    for symbol in SYMBOLS:
        metrics = result["metrics"][symbol]
        assert {row["n"] for row in metrics.values()} == {1}
    assert result["metrics"]["BTCUSDT"]["depth"]["brier"] == pytest.approx(
        0.675**2 + 0.1**2 + 0.775**2
    )
    assert result["metrics"]["ETHUSDT"]["depth"]["direction_correct"] == 1
    assert result["metrics"]["SOLUSDT"]["depth"]["moving_n"] == 0
    assert result["metrics"]["SOLUSDT"]["depth"]["log_loss"] == pytest.approx(
        -math.log(0.1)
    )


def test_missing_book_uses_exact_baseline_and_remains_visible(tmp_path):
    f = fixture(tmp_path, fitted=True, missing_book=True)
    result = verify(f["root"], now=f["now"])
    assert result["fallback_missing_book"] == 1
    assert (
        result["metrics"]["SOLUSDT"]["depth"]
        == result["metrics"]["SOLUSDT"]["baseline"]
    )


def test_scored_case_metadata_uses_exact_same_cases_and_retains_flats(tmp_path):
    f = fixture(tmp_path, fitted=True)
    result = verify(f["root"], now=f["now"])
    cases = result["scored_cases"]
    assert result["end_at"] == CUTOFF + 3600
    assert len(cases) == result["scored_outcomes"] == 3
    assert cases == sorted(cases, key=lambda item: (item["anchor"], item["symbol"]))
    assert [case["label"] for case in cases] == [2, 0, 1]
    for symbol in SYMBOLS:
        selected = [case for case in cases if case["symbol"] == symbol]
        assert len(selected) == 1
        case = selected[0]
        assert case["anchor"] == CUTOFF
        assert case["target_end"] == CUTOFF + 360
        assert case["source_received_at"] == CUTOFF + 361
        assert case["forecast_sha256"] == file_sha(
            f["root"] / "forecasts" / f"{CUTOFF}-{symbol}.json"
        )
        moving = sum(case["label"] != 1 for case in selected)
        for metrics in result["metrics"][symbol].values():
            assert metrics["n"] == len(selected)
            assert metrics["moving_n"] == moving


@pytest.mark.parametrize("late", [False, True])
def test_absent_seal_or_late_confirmation_never_scores(tmp_path, late):
    f = fixture(tmp_path, fitted=True, omit_seal=not late, late=late)
    result = verify(f["root"], now=f["now"])
    assert result["candidate_forecasts"] == result["scored_outcomes"] == 0
    assert result["scored_cases"] == []
    assert (
        len(result["late_confirmations"] if late else result["unconfirmed_forecasts"])
        == 3
    )
    assert not result["clock_unhealthy_forecasts"]
    assert result["missing_emissions"] == 6


def test_wrong_coefficient_fails_independent_optimum_check(tmp_path):
    f = fixture(tmp_path, fitted=True)
    path = f["root"] / "model.json"
    model = json.loads(path.read_text())
    model["state"]["coefficient"] = 0.6
    write(path, model)
    with pytest.raises(ValueError, match="independent ridge optimum"):
        verify(f["root"], now=f["now"])


def test_complete_membership_cannot_drop_an_unfavourable_training_anchor(tmp_path):
    f = fixture(tmp_path, fitted=True)
    path = f["root"] / "training-data.json"
    data = json.loads(path.read_text())
    data["membership"].pop()
    for values in data["data"].values():
        values.pop()
    write(path, data)
    with pytest.raises(ValueError, match="membership or data is incomplete"):
        verify(f["root"], now=f["now"])


@pytest.mark.parametrize(
    "kind", ["candidate", "flat", "control", "source_availability", "clock"]
)
def test_rehashed_forecast_semantic_tampering_is_rejected(tmp_path, kind):
    f = fixture(tmp_path, fitted=True)

    def mutate(value):
        if kind == "candidate":
            value["models"]["depth"] = [0.6, 0.1, 0.3]
        elif kind == "flat":
            value["models"]["depth"] = [0.675, 0.11, 0.215]
        elif kind == "control":
            value["models"]["price_control"] = [0.6, 0.1, 0.3]
        elif kind == "source_availability":
            value["original"]["available_at"] = CUTOFF + 13
        else:
            value["issued_monotonic"] -= 5

    change_forecast(f["root"], "BTCUSDT", mutate)
    with pytest.raises(ValueError):
        verify(f["root"], now=f["now"])


def test_changed_frozen_runtime_is_rejected(tmp_path):
    f = fixture(tmp_path)
    (f["root"] / "runtime/research/depth_information.py").write_text("changed\n")
    with pytest.raises(ValueError, match="frozen code"):
        verify(f["root"], now=f["now"])


def test_copied_outcome_price_tampering_is_rejected(tmp_path):
    f = fixture(tmp_path, fitted=True)
    path = f["root"] / "outcomes" / f"{CUTOFF}-BTCUSDT.json"
    value = json.loads(path.read_text())
    value["exit_price"] = "102.00"
    write(path, value)
    with pytest.raises(ValueError, match="copied outcome differs"):
        verify(f["root"], now=f["now"])


def test_source_chain_tampering_is_rejected(tmp_path):
    f = fixture(tmp_path)
    connection = sqlite3.connect(f["capture"] / "depth-events.sqlite")
    connection.execute("UPDATE events SET payload='{}' WHERE seq=2")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="event hash"):
        verify(f["root"], now=f["now"])
