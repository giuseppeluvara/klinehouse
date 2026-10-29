"""Independent SQLite fixtures exercise the auditor without production writers."""

import copy
import hashlib
import json
import math
from pathlib import Path
import runpy
import sqlite3

import pytest

from scripts.verify_shadow_run import canonical, sha, verify


START = 1800000000
ANCHOR = START + 4800
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def candle(stamp, close="100.0"):
    return [
        stamp * 1000,
        "100",
        "102",
        "98",
        close,
        "5",
        stamp * 1000 + 59999,
        "500",
        3,
        "2",
        "200",
        "0",
    ]


def receipt(symbol, body, at):
    return (
        "receipt",
        at,
        {
            "source": "spot_klines",
            "symbol": symbol,
            "body": body,
            "body_sha256": sha(body),
            "content_times": [row[0] / 1000 for row in body],
            "request_params": {
                "symbol": symbol,
                "interval": "1m",
                "startTime": START * 1000,
                "endTime": int(at) * 1000 - 1,
                "limit": 1000,
            },
            "requested_at": at - 0.2,
        },
    )


def fixture_events(snapshot_hash):
    events = [
        receipt(symbol, [candle(t) for t in range(START, ANCHOR, 60)], ANCHOR + 1 + i)
        for i, symbol in enumerate(SYMBOLS)
    ]
    for symbol in SYMBOLS:
        events.append(
            (
                "forecast",
                ANCHOR + 20,
                {
                    "anchor": ANCHOR,
                    "symbol": symbol,
                    "issued_at": ANCHOR + 19,
                    "snapshot_sha256": snapshot_hash,
                    "input_ids": [1, 2, 3],
                    "models": {
                        arm: [0.4, 0.1, 0.5]
                        for arm in ("baseline", "price_control", "positions")
                    },
                },
            )
        )
    exits = ("101", "99", "100.00")
    for i, (symbol, end) in enumerate(zip(SYMBOLS, exits)):
        body = [
            candle(t, end if t == ANCHOR + 300 else "100.0")
            for t in range(ANCHOR, ANCHOR + 360, 60)
        ]
        events.append(receipt(symbol, body, ANCHOR + 361 + i))
    for i, (symbol, end) in enumerate(zip(SYMBOLS, exits)):
        events.append(
            (
                "outcome",
                ANCHOR + 365,
                {
                    "anchor": ANCHOR,
                    "symbol": symbol,
                    "label": (2, 0, 1)[i],
                    "return_start": ANCHOR + 60,
                    "return_end": ANCHOR + 360,
                    "entry_price": "100.0",
                    "exit_price": end,
                    "return": math.log(float(end) / 100),
                    "entry_receipt_id": 7 + i,
                    "exit_receipt_id": 7 + i,
                    "input_ids": [7 + i],
                },
            )
        )
    return events


def with_commit_acks(events):
    """Insert post-commit acknowledgments and remap original receipt references."""
    mapping, extra = {}, 0
    for old_seq, (kind, _, _) in enumerate(events, 1):
        mapping[old_seq] = old_seq + extra
        extra += kind == "forecast"
    result = []
    for old_seq, (kind, received, original) in enumerate(events, 1):
        payload = copy.deepcopy(original)
        if "input_ids" in payload:
            payload["input_ids"] = [mapping.get(i, i) for i in payload["input_ids"]]
        for key in ("entry_receipt_id", "exit_receipt_id"):
            if key in payload:
                payload[key] = mapping.get(payload[key], payload[key])
        result.append((kind, received, payload))
        if kind == "forecast":
            result.append(
                (
                    "forecast_commit",
                    received,
                    {
                        "anchor": payload["anchor"],
                        "symbol": payload["symbol"],
                        "forecast_seq": mapping[old_seq],
                        "forecast_hash": None,
                        "committed_at": received,
                        "clock_healthy": True,
                        "eligible": received <= payload["anchor"] + 30,
                    },
                )
            )
    return result


def make_database(tmp_path, change=None, *, ack_change=None, acknowledgments=True):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text('{"frozen":true}\n')
    config = {
        "start_at": START,
        "end_at": START + 7 * 86400,
        "issuance_deadline_seconds": 30,
        "require_commit_ack": True,
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(config))
    events = fixture_events(config["snapshot_sha256"])
    if change:
        change(events)
    if acknowledgments:
        events = with_commit_acks(events)
    if ack_change:
        ack_change(events)
    database = tmp_path / "ledger.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE events(seq INTEGER PRIMARY KEY,kind TEXT,received_at REAL,received_ns INTEGER,anchor INTEGER,symbol TEXT,payload TEXT,prev_hash TEXT,digest TEXT)"
        )
        connection.executemany(
            "INSERT INTO meta VALUES (?,?)",
            [("config", canonical(config)), ("config_sha256", sha(config))],
        )
        previous = sha(config)
        hashes = {}
        for i, (kind, received, payload) in enumerate(events, 1):
            if kind == "forecast_commit" and payload.get("forecast_hash") is None:
                payload["forecast_hash"] = hashes.get(payload["forecast_seq"], "0" * 64)
            nanoseconds = round(received * 1_000_000_000)
            digest = sha(
                {
                    "seq": i,
                    "kind": kind,
                    "received_ns": nanoseconds,
                    "payload": payload,
                    "prev_hash": previous,
                }
            )
            connection.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    i,
                    kind,
                    received,
                    nanoseconds,
                    payload.get("anchor"),
                    payload.get("symbol"),
                    canonical(payload),
                    previous,
                    digest,
                ),
            )
            previous = digest
            hashes[i] = digest
    return database, snapshot, protocol


def test_independent_read_only_audit_reconstructs_both_scores_and_direction(tmp_path):
    database, snapshot, protocol = make_database(tmp_path)
    before = database.read_bytes()
    result = verify(database, snapshot=snapshot, protocol=protocol, now=ANCHOR + 366)
    assert database.read_bytes() == before
    assert result["verified"] and not result["technical_pipeline_observed"]
    assert result["eligible_forecast_count"] == 3
    assert result["accepted_active_outcomes_by_symbol"] == dict.fromkeys(SYMBOLS, 0)
    assert result["forecast_count"] == result["outcome_count"] == 3
    assert not result["statistical_superiority_verified"]
    delay = result["source_first_receipt_delay_seconds"]["spot_klines"]["BTCUSDT"]
    assert delay["count"] == 86
    assert delay["min"] == 1
    assert delay["max"] == ANCHOR - START - 59
    btc = result["metrics"]["BTCUSDT"]["baseline"]
    assert btc["brier"]["mean"] == pytest.approx(0.42)
    assert btc["log_loss"]["mean"] == pytest.approx(-math.log(0.5))
    assert btc["direction_accuracy"]["mean"] == 1
    assert btc["direction_brier"]["mean"] == pytest.approx((0.5 / 0.9 - 1) ** 2)
    assert "direction_brier" not in result["metrics"]["SOLUSDT"]["baseline"]
    assert (
        result["paired_mean_gains"]["BTCUSDT"]["positions_vs_baseline"]["log_loss"][
            "mean"
        ]
        == 0
    )


def test_uncorrected_digest_tamper_is_detected(tmp_path):
    database, _, _ = make_database(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE events SET digest=? WHERE seq=2", ("0" * 64,))
    with pytest.raises(ValueError, match="digest"):
        verify(database, now=ANCHOR + 366)


@pytest.mark.parametrize(
    "fault",
    (
        "late",
        "future_receipt",
        "wrong_label",
        "wrong_price",
        "wrong_return",
        "bad_probability",
        "old_request",
        "raw_time",
        "body_hash",
        "early_outcome",
        "changed_snapshot",
    ),
)
def test_semantic_errors_rejected_even_when_hash_chain_is_rebuilt(tmp_path, fault):
    def change(events):
        if fault == "late":
            events[3][2]["issued_at"] = ANCHOR + 31
        elif fault == "future_receipt":
            events[3][2]["input_ids"] = [1, 2, 7]
        elif fault == "wrong_label":
            events[9][2]["label"] = 0
        elif fault == "wrong_price":
            events[9][2]["exit_price"] = "102"
            events[9][2]["return"] = math.log(1.02)
        elif fault == "wrong_return":
            events[9][2]["return"] = 0.5
        elif fault == "bad_probability":
            events[3][2]["models"]["baseline"][0] = 0.8
        elif fault == "old_request":
            events[0][2]["request_params"]["startTime"] = (START - 60) * 1000
        elif fault == "raw_time":
            events[0][2]["content_times"][0] += 60
        elif fault == "body_hash":
            events[0][2]["body_sha256"] = "0" * 64
        elif fault == "early_outcome":
            events[9] = (events[9][0], ANCHOR + 359, events[9][2])
        elif fault == "changed_snapshot":
            events[3][2]["snapshot_sha256"] = "0" * 64

    database, _, _ = make_database(tmp_path, change)
    with pytest.raises(ValueError):
        verify(database, now=ANCHOR + 366)


def test_exact_decimal_direction_does_not_collapse_float_rounding(tmp_path):
    def change(events):
        p = events[9][2]
        p.update(
            entry_price="1000000000000000.01",
            exit_price="1000000000000000.02",
            label=2,
            **{"return": 0.0},
        )
        raw = events[6][2]
        raw["body"][0][4] = p["entry_price"]
        raw["body"][-1][4] = p["exit_price"]
        raw["body_sha256"] = sha(raw["body"])

    database, _, _ = make_database(tmp_path, change)
    result = verify(database, now=ANCHOR + 366)
    assert result["metrics"]["BTCUSDT"]["baseline"]["direction_accuracy"]["mean"] == 1


def test_empty_database_is_auditable_but_not_a_verified_pipeline(tmp_path):
    database, _, _ = make_database(tmp_path, lambda events: events.clear())
    result = verify(database, now=START + 31)
    assert result["verified"] and not result["technical_pipeline_observed"]
    assert result["unaccounted_anchor_symbols"] == [
        [START, symbol] for symbol in SYMBOLS
    ]


def test_duplicate_forecast_rejected_despite_rehashed_chain(tmp_path):
    def change(events):
        events.insert(6, copy.deepcopy(events[3]))

    database, _, _ = make_database(tmp_path, change)
    with pytest.raises(ValueError, match="duplicate"):
        verify(database, now=ANCHOR + 366)


def test_external_binding_detects_replaced_snapshot(tmp_path):
    database, snapshot, _ = make_database(tmp_path)
    snapshot.write_text('{"different":true}\n')
    with pytest.raises(ValueError, match="snapshot hash"):
        verify(database, snapshot=snapshot, now=ANCHOR + 366)


def test_complete_position_context_allows_nontrivial_corrections(tmp_path):
    def change(events):
        extra = []
        for symbol in SYMBOLS:
            for source, field in (
                ("open_interest", "sumOpenInterest"),
                ("position_ratio", "longShortRatio"),
            ):
                body = [
                    {"timestamp": stamp * 1000, "symbol": symbol, field: "1.2"}
                    for stamp in range(ANCHOR - 4500, ANCHOR - 899, 300)
                ]
                extra.append(
                    (
                        "receipt",
                        ANCHOR + 4 + len(extra),
                        {
                            "source": source,
                            "symbol": symbol,
                            "body": body,
                            "body_sha256": sha(body),
                            "content_times": [row["timestamp"] / 1000 for row in body],
                            "request_params": {
                                "symbol": symbol,
                                "period": "5m",
                                "startTime": START * 1000,
                                "endTime": ANCHOR * 1000,
                                "limit": 100,
                            },
                        },
                    )
                )
        events[3:3] = extra
        for kind, _, payload in events:
            if kind == "forecast":
                payload["input_ids"] = list(range(1, 10))
                payload["models"]["positions"] = [0.35, 0.1, 0.55]
                payload["models"]["price_control"] = [0.42, 0.1, 0.48]
                payload["diagnostics"] = {"correction_available": True}
            if kind == "outcome":
                payload["entry_receipt_id"] += 6
                payload["exit_receipt_id"] += 6
                payload["input_ids"] = [i + 6 for i in payload["input_ids"]]

    database, _, _ = make_database(tmp_path, change)
    result = verify(database, now=ANCHOR + 366)
    assert result["technical_pipeline_observed"]
    assert result["paired_mean_gains"]["BTCUSDT"]["positions_vs_baseline"]["log_loss"][
        "mean"
    ] == pytest.approx(math.log(0.55 / 0.5))


def test_partial_position_context_requires_exact_shared_fallback(tmp_path):
    def change(events):
        events[3][2]["models"]["positions"] = [0.35, 0.1, 0.55]

    database, _, _ = make_database(tmp_path, change)
    with pytest.raises(ValueError, match="common fallback"):
        verify(database, now=ANCHOR + 366)


def test_incomplete_spot_context_cannot_be_hidden_by_valid_probabilities(tmp_path):
    def change(events):
        receipt = events[0][2]
        del receipt["body"][-2]
        del receipt["content_times"][-2]
        receipt["body_sha256"] = sha(receipt["body"])

    database, _, _ = make_database(tmp_path, change)
    with pytest.raises(ValueError, match="63-minute"):
        verify(database, now=ANCHOR + 366)


def test_real_runner_ledger_passes_independent_end_to_end_audit(tmp_path, monkeypatch):
    # Production collection/inference/writer are exercised with an injected clock
    # and synthetic public-source responses. The auditor itself imports none of
    # those implementations and recalculates its evidence directly from SQLite.
    helpers = runpy.run_path(str(Path(__file__).with_name("test_shadow_runner.py")))
    clock = [ANCHOR + 10]
    instance, _, root = helpers["make_runner"](tmp_path, monkeypatch, clock)
    try:
        instance.check_clock()
        instance.collect()
        instance.decide()
        clock[0] = ANCHOR + 61
        instance.collect()
        instance.resolve()
        clock[0] = ANCHOR + 361
        instance.collect()
        instance.resolve()
        instance.decide()
        result = verify(
            root / "events.sqlite",
            snapshot=root / "model.json",
            protocol=root / "config.json",
            now=clock[0],
        )
        assert result["technical_pipeline_observed"]
        assert result["forecast_count"] == result["outcome_count"] == 3
        assert result["unaccounted_anchor_symbols"] == []
        assert result["matured_pending_outcomes"] == []
        assert result["external_protocol_verified"]
        assert result["external_snapshot_verified"]
        unique_labels = {}
        for event in instance.ledger.rows("receipt"):
            payload = event["payload"]
            unique_labels.setdefault(
                (payload["source"], payload["symbol"]), set()
            ).update(payload["content_times"])
        for (source, symbol), labels in unique_labels.items():
            assert result["source_first_receipt_delay_seconds"][source][symbol][
                "count"
            ] == len(labels)
    finally:
        instance.ledger.close()


def test_crash_before_ack_is_visible_and_never_scored(tmp_path):
    def change(events):
        events[:] = [event for event in events if event[0] != "outcome"]

    database, _, _ = make_database(tmp_path, change, acknowledgments=False)
    result = verify(database, now=ANCHOR + 366)
    assert result["forecast_count"] == 3
    assert result["eligible_forecast_count"] == 0
    assert result["unacknowledged_forecasts"] == [
        [ANCHOR, symbol] for symbol in SYMBOLS
    ]
    assert result["metrics"] == {}
    assert not result["technical_pipeline_observed"]


def test_outcome_without_ack_is_rejected_not_silently_omitted(tmp_path):
    database, _, _ = make_database(tmp_path, acknowledgments=False)
    with pytest.raises(ValueError, match="post-commit"):
        verify(database, now=ANCHOR + 366)


@pytest.mark.parametrize("reason", ("late", "clock_unhealthy"))
def test_ineligible_commits_are_accounted_and_not_scored(tmp_path, reason):
    def single(events):
        del events[4:]

    def acknowledgment(events):
        kind, received, payload = events[-1]
        payload["eligible"] = False
        if reason == "late":
            payload["committed_at"] = ANCHOR + 31
            received = ANCHOR + 31
        else:
            payload["clock_healthy"] = False
        events[-1] = (kind, received, payload)

    database, _, _ = make_database(tmp_path, single, ack_change=acknowledgment)
    result = verify(database, now=ANCHOR + 366)
    assert result["ineligible_commit_forecasts"] == [[ANCHOR, "BTCUSDT"]]
    assert result["eligible_forecast_count"] == result["outcome_count"] == 0
    assert result["unacknowledged_forecasts"] == []
    assert result["late_commit_forecasts"] == (
        [[ANCHOR, "BTCUSDT"]] if reason == "late" else []
    )
    assert result["clock_unhealthy_forecasts"] == (
        [[ANCHOR, "BTCUSDT"]] if reason == "clock_unhealthy" else []
    )


@pytest.mark.parametrize(
    "fault", ("hash", "sequence", "premature", "eligibility", "duplicate")
)
def test_post_commit_acknowledgment_provenance_is_checked(tmp_path, fault):
    def single(events):
        del events[4:]

    def acknowledgment(events):
        payload = events[-1][2]
        if fault == "hash":
            payload["forecast_hash"] = "0" * 64
        elif fault == "sequence":
            payload["forecast_seq"] = 1
        elif fault == "premature":
            payload["committed_at"] -= 1
        elif fault == "eligibility":
            payload["eligible"] = False
        else:
            events.append(copy.deepcopy(events[-1]))

    database, _, _ = make_database(tmp_path, single, ack_change=acknowledgment)
    with pytest.raises(ValueError, match="acknowledgment"):
        verify(database, now=ANCHOR + 366)
