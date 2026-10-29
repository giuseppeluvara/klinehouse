import json
import sqlite3

import pytest

from research.shadow_ledger import Ledger, MODELS


START = 1800000000
SHA = "a" * 64


def config():
    return dict(start_at=START, end_at=START + 7 * 86400, snapshot_sha256=SHA)


def receipt(stamp=START, symbol="BTCUSDT"):
    return dict(
        source="spot_klines",
        symbol=symbol,
        content_times=[stamp],
        body=[[stamp, "100"]],
    )


def forecast(inputs, anchor=START):
    return dict(
        anchor=anchor,
        symbol="BTCUSDT",
        issued_at=anchor + 10,
        models={model: [0.45, 0.1, 0.45] for model in MODELS},
        input_ids=inputs,
        snapshot_sha256=SHA,
    )


def outcome(inputs, anchor=START):
    return dict(
        anchor=anchor,
        symbol="BTCUSDT",
        label=2,
        return_start=anchor + 60,
        return_end=anchor + 360,
        entry_price="100.00",
        exit_price="101.00",
        input_ids=inputs,
    )


def commit_ack(event, committed_at, *, healthy=True):
    return dict(
        anchor=event["payload"]["anchor"],
        symbol=event["payload"]["symbol"],
        forecast_seq=event["seq"],
        forecast_hash=event["digest"],
        committed_at=committed_at,
        eligible=committed_at <= event["payload"]["anchor"] + 30 and healthy,
        clock_healthy=healthy,
    )


def test_full_cycle_durable_restart_and_read_only_reader(tmp_path):
    clock = [START + 5]
    path = tmp_path / "ledger.sqlite3"
    with Ledger(path, config(), clock=lambda: clock[0]) as ledger:
        source = ledger.append("receipt", receipt(), received_at=START + 1)
        clock[0] = START + 11
        assert ledger.append("forecast", forecast([source])) == 2
        with Ledger(path, config(), read_only=True) as reader:
            assert reader.status()["counts"] == dict(receipt=1, forecast=1)
            assert reader.verify()["events"] == 2
            with pytest.raises(ValueError, match="read-only"):
                reader.append("heartbeat", {})
        clock[0] = START + 361
        end = ledger.append("receipt", receipt(START + 360))
        ledger.append("outcome", outcome([source, end]))
        head = ledger.verify()
        assert ledger.checkpoint() is not None
    with Ledger(path, config()) as reopened:
        assert reopened.verify() == head
        assert reopened.last("forecast")["seq"] == 2
        assert [e["seq"] for e in reopened.rows(after_id=2)] == [3, 4]


def test_immutable_config_and_event_mutation_are_refused(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    with Ledger(path, config(), clock=lambda: START) as ledger:
        ledger.append("receipt", receipt())
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            ledger.connection.execute("UPDATE events SET payload='{}'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            ledger.connection.execute("DELETE FROM events")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            ledger.connection.execute("UPDATE meta SET value='{}'")
    changed = dict(config(), snapshot_sha256="b" * 64)
    with pytest.raises(ValueError, match="configuration differs"):
        Ledger(path, changed)


def test_hash_chain_detects_tampering_after_guard_bypass(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    with Ledger(path, config(), clock=lambda: START) as ledger:
        ledger.append("receipt", receipt())
    db = sqlite3.connect(path)
    db.execute("DROP TRIGGER events_no_update")
    changed = dict(receipt(), body=[[START, "999"]])
    db.execute("UPDATE events SET payload=?", (json.dumps(changed),))
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="hash chain"):
        Ledger(path, read_only=True)


def test_protected_or_future_receipts_and_nonfinite_values_rejected(tmp_path):
    with Ledger(tmp_path / "log", config(), clock=lambda: START + 5) as ledger:
        for stamp in (START - 1, START + 10):
            with pytest.raises(ValueError, match="source content"):
                ledger.append("receipt", receipt(stamp))
        with pytest.raises(ValueError, match="receipt time"):
            ledger.append("receipt", receipt(), received_at=START - 1)
        with pytest.raises(ValueError):
            ledger.append("receipt", dict(receipt(), body={"bad": float("nan")}))
        assert ledger.verify()["events"] == 0


def test_late_backdated_forecast_missing_and_duplicate_guards(tmp_path):
    clock = [START + 1]
    with Ledger(tmp_path / "log", config(), clock=lambda: clock[0]) as ledger:
        source = ledger.append("receipt", receipt())
        clock[0] = START + 31
        with pytest.raises(ValueError, match="late or backdated"):
            ledger.append("forecast", forecast([source]), received_at=START + 10)
        ledger.append(
            "missing", dict(anchor=START, symbol="BTCUSDT", reason="deadline")
        )
        clock[0] = START + 11
        with pytest.raises(sqlite3.IntegrityError):
            ledger.append("forecast", forecast([source]))
        with pytest.raises(sqlite3.IntegrityError):
            ledger.append(
                "missing", dict(anchor=START, symbol="BTCUSDT", reason="again")
            )
        assert ledger.verify()["events"] == 2


def test_forecast_requires_real_earlier_inputs_and_snapshot(tmp_path):
    clock = [START + 12]
    with Ledger(tmp_path / "log", config(), clock=lambda: clock[0]) as ledger:
        later = ledger.append("receipt", receipt())
        for inputs in ([], [99], [later], [later, later]):
            with pytest.raises(ValueError):
                ledger.append("forecast", forecast(inputs))
        earlier = ledger.append("receipt", receipt(), received_at=START + 2)
        wrong = dict(forecast([earlier]), snapshot_sha256="b" * 64)
        with pytest.raises(ValueError, match="snapshot"):
            ledger.append("forecast", wrong)
        bad = forecast([earlier])
        bad["models"]["positions"] = [0.7, 0.2, 0.2]
        with pytest.raises(ValueError, match="probabilities"):
            ledger.append("forecast", bad)
        ledger.append("forecast", forecast([earlier]))
        with pytest.raises(sqlite3.IntegrityError):
            ledger.append("forecast", forecast([earlier]))


def test_outcome_maturity_interval_and_duplicate_guards(tmp_path):
    clock = [START + 1]
    with Ledger(tmp_path / "log", config(), clock=lambda: clock[0]) as ledger:
        source = ledger.append("receipt", receipt())
        clock[0] = START + 11
        ledger.append("forecast", forecast([source]))
        with pytest.raises(ValueError, match="not matured"):
            ledger.append("outcome", outcome([source]))
        clock[0] = START + 361
        with pytest.raises(ValueError, match="interval"):
            ledger.append("outcome", dict(outcome([source]), return_end=START + 300))
        ledger.append("outcome", outcome([source]))
        with pytest.raises(sqlite3.IntegrityError):
            ledger.append("outcome", outcome([source]))


def test_failed_transaction_leaves_previous_complete_head(tmp_path):
    path = tmp_path / "log"
    with Ledger(path, config(), clock=lambda: START + 1) as ledger:
        ledger.append("receipt", receipt())
        previous = ledger.verify()
        ledger.connection.execute("BEGIN IMMEDIATE")
        ledger.connection.execute(
            "INSERT INTO events VALUES (2,'broken',0,0,NULL,NULL,'{}','x','y')"
        )
        # Closing without COMMIT models process termination during a transaction.
    with Ledger(path, config()) as reopened:
        assert reopened.verify() == previous


def test_prediction_outside_trial_unknown_symbol_and_invalid_config(tmp_path):
    for override in (
        dict(end_at=START),
        dict(end_at=START + 9 * 86400),
        dict(issuance_deadline_seconds=60),
        dict(snapshot_sha256="wrong"),
    ):
        with pytest.raises(ValueError):
            Ledger(tmp_path / str(len(str(override))), dict(config(), **override))
    with Ledger(tmp_path / "good", config(), clock=lambda: START + 11) as ledger:
        source = ledger.append("receipt", receipt(), received_at=START + 1)
        for bad in (
            dict(forecast([source]), anchor=START - 300),
            dict(forecast([source]), anchor=START + 1),
            dict(forecast([source]), symbol="OTHER"),
        ):
            with pytest.raises(ValueError):
                ledger.append("forecast", bad)


def test_final_content_boundary_delayed_receipts_and_body_digest(tmp_path):
    end = config()["end_at"]
    clock = [end + 100]
    with Ledger(tmp_path / "log", config(), clock=lambda: clock[0]) as ledger:
        ledger.append("receipt", receipt(end))
        with pytest.raises(ValueError, match="source content"):
            ledger.append("receipt", receipt(end + 1))
        with pytest.raises(ValueError, match="digest"):
            ledger.append("receipt", dict(receipt(end), body_sha256="wrong"))
        late_anchor = (end // 300) * 300
        with pytest.raises(ValueError, match="outside prospective"):
            ledger.append(
                "missing", dict(anchor=late_anchor, symbol="BTCUSDT", reason="end")
            )
        clock[0] = end + 121
        with pytest.raises(ValueError, match="collection deadline"):
            ledger.append("receipt", receipt(end))
        # The stop receipt can still describe the observed delayed shutdown.
        ledger.append("stop", dict(reason="expired", symbol="BTCUSDT"))
        assert ledger.verify()["events"] == 2


def test_acknowledgement_proves_prior_commit_even_when_ack_write_is_later(tmp_path):
    clock = [START + 1]
    cfg = dict(config(), require_commit_ack=True)
    with Ledger(tmp_path / "log", cfg, clock=lambda: clock[0]) as ledger:
        source = ledger.append("receipt", receipt())
        clock[0] = START + 11
        ledger.append("forecast", forecast([source]))
        event = ledger.last("forecast")
        observed_commit = clock[0]
        # A delayed acknowledgement describes the already observed completed
        # forecast commit, without backdating the acknowledgement event itself.
        clock[0] = START + 50
        ack = commit_ack(event, observed_commit)
        with pytest.raises(ValueError, match="provenance"):
            ledger.append("forecast_commit", dict(ack, forecast_hash="b" * 64))
        ledger.append("forecast_commit", ack)
        assert ledger.last()["received_at"] == START + 50
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            ledger.append("forecast_commit", ack)
        clock[0] = START + 361
        ledger.append("outcome", outcome([source]))
        assert ledger.status()["counts"]["outcome"] == 1
        assert ledger.verify()["events"] == 4


def test_commit_stall_cannot_be_certified_with_precommit_timestamp(tmp_path):
    clock = [START + 1]
    with Ledger(tmp_path / "log", dict(config(), require_commit_ack=True),
                clock=lambda: clock[0]) as ledger:
        source = ledger.append("receipt", receipt())
        clock[0] = START + 11

        def stall_during_commit(statement):
            if statement == "COMMIT":
                clock[0] = START + 70

        # SQLite's COMMIT callback advances the simulated wall clock during the
        # transaction, before append can observe durable completion and return.
        ledger.connection.set_trace_callback(stall_during_commit)
        ledger.append("forecast", forecast([source]))
        ledger.connection.set_trace_callback(None)
        event = ledger.last("forecast")
        assert event["received_at"] == START + 11
        assert clock[0] == START + 70
        with pytest.raises(ValueError, match="predates observed durable commit"):
            ledger.append("forecast_commit", commit_ack(event, START + 11))
        ledger.append("forecast_commit", commit_ack(event, clock[0]))
        assert ledger.last()["payload"]["eligible"] is False
        clock[0] = START + 361
        with pytest.raises(ValueError, match="eligible post-commit"):
            ledger.append("outcome", outcome([source]))
        assert ledger.verify()["events"] == 3


def test_crash_before_ack_cannot_retroactively_certify_a_forecast(tmp_path):
    path = tmp_path / "log"
    clock = [START + 1]
    cfg = dict(config(), require_commit_ack=True)
    with Ledger(path, cfg, clock=lambda: clock[0]) as ledger:
        source = ledger.append("receipt", receipt())
        clock[0] = START + 11
        ledger.append("forecast", forecast([source]))
        event = ledger.last("forecast")
    clock[0] = START + 12
    with Ledger(path, cfg, clock=lambda: clock[0]) as reopened:
        with pytest.raises(ValueError, match="not observed by this writer"):
            reopened.append("forecast_commit", commit_ack(event, START + 11))
        clock[0] = START + 361
        with pytest.raises(ValueError, match="eligible post-commit"):
            reopened.append("outcome", outcome([source]))
        assert reopened.verify()["events"] == 2


def test_unhealthy_clock_ack_is_retained_but_cannot_enable_scoring(tmp_path):
    clock = [START + 1]
    with Ledger(tmp_path / "log", dict(config(), require_commit_ack=True),
                clock=lambda: clock[0]) as ledger:
        source = ledger.append("receipt", receipt())
        clock[0] = START + 11
        ledger.append("forecast", forecast([source]))
        ack = commit_ack(ledger.last("forecast"), clock[0], healthy=False)
        with pytest.raises(ValueError, match="eligibility"):
            ledger.append("forecast_commit", dict(ack, eligible=True))
        ledger.append("forecast_commit", ack)
        clock[0] = START + 361
        with pytest.raises(ValueError, match="eligible post-commit"):
            ledger.append("outcome", outcome([source]))
