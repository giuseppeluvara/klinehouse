"""Independent audit failures; these tests import neither runner nor learner."""

import copy
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from scripts import verify_direction_orientation as audit


def sample(n=8):
    anchor = np.arange(n, dtype=np.int64) * 300
    p = np.column_stack(
        (np.linspace(0.3, 0.5, n), np.full(n, 0.1), np.linspace(0.6, 0.4, n))
    )
    return dict(
        anchor=anchor,
        issue=anchor.astype(float) + 10,
        label_at=anchor.astype(float) + 360,
        label=np.full(n, 2, dtype=np.int64),
        baseline=p,
        price_control=p.copy(),
        positions=p.copy(),
    )


def complete(source):
    return {
        **copy.deepcopy(source),
        **audit.independent_orientation(
            source["baseline"], source["issue"], source["label_at"], source["label"]
        ),
    }


def test_receipt_sorted_prefix_includes_late_feedback_once_and_only_same_lane():
    source = sample()
    source["label_at"][0] = source["issue"][6]
    source["label"][1] = 1
    source["label"][2] = 0
    result = complete(source)
    np.testing.assert_array_equal(result["feedback_count"][::2], [0, 0, 1, 3])
    assert result["feedback_count"][3] == 0  # lane one's flat is not feedback
    q = source["baseline"][:, 2] / source["baseline"][:, [0, 2]].sum(1)
    likelihood = np.array([1 - q[2], q[2], 0.5])
    np.testing.assert_allclose(result["weights"][4], likelihood / likelihood.sum())
    likelihood *= np.array([q[0], 1 - q[0], 0.5]) * np.array([q[4], 1 - q[4], 0.5])
    np.testing.assert_allclose(result["weights"][6], likelihood / likelihood.sum())
    assert audit.audit_prediction_data(result, source)["weights"] == 0


def test_audit_rejects_corrupt_posterior_even_when_probabilities_are_unchanged():
    source = sample()
    result = complete(source)
    result["weights"][4] = [0.8, 0.1, 0.1]
    with pytest.raises(ValueError, match="weights reconstruction"):
        audit.audit_prediction_data(result, source)


def test_audit_rejects_backdated_outcome_and_changed_source():
    source = sample()
    result = complete(source)
    result["label"][0] = 0
    with pytest.raises(ValueError, match="source differs: label"):
        audit.audit_prediction_data(result, source)
    source["label_at"][0] = source["issue"][0]
    with pytest.raises(ValueError, match="availability chronology"):
        complete(source)


def test_flat_probabilities_and_initial_neutrality_are_exact():
    source = sample()
    source["label"][:] = 1
    result = complete(source)
    np.testing.assert_array_equal(result["adaptive"][:, 0], result["adaptive"][:, 2])
    np.testing.assert_array_equal(result["adaptive"][:, 1], source["baseline"][:, 1])
    np.testing.assert_allclose(result["weights"], 1 / 3)
    assert not result["feedback_count"].any()
    result["neutral"][0, 1] += 1e-12
    with pytest.raises(ValueError, match="flat probability changed"):
        audit.audit_prediction_data(result, source)


def test_unknown_labels_and_future_label_changes_cannot_change_prior_prefix():
    source = sample(12)
    first = complete(source)
    source["label"][8:] = -1
    source["label_at"][8:] = np.nan
    second = complete(source)
    np.testing.assert_array_equal(first["weights"][:10], second["weights"][:10])
    np.testing.assert_array_equal(first["adaptive"][:10], second["adaptive"][:10])
    assert second["feedback_count"][-1] < first["feedback_count"][-1]


def test_score_denominators_keep_flats_and_original_selection_flat_errors():
    source = sample(4)
    source["label"][:] = [2, 1, 0, -1]
    source["label_at"][-1] = np.nan
    result = complete(source)
    score = audit.independent_scores(
        source["baseline"], source["label"], np.ones(4, bool)
    )
    assert (score["n"], score["moving_n"], score["flat_n"]) == (3, 2, 1)
    report = audit.independent_report(result)
    selected = report["models"]["baseline"]["own_selection"]
    assert selected["n"] == 2 and selected["original_accuracy_flat_errors"] == 0.5
    assert report["pending"] == 1
    changed = copy.deepcopy(report)
    changed["models"]["baseline"]["all"]["moving_n"] = 3
    with pytest.raises(ValueError, match="report value differs"):
        audit.compare_report(changed, report)


def test_historical_protection_is_checked_before_opening_numeric_payload(tmp_path):
    anchor = np.array([audit.BOUNDARY], dtype=np.int64)
    pred, labels = tmp_path / "pred.npz", tmp_path / "labels.npz"
    # Object payload would itself fail allow_pickle=False if opened too early.
    np.savez(pred, anchor=anchor, baseline=np.array([object()], dtype=object))
    np.savez(
        labels,
        anchor=anchor,
        end=anchor + 360,
        label=np.array([object()], dtype=object),
    )
    with pytest.raises(ValueError, match="protected or out-of-period"):
        audit.historical_source(pred, labels, 2026)


def test_original_2024_july_target_can_end_in_unprotected_august(tmp_path):
    anchor = np.array([int(np.datetime64("2024-07-31T23:55:00", "s").astype(np.int64))])
    p = np.array([[0.45, 0.1, 0.45]])
    pred, labels = tmp_path / "pred.npz", tmp_path / "labels.npz"
    np.savez(pred, anchor=anchor, baseline=p, price_control=p, positions=p)
    np.savez(labels, anchor=anchor, end=anchor + 360, label=np.array([2]))
    source = audit.historical_source(pred, labels, 2024)
    assert source["label_at"][0] > int(
        np.datetime64("2024-08-01", "s").astype(np.int64)
    )


def live_fixture(path, *, backdated=False):
    path.mkdir()
    (path / "model.json").write_text('{"synthetic":true}\n')
    anchor = 1793276100
    config = dict(
        require_commit_ack=True,
        start_at=anchor - 3600,
        end_at=anchor + 7200,
        snapshot_sha256=hashlib.sha256((path / "model.json").read_bytes()).hexdigest(),
    )
    (path / "config.json").write_text(json.dumps(config))
    c = sqlite3.connect(path / "events.sqlite")
    c.executescript(
        "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE events(seq INTEGER PRIMARY KEY,kind TEXT,received_at REAL,received_ns INTEGER,anchor INTEGER,symbol TEXT,payload TEXT,prev_hash TEXT,digest TEXT);"
    )
    previous = audit.object_hash(config)
    c.executemany(
        "INSERT INTO meta VALUES (?,?)",
        [("config", audit.canonical(config)), ("config_sha256", previous)],
    )
    seq = 0

    def append(kind, payload, received):
        nonlocal seq, previous
        seq += 1
        ns = int(received * 1e9)
        value = audit.object_hash(
            dict(
                seq=seq, kind=kind, received_ns=ns, payload=payload, prev_hash=previous
            )
        )
        c.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
            (
                seq,
                kind,
                received,
                ns,
                payload.get("anchor"),
                payload.get("symbol"),
                audit.canonical(payload),
                previous,
                value,
            ),
        )
        previous = value
        return seq, value

    for symbol in audit.SYMBOLS:
        p = dict(
            anchor=anchor,
            symbol=symbol,
            issued_at=anchor + 10,
            snapshot_sha256=config["snapshot_sha256"],
            models={arm: [0.4, 0.1, 0.5] for arm in audit.ARMS[:3]},
        )
        forecast_seq, forecast_hash = append("forecast", p, anchor + 10)
        ack = dict(
            anchor=anchor,
            symbol=symbol,
            forecast_seq=forecast_seq,
            forecast_hash=forecast_hash,
            committed_at=anchor + 11,
            clock_healthy=True,
            eligible=True,
        )
        append("forecast_commit", ack, anchor + 12)
    for symbol in audit.SYMBOLS:
        body = [
            [anchor * 1000, 0, 0, 0, "100"],
            [(anchor + 300) * 1000, 0, 0, 0, "101"],
        ]
        receipt_seq, _ = append(
            "receipt",
            dict(source="spot_klines", symbol=symbol, body=body),
            anchor + 361,
        )
        outcome = dict(
            anchor=anchor,
            symbol=symbol,
            return_start=anchor + 60,
            return_end=anchor + 360,
            entry_price="100",
            exit_price="101",
            label=2,
            entry_receipt_id=receipt_seq,
            exit_receipt_id=receipt_seq,
        )
        append("outcome", outcome, anchor + (359 if backdated else 362))
    c.commit()
    c.close()
    (path / "general-audit.json").write_text(
        json.dumps(dict(verified=True, event_count=seq, chain_tip=previous))
    )


def test_live_source_reconstructs_eligible_emissions_and_rejects_early_outcome(
    tmp_path,
):
    live_fixture(tmp_path / "valid")
    data = audit.live_sources(tmp_path / "valid")
    assert set(data) == set(audit.SYMBOLS)
    assert data["BTCUSDT"]["label"].tolist() == [2]
    live_fixture(tmp_path / "early", backdated=True)
    with pytest.raises(ValueError, match="live outcome is early"):
        audit.live_sources(tmp_path / "early")
