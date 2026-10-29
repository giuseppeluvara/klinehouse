"""Independent hand-built prospective evidence; no production writer or model."""

import copy
import hashlib
import json
import math
import sqlite3

import pytest

from scripts.verify_shadow_first_cycle import arithmetic, canonical, sha, verify


START = 1800000000
ANCHOR = START + 4800
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MODELS = {
    "baseline": [0.4, 0.1, 0.5],
    "price_control": [0.45, 0.1, 0.45],
    "positions": [0.47, 0.1, 0.43],
}


def candle(stamp, value="100.00"):
    return [
        stamp * 1000,
        value,
        value,
        value,
        value,
        "10",
        stamp * 1000 + 59999,
        "1000",
        4,
        "5",
        "500",
        "0",
    ]


def events(snapshot):
    result = []

    def receipt(source, symbol, body, at):
        times = [
            r[0] // 1000 if source == "spot_klines" else r["timestamp"] // 1000
            for r in body
        ]
        result.append(
            (
                "receipt",
                at,
                {
                    "source": source,
                    "symbol": symbol,
                    "body": body,
                    "body_sha256": sha(body),
                    "content_times": times,
                    "requested_at": at - 0.5,
                    "request_params": {
                        "symbol": symbol,
                        "startTime": START * 1000,
                        "endTime": int(at) * 1000 - 1,
                    },
                },
            )
        )
        return len(result)

    for symbol in SYMBOLS:
        receipt(
            "spot_klines",
            symbol,
            [candle(t) for t in range(ANCHOR - 3780, ANCHOR, 60)],
            ANCHOR + 1,
        )
        for source, field, value in (
            ("open_interest", "sumOpenInterest", "1000"),
            ("position_ratio", "longShortRatio", "1.2"),
        ):
            body = [
                {"symbol": symbol, "timestamp": t * 1000, field: value}
                for t in range(ANCHOR - 4500, ANCHOR - 899, 300)
            ]
            receipt(source, symbol, body, ANCHOR + 2)
    for symbol in SYMBOLS:
        result.append(
            (
                "forecast",
                ANCHOR + 10,
                {
                    "anchor": ANCHOR,
                    "symbol": symbol,
                    "issued_at": ANCHOR + 9,
                    "snapshot_sha256": snapshot,
                    "models": copy.deepcopy(MODELS),
                    "diagnostics": {"correction_available": True},
                    "input_ids": list(range(1, 10)),
                },
            )
        )
        result.append(
            (
                "forecast_commit",
                ANCHOR + 11,
                {
                    "anchor": ANCHOR,
                    "symbol": symbol,
                    "forecast_seq": len(result),
                    "forecast_hash": None,
                    "committed_at": ANCHOR + 10.5,
                    "clock_healthy": True,
                    "eligible": True,
                },
            )
        )
    for symbol, exit_price, label in zip(
        SYMBOLS, ("101.00", "99.00", "100.00"), (2, 0, 1)
    ):
        identifier = receipt(
            "spot_klines",
            symbol,
            [candle(ANCHOR), candle(ANCHOR + 300, exit_price)],
            ANCHOR + 361,
        )
        result.append(
            (
                "outcome",
                ANCHOR + 362,
                {
                    "anchor": ANCHOR,
                    "symbol": symbol,
                    "return_start": ANCHOR + 60,
                    "return_end": ANCHOR + 360,
                    "entry_price": "100.00",
                    "exit_price": exit_price,
                    "label": label,
                    "return": math.log(float(exit_price) / 100),
                    "entry_receipt_id": identifier,
                    "exit_receipt_id": identifier,
                    "input_ids": [identifier],
                },
            )
        )
    return result


def fixture(tmp_path, change=None):
    root = tmp_path / "run"
    root.mkdir()
    (root / "model.json").write_text('{"frozen":"test-only"}\n')
    snapshot = hashlib.sha256((root / "model.json").read_bytes()).hexdigest()
    config = {
        "start_at": START,
        "end_at": START + 7 * 86400,
        "snapshot_sha256": snapshot,
        "require_commit_ack": True,
        "issuance_deadline_seconds": 30,
    }
    (root / "config.json").write_text(json.dumps(config))
    data = events(snapshot)
    if change:
        change(data)
    with sqlite3.connect(root / "events.sqlite") as connection:
        connection.execute("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT)")
        connection.execute(
            "CREATE TABLE events(seq INTEGER PRIMARY KEY,kind TEXT,received_at REAL,received_ns INTEGER,anchor INTEGER,symbol TEXT,payload TEXT,prev_hash TEXT,digest TEXT)"
        )
        connection.executemany(
            "INSERT INTO meta VALUES (?,?)",
            [("config", canonical(config)), ("config_sha256", sha(config))],
        )
        previous, digests = sha(config), {}
        for seq, (kind, at, p) in enumerate(data, 1):
            p = copy.deepcopy(p)
            if kind == "forecast_commit" and p["forecast_hash"] is None:
                p["forecast_hash"] = digests[p["forecast_seq"]]
            ns = round(at * 1e9)
            envelope = dict(
                seq=seq, kind=kind, received_ns=ns, payload=p, prev_hash=previous
            )
            digest = sha(envelope)
            connection.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    seq,
                    kind,
                    at,
                    ns,
                    p.get("anchor"),
                    p.get("symbol"),
                    canonical(p),
                    previous,
                    digest,
                ),
            )
            previous = digest
            digests[seq] = digest
    return root


def test_exact_first_three_asset_cycle(tmp_path):
    root = fixture(tmp_path)
    result = verify(root, now=ANCHOR + 370)
    assert result["status"] == "verified"
    assert result["first_cycle"]["anchor"] == ANCHOR
    for symbol in SYMBOLS:
        item = result["first_cycle"]["symbols"][symbol]
        assert item["seconds_before_target_entry"] == 49.5
        assert item["input_last_received_at"] <= item["forecast_issued_at"]
        assert item["commit_observed_at"] < result["first_cycle"]["return_start"]
        assert item["outcome_received_at"] >= result["first_cycle"]["return_end"]
    btc = result["first_cycle"]["symbols"]["BTCUSDT"]["models"]["baseline"]["metrics"]
    assert btc["brier"] == pytest.approx(0.42)
    assert btc["log_loss"] == pytest.approx(math.log(2))
    assert btc["direction_brier"] == pytest.approx((4 / 9) ** 2)
    assert btc["direction_accuracy"] == 1
    sol = result["first_cycle"]["symbols"]["SOLUSDT"]["models"]["baseline"]["metrics"]
    assert sol["brier"] == pytest.approx(1.22)
    assert "direction_accuracy" not in sol
    assert not result["statistical_superiority_verified"]


def test_missing_last_outcome_remains_pending(tmp_path):
    root = fixture(tmp_path, lambda data: data.pop())
    result = verify(root, now=ANCHOR + 370)
    assert result["status"] == "pending"
    assert result["outcome_count"] == 2
    assert "first_cycle" not in result


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda d: d[10][2].update(forecast_seq=1), "acknowledgement forecast"),
        (lambda d: d[10][2].update(committed_at=ANCHOR + 61), "acknowledgement before"),
        (lambda d: d[9][2].update(issued_at=ANCHOR + 61), "late or duplicate"),
        (lambda d: d[16][2].update(label=0), "incorrect exact"),
        (lambda d: d[16][2].update(exit_price="102.00"), "cited raw price"),
        (lambda d: d[16][2].update(entry_receipt_id=1), "outcome source"),
        (lambda d: d[9][2].update(input_ids=[1, 2, 3]), "input receipt versions"),
        (
            lambda d: d[0][2]["request_params"].update(startTime=(START - 60) * 1000),
            "prospective boundary",
        ),
        (lambda d: d[9][2]["models"].update(baseline=[0.1, 0.1, 0.1]), "sum to one"),
    ],
)
def test_forged_but_rehashed_evidence_rejected(tmp_path, change, match):
    root = fixture(tmp_path, change)
    with pytest.raises(ValueError, match=match):
        verify(root, now=ANCHOR + 370)


def test_ties_predict_up_and_flats_do_not_receive_direction_scores():
    assert arithmetic([0.45, 0.1, 0.45], 2)["direction_accuracy"] == 1
    assert arithmetic([0.45, 0.1, 0.45], 0)["direction_accuracy"] == 0
    assert set(arithmetic([0.45, 0.1, 0.45], 1)) == {"brier", "log_loss"}


def test_general_audit_prefix_scores_and_identity(tmp_path):
    root = fixture(tmp_path)
    checked = verify(root, now=ANCHOR + 370)
    general = {
        "verified": True,
        "event_count": checked["event_count"],
        "chain_tip": checked["chain_tip"],
        "protocol_sha256": checked["protocol_sha256"],
        "snapshot_sha256": checked["snapshot_sha256"],
        "metrics": checked["aggregate_metrics"],
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(general))
    assert verify(root, general_audit=path, now=ANCHOR + 370)["general_audit_compared"]
    general["metrics"]["BTCUSDT"]["baseline"]["brier"]["mean"] += 0.01
    path.write_text(json.dumps(general))
    with pytest.raises(ValueError, match="metric arithmetic"):
        verify(root, general_audit=path, now=ANCHOR + 370)


def test_missing_ack_never_gets_an_outcome_score(tmp_path):
    def mutate(data):
        kind, at, p = data[10]
        data[10] = ("ack_missing_diagnostic", at, p)

    root = fixture(tmp_path, mutate)
    with pytest.raises(ValueError, match="eligible prior forecast"):
        verify(root, now=ANCHOR + 370)


def pre_target_capture(root, path):
    config = json.loads((root / "config.json").read_text())
    proof = {
        "anchor": ANCHOR,
        "captured_at": ANCHOR + 20,
        "protocol_sha256": sha(config),
        "snapshot_sha256": config["snapshot_sha256"],
        "forecasts": {},
        "commit_acks": {},
        "outcome_count": 0,
    }
    with sqlite3.connect(root / "events.sqlite") as connection:
        connection.row_factory = sqlite3.Row
        for event in connection.execute(
            "SELECT * FROM events WHERE seq<=15 ORDER BY seq"
        ):
            p = json.loads(event["payload"])
            if event["kind"] in ("forecast", "forecast_commit"):
                field = "forecasts" if event["kind"] == "forecast" else "commit_acks"
                proof[field][event["symbol"]] = {
                    "seq": event["seq"],
                    "digest": event["digest"],
                    "payload": p,
                }
            proof["ledger_head"] = {"seq": event["seq"], "digest": event["digest"]}
    path.write_text(json.dumps(proof))
    return proof


def test_read_only_backup_and_pre_target_capture(tmp_path):
    root = fixture(tmp_path)
    path = tmp_path / "pre-target.json"
    pre_target_capture(root, path)
    backup = tmp_path / "consistent.sqlite"
    with (
        sqlite3.connect(root / "events.sqlite") as source,
        sqlite3.connect(backup) as destination,
    ):
        source.backup(destination)
    result = verify(root, database=backup, pre_target_proof=path, now=ANCHOR + 370)
    assert result["status"] == "verified"
    assert result["pre_target_proof_verified"]
    assert result["database"] == str(backup)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda p: p.update(captured_at=ANCHOR + 61), "before target entry"),
        (
            lambda p: p["forecasts"]["BTCUSDT"]["payload"]["models"][
                "baseline"
            ].__setitem__(0, 0.3),
            "changed after capture",
        ),
        (lambda p: p.update(outcome_count=1), "already contained"),
    ],
)
def test_pre_target_capture_cannot_be_backfilled_or_changed(tmp_path, change, match):
    root = fixture(tmp_path)
    path = tmp_path / "pre-target.json"
    proof = pre_target_capture(root, path)
    change(proof)
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match=match):
        verify(root, pre_target_proof=path, now=ANCHOR + 370)
