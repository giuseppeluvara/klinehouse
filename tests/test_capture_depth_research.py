"""Offline tests for isolated public depth acquisition and causal evidence."""

import hashlib
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from scripts import capture_depth_research as depth


START = 1800000000


class Clock:
    def __init__(self, value=START + 0.1):
        self.value = value
        self.mono = 1000.0
        self.lock = threading.Lock()

    def time(self):
        return self.value

    def monotonic(self):
        return self.mono

    def advance(self, seconds):
        self.value += seconds
        self.mono += seconds


def book():
    return {
        "lastUpdateId": 25,
        "bids": [[str(100 - i), "3"] for i in range(20)],
        "asks": [[str(101 + i), "1"] for i in range(20)],
    }


def initialize(tmp_path, clock):
    root = tmp_path / "new-depth"
    config = depth.initialize(root, end_at=START + 3600, clock=clock.time)
    return root, config


def fetcher(clock, *, late=False, bad_clock=False):
    calls = []

    def request(source, symbol=None, **kwargs):
        with clock.lock:
            calls.append((source, symbol))
            start, mono = clock.time(), clock.monotonic()
            clock.advance(11 if source == "depth" and late else 0.05)
            data = (
                {
                    "serverTime": round(
                        (start + 30 if bad_clock else start + 0.025) * 1000
                    )
                }
                if source == "time"
                else book()
            )
            raw = json.dumps(data)
            return {
                "source": source,
                "symbol": symbol,
                "requested_at": start,
                "requested_monotonic": mono,
                "received_at": clock.time(),
                "received_monotonic": clock.monotonic(),
                "body": data,
                "raw_body": raw,
                "body_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "http_date": "synthetic",
                "http_status": 200,
            }

    request.calls = calls
    return request


def test_public_request_allowlist_has_no_generic_url_or_account_scope():
    assert depth.request_spec("depth", "BTCUSDT") == (
        "https://api.binance.com/api/v3/depth",
        {"symbol": "BTCUSDT", "limit": 20},
    )
    assert depth.request_spec("time") == ("https://api.binance.com/api/v3/time", {})
    for source, symbol in [
        ("order", "BTCUSDT"),
        ("depth", "BNBUSDT"),
        ("https://evil.test", None),
        ("time", "BTCUSDT"),
    ]:
        with pytest.raises(ValueError, match="public depth/time"):
            depth.request_spec(source, symbol)


def test_http_client_disables_credentials_proxies_and_redirects(monkeypatch):
    raw = json.dumps(book()).encode()
    seen = []

    class Response:
        status_code = 200
        headers = {"Date": "synthetic", "X-MBX-USED-WEIGHT-1M": "5"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, chunk_size):
            yield raw

    class Session:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            assert self.trust_env is False
            assert set(kwargs) == {"params", "timeout", "allow_redirects", "stream"}
            assert kwargs["timeout"] == (2, 5)
            assert kwargs["allow_redirects"] is False
            seen.append(url)
            return Response()

    monkeypatch.setattr(depth.requests, "Session", Session)
    receipt = depth.fetch("depth", "ETHUSDT", clock=lambda: START, monotonic=lambda: 5)
    assert receipt["body"] == book()
    assert receipt["raw_body"].encode() == raw
    assert receipt["body_sha256"] == hashlib.sha256(raw).hexdigest()
    assert seen == [depth.URLS["depth"]]


def test_exact_decimal_book_features_and_same_body_semantics():
    features = depth.book_features(book())
    assert features["qi1"] == features["qi20"] == "0.5"
    assert features["mid_price"] == "100.5"
    assert features["spread_price"] == "1"
    with depth.localcontext() as context:
        context.prec = 50
        assert depth.Decimal(features["spread_bps"]) == depth.Decimal(
            10000
        ) / depth.Decimal("100.5")
    assert "microprice" not in features


@pytest.mark.parametrize(
    "change",
    [
        lambda b: b["bids"].pop(),
        lambda b: b["bids"][0].__setitem__(1, "0"),
        lambda b: b["asks"][0].__setitem__(0, "100"),
        lambda b: b["bids"][1].__setitem__(0, "100"),
        lambda b: b["asks"][1].__setitem__(0, "100"),
        lambda b: b["asks"][0].__setitem__(1, "NaN"),
        lambda b: b["asks"][0].__setitem__(1, "Infinity"),
        lambda b: b["bids"][0].__setitem__(0, 100),
        lambda b: b.__setitem__("lastUpdateId", True),
        lambda b: b.__setitem__("futureField", 1),
    ],
)
def test_invalid_duplicate_crossed_and_changed_books_are_rejected(change):
    data = book()
    change(data)
    with pytest.raises(ValueError):
        depth.book_features(data)


def test_clock_bound_includes_round_trip_and_monotonic_clock():
    receipt = {
        "requested_at": 100,
        "received_at": 102.1,
        "requested_monotonic": 50,
        "received_monotonic": 52.1,
        "body": {"serverTime": 100000},
    }
    assert depth.clock_evidence(receipt)["healthy"] is False
    receipt["received_at"] = 101
    receipt["received_monotonic"] = 51
    assert depth.clock_evidence(receipt)["healthy"] is True
    receipt["received_at"] = 100.1
    assert depth.clock_evidence(receipt)["healthy"] is False


def test_new_root_frozen_configuration_and_pilot_path_guard(tmp_path):
    clock = Clock()
    root, _ = initialize(tmp_path, clock)
    with pytest.raises(FileExistsError):
        depth.initialize(root, end_at=START + 3600, clock=clock.time)
    assert depth.load_config(root)["candidate_predictions"] is False
    config = json.loads((root / "config.json").read_text())
    config["limits"]["levels"] = 10
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="frozen"):
        depth.load_config(root)
    with pytest.raises(ValueError, match="shadow pilot"):
        depth.root_path(tmp_path / "shadow-pilot-user" / "new")


def test_utc_end_requires_explicit_utc_and_prospective_bounds(tmp_path):
    assert depth.parse_utc("2026-11-05T11:10Z") == 1793877000
    for stamp in ("2026-11-05T11:10", "2026-11-05T11:10+01:00"):
        with pytest.raises(ValueError, match="UTC"):
            depth.parse_utc(stamp)
    with pytest.raises(ValueError, match="eight days"):
        depth.initialize(
            tmp_path / "bad", end_at=START + 9 * 86400, clock=lambda: START
        )


def test_full_three_book_cycle_durable_before_anchor_and_immutable_evidence(tmp_path):
    clock = Clock()
    root, config = initialize(tmp_path, clock)
    clock.advance(289.9)
    request = fetcher(clock)
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        records = depth.capture_cycle(
            journal,
            START + 300,
            fetcher=request,
            clock=clock.time,
            monotonic=clock.monotonic,
        )
        assert len(records) == 3
        assert all(
            r["received_at"] < START + 300 and r["committed_at"] < START + 300
            for r in records
        )
        assert journal.completed(START + 300) == set(depth.SYMBOLS)
        assert journal.verify()["events"] == 10
        assert len(request.calls) == 4
        assert (
            depth.capture_cycle(
                journal,
                START + 300,
                fetcher=request,
                clock=clock.time,
                monotonic=clock.monotonic,
            )
            == []
        )
        assert len(request.calls) == 4
        for sql in (
            "UPDATE events SET payload='{}'",
            "DELETE FROM events",
            "DELETE FROM meta",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                journal.connection.execute(sql)
    proof = json.loads((root / "first-cycle.json").read_text())
    assert proof["all_body_receipts_before_anchor"]
    assert proof["all_primary_commits_before_anchor"]
    assert proof["all_ack_commits_before_anchor"]
    assert (
        json.loads((root / "first-cycle-write-receipt.json").read_text())[
            "durable_write_observed_at"
        ]
        < START + 300
    )


def test_late_full_body_is_preserved_but_explicitly_missing(tmp_path):
    clock = Clock()
    root, config = initialize(tmp_path, clock)
    clock.advance(289.9)
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        assert (
            depth.capture_cycle(
                journal,
                START + 300,
                fetcher=fetcher(clock, late=True),
                clock=clock.time,
                monotonic=clock.monotonic,
            )
            == []
        )
        assert journal.completed(START + 300) == set(depth.SYMBOLS)
        rows = list(journal.connection.execute("SELECT kind,payload FROM events"))
        assert sum(r["kind"] == "book_receipt" for r in rows) == 3
        assert sum(r["kind"] == "missing" for r in rows) == 3
        assert not (root / "first-cycle.json").exists()


def test_unhealthy_clock_does_not_attempt_books(tmp_path):
    clock = Clock()
    root, config = initialize(tmp_path, clock)
    clock.advance(289.9)
    request = fetcher(clock, bad_clock=True)
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        assert (
            depth.capture_cycle(
                journal,
                START + 300,
                fetcher=request,
                clock=clock.time,
                monotonic=clock.monotonic,
            )
            == []
        )
        assert request.calls == [("time", None)]
        assert journal.completed(START + 300) == set(depth.SYMBOLS)


def test_slow_full_commit_cannot_receive_a_backdated_eligible_ack(tmp_path):
    clock = Clock()
    root, config = initialize(tmp_path, clock)
    clock.advance(294.9)
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        commit = journal._commit

        def slow_commit():
            commit()
            clock.advance(40)

        journal._commit = slow_commit
        record = journal.append(
            "book_receipt",
            {"synthetic": True},
            anchor=START + 300,
            symbol="BTCUSDT",
            eligible=True,
            clock_healthy=True,
        )
        assert record["eligible"] is False
        ack = json.loads(
            journal.connection.execute(
                "SELECT payload FROM events WHERE kind='commit_ack' ORDER BY seq DESC LIMIT 1"
            ).fetchone()[0]
        )
        assert ack["committed_at"] == START + 335
        assert ack["durable_before_anchor"] is False
        journal.verify()


def test_crash_after_primary_commit_is_not_retroactively_acknowledged(tmp_path):
    clock = Clock()
    root, config = initialize(tmp_path, clock)
    clock.advance(289.9)
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        commit = journal._commit

        def crash():
            commit()
            raise RuntimeError("simulated power loss after commit")

        journal._commit = crash
        with pytest.raises(RuntimeError):
            journal.append(
                "book_receipt",
                {"synthetic": True},
                anchor=START + 300,
                symbol="BTCUSDT",
                eligible=True,
                clock_healthy=True,
            )
    clock.advance(20)
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        assert journal.completed(START + 300) == set()
        depth.capture_cycle(
            journal,
            START + 300,
            clock=clock.time,
            monotonic=clock.monotonic,
            fetcher=lambda *a, **kw: pytest.fail("no historical book backfill"),
        )
        acked = [
            json.loads(r[0])["record_seq"]
            for r in journal.connection.execute(
                "SELECT payload FROM events WHERE kind='commit_ack'"
            )
        ]
        assert 3 not in acked
        assert journal.completed(START + 300) == set(depth.SYMBOLS)


def test_restart_inside_capture_window_never_retries_unacknowledged_book(tmp_path):
    clock = Clock()
    root, config = initialize(tmp_path, clock)
    clock.advance(289.9)
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        commit = journal._commit

        def crash():
            commit()
            raise RuntimeError("crash after durable book")

        journal._commit = crash
        with pytest.raises(RuntimeError):
            journal.append(
                "book_receipt",
                {"synthetic": True},
                anchor=START + 300,
                symbol="BTCUSDT",
                eligible=True,
                clock_healthy=True,
            )
    request = fetcher(clock)
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        accepted = depth.capture_cycle(
            journal,
            START + 300,
            fetcher=request,
            clock=clock.time,
            monotonic=clock.monotonic,
        )
        assert {row["symbol"] for row in accepted} == {"ETHUSDT", "SOLUSDT"}
        assert ("depth", "BTCUSDT") not in request.calls
        assert journal.completed(START + 300) == set(depth.SYMBOLS)
        missing = list(
            journal.connection.execute(
                "SELECT symbol,payload FROM events WHERE kind='missing'"
            )
        )
        assert len(missing) == 1 and missing[0]["symbol"] == "BTCUSDT"
        assert "unacknowledged" in missing[0]["payload"]
        journal.verify()


@pytest.mark.parametrize("jump_commit", [1, 2])
def test_wall_rollback_during_primary_or_ack_fsync_fails_closed(tmp_path, jump_commit):
    clock = Clock()
    root, config = initialize(tmp_path, clock)
    clock.advance(289.9)
    reference = [clock.time(), clock.monotonic()]
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        commit, calls = journal._commit, []

        def jump():
            commit()
            calls.append(1)
            if len(calls) == jump_commit:
                clock.value -= 1
                clock.mono += 40

        journal._commit = jump
        record = journal.append(
            "book_receipt",
            {"synthetic": True},
            anchor=START + 300,
            symbol="BTCUSDT",
            eligible=True,
            clock_healthy=True,
            clock_reference=reference,
        )
        assert record["eligible"] is False
        assert record["clock_healthy"] is False
        assert journal.completed(START + 300) == set()
        assert journal.status("test")["accepted_books"] == 0
        journal.verify()


def test_wall_jump_before_write_is_caught_by_postcommit_probe_reference(tmp_path):
    clock = Clock()
    root, config = initialize(tmp_path, clock)
    clock.advance(289.9)
    reference = [clock.time(), clock.monotonic()]
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        write = journal._write

        def jump(kind, *args):
            if kind == "book_receipt":
                clock.value -= 1
                clock.mono += 40
            return write(kind, *args)

        journal._write = jump
        record = journal.append(
            "book_receipt",
            {"synthetic": True},
            anchor=START + 300,
            symbol="BTCUSDT",
            eligible=True,
            clock_healthy=True,
            clock_reference=reference,
        )
        assert record["commit_clock_healthy"] is True
        assert record["eligible"] is False
        assert record["clock_healthy"] is False
        journal.verify()


def test_hash_chain_detects_bypassed_sqlite_guard(tmp_path):
    clock = Clock()
    root, config = initialize(tmp_path, clock)
    with depth.Journal(root, config, clock=clock.time) as journal:
        journal.connection.execute("DROP TRIGGER events_no_update")
        journal.connection.execute("UPDATE events SET payload='{}' WHERE seq=1")
        journal.connection.commit()
        with pytest.raises(ValueError, match="hash chain"):
            journal.verify()


def test_resource_guards_preserve_files(tmp_path, monkeypatch):
    existing = tmp_path / "evidence"
    existing.write_text("keep")
    assert (
        depth.resource_reason(tmp_path, disk_usage=lambda p: SimpleNamespace(free=0))
        == "free_space_below_40_GiB"
    )
    monkeypatch.setattr(depth, "directory_bytes", lambda p: 32 * 1024**2)
    assert (
        depth.resource_reason(
            tmp_path, disk_usage=lambda p: SimpleNamespace(free=50 * 1024**3)
        )
        == "32_MiB_budget_reserve_reached"
    )
    assert existing.read_text() == "keep"


def test_once_run_records_missed_anchor_without_backfill(tmp_path):
    clock = Clock()
    root, _ = initialize(tmp_path, clock)
    clock.advance(301)
    assert (
        depth.run(
            root,
            once=True,
            clock=clock.time,
            monotonic=clock.monotonic,
            fetcher=lambda *a, **kw: pytest.fail("missed books cannot be recovered"),
        )
        == 0
    )
    status = json.loads((root / "status.json").read_text())
    assert status["counts"]["missing"] == 3
    assert status["accepted_books"] == 0


def test_sleep_is_bounded_and_once_does_not_change_five_minute_grid(tmp_path):
    clock = Clock()
    root, _ = initialize(tmp_path, clock)
    pauses = []

    def sleep(seconds):
        assert 0 < seconds <= 30
        pauses.append(seconds)
        clock.advance(seconds)

    assert (
        depth.run(
            root,
            once=True,
            clock=clock.time,
            monotonic=clock.monotonic,
            sleep=sleep,
            fetcher=fetcher(clock),
        )
        == 0
    )
    assert len(pauses) >= 9
    assert json.loads((root / "first-cycle.json").read_text())["anchor"] == START + 300


def test_status_fsync_crossing_capture_window_never_requests_negative_sleep(
    tmp_path, monkeypatch
):
    clock = Clock()
    root, _ = initialize(tmp_path, clock)
    clock.advance(289)
    original_status = depth.Journal.status
    pauses = []

    def slow_status(self, phase, **kwargs):
        result = original_status(self, phase, **kwargs)
        if phase == "waiting":
            clock.advance(2)
        return result

    def sleep(seconds):
        assert 0 <= seconds <= 30
        pauses.append(seconds)
        clock.advance(seconds)

    monkeypatch.setattr(depth.Journal, "status", slow_status)
    assert (
        depth.run(
            root,
            once=True,
            clock=clock.time,
            monotonic=clock.monotonic,
            sleep=sleep,
            fetcher=fetcher(clock),
        )
        == 0
    )
    assert pauses == [0]
