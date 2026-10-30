"""Offline tests for isolated public depth acquisition and causal evidence."""

import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from types import ModuleType, SimpleNamespace

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
    monkeypatch.setattr(depth, "directory_bytes", lambda p: 60 * 1024**2)
    assert (
        depth.resource_reason(
            tmp_path, disk_usage=lambda p: SimpleNamespace(free=50 * 1024**3)
        )
        == "64_MiB_budget_reserve_reached"
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


@pytest.fixture
def legacy_capture(tmp_path):
    """Authentic old code produces the exact approved config in an isolated tmp."""
    path = (
        Path(__file__).resolve().parents[1]
        / "reports/depth-information-20261030/frozen-source/capture_depth_research_v1.py"
    )
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == depth.LEGACY_CODE_SHA256
    legacy = ModuleType("authentic_legacy_depth_fixture")
    legacy.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), legacy.__dict__)
    clock = Clock(1793374413.458949)
    root = tmp_path / "depth-research-20261030"
    config = legacy.initialize(root, end_at=1793877000.0, clock=clock.time)
    assert depth.sha(config) == depth.LEGACY_CONFIG_SHA256
    (root / "runtime").mkdir()
    (root / "runtime/capture_depth_research.py").write_bytes(raw)
    first = 1793374500
    clock.advance(first - 10 - clock.time())
    request = fetcher(clock)

    def complete_receipt(source, symbol=None, **kwargs):
        receipt = request(source, symbol, **kwargs)
        receipt.update(
            url=depth.URLS[source],
            params={} if source == "time" else {"symbol": symbol, "limit": 20},
            body_bytes=len(receipt["raw_body"].encode()),
        )
        return receipt

    with legacy.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        legacy.capture_cycle(
            journal,
            first,
            fetcher=complete_receipt,
            clock=clock.time,
            monotonic=clock.monotonic,
        )
    return root, config, clock, legacy


def prepare_legacy(fixture, monkeypatch):
    root, config, clock, _ = fixture
    document = depth.prepare_budget_amendment(
        root, clock=clock.time, monotonic=clock.monotonic
    )
    runtime = root / document["amended_runtime_path"]
    monkeypatch.setattr(depth, "__file__", str(runtime))
    return root, config, clock, document


def test_budget_amendment_preparation_keeps_legacy_bytes_and_journal(legacy_capture):
    root, config, clock, legacy = legacy_capture
    protected = [
        root / "config.json",
        root / "config-sha256.json",
        root / "runtime/capture_depth_research.py",
        root / "depth-events.sqlite",
    ]
    before = {path: path.read_bytes() for path in protected}
    # The old process can retain its writer lock: preparation only reads its DB.
    with legacy.writer_lock(root):
        document = depth.prepare_budget_amendment(
            root, clock=clock.time, monotonic=clock.monotonic
        )
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert document["old_limits"] == config["limits"]
    assert document["new_limits"] == {
        **config["limits"],
        "max_storage_bytes": 64 * 1024**2,
    }
    assert document["first_anchor"] == 1793374500
    assert document["training_cutoff"] == 1793460900
    assert document["registered_at"] < document["training_cutoff"]
    assert document["data_selection_and_model_changes"] is False
    assert (
        depth.file_sha256(root / document["amended_runtime_path"])
        == depth.code_sha256()
    )
    with pytest.raises(FileExistsError):
        depth.prepare_budget_amendment(
            root, clock=clock.time, monotonic=clock.monotonic
        )


def test_legacy_requires_amendment_and_exact_versioned_runtime(
    legacy_capture, monkeypatch
):
    root, config, clock, _ = legacy_capture
    with pytest.raises(ValueError, match="requires its budget amendment"):
        depth.load_config(root)
    document = depth.prepare_budget_amendment(
        root, clock=clock.time, monotonic=clock.monotonic
    )
    with pytest.raises(ValueError, match="runtime bytes/path"):
        depth.load_config(root)
    monkeypatch.setattr(depth, "__file__", str(root / document["amended_runtime_path"]))
    assert depth.load_config(root) == config


def test_amendment_appends_once_then_restart_reuses_original_seed(
    legacy_capture, monkeypatch
):
    root, config, clock, document = prepare_legacy(legacy_capture, monkeypatch)
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        before = [
            tuple(row) for row in journal.connection.execute("SELECT * FROM events")
        ]
        with depth.writer_lock(root):
            assert depth.activate_budget_amendment(journal) == document
        assert journal.verify()["events"] == len(before) + 2
        assert journal.effective_limits == depth.LIMITS
        assert (
            journal.status("test")["budget_amendment_sha256"]
            == document["amendment_sha256"]
        )
        assert [
            tuple(row)
            for row in journal.connection.execute(
                "SELECT * FROM events WHERE seq<=?", (len(before),)
            )
        ] == before
        after = journal.verify()
    # A valid prior activation remains valid after the cutoff; no second event/ack.
    clock.advance(86400)
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        depth.activate_budget_amendment(journal)
        assert journal.verify() == after
        assert journal.config["limits"]["max_storage_bytes"] == 32 * 1024**2
        assert journal.effective_limits["max_storage_bytes"] == 64 * 1024**2


def test_new_producer_marker_follows_resource_amendment(legacy_capture, monkeypatch):
    root, _, clock, document = prepare_legacy(legacy_capture, monkeypatch)
    clock.advance(310)
    assert (
        depth.run(
            root,
            once=True,
            clock=clock.time,
            monotonic=clock.monotonic,
            fetcher=lambda *a, **kw: pytest.fail(
                "missed anchor must not make a request"
            ),
        )
        == 0
    )
    with sqlite3.connect(root / "depth-events.sqlite") as connection:
        event = connection.execute(
            "SELECT seq FROM events WHERE kind='resource_budget_amended'"
        ).fetchone()[0]
        marker_seq, raw = connection.execute(
            "SELECT seq,payload FROM events WHERE kind='process_started' ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        marker = json.loads(raw)
        assert event < marker_seq
        assert marker["producer_code_sha256"] == document["amended_code_sha256"]
        assert marker["amendment_sha256"] == document["amendment_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.__setitem__("root", "/somewhere/else"),
        lambda d: d.__setitem__("original_code_sha256", "0" * 64),
        lambda d: d.__setitem__("config_sha256", "0" * 64),
        lambda d: d.__setitem__("reason", "different scope"),
        lambda d: d.__setitem__("amended_runtime_path", "../escape.py"),
        lambda d: d["new_limits"].__setitem__("levels", 10),
        lambda d: d["new_limits"].__setitem__("min_free_bytes", 0),
        lambda d: d["old_limits"].__setitem__("max_storage_bytes", 64 * 1024**2),
        lambda d: d.__setitem__("registered_at", d["training_cutoff"]),
        lambda d: d.__setitem__("training_cutoff", d["training_cutoff"] + 300),
        lambda d: d.__setitem__("end_at", d["end_at"] + 300),
        lambda d: d.__setitem__("symbols", ["BTCUSDT"]),
        lambda d: d.__setitem__("data_selection_and_model_changes", True),
    ],
)
def test_resigned_amendment_cannot_change_scope_or_identity(
    legacy_capture, monkeypatch, change
):
    root, config, _, document = prepare_legacy(legacy_capture, monkeypatch)
    change(document)
    body = {k: v for k, v in document.items() if k != "amendment_sha256"}
    document["amendment_sha256"] = depth.sha(body)
    (root / "budget-amendment.json").write_text(json.dumps(document))
    with pytest.raises(ValueError):
        depth.load_budget_amendment(root, config, require_runtime=True)


@pytest.mark.parametrize("target", ["manifest", "original_runtime", "amended_runtime"])
def test_amendment_rejects_changed_bytes(legacy_capture, monkeypatch, target):
    root, config, _, document = prepare_legacy(legacy_capture, monkeypatch)
    if target == "manifest":
        document["amendment_sha256"] = "0" * 64
        (root / "budget-amendment.json").write_text(json.dumps(document))
    else:
        name = (
            "original_runtime_path"
            if target == "original_runtime"
            else "amended_runtime_path"
        )
        with (root / document[name]).open("ab") as stream:
            stream.write(b"\n# changed bytes\n")
    with pytest.raises(ValueError):
        depth.load_budget_amendment(root, config, require_runtime=True)


def test_amendment_preparation_and_first_activation_refuse_after_cutoff(
    legacy_capture, monkeypatch
):
    root, config, clock, document = prepare_legacy(legacy_capture, monkeypatch)
    clock.advance(document["training_cutoff"] - clock.time())
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        before = journal.verify()
        with pytest.raises(ValueError, match="activation must precede cutoff"):
            depth.activate_budget_amendment(journal)
        assert journal.verify() == before
    # Another fixture root with genuine bytes but no prior manifest also refuses.
    (root / "budget-amendment.json").rename(root / "preserved-old-amendment.json")
    with pytest.raises(ValueError, match="before cutoff"):
        depth.prepare_budget_amendment(
            root, clock=clock.time, monotonic=clock.monotonic
        )
    with pytest.raises(ValueError, match="shadow pilot"):
        depth.prepare_budget_amendment(root.parent / "shadow-pilot-20261029-v2")


def test_amendment_clock_jump_during_ack_invalidates_and_never_recovers(
    legacy_capture, monkeypatch
):
    root, config, clock, _ = prepare_legacy(legacy_capture, monkeypatch)
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        commit = journal._commit
        calls = []

        def jump_during_ack():
            commit()
            calls.append(1)
            if len(calls) == 2:
                clock.value -= 1
                clock.mono += 40

        journal._commit = jump_during_ack
        with pytest.raises(ValueError, match="confirmation failed"):
            depth.activate_budget_amendment(journal)
        assert (
            journal.connection.execute(
                "SELECT COUNT(*) FROM events WHERE kind='invalidated'"
            ).fetchone()[0]
            == 1
        )
    with depth.Journal(
        root, config, clock=clock.time, monotonic=clock.monotonic
    ) as journal:
        with pytest.raises(ValueError, match="event/confirmation differs"):
            depth.activate_budget_amendment(journal)
        assert (
            journal.connection.execute(
                "SELECT COUNT(*) FROM events WHERE kind='resource_budget_amended'"
            ).fetchone()[0]
            == 1
        )


def test_amendment_publication_crash_is_private_and_retryable(
    legacy_capture, monkeypatch
):
    root, _, clock, _ = legacy_capture
    link = depth.os.link

    def crash_manifest_publish(source, target):
        if Path(target).name == "budget-amendment.json":
            raise RuntimeError("synthetic crash before publication")
        return link(source, target)

    monkeypatch.setattr(depth.os, "link", crash_manifest_publish)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        depth.prepare_budget_amendment(
            root, clock=clock.time, monotonic=clock.monotonic
        )
    assert not (root / "budget-amendment.json").exists()
    assert list(root.glob(".budget-amendment.json.pending-*"))
    monkeypatch.setattr(depth.os, "link", link)
    document = depth.prepare_budget_amendment(
        root, clock=clock.time, monotonic=clock.monotonic
    )
    assert document["config_sha256"] == depth.LEGACY_CONFIG_SHA256


def test_new_budget_preserves_40_gib_free_guard_and_four_mib_reserve(
    legacy_capture, monkeypatch
):
    root, _, clock, _ = legacy_capture
    monkeypatch.setattr(
        depth.shutil, "disk_usage", lambda p: SimpleNamespace(free=39 * 1024**3)
    )
    with pytest.raises(ValueError, match="insufficient free space"):
        depth.prepare_budget_amendment(
            root, clock=clock.time, monotonic=clock.monotonic
        )
    assert (
        depth.resource_reason(
            root, disk_usage=lambda p: SimpleNamespace(free=39 * 1024**3)
        )
        == "free_space_below_40_GiB"
    )
    monkeypatch.setattr(depth, "directory_bytes", lambda p: 60 * 1024**2 - 1)
    assert (
        depth.resource_reason(
            root, disk_usage=lambda p: SimpleNamespace(free=50 * 1024**3)
        )
        is None
    )
    monkeypatch.setattr(depth, "directory_bytes", lambda p: 60 * 1024**2)
    assert (
        depth.resource_reason(
            root, disk_usage=lambda p: SimpleNamespace(free=50 * 1024**3)
        )
        == "64_MiB_budget_reserve_reached"
    )
