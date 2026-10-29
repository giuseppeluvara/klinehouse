"""Prospective runner tests: injected clock and in-memory public responses only."""

import copy
import json
import math
from threading import Barrier

import pytest

from research import shadow_model as model
from research import shadow_sources as sources
from scripts import run_shadow as runner
from scripts.verify_shadow_run import verify


START = 1800000000


def small_snapshot():
    return {
        "schema": model.SCHEMA,
        "cutoff": model.FROZEN_CUTOFF,
        "symbols": list(model.SYMBOLS),
        "metrics_lag_seconds": 900,
        "baseline_beta": 0.5,
        "flat_model": {
            "cutoff": model.FROZEN_CUTOFF,
            "mean": [0.0] * 23,
            "scale": [1.0] * 23,
            "coef": [[0.0] * 23 for _ in range(3)],
            "intercept": [0.0, -3.0, 0.0],
        },
        "frequency": {s: [0.45, 0.1, 0.45] for s in model.SYMBOLS},
        "corrections": {
            arm: {
                "cutoff": model.FROZEN_CUTOFF,
                "coefficients": [0.1, -0.1],
                "scale": [1.0, 1.0],
                "baseline_beta": 0.5,
            }
            for arm in ("price_control", "positions")
        },
    }


def candle(stamp):
    minute = (stamp - START) // 60
    close = f"{100 + minute * 0.01 + math.sin(minute) * 0.001:.8f}"
    return [
        stamp * 1000,
        close,
        close,
        close,
        close,
        "10",
        stamp * 1000 + 59999,
        "1000",
        100,
        "5",
        "500",
        "0",
    ]


class FakePublicSources:
    def __init__(self, clock, missing=()):
        self.clock = clock
        self.missing = set(missing)
        self.calls = []

    def __call__(self, source, symbol, start_at, end_at, *, recovery=False):
        now = self.clock[0]
        params = sources.request_parameters(
            source, symbol, start_at, end_at, now, recovery
        )
        self.calls.append((source, symbol, params))
        if params is None or (source, symbol) in self.missing:
            return None
        first = math.ceil(
            params["startTime"] / 1000 / (60 if source == "spot_klines" else 300)
        )
        step = 60 if source == "spot_klines" else 300
        labels = range(
            first * step, math.floor(params["endTime"] / 1000 / step) * step + 1, step
        )
        if source == "spot_klines":
            body = [candle(stamp) for stamp in labels]
        else:
            key = "sumOpenInterest" if source == "open_interest" else "longShortRatio"
            base = 1000 if source == "open_interest" else 1
            body = [
                dict(
                    symbol=symbol,
                    timestamp=stamp * 1000,
                    **{key: str(base + (stamp - START) / 300000)},
                )
                for stamp in labels
            ]
        times = sources.validate_body(source, body, params, start_at, end_at, now)
        return dict(
            source=source,
            symbol=symbol,
            requested_at=now,
            request_params=params,
            body=body,
            content_times=times,
            body_sha256=sources.digest(body),
            received_at=now,
            http_status=200,
        )


def make_runner(tmp_path, monkeypatch, clock, missing=(), monotonic=None):
    snapshot = tmp_path / "model-source.json"
    snapshot.write_text(json.dumps(small_snapshot()))
    # Initialization itself must happen before its registered start.
    monkeypatch.setattr(runner.time, "time", lambda: START - 60)
    root = tmp_path / "trial"
    runner.initialize(root, snapshot, start_at=START)
    fake = FakePublicSources(clock, missing)
    instance = runner.Runner(
        root,
        clock=lambda: clock[0],
        monotonic=monotonic or (lambda: clock[0]),
        fetcher=fake,
        probe=lambda: {"healthy": True, "samples": []},
    )
    return instance, fake, root


def test_full_three_asset_emission_maturation_and_restart(tmp_path, monkeypatch):
    anchor = START + 4800
    clock = [anchor + 10]
    instance, fake, root = make_runner(tmp_path, monkeypatch, clock)
    try:
        instance.check_clock()
        instance.collect()
        instance.decide()
        emissions = list(instance.ledger.rows("forecast"))
        assert len(emissions) == 3
        commits = list(instance.ledger.rows("forecast_commit"))
        assert len(commits) == 3
        for forecast, commit in zip(emissions, commits):
            ack = commit["payload"]
            assert ack["forecast_seq"] == forecast["seq"]
            assert ack["forecast_hash"] == forecast["digest"]
            assert ack["eligible"] is True
            assert ack["committed_at"] <= anchor + 30
        assert {e["payload"]["symbol"] for e in emissions} == set(model.SYMBOLS)
        assert all(
            e["payload"]["diagnostics"]["correction_available"] for e in emissions
        )
        assert all(e["payload"]["issued_at"] < anchor + 60 for e in emissions)
        assert all(call[2]["startTime"] >= START * 1000 for call in fake.calls)
        instance.decide()
        assert len(list(instance.ledger.rows("forecast"))) == 3
        instance.resolve()
        assert not list(instance.ledger.rows("outcome"))
        clock[0] = anchor + 61
        instance.collect()  # Capture the entry candle once it has closed.
        instance.resolve()
        assert not list(instance.ledger.rows("outcome"))
        clock[0] = anchor + 361
        instance.collect()
        instance.resolve()
        outcomes = list(instance.ledger.rows("outcome"))
        assert len(outcomes) == 3
        for event in outcomes:
            payload = event["payload"]
            assert payload["entry_price"] == candle(anchor)[4]
            assert payload["exit_price"] == candle(anchor + 300)[4]
            assert payload["label"] == 2
            assert payload["return_start"] == anchor + 60
            assert payload["return_end"] == anchor + 360
        audit = verify(
            root / "events.sqlite",
            snapshot=root / "model.json",
            protocol=root / "config.json",
            now=clock[0],
        )
        assert audit["verified"]
        assert audit["technical_pipeline_observed"]
        assert audit["eligible_forecast_count"] == audit["outcome_count"] == 3
        previous = instance.ledger.verify()
    finally:
        instance.ledger.close()
    restarted = runner.Runner(
        root,
        clock=lambda: clock[0],
        monotonic=lambda: clock[0],
        fetcher=fake,
        probe=lambda: {"healthy": True, "samples": []},
    )
    try:
        assert restarted.ledger.verify() == previous
        assert len(restarted.forecasts) == len(restarted.scored) == 3
        restarted.resolve()
        assert len(list(restarted.ledger.rows("outcome"))) == 3
        restarted.decide()
        # The interrupted next anchor is recorded missing, never backfilled.
        late = [
            e
            for e in restarted.ledger.rows("missing")
            if e["payload"]["anchor"] == anchor + 300
        ]
        assert len(late) == 3
        assert len(list(restarted.ledger.rows("forecast"))) == 3
    finally:
        restarted.ledger.close()


def test_missing_one_metric_disables_both_corrections_for_every_asset(
    tmp_path, monkeypatch
):
    clock = [START + 4800 + 10]
    instance, _, _ = make_runner(
        tmp_path, monkeypatch, clock, missing={("position_ratio", "SOLUSDT")}
    )
    try:
        instance.check_clock()
        instance.collect()
        instance.decide()
        emissions = list(instance.ledger.rows("forecast"))
        assert len(emissions) == 3
        for event in emissions:
            p = event["payload"]
            assert not p["diagnostics"]["correction_available"]
            assert p["diagnostics"]["fallback_assets"] == ["SOLUSDT"]
            assert len({tuple(v) for v in p["models"].values()}) == 1
    finally:
        instance.ledger.close()


def test_decision_delay_clock_failure_and_warmup_are_visible_missing(
    tmp_path, monkeypatch
):
    clock = [START + 5]
    instance, _, _ = make_runner(tmp_path, monkeypatch, clock)
    try:
        instance.collect()
        instance.decide()
        assert not instance.decided
        clock[0] = START + 10
        instance.decide()
        reasons = [e["payload"]["reason"] for e in instance.ledger.rows("missing")]
        assert reasons == ["clock_unverified_or_stale"] * 3
        clock[0] = START + 310
        instance.check_clock()
        instance.collect()
        instance.decide()
        assert not instance.forecasts
        warmup = [
            e
            for e in instance.ledger.rows("missing")
            if e["payload"]["anchor"] == START + 300
        ]
        assert len(warmup) == 3
        assert all(
            e["payload"]["reason"].startswith("incomplete_or_invalid_context")
            for e in warmup
        )
    finally:
        instance.ledger.close()


def test_late_inference_marks_missing_instead_of_emitting_after_deadline(
    tmp_path, monkeypatch
):
    anchor = START + 4800
    clock = [anchor + 10]
    instance, _, _ = make_runner(tmp_path, monkeypatch, clock)
    original = runner.predict

    def slow(*args):
        result = original(*args)
        clock[0] = anchor + 31
        return result

    monkeypatch.setattr(runner, "predict", slow)
    try:
        instance.check_clock()
        instance.collect()
        instance.decide()
        assert not instance.forecasts
        missing = [
            e
            for e in instance.ledger.rows("missing")
            if e["payload"]["anchor"] == anchor
        ]
        assert len(missing) == 3
        assert all(
            e["payload"]["reason"] == "missed_deadline_or_process_downtime"
            for e in missing
        )
    finally:
        instance.ledger.close()


@pytest.mark.parametrize("utc_delta,monotonic_delta", [(-10, 1), (10, 1), (1, -1)])
def test_wallclock_rollback_or_monotonic_disagreement_prevents_forecasts(
    tmp_path, monkeypatch, utc_delta, monotonic_delta
):
    anchor = START + 4800
    clock, monotonic = [anchor + 20], [1000.0]
    instance, _, _ = make_runner(
        tmp_path, monkeypatch, clock, monotonic=lambda: monotonic[0]
    )
    try:
        instance.check_clock()
        instance.collect()
        clock[0] += utc_delta
        monotonic[0] += monotonic_delta
        instance.decide()
        assert not list(instance.ledger.rows("forecast"))
        assert not instance.forecasts
        missing = [
            event
            for event in instance.ledger.rows("missing")
            if event["payload"]["anchor"] == anchor
        ]
        assert len(missing) == 3
        assert all(
            event["payload"]["reason"] == "clock_unverified_or_stale"
            for event in missing
        )
    finally:
        instance.ledger.close()


def test_forecast_committed_after_deadline_is_retained_but_never_scored(
    tmp_path, monkeypatch
):
    anchor = START + 4800
    clock = [anchor + 10]
    instance, fake, root = make_runner(tmp_path, monkeypatch, clock)
    original = instance.ledger.append

    def delayed_commit(kind, payload, received_at=None):
        sequence = original(kind, payload, received_at=received_at)
        if kind == "forecast":
            # Model a slow disk commit crossing even the beginning of the target.
            clock[0] = anchor + 61
        return sequence

    monkeypatch.setattr(instance.ledger, "append", delayed_commit)
    try:
        instance.check_clock()
        instance.collect()
        instance.decide()
        assert len(list(instance.ledger.rows("forecast"))) == 1
        commits = list(instance.ledger.rows("forecast_commit"))
        assert len(commits) == 1
        assert commits[0]["payload"]["eligible"] is False
        assert commits[0]["payload"]["committed_at"] == anchor + 61
        assert not instance.forecasts
        instance.collect()
        clock[0] = anchor + 361
        instance.collect()
        instance.resolve()
        assert not list(instance.ledger.rows("outcome"))
    finally:
        instance.ledger.close()
    restarted = runner.Runner(
        root,
        clock=lambda: clock[0],
        monotonic=lambda: clock[0],
        fetcher=fake,
        probe=lambda: {"healthy": True, "samples": []},
    )
    try:
        assert not restarted.forecasts
        assert (anchor, "BTCUSDT") in restarted.decided
        restarted.resolve()
        restarted.decide()
        assert not list(restarted.ledger.rows("outcome"))
        assert len(list(restarted.ledger.rows("forecast"))) == 1
    finally:
        restarted.ledger.close()


def test_restart_excludes_committed_forecast_without_completion_ack(
    tmp_path, monkeypatch
):
    anchor = START + 4800
    clock = [anchor + 10]
    instance, fake, root = make_runner(tmp_path, monkeypatch, clock)
    original = instance.ledger.append

    def crash_before_ack(kind, payload, received_at=None):
        sequence = original(kind, payload, received_at=received_at)
        if kind == "forecast":
            raise RuntimeError(
                "simulated interruption before completion acknowledgement"
            )
        return sequence

    monkeypatch.setattr(instance.ledger, "append", crash_before_ack)
    try:
        instance.check_clock()
        instance.collect()
        with pytest.raises(RuntimeError, match="simulated interruption"):
            instance.decide()
        assert len(list(instance.ledger.rows("forecast"))) == 1
        assert not list(instance.ledger.rows("forecast_commit"))
    finally:
        instance.ledger.close()
    restarted = runner.Runner(
        root,
        clock=lambda: clock[0],
        monotonic=lambda: clock[0],
        fetcher=fake,
        probe=lambda: {"healthy": True, "samples": []},
    )
    try:
        assert not restarted.forecasts
        assert (anchor, "BTCUSDT") in restarted.decided
        clock[0] = anchor + 361
        restarted.collect()
        restarted.resolve()
        assert not list(restarted.ledger.rows("outcome"))
    finally:
        restarted.ledger.close()


def test_clock_disagreement_during_commit_excludes_otherwise_timely_forecasts(
    tmp_path, monkeypatch
):
    anchor = START + 4800
    clock, monotonic = [anchor + 10], [1000.0]
    instance, _, root = make_runner(
        tmp_path, monkeypatch, clock, monotonic=lambda: monotonic[0]
    )
    original = instance.ledger.append

    def changed_clock(kind, payload, received_at=None):
        sequence = original(kind, payload, received_at=received_at)
        if kind == "forecast":
            clock[0] = anchor + 20
            monotonic[0] = 1001.0
        return sequence

    monkeypatch.setattr(instance.ledger, "append", changed_clock)
    try:
        instance.check_clock()
        instance.collect()
        instance.decide()
        commits = list(instance.ledger.rows("forecast_commit"))
        assert commits
        assert all(event["payload"]["committed_at"] <= anchor + 30 for event in commits)
        assert all(event["payload"]["clock_healthy"] is False for event in commits)
        assert all(event["payload"]["eligible"] is False for event in commits)
        assert not instance.forecasts
        audit = verify(
            root / "events.sqlite",
            snapshot=root / "model.json",
            protocol=root / "config.json",
            now=clock[0],
        )
        assert audit["verified"] and audit["eligible_forecast_count"] == 0
    finally:
        instance.ledger.close()


def test_unverified_clock_prevents_polling_and_scoring_matured_pending_targets(
    tmp_path, monkeypatch
):
    anchor = START + 4800
    clock = [anchor + 10]
    instance, fake, _ = make_runner(tmp_path, monkeypatch, clock)
    try:
        instance.collect()
        assert not fake.calls
        assert not list(instance.ledger.rows("receipt"))
        instance.check_clock()
        instance.collect()
        instance.decide()
        assert len(instance.forecasts) == 3
        clock[0] = anchor + 61
        instance.collect()
        clock[0] = anchor + 361
        instance.collect()
        # Both exact target prices are already present, so clock health is the
        # only condition stopping premature trust in a locally advanced UTC.
        for symbol in model.SYMBOLS:
            cached = instance.cache.data["spot_klines", symbol]
            assert anchor in cached and anchor + 300 in cached
        instance.clock_ok = False
        calls = len(fake.calls)
        instance.collect()
        instance.resolve()
        assert len(fake.calls) == calls
        assert not instance.scored
        assert not list(instance.ledger.rows("outcome"))
        instance.check_clock()
        instance.resolve()
        assert len(instance.scored) == 3
        assert len(list(instance.ledger.rows("outcome"))) == 3
    finally:
        instance.ledger.close()


def test_clock_jump_during_poll_discards_the_entire_received_batch(
    tmp_path, monkeypatch
):
    anchor = START + 4800
    clock, monotonic = [anchor + 10], [1000.0]
    instance, fake, _ = make_runner(
        tmp_path, monkeypatch, clock, monotonic=lambda: monotonic[0]
    )
    barrier = Barrier(9)

    def jump_after_responses(*args, **kwargs):
        result = fake(*args, **kwargs)
        if barrier.wait(timeout=5) == 0:
            clock[0] = anchor + 20
        return result

    instance.fetcher = jump_after_responses
    try:
        instance.check_clock()
        instance.collect()
        assert len(fake.calls) == 9
        assert not list(instance.ledger.rows("receipt"))
        assert all(not rows for rows in instance.cache.data.values())
        rejected = list(instance.ledger.rows("clock_rejected_batch"))
        assert len(rejected) == 1
        assert rejected[0]["payload"]["responses_discarded"] == 9
        assert instance.recovery is True
        assert not instance.valid_clock()
    finally:
        instance.ledger.close()


def test_runner_stops_at_registered_end_and_accounts_all_eligible_anchors(
    tmp_path, monkeypatch
):
    clock = [START + 7 * 86400 + 120]
    instance, fake, root = make_runner(tmp_path, monkeypatch, clock)
    instance.run(max_cycles=1)
    status = json.loads((root / "status.json").read_text())
    assert status["phase"] == "completed"
    assert status["counts"]["completed"] == 1
    assert status["counts"]["missing"] == 2015 * 3
    assert not fake.calls


@pytest.mark.parametrize("source", tuple(sources.URLS))
def test_source_request_windows_cannot_reopen_pre_activation_history(source):
    end = START + 7 * 86400
    for recovery in (False, True):
        assert (
            sources.request_parameters(source, "BTCUSDT", START, end, START, recovery)
            is None
        )
        for now in (START + 60, START + 4800, end + 30, end + 120):
            params = sources.request_parameters(
                source, "BTCUSDT", START, end, now, recovery
            )
            if params is not None:
                assert (
                    START * 1000
                    <= params["startTime"]
                    <= params["endTime"]
                    <= end * 1000
                )
                assert params["limit"] <= 100
    with pytest.raises(ValueError):
        sources.request_parameters(source, "DOGEUSDT", START, end, START + 500)


def test_receipt_validation_rejects_backfill_future_duplicate_and_invalid_body():
    now, end = START + 300, START + 7 * 86400
    params = sources.request_parameters("spot_klines", "BTCUSDT", START, end, now, True)
    good = [candle(START + 120)]
    assert sources.validate_body("spot_klines", good, params, START, end, now) == [
        START + 120
    ]
    for body in ([candle(START - 60)], [candle(now)], good + good, {"bad": True}):
        with pytest.raises(ValueError):
            sources.validate_body("spot_klines", body, params, START, end, now)
    bad = copy.deepcopy(good)
    bad[0][9] = "11"
    with pytest.raises(ValueError, match="buy volume"):
        sources.validate_body("spot_klines", bad, params, START, end, now)


def test_fetch_uses_only_registered_get_endpoint_and_records_actual_receipt_time():
    timestamps = iter((START + 300, START + 300.25))
    calls = []

    class Response:
        status_code = 200
        content = b"public response"

        def json(self):
            return [candle(START + 240)]

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    observed = sources.fetch(
        "spot_klines",
        "BTCUSDT",
        START,
        START + 7 * 86400,
        clock=lambda: next(timestamps),
        get=get,
    )
    assert observed["requested_at"] == START + 300
    assert observed["received_at"] == START + 300.25
    assert calls[0][0] == sources.URLS["spot_klines"]
    assert calls[0][1]["allow_redirects"] is False
    assert "headers" not in calls[0][1]
    assert observed["body_sha256"] == sources.digest(observed["body"])


def test_cache_preserves_revisions_and_input_provenance_without_mutating_issued_forecasts():
    cache = runner.Cache()
    body = [candle(START)]
    original = dict(
        source="spot_klines",
        symbol="BTCUSDT",
        body=body,
        content_times=[START],
        body_sha256=sources.digest(body),
    )
    cache.add(1, original)
    changed = copy.deepcopy(original)
    changed["body"][0][4] = "999.0"
    changed["body_sha256"] = sources.digest(changed["body"])
    cache.add(2, changed)
    assert cache.revisions == 1
    assert cache.data["spot_klines", "BTCUSDT"][START] == (changed["body"][0], 2)
    assert original["body"][0][4] != "999.0"
