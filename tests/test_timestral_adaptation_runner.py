"""End-to-end causal ordering and immutable holdout boundaries with tiny fake data."""

from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pytest

from research import timestral_experiment as shared
from research.timestral_long_horizon import Hour
from scripts import run_timestral_adaptation as runner
from scripts import analyze_timestral_adaptation as analysis


class FakeModel:
    def __init__(self, config):
        self.config = config

    def predict(self, context, config, covariates):
        point = (context[-1].log_close - context[-5].log_close) * (.4 if covariates else .3)
        return {"point": point, "quantiles": [point - .01, point, point + .01]}


def setup_source(tmp_path, monkeypatch, future_shift=False):
    config = {"symbol": "BTCUSDT", "development_start": "2025-09-01T00:00:00Z",
              "fit_start": "2025-09-12T00:00:00Z", "validation_start": "2025-09-14T00:00:00Z",
              "test_start": "2025-09-16T00:00:00Z", "test_end": "2025-09-18T00:00:00Z",
              "development_archive": str(tmp_path / "development/archive"),
              "test_archive": str(tmp_path / "test/archive"), "model_cache": str(tmp_path / "model"),
              "lookback": 256, "horizons": [4, 24], "batch_size": 8192, "torch_threads": 1,
              "local_files_only": True, "min_train": 1, "min_validation": 1}
    monkeypatch.setattr(runner.shared, "verified_source_snapshot", lambda config: ({"immutable": True}, []))
    monkeypatch.setattr(runner.shared, "source_snapshot", lambda config: {"immutable": True})
    monkeypatch.setattr(runner, "verify_public_provenance", lambda source, entries: {})
    monkeypatch.setattr(runner, "code_hashes", lambda: {analysis.ANALYSIS_KEY: analysis.digest(analysis.__file__)})
    consumed = []

    def hours(source, audit):
        start, end = shared.utc(source["start"]), shared.utc(source["end"])
        for i in range(int((end - start).total_seconds() // 3600)):
            time = start + timedelta(hours=i)
            absolute = (time - shared.utc(config["development_start"])).total_seconds() / 3600
            close = 10 + .00001 * absolute + .002 * np.sin(absolute / 17)
            if future_shift and time >= shared.utc("2025-09-13"):
                close += .5
            consumed.append(time)
            yield Hour(time, float(close), (.001, 6., .1))
        audit.update(valid_hours=int((end - start).total_seconds() // 3600), invalid_hours=0)

    monkeypatch.setattr(runner.hourly, "read_hours", hours)
    return config, consumed


def test_development_emits_before_future_and_prefix_invariant(tmp_path, monkeypatch):
    config, consumed = setup_source(tmp_path, monkeypatch)

    class OrderedModel(FakeModel):
        def predict(self, context, config, covariates):
            assert max(consumed) == context[-1].time
            return super().predict(context, config, covariates)

    first = tmp_path / "first"
    result = runner.development(config, first, OrderedModel)
    assert result["counts"] == {"4": 24, "24": 4}
    original = runner.read_jsonl(first / "examples.jsonl")
    config, _ = setup_source(tmp_path, monkeypatch, future_shift=True)
    runner.development(config, tmp_path / "second", FakeModel)
    changed = runner.read_jsonl(tmp_path / "second/examples.jsonl")
    old = {r["sample_id"]: r for r in original}
    new = {r["sample_id"]: r for r in changed}
    anchor = "2025-09-13T00:00:00Z"
    for row in original:
        if row["anchor"] <= anchor:
            assert new[row["sample_id"]]["features"] == row["features"]
            assert new[row["sample_id"]]["predictions"] == row["predictions"]
    affected = next(r for r in original if r["anchor"] == anchor and r["horizon_hours"] == 24)
    assert old[affected["sample_id"]]["actual"] != new[affected["sample_id"]]["actual"]
    with pytest.raises(FileExistsError):
        runner.development(config, first, FakeModel)


def test_complete_freeze_then_holdout_is_immutable(tmp_path, monkeypatch):
    config, consumed = setup_source(tmp_path, monkeypatch)
    dev, frozen = tmp_path / "dev-run", tmp_path / "frozen"
    runner.development(config, dev, FakeModel)
    protocol = runner.freeze(dev, frozen)
    before = (frozen / "bundle.json").read_bytes()
    archive = Path(config["test_archive"])
    archive.mkdir(parents=True)
    (archive.parent / "provenance.json").write_text(json.dumps({
        "retrieved_at": shared.stamp(shared.utc(protocol["registered_at"]) + timedelta(seconds=1)),
        "start": config["test_start"], "end": config["test_end"]}))
    consumed.clear()

    class HoldoutOrderedModel(FakeModel):
        def predict(self, context, config, covariates):
            if not consumed:
                assert context[-1].time == shared.utc("2025-09-15T23:00:00Z")
            else:
                assert consumed[-1] == context[-1].time
            return super().predict(context, config, covariates)

    result = runner.holdout(frozen, tmp_path / "test-run", HoldoutOrderedModel)
    assert result["counts"] == {"4": 12, "24": 2}
    assert (frozen / "bundle.json").read_bytes() == before
    rows = runner.read_jsonl(tmp_path / "test-run/evaluation.jsonl")
    assert all(set(row["predictions"]) == set(runner.MODEL_NAMES) for row in rows)
    assert all(row["anchor"] >= config["test_start"] and row["label_end"] <= config["test_end"] for row in rows)
    assert max(row["label_end"] for row in rows) == config["test_end"]
    _, _, groups = analysis.verified_records(tmp_path / "test-run")
    assert {h: len(group) for h, group in groups.items()} == {4: 12, 24: 2}
    with pytest.raises(FileExistsError):
        runner.holdout(frozen, tmp_path / "second-test", FakeModel)


@pytest.mark.parametrize("existing", ["archive", "downloads"])
def test_freeze_rejects_existing_test_archive(tmp_path, monkeypatch, existing):
    config, _ = setup_source(tmp_path, monkeypatch)
    runner.development(config, tmp_path / "dev", FakeModel)
    (Path(config["test_archive"]).parent / existing).mkdir(parents=True)
    with pytest.raises(ValueError, match="before acquiring"):
        runner.freeze(tmp_path / "dev", tmp_path / "freeze")


def test_freeze_rejects_changed_development(tmp_path, monkeypatch):
    config, _ = setup_source(tmp_path, monkeypatch)
    runner.development(config, tmp_path / "dev", FakeModel)
    path = tmp_path / "dev/examples.jsonl"
    path.chmod(0o644)
    with path.open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="output changed"):
        runner.freeze(tmp_path / "dev", tmp_path / "freeze")


def test_rejects_protected_period(tmp_path, monkeypatch):
    config, _ = setup_source(tmp_path, monkeypatch)
    config["test_end"] = "2026-09-01T00:00:00Z"
    with pytest.raises(ValueError, match="protected"):
        runner.validate_config(config)


def test_gap_invalidates_targets_and_rebuilds_complete_context(tmp_path, monkeypatch):
    config, _ = setup_source(tmp_path, monkeypatch)
    original = runner.hourly.read_hours

    def with_gap(source, audit):
        for hour in original(source, audit):
            if hour.time != shared.utc("2025-09-12T02:00:00Z"):
                yield hour

    monkeypatch.setattr(runner.hourly, "read_hours", with_gap)
    result = runner.development(config, tmp_path / "dev", FakeModel)
    assert result["data"]["invalid_targets"] == 2
    assert result["data"]["hour_gap_resets"] == 1
    assert result["counts"] == {"4": 0, "24": 0}


def test_official_provenance_must_bind_verified_archive(tmp_path):
    source = {"symbol": "BTCUSDT", "archive_root": str(tmp_path / "archive"),
              "start": "2025-09-01T00:00:00Z", "end": "2025-09-02T00:00:00Z"}
    entry = {"symbol": "BTCUSDT", "start": source["start"], "end": source["end"], "sha256": "original"}
    name = "BTCUSDT-1s-2025-09-01.zip"
    url = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/" + name
    (tmp_path / "downloads").mkdir()
    (tmp_path / "downloads" / name).write_bytes(b"verified ZIP fixture")
    provenance = {"source": "binance_public_data", "market": "binance_spot", "interval": "1s",
                  "symbol": "BTCUSDT", "start": source["start"], "end": source["end"],
                  "files": [{"archive_entry": entry, "day": source["start"], "filename": name,
                             "url": url, "checksum_url": url + ".CHECKSUM",
                             "sha256": shared.digest_file(tmp_path / "downloads" / name)}]}
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    assert runner.verify_public_provenance(source, [entry])["symbol"] == "BTCUSDT"
    with pytest.raises(ValueError, match="bound"):
        runner.verify_public_provenance(source, [{**entry, "sha256": "other"}])
    (tmp_path / "downloads" / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="ZIP changed"):
        runner.verify_public_provenance(source, [entry])
