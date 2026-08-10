from __future__ import annotations
# ruff: noqa: E702

import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import rapid_evidence as rapid
from research import sequence_experiment as seq


def test_protocol_and_frozen_bundle_hashes():
    freeze = rapid.verify_bundle()
    assert rapid.sha(rapid.PROTOCOL) == rapid.PROTOCOL_SHA
    assert freeze["git_commit"] == "1a4cd9c723d94e5280c5532b5a56c24d5ade027b"
    assert freeze["final_samples"] == 52_424


def test_oof_identity_is_exactly_the_frozen_validation_ids():
    data = seq.load_dataset(); all_ids = np.asarray(data["anchor"])
    oof = pd.read_parquet(seq.ROOT / "validation" / "oof.parquet").sort_values("sample_id")
    expected = []
    for fold in (1, 2, 3):
        saved = np.load(rapid.BUNDLE / f"fold-{fold}-ids.npz")
        validation = saved["validation"]
        assert np.array_equal(all_ids[rapid.ids_to_rows(all_ids, validation)], validation)
        expected.append(validation)
    expected = np.sort(np.concatenate(expected))
    assert len(expected) == len(np.unique(expected)) == 13_104
    assert np.array_equal(oof.sample_id.to_numpy(), expected)


def test_shuffle_is_sample_specific_deterministic_and_preserves_multiset():
    x = np.arange(3 * 12 * 2, dtype=np.float32).reshape(3, 12, 2)
    ids = np.array([1_700_000_000_000, 1_700_003_600_000, 1_700_007_200_000])
    first = rapid.shuffled_steps(x, ids); second = rapid.shuffled_steps(x, ids)
    assert np.array_equal(first, second)
    assert not np.array_equal(first[0], x[0])
    assert not np.array_equal(first[0] - x[0, 0], first[1] - x[1, 0])
    for before, after in zip(x, first):
        assert sorted(map(tuple, before)) == sorted(map(tuple, after))


def test_aggregate_has_exact_48_causal_statistics():
    x = np.arange(2 * 120 * 12, dtype=np.float32).reshape(2, 120, 12)
    result = rapid.aggregate_48(x)
    assert result.shape == (2, 48)
    np.testing.assert_allclose(result[:, :12], x[:, -1])
    np.testing.assert_allclose(result[:, 12:24], x.mean(1), rtol=1e-6)
    np.testing.assert_allclose(result[:, 24:36], x.std(1), rtol=1e-6)
    np.testing.assert_allclose(result[:, 36:], 12.0, rtol=1e-6)


def test_fold_class_means_use_only_training_rows():
    y = np.array([0, 1, 2, 0, 1, 2]); returns = np.array([-3., 0., 3., -6., 1., 9.])
    train = np.array([0, 1, 2]); expected = rapid.fold_class_means(y, returns, train)
    changed = returns.copy(); changed[3:] = 10_000
    np.testing.assert_array_equal(rapid.fold_class_means(y, changed, train), expected)


def test_ensemble_is_arithmetic_mean_of_seed_probabilities():
    probabilities = [np.eye(3), np.roll(np.eye(3), 1, axis=1), np.full((3, 3), 1 / 3)]
    np.testing.assert_allclose(rapid.ensemble(probabilities), np.stack(probabilities).mean(0))


def test_shadow_call_graph_has_no_fit_path():
    tree = ast.parse(inspect.getsource(rapid.shadow_once))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            calls.append(name)
    assert not any("fit" in name.lower() or "train" in name.lower() for name in calls)
    assert "fit_neural" not in inspect.getsource(rapid.shadow_once)


def test_jsonl_is_append_only_and_timestamp_idempotent(tmp_path):
    path = tmp_path / "events.jsonl"; first = {"sample_id": 1, "value": "first"}
    assert rapid.append_unique(path, first)
    prefix = path.read_bytes()
    assert not rapid.append_unique(path, {"sample_id": 1, "value": "changed"})
    assert path.read_bytes() == prefix
    assert rapid.append_unique(path, {"sample_id": 2, "value": "second"})
    assert path.read_bytes().startswith(prefix)
    assert [json.loads(line)["sample_id"] for line in path.read_text().splitlines()] == [1, 2]


def test_label_maturation_is_separate_append_only_event(tmp_path, monkeypatch):
    monkeypatch.setattr(rapid, "OUT", tmp_path)
    anchor = pd.Timestamp("2026-08-10T10:00:00Z")
    event = {"sample_id": int(anchor.timestamp() * 1000), "anchor": anchor.isoformat(), "models": {}}
    rapid.append_unique(tmp_path / "events.jsonl", event); original = (tmp_path / "events.jsonl").read_bytes()
    now = anchor + pd.Timedelta(hours=2)
    assert rapid.mature_shadow(now, lambda _: 0.003) == 1
    assert rapid.mature_shadow(now, lambda _: 0.003) == 0
    assert (tmp_path / "events.jsonl").read_bytes() == original
    label = json.loads((tmp_path / "maturations.jsonl").read_text())
    assert label["sample_id"] == event["sample_id"] and label["y"] == 2
