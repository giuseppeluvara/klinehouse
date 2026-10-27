"""Causality, tie handling, hierarchical probabilities and directional signal."""

import numpy as np
import pytest

from research import timestral_microdirection as m


def data(n=400):
    rng = np.random.default_rng(3)
    return {
        "time": 1767225600 + np.arange(n) * 60,
        "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n))),
        "volume": np.full(n, 2.0),
        "taker": rng.uniform(0, 2, n),
        "trades": np.full(n, 10.0),
        "tail_flow": rng.uniform(-1, 1, n),
    }


def test_future_changes_cannot_change_features():
    d = data()
    i, x = m.features(d)
    altered = {k: v.copy() for k, v in d.items()}
    for k in ("close", "volume", "taker", "trades"):
        altered[k][201:] *= 2
    j, y = m.features(altered)
    assert np.array_equal(i, j)
    assert np.array_equal(x[i <= 200], y[i <= 200])


def test_delay_target_and_preserved_ties():
    d = data()
    d["close"][100:103] = [100.0, 102.0, 102.0]
    a = m.samples(d, 1, 0)
    b = m.samples(d, 1, 1)
    assert a["label"][a["index"] == 100].item() == 2
    assert b["label"][b["index"] == 100].item() == 1
    assert b["end"][b["index"] == 100].item() == d["time"][102] + 60
    assert b["return"][b["index"] == 100].item() == 0


def test_missing_minute_invalidates_context_and_future_path():
    d = data()
    d = {k: np.delete(v, 100) for k, v in d.items()}
    s = m.samples(d, 5, 1)
    anchors = s["anchor"]
    missing = 1767225600 + 100 * 60
    assert not np.any((anchors > missing - 5 * 60) & (anchors <= missing + 62 * 60))


def test_fit_rejects_immature_labels_and_predict_rejects_future_fit():
    cells = np.zeros((100, 3), dtype=int)
    labels = np.zeros(100, dtype=int)
    with pytest.raises(ValueError, match="matured"):
        m.fit(cells, labels, np.ones(100, dtype=int) * 101, 100)
    p = m.fit(cells, labels, np.ones(100, dtype=int) * 100, 100)
    with pytest.raises(ValueError, match="future"):
        m.predict(p, cells, np.full(100, 99))
    with pytest.raises(ValueError, match="protected"):
        m.predict(p, cells, np.full(100, m.BOUNDARY))


def test_empty_cells_back_off_and_probabilities_normalize():
    p = m.fit(np.zeros((100, 3), dtype=int), np.arange(100) % 3, np.zeros(100), 100)
    assert np.allclose(p["price"]["probabilities"][1], p["frequency"])
    assert np.allclose(p["flow"]["probabilities"][1], p["price"]["probabilities"][0])
    for name in m.NAMES[1:]:
        table = np.asarray(p[name]["probabilities"])
        assert (table > 0).all()
        assert np.allclose(table.sum(axis=1), 1)


def test_flow_can_learn_a_known_conditional_direction():
    rng = np.random.default_rng(4)
    flow = rng.integers(0, 5, size=10000)
    cells = np.stack([np.zeros(10000, dtype=int), flow, flow * 3], axis=1)
    labels = np.where(flow < 2, 0, 2)
    p = m.fit(cells, labels, np.zeros(10000), 100)
    predictions = m.predict(p, cells, np.full(10000, 101))
    assert np.mean(predictions[:, 2].argmax(axis=1) == labels) == 1
    assert np.mean(predictions[:, 1].argmax(axis=1) == labels) < 0.7


def test_protected_source_rejected_before_feature_calculation():
    d = data()
    d["time"] += m.BOUNDARY - d["time"][-1]
    with pytest.raises(ValueError, match="protected"):
        m.features(d)
