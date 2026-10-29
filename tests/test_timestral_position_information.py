import csv
import io
import zipfile

import numpy as np
import pytest

from research.timestral_position_information import (
    ASSUMED_LAG_SECONDS,
    BOUNDARY,
    TIMING_CERTIFIED,
    metrics_features,
)
from scripts import prepare_position_information as source


def sample(n=50):
    label = 1704067200 + 300 * np.arange(n)
    values = np.ones((n, 6))
    values[:, 0] = 100 * np.exp(np.arange(n) * 0.01)
    values[:, 3] = 1 + np.arange(n) * 0.05
    return dict(
        label=label, values=values, file_day=label // 86400, row=np.arange(n) + 2
    )


def test_exact_lag_endpoints_and_predeclared_transforms():
    d = sample()
    anchors = d["label"][[12, 20]] + ASSUMED_LAG_SECONDS
    actual, valid = metrics_features(d, anchors, np.array([2.0, -9.0]))
    assert valid.all()
    for r, j in enumerate((12, 20)):
        old = j - 12
        oi = np.log(d["values"][j, 0]) - np.log(d["values"][old, 0])
        a, b = d["values"][[j, old], 3]
        assert actual[r, 0] == pytest.approx((2 if r == 0 else -3) * oi)
        assert actual[r, 1] == pytest.approx((a - 1) / (a + 1) - (b - 1) / (b + 1))
    assert TIMING_CERTIFIED is False


def test_missing_rows_and_missing_exact_latest_are_not_filled():
    d = sample()
    anchors = d["label"][[11, 15, 35]] + ASSUMED_LAG_SECONDS
    missing = {key: np.delete(value, 8, axis=0) for key, value in d.items()}
    x, valid = metrics_features(missing, anchors, np.ones(3))
    np.testing.assert_array_equal(valid, [False, False, True])
    np.testing.assert_array_equal(x[:2], np.zeros((2, 2)))
    latest_missing = {key: np.delete(value, 35, axis=0) for key, value in d.items()}
    x, valid = metrics_features(latest_missing, anchors[-1:], np.ones(1))
    assert not valid[0]
    np.testing.assert_array_equal(x, np.zeros((1, 2)))


def test_all_duplicates_offgrid_and_invalid_value_rows_are_unusable():
    d = sample()
    anchor = d["label"][15:16] + ASSUMED_LAG_SECONDS
    duplicated = {
        key: np.concatenate((value, value[8:9]), axis=0) for key, value in d.items()
    }
    # Both identical copies and conflicting copies invalidate the label.
    for conflict in (False, True):
        if conflict:
            duplicated["values"][-1, 0] *= 1.2
        x, valid = metrics_features(duplicated, anchor, np.ones(1))
        assert not valid[0]
        np.testing.assert_array_equal(x, np.zeros((1, 2)))
    for column, value in ((0, 0), (0, -1), (3, 0), (3, -1), (0, np.nan), (3, np.nan)):
        bad = {key: a.copy() for key, a in d.items()}
        bad["values"][8, column] = value
        assert not metrics_features(bad, anchor, np.ones(1))[1][0]
    offgrid = {key: a.copy() for key, a in d.items()}
    offgrid["label"][8] += 2
    assert not metrics_features(offgrid, anchor, np.ones(1))[1][0]


def test_unused_missing_columns_do_not_discard_valid_position_information():
    d = sample()
    anchor = d["label"][15:16] + ASSUMED_LAG_SECONDS
    expected = metrics_features(d, anchor, np.ones(1))
    d["values"][:, [1, 2, 4, 5]] = np.nan
    actual = metrics_features(d, anchor, np.ones(1))
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    assert actual[1][0]


def test_future_append_or_changes_do_not_affect_earlier_features():
    d = sample()
    anchors = d["label"][[12, 20]] + ASSUMED_LAG_SECONDS
    short = {key: value[:21].copy() for key, value in d.items()}
    first = metrics_features(short, anchors, np.ones(2))
    for a, b in zip(first, metrics_features(d, anchors, np.ones(2))):
        np.testing.assert_array_equal(a, b)
    d["values"][21:, 0] *= 4
    d["values"][21:, 3] *= 0.5
    for a, b in zip(first, metrics_features(d, anchors, np.ones(2))):
        np.testing.assert_array_equal(a, b)


def test_plausible_future_content_and_target_guard_precede_feature_selection():
    d = sample()
    anchor = d["label"][15:16] + ASSUMED_LAG_SECONDS
    protected = {key: value.copy() for key, value in d.items()}
    protected["label"][-1] = BOUNDARY - 300
    with pytest.raises(ValueError):
        metrics_features(protected, anchor, np.ones(1))
    with pytest.raises(ValueError):
        metrics_features(d, np.array([BOUNDARY - 300]), np.ones(1))
    with pytest.raises(ValueError):
        metrics_features(d, anchor + 1, np.ones(1))


def test_raw_clipping_and_empty_input_keep_explicit_validity():
    d = sample()
    d["values"][12, 0] = 1e20
    anchor = d["label"][12:13] + ASSUMED_LAG_SECONDS
    x, valid = metrics_features(d, anchor, np.array([100.0]))
    assert valid[0] and x[0, 0] == 8
    empty = {key: value[:0] for key, value in d.items()}
    x, valid = metrics_features(empty, anchor, np.ones(1))
    np.testing.assert_array_equal(x, np.zeros((1, 2)))
    assert not valid.any()


def test_registered_source_requests_exclude_july31_and_protected_days():
    requests = list(source.items())
    assert len(requests) == 2730
    assert len({(symbol, day) for _, symbol, day in requests}) == len(requests)
    for cohort, symbol, day in requests:
        assert f"{cohort - 1}-10-01" <= day < f"{cohort}-07-31"
        assert day < "2026-08-01"
        assert source.url(symbol, day).endswith(f"{symbol}-metrics-{day}.zip")
    for day in ("2024-07-31", "2025-07-31", "2026-07-31", "2026-08-01"):
        with pytest.raises(ValueError):
            source.url("BTCUSDT", day)
    with pytest.raises(ValueError):
        source.url("DOGEUSDT", "2026-07-30")


def test_source_decoder_guards_protected_content_before_numeric_parsing(tmp_path):
    def archive(name, row):
        text = io.StringIO()
        writer = csv.writer(text)
        writer.writerow(source.HEADER)
        writer.writerow(row)
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as zipped:
            zipped.writestr("metrics.csv", text.getvalue())
        return path

    path = archive(
        "july30.zip",
        ["2026-07-30 12:00:00", "BTCUSDT", "100", "200", "1", "2", "3", "4"],
    )
    rows, excluded = source.decode(path, "BTCUSDT", "2026-07-30")
    assert rows.shape == (1, 9) and excluded == 0
    expected_time = np.datetime64("2026-07-30T12:00:00", "s").astype(np.int64)
    assert rows[0, 0] == expected_time and rows[0, 2] == 2
    np.testing.assert_array_equal(rows[0, 3:], [100, 200, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        source.decode(path, "ETHUSDT", "2026-07-30")
    protected = archive(
        "hypothetical-july31.zip",
        ["2026-07-31 23:55:00", "BTCUSDT", *(["invalid-number"] * 6)],
    )
    rows, excluded = source.decode(protected, "BTCUSDT", "2026-07-31")
    assert rows.shape == (0, 9) and excluded == 1
