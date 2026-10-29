#!/usr/bin/env python3
"""Independent metrics structure, conditional features and paired-score audit."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_universal_direction import (  # noqa: E402
    ROOT,
    ASSETS,
    check,
    same,
    digest,
    verify_metrics,
)
from scripts.verify_futures_flow import (  # noqa: E402
    SOURCE,
    MICRO,
    epoch,
    features as reconstruct_old_features,
)
from scripts.verify_volume_center import reconstruct_centers  # noqa: E402
from scripts.verify_flat_repair import frequency_audit  # noqa: E402
from scripts.verify_temporal_adaptation import paired_check  # noqa: E402
from scripts.verify_lead_lag import (  # noqa: E402
    raw_features,
    offset_fit_check,
    sigmoid,
)

FUTURES = ROOT / "reports/timestral-futures-flow-20261027"
CENTER = ROOT / "reports/timestral-volume-center-20261027"
RAW = ROOT / "research/out/position-information-20261029/raw"
DEFAULT = ROOT / "reports/timestral-position-information-20261029"
BOUNDARY = 1785542400
ARMS = ("price_control", "positions")
PAIRS = (
    ("price_control", "baseline"),
    ("positions", "baseline"),
    ("positions", "price_control"),
)
HEADER = (
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)


def source_requests():
    result = []
    for year in (2024, 2025, 2026):
        first = datetime(year - 1, 10, 1, tzinfo=timezone.utc)
        last = datetime(year, 7, 31, tzinfo=timezone.utc)
        for symbol in ASSETS:
            day = first
            while day < last:
                date = day.strftime("%Y-%m-%d")
                result.append(
                    dict(
                        cohort=year,
                        symbol=symbol,
                        day=date,
                        url=f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{date}.zip",
                    )
                )
                day += timedelta(days=1)
    check(len(result) == 2730, "registered daily source count")
    return result


def conditional_features(data, anchors, hour):
    """Independently match all thirteen required labels without filling gaps."""
    label, values = data["label"], data["values"]
    check(label.ndim == 1 and values.shape == (len(label), 6), "metrics feature schema")
    check(
        np.issubdtype(label.dtype, np.integer)
        and np.all(label >= 0)
        and np.all(label + 300 < BOUNDARY),
        "plausibly protected metric content",
    )
    check(
        anchors.ndim == 1
        and hour.shape == anchors.shape
        and np.all(np.isfinite(hour))
        and np.all(anchors % 300 == 0)
        and np.all(anchors >= 0)
        and np.all(anchors + 360 <= BOUNDARY),
        "conditional feature forecast scope",
    )
    unique, occurrences = np.unique(label, return_counts=True)
    duplicated = unique[occurrences > 1]
    eligible = (
        ~np.isin(label, duplicated)
        & (label % 300 == 0)
        & np.isfinite(values[:, [0, 3]]).all(axis=1)
        & (values[:, 0] > 0)
        & (values[:, 3] > 0)
    )
    ordered = np.argsort(label[eligible])
    times, stocks = label[eligible][ordered], values[eligible][ordered]
    x = np.zeros((len(anchors), 2))
    valid = np.zeros(len(anchors), dtype=bool)
    if not len(times):
        return x, valid
    wanted = anchors[:, None] - 900 - 300 * np.arange(12, -1, -1)
    matches = np.searchsorted(times, wanted)
    bounded = np.minimum(matches, len(times) - 1)
    valid = np.all((matches < len(times)) & (times[bounded] == wanted), axis=1)
    start, end = bounded[valid, 0], bounded[valid, -1]
    change = np.log(stocks[end, 0] / stocks[start, 0])
    current = 1 - 2 / (stocks[end, 3] + 1)
    old = 1 - 2 / (stocks[start, 3] + 1)
    x[valid] = np.clip(
        np.column_stack((np.clip(hour[valid], -3, 3) * change, current - old)), -8, 8
    )
    return x, valid


def flow_check(labels, ratios, source, shift):
    """Reconstruct five exact minute opens with independent matrix indexing."""
    desired = labels[:, None] + shift + np.arange(5) * 60
    positions = np.searchsorted(source["time"], desired)
    bounded = np.minimum(positions, len(source["time"]) - 1)
    valid = (
        np.isfinite(ratios)
        & (ratios > 0)
        & np.all(
            (positions < len(source["time"])) & (source["time"][bounded] == desired),
            axis=1,
        )
    )
    positions = bounded[valid]
    buy = source["taker"][positions].sum(axis=1)
    sell = source["volume"][positions].sum(axis=1) - buy
    valid_volume = (buy > 0) & (sell > 0)
    expected = buy[valid_volume] / sell[valid_volume]
    actual = ratios[valid][valid_volume]
    if len(actual) < 2 or np.std(actual) == 0 or np.std(expected) == 0:
        return dict(n=len(actual), correlation=None, median_relative_error=None)
    a, b = actual - actual.mean(), expected - expected.mean()
    return dict(
        n=len(actual),
        correlation=float((a @ b) / np.sqrt((a @ a) * (b @ b))),
        median_relative_error=float(np.median(np.abs(expected / actual - 1))),
    )


def flow_match(values):
    accepted = []
    for shift, other in (("-300", "0"), ("0", "-300")):
        x, alternative = values[shift], values[other]
        if (
            x["n"] >= 200
            and x["correlation"] is not None
            and alternative["median_relative_error"] is not None
            and x["correlation"] >= 0.98
            and x["median_relative_error"] <= 0.05
            and x["median_relative_error"] <= 0.2 * alternative["median_relative_error"]
        ):
            accepted.append(int(shift))
    return accepted[0] if len(accepted) == 1 else None


def raw_rows(receipt):
    path = RAW / receipt["url"].rsplit("/", 1)[-1]
    check(
        digest(path) == receipt["sha256"] and path.stat().st_size == receipt["bytes"],
        "actual metrics ZIP",
    )
    check(
        path.with_suffix(".zip.CHECKSUM").read_text().split()[0] == receipt["sha256"],
        "official checksum receipt",
    )
    day = int(
        datetime.strptime(receipt["day"], "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    rows, excluded = [], 0
    with zipfile.ZipFile(path) as archive:
        check(
            len(archive.infolist()) == 1
            and archive.infolist()[0].file_size <= 1_000_000,
            "bounded single metrics ZIP member",
        )
        with archive.open(archive.namelist()[0]) as file:
            reader = csv.reader(io.TextIOWrapper(file))
            check(tuple(next(reader)) == HEADER, "exact metrics header")
            for line, row in enumerate(reader, 2):
                check(
                    len(row) == 8 and row[1] == receipt["symbol"],
                    "metrics raw symbol and width",
                )
                stamp = int(
                    datetime.fromisoformat(row[0])
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
                check(day <= stamp <= day + 86400, "raw row within file day")
                if stamp + 300 >= BOUNDARY:
                    excluded += 1
                    continue
                values = [float(v) if v.strip() else float("nan") for v in row[2:]]
                rows.append([stamp, day, line, *values])
    check(
        excluded == receipt["excluded_protected_rows"] and len(rows) == receipt["rows"],
        "raw count and protected exclusions",
    )
    return np.asarray(rows, dtype=float).reshape(-1, 9)


def verify_sources(root, raw=False):
    protocol = json.loads((root / "source-protocol.json").read_text())
    for name, sha in {**protocol["code"], **protocol["inputs"]}.items():
        check(digest(ROOT / name) == sha, f"source protocol input/code {name}")
    requested = source_requests()
    check(protocol["requests"] == requested, "exact allowed acquisition plan")
    sources = json.loads((root / "sources.json").read_text())
    audit = json.loads((root / "timestamp-audit.json").read_text())
    check(
        protocol["at"] < sources["at"] < audit["at"],
        "source acquisition then timestamp audit",
    )
    check(
        len(sources["sources"]) == sources["requested_files"] == 2730,
        "all attempted sources retained",
    )
    keys = {f"{year}-{symbol}" for year in (2024, 2025, 2026) for symbol in ASSETS}
    check(
        set(sources["arrays"]) == {f"{key}-metrics.npz" for key in keys}
        and set(audit["cohorts"]) == keys,
        "all nine source cohorts",
    )
    check(
        audit["observation_time_verified_for_positions"] is False
        and audit["historical_publication_verified"] is False
        and audit["flow_alignment_does_not_verify_stock_fields"] is True
        and audit["protected_data_opened"] is False,
        "field-specific timing limitations preserved",
    )
    arrays, by_day = {}, {}
    total = zip_bytes = missing = verified = 0
    for key in sorted(keys):
        path = root / f"{key}-metrics.npz"
        check(digest(path) == sources["arrays"][path.name], "metrics array hash")
        data = dict(np.load(path))
        check(
            set(data) == {"label", "file_day", "row", "values"}, "metrics array fields"
        )
        n = len(data["label"])
        check(
            data["label"].shape == data["file_day"].shape == data["row"].shape == (n,)
            and data["values"].shape == (n, 6),
            "metrics array shapes",
        )
        for name in ("label", "file_day", "row"):
            check(np.issubdtype(data[name].dtype, np.integer), "integer raw identities")
        year = int(key[:4])
        check(
            n > 0
            and np.all(data["row"] >= 2)
            and data["file_day"].min() >= epoch(year - 1, 10)
            and data["file_day"].max() < epoch(year, 8) - 86400
            and np.all(data["file_day"] % 86400 == 0)
            and np.all(data["label"] >= data["file_day"])
            and np.all(data["label"] <= data["file_day"] + 86400)
            and np.all(data["label"] + 300 < BOUNDARY),
            "admitted file dates and content boundary",
        )
        check(np.all(np.diff(data["file_day"]) >= 0), "source file order")
        arrays[key] = data
        unique, inverse, occurrences = np.unique(
            data["label"], return_inverse=True, return_counts=True
        )
        duplicate = occurrences[inverse] > 1
        bad = (
            duplicate
            | (data["label"] % 300 != 0)
            | ~np.isfinite(data["values"][:, [0, 3]]).all(axis=1)
            | np.any(data["values"][:, [0, 3]] <= 0, axis=1)
        )
        saved = audit["cohorts"][key]
        check(
            saved["rows"] == n
            and saved["duplicate_rows"] == int(duplicate.sum())
            and saved["invalid_stock_rows"] == int(bad.sum())
            and saved["offgrid_rows"] == int(np.sum(data["label"] % 300 != 0)),
            "raw invalid and duplicate accounting",
        )
        future_path = FUTURES / f"{key}-futures.npz"
        check(
            str(future_path.relative_to(ROOT)) in protocol["inputs"],
            "frozen futures audit input",
        )
        future = dict(np.load(future_path))
        check(future["time"].max() < BOUNDARY, "protected flow input")
        days = np.unique(data["file_day"])
        check(
            set(saved["days"]) == {str(int(day)) for day in days},
            "every acquired file day audited",
        )
        matches = []
        for day in days:
            take = data["file_day"] == day
            by_day[(key, int(day))] = np.flatnonzero(take)
            line = data["row"][take]
            check(np.all(np.diff(line) > 0), "unique retained file row identities")
            actual = {
                str(shift): flow_check(
                    data["label"][take], data["values"][take, 5], future, shift
                )
                for shift in (-300, 0)
            }
            record = saved["days"][str(int(day))]
            check(
                set(record["comparisons"]) == {"-300", "0"},
                "both flow timing hypotheses",
            )
            for shift, values in actual.items():
                expected = record["comparisons"][shift]
                check(expected["n"] == values["n"], "exact finite flow membership")
                for field in ("correlation", "median_relative_error"):
                    if values[field] is None:
                        check(expected[field] is None, "undefined flow diagnostic")
                    else:
                        same(expected[field], values[field])
            match = flow_match(actual)
            check(record["matched_shift"] == match, "source-only flow match rule")
            matches.append(match)
        check(
            saved["flow_matches"]
            == {str(shift): matches.count(shift) for shift in (-300, 0, None)},
            "all per-day timing classifications",
        )
        total += n
    receipt_rows = {key: 0 for key in keys}
    for expected, receipt in zip(requested, sources["sources"]):
        check(
            all(receipt[k] == v for k, v in expected.items()),
            "request receipt identity/order",
        )
        key = f"{receipt['cohort']}-{receipt['symbol']}"
        day = int(
            datetime.strptime(receipt["day"], "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        indices = by_day.get((key, day), np.array([], dtype=int))
        if receipt["status"] == "unavailable_404":
            check(len(indices) == 0, "unavailable files not fabricated")
            missing += 1
            continue
        check(
            receipt["status"] == "verified"
            and len(receipt["sha256"]) == 64
            and 0 < receipt["bytes"] <= 1_000_000
            and receipt["rows"] == len(indices),
            "verified source receipt",
        )
        zip_bytes += receipt["bytes"]
        receipt_rows[key] += receipt["rows"]
        verified += 1
        if raw:
            decoded = raw_rows(receipt)
            data = arrays[key]
            stored = np.column_stack(
                (
                    data["label"][indices],
                    data["file_day"][indices],
                    data["row"][indices],
                    data["values"][indices],
                )
            )
            np.testing.assert_allclose(stored, decoded, rtol=0, atol=0, equal_nan=True)
    check(
        zip_bytes == sources["zip_bytes"] and zip_bytes <= 100_000_000,
        "total ZIP byte budget",
    )
    check(
        all(receipt_rows[key] == len(arrays[key]["label"]) for key in keys),
        "all source rows accounted for",
    )
    return arrays, dict(
        verified=True,
        arrays=9,
        requested_files=2730,
        verified_files=verified,
        unavailable_files=missing,
        source_rows=total,
        raw_redecoded=bool(raw),
        daily_flow_comparisons=2 * len(by_day),
        position_time_verified=False,
        historical_publication_verified=False,
        protected_data_opened=False,
    )


def verify(root, raw=False):
    protocol = json.loads((root / "protocol.json").read_text())
    check(
        digest(root / "protocol.json")
        == json.loads((root / "protocol.sha256.json").read_text())["sha256"],
        "protocol hash",
    )
    check(
        {
            "scripts/verify_position_information.py",
            "scripts/run_position_information.py",
            "research/timestral_position_information.py",
        }
        <= set(protocol["code"]),
        "new code and auditor frozen before scoring",
    )
    manifest = json.loads((root / "manifest.json").read_text())
    required = {
        "protocol.json",
        "protocol.sha256.json",
        "source-protocol.json",
        "sources.json",
        "timestamp-audit.json",
        "oi-timestamp-audit.json",
        "opened.json",
        "fits.json",
        "results.json",
        "summary.json",
    }
    required.update(
        f"{year}-{s}-{suffix}.npz"
        for year in (2024, 2025, 2026)
        for s in ASSETS
        for suffix in ("metrics", "predictions")
    )
    check(required <= set(manifest), "complete execution manifest")
    for name, sha in manifest.items():
        check(digest(root / name) == sha, f"artifact hash {name}")
    for name, sha in {**protocol["code"], **protocol["inputs"]}.items():
        check(digest(ROOT / name) == sha, f"code/input hash {name}")
    check(
        json.loads((root / "sources.json").read_text())["at"]
        < json.loads((root / "timestamp-audit.json").read_text())["at"]
        < protocol["at"]
        < json.loads((root / "opened.json").read_text())["at"],
        "source preparation, freeze and score chronology",
    )

    def registered(path):
        check(str(path.relative_to(ROOT)) in protocol["inputs"], "frozen audit input")

    for path in (
        root / "source-protocol.json",
        root / "sources.json",
        root / "timestamp-audit.json",
        root / "oi-timestamp-audit.json",
        FUTURES / "sources.json",
        CENTER / "fits.json",
    ):
        registered(path)
    oi_audit = json.loads((root / "oi-timestamp-audit.json").read_text())
    check(
        oi_audit.get("protected_data_opened") is False
        and oi_audit.get("outcomes_opened") is False
        and oi_audit.get("historical_publication_verified") is False
        and oi_audit.get("position_timing_verified") is False
        and oi_audit.get("oi_sample_alignment_only") is True,
        "OI sample evidence does not certify positions or publication",
    )
    check(
        oi_audit["plan"]["dates"]
        == ["2024-01-01", "2025-01-01", "2026-06-01", "2026-07-01"]
        and oi_audit["plan"]["no_fits"] is True
        and oi_audit["plan"]["no_outcomes"] is True,
        "bounded source-only OI probe",
    )
    if "at" in oi_audit:
        check(
            oi_audit["at"] < protocol["at"], "OI probe precedes candidate registration"
        )
    metrics, source_audit = verify_sources(root, raw=raw)
    futures = {}
    for year in (2024, 2025, 2026):
        for symbol in ASSETS:
            key = f"{year}-{symbol}"
            path = FUTURES / f"{key}-futures.npz"
            registered(path)
            registered(root / f"{key}-metrics.npz")
            futures[key] = dict(np.load(path))
            check(
                futures[key]["time"].max() < BOUNDARY, "protected prior futures source"
            )
    fits = json.loads((root / "fits.json").read_text())
    old_fits = json.loads((CENTER / "fits.json").read_text())
    results = json.loads((root / "results.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    expected_keys = {f"{year}-{s}" for year in (2024, 2025, 2026) for s in ASSETS}
    check(set(results) == expected_keys, "all nine evaluation cohorts")
    check(
        set(fits)
        == {
            f"{year}-{month}-{arm}"
            for year in (2024, 2025, 2026)
            for month in range(1, 8)
            for arm in ARMS
        },
        "all forty-two fits",
    )
    maximum = 0.0
    cases = fallback = 0
    for year in (2024, 2025, 2026):
        spot = {}
        for a, symbol in enumerate(ASSETS):
            path = (
                SOURCE / "history" / f"{symbol}-minutes.npz"
                if year < 2026
                else MICRO
                / ("micro/minutes.npz" if a == 0 else f"transfer/{symbol}-minutes.npz")
            )
            registered(path)
            source = dict(np.load(path))
            check(source["time"].max() < BOUNDARY, "protected spot input")
            use = (source["time"] >= epoch(year - 1, 10)) & (
                source["time"] < epoch(year, 8)
            )
            spot[symbol] = {key: value[use] for key, value in source.items()}
        t, end, x, y, old_available = reconstruct_old_features(FUTURES, year, futures)
        center = reconstruct_centers(year, t)["uniform"]
        common = np.clip(-(x[:, :, 3] + 2 * x[:, :, 16]) / 3, -3, 3)
        agreement = common * (-x[:, :, 3]) > 0
        signal = np.where(agreement, center[None, :], common)
        fields = {name: [] for name in ARMS}
        availability = []
        for symbol in ASSETS:
            pc, _, spot_ok = raw_features(spot[symbol], spot[symbol], t)
            position, stock_ok = conditional_features(
                metrics[f"{year}-{symbol}"], t, pc[:, 1]
            )
            fields["price_control"].append(pc)
            fields["positions"].append(position)
            availability.append(spot_ok & stock_ok)
        valid = np.all(availability, axis=0)
        fields = {name: np.stack(values) for name, values in fields.items()}
        pieces = {name: [] for name in (*ARMS, "baseline")}
        indices = []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (t >= cutoff - 90 * 86400) & (end <= cutoff) & old_available & valid
            test = (t >= cutoff) & (t < epoch(year, month + 1))
            labels = y[:, train].reshape(-1)
            moving = labels != 1
            binary = (labels[moving] == 2).astype(int)
            tt, ee = np.tile(t[train], 3)[moving], np.tile(end[train], 3)[moving]
            ids = np.concatenate(
                [np.column_stack((np.full(train.sum(), a), t[train])) for a in range(3)]
            ).astype("<i8")[moving]
            beta = old_fits[f"{year}-{month}-uniform"]["adjusted"]["beta"]
            check(np.isfinite(beta) and 0 <= beta <= 8, "frozen baseline coefficient")
            offset = beta * signal
            pieces["baseline"].append(sigmoid(offset[:, test]))
            for name in ARMS:
                state = fits[f"{year}-{month}-{name}"]
                maximum = max(
                    maximum,
                    offset_fit_check(
                        state,
                        fields[name][:, train].reshape(-1, 2)[moving],
                        binary,
                        offset[:, train].reshape(-1)[moving],
                        tt,
                        ee,
                        ids,
                        cutoff,
                        beta,
                    ),
                )
                standardized = np.clip(fields[name][:, test] / state["scale"], -3, 3)
                pred = sigmoid(
                    offset[:, test] + standardized @ np.asarray(state["coefficients"])
                )
                pred[:, ~valid[test]] = pieces["baseline"][-1][:, ~valid[test]]
                pieces[name].append(pred)
            indices.append(np.flatnonzero(test))
        indices = np.concatenate(indices)
        predictions = {
            name: np.concatenate(values, axis=1) for name, values in pieces.items()
        }
        for a, symbol in enumerate(ASSETS):
            key = f"{year}-{symbol}"
            folder, prefix = (
                (SOURCE, symbol) if year == 2026 else (SOURCE / "history", key)
            )
            paths = (
                folder / f"{prefix}-predictions.npz",
                folder / f"{prefix}-outcomes.npz",
                CENTER / f"{key}-predictions.npz",
            )
            for path in paths:
                registered(path)
            prior, outcome, old = [dict(np.load(path)) for path in paths]
            saved = dict(np.load(root / f"{key}-predictions.npz"))
            check(
                set(saved) == {"anchor", "available", *ARMS, "baseline"},
                "saved predictions schema",
            )
            for d in (prior, outcome, old, saved):
                same(d["anchor"], t[indices])
            same(outcome["label"], y[a, indices])
            same(outcome["end"], end[indices])
            check(outcome["end"].max() <= BOUNDARY, "protected target boundary")
            frequency_audit(year, symbol, prior, outcome)
            same(saved["available"], valid[indices])
            same(predictions["baseline"][a], old["uniform_up"])
            flat = (
                prior["cross" if year == 2026 else "probability"][:, 1]
                + prior["frequency"][:, 1]
            ) / 2
            p = {}
            for name, q in predictions.items():
                up = q[a]
                p[name] = np.column_stack(
                    ((1 - flat) * (1 - up), flat, (1 - flat) * up)
                )
                same(saved[name], p[name])
                np.testing.assert_allclose(
                    saved[name].sum(axis=1), 1, atol=1e-12, rtol=0
                )
                check(
                    np.all(
                        np.isfinite(saved[name]) & (saved[name] > 0) & (saved[name] < 1)
                    ),
                    "finite normalized forecasts",
                )
                same(saved[name][:, 1], flat)
                if name != "baseline":
                    same(
                        saved[name][~valid[indices]], saved["baseline"][~valid[indices]]
                    )
            r = results[key]
            check(
                set(r) == {"n", "missing_contexts", "original", "comparisons"}
                and r["n"] == len(indices)
                and r["missing_contexts"] == int((~valid[indices]).sum()),
                "unchanged membership and shared fallback",
            )
            check(set(r["original"]) == {*ARMS, "baseline"}, "all original evaluations")
            for name in p:
                verify_metrics(
                    r["original"][name],
                    y[a, indices],
                    p[name],
                    prior["frequency"],
                    prior["prior_reversal"],
                    t[indices],
                )
            selected = np.maximum(p["baseline"][:, 0], p["baseline"][:, 2]) >= 0.52
            check(
                set(r["comparisons"])
                == {f"{name}_vs_{control}" for name, control in PAIRS},
                "all three paired comparisons",
            )
            for name, control in PAIRS:
                comparison = r["comparisons"][f"{name}_vs_{control}"]
                check(
                    set(comparison) == {"full", "selected", "direction"}, "all scopes"
                )
                for scope, mask in (
                    ("full", np.ones(len(indices), dtype=bool)),
                    ("selected", selected),
                    ("direction", y[a, indices] != 1),
                ):
                    paired_check(
                        comparison[scope],
                        y[a, indices],
                        p[name],
                        p[control],
                        t[indices],
                        mask,
                        directional=(scope == "direction"),
                    )
            cases += len(indices)
            fallback += r["missing_contexts"]
    original = {
        name: sum(r["original"][name]["gate"] for r in results.values())
        for name in (*ARMS, "baseline")
    }
    counts = {
        f"{name}_vs_{control}": {
            scope: sum(
                r["comparisons"][f"{name}_vs_{control}"][scope]["improvement_gate"]
                for r in results.values()
            )
            for scope in ("full", "selected", "direction")
        }
        for name, control in PAIRS
    }
    success = original["positions"] == 9 and all(
        counts[f"positions_vs_{control}"][scope] == 9
        for control in ("baseline", "price_control")
        for scope in ("full", "direction")
    )
    expected_summary = dict(
        original_passes=original,
        paired_gates=counts,
        empirical_stable_improvement=success,
        timing_certified=False,
        stable_improvement=False,
        anchors=cases,
    )
    check(
        cases == 550362 and summary == expected_summary, "registered primary criteria"
    )
    return dict(
        verified=True,
        conditional_replay_verified=True,
        source_audit=source_audit,
        fits=42,
        cases=cases,
        cohorts=9,
        past_frequency_windows=63,
        original_evaluations=27,
        full_comparisons=27,
        selected_comparisons=27,
        direction_comparisons=27,
        maximum_fit_gradient=maximum,
        missing_contexts=fallback,
        protected_data_opened=False,
        empirical_stable_improvement=success,
        timing_certified=False,
        stable_improvement=False,
        historical_publication_verified=False,
        position_time_verified=False,
        summary=summary,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Independently decode every cached metrics ZIP",
    )
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="Audit inputs and flow timing without opening outcomes",
    )
    args = parser.parse_args()
    args.root = args.root.resolve()
    result = (
        verify_sources(args.root, raw=args.raw)[1]
        if args.sources_only
        else verify(args.root, raw=args.raw)
    )
    print(json.dumps(result, sort_keys=True))
