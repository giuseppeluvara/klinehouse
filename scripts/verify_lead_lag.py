#!/usr/bin/env python3
"""Independent cached-price, offset-fit and lead-lag score audit; no fitting."""

import argparse
import csv
import hashlib
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

FUTURES = ROOT / "reports/timestral-futures-flow-20261027"
CENTER = ROOT / "reports/timestral-volume-center-20261027"
RAW = ROOT / "research/out/futures-flow-20261027/raw"
ARMS = ("price_control", "lead_lag")
PAIRS = (
    ("price_control", "baseline"),
    ("lead_lag", "baseline"),
    ("lead_lag", "price_control"),
)
BOUNDARY = 1785542400


def sigmoid(x):
    return (1 + np.tanh(x / 2)) / 2


def raw_features(spot, futures, anchors):
    """Use explicit historical indices and convolution, never production features."""
    js = []
    valid = np.ones(len(anchors), dtype=bool)
    for source, required in ((spot, 61), (futures, 60)):
        t, c = source["time"], source["close"]
        check(
            t.ndim == 1 and c.shape == t.shape and len(t) >= required + 1,
            "source shapes and trailing context",
        )
        check(
            np.all(np.diff(t) > 0)
            and np.all(t % 60 == 0)
            and t.max() < BOUNDARY
            and np.all(np.isfinite(c) & (c > 0)),
            "finite pre-protected source",
        )
        j = np.searchsorted(t, anchors - 120)
        ok = (j >= required) & (j < len(t))
        ids = np.flatnonzero(ok)
        ok[ids] &= (t[j[ids]] == anchors[ids] - 120) & (
            t[j[ids]] - t[j[ids] - required] == required * 60
        )
        valid &= ok
        js.append(j)
    price = np.zeros((len(anchors), 2))
    lead = np.zeros_like(price)
    ids = np.flatnonzero(valid)
    j, k = js[0][ids], js[1][ids]
    # These sixty returns end at j-1, excluding the latest observed return j.
    returns = np.r_[0.0, np.log(spot["close"][1:] / spot["close"][:-1])]
    rms = np.sqrt(
        np.maximum(
            np.convolve(returns * returns, np.ones(60) / 60, "valid")[j - 60],
            1e-16,
        )
    )
    for column, horizon in enumerate((5, 60)):
        spot_return = np.log(spot["close"][j] / spot["close"][j - horizon])
        future_return = np.log(futures["close"][k] / futures["close"][k - horizon])
        denominator = rms * np.sqrt(horizon)
        price[ids, column] = np.clip(spot_return / denominator, -8, 8)
        lead[ids, column] = np.clip((future_return - spot_return) / denominator, -8, 8)
    return price, lead, valid


def offset_fit_check(state, x, y, offset, anchors, ends, ids, cutoff, baseline_beta):
    n = len(y)
    check(
        x.shape == (n, 2)
        and offset.shape == anchors.shape == ends.shape == (n,)
        and ids.shape == (n, 2)
        and n >= 100
        and np.all(np.isfinite(x))
        and np.all(np.abs(x) <= 8)
        and np.all(np.isfinite(offset))
        and np.all(np.isin(y, (0, 1)))
        and np.all(anchors >= cutoff - 90 * 86400)
        and np.all(ends <= cutoff)
        and cutoff < BOUNDARY,
        "registered mature training samples",
    )
    same(ends, anchors + 360)
    same(ids[:, 1], anchors)
    check(np.all(np.isin(ids[:, 0], (0, 1, 2))), "registered pooled asset IDs")
    days, inverse, counts = np.unique(
        anchors // 86400, return_inverse=True, return_counts=True
    )
    check(len(days) >= 80, "at least eighty observed training days")
    w = 1 / (len(days) * counts[inverse])
    scale = np.maximum(np.sqrt(np.sum(w[:, None] * x * x, axis=0)), 1e-8)
    same(state["scale"], scale)
    z = np.clip(x / scale, -3, 3)
    coefficients = np.asarray(state["coefficients"])
    check(
        coefficients.shape == (2,) and np.all(np.isfinite(coefficients)),
        "exactly two finite correction coefficients",
    )
    logits = offset + z @ coefficients
    probability = sigmoid(logits)
    gradient = (
        np.sum((w * (probability - y))[:, None] * z, axis=0) + 0.25 * coefficients
    )
    maximum = float(np.max(np.abs(gradient)))
    check(maximum <= 1e-7, "strictly convex ridge optimum gradient")
    same(state["gradient"], gradient)
    same(state["max_abs_gradient"], maximum)
    loss = float(w @ (np.logaddexp(0, logits) - y * logits))
    same(state["training_loss"], loss)
    same(state["objective"], loss + 0.25 / 2 * np.sum(coefficients**2))
    same(state["baseline_beta"], baseline_beta)
    check(
        state["n"] == n
        and state["cutoff"] == cutoff
        and state["days"] == len(days)
        and state["penalty"] == 0.25
        and state["first_anchor"] == int(anchors.min())
        and state["last_anchor"] == int(anchors.max())
        and state["max_end"] == int(ends.max()),
        "saved fit scope and fixed penalty",
    )
    check(
        state["training_ids_sha256"]
        == hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
        "training IDs",
    )
    check(
        state["training_labels_sha256"]
        == hashlib.sha256(y.astype("<i8").tobytes()).hexdigest(),
        "training labels",
    )
    return maximum


def verify_sources(root, raw=False):
    """Validate source-only preparation without opening any outcome arrays."""
    provenance = json.loads((root / "sources.json").read_text())
    old = json.loads((FUTURES / "sources.json").read_text())
    old_manifest = json.loads((FUTURES / "manifest.json").read_text())
    check(
        digest(FUTURES / "sources.json") == old_manifest["sources.json"],
        "frozen prior source receipt",
    )
    expected = []
    for cohort in (2024, 2025, 2026):
        for symbol in ASSETS:
            for number in range((cohort - 1) * 12 + 9, cohort * 12 + 7):
                year, month = number // 12, number % 12 + 1
                name = f"{symbol}-1m-{year}-{month:02d}.zip"
                expected.append(
                    f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1m/{name}"
                )
    check(
        [r["url"] for r in provenance["sources"]] == expected and len(expected) == 90,
        "exactly ninety admitted cached sources",
    )
    old_records = {r["url"]: r for r in old["sources"]}
    check(set(old_records) == set(expected), "same original archive scope")
    keys = {f"{year}-{s}" for year in (2024, 2025, 2026) for s in ASSETS}
    check(
        set(provenance["arrays"]) == {f"{key}-prices.npz" for key in keys},
        "all nine source price arrays",
    )
    arrays = {}
    for key in sorted(keys):
        path = root / f"{key}-prices.npz"
        check(digest(path) == provenance["arrays"][path.name], "source array hash")
        arrays[key] = dict(np.load(path))
        d = arrays[key]
        year = int(key[:4])
        check(
            set(d) == {"time", "close"}
            and d["time"].ndim == 1
            and d["close"].shape == d["time"].shape
            and len(d["time"]) > 0,
            "source price schema",
        )
        t, c = d["time"], d["close"]
        check(
            np.issubdtype(t.dtype, np.integer)
            and np.all(np.diff(t) > 0)
            and np.all(t % 60 == 0)
            and t.min() >= epoch(year - 1, 10)
            and t.max() < min(epoch(year, 8), BOUNDARY)
            and np.all(np.isfinite(c) & (c > 0)),
            "source values and protected limit",
        )
    counts = {key: 0 for key in keys}
    for receipt in provenance["sources"]:
        old_receipt = old_records[receipt["url"]]
        for field in ("sha256", "rows", "bytes"):
            check(receipt[field] == old_receipt[field], "cached source provenance")
        name = receipt["url"].rsplit("/", 1)[-1]
        symbol, _, year_text, month_text = name.removesuffix(".zip").split("-")
        year, month = int(year_text), int(month_text)
        key = f"{year + (month >= 10)}-{symbol}"
        counts[key] += receipt["rows"]
        if not raw:
            continue
        path = RAW / name
        check(
            digest(path) == receipt["sha256"]
            and path.stat().st_size == receipt["bytes"],
            "actual cached raw archive",
        )
        check(
            path.with_suffix(".zip.CHECKSUM").read_text().split()[0]
            == receipt["sha256"],
            "cached official checksum",
        )
        times, closes = [], []
        with zipfile.ZipFile(path) as archive:
            check(
                len(archive.infolist()) == 1
                and archive.infolist()[0].file_size <= 32_000_000,
                "bounded single-member raw archive",
            )
            with archive.open(archive.namelist()[0]) as file:
                for row in csv.reader(io.TextIOWrapper(file)):
                    if row[0] == "open_time":
                        continue
                    check(len(row) == 12, "raw kline columns")
                    ts, close_ms = int(row[0]), int(row[6])
                    o, high, low, close = map(float, row[1:5])
                    volume, buys, trades = float(row[5]), float(row[9]), int(row[8])
                    check(
                        ts % 60000 == 0
                        and close_ms == ts + 59999
                        and epoch(year, month)
                        <= ts // 1000
                        < epoch(year + (month == 12), month % 12 + 1)
                        and ts // 1000 < BOUNDARY,
                        "raw millisecond candle times",
                    )
                    check(
                        np.isfinite([o, high, low, close, volume, buys]).all()
                        and 0 < low <= min(o, close) <= max(o, close) <= high
                        and 0 <= buys <= volume
                        and trades >= 0,
                        "raw OHLC and activity constraints",
                    )
                    times.append(ts // 1000)
                    closes.append(close)
        times = np.asarray(times)
        check(
            len(times) == receipt["rows"] and np.all(np.diff(times) > 0),
            "raw row count and chronology",
        )
        expected_minutes = (
            epoch(year + (month == 12), month % 12 + 1) - epoch(year, month)
        ) // 60
        check(
            old_receipt["missing_minutes"] == expected_minutes - len(times),
            "original source gaps preserved",
        )
        indices = np.searchsorted(arrays[key]["time"], times)
        same(arrays[key]["time"][indices], times)
        same(arrays[key]["close"][indices], closes)
    check(
        all(counts[key] == len(arrays[key]["time"]) for key in keys),
        "all original rows retained without interpolation",
    )
    total = sum(counts.values())
    check(total == 3944160, "all admitted raw price rows")
    return arrays, dict(
        verified=True,
        arrays=9,
        source_zip_files=90,
        source_rows=total,
        raw_redecoded=bool(raw),
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
            "scripts/verify_lead_lag.py",
            "scripts/run_lead_lag.py",
            "research/timestral_lead_lag.py",
        }
        <= set(protocol["code"]),
        "new code and auditor frozen before scoring",
    )
    manifest = json.loads((root / "manifest.json").read_text())
    required = {
        "protocol.json",
        "protocol.sha256.json",
        "sources.json",
        "opened.json",
        "fits.json",
        "results.json",
        "summary.json",
    }
    required.update(
        f"{year}-{s}-{suffix}.npz"
        for year in (2024, 2025, 2026)
        for s in ASSETS
        for suffix in ("prices", "predictions")
    )
    check(required <= set(manifest), "complete execution manifest")
    for name, sha in manifest.items():
        check(digest(root / name) == sha, f"artifact hash {name}")
    for name, sha in {**protocol["code"], **protocol["inputs"]}.items():
        check(digest(ROOT / name) == sha, f"code/input hash {name}")
    check(
        json.loads((root / "sources.json").read_text())["at"]
        < protocol["at"]
        < json.loads((root / "opened.json").read_text())["at"],
        "source preparation, freeze and score chronology",
    )

    def registered(path):
        check(str(path.relative_to(ROOT)) in protocol["inputs"], "frozen audit input")

    for path in (root / "sources.json", FUTURES / "sources.json", CENTER / "fits.json"):
        registered(path)
    prices, source_audit = verify_sources(root, raw=raw)
    futures = {}
    for year in (2024, 2025, 2026):
        for symbol in ASSETS:
            key = f"{year}-{symbol}"
            path = FUTURES / f"{key}-futures.npz"
            registered(path)
            registered(root / f"{key}-prices.npz")
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
            pc, ll, ok = raw_features(spot[symbol], prices[f"{year}-{symbol}"], t)
            fields["price_control"].append(pc)
            fields["lead_lag"].append(ll)
            availability.append(ok)
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
    success = original["lead_lag"] == 9 and all(
        counts[f"lead_lag_vs_{control}"][scope] == 9
        for control in ("baseline", "price_control")
        for scope in ("full", "direction")
    )
    expected_summary = dict(
        original_passes=original,
        paired_gates=counts,
        stable_improvement=success,
        anchors=cases,
    )
    check(
        cases == 550362 and summary == expected_summary, "registered primary criteria"
    )
    return dict(
        verified=True,
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
        stable_improvement=success,
        summary=summary,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "reports/timestral-lead-lag-20261028"
    )
    parser.add_argument(
        "--raw", action="store_true", help="Independently decode all ninety cached ZIPs"
    )
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="Audit prepared prices without opening outcomes",
    )
    args = parser.parse_args()
    result = (
        verify_sources(args.root, raw=args.raw)[1]
        if args.sources_only
        else verify(args.root, raw=args.raw)
    )
    print(json.dumps(result, sort_keys=True))
