"""Verify saved evidence independently, without training or model deserialization.

Run from any directory. Gzip files decompress to the exact frozen bytes. All
checks are explicit and remain active under python -O. This is an arithmetic,
provenance and causality audit, not independent predictive validation.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "reports/timestral-diagnosis-20261026"


def require(value, message):
    if not value:
        raise ValueError(message)


def data(path):
    return (
        path.read_bytes()
        if path.exists()
        else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    )


def sha(path):
    return hashlib.sha256(data(path)).hexdigest()


def read(path):
    return json.loads(data(path))


def rows(path):
    return [json.loads(line) for line in data(path).splitlines()]


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def equal(a, b, label):
    require(
        np.asarray(a).shape == np.asarray(b).shape
        and np.allclose(a, b, rtol=1e-9, atol=1e-12),
        label,
    )


def indexed(records, key="sample_id"):
    values = {r[key]: r for r in records}
    require(len(values) == len(records), "duplicate identities")
    return values


def audit_metrics(group, points, actual, metric, screen, history=False):
    y, p = np.asarray(actual), np.asarray(points)
    require(np.isfinite(y).all() and np.isfinite(p).all(), "nonfinite prediction")
    errors = p - y
    # Use scalar summation, not the production reductions.
    mae = math.fsum(abs(float(e)) for e in errors) / len(y)
    baseline = math.fsum(abs(float(v)) for v in y) / len(y)
    rmse = math.sqrt(math.fsum(float(e) ** 2 for e in errors) / len(y))
    equal(
        [metric["mae"], metric["rmse"], metric["samples"]],
        [mae, rmse, len(y)],
        "point metrics",
    )
    differences = abs(errors) - abs(y)
    months = sorted({r["anchor"][:7] for r in group})
    wins = sum(
        math.fsum(
            float(differences[i])
            for i, r in enumerate(group)
            if r["anchor"][:7] == month
        )
        < 0
        for month in months
    )
    start = (
        utc(group[0]["anchor"]).replace(day=1, hour=0)
        if history
        else utc("2026-01-01T00:00:00Z")
    )
    blocks = [(utc(r["anchor"]) - start).days // 7 for r in group]
    n = max(blocks) + 1
    members = [np.flatnonzero(np.asarray(blocks) == b) for b in range(n)]
    # Expand sampled weekly blocks to actual rows, independent of block-sum implementation.
    choices = np.random.default_rng(20261026).integers(0, n, (4000, n))
    samples = [
        float(differences[np.concatenate([members[b] for b in draw])].mean())
        for draw in choices
    ]
    quantiles = [0.05 / 24, 1 - 0.05 / 24] if history else [0.003125, 0.996875]
    interval = np.quantile(samples, quantiles)
    equal(
        interval,
        screen["interval99_5833" if history else "interval99_375"],
        "weekly bootstrap",
    )
    gain = 1 - mae / baseline
    equal(gain, screen["gain" if history else "mae_improvement"], "MAE gain")
    require(wins == screen["monthly_wins"], "monthly wins")
    checks = [
        gain >= 0.01,
        np.mean(errors * errors) <= np.mean(y * y),
        wins >= (math.ceil(2 * len(months) / 3) if history else 5),
        interval[1] < 0,
    ]
    require(
        all(checks)
        == screen["passes_historical_screen" if history else "passes_replay_screen"],
        "screen decision",
    )
    require(screen["solution_found"] is False, "historical screen misrepresented")


def main():
    frozen = {}
    for family in ("run", "median", "funding", "history"):
        base = HERE / family
        p, r = read(base / "protocol.json"), read(base / "results.json")
        require(
            sha(base / "protocol.json")
            == read(base / "protocol.sha256.json")["sha256"]
            == r["protocol_sha256"],
            "protocol hash",
        )
        require(
            utc(p["registered_at"]) <= utc(read(base / "opened.json")["at"]),
            "registration after execution",
        )
        require(
            all(sha(ROOT / name) == value for name, value in p["code_sha256"].items()),
            "frozen code hash",
        )
        source = HERE / "history-source" if family == "history" else SOURCE
        require(
            all(sha(source / name) == value for name, value in p["inputs"].items()),
            "source hash",
        )
        require(
            all(
                sha(base / name) == value for name, value in r["output_sha256"].items()
            ),
            "output hash",
        )
        frozen[family] = r
    require(
        sha(HERE / "history-plan.json")
        == read(HERE / "history/protocol.json")["plan_sha256"],
        "acquisition plan hash",
    )
    fp = read(HERE / "funding/protocol.json")
    require(
        all(
            sha(HERE / "funding-source" / name) == value
            for name, value in fp["funding_inputs"].items()
        ),
        "funding source hash",
    )
    old_rows = rows(SOURCE / "run/predictions.jsonl")
    old = indexed(old_rows)
    old_outcomes = indexed(rows(SOURCE / "run/outcomes.jsonl"))
    hours = rows(SOURCE / "hours.jsonl")
    hindex = {r["time"]: i for i, r in enumerate(hours)}
    expected = {
        k for k, r in old.items() if utc(r["anchor"]) >= utc("2026-01-01T00:00:00Z")
    }
    require(len(expected) == 1484, "evaluation size")
    groups = defaultdict(list)
    forecasts = rows(HERE / "run/forecasts.jsonl")
    for r in forecasts:
        ref = old[r["sample_id"]]
        i, h = hindex[r["anchor"]], r["horizon_hours"]
        length = int(r["condition"].split("_")[1])
        require(
            hours[i - length]["time"] == r["input_start"]
            and ref["anchor"] == r["anchor"],
            "capacity context",
        )
        if r["condition"] == "5m_256":
            equal(
                r["prediction"]["point"],
                ref["predictions"]["timestral_price"]["point"],
                "5M reproduction",
            )
        groups[(r["condition"], h)].append(r)
    require(len(forecasts) == 5936 and len(groups) == 8, "capacity completeness")
    for (name, h), group in groups.items():
        require(
            {r["sample_id"] for r in group}
            == {k for k in expected if old[k]["horizon_hours"] == h},
            "capacity paired samples",
        )
        result = frozen["run"]["horizons"][str(h)]
        audit_metrics(
            group,
            [r["prediction"]["point"] for r in group],
            [old_outcomes[r["sample_id"]]["actual_log_return"] for r in group],
            result["metrics"][name],
            result["screens"][name],
        )
    settlements = read(HERE / "funding-source/funding.json")
    for family in ("median", "funding"):
        pred = rows(HERE / family / "predictions.jsonl")
        out = indexed(rows(HERE / family / "outcomes.jsonl"))
        require(
            set(indexed(pred)) == set(out) == expected, "calibration sample pairing"
        )
        fits = read(
            HERE / family / ("parameters.json" if family == "median" else "fits.json")
        )
        for r in pred:
            ref = old[r["sample_id"]]
            i, h, anchor = hindex[r["anchor"]], r["horizon_hours"], utc(r["anchor"])
            require(
                "actual" not in r and anchor < utc("2026-08-01T00:00:00Z"),
                "future or joined forecast",
            )
            equal(
                out[r["sample_id"]]["actual"],
                hours[i + h - 1]["log_close"] - hours[i - 1]["log_close"],
                "target from closes",
            )
            state = [
                *(
                    (hours[i - 1]["log_close"] - hours[i - 1 - lag]["log_close"])
                    / r["scale"]
                    for lag in (4, 24, 168)
                ),
                np.mean([x["features"][2] for x in hours[i - 24 : i]]),
                np.mean([x["features"][1] for x in hours[i - 24 : i]])
                - np.mean([x["features"][1] for x in hours[i - 168 : i]]),
                ref["predictions"]["timestral_price"]["point"] / r["scale"],
                ref["predictions"]["timestral_returns"]["point"] / r["scale"],
            ]
            equal(state, r["features"][:7], "causal state features")
            if family == "funding":
                known = [
                    v
                    for v in settlements
                    if utc(v["time"]) + timedelta(hours=1) <= anchor
                ][-3:]
                require(
                    len(known) == 3
                    and known[-1]["time"] == r["latest_funding_settlement"],
                    "settlement availability",
                )
                equal(
                    r["features"][-2:],
                    [
                        known[-1]["rate"] / known[-1]["interval_hours"] * 10000,
                        sum(v["rate"] for v in known)
                        / sum(v["interval_hours"] for v in known)
                        * 10000,
                    ],
                    "funding features",
                )
            else:
                for name, forecast in r["predictions"].items():
                    fit = fits[f"{h}:{r['anchor'][:7]}:{name}"]
                    n = len(fit["mean"])
                    z = np.r_[
                        1.0,
                        np.clip(
                            (np.asarray(r["features"][:n]) - fit["mean"]) / fit["std"],
                            -8,
                            8,
                        ),
                    ]
                    equal(
                        r["scale"] * np.dot(z, fit["beta"]),
                        forecast["point"],
                        "median forecast reconstruction",
                    )
        for key, fit in fits.items():
            h = int(key.split(":" if family == "median" else "-")[0])
            cutoff = utc(fit["cutoff"])
            training = [
                v
                for v in old_rows
                if v["horizon_hours"] == h
                and utc(v["label_end"]) <= cutoff
                and utc(v["anchor"]) >= cutoff - timedelta(days=180)
            ]
            require(
                len(training) == fit["samples"]
                and digest([v["sample_id"] for v in training])
                == fit["sample_ids_sha256"],
                "training membership",
            )
            require(utc(fit["latest_label_end"]) <= cutoff, "unmatured fit")
        for h in (4, 24):
            group = [r for r in pred if r["horizon_hours"] == h]
            result = frozen[family]["horizons"][str(h)]
            for name, screen in result["screens"].items():
                audit_metrics(
                    group,
                    [r["predictions"][name]["point"] for r in group],
                    [out[r["sample_id"]]["actual"] for r in group],
                    result["metrics"][name],
                    screen,
                )
    native = rows(HERE / "history-source/hours.jsonl")
    provenance = read(HERE / "history-source/provenance.json")
    require(
        sha(HERE / "history-source/hours.jsonl") == provenance["hours_sha256"],
        "native source hash",
    )
    times = [utc(r["time"]) for r in native]
    require(
        len(native) == provenance["rows"]
        and all(a < b for a, b in zip(times, times[1:]))
        and times[-1] < utc("2026-08-01T00:00:00Z"),
        "native range/order",
    )
    native_index = {t: i for i, t in enumerate(times)}
    matched = [r for r in native if r["time"] in hindex]
    require(len(matched) == 8016, "source intersection")
    equal(
        [math.log(r["close"]) for r in matched],
        [hours[hindex[r["time"]]]["log_close"] for r in matched],
        "native/1s hourly closes",
    )
    # Independently enumerate complete windows via timestamp membership, not cumulative gap counts.
    eligible = defaultdict(list)
    expected_outcome = {}
    for anchor in times[169:]:
        i = native_index[anchor]
        for h in (4, 24):
            if anchor.hour % h or any(
                anchor + timedelta(hours=offset) not in native_index
                for offset in range(-169, h)
            ):
                continue
            eligible[h].append(anchor)
            expected_outcome[(anchor, h)] = math.log(
                native[native_index[anchor + timedelta(hours=h - 1)]]["close"]
                / native[i - 1]["close"]
            )
    pred = rows(HERE / "history/predictions.jsonl")
    out = indexed(rows(HERE / "history/outcomes.jsonl"))
    require(set(indexed(pred)) == set(out), "history pairing")
    fits = read(HERE / "history/fits.json")
    for key, fit in fits.items():
        _, _, h, days = key.split("-")
        h, days, cutoff = int(h), int(days), utc(fit["cutoff"])
        training = [
            t
            for t in eligible[h]
            if cutoff - timedelta(days=days) <= t and t + timedelta(hours=h) <= cutoff
        ]
        require(
            len(training) == fit["samples"]
            and digest([t.isoformat().replace("+00:00", "Z") for t in training])
            == fit["training_anchors_sha256"],
            "native training membership",
        )
        require(utc(fit["latest_label_end"]) <= cutoff, "native future fit")
    groups = defaultdict(list)
    for r in pred:
        anchor, h = utc(r["anchor"]), r["horizon_hours"]
        require(
            "actual" not in r and utc(fits[r["fit_key"]]["cutoff"]) <= anchor,
            "history causal fit",
        )
        equal(
            out[r["sample_id"]]["actual"],
            expected_outcome[(anchor, h)],
            "history target",
        )
        context = [
            native[native_index[anchor - timedelta(hours=offset)]]
            for offset in range(169, 0, -1)
        ]
        logs = [math.log(v["close"]) for v in context]
        returns = [b - a for a, b in zip(logs, logs[1:])]
        vol24 = math.sqrt(math.fsum(v * v for v in returns[-24:]))
        vol168 = math.sqrt(math.fsum(v * v for v in returns))
        scale = max(vol24 * math.sqrt(h / 24), 1e-6)
        total = math.fsum(v["volume"] for v in context[-24:])
        features = [
            *((logs[-1] - logs[-1 - lag]) / scale for lag in (4, 24, 168)),
            vol24 / max(vol168 / math.sqrt(7), 1e-6),
            math.log(context[-1]["high"] / context[-1]["low"]) / scale,
            math.log1p(total / 24)
            - math.log1p(math.fsum(v["volume"] for v in context[-168:]) / 168),
            2 * math.fsum(v["taker"] for v in context[-24:]) / total - 1
            if total
            else 0.0,
            math.sin(2 * math.pi * anchor.hour / 24),
            math.cos(2 * math.pi * anchor.hour / 24),
            math.sin(2 * math.pi * anchor.weekday() / 7),
            math.cos(2 * math.pi * anchor.weekday() / 7),
        ]
        equal(scale, r["scale"], "native scale")
        equal(features, r["features"], "native features")
        groups[(r["cohort"], str(h), str(r["window_days"]))].append(r)
    for (year, h, days), group in groups.items():
        first = utc(year + "-01-01T00:00:00Z")
        last = (
            utc("2026-08-01T00:00:00Z")
            if year == "2026"
            else utc(str(int(year) + 1) + "-01-01T00:00:00Z")
        )
        require(
            {utc(r["anchor"]) for r in group}
            == {t for t in eligible[int(h)] if first <= t < last},
            "complete historical cohort",
        )
        m = frozen["history"]["cohorts"][year][h][days]
        audit_metrics(
            group,
            [r["point"] for r in group],
            [out[r["sample_id"]]["actual"] for r in group],
            m,
            m,
            True,
        )
    require(len(groups) == 12, "all historical comparisons")
    return {
        "verified": True,
        "capacity_forecasts": len(forecasts),
        "calibration_anchors_per_family": len(expected),
        "history_forecasts": len(pred),
        "native_hours": len(native),
        "native_vs_1s_matching_hours": 8016,
        "metric_comparisons": 30,
        "inference_rerun": False,
        "independent_predictive_validation": False,
        "scope": "frozen hashes, exact sample sets, past-only features/training, target endpoints, point metrics, weekly bootstrap, median prediction reconstruction; neural/tree inference not replayed",
    }


def audit_local(root):
    """Check retained acquisition bytes and model hashes, never load pickles."""
    counts = {"raw_archives": 0, "model_files": 0, "invalid_native_rows": 0}
    for family in ("funding-source", "history-source"):
        provenance = read(HERE / family / "provenance.json")
        reconstructed = []
        for source in provenance["sources"]:
            path = root / family / source["url"].rsplit("/", 1)[-1]
            require(
                sha(path)
                == source["sha256"]
                == Path(str(path) + ".CHECKSUM").read_text().split()[0],
                "raw archive checksum",
            )
            with zipfile.ZipFile(path) as archive:
                require(len(archive.namelist()) == 1, "archive member count")
                content = archive.read(archive.namelist()[0]).decode()
            counts["raw_archives"] += 1
            if family == "funding-source":
                for row in csv.DictReader(io.StringIO(content)):
                    t = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
                        milliseconds=int(row["calc_time"])
                    )
                    reconstructed.append(
                        {
                            "time": t.isoformat().replace("+00:00", "Z"),
                            "interval_hours": float(row["funding_interval_hours"]),
                            "rate": float(row["last_funding_rate"]),
                        }
                    )
            else:
                invalid = []
                for values in csv.reader(io.StringIO(content)):
                    if values[0] == "open_time":
                        continue
                    opened, closed = [int(values[i]) for i in (0, 6)]
                    unit = 1_000_000 if opened >= 10**15 else 1000
                    t = datetime.fromtimestamp(opened / unit, timezone.utc)
                    o, hi, lo, close, volume = map(float, values[1:6])
                    trades, taker = float(values[8]), float(values[9])
                    stamp = t.isoformat().replace("+00:00", "Z")
                    valid = (
                        opened <= closed < opened + 3600 * unit
                        and all(
                            math.isfinite(x)
                            for x in (o, hi, lo, close, volume, trades, taker)
                        )
                        and 0 < lo <= min(o, close) <= max(o, close) <= hi
                        and min(volume, trades, taker) >= 0
                        and taker <= volume + 1e-8
                    )
                    if not valid:
                        invalid.append({"time": stamp, "raw": values})
                        counts["invalid_native_rows"] += 1
                        continue
                    reconstructed.append(
                        {
                            "time": stamp,
                            "close": close,
                            "high": hi,
                            "low": lo,
                            "volume": volume,
                            "trades": trades,
                            "taker": taker,
                        }
                    )
                require(
                    invalid == source["invalid_candles"], "source anomaly accounting"
                )
        expected = (
            read(HERE / family / "funding.json")
            if family == "funding-source"
            else rows(HERE / family / "hours.jsonl")
        )
        require(reconstructed == expected, "raw/normalized source reconstruction")
    for family in ("funding", "history"):
        for key, fit in read(HERE / family / "fits.json").items():
            require(
                sha(root / family / "models" / (key + ".pkl")) == fit["model_sha256"],
                "retained model hash",
            )
            counts["model_files"] += 1
    p = read(HERE / "run/protocol.json")
    cache = ROOT / "research/out/timestral-pilot-20261025/model-cache"
    for size, spec in p["model_specifications"].items():
        path = (
            cache
            / ("models--" + spec["id"].replace("/", "--"))
            / "snapshots"
            / spec["revision"]
        )
        require(
            all(
                sha(path / name) == value
                for name, value in p["checkpoints"][size].items()
            ),
            "retained checkpoint hash",
        )
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-artifacts", type=Path)
    args = parser.parse_args()
    result = main()
    if args.local_artifacts:
        result["retained_sources_and_models"] = audit_local(args.local_artifacts)
    print(json.dumps(result, indent=2))
