#!/usr/bin/env python3
"""Independent saved sign/variance audit: no experiment imports or fitting."""

from collections import defaultdict
from datetime import datetime, timedelta
import argparse
import csv
import gzip
import hashlib
import json
import io
import math
from pathlib import Path
import zipfile

import numpy as np
from scipy.special import digamma, gammainc, gammaincc, expit

ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def raw(path):
    return (
        path.read_bytes()
        if path.exists()
        else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    )


def read(path):
    return json.loads(raw(path))


def lines(path):
    return [json.loads(x) for x in raw(path).splitlines()]


def sha(path):
    return hashlib.sha256(raw(path)).hexdigest()


def utc(t):
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def stamp(t):
    return t.isoformat().replace("+00:00", "Z")


def digest(x):
    return hashlib.sha256(
        json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def same(a, b, why):
    require(
        np.asarray(a).shape == np.asarray(b).shape
        and np.isfinite(a).all()
        and np.isfinite(b).all()
        and np.allclose(a, b, rtol=1e-8, atol=2e-11),
        why,
    )


def reconstruct(hours):
    times = [utc(r["time"]) for r in hours]
    require(
        all(t.minute == t.second == t.microsecond == 0 for t in times)
        and all(a < b for a, b in zip(times, times[1:]))
        and times[-1] < utc("2026-08-01T00:00:00Z"),
        "source grid/protected dates",
    )
    logs = np.log([r["close"] for r in hours])
    rows = {}
    for i in range(721, len(hours)):
        t = times[i]
        for h in (4, 24):
            if (
                t.hour % h
                or i + h > len(hours)
                or times[i + h - 1] - times[i - 721] != timedelta(hours=720 + h)
            ):
                continue
            context = hours[i - 721 : i]
            past = np.diff(logs[i - 721 : i])
            squares = past**2
            vol24 = math.sqrt(math.fsum(squares[-24:]))
            vol168 = math.sqrt(math.fsum(squares[-168:]))
            scale = max(vol24 * math.sqrt(h / 24), 1e-6)
            vols = [max(math.fsum(squares[-n:]) / n, 1e-12) for n in (24, 168, 720)]
            volume = math.fsum(r["volume"] for r in context[-24:])
            volume168 = math.fsum(r["volume"] for r in context[-168:])
            state = [
                *((logs[i - 1] - logs[i - 1 - lag]) / scale for lag in (4, 24, 168)),
                vol24 / max(vol168 / math.sqrt(7), 1e-6),
                math.log(context[-1]["high"] / context[-1]["low"]) / scale,
                math.log1p(volume / 24) - math.log1p(volume168 / 168),
                2 * math.fsum(r["taker"] for r in context[-24:]) / volume - 1
                if volume
                else 0.0,
                math.sin(2 * math.pi * t.hour / 24),
                math.cos(2 * math.pi * t.hour / 24),
                math.sin(2 * math.pi * t.weekday() / 7),
                math.cos(2 * math.pi * t.weekday() / 7),
            ]
            weights = [2 ** (-age / 24) for age in range(719, -1, -1)]
            ewma = h * max(
                math.fsum(v * w for v, w in zip(squares, weights)) / math.fsum(weights),
                1e-12,
            )
            future = np.diff(logs[i - 1 : i + h])
            identity = f"{stamp(t)}:{h}"
            rows[identity] = {
                "sample_id": identity,
                "anchor": stamp(t),
                "label_end": stamp(t + timedelta(hours=h)),
                "horizon_hours": h,
                "state": state,
                "variance_features": np.log(vols).tolist(),
                "variance_persistence": h * vols[0],
                "variance_ewma": ewma,
                "return": float(logs[i + h - 1] - logs[i - 1]),
                "variance": max(math.fsum(v * v for v in future), 1e-12),
            }
    return rows


def verify_fits(fits, source):
    gradients = []
    for key, p in fits.items():
        h = 4 if key.startswith(("ETH", "SOL")) else int(key.split(":")[-1])
        cutoff = utc(p["cutoff"])
        train = [
            r
            for r in source.values()
            if r["horizon_hours"] == h
            and utc(r["label_end"]) <= cutoff
            and utc(r["anchor"]) >= cutoff - timedelta(days=1095)
        ]
        require(
            len(train) == p["samples"]
            and digest([r["sample_id"] for r in train]) == p["sample_ids_sha256"],
            "training membership",
        )
        require(utc(p["latest_label_end"]) <= cutoff, "future training label")
        for kind, column in [("direction", "state"), ("variance", "variance_features")]:
            group = (
                [r for r in train if r["return"] != 0] if kind == "direction" else train
            )
            f = p[kind]
            x = np.asarray([r[column] for r in group])
            y = np.asarray(
                [
                    int(r["return"] > 0) if kind == "direction" else r["variance"]
                    for r in group
                ]
            )
            same(x.mean(axis=0), f["mean"], "training normalization mean")
            same(
                np.maximum(x.std(axis=0), 1e-8),
                f["scale"],
                "training normalization scale",
            )
            z = np.clip((x - f["mean"]) / f["scale"], -8, 8)
            eta = z @ f["coef"] + f["intercept"]
            if kind == "direction":
                residual = expit(eta) - y
                gradient = z.T @ residual / len(group) + np.asarray(f["coef"]) / (
                    0.1 * len(group)
                )
                same(
                    f["baseline"], (y.sum() + 1) / (len(y) + 2), "training class prior"
                )
            else:
                same(f["unit"], y.mean(), "variance units")
                residual = 1 - y / (f["unit"] * np.exp(eta))
                gradient = z.T @ residual / len(group) + 0.01 * np.asarray(f["coef"])
            error = max(abs(float(residual.mean())), float(np.abs(gradient).max()))
            require(error < 1e-5, "fitted objective gradient not stationary")
            gradients.append(error)
    return max(gradients)


def predicted(p, row, kind):
    f = p[kind]
    x = np.asarray(row["state" if kind == "direction" else "variance_features"])
    z = np.clip((x - f["mean"]) / f["scale"], -8, 8)
    eta = float(z @ f["coef"] + f["intercept"])
    return float(expit(eta)) if kind == "direction" else f["unit"] * math.exp(eta)


def bootstrap(rows, differences, level):
    start = utc(rows[0]["anchor"]).replace(day=1, hour=0)
    blocks = np.asarray([(utc(r["anchor"]) - start).days // 7 for r in rows])
    n = int(blocks.max() + 1)
    members = [np.flatnonzero(blocks == i) for i in range(n)]
    draws = np.random.default_rng(20261027).integers(0, n, (4000, n))
    values = [
        float(differences[np.concatenate([members[b] for b in draw])].mean())
        for draw in draws
    ]
    return np.quantile(values, [0.05 / level, 1 - 0.05 / level])


def screen(rows, model_loss, base_loss, secondary, secondary_base, saved, level):
    diff = model_loss - base_loss
    gain = 1 - model_loss.mean() / base_loss.mean()
    months = sorted({r["anchor"][:7] for r in rows})
    wins = sum(
        math.fsum(
            float(diff[i]) for i, r in enumerate(rows) if r["anchor"][:7] == month
        )
        < 0
        for month in months
    )
    interval = bootstrap(rows, diff, level)
    same(gain, saved["gain"], "score gain")
    same(
        interval,
        saved["interval99_791667" if level == 48 else "interval99_583333"],
        "weekly bootstrap",
    )
    require(wins == saved["monthly_wins"], "monthly wins")
    passed = (
        gain >= 0.01
        and secondary <= secondary_base
        and wins >= math.ceil(2 * len(months) / 3)
        and interval[1] < 0
    )
    require(bool(passed) == saved["passes"], "screen decision")


def qlike(actual, mean):
    return actual / mean - np.log(actual / mean) - 1


def verify_bundle(base, source_path, archived_code=None):
    p, r = read(base / "protocol.json"), read(base / "results.json")
    require(
        sha(base / "protocol.json")
        == read(base / "protocol.sha256.json")["sha256"]
        == r["protocol_sha256"],
        "protocol hash",
    )
    require(
        utc(p["registered_at"]) <= utc(read(base / "opened.json")["at"]),
        "registration chronology",
    )
    for name, value in p["code_sha256"].items():
        target = (
            archived_code
            if archived_code and name == "scripts/run_variance_transfer.py"
            else ROOT / name
        )
        require(sha(target) == value, "code hash: " + name)
    require(
        all(sha(base / name) == value for name, value in r["output_sha256"].items()),
        "saved output hashes",
    )
    if "plan_sha256" in p:
        require(sha(base / "plan.json") == p["plan_sha256"], "plan hash")
    if "source_sha256" in p:
        require(sha(source_path) == p["source_sha256"], "source hash")
    return p, r


def check_calibration(calibration, fit, source):
    cutoff = utc(fit["cutoff"])
    # The caller supplies only the appropriate horizon here.
    train = [
        r
        for r in source.values()
        if utc(r["label_end"]) <= cutoff
        and utc(r["anchor"]) >= cutoff - timedelta(days=1095)
    ]
    require(
        len(train) == calibration["samples"] == fit["samples"],
        "calibration sample count",
    )
    for name, c in calibration["parameters"].items():
        raw_pred = np.asarray(
            [
                predicted(fit, r, "variance")
                if name == "gamma"
                else r["variance_" + name]
                for r in train
            ]
        )
        ratio = np.asarray([r["variance"] for r in train]) / raw_pred
        same(c["bias"], ratio.mean(), "mean calibration")
        same(
            c["log_shift"], np.log(ratio / c["bias"]).mean(), "expected-log calibration"
        )
        same(
            c["log_shift"],
            digamma(c["shape"]) - math.log(c["shape"]),
            "Gamma shape equation",
        )
        same(
            [
                gammainc(c["shape"], c["shape"] * c["lower_ratio"]),
                gammainc(c["shape"], c["shape"] * c["upper_ratio"]),
            ],
            [0.05, 0.95],
            "Gamma quantiles",
        )


def audit(report):
    btc_source = (
        ROOT / "reports/timestral-capacity-20261026/history-source/hours.jsonl.gz"
    )
    sources = {
        "BTCUSDT": reconstruct(
            [
                json.loads(x)
                for x in gzip.decompress(btc_source.read_bytes()).splitlines()
            ]
        )
    }
    p, first = verify_bundle(report / "decomposition", btc_source)
    fits = read(report / "decomposition/parameters.json")
    gradient = verify_fits(fits, sources["BTCUSDT"])
    predictions = lines(report / "decomposition/predictions.jsonl")
    outcomes = {
        r["sample_id"]: r for r in lines(report / "decomposition/outcomes.jsonl")
    }
    require(
        len(outcomes) == len(predictions) == len({r["sample_id"] for r in predictions}),
        "duplicate predictions",
    )
    expected = {
        k
        for k, r in sources["BTCUSDT"].items()
        if "2024-01-01" <= r["anchor"] < "2026-08-01"
    }
    require(set(outcomes) == expected, "complete BTC evaluation set")
    for r in predictions:
        ref = sources["BTCUSDT"][r["sample_id"]]
        fit = fits[r["fit_key"]]
        require(
            utc(fit["cutoff"]) <= utc(r["anchor"])
            and "return" not in r
            and "variance" not in r,
            "forecast causality",
        )
        for key in (
            "state",
            "variance_features",
            "variance_persistence",
            "variance_ewma",
        ):
            same(r[key], ref[key], "origin inputs")
        for key in ("return", "variance"):
            same(outcomes[r["sample_id"]][key], ref[key], "target endpoint")
        for kind in ("direction", "variance"):
            same(
                r["predictions"][kind],
                predicted(fit, ref, kind),
                "coefficient prediction",
            )
        same(
            [r["predictions"]["direction_prior"], r["predictions"]["variance_mean"]],
            [fit["direction"]["baseline"], fit["variance"]["baseline"]],
            "baseline fit",
        )
    for year, hs in first["cohorts"].items():
        for h, tasks in hs.items():
            for kind, scores in tasks.items():
                group = [
                    r
                    for r in predictions
                    if r["cohort"] == year
                    and r["horizon_hours"] == int(h)
                    and (kind != "direction" or outcomes[r["sample_id"]]["return"] != 0)
                ]
                y = np.asarray(
                    [
                        int(outcomes[r["sample_id"]]["return"] > 0)
                        if kind == "direction"
                        else outcomes[r["sample_id"]]["variance"]
                        for r in group
                    ]
                )
                losses = {}
                secondary = {}
                for name, m in scores["metrics"].items():
                    value = np.asarray([r["predictions"][name] for r in group])
                    losses[name] = (
                        (value - y) ** 2 if kind == "direction" else qlike(y, value)
                    )
                    secondary[name] = (
                        float(
                            np.mean(-(y * np.log(value) + (1 - y) * np.log1p(-value)))
                        )
                        if kind == "direction"
                        else float(np.mean((np.log(y) - np.log(value)) ** 2))
                    )
                    same(
                        [m["primary"], m["secondary"]],
                        [losses[name].mean(), secondary[name]],
                        "first scores",
                    )
                for name, s in scores["screens"].items():
                    screen(
                        group,
                        losses[kind],
                        losses[name],
                        secondary[kind],
                        secondary[name],
                        s,
                        48,
                    )
    comparison_count = 24
    distribution_count = 0
    for family in ("distribution", "transfer"):
        base = report / family
        if not base.exists():
            continue
        archived = base / "original-run_variance_transfer.py"
        p, result = verify_bundle(
            base, btc_source, archived if archived.exists() else None
        )
        if family == "distribution":
            require(
                p["prior_parameters_sha256"]
                == sha(report / "decomposition/parameters.json"),
                "frozen mean fits",
            )
            family_fits = fits
        else:
            original = archived.read_text()
            require(
                (ROOT / "scripts/run_variance_transfer.py").read_text()
                == original.replace("    p = verify(root)\n", "    verify(root)\n"),
                "transfer wrapper must differ only by documented unused-variable removal",
            )
            acquisition = read(base / "acquisition.json")
            require(
                sha(base / "acquisition.json") == result["acquisition_sha256"],
                "acquisition hash",
            )
            require(
                utc(p["registered_at"])
                <= utc(acquisition["completed_at"])
                <= utc(read(base / "opened.json")["at"]),
                "transfer chronology",
            )
            for symbol, hashes in acquisition["sources"].items():
                require(
                    all(
                        sha(base / "sources" / symbol / name) == value
                        for name, value in hashes.items()
                    ),
                    "transfer source hashes",
                )
                sources[symbol] = reconstruct(
                    [
                        json.loads(x)
                        for x in gzip.decompress(
                            (base / "sources" / symbol / "hours.jsonl.gz").read_bytes()
                        ).splitlines()
                    ]
                )
            family_fits = read(base / "parameters.json")
            for symbol in ("ETHUSDT", "SOLUSDT"):
                gradient = max(
                    gradient,
                    verify_fits(
                        {k: v for k, v in family_fits.items() if k.startswith(symbol)},
                        sources[symbol],
                    ),
                )
        calibration = read(base / "calibrations.json")
        for key, c in calibration.items():
            symbol = key.split(":")[0] if family == "transfer" else "BTCUSDT"
            h = 4 if family == "transfer" else int(key.split(":")[-1])
            check_calibration(
                c,
                family_fits[key],
                {k: r for k, r in sources[symbol].items() if r["horizon_hours"] == h},
            )
        pred = lines(base / "predictions.jsonl")
        out = {r["sample_id"]: r for r in lines(base / "outcomes.jsonl")}
        groups = defaultdict(list)
        require(
            len(pred) == len(out) == len({r["sample_id"] for r in pred}),
            "distribution pairing",
        )
        for r in pred:
            symbol = r.get("symbol", "BTCUSDT")
            identity = (
                r["sample_id"][len(symbol) + 1 :]
                if family == "transfer"
                else r["sample_id"]
            )
            ref = sources[symbol][identity]
            require(
                utc(family_fits[r["fit_key"]]["cutoff"]) <= utc(r["anchor"])
                and "variance" not in r,
                "distribution causality",
            )
            same(out[r["sample_id"]]["variance"], ref["variance"], "variance target")
            for name, c in calibration[r["fit_key"]]["parameters"].items():
                raw_mean = (
                    predicted(family_fits[r["fit_key"]], ref, "variance")
                    if name == "gamma"
                    else ref["variance_" + name]
                )
                mean = raw_mean * c["bias"]
                saved = r["distribution"][name]
                same(
                    [
                        saved["mean"],
                        saved["expected_log"],
                        saved["lower90"],
                        saved["upper90"],
                        saved["probability_above_recent"],
                    ],
                    [
                        mean,
                        math.log(mean) + c["log_shift"],
                        mean * c["lower_ratio"],
                        mean * c["upper_ratio"],
                        gammaincc(
                            c["shape"], ref["variance_persistence"] * c["shape"] / mean
                        ),
                    ],
                    "distribution reconstruction",
                )
            groups[(symbol, r["cohort"], str(r["horizon_hours"]))].append(r)
        for (symbol, year, h), group in groups.items():
            expected = {
                k
                for k, r in sources[symbol].items()
                if r["anchor"].startswith(year)
                and r["anchor"] < "2026-08-01"
                and r["horizon_hours"] == int(h)
            }
            observed = {
                r["sample_id"][len(symbol) + 1 :]
                if family == "transfer"
                else r["sample_id"]
                for r in group
            }
            require(observed == expected, "complete asset cohort")
            saved = (
                result["cohorts"][year][h]
                if family == "distribution"
                else result["assets"][symbol][year]
            )
            y = np.asarray([out[r["sample_id"]]["variance"] for r in group])
            losses = {}
            secondary = {}
            for name, m in saved["metrics"].items():
                f = [r["distribution"][name] for r in group]
                mean = np.asarray([r["mean"] for r in f])
                elog = np.asarray([r["expected_log"] for r in f])
                lower = np.asarray([r["lower90"] for r in f])
                upper = np.asarray([r["upper90"] for r in f])
                prob = np.asarray([r["probability_above_recent"] for r in f])
                threshold = np.asarray([r["variance_persistence"] for r in group])
                losses[name] = qlike(y, mean)
                secondary[name] = float(np.mean((np.log(y) - elog) ** 2))
                interval = (
                    upper
                    - lower
                    + 20 * np.maximum(lower - y, 0)
                    + 20 * np.maximum(y - upper, 0)
                )
                same(
                    [
                        m["qlike"],
                        m["expected_log_mse"],
                        m["coverage90"],
                        m["interval_score_scaled"],
                        m["event_brier"],
                    ],
                    [
                        losses[name].mean(),
                        secondary[name],
                        np.mean((y >= lower) & (y <= upper)),
                        np.mean(interval / threshold),
                        np.mean((prob - (y > threshold)) ** 2),
                    ],
                    "distribution scores",
                )
            for name, s in saved["screens"].items():
                screen(
                    group,
                    losses["gamma"],
                    losses[name],
                    secondary["gamma"],
                    secondary[name],
                    s,
                    24,
                )
                comparison_count += 1
            require(
                saved["passes_both_baselines"]
                == all(s["passes"] for s in saved["screens"].values()),
                "combined baseline decision",
            )
        distribution_count += len(pred)
        if family == "distribution":
            expected_pass = {
                h: all(
                    result["cohorts"][y][h]["passes_both_baselines"]
                    for y in result["cohorts"]
                )
                for h in ("4", "24")
            }
            require(
                expected_pass == result["passes_all_cohorts"], "overall BTC decision"
            )
        else:
            require(
                all(
                    v["passes_both_baselines"]
                    for years in result["assets"].values()
                    for v in years.values()
                )
                == result["passes_all_assets_and_cohorts"],
                "overall transfer decision",
            )
    return {
        "verified": True,
        "primary_comparisons": comparison_count,
        "btc_decomposition_forecasts": len(predictions),
        "distribution_forecasts": distribution_count,
        "maximum_objective_gradient_error": gradient,
        "source_assets": list(sources),
        "inference_or_fit_rerun": False,
        "scope": "Source reconstruction, exact training/evaluation membership, optimizer stationarity, forecasts from coefficients, calibration, target endpoints, metrics and expanded weekly bootstrap. Not prospective/trading validation.",
    }


def raw_transfer(report, root):
    count, excluded = 0, 0
    for symbol in ("ETHUSDT", "SOLUSDT"):
        base = report / "transfer/sources" / symbol
        provenance = read(base / "provenance.json")
        reconstructed = []
        for source in provenance["files"]:
            path = root / "sources" / symbol / source["url"].rsplit("/", 1)[-1]
            require(
                sha(path)
                == source["sha256"]
                == Path(str(path) + ".CHECKSUM").read_text().split()[0],
                "raw transfer archive checksum",
            )
            with zipfile.ZipFile(path) as archive:
                require(len(archive.namelist()) == 1, "archive member count")
                content = archive.read(archive.namelist()[0]).decode()
            invalid = []
            for values in csv.reader(io.StringIO(content)):
                if values[0] == "open_time":
                    continue
                opened, closed = int(values[0]), int(values[6])
                unit = 1_000_000 if opened >= 10**15 else 1000
                t = utc("1970-01-01T00:00:00Z") + timedelta(seconds=opened / unit)
                o, high, low, close, volume = map(float, values[1:6])
                trades, taker = float(values[8]), float(values[9])
                valid = (
                    opened <= closed < opened + 3600 * unit
                    and all(
                        math.isfinite(v)
                        for v in (o, high, low, close, volume, trades, taker)
                    )
                    and 0 < low <= min(o, close) <= max(o, close) <= high
                    and min(volume, trades, taker) >= 0
                    and taker <= volume + 1e-8
                )
                if not valid:
                    invalid.append(values)
                    continue
                reconstructed.append(
                    {
                        "time": stamp(t),
                        "close": close,
                        "high": high,
                        "low": low,
                        "volume": volume,
                        "trades": trades,
                        "taker": taker,
                    }
                )
            require(invalid == source["invalid_rows"], "transfer anomaly accounting")
            count += 1
            excluded += len(invalid)
        expected = [
            json.loads(x)
            for x in gzip.decompress(
                (base / "hours.jsonl.gz").read_bytes()
            ).splitlines()
        ]
        require(reconstructed == expected, "raw transfer normalization")
    return {"archives": count, "excluded_malformed_rows": excluded}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/timestral-decomposition-20261027"
    )
    parser.add_argument("--raw-transfer", type=Path)
    args = parser.parse_args()
    result = audit(args.report)
    if args.raw_transfer:
        result["raw_transfer_audit"] = raw_transfer(args.report, args.raw_transfer)
    print(json.dumps(result, indent=2))
