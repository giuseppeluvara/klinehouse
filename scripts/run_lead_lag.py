#!/usr/bin/env python3
"""Preregistered information in traded perpetual prices relative to spot."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_universal as shared  # noqa: E402
from research import timestral_volume_center as center  # noqa: E402
from research.timestral_futures_flow import combine  # noqa: E402
from scripts import run_volume_center as previous  # noqa: E402
from scripts.run_futures_flow import (  # noqa: E402
    ROOT,
    RAW,
    digest,
    write,
    epoch,
    sources,
    prior_path,
    requests_list,
    url,
    decode,
    examples,
)
from scripts.run_flat_repair import outcome_path  # noqa: E402
from scripts.run_temporal_adaptation import paired_metrics  # noqa: E402

DEFAULT = ROOT / "reports/timestral-lead-lag-20261028"
ARMS = ("price_control", "lead_lag")
PAIRS = (
    ("price_control", "baseline"),
    ("lead_lag", "baseline"),
    ("lead_lag", "price_control"),
)
CODE = tuple(
    dict.fromkeys(
        (
            "research/timestral_lead_lag.py",
            "scripts/run_lead_lag.py",
            "scripts/verify_lead_lag.py",
            "tests/test_timestral_lead_lag.py",
            "scripts/run_temporal_adaptation.py",
            "scripts/verify_temporal_adaptation.py",
            "scripts/verify_universal_direction.py",
            "scripts/verify_futures_flow.py",
            "scripts/verify_volume_center.py",
            "scripts/verify_flat_repair.py",
            *previous.CODE,
        )
    )
)
PLAN = {
    "purpose": "Test NEW traded-perpetual price information, rather than changing weights among old experts. The earlier futures-flow study explicitly excluded futures prices; premium-index information was different. No open-interest/position metrics downloaded or scored: source chosen on timestamp integrity before candidate scores.",
    "status": "Sequential retrospective research on already explored Jan-Jul2024/2025/2026 BTC/ETH/SOL, not full years, pristine validation or prospective performance. New candidate scores opened only after source preparation and protocol/code hashes frozen.",
    "sources": "Reuse90 locally cached official USD-M monthly1m trade-kline ZIPs from Oct(previousyear) through July, checking frozen SHA256 and cached official checksums. Decode candle timestamp/OHLC/volume validity; keep actual close price, not premium/index/mark. No network, no downloads. Source-derived arrays retain gaps without interpolation.",
    "timing": "Latest included candle OPEN=anchor-120sec and CLOSE=anchor-61sec: one full minute after closure before anchor. Futures61 contiguous closes and spot62 contiguous closes required. Target remains spot return anchor+60 toanchor+360sec. Exchange archive revisions/publication vintage remain unverified; declared delay is conditional replay timing, not point-in-time certification.",
    "features": "Horizon5 and60minutes fixed. Price-control raw features=own spot logreturn_h/(prior60minute spot-return RMS*sqrt(h)); lead-lag raw features=(own perpetual logreturn_h-own spot logreturn_h)/same denominator. RMS uses returns at indices j-60 through j-1, excludes latestreturn j, floor1e-8. Rawclip+-8. No lead-lag coefficient sign imposed, no assetIDs/intercept or new cross-asset weighting.",
    "baseline": "Freeze each existing monthly uncertainty-shrunk uniform-center beta; apply its existing causal uniform signal. Recombine ALL arms with exactly mixed_flat=(frozen cross/probability flat+pastfrequency flat)/2. Baseline evaluation probabilities equal prior flat-repair mixed model. New feature arms only modify conditional up/down logit, holding flat probability fixed.",
    "fit": "Monthly90day matured nonflat pooled3asset samples, identical for both arms, intersect prior futures availability with new context availability across all3assets. Keep original baseline beta trained on original membership, no refit. Offsetlogistic two coefficients each, equal UTCday weights, weighted trainRMS perfeature floor1e-8, normalize thenclip+-3. Minimize mean weighted binary logloss +0.25/2*sum(beta^2). Fixed ridge, no tuning/sweep,42fits. Gradient tolerance1e-7, >=80 distinct training days. Offset computed using current monthly baseline coefficient on those past rows; not an out-of-fold stack.",
    "gaps": "Both corrections fall back exactly to baseline for any missing new context across all3assets; all550362 original evaluation anchors retained. Training missing contexts excluded identically, never use outcomes to choose prediction membership. Evaluate all original nonflat labels only for conditional metrics; future flats cannot be known operationally.",
    "comparison": "Primary lead_lag versus baseline AND matched two-coefficient price_control. Price_control_vs_baseline diagnostic. Each comparison full3class Brier(sum squared), natural logloss; conditional moving single-square Brier, binary logloss and accuracy(tiesup). Also report fullscores on fixed baseline52percent selected subset as diagnostic. Original unconditional gates unchanged for both arms and baseline.",
    "uncertainty": "12000 paired7day moving blocks seed271032; quantiles1/1080,1079/1080 inherited unchanged (conservative Bonferroni27 comparisons versus18 primary pair/cohorts). Per fixed endpoint/scope, not simultaneous across endpoints, scopes or prior sequential studies. No bootstrap refit. Each paired gate requires ALLmetric lowergainbounds>0 and>=5positive months for EVERYmetric.",
    "success": "Stable improvement only if lead_lag passes all9 original gates AND all18 primary full-all-anchor gates AND all18 primary conditional-direction gates against bothcontrols. Fixed selected subset diagnostic not an extra success criterion. Do not promote price_control or redefine success after observing results; report every cohort, all results and all failures.",
    "protected": "No source/outcome from2026-08-01 onward. Preserve all old artifacts/code/userfiles/credentials. Small new arrays using existing environment; no orders or deployment. A verified replay or positive historical score is not universal superiority or profitability.",
    "references": [
        "https://github.com/binance/binance-public-data",
        "https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data",
        "https://github.com/binance/binance-public-data/issues/491",
    ],
}


def prepare(root):
    """Extract new price columns from existing checksummed sources, no outcomes."""
    root.mkdir(parents=True, exist_ok=True)
    old = json.loads((previous.FUTURES / "sources.json").read_text())
    known = {r["url"]: r for r in old["sources"]}
    pieces, receipts = {}, []
    for cohort, symbol, year, month in requests_list():
        address = url(symbol, year, month)
        path = RAW / address.rsplit("/", 1)[-1]
        sha = digest(path)
        if (
            sha != known[address]["sha256"]
            or sha != path.with_suffix(".zip.CHECKSUM").read_text().split()[0]
        ):
            raise ValueError("frozen or official raw checksum differs")
        rows = []
        with zipfile.ZipFile(path) as z:
            if len(z.infolist()) != 1 or z.infolist()[0].file_size > 32_000_000:
                raise ValueError("raw ZIP expansion guard")
            with z.open(z.namelist()[0]) as f:
                for row in csv.reader(io.TextIOWrapper(f)):
                    if row[0] == "open_time":
                        continue
                    t, _, _, _ = decode(row, year, month)
                    rows.append((t, float(row[4])))
        a = np.asarray(rows)
        if len(a) != known[address]["rows"] or np.any(np.diff(a[:, 0]) <= 0):
            raise ValueError("raw membership changed")
        pieces.setdefault(f"{cohort}-{symbol}", []).append(a)
        receipts.append(
            dict(url=address, sha256=sha, rows=len(a), bytes=path.stat().st_size)
        )
        if len(receipts) % 10 == 0:
            print("verified cached price ZIPs", len(receipts), "/90", flush=True)
    arrays = {}
    for key, rows in pieces.items():
        a = np.concatenate(rows)
        path = root / f"{key}-prices.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, time=a[:, 0].astype(np.int64), close=a[:, 1])
        arrays[path.name] = digest(path)
    write(
        root / "sources.json",
        dict(
            at=datetime.now(timezone.utc).isoformat(), sources=receipts, arrays=arrays
        ),
    )


def input_paths(root):
    paths = [
        previous.DEFAULT / "fits.json",
        previous.FUTURES / "sources.json",
        root / "sources.json",
    ]
    for year in (2024, 2025, 2026):
        paths.extend(sources(year).values())
        for s in shared.SYMBOLS:
            paths.extend(
                [
                    prior_path(year, s),
                    outcome_path(year, s),
                    previous.FUTURES / f"{year}-{s}-futures.npz",
                    previous.DEFAULT / f"{year}-{s}-predictions.npz",
                    root / f"{year}-{s}-prices.npz",
                ]
            )
    return paths


def register(root):
    write(
        root / "protocol.json",
        dict(
            at=datetime.now(timezone.utc).isoformat(),
            plan=PLAN,
            code={p: digest(ROOT / p) for p in CODE},
            inputs={str(p.relative_to(ROOT)): digest(p) for p in input_paths(root)},
        ),
    )
    write(root / "protocol.sha256.json", dict(sha256=digest(root / "protocol.json")))


def run(root):
    from research import timestral_lead_lag as model

    protocol = json.loads((root / "protocol.json").read_text())
    if (
        protocol["plan"] != PLAN
        or digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
        or any(
            digest(ROOT / p) != h
            for p, h in {**protocol["code"], **protocol["inputs"]}.items()
        )
    ):
        raise ValueError("frozen experiment changed")
    write(root / "opened.json", dict(at=datetime.now(timezone.utc).isoformat()))
    oldfits = json.loads((previous.DEFAULT / "fits.json").read_text())
    fits, results = {}, {}
    for year in (2024, 2025, 2026):
        sample = examples(previous.FUTURES, year)
        t, end, y = sample["anchor"], sample["end"], sample["y"]
        spot = previous.spot_data(year)
        signals, _ = center.signals(sample["x"], center.centers(spot, t))
        zs = signals["uniform"]
        price, lead, available = [], [], []
        for s in shared.SYMBOLS:
            price_features, lead_features, context_valid = model.features(
                spot[s], dict(np.load(root / f"{year}-{s}-prices.npz")), t
            )
            price.append(price_features)
            lead.append(lead_features)
            available.append(context_valid)
        valid = np.all(available, axis=0)
        features = dict(price_control=np.stack(price), lead_lag=np.stack(lead))
        q = {name: [] for name in (*ARMS, "baseline")}
        indices = []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (
                (t >= cutoff - 90 * 86400)
                & (end <= cutoff)
                & sample["available"]
                & valid
            )
            test = (t >= cutoff) & (t < epoch(year, month + 1))
            labels = y[:, train].reshape(-1)
            moving = labels != 1
            yy = (labels[moving] == 2).astype(int)
            tt, ee = np.tile(t[train], 3)[moving], np.tile(end[train], 3)[moving]
            ids = np.concatenate(
                [np.column_stack((np.full(train.sum(), a), t[train])) for a in range(3)]
            ).astype("<i8")[moving]
            baseline = oldfits[f"{year}-{month}-uniform"]["adjusted"]
            offset = baseline["beta"] * zs
            q["baseline"].append(expit(offset[:, test]))
            for name in ARMS:
                state = model.offsetfit(
                    features[name][:, train].reshape(-1, 2)[moving],
                    yy,
                    offset[:, train].reshape(-1)[moving],
                    tt,
                    ee,
                    cutoff,
                )
                state["training_ids_sha256"] = hashlib.sha256(ids.tobytes()).hexdigest()
                state["training_labels_sha256"] = hashlib.sha256(
                    yy.astype("<i8").tobytes()
                ).hexdigest()
                state["baseline_beta"] = baseline["beta"]
                fits[f"{year}-{month}-{name}"] = state
                pred = np.stack(
                    [
                        model.predict(
                            state, features[name][a, test], offset[a, test], t[test]
                        )
                        for a in range(3)
                    ]
                )
                pred[:, ~valid[test]] = q["baseline"][-1][:, ~valid[test]]
                q[name].append(pred)
            indices.append(np.flatnonzero(test))
            print("fitted", year, month, flush=True)
        ids = np.concatenate(indices)
        q = {k: np.concatenate(v, axis=1) for k, v in q.items()}
        for a, s in enumerate(shared.SYMBOLS):
            key = f"{year}-{s}"
            prior = dict(np.load(prior_path(year, s)))
            old = dict(np.load(previous.DEFAULT / f"{key}-predictions.npz"))
            outcome = dict(np.load(outcome_path(year, s)))
            anchors, labels = t[ids], y[a, ids]
            if (
                not np.array_equal(anchors, prior["anchor"])
                or not np.array_equal(anchors, old["anchor"])
                or not np.array_equal(anchors, outcome["anchor"])
                or not np.array_equal(labels, outcome["label"])
                or outcome["end"].max() > shared.BOUNDARY
            ):
                raise ValueError("changed evaluation membership/labels")
            np.testing.assert_allclose(
                q["baseline"][a], old["uniform_up"], atol=1e-14, rtol=0
            )
            raw = prior["cross" if year == 2026 else "probability"]
            flat = (raw[:, 1] + prior["frequency"][:, 1]) / 2
            p = {name: combine(qq[a], flat) for name, qq in q.items()}
            selected = np.maximum(p["baseline"][:, 0], p["baseline"][:, 2]) >= 0.52
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(
                    f,
                    anchor=anchors,
                    available=valid[ids],
                    **{name: pp for name, pp in p.items()},
                )
            results[key] = dict(
                n=len(ids),
                missing_contexts=int((~valid[ids]).sum()),
                original={},
                comparisons={},
            )
            for name, pp in p.items():
                results[key]["original"][name] = shared.evaluate(
                    labels, pp, prior["frequency"], prior["prior_reversal"], anchors
                )
            for name, control in PAIRS:
                results[key]["comparisons"][f"{name}_vs_{control}"] = dict(
                    full=paired_metrics(
                        labels, p[name], p[control], anchors, np.ones(len(ids), bool)
                    ),
                    selected=paired_metrics(
                        labels, p[name], p[control], anchors, selected
                    ),
                    direction=paired_metrics(
                        labels,
                        p[name],
                        p[control],
                        anchors,
                        np.ones(len(ids), bool),
                        direction=True,
                    ),
                )
            print("scored", key, flush=True)
    write(root / "fits.json", fits)
    write(root / "results.json", results)
    original = {
        name: sum(r["original"][name]["gate"] for r in results.values())
        for name in (*ARMS, "baseline")
    }
    counts = {
        pair: {
            scope: sum(
                r["comparisons"][pair][scope]["improvement_gate"]
                for r in results.values()
            )
            for scope in ("full", "selected", "direction")
        }
        for pair in results[next(iter(results))]["comparisons"]
    }
    success = original["lead_lag"] == 9 and all(
        counts[f"lead_lag_vs_{control}"][scope] == 9
        for control in ("baseline", "price_control")
        for scope in ("full", "direction")
    )
    write(
        root / "summary.json",
        dict(
            original_passes=original,
            paired_gates=counts,
            stable_improvement=success,
            anchors=sum(r["n"] for r in results.values()),
        ),
    )
    write(
        root / "manifest.json",
        {p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "register", "run"))
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.command](args.root.resolve())
