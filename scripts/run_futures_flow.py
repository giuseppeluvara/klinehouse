#!/usr/bin/env python3
"""Registered derivatives-information experiment with bounded public sources."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_futures_flow as model  # noqa: E402
from research import timestral_universal as shared  # noqa: E402
from scripts.run_timestral_microdirection import ROOT, digest, write  # noqa: E402

DEFAULT = ROOT / "reports/timestral-futures-flow-20261027"
RAW = ROOT / "research/out/futures-flow-20261027/raw"
UNIVERSAL = ROOT / "reports/timestral-universal-20261027"
MICRO = ROOT / "reports/timestral-microdirection-20261027"
CODE = (
    "research/timestral_premium_direction.py",
    "research/timestral_futures_flow.py",
    "research/timestral_universal.py",
    "scripts/run_futures_flow.py",
    "scripts/run_timestral_microdirection.py",
)
PLAN = {
    "purpose": "Test new futures volume intensity and aggressor flow for conditional up/down direction, holding flat probability fixed; flow must improve BOTH spot and volume controls.",
    "status": "Retrospective exploration on already explored spot outcomes, Jan-Jul2024/2025/2026; new futures trading-volume source. Not prospective or pristine validation.",
    "source": "Binance USD-M monthly trade klines1m, Oct(previous year)-Jul(evaluation year), BTC/ETH/SOL, 90 ZIPs. Millisecond time, base volume, taker-buy base volume, trade count retained after OHLC validation. Aggressive sell=volume-buy, not short positions or order book. Official SHA256 checksums, 8MB ZIP/32MB expanded per file maximum, two download workers.",
    "gaps": "Reject invalid/duplicate/unordered records; retain missing minutes without filling. Require121 contiguous minutes on ALL3assets for the same training rows in all3arms. Both futures arms fall back to spot on missing evaluation context; retain ALL original evaluation anchors.",
    "availability": "Latest source open<=anchor-120sec, close<=anchor-60sec. Unknown archive revisions/publication latency: causal replay conditional on declared one-minute lag, not verified point-in-time vintage.",
    "features": list(model.FEATURES),
    "representation": "Volume/trade activity=log(latest/mean(previous120minutes)), clipped+-4; volume denominator floor1e-12, trade floor1, ratio floor1e-6. Activity interacted with signed spot features. Flowh=(2*sum(buy)-sum(volume))/sum(volume), zero if no volume, h=1/5/30/120. Latest included in flow, excluded from activity reference. Cross-asset average excludes own. All12 additions odd under simultaneous price and aggressor reflection. No futures price features.",
    "target": "Five-minute spot return from anchor+1min to anchor+6min. Binary fit on matured non-flat outcomes; recombine with identical frozen cross-model flat probabilities for ALL-anchor original direction gate.",
    "training": "Monthly90day rolling pooled3asset logistic, C=.1 max500 tol1e-8, reflected augmentation, train-only normalization and+-8 clip. Spot23, volume27, flow35 features;63fits, no asset identifier. No parameter/signal sweep after scores.",
    "comparisons": ["volume_vs_spot", "flow_vs_spot", "flow_vs_volume"],
    "primary": "General flow claim requires ALL9 cohorts pass information gates for BOTH flow_vs_spot AND flow_vs_volume, AND original unconditional flow direction gates. Each conditional gate: accuracy lower99.444%>50%, paired Brier/logloss gain lower99.444%>0 against named control AND past frequency, >=5months Brier gain>0 versus control, >=100 forecasts each direction. Volume_vs_spot is a separate ablation.",
    "uncertainty": "4000 paired7day moving-block replicates, seed271032; Bonferroni9 per endpoint separately, not simultaneous correction over comparisons/endpoints/research history. No bootstrap refitting. Conditional evaluation cannot identify future flats or demonstrate tradable coverage.",
    "protected": "All market sources/outcomes before2026-08-01; preexisting artifacts unchanged. Negative results retained; no costs/profitability/deployment claim.",
    "references": [
        "https://github.com/binance/binance-public-data",
        "https://developers.binance.info/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data",
    ],
}


def epoch(year, month):
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())


def sources(year):
    return {
        s: UNIVERSAL / "history" / f"{s}-minutes.npz"
        if year < 2026
        else MICRO / "micro/minutes.npz"
        if a == 0
        else MICRO / f"transfer/{s}-minutes.npz"
        for a, s in enumerate(shared.SYMBOLS)
    }


def prior_path(year, s):
    return (
        UNIVERSAL / f"{s}-predictions.npz"
        if year == 2026
        else UNIVERSAL / "history" / f"{year}-{s}-predictions.npz"
    )


def requests_list():
    return [
        (year, s, n // 12, n % 12 + 1)
        for year in (2024, 2025, 2026)
        for s in shared.SYMBOLS
        for n in range((year - 1) * 12 + 9, year * 12 + 7)
    ]


def url(s, year, month):
    if s not in shared.SYMBOLS or (year, month) >= (2026, 8):
        raise ValueError("unregistered/protected source")
    return f"https://data.binance.vision/data/futures/um/monthly/klines/{s}/1m/{s}-1m-{year}-{month:02d}.zip"


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    inputs = {
        str(p.relative_to(ROOT)): digest(p)
        for year in (2024, 2025, 2026)
        for p in [
            *sources(year).values(),
            *[prior_path(year, s) for s in shared.SYMBOLS],
        ]
    }
    write(
        root / "protocol.json",
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "plan": PLAN,
            "code": {p: digest(ROOT / p) for p in CODE},
            "inputs": inputs,
            "urls": [url(s, y, m) for _, s, y, m in requests_list()],
        },
    )
    write(root / "protocol.sha256.json", {"sha256": digest(root / "protocol.json")})


def validate(root):
    p = json.loads((root / "protocol.json").read_text())
    if (
        p["plan"] != PLAN
        or p["code"] != {name: digest(ROOT / name) for name in CODE}
        or digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
        or any(digest(ROOT / name) != sha for name, sha in p["inputs"].items())
    ):
        raise ValueError("frozen experiment changed")
    return p


def decode(row, year, month):
    if len(row) != 12:
        raise ValueError("futures kline width")
    t, end = int(row[0]), int(row[6])
    o, h, low, c = map(float, row[1:5])
    volume, taker, trades = float(row[5]), float(row[9]), int(row[8])
    upper = epoch(year + (month == 12), month % 12 + 1)
    if (
        t % 60000
        or not epoch(year, month) <= t // 1000 < upper
        or t // 1000 >= shared.BOUNDARY
        or end != t + 59999
        or not np.isfinite([o, h, low, c, volume, taker, trades]).all()
        or not 0 < low <= min(o, c) <= max(o, c) <= h
        or not 0 <= taker <= volume
        or trades < 0
    ):
        raise ValueError("invalid or protected futures candle")
    return t // 1000, volume, taker, trades


def download(item):
    cohort, s, year, month = item
    address = url(s, year, month)
    name = address.rsplit("/", 1)[-1]
    path = RAW / name
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
            )
        ),
    )
    checksum = session.get(address + ".CHECKSUM", timeout=30)
    checksum.raise_for_status()
    if not path.exists():
        response = session.get(address, timeout=60, stream=True)
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(65536):
            content.extend(chunk)
            if len(content) > 8_000_000:
                raise ValueError("futures download budget exceeded")
        if hashlib.sha256(content).hexdigest() != checksum.text.split()[0]:
            raise ValueError("official checksum mismatch")
        with path.open("xb") as f:
            f.write(content)
        path.with_suffix(".zip.CHECKSUM").write_text(checksum.text)
    if digest(path) != checksum.text.split()[0]:
        raise ValueError("cached futures source changed")
    rows = []
    with zipfile.ZipFile(path) as z:
        if len(z.infolist()) != 1 or z.infolist()[0].file_size > 32_000_000:
            raise ValueError("unexpected ZIP members or expansion")
        with z.open(z.namelist()[0]) as f:
            for row in csv.reader(io.TextIOWrapper(f)):
                if row[0] == "open_time":
                    continue
                rows.append(decode(row, year, month))
    a = np.asarray(rows)
    expected = np.arange(
        epoch(year, month), epoch(year + (month == 12), month % 12 + 1), 60
    )
    if np.any(np.diff(a[:, 0]) <= 0) or not np.isin(a[:, 0], expected).all():
        raise ValueError("duplicate, unordered or out-of-period futures rows")
    return (
        cohort,
        s,
        a,
        {
            "url": address,
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "rows": len(a),
            "missing_minutes": len(expected) - len(a),
        },
    )


def prepare(root):
    validate(root)
    RAW.mkdir(parents=True, exist_ok=True)
    grouped, provenance = {}, []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for year, s, a, receipt in pool.map(download, requests_list()):
            grouped.setdefault(f"{year}-{s}", []).append(a)
            provenance.append(receipt)
            if len(provenance) % 10 == 0:
                print("verified futures ZIPs", len(provenance), "/90", flush=True)
    for key, pieces in grouped.items():
        a = np.concatenate(pieces)
        with (root / f"{key}-futures.npz").open("xb") as f:
            np.savez_compressed(
                f,
                time=a[:, 0].astype(np.int64),
                volume=a[:, 1],
                taker=a[:, 2],
                trades=a[:, 3].astype(np.int64),
            )
    write(
        root / "sources.json",
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "sources": provenance,
            "arrays": {p.name: digest(p) for p in sorted(root.glob("*-futures.npz"))},
        },
    )


def examples(root, year):
    data = {}
    for s, p in sources(year).items():
        d = dict(np.load(p))
        mask = (d["time"] >= epoch(year - 1, 10)) & (d["time"] < epoch(year, 8) + 360)
        data[s] = {k: v[mask] for k, v in d.items()}
    sample = shared.examples(data)
    futures = {
        s: dict(np.load(root / f"{year}-{s}-futures.npz")) for s in shared.SYMBOLS
    }
    valid, extra = model.futures_features(futures, sample["anchor"], sample["x"])
    all_extra = np.zeros((3, len(sample["anchor"]), 12))
    all_extra[:, valid] = extra
    return {
        **sample,
        "available": valid,
        "x": np.concatenate((sample["x"], all_extra), axis=2),
    }


def run(root):
    validate(root)
    provenance = json.loads((root / "sources.json").read_text())
    if any(digest(root / p) != h for p, h in provenance["arrays"].items()):
        raise ValueError("prepared sources changed")
    write(root / "opened.json", {"at": datetime.now(timezone.utc).isoformat()})
    fits, results = {}, {}
    for year in (2024, 2025, 2026):
        sample = examples(root, year)
        predictions = {name: [] for name, _ in model.ARMS}
        ids = []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (sample["anchor"] >= cutoff - 90 * 86400) & (
                sample["end"] <= cutoff
            )
            train &= sample["available"]
            test = (sample["anchor"] >= cutoff) & (
                sample["anchor"] < epoch(year, month + 1)
            )
            y = np.concatenate([sample["y"][a, train] for a in range(3)])
            ends = np.tile(sample["end"][train], 3)
            membership = np.concatenate(
                [
                    np.column_stack((np.full(train.sum(), a), sample["anchor"][train]))
                    for a in range(3)
                ]
            ).astype("<i8")
            for name, columns in model.ARMS:
                x = np.concatenate([sample["x"][a, train, :columns] for a in range(3)])
                moving = y != 1
                state = model.fit(
                    x[moving], (y[moving] == 2).astype(int), ends[moving], cutoff
                )
                state["training_ids_sha256"] = hashlib.sha256(
                    membership[moving].tobytes()
                ).hexdigest()
                state["training_labels_sha256"] = hashlib.sha256(
                    (y[moving] == 2).astype("<i8").tobytes()
                ).hexdigest()
                fits[f"{year}-{month}-{name}"] = state
                predictions[name].append(
                    np.stack(
                        [
                            model.predict(
                                state,
                                sample["x"][a, test, :columns],
                                sample["anchor"][test],
                            )
                            for a in range(3)
                        ]
                    )
                )
                if name != "spot":
                    missing = ~sample["available"][test]
                    predictions[name][-1][:, missing] = predictions["spot"][-1][
                        :, missing
                    ]
            ids.append(np.flatnonzero(test))
            print("fitted", year, month, flush=True)
        ids = np.concatenate(ids)
        q = {k: np.concatenate(v, axis=1) for k, v in predictions.items()}
        for a, s in enumerate(shared.SYMBOLS):
            key = f"{year}-{s}"
            prior = dict(np.load(prior_path(year, s)))
            t, y = sample["anchor"][ids], sample["y"][a, ids]
            if not np.array_equal(t, prior["anchor"]):
                raise ValueError("evaluation anchors changed")
            raw = prior["cross" if year == 2026 else "probability"]
            flat = raw[:, 1]
            frequency = prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1])
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(
                    f,
                    anchor=t,
                    spot_up=q["spot"][a],
                    volume_up=q["volume"][a],
                    flow_up=q["flow"][a],
                    futures_available=sample["available"][ids],
                )
            results[key] = {
                "futures_missing_contexts": int((~sample["available"][ids]).sum()),
                "comparisons": {
                    f"{candidate}_vs_{control}": model.comparison(
                        y, q[candidate][a], q[control][a], frequency, t
                    )
                    for candidate, control in model.COMPARISONS
                },
            }
            for name, _ in model.ARMS:
                results[key][name] = shared.evaluate(
                    y,
                    model.combine(q[name][a], flat),
                    prior["frequency"],
                    prior["prior_reversal"],
                    t,
                )
            print(
                key,
                {
                    k: v["information_gate"]
                    for k, v in results[key]["comparisons"].items()
                },
                "direction",
                results[key]["flow"]["gate"],
                flush=True,
            )
    write(root / "fits.json", fits)
    write(
        root / "results.json",
        {
            "cohorts": results,
            "all_information_gates": all(
                r["comparisons"][c]["information_gate"]
                for r in results.values()
                for c in ("flow_vs_spot", "flow_vs_volume")
            ),
            "all_direction_gates": all(r["flow"]["gate"] for r in results.values()),
        },
    )
    write(
        root / "manifest.json",
        {p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "prepare", "run"))
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"register": register, "prepare": prepare, "run": run}[args.action](args.root)
