#!/usr/bin/env python3
"""Locked shared-direction replication on January-July 2024 and 2025."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_universal as model  # noqa: E402
from research import timestral_reversal as reversal  # noqa: E402
from scripts.run_timestral_microdirection import ROOT, digest, write  # noqa: E402

CODE = (
    "research/timestral_universal.py",
    "research/timestral_reversal.py",
    "research/timestral_microdirection.py",
    "scripts/run_universal_history.py",
    "scripts/run_timestral_microdirection.py",
)
PLAN = {
    "purpose": "Locked temporal replication of the PRIMARY shared cross-market model on two older cohorts; no parameter or threshold changes.",
    "cohorts": [2024, 2025],
    "evaluation": "January-July each year; same 5m target, 1m delay, 90-day monthly fitting, all three assets.",
    "source": "Native1m BTC/ETH/SOL October2023-July2025; parse milliseconds before2025 and microseconds from2025. Fixed URLs, SHA256 receipts, bounded files. No protected sources.",
    "missing": "Retain invalid source rows in provenance; intersect asset timestamps and split at gaps; discard contexts/targets crossing segment boundaries. No forward fill.",
    "model": "Exact frozen timestral_universal.fit/predict, all23 features and shared coefficients; no new variants or selection.",
    "baselines": "Past per-asset frequency and the frozen symmetric reversal rule fitted on the same matured samples.",
    "gate": "Unchanged individual asset criterion from the 2026 experiment. Replication passes only if all six asset/cohort checks pass. Bonferroni interval correction within each three-asset cohort; no claim of prospective validation.",
    "context": "Registered after positive January-July2026 shared-model results and before acquiring these minute files. Older hourly prices were used in earlier studies; these are retrospective replications.",
    "storage": "66 bounded monthly ZIPs, no extracted CSV copies; estimated increment including evidence and Git <0.5GiB.",
}


def epoch(year, month):
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())


def register(root):
    urls = []
    for symbol in model.SYMBOLS:
        for number in range(2023 * 12 + 9, 2025 * 12 + 7):
            year, month = divmod(number, 12)
            name = f"{symbol}-1m-{year}-{month + 1:02d}.zip"
            urls.append(
                f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{name}"
            )
    write(
        root / "protocol.json",
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "plan": PLAN,
            "urls": urls,
            "code": {n: digest(ROOT / n) for n in CODE},
            "prior_results_sha256": digest(
                ROOT / "reports/timestral-universal-20261027/results.json"
            ),
        },
    )
    write(root / "protocol.sha256.json", {"sha256": digest(root / "protocol.json")})


def validate(root):
    p = json.loads((root / "protocol.json").read_text())
    if (
        p["plan"] != PLAN
        or p["code"] != {n: digest(ROOT / n) for n in CODE}
        or digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
    ):
        raise ValueError("frozen replication changed")
    return p


def decode(row, year, month):
    if len(row) != 12:
        raise ValueError("invalid kline width")
    factor = 1000 if year < 2025 else 1
    t, ct = int(row[0]) * factor, int(row[6]) * factor
    when = datetime.fromtimestamp(t / 1e6, timezone.utc)
    if (
        when.year != year
        or when.month != month
        or t % 60000000
        or not epoch(2023, 10) <= t // 1000000 < epoch(2025, 8)
    ):
        raise ValueError("misplaced or unregistered source")
    o, hi, lo, c, v = map(float, row[1:6])
    tr, buy = float(row[8]), float(row[9])
    if (
        not t <= ct < t + 60000000
        or not np.isfinite([o, hi, lo, c, v, tr, buy]).all()
        or not 0 < lo <= min(o, c) <= max(o, c) <= hi
        or min(v, tr, buy) < 0
        or buy > v + 1e-8
    ):
        return None
    return t // 1000000, c, v, buy, tr


def prepare(root):
    p = validate(root)
    if (root / "sources.json").exists():
        raise ValueError("source already prepared")
    raw = root / "raw"
    raw.mkdir(exist_ok=True)

    def acquire(url):
        session = requests.Session()
        session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=3,
                    backoff_factor=0.5,
                    status_forcelist=(429, 500, 502, 503, 504),
                )
            ),
        )
        name = url.rsplit("/", 1)[-1]
        receipt = session.get(url + ".CHECKSUM", timeout=30)
        receipt.raise_for_status()
        tokens = receipt.text.split()
        if (
            len(tokens) != 2
            or not re.fullmatch("[0-9a-f]{64}", tokens[0])
            or tokens[1].lstrip("*") != name
        ):
            raise ValueError("invalid checksum receipt")
        path = raw / name
        if path.exists():
            payload = path.read_bytes()
        else:
            payload = bytearray()
            with session.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                for block in response.iter_content(65536):
                    payload.extend(block)
                    if len(payload) > 10_000_000:
                        raise ValueError("monthly ZIP exceeds bound")
        if (
            len(payload) > 10_000_000
            or hashlib.sha256(payload).hexdigest() != tokens[0]
        ):
            raise ValueError("corrupt source")
        if not path.exists():
            with path.open("xb") as f:
                f.write(payload)
        check = raw / (name + ".CHECKSUM")
        if check.exists():
            if check.read_text().split()[0] != tokens[0]:
                raise ValueError("checksum receipt changed")
        else:
            check.write_text(receipt.text)
        symbol, _, year, month = name.removesuffix(".zip").split("-")
        rows, invalid = [], []
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            entries = z.infolist()
            if (
                len(entries) != 1
                or entries[0].filename != name.replace(".zip", ".csv")
                or entries[0].file_size > 20_000_000
            ):
                raise ValueError("unexpected archive")
            with z.open(entries[0]) as f:
                for values in csv.reader(io.TextIOWrapper(f)):
                    if values[0] == "open_time":
                        continue
                    row = decode(values, int(year), int(month))
                    if row is None:
                        invalid.append(values)
                    else:
                        rows.append(row)
        return (
            symbol,
            rows,
            {
                "url": url,
                "sha256": tokens[0],
                "bytes": len(payload),
                "rows": len(rows),
                "invalid": invalid,
            },
        )

    groups = {s: [] for s in model.SYMBOLS}
    sources = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for n, (symbol, rows, receipt) in enumerate(executor.map(acquire, p["urls"])):
            groups[symbol].extend(rows)
            sources.append(receipt)
            if (n + 1) % 6 == 0:
                print(f"acquired {n + 1}/66 months", flush=True)
    derived = {}
    for symbol, rows in groups.items():
        a = np.asarray(rows)
        if np.any(np.diff(a[:, 0]) <= 0):
            raise ValueError("unordered native minute rows")
        d = {
            k: a[:, j]
            for j, k in enumerate(("time", "close", "volume", "taker", "trades"))
        }
        d["time"] = d["time"].astype(np.int64)
        with (root / f"{symbol}-minutes.npz").open("xb") as f:
            np.savez_compressed(f, **d)
        derived[symbol] = {
            "sha256": digest(root / f"{symbol}-minutes.npz"),
            "minutes": len(a),
            "gaps": int(np.sum(np.diff(a[:, 0]) != 60)),
        }
    write(root / "sources.json", {"sources": sources, "derived": derived})


def examples(datasets):
    common = datasets[model.SYMBOLS[0]]["time"]
    for data in datasets.values():
        common = np.intersect1d(common, data["time"])
    aligned = {
        s: {k: v[np.searchsorted(d["time"], common)] for k, v in d.items()}
        for s, d in datasets.items()
    }
    boundaries = np.r_[0, np.flatnonzero(np.diff(common) != 60) + 1, len(common)]
    chunks = []
    cells = []
    signs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 70:
            continue
        data = {s: {k: v[start:end] for k, v in d.items()} for s, d in aligned.items()}
        sample = model.examples(data)
        indices = np.searchsorted(common[start:end], sample["anchor"] - 60)
        states = [reversal.states(d, indices) for d in data.values()]
        cells.append(np.stack([v[0] for v in states]))
        signs.append(np.stack([v[1] for v in states]))
        chunks.append(sample)
    if not chunks:
        raise ValueError("no complete common contexts")
    return (
        {
            k: np.concatenate([c[k] for c in chunks], axis=1 if k in ("x", "y") else 0)
            for k in chunks[0]
        },
        np.concatenate(cells, axis=1),
        np.concatenate(signs, axis=1),
    )


def run(root):
    validate(root)
    sources = json.loads((root / "sources.json").read_text())
    for s in model.SYMBOLS:
        if digest(root / f"{s}-minutes.npz") != sources["derived"][s]["sha256"]:
            raise ValueError("derived input changed")
    write(root / "opened.json", {"at": datetime.now(timezone.utc).isoformat()})
    sample, cells, signs = examples(
        {s: dict(np.load(root / f"{s}-minutes.npz")) for s in model.SYMBOLS}
    )
    fits, baselines, results = {}, {}, {}
    for year in (2024, 2025):
        predictions = []
        frequency = [[] for _ in range(3)]
        prior = [[] for _ in range(3)]
        indices = []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (sample["anchor"] >= cutoff - 90 * 86400) & (
                sample["end"] <= cutoff
            )
            test = (sample["anchor"] >= cutoff) & (
                sample["anchor"] < epoch(year, month + 1)
            )
            x = np.concatenate([sample["x"][a, train] for a in range(3)])
            y = np.concatenate([sample["y"][a, train] for a in range(3)])
            f = model.fit(x, y, np.tile(sample["end"][train], 3), cutoff)
            ids = np.concatenate(
                [
                    np.column_stack((np.full(train.sum(), a), sample["anchor"][train]))
                    for a in range(3)
                ]
            ).astype("<i8")
            f.update(
                assets=[0, 1, 2],
                columns=23,
                training_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
            )
            fits[f"{year}_{month}"] = f
            predictions.append(
                np.stack(
                    [
                        model.predict(f, sample["x"][a, test], sample["anchor"][test])
                        for a in range(3)
                    ]
                )
            )
            indices.append(np.flatnonzero(test))
            for a, s in enumerate(model.SYMBOLS):
                b = reversal.fit(
                    cells[a, train],
                    signs[a, train],
                    sample["y"][a, train],
                    sample["end"][train],
                    cutoff,
                )
                baselines[f"{year}_{month}_{s}"] = b
                q, _ = reversal.predict(
                    b, cells[a, test], signs[a, test], sample["anchor"][test]
                )
                frequency[a].append(q[:, 0])
                prior[a].append(q[:, 2])
            print(f"fitted historical {year}-{month:02d}", flush=True)
        pred = np.concatenate(predictions, axis=1)
        ii = np.concatenate(indices)
        for a, s in enumerate(model.SYMBOLS):
            base, old = np.concatenate(frequency[a]), np.concatenate(prior[a])
            with (root / f"{year}-{s}-predictions.npz").open("xb") as f:
                np.savez_compressed(
                    f,
                    anchor=sample["anchor"][ii],
                    probability=pred[a],
                    frequency=base,
                    prior_reversal=old,
                )
            with (root / f"{year}-{s}-outcomes.npz").open("xb") as f:
                np.savez_compressed(
                    f,
                    anchor=sample["anchor"][ii],
                    end=sample["end"][ii],
                    label=sample["y"][a, ii],
                )
            results[f"{year}-{s}"] = model.evaluate(
                sample["y"][a, ii], pred[a], base, old, sample["anchor"][ii]
            )
            print(
                year,
                s,
                json.dumps(
                    {
                        k: results[f"{year}-{s}"].get(k)
                        for k in ("accuracy", "coverage", "gate")
                    }
                ),
                flush=True,
            )
    write(root / "fits.json", fits)
    write(root / "baselines.json", baselines)
    write(
        root / "results.json",
        {
            "cohorts": results,
            "replication_gate": all(v["gate"] for v in results.values()),
        },
    )
    write(
        root / "manifest.json",
        {
            p.name: digest(p)
            for p in sorted(root.iterdir())
            if p.suffix in (".npz", ".json")
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {"register": register, "prepare": prepare, "run": run}[args.action](args.output)
