#!/usr/bin/env python3
"""Frozen five-minute reversal transfer on native ETH/SOL minute candles."""

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
from research import timestral_microdirection as micro  # noqa: E402
from research import timestral_reversal as reversal  # noqa: E402
from scripts.run_timestral_microdirection import ROOT, digest, epoch, write  # noqa: E402

CODE = (
    "research/timestral_reversal.py",
    "scripts/run_direction_transfer.py",
    "research/timestral_microdirection.py",
    "scripts/run_timestral_microdirection.py",
)
SYMBOLS = ("ETHUSDT", "SOLUSDT")
PLAN = {
    "method": "Exact frozen symmetric reversal method, fit separately per asset; no altered thresholds, priors, features or gates.",
    "horizon_minutes": 5,
    "delay_minutes": 1,
    "train_days": 90,
    "evaluation": "2026-01-01 <= anchor < 2026-08-01 and label end <=2026-08-01",
    "source": "Native Binance Spot 1m from September 2025 to July 2026. No last-ten-second features available or used; tail_flow placeholder zero is ignored by reversal.",
    "primary": "Both ETHUSDT and SOLUSDT must pass the unchanged reversal gate. BTC selection is explicitly exploratory; correlated assets are not independent replications.",
    "diagnostic": "Report accuracy and its original 7-day block interval separately from the full probability gate, without replacing that gate.",
    "prior_findings": "BTC selected 5m accuracy 51.62%, but probability gate failed; this registration precedes acquiring new minute data.",
    "storage": "22 bounded ZIPs <=10MB each; streamed parsing, no extracted CSV copies; total estimated increment including reports <0.3GiB.",
}


def register(root):
    urls = []
    for symbol in SYMBOLS:
        for number in range(2025 * 12 + 8, 2026 * 12 + 7):
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
            "code": {n: digest(ROOT / n) for n in CODE},
            "urls": urls,
            "prior_results_sha256": digest(
                ROOT / "research/out/timestral-reversal-20261027/results.json"
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
        raise ValueError("frozen transfer changed")
    return p


def decode(values, year, month):
    if len(values) != 12:
        raise ValueError("invalid native minute width")
    # All registered source dates use microseconds, per official Spot format.
    t, ct = int(values[0]), int(values[6])
    when = datetime.fromtimestamp(t / 1e6, timezone.utc)
    if (
        when.year != year
        or when.month != month
        or t % 60000000
        or t >= micro.BOUNDARY * 1000000
    ):
        raise ValueError("misplaced or protected minute")
    o, high, low, close, volume = map(float, values[1:6])
    trades, taker = float(values[8]), float(values[9])
    if (
        not t <= ct < t + 60000000
        or not np.isfinite([o, high, low, close, volume, trades, taker]).all()
        or not 0 < low <= min(o, close) <= max(o, close) <= high
        or min(volume, trades, taker) < 0
        or taker > volume + 1e-8
    ):
        return None
    return t // 1000000, close, volume, taker, trades


def prepare(root):
    p = validate(root)
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    if (root / "sources.json").exists():
        raise ValueError("source already prepared")

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
        target = raw / name
        if target.exists():
            payload = target.read_bytes()
        else:
            with session.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                payload = bytearray()
                for block in response.iter_content(65536):
                    payload.extend(block)
                    if len(payload) > 10_000_000:
                        raise ValueError("monthly download exceeds bound")
        if (
            len(payload) > 10_000_000
            or hashlib.sha256(payload).hexdigest() != tokens[0]
        ):
            raise ValueError("source checksum mismatch")
        if not target.exists():
            with target.open("xb") as f:
                f.write(payload)
        check = raw / (name + ".CHECKSUM")
        if not check.exists():
            check.write_text(receipt.text)
        elif check.read_text().split()[0] != tokens[0]:
            raise ValueError("checksum receipt changed")
        symbol, _, year, month = name.removesuffix(".zip").split("-")
        rows, invalid = [], []
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            entries = z.infolist()
            if (
                len(entries) != 1
                or entries[0].filename != name.replace(".zip", ".csv")
                or entries[0].file_size > 20_000_000
            ):
                raise ValueError("unexpected CSV archive")
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

    grouped = {s: [] for s in SYMBOLS}
    sources = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, (symbol, rows, receipt) in enumerate(executor.map(acquire, p["urls"])):
            grouped[symbol].extend(rows)
            sources.append(receipt)
            print(f"verified {i + 1}/22 source months", flush=True)
    derived = {}
    for symbol, rows in grouped.items():
        a = np.asarray(rows)
        if np.any(np.diff(a[:, 0]) <= 0):
            raise ValueError("duplicate or unordered minutes")
        d = {
            k: a[:, i]
            for i, k in enumerate(("time", "close", "volume", "taker", "trades"))
        }
        d["time"] = d["time"].astype(np.int64)
        d["tail_flow"] = np.zeros(len(a))
        with (root / f"{symbol}-minutes.npz").open("xb") as f:
            np.savez_compressed(f, **d)
        derived[symbol] = {
            "sha256": digest(root / f"{symbol}-minutes.npz"),
            "minutes": len(a),
            "gaps": int(np.sum(np.diff(a[:, 0]) != 60)),
        }
    write(root / "sources.json", {"sources": sources, "derived": derived})


def run(root):
    validate(root)
    sources = json.loads((root / "sources.json").read_text())
    write(root / "opened.json", {"at": datetime.now(timezone.utc).isoformat()})
    results, fits = {}, {}
    for symbol in SYMBOLS:
        if (
            digest(root / f"{symbol}-minutes.npz")
            != sources["derived"][symbol]["sha256"]
        ):
            raise ValueError("derived source changed")
        data = dict(np.load(root / f"{symbol}-minutes.npz"))
        s = micro.samples(data, 5, 1)
        cells, signs = reversal.states(data, s["index"])
        ps, ss, ii = [], [], []
        for month in range(1, 8):
            cutoff = epoch(f"2026-{month:02d}-01T00:00:00Z")
            stop = epoch(f"2026-{month + 1:02d}-01T00:00:00Z")
            train = (s["anchor"] >= cutoff - 90 * 86400) & (s["end"] <= cutoff)
            test = (s["anchor"] >= cutoff) & (s["anchor"] < stop)
            f = reversal.fit(
                cells[train], signs[train], s["label"][train], s["end"][train], cutoff
            )
            f["training_anchor_sha256"] = hashlib.sha256(
                s["anchor"][train].astype("<i8").tobytes()
            ).hexdigest()
            fits[f"{symbol}_{month}"] = f
            p, sel = reversal.predict(f, cells[test], signs[test], s["anchor"][test])
            ps.append(p)
            ss.append(sel)
            ii.append(np.flatnonzero(test))
        p, sel, ids = np.concatenate(ps), np.concatenate(ss), np.concatenate(ii)
        with (root / f"{symbol}-predictions.npz").open("xb") as f:
            np.savez_compressed(
                f,
                anchor=s["anchor"][ids],
                probability=p,
                selected=sel,
                cells=cells[ids],
                signs=signs[ids],
            )
        with (root / f"{symbol}-outcomes.npz").open("xb") as f:
            np.savez_compressed(
                f,
                anchor=s["anchor"][ids],
                end=s["end"][ids],
                label=s["label"][ids],
                returns=s["return"][ids],
            )
        results[symbol] = reversal.evaluate(s["label"][ids], p, s["anchor"][ids], sel)
        print(symbol, json.dumps(results[symbol]), flush=True)
    write(root / "fits.json", fits)
    write(
        root / "results.json",
        {"runs": results, "primary_gate": all(v["gate"] for v in results.values())},
    )
    write(
        root / "manifest.json",
        {
            p.name: digest(p)
            for p in sorted(root.iterdir())
            if p.suffix in (".json", ".npz")
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {"register": register, "prepare": prepare, "run": run}[args.action](args.output)
