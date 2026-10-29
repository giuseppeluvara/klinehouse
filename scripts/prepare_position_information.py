#!/usr/bin/env python3
"""Bounded metrics acquisition and field-specific timestamp diagnostics."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone, timedelta
from functools import lru_cache
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
from scripts.run_futures_flow import ROOT, digest, write  # noqa: E402

DEFAULT = ROOT / "reports/timestral-position-information-20261029"
RAW = ROOT / "research/out/position-information-20261029/raw"
FUTURES = ROOT / "reports/timestral-futures-flow-20261027"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
BOUNDARY = 1785542400
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
PLAN = {
    "scope": "BTC/ETH/SOL daily USD-M metrics, Oct(previousyear) through July30 for2024/2025/2026 cohorts. July31 files deliberately excluded to avoid plausible Aug1 snapshots; all eventual evaluation anchors retained via baseline fallback.",
    "budget": "Four workers, existing environment, <=1MB ZIP and<=1MB expansion each; total ZIP budget100MB, planned total study0.4GiB. Official checksum required, preserve existing cached pilot24files. No market file dated2026-08-01 orlater requested. No tradingcredentials.",
    "parse": "Exact8column header. Preserve rowlabel, fileday,rowindex and all6values. Empty/nonpositive OIqty or positionratio invalidates context; off-grid and all instances of duplicate labels invalidated, never round or keep-last. Any plausible content time label+300>=2026Aug1 rejected before numeric parsing. Canonical300sec grid retains missing rows as NaN, no filling.",
    "flow_audit": "For each symbol/filedate compare archive takerratio with existing1m futures buy/(volume-buy) over[T-300,T) and[T,T+300). Report both correlations and median relative errors. Diagnostic match requires >=200 finitepairs, correlation>=.98, medianrelativeerror<=.05 and besterror<=.2*othererror. These thresholds are fixed after pilot SOURCE inspection, before bulkdownload and before new modeloutcomes. A flow match does NOT certify stock-column timestamps, position definitions or historical publication availability.",
    "field_status": "Separate label/structure verification, flow-content alignment, OI independent sample evidence, position timing, and publication/vintage. Never infer universalJune25shift or applyflowshift to OI/positions. The pilot contradicted uniformhistorical labelrange claims. Source-only diagnostics use no prediction outcomes.",
}


def items():
    for cohort in (2024, 2025, 2026):
        begin = datetime(cohort - 1, 10, 1, tzinfo=timezone.utc)
        stop = datetime(cohort, 7, 31, tzinfo=timezone.utc)
        for symbol in SYMBOLS:
            for offset in range((stop - begin).days):
                yield (
                    cohort,
                    symbol,
                    (begin + timedelta(days=offset)).strftime("%Y-%m-%d"),
                )


@lru_cache(maxsize=1)
def allowed_sources():
    return {(s, d) for _, s, d in items()}


def url(symbol, day):
    if (
        symbol not in SYMBOLS
        or day >= "2026-08-01"
        or (symbol, day) not in allowed_sources()
    ):
        raise ValueError("unregistered or protected metrics source")
    return f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip"


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    paths = [
        FUTURES / f"{year}-{s}-futures.npz"
        for year in (2024, 2025, 2026)
        for s in SYMBOLS
    ]
    paths.append(FUTURES / "sources.json")
    write(
        root / "source-protocol.json",
        dict(
            at=datetime.now(timezone.utc).isoformat(),
            plan=PLAN,
            code={"scripts/prepare_position_information.py": digest(Path(__file__))},
            inputs={str(p.relative_to(ROOT)): digest(p) for p in paths},
            requests=[
                dict(cohort=y, symbol=s, day=d, url=url(s, d)) for y, s, d in items()
            ],
        ),
    )


def validate(root):
    p = json.loads((root / "source-protocol.json").read_text())
    if p["plan"] != PLAN or any(
        digest(ROOT / f) != h for f, h in {**p["code"], **p["inputs"]}.items()
    ):
        raise ValueError("frozen source protocol changed")
    return p


def decode(path, symbol, day):
    base = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())
    rows, rejected_boundary = [], 0
    with zipfile.ZipFile(path) as z:
        if len(z.infolist()) != 1 or z.infolist()[0].file_size > 1_000_000:
            raise ValueError("metrics expansion budget")
        with z.open(z.namelist()[0]) as f:
            reader = csv.reader(io.TextIOWrapper(f))
            if tuple(next(reader)) != HEADER:
                raise ValueError("metrics schema changed")
            for line, row in enumerate(reader, 2):
                if len(row) != 8 or row[1] != symbol:
                    raise ValueError("metrics symbol or row width")
                stamp = int(
                    datetime.fromisoformat(row[0])
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
                if not base <= stamp <= base + 86400:
                    raise ValueError("metric row outside its file day")
                if stamp + 300 >= BOUNDARY:
                    rejected_boundary += 1
                    continue
                values = [float(v) if v.strip() else float("nan") for v in row[2:]]
                rows.append((stamp, base, line, *values))
    return np.asarray(rows, dtype=float).reshape(-1, 9), rejected_boundary


def fetch(item):
    cohort, symbol, day = item
    address = url(symbol, day)
    path = RAW / address.rsplit("/", 1)[-1]
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
            )
        ),
    )
    receipt = dict(cohort=cohort, symbol=symbol, day=day, url=address)
    checksum = path.with_suffix(".zip.CHECKSUM")
    if path.exists() and checksum.exists():
        expected = checksum.read_text().split()[0]
    else:
        response = session.get(address + ".CHECKSUM", timeout=30)
        if response.status_code == 404:
            return receipt | dict(status="unavailable_404"), None
        response.raise_for_status()
        expected = response.text.split()[0]
        if len(expected) != 64:
            raise ValueError("invalid official checksum")
        reply = session.get(address, timeout=45, stream=True)
        reply.raise_for_status()
        content = bytearray()
        for chunk in reply.iter_content(65536):
            content.extend(chunk)
            if len(content) > 1_000_000:
                raise ValueError("metrics ZIP budget")
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("official metrics checksum mismatch")
        with path.open("xb") as f:
            f.write(content)
        with checksum.open("x") as f:
            f.write(response.text)
    if digest(path) != expected:
        raise ValueError("cached metrics checksum mismatch")
    rows, excluded = decode(path, symbol, day)
    return receipt | dict(
        status="verified",
        sha256=expected,
        bytes=path.stat().st_size,
        rows=len(rows),
        excluded_protected_rows=excluded,
    ), rows


def download(root):
    validate(root)
    RAW.mkdir(parents=True, exist_ok=True)
    receipts, grouped = [], {}
    budget = 0
    requests_ = list(items())
    with ThreadPoolExecutor(max_workers=4) as pool:
        for receipt, rows in pool.map(fetch, requests_):
            receipts.append(receipt)
            budget += receipt.get("bytes", 0)
            if budget > 100_000_000:
                raise ValueError("total metrics download budget")
            if rows is not None:
                grouped.setdefault(
                    f"{receipt['cohort']}-{receipt['symbol']}", []
                ).append(rows)
            if len(receipts) % 150 == 0:
                print("metrics sources", len(receipts), "/", len(requests_), flush=True)
    hashes = {}
    for key, pieces in grouped.items():
        rows = np.concatenate(pieces)
        path = root / f"{key}-metrics.npz"
        with path.open("xb") as f:
            np.savez_compressed(
                f,
                label=rows[:, 0].astype(np.int64),
                file_day=rows[:, 1].astype(np.int64),
                row=rows[:, 2].astype(np.int32),
                values=rows[:, 3:],
            )
        hashes[path.name] = digest(path)
    write(
        root / "sources.json",
        dict(
            at=datetime.now(timezone.utc).isoformat(),
            sources=receipts,
            arrays=hashes,
            zip_bytes=budget,
            requested_files=len(requests_),
        ),
    )


def flow_comparison(times, ratios, futures, shift):
    j = np.searchsorted(futures["time"], times + shift)
    good = np.isfinite(ratios) & (ratios > 0) & (j < len(futures["time"]) - 4)
    indices = np.flatnonzero(good)
    k = j[indices]
    good[indices] &= (futures["time"][k] == times[indices] + shift) & (
        futures["time"][k + 4] - futures["time"][k] == 240
    )
    indices = np.flatnonzero(good)
    k = j[indices]
    volume = sum(futures["volume"][k + n] for n in range(5))
    buys = sum(futures["taker"][k + n] for n in range(5))
    ok = (volume > buys) & (buys > 0)
    predicted = buys[ok] / (volume[ok] - buys[ok])
    actual = ratios[indices][ok]
    if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return dict(n=len(actual), correlation=None, median_relative_error=None)
    return dict(
        n=len(actual),
        correlation=float(np.corrcoef(predicted, actual)[0, 1]),
        median_relative_error=float(np.median(abs(predicted - actual) / actual)),
    )


def audit(root):
    validate(root)
    provenance = json.loads((root / "sources.json").read_text())
    cohorts = {}
    for filename, sha in provenance["arrays"].items():
        if digest(root / filename) != sha:
            raise ValueError("prepared metrics changed")
        key = filename.removesuffix("-metrics.npz")
        d = dict(np.load(root / filename))
        futures = dict(np.load(FUTURES / f"{key}-futures.npz"))
        unique, counts = np.unique(d["label"], return_counts=True)
        duplicate = np.isin(d["label"], unique[counts > 1])
        days = {}
        for day in np.unique(d["file_day"]):
            take = d["file_day"] == day
            values = {
                str(shift): flow_comparison(
                    d["label"][take], d["values"][take, 5], futures, shift
                )
                for shift in (-300, 0)
            }
            match = []
            for shift, other in (("-300", "0"), ("0", "-300")):
                x, alt = values[shift], values[other]
                if (
                    x["n"] >= 200
                    and x["correlation"] is not None
                    and alt["median_relative_error"] is not None
                    and x["correlation"] >= 0.98
                    and x["median_relative_error"] <= 0.05
                    and x["median_relative_error"] <= 0.2 * alt["median_relative_error"]
                ):
                    match.append(shift)
            days[str(int(day))] = dict(
                comparisons=values,
                matched_shift=int(match[0]) if len(match) == 1 else None,
            )
        invalid = (
            duplicate
            | (d["label"] % 300 != 0)
            | ~np.isfinite(d["values"][:, [0, 3]]).all(axis=1)
            | np.any(d["values"][:, [0, 3]] <= 0, axis=1)
        )
        cohorts[key] = dict(
            rows=len(d["label"]),
            duplicate_rows=int(duplicate.sum()),
            invalid_stock_rows=int(invalid.sum()),
            offgrid_rows=int(np.sum(d["label"] % 300 != 0)),
            days=days,
            flow_matches={
                str(shift): sum(v["matched_shift"] == shift for v in days.values())
                for shift in (-300, 0, None)
            },
        )
    write(
        root / "timestamp-audit.json",
        dict(
            at=datetime.now(timezone.utc).isoformat(),
            cohorts=cohorts,
            observation_time_verified_for_positions=False,
            historical_publication_verified=False,
            flow_alignment_does_not_verify_stock_fields=True,
            protected_data_opened=False,
        ),
    )
    print(
        json.dumps(
            {
                k: {
                    f: v[f]
                    for f in (
                        "rows",
                        "duplicate_rows",
                        "invalid_stock_rows",
                        "offgrid_rows",
                        "flow_matches",
                    )
                }
                for k, v in cohorts.items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "download", "audit"))
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.command](args.root.resolve())
