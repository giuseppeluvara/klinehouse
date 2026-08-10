"""Frozen PF_XBTUSD historical-profitability sprint.

The module intentionally contains no private API, WebSocket, paper, or order
execution path.  Generated data lives below the already ignored research/out/.
"""
# ruff: noqa: E402, E701, E702

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import subprocess
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import httpx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out" / "profitability"
PROTOCOL = BASE / "PROFITABILITY_SPRINT.md"
SYMBOL = "PF_XBTUSD"
INSTRUMENTS_URL = "https://futures.kraken.com/derivatives/api/v3/instruments"
CHARTS = "https://futures.kraken.com/api/charts/v1"
FUNDING_URL = (
    "https://assets-cms.kraken.com/files/51n36hrp/facade/"
    "4b70936c1227e4ae5514cba8bf41a5561cf13bd7.zip?dl="
)
MINUTE = 60_000
HOUR = 60 * MINUTE
LOOKBACK = 120
MAX_HORIZON = 4
SEEDS = (7, 17, 29)
FEATURES = (
    "trade_log_return", "mark_log_return", "trade_range", "log_volume",
    "log_trade_count", "aggressor_imbalance", "trade_mark_basis",
)
MODELS = ("ridge", "hgb", "transformer")
HORIZONS = (1, 4)
FUNDING_COLUMNS = ("timestamp", "tradeable", "absolute_rate", "relative_rate")


def utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def last_complete_quarter(now: pd.Timestamp | None = None) -> pd.Timestamp:
    stamp = utc(now or pd.Timestamp.now(tz="UTC"))
    month = 3 * ((stamp.month - 1) // 3) + 1
    return pd.Timestamp(year=stamp.year, month=month, day=1, tz="UTC")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: dict, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")


def git_value(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=BASE, text=True, check=True, capture_output=True,
    ).stdout.strip()


def runtime() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "max_rss_mb": usage.ru_maxrss / (1024 * 1024 if os.uname().sysname == "Darwin" else 1024),
        "utc": datetime.now(timezone.utc).isoformat(),
    }


def get_json(client: httpx.Client, url: str, params: dict | None = None) -> dict:
    for attempt in range(6):
        try:
            response = client.get(url, params=params)
            if response.status_code == 200:
                payload = response.json()
                if payload.get("errors"):
                    raise RuntimeError(f"Kraken errors at {response.url}: {payload['errors']}")
                return payload
            if response.status_code != 429 and response.status_code < 500:
                response.raise_for_status()
        except (httpx.TransportError, httpx.TimeoutException):
            if attempt == 5:
                raise
        time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Kraken request failed after retries: {url}")


def normalized_seconds(values) -> np.ndarray:
    raw = np.asarray(values, dtype=np.int64)
    return np.where(raw > 100_000_000_000, raw // 1000, raw)


def parse_candles(payload: dict, source: str) -> pd.DataFrame:
    rows = payload.get("candles", [])
    columns = ("time", "open", "high", "low", "close", "volume")
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{source} candle schema missing {sorted(missing)}")
    frame = frame.loc[:, columns].copy()
    frame["time"] = normalized_seconds(frame["time"])
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame.iloc[:, 1:].to_numpy(float)).all():
        raise RuntimeError(f"non-finite {source} candle")
    if ((frame["open"] <= 0) | (frame["high"] <= 0) | (frame["low"] <= 0) |
            (frame["close"] <= 0) | (frame["volume"] < 0)).any():
        raise RuntimeError(f"invalid {source} candle")
    return frame


def parse_analytics(payload: dict, source: str) -> pd.DataFrame:
    result = payload.get("result", {})
    timestamps = normalized_seconds(result.get("timestamp", []))
    data = result.get("data", [])
    if source == "trade_count":
        values = np.asarray(data, dtype=float)
        if len(values) != len(timestamps): raise RuntimeError("trade-count length mismatch")
        return pd.DataFrame({"time": timestamps, "trade_count": values})
    if source == "cvd":
        required = ("buy_volume", "sell_volume")
        if not isinstance(data, dict) or not set(required).issubset(data):
            raise RuntimeError("CVD schema mismatch")
        frame = pd.DataFrame({"time": timestamps})
        for column in required:
            values = np.asarray(data[column], dtype=float)
            if len(values) != len(timestamps): raise RuntimeError(f"CVD {column} length mismatch")
            frame[column] = values
        return frame
    if source == "funding":
        if not isinstance(data, dict) or "relativeRate" not in data:
            raise RuntimeError("funding schema mismatch")
        values = data["relativeRate"]
        close = np.asarray([float(row[-1] if isinstance(row, list) else row) for row in values])
        if len(close) != len(timestamps): raise RuntimeError("funding length mismatch")
        return pd.DataFrame({"time": timestamps, "relative_rate": close})
    raise ValueError(source)


def instrument(client: httpx.Client) -> dict:
    payload = get_json(client, INSTRUMENTS_URL)
    matches = [row for row in payload.get("instruments", []) if row.get("symbol") == SYMBOL]
    if len(matches) != 1: raise RuntimeError(f"expected exactly one {SYMBOL} instrument")
    return matches[0]


def validate_instrument(row: dict) -> dict:
    platforms = set(row.get("platformsPermitted", []))
    checks = {
        "symbol": row.get("symbol") == SYMBOL,
        "linear_type": row.get("type") == "flexible_futures",
        "base_quote": row.get("base") == "BTC" and row.get("quote") == "USD",
        "contract_size": float(row.get("contractSize", math.nan)) == 1.0,
        "lot_precision": int(row.get("contractValueTradePrecision", -1)) == 4,
        "tradeable": row.get("tradeable") is True,
        "not_expired": row.get("isExpired") is False,
        "eea_platform": bool(platforms & {"europa", "europa_crypto"}),
        "no_declared_country_ban": not row.get("countriesBanned", []),
    }
    return {"passed": all(checks.values()), "checks": checks}


def first_market_rows(client: httpx.Client, opening: pd.Timestamp) -> dict:
    since = int((opening - pd.Timedelta(days=1)).timestamp())
    end = int((opening + pd.Timedelta(days=14)).timestamp())
    result = {}
    for source in ("trade", "mark"):
        payload = get_json(client, f"{CHARTS}/{source}/{SYMBOL}/1m", {"from": since, "to": end})
        frame = parse_candles(payload, source)
        result[source] = int(frame.time.min()) if len(frame) else None
    for source, kind in (("trade_count", "trade-count"), ("cvd", "cvd")):
        payload = get_json(client, f"{CHARTS}/analytics/{SYMBOL}/{kind}",
                           {"since": since, "to": end, "interval": 60})
        frame = parse_analytics(payload, source)
        result[source] = int(frame.time.min()) if len(frame) else None
    return result


def first_funding_api(client: httpx.Client, opening: pd.Timestamp,
                      cutoff: pd.Timestamp) -> pd.DataFrame:
    cursor, stop, frames = int(opening.timestamp()), int(cutoff.timestamp()), []
    while cursor < stop:
        payload = get_json(client, f"{CHARTS}/analytics/{SYMBOL}/funding", {
            "since": cursor, "to": stop, "interval": 3600,
        })
        frame = parse_analytics(payload, "funding")
        if frame.empty: break
        frame = frame[(frame.time >= cursor) & (frame.time <= stop)]
        if frame.empty: break
        frames.append(frame)
        following = int(frame.time.max()) + 3600
        if following <= cursor: raise RuntimeError("no funding pagination progress")
        cursor = following
        if not payload.get("result", {}).get("more"): break
    if not frames: return pd.DataFrame(columns=("time", "relative_rate"))
    return pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time")


def download_funding_export(client: httpx.Client) -> tuple[Path, Path]:
    folder = OUT / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    archive_path, csv_path = folder / "funding-rates.zip", folder / f"{SYMBOL}.csv"
    if not archive_path.exists():
        response = client.get(FUNDING_URL)
        response.raise_for_status()
        partial = archive_path.with_suffix(".part")
        partial.write_bytes(response.content)
        with zipfile.ZipFile(partial) as archive:
            archive.getinfo(f"exports/{SYMBOL}.csv")
        partial.replace(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read(f"exports/{SYMBOL}.csv")
    if not csv_path.exists() or csv_path.read_bytes() != payload:
        partial_csv = csv_path.with_suffix(".part")
        partial_csv.write_bytes(payload)
        partial_csv.replace(csv_path)
    return archive_path, csv_path


def read_funding_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if tuple(frame.columns) != FUNDING_COLUMNS:
        raise RuntimeError(f"funding CSV schema changed: {tuple(frame.columns)}")
    if set(frame.tradeable) != {SYMBOL}: raise RuntimeError("funding CSV contains wrong product")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    frame["relative_rate"] = pd.to_numeric(frame.relative_rate, errors="raise")
    if frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
        raise RuntimeError("funding CSV timestamps are not unique and ordered")
    if not np.isfinite(frame.relative_rate).all(): raise RuntimeError("non-finite funding rate")
    return frame


def funding_precheck(export: pd.DataFrame, api: pd.DataFrame,
                     start: pd.Timestamp, cutoff: pd.Timestamp) -> dict:
    api_times = pd.to_datetime(api.time, unit="s", utc=True) if len(api) else pd.DatetimeIndex([])
    export_times = export.timestamp
    overlap = export.merge(
        pd.DataFrame({"timestamp": api_times, "api_rate": api.relative_rate.to_numpy()}),
        on="timestamp", how="inner",
    )
    max_difference = None if overlap.empty else float(
        np.max(np.abs(overlap.relative_rate - overlap.api_rate))
    )
    export_first = export_times.min()
    export_last = export_times.max()
    api_first = api_times.min() if len(api_times) else None
    api_last = api_times.max() if len(api_times) else None
    gap_hours = None
    if api_first is not None and export_last < api_first:
        gap_hours = float((api_first - export_last) / pd.Timedelta(hours=1))
    covers_start = export_first <= start
    covers_end = (export_last >= cutoff) or (api_last is not None and api_last >= cutoff)
    continuous_join = api_first is not None and export_last >= api_first
    overlap_agrees = bool(len(overlap)) and max_difference <= 1e-12
    checks = {
        "covers_common_start": bool(covers_start),
        "covers_cutoff": bool(covers_end),
        "export_api_overlap": bool(continuous_join and len(overlap)),
        "overlap_rate_tolerance_1e-12": bool(overlap_agrees),
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "export_first": export_first.isoformat(), "export_last": export_last.isoformat(),
        "api_first": api_first.isoformat() if api_first is not None else None,
        "api_last": api_last.isoformat() if api_last is not None else None,
        "gap_hours": gap_hours, "overlap_rows": int(len(overlap)),
        "overlap_max_abs_difference": max_difference,
    }


def prior_freeze() -> dict:
    paths = (
        BASE / "PLAN.md", BASE / "RAPID_EVIDENCE.md",
        BASE / "research" / "sequence_experiment.py",
        BASE / "research" / "rapid_evidence.py",
        BASE / "tests" / "test_sequence_experiment.py",
        BASE / "tests" / "test_rapid_evidence.py",
    )
    return {str(path.relative_to(BASE)): sha(path) for path in paths}


def audit(cutoff: str | None = None, client_factory: Callable = httpx.Client) -> dict:
    started = time.monotonic()
    if (OUT / "manifest.json").exists():
        raise FileExistsError("profitability sprint is already frozen")
    end = utc(cutoff) if cutoff else last_complete_quarter()
    if end != last_complete_quarter(end): raise ValueError("cutoff must start a UTC quarter")
    with client_factory(timeout=120, follow_redirects=True) as client:
        product = instrument(client)
        product_check = validate_instrument(product)
        opening = utc(product["openingDate"])
        first = first_market_rows(client, opening)
        first_values = [value for value in first.values() if value is not None]
        common_first = pd.to_datetime(max(first_values), unit="s", utc=True)
        quarter_month = 3 * ((common_first.month - 1) // 3) + 1
        quarter_start = pd.Timestamp(common_first.year, quarter_month, 1, tz="UTC")
        common_start = quarter_start + pd.DateOffset(months=3)
        funding_api = first_funding_api(client, opening, end)
        archive, csv_path = download_funding_export(client)
    funding = read_funding_csv(csv_path)
    funding_check = funding_precheck(funding, funding_api, common_start, end)
    prechecks = {
        "instrument": product_check["passed"],
        "four_market_sources_found": len(first_values) == 4,
        "at_least_24_months": end >= common_start + pd.DateOffset(months=24),
        "funding": funding_check["passed"],
    }
    status = "precheck_pass" if all(prechecks.values()) else "data_no_go"
    report = {
        "status": status, "prechecks": prechecks, "product": product,
        "product_validation": product_check,
        "first_available": {
            key: pd.to_datetime(value, unit="s", utc=True).isoformat() if value else None
            for key, value in first.items()
        },
        "common_start": common_start.isoformat(), "cutoff": end.isoformat(),
        "funding": funding_check,
        "sources": {
            "funding_zip": {"url": FUNDING_URL, "bytes": archive.stat().st_size, "sha256": sha(archive)},
            "funding_csv": {"bytes": csv_path.stat().st_size, "sha256": sha(csv_path)},
        },
        "note": (
            "Market-source full downloads and training are forbidden after a decisive precheck no-go."
            if status == "data_no_go" else "Run prepare for full minute coverage and alignment audit."
        ),
        "elapsed_seconds": time.monotonic() - started, **runtime(),
    }
    manifest = {
        "state": status, "protocol_sha256": sha(PROTOCOL),
        "base_commit": git_value("rev-parse", "HEAD"), "prior_freeze": prior_freeze(),
        "audit_sha256": None, "created_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL, "common_start": common_start.isoformat(), "cutoff": end.isoformat(),
    }
    write_json(OUT / "audit.json", report, exclusive=True)
    manifest["audit_sha256"] = sha(OUT / "audit.json")
    write_json(OUT / "manifest.json", manifest, exclusive=True)
    return report


def _more(payload: dict, source: str) -> bool:
    return bool(payload.get("more_candles")) if source in {"trade", "mark"} else bool(
        payload.get("result", {}).get("more")
    )


def download_series(client: httpx.Client, source: str, start: pd.Timestamp,
                    end: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    cursor, stop, frames, raw_count = int(start.timestamp()), int(end.timestamp()), [], 0
    while cursor < stop:
        if source in {"trade", "mark"}:
            payload = get_json(client, f"{CHARTS}/{source}/{SYMBOL}/1m",
                               {"from": cursor, "to": stop})
            frame = parse_candles(payload, source)
        else:
            kind = "trade-count" if source == "trade_count" else "cvd"
            payload = get_json(client, f"{CHARTS}/analytics/{SYMBOL}/{kind}",
                               {"since": cursor, "to": stop, "interval": 60})
            frame = parse_analytics(payload, source)
        if frame.empty: break
        frame = frame[(frame.time >= cursor) & (frame.time < stop)]
        if frame.empty: break
        frames.append(frame); raw_count += len(frame)
        next_cursor = int(frame.time.max()) + 60
        if next_cursor <= cursor: raise RuntimeError(f"no pagination progress for {source}")
        cursor = next_cursor
        if not _more(payload, source) and cursor < stop:
            # A short page can still be followed by a documented gap; a direct
            # jump would hide it, so terminate and let coverage fail.
            break
    if not frames: return pd.DataFrame(), 0
    combined = pd.concat(frames, ignore_index=True).sort_values("time")
    combined = combined.drop_duplicates("time", keep="first").reset_index(drop=True)
    return combined, raw_count


def quarter_ranges(start: pd.Timestamp, end: pd.Timestamp):
    cursor = start
    while cursor < end:
        following = cursor + pd.DateOffset(months=3)
        yield cursor, min(following, end)
        cursor = following


def coverage_report(frames: dict[str, pd.DataFrame], start: pd.Timestamp,
                    end: pd.Timestamp) -> dict:
    report = {}
    for source, frame in frames.items():
        times = np.sort(frame.time.to_numpy(np.int64))
        quarters = []
        for q0, q1 in quarter_ranges(start, end):
            expected = int((q1 - q0) / pd.Timedelta(minutes=1))
            lo, hi = int(q0.timestamp()), int(q1.timestamp())
            observed = int(np.searchsorted(times, hi) - np.searchsorted(times, lo))
            quarters.append({"start": q0.isoformat(), "end": q1.isoformat(),
                             "expected": expected, "observed": observed,
                             "coverage": observed / expected})
        report[source] = quarters
    return report


def aligned_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    trade = frames["trade"].rename(columns={column: f"trade_{column}" for column in
        ("open", "high", "low", "close", "volume")})
    mark = frames["mark"].rename(columns={column: f"mark_{column}" for column in
        ("open", "high", "low", "close", "volume")})
    frame = trade.merge(mark, on="time", how="inner", validate="one_to_one")
    frame = frame.merge(frames["trade_count"], on="time", how="inner", validate="one_to_one")
    frame = frame.merge(frames["cvd"], on="time", how="inner", validate="one_to_one")
    frame = frame.sort_values("time").reset_index(drop=True)
    contiguous = np.r_[False, np.diff(frame.time.to_numpy(np.int64)) == 60]
    trade_close, mark_close = frame.trade_close.to_numpy(float), frame.mark_close.to_numpy(float)
    trade_return = np.full(len(frame), np.nan)
    mark_return = np.full(len(frame), np.nan)
    indexes = np.flatnonzero(contiguous)
    trade_return[indexes] = np.log(trade_close[indexes] / trade_close[indexes - 1])
    mark_return[indexes] = np.log(mark_close[indexes] / mark_close[indexes - 1])
    buy, sell = frame.buy_volume.to_numpy(float), frame.sell_volume.to_numpy(float)
    total = buy + sell
    imbalance = np.divide(buy - sell, total, out=np.zeros(len(frame)), where=total > 0)
    values = np.column_stack((
        trade_return, mark_return,
        (frame.trade_high.to_numpy(float) - frame.trade_low.to_numpy(float)) / frame.trade_open,
        np.log1p(frame.trade_volume.to_numpy(float)), np.log1p(frame.trade_count.to_numpy(float)),
        imbalance, trade_close / mark_close - 1,
    ))
    for index, name in enumerate(FEATURES): frame[name] = values[:, index]
    return frame


def aggregate_28(sequences: np.ndarray) -> np.ndarray:
    values = np.asarray(sequences, dtype=np.float64)
    t = np.arange(values.shape[1], dtype=float)
    centered = t - t.mean()
    slope = np.einsum("ntc,t->nc", values, centered) / np.square(centered).sum()
    return np.concatenate((values[:, -1], values.mean(1), values.std(1), slope), axis=1).astype(np.float32)


def build_arrays(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, np.ndarray]:
    times = frame.time.to_numpy(np.int64)
    lookup = {int(value): index for index, value in enumerate(times)}
    feature_values = frame.loc[:, FEATURES].to_numpy(np.float32)
    fields = {name: [] for name in (
        "x", "anchor", "entry", "normal_exit_1", "normal_exit_4",
        "stop_time_1", "stop_time_4", "stop_exit_1", "stop_exit_4",
        "target_1", "target_4",
    )}
    first, stop = int(start.timestamp()), int(end.timestamp())
    for anchor in range(first, stop, 3600):
        if anchor + MAX_HORIZON * 3600 > stop:
            continue
        support_times = np.arange(anchor - LOOKBACK * 60, anchor, 60, dtype=np.int64)
        future_times = np.arange(anchor, anchor + MAX_HORIZON * 3600 + 60, 60, dtype=np.int64)
        try:
            support = np.asarray([lookup[int(value)] for value in support_times])
            future = np.asarray([lookup[int(value)] for value in future_times])
        except KeyError:
            continue
        sequence = feature_values[support]
        future_frame = frame.iloc[future]
        if not np.isfinite(sequence).all(): continue
        entry = float(future_frame.trade_open.iloc[0])
        if not np.isfinite(entry) or entry <= 0: continue
        row = {"x": sequence, "anchor": anchor, "entry": entry}
        valid = True
        for horizon in HORIZONS:
            minutes = horizon * 60
            normal_exit = float(future_frame.trade_close.iloc[minutes - 1])
            scan = future_frame.iloc[:minutes]
            triggered = np.flatnonzero(scan.mark_low.to_numpy(float) <= entry * 0.99)
            if len(triggered):
                position = int(triggered[0])
                next_open = float(future_frame.trade_open.iloc[position + 1])
                stop_exit = min(entry * 0.99, next_open)
                stop_time = anchor + (position + 1) * 60
            else:
                stop_exit, stop_time = math.nan, -1
            if not np.isfinite(normal_exit) or (len(triggered) and not np.isfinite(stop_exit)):
                valid = False; break
            row[f"normal_exit_{horizon}"] = normal_exit
            row[f"stop_time_{horizon}"] = stop_time
            row[f"stop_exit_{horizon}"] = stop_exit
            row[f"target_{horizon}"] = 10_000 * (normal_exit / entry - 1)
        if valid:
            for name in fields: fields[name].append(row[name])
    result = {name: np.asarray(value, np.float32 if name == "x" else np.float64)
              for name, value in fields.items()}
    if not len(result["x"]):
        result["x"] = np.empty((0, LOOKBACK, len(FEATURES)), dtype=np.float32)
    result["anchor"] = result["anchor"].astype(np.int64)
    for name in ("stop_time_1", "stop_time_4"): result[name] = result[name].astype(np.int64)
    result["aggregate"] = aggregate_28(result["x"])
    return result


def prepare(client_factory: Callable = httpx.Client) -> dict:
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["protocol_sha256"] != sha(PROTOCOL): raise RuntimeError("protocol changed after freeze")
    if manifest["state"] != "precheck_pass": raise RuntimeError(f"data gate forbids prepare: {manifest['state']}")
    start, end = utc(manifest["common_start"]), utc(manifest["cutoff"])
    frames, records = {}, {}
    with client_factory(timeout=120, follow_redirects=True) as client:
        for source in ("trade", "mark", "trade_count", "cvd"):
            frame, raw_count = download_series(client, source, start, end)
            path = OUT / "sources" / f"{source}.parquet"; frame.to_parquet(path, index=False)
            frames[source] = frame
            records[source] = {"raw_rows": raw_count, "rows": len(frame), "sha256": sha(path)}
    coverage = coverage_report(frames, start, end)
    aligned = aligned_features(frames)
    arrays = build_arrays(aligned, start, end)
    folds = temporal_folds(arrays["anchor"], start, end)
    theoretical = {fold.number: int((fold.end - fold.start) / pd.Timedelta(hours=1)) for fold in folds}
    anchor_coverage = {fold.number: len(fold.validation) / theoretical[fold.number] for fold in folds}
    source_pass = all(row["coverage"] >= 0.995 for rows in coverage.values() for row in rows)
    anchor_pass = len(folds) >= 5 and all(value >= 0.95 for value in anchor_coverage.values())
    if not source_pass or not anchor_pass:
        status = "data_no_go"
    else:
        status = "data_gate_pass"
        dataset = OUT / "dataset.npz"; np.savez_compressed(dataset, **arrays)
        records["dataset"] = {"sha256": sha(dataset), "array_hash": array_hash(*arrays.values())}
    report = {"status": status, "coverage": coverage, "anchor_coverage": anchor_coverage,
              "sources": records, "folds": [fold.as_dict() for fold in folds]}
    write_json(OUT / "prepare.json", report, exclusive=True)
    manifest["state"] = status; manifest["prepare_sha256"] = sha(OUT / "prepare.json")
    manifest["sources"] = records
    write_json(manifest_path, manifest)
    return report


@dataclass(frozen=True)
class Fold:
    number: int
    start: pd.Timestamp
    end: pd.Timestamp
    train: np.ndarray
    validation: np.ndarray

    def as_dict(self) -> dict:
        return {"number": self.number, "start": self.start.isoformat(), "end": self.end.isoformat(),
                "train": len(self.train), "validation": len(self.validation)}


def temporal_folds(anchor: np.ndarray, start: pd.Timestamp, end: pd.Timestamp) -> list[Fold]:
    validation_start = start + pd.DateOffset(months=24)
    result, number = [], 1
    while validation_start + pd.DateOffset(months=3) <= end:
        validation_end = validation_start + pd.DateOffset(months=3)
        v0, v1 = int(validation_start.timestamp()), int(validation_end.timestamp())
        validation = np.flatnonzero((anchor >= v0) & (anchor + MAX_HORIZON * 3600 <= v1))
        # Protect validation lookback from every training label using the strict 4h end.
        train = np.flatnonzero(anchor + MAX_HORIZON * 3600 <= v0 - LOOKBACK * 60)
        result.append(Fold(number, validation_start, validation_end, train, validation))
        number += 1; validation_start = validation_end
    return result


def quantity(entry_price: float) -> float:
    return math.floor((100.0 / entry_price) / 0.0001 + 1e-12) * 0.0001


def funding_cost(funding: pd.DataFrame, entry_time: int, exit_time: int,
                 notional: float) -> tuple[float, float, int]:
    elapsed_hours = (exit_time - entry_time) / 3600
    rows = funding[(funding.time > entry_time) & (funding.time <= exit_time)]
    actual_rate = float(np.maximum(rows.relative_rate.to_numpy(float), 0).sum())
    applied_rate = max(actual_rate, 0.0001 * elapsed_hours)
    return notional * applied_rate, applied_rate, len(rows)


def trade_economics(entry: float, exit_price: float, qty: float, stopped: bool,
                    funding_usd: float) -> dict:
    entry_notional, exit_notional = qty * entry, qty * exit_price
    gross = qty * (exit_price - entry)
    fees = entry_notional * 0.0002 + exit_notional * (0.0005 if stopped else 0.0002)
    adverse = entry_notional * 0.0004
    net = gross - fees - adverse - funding_usd
    return {
        "quantity_btc": qty, "entry_notional_usd": entry_notional,
        "exit_notional_usd": exit_notional, "gross_usd": gross, "fees_usd": fees,
        "adverse_selection_usd": adverse, "funding_usd": funding_usd, "net_usd": net,
        "gross_bps": gross / entry_notional * 10_000,
        "net_bps": net / entry_notional * 10_000,
    }


def fit_scaler(x: np.ndarray, axes=(0,)) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(x).mean(axis=axes, dtype=np.float64)
    std = np.asarray(x).std(axis=axes, dtype=np.float64)
    std = np.where(std == 0, 1, std)
    return mean.astype(np.float32), std.astype(np.float32)


def make_transformer(channels: int = len(FEATURES)):
    import torch
    from torch import nn

    class Regressor(nn.Module):
        def __init__(self):
            super().__init__(); width = 32
            self.project = nn.Linear(channels, width)
            position = torch.arange(LOOKBACK).unsqueeze(1)
            scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10_000) / width))
            encoding = torch.zeros(LOOKBACK, width)
            encoding[:, 0::2] = torch.sin(position * scale)
            encoding[:, 1::2] = torch.cos(position * scale)
            self.register_buffer("position", encoding)
            layer = nn.TransformerEncoderLayer(width, 4, 64, 0.1, batch_first=True,
                                               norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Linear(width, 1)

        def forward(self, values):
            return self.head(self.encoder(self.project(values) + self.position).mean(1)).squeeze(1)

    return Regressor()


def set_seed(seed: int, deterministic: bool = True) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(deterministic)


def train_transformer(x: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray,
                      seed: int, device: str, epochs: int = 10, batch: int = 256):
    import torch
    from torch import nn
    set_seed(seed, device == "cpu")
    model = make_transformer(x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    loss_function = nn.SmoothL1Loss(beta=20.0)
    generator = np.random.default_rng(seed)
    for _ in range(epochs):
        order = generator.permutation(len(x)); model.train()
        for offset in range(0, len(order), batch):
            rows = order[offset:offset + batch]
            xb = ((np.asarray(x[rows]) - mean) / std).astype(np.float32)
            inputs = torch.from_numpy(xb).to(device)
            target = torch.from_numpy(np.asarray(y[rows], np.float32)).to(device)
            optimizer.zero_grad(); loss = loss_function(model(inputs), target)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    return model


def transformer_predict(model, x: np.ndarray, mean: np.ndarray, std: np.ndarray,
                        device: str, batch: int = 256) -> np.ndarray:
    import torch
    model.eval(); output = []
    with torch.no_grad():
        for offset in range(0, len(x), batch):
            values = ((np.asarray(x[offset:offset + batch]) - mean) / std).astype(np.float32)
            output.append(model(torch.from_numpy(values).to(device)).cpu().numpy())
    return np.concatenate(output)


def fit_predictions(kind: str, arrays: dict, train: np.ndarray, calibration: np.ndarray,
                    validation: np.ndarray, horizon: int, device: str):
    y = arrays[f"target_{horizon}"][train]
    if kind == "ridge":
        scaler = StandardScaler().fit(arrays["aggregate"][train])
        model = Ridge(alpha=1.0).fit(scaler.transform(arrays["aggregate"][train]), y)
        def predict(rows): return model.predict(scaler.transform(arrays["aggregate"][rows]))
        return predict(calibration), predict(validation), {"model": model, "scaler": scaler}
    if kind == "hgb":
        model = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=31,
            learning_rate=0.05, l2_regularization=1.0, random_state=7).fit(
                arrays["aggregate"][train], y)
        def predict(rows): return model.predict(arrays["aggregate"][rows])
        return predict(calibration), predict(validation), {"model": model}
    mean, std = fit_scaler(arrays["x"][train], axes=(0, 1))
    calibration_predictions, validation_predictions, models = [], [], []
    for seed in SEEDS:
        model = train_transformer(arrays["x"][train], y, mean, std, seed, device)
        calibration_predictions.append(transformer_predict(model, arrays["x"][calibration], mean, std, device))
        validation_predictions.append(transformer_predict(model, arrays["x"][validation], mean, std, device))
        models.append(model)
    return (np.mean(calibration_predictions, axis=0), np.mean(validation_predictions, axis=0),
            {"models": models, "mean": mean, "std": std})


def backtest(arrays: dict, rows: np.ndarray, scores: np.ndarray, threshold: float,
             horizon: int, fold: int, funding: pd.DataFrame) -> pd.DataFrame:
    trades, next_entry = [], -1
    for row, score in zip(rows, scores):
        anchor = int(arrays["anchor"][row])
        if score <= threshold or anchor < next_entry: continue
        entry = float(arrays["entry"][row]); qty = quantity(entry)
        if qty < 0.0001: continue
        stopped = int(arrays[f"stop_time_{horizon}"][row]) >= 0
        if stopped:
            exit_time = int(arrays[f"stop_time_{horizon}"][row])
            exit_price = float(arrays[f"stop_exit_{horizon}"][row])
        else:
            exit_time = anchor + horizon * 3600
            exit_price = float(arrays[f"normal_exit_{horizon}"][row])
        funding_usd, funding_rate, funding_rows = funding_cost(
            funding, anchor, exit_time, qty * entry,
        )
        economics = trade_economics(entry, exit_price, qty, stopped, funding_usd)
        trades.append({
            "fold": fold, "anchor": anchor, "exit_time": exit_time, "score": float(score),
            "threshold": threshold, "entry_price": entry, "exit_price": exit_price,
            "stopped": stopped, "funding_rate": funding_rate, "funding_rows": funding_rows,
            **economics,
        })
        next_entry = exit_time
    return pd.DataFrame(trades)


def bootstrap_ci(ledger: pd.DataFrame, draws: int = 10_000, seed: int = 7) -> dict:
    if ledger.empty: return {"ci95": [math.nan, math.nan], "ci9917": [math.nan, math.nan]}
    rows = ledger.copy()
    rows["week"] = (rows.anchor.astype(np.int64) // (7 * 86400)).astype(str)
    blocks = {(fold, week): group.net_bps.to_numpy(float) for (fold, week), group in rows.groupby(["fold", "week"])}
    fold_keys = {fold: sorted([key for key in blocks if key[0] == fold]) for fold in sorted(rows.fold.unique())}
    generator = np.random.default_rng(seed); estimates = np.empty(draws)
    for draw in range(draws):
        values = []
        for keys in fold_keys.values():
            selected = generator.choice(len(keys), len(keys), replace=True)
            values.extend(blocks[keys[index]] for index in selected)
        estimates[draw] = np.concatenate(values).mean()
    return {
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "ci9917": np.quantile(estimates, [0.004167, 0.995833]).tolist(),
        "draws": draws, "seed": seed,
    }


def candidate_summary(ledger: pd.DataFrame, expected_folds: list[int]) -> dict:
    if ledger.empty:
        fold_net, fold_counts = {}, {}
        positive = []
    else:
        fold_net = ledger.groupby("fold").net_usd.sum().to_dict()
        fold_counts = ledger.groupby("fold").size().to_dict()
        positive = [fold for fold in expected_folds if fold_net.get(fold, 0) > 0]
    gross_profit = float(ledger.loc[ledger.net_usd > 0, "net_usd"].sum()) if len(ledger) else 0
    gross_loss = float(-ledger.loc[ledger.net_usd < 0, "net_usd"].sum()) if len(ledger) else 0
    profit_factor = gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else 0)
    cumulative = ledger.net_usd.cumsum() if len(ledger) else pd.Series(dtype=float)
    drawdown = cumulative - cumulative.cummax() if len(cumulative) else cumulative
    intervals = bootstrap_ci(ledger)
    checks = {
        "at_least_100_trades": len(ledger) >= 100,
        "ten_trades_in_five_folds": sum(fold_counts.get(fold, 0) >= 10 for fold in expected_folds) >= 5,
        "positive_in_five_folds": len(positive) >= 5,
        "positive_in_last_two_folds": all(fold_net.get(fold, 0) > 0 for fold in expected_folds[-2:]),
        "ci9917_lower_above_zero": intervals["ci9917"][0] > 0,
        "profit_factor_at_least_1_1": profit_factor >= 1.1,
    }
    return {
        "trades": int(len(ledger)), "gross_usd": float(ledger.gross_usd.sum()) if len(ledger) else 0,
        "net_usd": float(ledger.net_usd.sum()) if len(ledger) else 0,
        "mean_gross_bps": float(ledger.gross_bps.mean()) if len(ledger) else math.nan,
        "mean_net_bps": float(ledger.net_bps.mean()) if len(ledger) else math.nan,
        "fees_usd": float(ledger.fees_usd.sum()) if len(ledger) else 0,
        "funding_usd": float(ledger.funding_usd.sum()) if len(ledger) else 0,
        "adverse_selection_usd": float(ledger.adverse_selection_usd.sum()) if len(ledger) else 0,
        "profit_factor": profit_factor,
        "max_drawdown_usd": float(drawdown.min()) if len(drawdown) else 0,
        "fold_counts": fold_counts, "fold_net_usd": fold_net,
        **intervals, "gate": checks, "passed": all(checks.values()),
    }


def select_candidate(summaries: dict[str, dict]) -> str | None:
    passed = [name for name, value in summaries.items() if value["passed"]]
    if not passed: return None
    best_lower = max(summaries[name]["ci9917"][0] for name in passed)
    tied = [name for name in passed if summaries[name]["ci9917"][0] >= best_lower - 1.0]
    priority = {"ridge": 0, "hgb": 1, "transformer": 2}
    return min(tied, key=lambda name: (priority[name.rsplit("_", 1)[0]], int(name.rsplit("_", 1)[1][:-1])))


def choose_device(requested: str) -> str:
    if requested != "auto": return requested
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def run(device: str = "auto") -> dict:
    started = time.monotonic(); result_path = OUT / "report.json"
    if result_path.exists(): raise FileExistsError("historical results already exist")
    manifest = json.loads((OUT / "manifest.json").read_text())
    if manifest["protocol_sha256"] != sha(PROTOCOL): raise RuntimeError("protocol changed after freeze")
    if manifest["state"] != "data_gate_pass": raise RuntimeError(f"data gate forbids training: {manifest['state']}")
    dataset = OUT / "dataset.npz"
    if sha(dataset) != manifest["sources"]["dataset"]["sha256"]: raise RuntimeError("dataset hash changed")
    loaded = np.load(dataset); arrays = {name: loaded[name] for name in loaded.files}
    funding_csv = OUT / "sources" / f"{SYMBOL}.csv"
    funding = read_funding_csv(funding_csv)
    funding = pd.DataFrame({"time": funding.timestamp.astype("int64") // 1_000_000_000,
                            "relative_rate": funding.relative_rate})
    start, end = utc(manifest["common_start"]), utc(manifest["cutoff"])
    folds = temporal_folds(arrays["anchor"], start, end)
    device = choose_device(device); summaries, ledgers = {}, {}
    for kind in MODELS:
        for horizon in HORIZONS:
            name = f"{kind}_{horizon}h"; pieces = []
            for fold in folds:
                calibration_start = int((fold.start - pd.DateOffset(months=6)).timestamp())
                calibration = fold.train[arrays["anchor"][fold.train] >= calibration_start]
                cal_score, val_score, _ = fit_predictions(
                    kind, arrays, fold.train, calibration, fold.validation, horizon, device,
                )
                threshold = float(np.quantile(cal_score, 0.975))
                pieces.append(backtest(arrays, fold.validation, val_score, threshold,
                                       horizon, fold.number, funding))
            ledger = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
            ledger_path = OUT / "results" / f"{name}.parquet"
            ledger_path.parent.mkdir(parents=True, exist_ok=True); ledger.to_parquet(ledger_path, index=False)
            ledgers[name] = {"sha256": sha(ledger_path), "path": str(ledger_path.relative_to(BASE))}
            summaries[name] = candidate_summary(ledger, [fold.number for fold in folds])
    winner = select_candidate(summaries)
    report = {
        "status": "historical_go" if winner else "historical_no_go", "winner": winner,
        "conditional_on_maker_fill": True, "candidates": summaries,
        "folds": [fold.as_dict() for fold in folds], "ledgers": ledgers,
        "device": device, "elapsed_seconds": time.monotonic() - started, **runtime(),
    }
    write_json(result_path, report, exclusive=True)
    manifest["state"] = report["status"]; manifest["report_sha256"] = sha(result_path)
    manifest["ledgers"] = ledgers
    if winner:
        # The exact fold models remain the evidentiary bundle. A deployment
        # refit is intentionally deferred until a separately authorized sprint.
        bundle = {"winner": winner, "protocol_sha256": sha(PROTOCOL),
                  "dataset_sha256": sha(dataset), "report_sha256": sha(result_path)}
        bundle_path = OUT / "bundle.json"; write_json(bundle_path, bundle, exclusive=True)
        manifest["bundle_sha256"] = sha(bundle_path)
    write_json(OUT / "manifest.json", manifest)
    return report


def verify() -> dict:
    manifest = json.loads((OUT / "manifest.json").read_text())
    checks = {
        "protocol": manifest["protocol_sha256"] == sha(PROTOCOL),
        "audit": manifest["audit_sha256"] == sha(OUT / "audit.json"),
        "base_commit_exists": bool(git_value("cat-file", "-t", manifest["base_commit"])),
        "prior_freeze": manifest["prior_freeze"] == prior_freeze(),
    }
    for key in ("prepare", "report"):
        if f"{key}_sha256" in manifest:
            checks[key] = manifest[f"{key}_sha256"] == sha(OUT / f"{key}.json")
    for name, record in manifest.get("sources", {}).items():
        if "path" in record: checks[f"source_{name}"] = sha(BASE / record["path"]) == record["sha256"]
    result = {"passed": all(checks.values()), "state": manifest["state"], "checks": checks, **runtime()}
    if not result["passed"]: raise RuntimeError(f"verification failed: {checks}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit"); audit_parser.add_argument("--cutoff")
    commands.add_parser("prepare")
    run_parser = commands.add_parser("run"); run_parser.add_argument("--device", default="auto")
    commands.add_parser("verify")
    args = parser.parse_args(argv)
    if args.command == "audit": result = audit(args.cutoff)
    elif args.command == "prepare": result = prepare()
    elif args.command == "run": result = run(args.device)
    else: result = verify()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
