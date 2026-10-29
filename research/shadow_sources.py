"""Bounded, public GET-only acquisition for a new prospective shadow stream."""

from __future__ import annotations

import hashlib
import json
import math
import time

import requests

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
URLS = {
    "spot_klines": "https://api.binance.com/api/v3/klines",
    "open_interest": "https://fapi.binance.com/futures/data/openInterestHist",
    "position_ratio": "https://fapi.binance.com/futures/data/topLongShortPositionRatio",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def public_get(url, **kwargs):
    """Do not read .netrc, proxy credentials or API-key environment settings."""
    with requests.Session() as session:
        session.trust_env = False
        return session.get(url, **kwargs)


def request_parameters(source, symbol, start_at, end_at, now, recovery=False):
    """Never request a candle/metric before this run's explicit activation."""
    if source not in URLS or symbol not in SYMBOLS:
        raise ValueError("unregistered source or symbol")
    if start_at % 60 or end_at <= start_at:
        raise ValueError("aligned prospective start required")
    if source == "spot_klines":
        stop = min(math.floor(now / 60) * 60, end_at)
        lookback = 67 * 60 if recovery else 3 * 60
        params = {"interval": "1m", "limit": 100}
    else:
        stop = min(math.floor(now / 300) * 300, end_at)
        lookback = 85 * 60 if recovery else 15 * 60
        params = {"period": "5m", "limit": 100}
    begin = max(start_at, stop - lookback)
    if stop <= begin:
        return None
    params.update(
        symbol=symbol,
        startTime=int(begin * 1000),
        endTime=min(
            int(end_at * 1000 - 1), int(stop * 1000 - (source == "spot_klines"))
        ),
    )
    return params


def validate_body(source, body, params, start_at, end_at, received_at):
    if not isinstance(body, list) or len(body) > 100:
        raise ValueError("invalid public response")
    content_times = []
    for row in body:
        if source == "spot_klines":
            if not isinstance(row, list) or len(row) != 12:
                raise ValueError("invalid kline schema")
            stamp = int(row[0]) / 1000
            if stamp % 60 or int(row[6]) != int(row[0]) + 59999:
                raise ValueError("invalid kline interval")
            if stamp + 60 > min(end_at, received_at):
                raise ValueError("unclosed candle")
            numeric = [float(row[k]) for k in (1, 2, 3, 4, 5, 7, 8, 9, 10)]
            if not all(math.isfinite(x) and x >= 0 for x in numeric):
                raise ValueError("invalid candle values")
            if min(numeric[:4]) <= 0 or float(row[9]) > float(row[5]):
                raise ValueError("invalid candle prices or buy volume")
            if (
                float(row[3]) > min(float(row[1]), float(row[4]))
                or float(row[2]) < max(float(row[1]), float(row[4]))
                or float(row[8]) != int(float(row[8]))
            ):
                raise ValueError("inconsistent OHLC or trade count")
        else:
            if not isinstance(row, dict) or row.get("symbol") != params["symbol"]:
                raise ValueError("invalid metric symbol")
            stamp = int(row["timestamp"]) / 1000
            key = "sumOpenInterest" if source == "open_interest" else "longShortRatio"
            value = float(row[key])
            if stamp % 300 or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid metric value or grid")
        if not start_at <= stamp < end_at or stamp > received_at:
            raise ValueError("source outside prospective interval")
        if not params["startTime"] <= stamp * 1000 <= params["endTime"]:
            raise ValueError("source outside requested window")
        content_times.append(stamp)
    if content_times != sorted(set(content_times)):
        raise ValueError("duplicate or unordered response labels")
    return content_times


def fetch(
    source, symbol, start_at, end_at, *, recovery=False, clock=time.time, get=public_get
):
    requested_at = clock()
    params = request_parameters(
        source, symbol, start_at, end_at, requested_at, recovery
    )
    if params is None:
        return None
    # Deliberately no environment credentials, generic URL, POST method or orders.
    response = get(URLS[source], params=params, timeout=(3, 5), allow_redirects=False)
    received_at = clock()
    if response.status_code != 200:
        raise RuntimeError(f"public HTTP status {response.status_code}")
    if len(response.content) > 100_000:
        raise ValueError("oversized public response")
    body = response.json()
    times = validate_body(source, body, params, start_at, end_at, received_at)
    return {
        "source": source,
        "symbol": symbol,
        "requested_at": requested_at,
        "request_params": params,
        "body": body,
        "content_times": times,
        "body_sha256": digest(body),
        "received_at": received_at,
        "http_status": 200,
    }


def clock_probe(*, clock=time.time, get=public_get):
    samples = []
    for url in (
        "https://api.binance.com/api/v3/time",
        "https://fapi.binance.com/fapi/v1/time",
    ):
        before = clock()
        response = get(url, timeout=(3, 5), allow_redirects=False)
        after = clock()
        response.raise_for_status()
        server = int(response.json()["serverTime"]) / 1000
        samples.append(
            {
                "url": url,
                "requested_at": before,
                "received_at": after,
                "server_time": server,
                "rtt": after - before,
                "offset": server - (before + after) / 2,
            }
        )
    return {
        "samples": samples,
        "healthy": all(
            s["rtt"] >= 0 and abs(s["offset"]) + s["rtt"] / 2 <= 2 for s in samples
        ),
    }
