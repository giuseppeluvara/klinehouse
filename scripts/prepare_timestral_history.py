#!/usr/bin/env python3
"""Build a separate native-hourly research source; never mix it into the 1s archive."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import argparse
import csv
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
from research import timestral_experiment as shared  # noqa: E402


def timestamp(value):
    number = int(value)
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
        microseconds=number if number >= 10**15 else number * 1000
    )


def decode_row(values, year, month):
    """Retain source anomalies as gaps, while rejecting misplaced timestamps."""
    if len(values) != 12:
        raise ValueError("invalid kline width")
    opened, closed = timestamp(values[0]), timestamp(values[6])
    if (
        opened.year != year
        or opened.month != month
        or opened >= shared.utc("2026-08-01")
        or opened.minute
        or opened.second
        or opened.microsecond
    ):
        raise ValueError("misplaced or protected native-hourly candle")
    o, high, low, close, volume = map(float, values[1:6])
    trades, taker = float(values[8]), float(values[9])
    if (
        not opened <= closed < opened + timedelta(hours=1)
        or not np.isfinite([o, high, low, close, volume, trades, taker]).all()
        or not 0 < low <= min(o, close) <= max(o, close) <= high
        or min(volume, trades, taker) < 0
        or taker > volume + 1e-8
    ):
        return None
    return {
        "time": shared.stamp(opened),
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "trades": trades,
        "taker": taker,
    }


def prepare(output, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    if any((output / n).exists() for n in ("hours.jsonl", "provenance.json")):
        raise ValueError("will not overwrite a normalized source")

    def download(number):
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
        year, month = number // 12, number % 12 + 1
        name = f"BTCUSDT-1h-{year}-{month:02d}.zip"
        url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/" + name
        checksum = session.get(url + ".CHECKSUM", timeout=30)
        checksum.raise_for_status()
        target = output / name
        if resume and target.exists():
            data = target.read_bytes()
        else:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            data = response.content
        digest = hashlib.sha256(data).hexdigest()
        if digest != checksum.text.split()[0] or len(data) > 2_000_000:
            raise ValueError("corrupt or oversized monthly file")
        if not target.exists():
            with target.open("xb") as handle:
                handle.write(data)
        receipt = output / (name + ".CHECKSUM")
        if receipt.exists():
            if receipt.read_text().split()[0] != digest:
                raise ValueError("saved checksum changed")
        else:
            with receipt.open("x") as handle:
                handle.write(checksum.text)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if (
                len(archive.infolist()) != 1
                or archive.infolist()[0].file_size > 2_000_000
            ):
                raise ValueError("unexpected monthly archive")
            raw = archive.read(archive.namelist()[0]).decode()
        rows, invalid = [], []
        for values in csv.reader(io.StringIO(raw)):
            if values[0] == "open_time":
                continue
            row = decode_row(values, year, month)
            if row is None:
                invalid.append(
                    {"time": shared.stamp(timestamp(values[0])), "raw": values}
                )
            else:
                rows.append(row)
        return rows, {
            "url": url,
            "sha256": digest,
            "bytes": len(data),
            "rows": len(rows),
            "invalid_candles": invalid,
        }

    rows = []
    sources = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, (group, source) in enumerate(
            executor.map(download, range(2020 * 12 + 7, 2026 * 12 + 7))
        ):
            rows.extend(group)
            sources.append(source)
            if (i + 1) % 12 == 0:
                print(f"Acquired {i + 1}/72 months", file=sys.stderr, flush=True)
    if any(
        shared.utc(b["time"]) <= shared.utc(a["time"]) for a, b in zip(rows, rows[1:])
    ):
        raise ValueError("duplicate or unordered native hours")
    with (output / "hours.jsonl").open("x") as handle:
        for row in rows:
            handle.write(shared.canonical(row).decode() + "\n")
    result = {
        "created_at": shared.stamp(datetime.now(timezone.utc)),
        "source_cadence": "native Binance Spot1h, not raw1s",
        "rows": len(rows),
        "first": rows[0]["time"],
        "last": rows[-1]["time"],
        "sources": sources,
        "hours_sha256": shared.digest_file(output / "hours.jsonl"),
        "gaps": sum(
            shared.utc(b["time"]) - shared.utc(a["time"]) != timedelta(hours=1)
            for a, b in zip(rows, rows[1:])
        ),
    }
    shared.exclusive_json(output / "provenance.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    r = prepare(args.output, args.resume)
    print(
        json.dumps(
            {
                "rows": r["rows"],
                "gaps": r["gaps"],
                "download_bytes": sum(x["bytes"] for x in r["sources"]),
            }
        )
    )
