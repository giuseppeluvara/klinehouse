#!/usr/bin/env python3
"""Fetch only the small pre-August2026 settled BTC perpetual funding archive."""

from datetime import datetime, timezone
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_experiment as shared  # noqa: E402


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    sources = []
    for number in range(2025 * 12 + 8, 2026 * 12 + 7):
        year, month = number // 12, number % 12 + 1
        name = f"BTCUSDT-fundingRate-{year}-{month:02d}.zip"
        url = (
            "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/"
            + name
        )
        checksum = requests.get(url + ".CHECKSUM", timeout=30)
        checksum.raise_for_status()
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        if (
            len(response.content) > 1_000_000
            or hashlib.sha256(response.content).hexdigest() != checksum.text.split()[0]
        ):
            raise ValueError("oversized or corrupt funding archive")
        (output / name).write_bytes(response.content)
        (output / (name + ".CHECKSUM")).write_text(checksum.text)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            if (
                len(archive.infolist()) != 1
                or archive.infolist()[0].file_size > 1_000_000
            ):
                raise ValueError("unexpected funding archive members")
            raw = archive.read(archive.namelist()[0]).decode()
        for row in csv.DictReader(io.StringIO(raw)):
            time = datetime.fromtimestamp(int(row["calc_time"]) / 1000, timezone.utc)
            if (
                time.year != year
                or time.month != month
                or time >= shared.utc("2026-08-01")
            ):
                raise ValueError("out-of-period funding record")
            records.append(
                {
                    "time": shared.stamp(time),
                    "interval_hours": float(row["funding_interval_hours"]),
                    "rate": float(row["last_funding_rate"]),
                }
            )
        sources.append(
            {
                "url": url,
                "sha256": hashlib.sha256(response.content).hexdigest(),
                "bytes": len(response.content),
            }
        )
    if any(
        shared.utc(b["time"]) <= shared.utc(a["time"])
        for a, b in zip(records, records[1:])
    ):
        raise ValueError("duplicate or unordered funding timestamps")
    shared.exclusive_json(output / "funding.json", records)
    result = {
        "created_at": shared.stamp(datetime.now(timezone.utc)),
        "rows": len(records),
        "sources": sources,
        "funding_sha256": shared.digest_file(output / "funding.json"),
        "availability": "assumed final settlement timestamp plus1hour; historical publication latency not verified",
    }
    shared.exclusive_json(output / "provenance.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = prepare(parser.parse_args().output)
    print(
        json.dumps(
            {
                "rows": result["rows"],
                "downloaded_bytes": sum(r["bytes"] for r in result["sources"]),
            }
        )
    )
