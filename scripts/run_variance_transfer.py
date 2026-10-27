#!/usr/bin/env python3
"""Frozen transfer of the four-hour variance method to ETH and SOL."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_experiment as shared  # noqa: E402
from research import timestral_decomposition as decomposition  # noqa: E402
from research import timestral_distribution as distribution  # noqa: E402
from scripts.prepare_timestral_history import decode_row  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("ETHUSDT", "SOLUSDT")
CODE = [
    "scripts/run_variance_transfer.py",
    "scripts/prepare_timestral_history.py",
    "research/timestral_decomposition.py",
    "research/timestral_distribution.py",
    "research/timestral_history.py",
    "research/timestral_funding.py",
    "research/timestral_experiment.py",
]


def hashes():
    return {n: shared.digest_file(ROOT / n) for n in CODE}


def register(root):
    root.mkdir(parents=True, exist_ok=False)
    p = {
        "registered_at": shared.stamp(datetime.now(timezone.utc)),
        "code_sha256": hashes(),
        "symbols": SYMBOLS,
        "acquisition": "Native1h monthly Binance Spot ZIPs Aug2020-Jul2026; checksum verified; no Aug2026 or later data. Separate from1s archive. Source bytes bound in acquisition receipt before fitting.",
        "hypothesis": "Transfer the BTC four-hour conditional variance method without any hyperparameter changes to two other assets. Method fixed before acquisition; source history not used to choose it in this experiment. This does not establish globally unseen data or prospective performance.",
        "design": "Identical721hour past context,1095day monthly mature training, Gamma mean model alpha0.01,log variance24/168/720hours, training standardization clip8, same training-only multiplicative mean calibration and Gamma shape for candidate and both dynamic baselines. Refit on each asset own past only.4h horizon only.",
        "gate": "Must pass on BOTH assets in ALL2024/2025/Jan-Jul2026 cohorts: >=1%QLIKE gain vs BOTH calibrated persistence24h and EWMA24h; expected-log MSE no worse; >=2/3monthly wins; upper4000draw weekly bootstrap99.583333%below0,seed20261027.12primary comparisons. Coverage and event scores descriptive. No re-selection or tuning after outcomes.",
        "direction_and_trading": "Not forecast or validated. This is conditional realized-variance risk prediction, not price profit.",
        "btc_protocol_sha256": shared.digest_file(
            ROOT / "research/out/timestral-distribution-20261027/protocol.json"
        ),
    }
    shared.exclusive_json(root / "protocol.json", p)
    shared.exclusive_json(
        root / "protocol.sha256.json",
        {"sha256": shared.digest_file(root / "protocol.json")},
    )
    return p


def verify(root):
    p = json.loads((root / "protocol.json").read_text())
    if (
        hashes() != p["code_sha256"]
        or shared.digest_file(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
    ):
        raise ValueError("transfer protocol or code changed")
    return p


def prepare(root):
    verify(root)
    if (root / "acquisition.json").exists():
        raise ValueError("acquisition already complete")
    sources = {}
    for symbol in SYMBOLS:
        base = root / "sources" / symbol
        base.mkdir(parents=True, exist_ok=True)
        if (base / "hours.jsonl.gz").exists():
            provenance = json.loads((base / "provenance.json").read_text())
            if (
                shared.digest_file(base / "hours.jsonl.gz")
                != provenance["hours_gzip_sha256"]
            ):
                raise ValueError("completed source changed")
            sources[symbol] = provenance
            continue

        def download(number):
            year, month = number // 12, number % 12 + 1
            name = f"{symbol}-1h-{year}-{month:02d}.zip"
            url = (
                "https://data.binance.vision/data/spot/monthly/klines/"
                + symbol
                + "/1h/"
                + name
            )
            with requests.Session() as session:
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
                checksum = session.get(url + ".CHECKSUM", timeout=30)
                checksum.raise_for_status()
                path = base / name
                if path.exists():
                    raw = path.read_bytes()
                else:
                    response = session.get(url, timeout=30)
                    response.raise_for_status()
                    raw = response.content
            digest = hashlib.sha256(raw).hexdigest()
            if len(raw) > 2_000_000 or digest != checksum.text.split()[0]:
                raise ValueError("corrupt/oversized source")
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(raw)
            receipt = Path(str(path) + ".CHECKSUM")
            if receipt.exists():
                if receipt.read_text().split()[0] != digest:
                    raise ValueError("saved receipt changed")
            else:
                with receipt.open("x") as stream:
                    stream.write(checksum.text)
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if (
                    len(archive.infolist()) != 1
                    or archive.infolist()[0].file_size > 2_000_000
                ):
                    raise ValueError("unexpected archive")
                content = archive.read(archive.namelist()[0]).decode()
            values = []
            invalid = []
            for row in csv.reader(io.StringIO(content)):
                if row[0] == "open_time":
                    continue
                parsed = decode_row(row, year, month)
                if parsed is None:
                    invalid.append(row)
                else:
                    values.append(parsed)
            return values, {
                "url": url,
                "sha256": digest,
                "bytes": len(raw),
                "rows": len(values),
                "invalid_rows": invalid,
            }

        hours = []
        files = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, (group, metadata) in enumerate(
                executor.map(download, range(2020 * 12 + 7, 2026 * 12 + 7))
            ):
                hours.extend(group)
                files.append(metadata)
                if (i + 1) % 24 == 0:
                    print(f"{symbol}: {i + 1}/72 months", file=sys.stderr, flush=True)
        if any(
            shared.utc(b["time"]) <= shared.utc(a["time"])
            for a, b in zip(hours, hours[1:])
        ):
            raise ValueError("duplicate/unordered hours")
        encoded = b"".join(shared.canonical(r) + b"\n" for r in hours)
        with (base / "hours.jsonl.gz").open("xb") as stream:
            stream.write(gzip.compress(encoded, mtime=0))
        provenance = {
            "symbol": symbol,
            "created_at": shared.stamp(datetime.now(timezone.utc)),
            "rows": len(hours),
            "first": hours[0]["time"],
            "last": hours[-1]["time"],
            "files": files,
            "hours_gzip_sha256": shared.digest_file(base / "hours.jsonl.gz"),
            "hours_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        shared.exclusive_json(base / "provenance.json", provenance)
        sources[symbol] = provenance
    answer = {
        "completed_at": shared.stamp(datetime.now(timezone.utc)),
        "protocol_sha256": shared.digest_file(root / "protocol.json"),
        "sources": {
            s: {
                n: shared.digest_file(root / "sources" / s / n)
                for n in ("hours.jsonl.gz", "provenance.json")
            }
            for s in SYMBOLS
        },
    }
    shared.exclusive_json(root / "acquisition.json", answer)
    return answer


def run(root):
    verify(root)
    acquisition = json.loads((root / "acquisition.json").read_text())
    if acquisition["protocol_sha256"] != shared.digest_file(
        root / "protocol.json"
    ) or any(
        shared.digest_file(root / "sources" / s / n) != v
        for s, names in acquisition["sources"].items()
        for n, v in names.items()
    ):
        raise ValueError("acquired data changed")
    shared.exclusive_json(
        root / "opened.json", {"at": shared.stamp(datetime.now(timezone.utc))}
    )
    started = time.perf_counter()
    fits = {}
    calibrations = {}
    evaluated = []
    audits = {}
    with (
        (root / "predictions.jsonl").open("x") as pf,
        (root / "outcomes.jsonl").open("x") as of,
    ):
        for symbol in SYMBOLS:
            hours = [
                json.loads(line)
                for line in gzip.decompress(
                    (root / "sources" / symbol / "hours.jsonl.gz").read_bytes()
                ).splitlines()
            ]
            all_rows, audits[symbol] = decomposition.examples(hours)
            rows = [r for r in all_rows if r["horizon_hours"] == 4]
            months = sorted(
                {
                    r["anchor"][:7]
                    for r in rows
                    if "2024-01-01" <= r["anchor"] < "2026-08-01"
                }
            )
            for month in months:
                cutoff = shared.utc(month + "-01")
                key = symbol + ":" + month
                training = [
                    r
                    for r in rows
                    if shared.utc(r["label_end"]) <= cutoff
                    and shared.utc(r["anchor"]) >= cutoff - timedelta(days=1095)
                ]
                fits[key] = decomposition.fit(training, cutoff, 4)
                calibrations[key] = distribution.calibrate(training, fits[key], cutoff)
                for r in rows:
                    if not r["anchor"].startswith(month):
                        continue
                    forecast = {
                        k: v for k, v in r.items() if k not in ("return", "variance")
                    }
                    forecast.update(
                        sample_id=symbol + ":" + r["sample_id"],
                        symbol=symbol,
                        cohort=month[:4],
                        fit_key=key,
                        distribution=distribution.predict(
                            calibrations[key], fits[key], r
                        ),
                    )
                    shared.append_record(pf, forecast)
                    shared.append_record(
                        of,
                        {"sample_id": forecast["sample_id"], "variance": r["variance"]},
                    )
                    evaluated.append({**forecast, "variance": r["variance"]})
            print("Completed fixed transfer: " + symbol, file=sys.stderr, flush=True)
    results = {
        symbol: {
            year: distribution.score(
                [r for r in evaluated if r["symbol"] == symbol and r["cohort"] == year]
            )
            for year in ("2024", "2025", "2026")
        }
        for symbol in SYMBOLS
    }
    shared.exclusive_json(root / "parameters.json", fits)
    shared.exclusive_json(root / "calibrations.json", calibrations)
    answer = {
        "assets": results,
        "passes_all_assets_and_cohorts": all(
            v["passes_both_baselines"] for s in results.values() for v in s.values()
        ),
        "direction_validated": False,
        "trading_validation": False,
        "prospective_validation": False,
        "source_audits": audits,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": shared.peak_rss_bytes(),
        "protocol_sha256": shared.digest_file(root / "protocol.json"),
        "acquisition_sha256": shared.digest_file(root / "acquisition.json"),
        "output_sha256": {
            n: shared.digest_file(root / n)
            for n in (
                "predictions.jsonl",
                "outcomes.jsonl",
                "parameters.json",
                "calibrations.json",
            )
        },
    }
    shared.exclusive_json(root / "results.json", answer)
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["register", "prepare", "run"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    r = {"register": register, "prepare": prepare, "run": run}[args.phase](args.root)
    print(
        json.dumps(
            {
                k: r[k]
                for k in (
                    "registered_at",
                    "completed_at",
                    "passes_all_assets_and_cohorts",
                    "wall_seconds",
                )
                if k in r
            }
        )
    )
