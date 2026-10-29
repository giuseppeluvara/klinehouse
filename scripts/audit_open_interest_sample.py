#!/usr/bin/env python3
"""Reproduce the fixed open-interest timing sample entirely from local files.

No network, model, outcome reader, or production feature code is used. The raw
third-party captures stay private; the public report contains derived summaries
and hashes only. Agreement on these samples is not a publication-time guarantee.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "research/out/position-information-20261029/oi-probe"
DEFAULT_REPORT = (
    ROOT / "reports/timestral-position-information-20261029/oi-timestamp-audit.json"
)
DATES = ("2024-01-01", "2025-01-01", "2026-06-01", "2026-07-01")
SYMBOLS = ("btcusdt", "ethusdt", "solusdt")
BOUNDARY = 1785542400
LIMITATIONS = (
    "Descriptive source audit of 24 fixed hourly labels on four dates per asset. "
    "No top-position validation, no all-day/all-history point-in-time certification. "
    "Exact recurring quantity matches do not prove the archive row publication time. "
    "No inference from flow timing to other fields."
)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(body):
    return hashlib.sha256(body).hexdigest()


def epoch(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def same(actual, expected, where="report"):
    """Compare all computed fields, with only round-off tolerance for floats."""
    if isinstance(expected, dict):
        check(isinstance(actual, dict), where + ": expected object")
        check(set(actual) == set(expected), where + ": different keys")
        for key in expected:
            same(actual[key], expected[key], where + "." + key)
    elif isinstance(expected, list):
        check(isinstance(actual, list), where + ": expected list")
        check(len(actual) == len(expected), where + ": different lengths")
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, f"{where}[{i}]")
    elif isinstance(expected, float):
        check(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            where + ": numeric mismatch",
        )
    else:
        check(
            type(actual) is type(expected) and actual == expected, where + ": mismatch"
        )


def read_captures(source):
    expected = {f"{day}-{hour:02d}04" for day in DATES for hour in range(24)}
    for extension in (".body", ".decoded.txt", ".receipt.json"):
        found = {
            p.name.removesuffix(extension)
            for p in (source / "hourly").glob("*" + extension)
        }
        check(
            found == expected,
            "The hourly directory must contain exactly 96 known captures",
        )
    observations, receipts, hashes = {}, [], {}
    for day in DATES:
        base = epoch(day)
        for hour in range(24):
            stem = f"{day}-{hour:02d}04"
            receipt = json.loads(
                (source / "hourly" / (stem + ".receipt.json")).read_text()
            )
            body = (source / "hourly" / (stem + ".body")).read_bytes()
            check(
                0 < len(body) < 64 * 1024,
                "Compressed capture outside recorded size limit",
            )
            check(
                receipt["status"] == 200 and receipt["auth"] is False,
                "Invalid HTTP receipt",
            )
            check(
                not receipt.get("body_limit_reached")
                and not receipt.get("parse_error"),
                "Incomplete capture",
            )
            check(
                receipt["date"] == day and receipt["hour"] == hour,
                "Wrong receipt date/hour",
            )
            check(
                receipt["bytes"] == len(body) and receipt["sha256"] == digest(body),
                "Capture body hash/size mismatch",
            )
            url = urlsplit(receipt["url"])
            query = parse_qs(url.query, strict_parsing=True)
            check(
                url.scheme == "https" and url.netloc == "api.tardis.dev",
                "Wrong source host",
            )
            check(
                url.path == "/v1/data-feeds/binance-futures" and not url.fragment,
                "Wrong raw endpoint",
            )
            check(
                set(query) == {"from", "offset", "sliceSize", "filters"},
                "Unexpected request parameter",
            )
            check(
                query["from"] == [day]
                and query["offset"] == [str(hour * 60 + 4)]
                and query["sliceSize"] == ["7"],
                "Unexpected capture interval",
            )
            check(len(query["filters"]) == 1, "Ambiguous filter")
            same(
                json.loads(query["filters"][0]),
                [{"channel": "openInterest", "symbols": list(SYMBOLS)}],
                "request filters",
            )
            headers = {key.lower(): value for key, value in receipt["headers"].items()}
            check(
                headers.get("content-encoding") == "gzip", "Unexpected content encoding"
            )
            with gzip.GzipFile(fileobj=io.BytesIO(body)) as compressed:
                plain = compressed.read(128 * 1024 + 1)
            check(len(plain) <= 128 * 1024, "Decoded capture too large")
            check(
                plain == (source / "hourly" / (stem + ".decoded.txt")).read_bytes(),
                "Decoded cache differs from raw body",
            )
            check(len(plain) == receipt["decoded_bytes"], "Decoded size mismatch")
            rows = []
            for line in plain.decode("utf-8").splitlines():
                if not line.strip():
                    continue  # Tardis represents disconnects with empty lines.
                captured, payload = line.split(" ", 1)
                local = epoch(captured)
                message = json.loads(payload)
                data = message["data"]
                symbol = data["symbol"].lower()
                native = float(data["time"]) / 1000
                value = float(data["openInterest"])
                check(
                    symbol in SYMBOLS and message["stream"] == symbol + "@openInterest",
                    "Unexpected channel/symbol",
                )
                check(
                    message.get("generated") is True, "Expected generated REST snapshot"
                )
                check(
                    base + (hour * 60 + 4) * 60
                    <= local
                    < base + (hour * 60 + 11) * 60
                    < BOUNDARY,
                    "Receipt outside fixed pre-holdout capture",
                )
                check(
                    base <= native < base + 86400 < BOUNDARY,
                    "Native timestamp outside sampled pre-holdout day",
                )
                check(
                    math.isfinite(value) and value > 0, "Invalid open-interest quantity"
                )
                rows.append(
                    dict(symbol=symbol, native=native, local=local, value=value)
                )
            check(len(rows) == receipt["messages"], "Message-count mismatch")
            check({r["symbol"] for r in rows} == set(SYMBOLS), "Missing sampled symbol")
            same(
                receipt["native_range"],
                [min(r["native"] for r in rows), max(r["native"] for r in rows)],
                "native range",
            )
            same(
                receipt["local_range"],
                [min(r["local"] for r in rows), max(r["local"] for r in rows)],
                "receipt range",
            )
            observations[(day, hour)] = rows
            receipts.append(receipt)
            hashes["hourly/" + stem + ".body"] = digest(body)
    return observations, receipts, hashes


def read_metrics(source, day, symbol):
    name = f"{symbol.upper()}-metrics-{day}.zip"
    # Preserve the original pilot's source identity even when the later full
    # acquisition adds another copy of the June files to the main raw folder.
    path = (
        source / "metrics-reference" if day == "2026-06-01" else source.parent / "raw"
    ) / name
    body = path.read_bytes()
    checksum = Path(str(path) + ".CHECKSUM").read_text().split()
    check(
        len(checksum) == 2 and checksum[1] == name, "Invalid official checksum record"
    )
    check(checksum[0] == digest(body), "Metrics checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        check(
            archive.namelist() == [name.removesuffix(".zip") + ".csv"],
            "Unexpected metrics archive member",
        )
        with archive.open(archive.namelist()[0]) as member:
            rows = list(csv.DictReader(io.TextIOWrapper(member)))
    labels = [row["create_time"] for row in rows]
    check(
        len(labels) == len(set(labels)) == 288, "Incomplete or duplicate metrics labels"
    )
    base = epoch(day)
    check(
        sorted(epoch(label) for label in labels)
        == [base + 300 * i for i in range(288)],
        "Unexpected daily metrics grid",
    )
    check(base + 86400 < BOUNDARY, "Metrics day overlaps protected boundary")
    check(all(row["symbol"] == symbol.upper() for row in rows), "Wrong metrics symbol")
    audit = dict(
        path=str(path.relative_to(source.parent)),
        sha256=digest(body),
        rows=len(rows),
        unique_labels=len(set(labels)),
        first_label=min(labels),
        last_label=max(labels),
        checksum_passed=True,
    )
    return dict(zip(labels, rows)), audit


def reconstruct(source):
    plan = json.loads((source / "hourly-plan.json").read_text())
    check(
        tuple(plan["dates"]) == DATES and tuple(plan["symbols"]) == SYMBOLS,
        "Sample changed",
    )
    check(
        plan["timestamp_hypotheses_seconds_after_label"] == [0, 300],
        "Hypotheses changed",
    )
    check(
        plan["nearest_native_time_tolerance_seconds"] == 15,
        "Nearest-time tolerance changed",
    )
    check(
        plan["no_outcomes"] is True and plan["no_fits"] is True,
        "Wrong source-only plan",
    )
    observations, receipts, hashes = read_captures(source)
    details, sources = [], []
    for day in DATES:
        for symbol in SYMBOLS:
            metrics, audit = read_metrics(source, day, symbol)
            sources.append(audit)
            for hour in range(24):
                label_text = f"{day} {hour:02d}:05:00"
                label = epoch(label_text)
                value = float(metrics[label_text]["sum_open_interest"])
                check(math.isfinite(value) and value > 0, "Invalid sampled archive OI")
                row = dict(
                    date=day,
                    symbol=symbol.upper(),
                    label=label_text,
                    archive_oi=value,
                    hypotheses={},
                )
                raw = [r for r in observations[(day, hour)] if r["symbol"] == symbol]
                for shift in (0, 300):
                    nearest = min(
                        raw,
                        key=lambda r: (abs(r["native"] - label - shift), r["local"]),
                    )
                    distance = abs(nearest["native"] - label - shift)
                    if distance <= 15:
                        row["hypotheses"][str(shift)] = dict(
                            raw_oi=nearest["value"],
                            native=nearest["native"],
                            local=nearest["local"],
                            native_distance_seconds=distance,
                            relative_error=abs(value - nearest["value"]) / value,
                            arrival_after_label_seconds=nearest["local"] - label,
                        )
                exact = [r for r in raw if r["value"] == value]
                row["exact_value_seen"] = bool(exact)
                row["first_exact_arrival_after_label_seconds"] = (
                    min(r["local"] - label for r in exact) if exact else None
                )
                details.append(row)
    summary = []
    for day in DATES:
        for symbol in SYMBOLS:
            rows = [
                r for r in details if r["date"] == day and r["symbol"].lower() == symbol
            ]
            paired = [r for r in rows if set(r["hypotheses"]) == {"0", "300"}]
            arrivals = [
                r["first_exact_arrival_after_label_seconds"]
                for r in rows
                if r["exact_value_seen"]
            ]
            group = dict(
                date=day,
                symbol=symbol.upper(),
                labels=len(rows),
                paired_hypotheses=len(paired),
                exact_values_seen=len(arrivals),
                exact_arrival_after_label_seconds_range=[min(arrivals), max(arrivals)]
                if arrivals
                else None,
                hypotheses={},
            )
            for shift in ("0", "300"):
                selected = [r for r in rows if shift in r["hypotheses"]]
                archive = np.array([r["archive_oi"] for r in selected])
                raw = np.array([r["hypotheses"][shift]["raw_oi"] for r in selected])
                # Successive available hourly samples, including any skipped hour.
                change_error = abs(np.diff(np.log(archive)) - np.diff(np.log(raw)))
                other = "300" if shift == "0" else "0"
                group["hypotheses"][shift] = dict(
                    matches=len(selected),
                    median_relative_error=float(np.median(abs(archive - raw) / archive))
                    if len(archive)
                    else None,
                    median_logchange_error=float(np.median(change_error))
                    if len(change_error)
                    else None,
                    lower_error_than_other=sum(
                        r["hypotheses"][shift]["relative_error"]
                        < r["hypotheses"][other]["relative_error"]
                        for r in paired
                    ),
                    maximum_native_distance_seconds=max(
                        (
                            r["hypotheses"][shift]["native_distance_seconds"]
                            for r in selected
                        ),
                        default=None,
                    ),
                )
            summary.append(group)
    delays = [r["local"] - r["native"] for rows in observations.values() for r in rows]
    check(len(details) == 288 and len(sources) == 12, "Wrong sample dimensions")
    return dict(
        plan=plan,
        sources=sources,
        capture_hashes=hashes,
        request_count=len(receipts),
        downloaded_bytes=sum(r["bytes"] for r in receipts),
        decoded_bytes=sum(r["decoded_bytes"] for r in receipts),
        status_counts=dict(Counter(str(r["status"]) for r in receipts)),
        messages=len(delays),
        native_to_receipt_seconds_range=[min(delays), max(delays)],
        summary=summary,
        detail=details,
        protected_data_opened=False,
        outcomes_opened=False,
        limitations=LIMITATIONS,
    )


def verify(source=DEFAULT_SOURCE, report_path=DEFAULT_REPORT):
    result = reconstruct(source)
    full_path = source / "hourly-comparison.json"
    same(json.loads(full_path.read_text()), result, "private full comparison")
    saved = json.loads(report_path.read_text())
    compact = {k: v for k, v in result.items() if k != "detail"}
    check(set(compact) <= set(saved), "Public report is missing computed fields")
    same({key: saved[key] for key in compact}, compact, "public summary")
    check(
        saved["private_full_comparison_sha256"] == digest(full_path.read_bytes()),
        "Full-comparison provenance mismatch",
    )
    check(
        saved["historical_publication_verified"] is False
        and saved["position_timing_verified"] is False
        and saved["oi_sample_alignment_only"] is True,
        "Unsupported timing certification",
    )
    return dict(
        verified=True,
        raw_captures=result["request_count"],
        raw_messages=result["messages"],
        metrics_files=len(result["sources"]),
        sampled_labels=len(result["detail"]),
        all_checksums_passed=True,
        outcomes_opened=False,
        protected_data_opened=False,
        timing_certified=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check committed summary and full private comparison against raw captures",
    )
    args = parser.parse_args()
    if args.verify:
        result = verify(args.source_root, args.report)
    else:
        result = reconstruct(args.source_root)
        result.pop("detail")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
