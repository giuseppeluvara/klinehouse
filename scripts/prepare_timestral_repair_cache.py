#!/usr/bin/env python3
"""Reuse three existing verified archives; never download or open protected dates."""

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_experiment as shared, timestral_long_horizon as hourly  # noqa: E402


def prepare(root, adaptation_root, long_horizon_protocol, plan):
    root, adaptation_root = Path(root), Path(adaptation_root)
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    plan = json.loads(Path(plan).read_text())
    shared.exclusive_json(root / "plan.json", plan)
    cache = adaptation_root / "development-run/hours.jsonl"
    results = json.loads((adaptation_root / "development-run/results.json").read_text())
    if shared.digest_file(cache) != results["output_sha256"]["hours.jsonl"]:
        raise ValueError("development cache changed")
    hours = [json.loads(line) for line in cache.read_text().splitlines()]
    sources = [{"cache": str(cache), "sha256": shared.digest_file(cache), "hours": len(hours)}]
    for path, start, end in ((adaptation_root / "test-run/protocol.json", "2026-01-01", "2026-04-01"),
                             (Path(long_horizon_protocol), "2026-04-01", "2026-08-01")):
        protocol = json.loads(path.read_text())
        config = protocol["config"]
        config = {**config, "archive_root": protocol["source"]["archive_root"], "source": "archive",
                  "start": protocol.get("test_start", config.get("start")),
                  "end": protocol.get("test_end", config.get("end"))}
        if shared.utc(config["start"]) != shared.utc(start) or shared.utc(config["end"]) != shared.utc(end):
            raise ValueError("unexpected or protected source period")
        snapshot, _ = shared.verified_source_snapshot(config)
        if snapshot != protocol["source"]:
            raise ValueError("source catalog changed")
        audit = {}
        hours.extend({"time": shared.stamp(row.time), "log_close": row.log_close, "features": list(row.features)}
                     for row in hourly.read_hours(config, audit))
        sources.append({"protocol": str(path), "protocol_sha256": shared.digest_file(path),
                        "source": snapshot, "data": audit})
    if (len(hours) != 8016 or shared.utc(hours[0]["time"]) != shared.utc("2025-09-01")
            or shared.utc(hours[-1]["time"]) != shared.utc("2026-07-31T23:00:00Z")
            or any((shared.utc(b["time"]) - shared.utc(a["time"])).total_seconds() != 3600 for a, b in zip(hours, hours[1:]))):
        raise ValueError("missing, duplicate or out-of-period hours")
    with (root / "hours.jsonl").open("x") as handle:
        for row in hours:
            handle.write(shared.canonical(row).decode() + "\n")
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(), "hours": len(hours),
                  "hours_sha256": shared.digest_file(root / "hours.jsonl"), "sources": sources,
                  "seconds": time.perf_counter() - started, "peak_rss_bytes": shared.peak_rss_bytes()}
    shared.exclusive_json(root / "cache-provenance.json", provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--adaptation-root", type=Path, default="research/out/timestral-adaptation-20261025")
    parser.add_argument("--long-horizon-protocol", type=Path, default="reports/timestral-long-horizon-20261025/protocol.json")
    parser.add_argument("--plan", type=Path, default="configs/timestral_repair.json")
    args = parser.parse_args()
    result = prepare(args.root, args.adaptation_root, args.long_horizon_protocol, args.plan)
    print(json.dumps({key: result[key] for key in ("hours", "seconds", "peak_rss_bytes")}))


if __name__ == "__main__":
    main()
