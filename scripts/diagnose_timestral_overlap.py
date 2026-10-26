#!/usr/bin/env python3
"""Compare dense/disjoint h-hour returns over identical past price windows."""

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_cadence as cadence  # noqa: E402
from research import timestral_diagnostics as diagnosis  # noqa: E402
from research import timestral_experiment as shared  # noqa: E402
from research.timestral_long_horizon import Hour  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/timestral-diagnosis-20261026"
INPUTS = ("hours.jsonl", "run/evaluation.jsonl")
CODE = ("research/timestral_cadence.py", "scripts/diagnose_timestral_overlap.py", "research/timestral_diagnostics.py",
        "research/timestral_experiment.py", "research/timestral_long_horizon.py")


def hashes():
    return {name: shared.digest_file(ROOT / name) for name in CODE}


def register(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    p = {"registered_at": shared.stamp(datetime.now(timezone.utc)), "code_sha256": hashes(),
         "source": str(SOURCE), "input_sha256": {n: shared.digest_file(SOURCE / n) for n in INPUTS},
         "status": "one exploratory ablation prompted by diagnosed overlap; already inspected history, no fresh test",
         "hypothesis": "Dense rolling h-hour returns have mechanical lag1 persistence. Disjoint h-hour returns remove shared increments; test whether this reduces copying of the last known return and improves error.",
         "design": "Both inputs derive from identical 64*h+1 consecutive hourly closes ending before anchor. Dense has63*h+1 values and predicts h steps; disjoint has64 values and predicts1 step. Both target log(P[t+h-1]/P[t-1]). Cadence and token count necessarily change; raw information spans do not. Neither sees future values.",
         "comparison": "All1484 Jan-Jul anchors, 4/24h. Report all monthly and total metrics, paired weekly bootstrap 2000 draws seed20261026, two-sided98.75% for four disjoint-vs-dense/zero comparisons. No tuning, early stopping, winner deployment, or fresh-test claim.",
         "controls": "Known constant, increasing/decreasing linear and four sine phases through both adapters at both horizons. Dense may legitimately forecast noise poorly; all controls retained.",
         "difference_from_prior": "History is64 horizon periods, not256hours. At24h both dense and disjoint now see64days. Compare the paired variants to isolate cadence; comparison with the prior report alone would confound history length.",
         "model": {"id": shared.MODEL_ID, "code_revision": shared.CODE_REVISION, "weight_revision": shared.WEIGHT_REVISION},
         "model_config": {"lookback": 1568, "torch_threads": 2, "local_files_only": True,
                          "model_cache": str(ROOT / "research/out/timestral-pilot-20261025/model-cache")}}
    shared.exclusive_json(output / "protocol.json", p)
    shared.exclusive_json(output / "protocol.sha256.json", {"sha256": shared.digest_file(output / "protocol.json")})
    return p


def verify(output):
    output = Path(output)
    p = json.loads((output / "protocol.json").read_text())
    if (shared.digest_file(output / "protocol.json") != json.loads((output / "protocol.sha256.json").read_text())["sha256"]
            or p["code_sha256"] != hashes() or p["source"] != str(SOURCE)
            or any(shared.digest_file(SOURCE / n) != v for n, v in p["input_sha256"].items())):
        raise ValueError("registered inputs or code changed")
    return p


def summary(rows):
    y = np.asarray([r["actual"] for r in rows])
    result = {}
    for name in ("unchanged", "dense", "disjoint"):
        p = np.zeros(len(rows)) if name == "unchanged" else np.asarray([r["predictions"][name]["point"] for r in rows])
        result[name] = {"n": len(rows), "mae": float(abs(p - y).mean()), "rmse": float(np.sqrt(np.mean((p - y)**2))),
                        "prediction_future_correlation": diagnosis.correlation(p, y),
                        "prediction_past_correlation": diagnosis.correlation(p, [r["past_h_return"] for r in rows])}
    return result


def run(output):
    output = Path(output)
    protocol = verify(output)
    shared.exclusive_json(output / "opened.json", {"at": shared.stamp(datetime.now(timezone.utc)),
                                                  "protocol_sha256": shared.digest_file(output / "protocol.json")})
    started = time.perf_counter()
    hours = [Hour(shared.utc(r["time"]), r["log_close"], tuple(r["features"]))
             for r in (json.loads(line) for line in (SOURCE / "hours.jsonl").read_text().splitlines())]
    lookup = {shared.stamp(r.time): i for i, r in enumerate(hours)}
    prior = [json.loads(line) for line in (SOURCE / "run/evaluation.jsonl").read_text().splitlines()]
    if len(prior) != 1484 or len(hours) != 8016 or hours[-1].time >= shared.utc("2026-08-01"):
        raise ValueError("unexpected or protected source")
    model = shared.Timestral(protocol["model_config"])
    rows = []
    with (output / "predictions.jsonl").open("x") as predictions, (output / "outcomes.jsonl").open("x") as outcomes:
        for i, row in enumerate(prior):
            idx, h = lookup[row["anchor"]], row["horizon_hours"]
            context = hours[idx - 64 * h - 1:idx]
            forecasts = cadence.forecast(model, context, row["anchor"], h)
            digest = hashlib.sha256(shared.canonical([[shared.stamp(r.time), r.log_close] for r in context])).hexdigest()
            sample_id = hashlib.sha256(shared.canonical(["cadence", row["anchor"], h, digest])).hexdigest()
            record = {"sample_id": sample_id, "reference_sample_id": row["sample_id"], "anchor": row["anchor"],
                      "horizon": h, "input_start": shared.stamp(context[0].time), "input_sha256": digest,
                      "predictions": forecasts, "past_h_return": context[-1].log_close - context[-h-1].log_close}
            shared.append_record(predictions, record)
            actual = hours[idx + h - 1].log_close - context[-1].log_close
            if abs(actual - row["actual_log_return"]) > 1e-14:
                raise ValueError("target mismatch")
            shared.append_record(outcomes, {"sample_id": sample_id, "actual": actual})
            rows.append({**record, "actual": actual})
            if (i + 1) % 400 == 0:
                print(f"Cadence pairs: {i + 1}/{len(prior)}", file=sys.stderr, flush=True)
    results = {}
    for h in (4, 24):
        group = [r for r in rows if r["horizon"] == h]
        results[str(h)] = {"metrics": summary(group),
                           "monthly": {m: summary([r for r in group if r["anchor"][:7] == m]) for m in sorted({r["anchor"][:7] for r in group})},
                           "disjoint_vs_dense": cadence.paired_mae_interval(group, "disjoint", "dense"),
                           "disjoint_vs_unchanged": cadence.paired_mae_interval(group, "disjoint", "unchanged")}
    # The same known deterministic patterns, extended to cover the larger span.
    controls = []
    from datetime import timedelta
    origin = shared.utc("2000-01-01")
    for h in (4, 24):
        n = 64 * h + 1
        t = np.arange(n + h)
        cases = {"constant": np.full(len(t), 10.), "linear_up": 10. + .0002*t, "linear_down": 10. - .0002*t}
        cases.update({f"sine_{phase}": 10. + .01*np.sin(2*np.pi*t/64 + phase*np.pi/2) for phase in range(4)})
        for name, values in cases.items():
            context = [Hour(origin + timedelta(hours=i), float(v), (0., 0., 0.)) for i, v in enumerate(values[:n])]
            predictions = cadence.forecast(model, context, origin + timedelta(hours=n), h)
            controls.append({"case": name, "horizon": h, "predictions": predictions, "actual": float(values[n+h-1]-values[n-1])})
    shared.exclusive_json(output / "controls.json", controls)
    result = {"status": protocol["status"], "horizons": results, "wall_seconds": time.perf_counter() - started,
              "peak_rss_bytes": shared.peak_rss_bytes(), "environment": shared.environment(),
              "checkpoint_files_sha256": model.checkpoint_files, "protocol_sha256": shared.digest_file(output / "protocol.json"),
              "output_sha256": {n: shared.digest_file(output / n) for n in ("predictions.jsonl", "outcomes.jsonl", "controls.json")}}
    shared.exclusive_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = register(args.output) if args.phase == "register" else run(args.output)
    print(json.dumps({k: result[k] for k in ("status", "wall_seconds", "peak_rss_bytes") if k in result}))


if __name__ == "__main__":
    main()
