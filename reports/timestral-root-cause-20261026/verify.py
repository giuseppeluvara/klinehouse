"""Reconstruct saved diagnostics without importing either diagnostic pipeline.

Run from the repository root. No inference, raw archive reads or output writes.
This checks calculations and saved provenance, not independent predictive skill.
"""

from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "reports/timestral-diagnosis-20261026"
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_text())
lines = lambda path: [json.loads(line) for line in path.read_text().splitlines()]
close = lambda a, b: math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-13)

hours = lines(SOURCE / "hours.jsonl")
prior = lines(SOURCE / "run/evaluation.jsonl")
lookup = {r["time"]: i for i, r in enumerate(hours)}
old = {r["sample_id"]: r for r in prior}
reports = {}
for folder in ("run", "overlap"):
    base = HERE / folder
    p, r = read(base / "protocol.json"), read(base / "results.json")
    assert sha(base / "protocol.json") == read(base / "protocol.sha256.json")["sha256"] == r["protocol_sha256"]
    assert all(sha(ROOT / n) == value for n, value in p["code_sha256"].items())
    assert all(sha(SOURCE / n) == value for n, value in p["input_sha256"].items())
    assert all(sha(base / n) == value for n, value in r["output_sha256"].items())
    reports[folder] = r

probes = lines(HERE / "run/affine-probes.jsonl")
assert len(probes) == len(old) == 1484
assert {r["sample_id"] for r in probes} == set(old)
for r in probes:
    original = old[r["sample_id"]]
    idx, h = lookup[r["anchor"]], r["horizon"]
    assert r["saved"] == original["predictions"]["timestral_price"]
    assert close(r["actual"], hours[idx+h-1]["log_close"] - hours[idx-1]["log_close"])
    assert close(r["past_h_return"], hours[idx-1]["log_close"] - hours[idx-h-1]["log_close"])
for h in (4, 24):
    group = [r for r in probes if r["horizon"] == h]
    metric = reports["run"]["horizons"][str(h)]["affine"]
    assert close(metric["max_point_difference_bps"], max(abs(r["saved"]["point"]-r["transformed"]["point"]) for r in group)*10000)
    for key, model in (("saved_mae", "saved"), ("transformed_mae", "transformed")):
        assert close(metric[key], math.fsum(abs(r[model]["point"]-r["actual"]) for r in group)/len(group))
    for period, cutoff in (("jan_mar", "2026-04-01"), ("jan_jul", "2026-08-01")):
        rows = [r for r in prior if r["horizon_hours"] == h and r["anchor"] < cutoff]
        for model, key in (("unchanged_static", "saved_static"), ("unchanged_dynamic", "dynamic")):
            scores = []
            for r in rows:
                lower, _, upper = r["predictions"][model]["quantiles"]
                y = r["actual_log_return"]
                scores.append(upper-lower + (20*(lower-y) if y<lower else 20*(y-upper) if y>upper else 0))
            saved = reports["run"]["horizons"][str(h)]["intervals"][period][key]
            assert close(saved["interval_score90"], math.fsum(scores)/len(scores))
            assert close(saved["tail_pinball"]*40, saved["interval_score90"])

predictions, outcomes = lines(HERE / "overlap/predictions.jsonl"), lines(HERE / "overlap/outcomes.jsonl")
assert len(predictions) == len(outcomes) == 1484
assert len({r["sample_id"] for r in predictions}) == 1484
assert {r["reference_sample_id"] for r in predictions} == set(old)
joined = []
for p, o in zip(predictions, outcomes):
    assert p["sample_id"] == o["sample_id"] and "actual" not in p
    idx, h = lookup[p["anchor"]], p["horizon"]
    context = hours[idx-64*h-1:idx]
    assert len(context) == 64*h+1
    assert context[0]["time"] == p["input_start"]
    assert datetime.fromisoformat(context[-1]["time"]) + timedelta(hours=1) == datetime.fromisoformat(p["anchor"])
    encoded = json.dumps([[r["time"], r["log_close"]] for r in context], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(encoded).hexdigest() == p["input_sha256"]
    assert close(o["actual"], hours[idx+h-1]["log_close"] - context[-1]["log_close"])
    assert p["predictions"]["dense"]["input_points"] == 63*h+1
    assert p["predictions"]["dense"]["forecast_steps"] == h
    assert p["predictions"]["disjoint"]["input_points"] == 64
    assert p["predictions"]["disjoint"]["forecast_steps"] == 1
    joined.append({**p, **o})

for h in (4, 24):
    group = [r for r in joined if r["horizon"] == h]
    result = reports["overlap"]["horizons"][str(h)]
    for name, saved in result["metrics"].items():
        errors = [(0 if name == "unchanged" else r["predictions"][name]["point"]) - r["actual"] for r in group]
        assert close(saved["mae"], math.fsum(abs(x) for x in errors)/len(errors))
        assert close(saved["rmse"], math.sqrt(math.fsum(x*x for x in errors)/len(errors)))
    # Expand sampled blocks to their actual row indices, rather than reuse block sums.
    blocks = np.asarray([(datetime.fromisoformat(r["anchor"])-datetime.fromisoformat("2026-01-01T00:00:00+00:00")).days//7 for r in group])
    n = int(blocks.max()+1)
    rng = np.random.default_rng(20261026)
    indices = [np.flatnonzero(blocks == b) for b in range(n)]
    draws = [np.concatenate([indices[b] for b in draw]) for draw in rng.integers(0, n, size=(2000, n))]
    for reference in ("dense", "unchanged"):
        differences = np.asarray([abs(r["predictions"]["disjoint"]["point"]-r["actual"])
                                  -abs((0 if reference == "unchanged" else r["predictions"][reference]["point"])-r["actual"]) for r in group])
        interval = np.quantile([differences[d].mean() for d in draws], [.00625, .99375])
        np.testing.assert_allclose(interval, result[f"disjoint_vs_{reference}"]["interval98_75"], rtol=1e-10, atol=1e-13)

print(json.dumps({"passed": True, "affine_samples": len(probes), "cadence_pairs": len(joined),
                  "bootstrap_intervals_recomputed": 4, "interval_comparisons_recomputed": 4,
                  "limits": "No rerun of inference; no independent market holdout; cache source provenance reused."}))
