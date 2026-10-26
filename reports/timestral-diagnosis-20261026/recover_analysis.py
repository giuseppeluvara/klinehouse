"""Recover only final analysis after the documented NumPy bool serialization error."""
from pathlib import Path
import hashlib
import json
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path.cwd()))
from research import timestral_experiment as shared, timestral_repair as repair
from scripts import run_timestral_repair as runner

root = Path(__file__).resolve().parent
run = root / "run"
protocol = json.loads((run / "protocol.json").read_text())
receipt = json.loads((run / "recovery-receipt.json").read_text())
for name, digest in receipt["saved_output_sha256"].items():
    assert shared.digest_file(run / name) == digest
for name in ("research/timestral_repair.py", "scripts/run_timestral_repair.py"):
    assert shared.digest_file(run / ("original-" + Path(name).name)) == protocol["code_sha256"][name]
original = (run / "original-timestral_repair.py").read_text()
fixed = Path("research/timestral_repair.py").read_text()
assert fixed == original.replace(
    '"two_monthly_wins": sum(difference[indices].mean() < 0 for indices in groups.values()) >= 2,',
    '"two_monthly_wins": bool(sum(difference[indices].mean() < 0 for indices in groups.values()) >= 2),')
assert shared.digest_file("scripts/run_timestral_repair.py") == protocol["code_sha256"]["scripts/run_timestral_repair.py"]

predictions = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
outcomes = [json.loads(line) for line in (run / "outcomes.jsonl").read_text().splitlines()]
assert len(predictions) == len(outcomes) == 2128
history, decisions = [], {}
evaluation_start = shared.utc(protocol["evaluation_start"])
for row, outcome in zip(predictions, outcomes):
    assert row["sample_id"] == outcome["sample_id"]
    anchor = shared.utc(row["anchor"])
    if anchor >= evaluation_start:
        key = row["decision_key"]
        if key not in decisions:
            decisions[key] = repair.choose_model(history, anchor, row["horizon_hours"])
        assert row["selected"] == decisions[key]["selected"]
    history.append({**row, **outcome})
rows = [row for row in history if shared.utc(row["anchor"]) >= evaluation_start]
saved = [json.loads(line) for line in (run / "evaluation.jsonl").read_text().splitlines()]
assert rows == saved and len(rows) == 1484
analysis = runner.analyze(rows, decisions)
resource = (root / "run-resource.txt").read_text()
analysis.update(csv_rows=7420, all_predictions=len(history),
                wall_seconds=float(re.search(r"([\d.]+) real", resource).group(1)),
                peak_rss_bytes=int(re.search(r"(\d+)\s+maximum resident set size", resource).group(1)),
                environment=shared.environment(), protocol_sha256=shared.digest_file(run / "protocol.json"),
                output_sha256=receipt["saved_output_sha256"],
                recovery={"at": datetime.now(timezone.utc).isoformat(), "reason": receipt["failure"],
                          "inference_rerun": False, "forecast_bytes_unchanged": True,
                          "only_code_change": "Python bool conversion of the monthly-win flag",
                          "recovered_code_sha256": runner.code_hashes(), "recovery_script_sha256": shared.digest_file(__file__)})
shared.exclusive_json(run / "analysis-recovered.json", analysis)
print(json.dumps({"recovered": True, "rows": len(rows), "forecast_bytes_unchanged": True,
                  "selected": {h: v["selected_counts"] for h, v in analysis["horizons"].items()}}))
