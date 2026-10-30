#!/usr/bin/env python3
"""Register and execute a separate orientation experiment on allowed saved data."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.direction_orientation import replay  # noqa: E402

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
BASE = ROOT / "reports/timestral-position-information-20261029"
OLD = ROOT / "reports/timestral-universal-20261027"
CHECKPOINT = ROOT / "research/out/direction-diagnosis-20261030/live-checkpoint"
DEFAULT = ROOT / "reports/direction-orientation-20261030-v3"
BOUNDARY = 1785542400
ARMS = ("baseline", "price_control", "positions", "neutral", "mirrored", "adaptive")
CODE = (
    "research/direction_orientation.py",
    "scripts/run_direction_orientation.py",
    "tests/test_direction_orientation.py",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def outcome_path(year, symbol):
    return OLD / (
        f"{symbol}-outcomes.npz"
        if year == 2026
        else f"history/{year}-{symbol}-outcomes.npz"
    )


def inputs():
    return [
        p
        for y in (2024, 2025, 2026)
        for s in SYMBOLS
        for p in (BASE / f"{y}-{s}-predictions.npz", outcome_path(y, s))
    ] + [
        CHECKPOINT / name
        for name in ("events.sqlite", "config.json", "model.json", "general-audit.json")
    ]


def register(root):
    root.mkdir()
    protocol = {
        "registered_at": time.time(),
        "purpose": "Test causal orientation of the same hourly center after the live contrarian ranking failure; never tune the running pilot.",
        "primary": "adaptive: equal-prior conditional Bayesian mixture of q,1-q,0.5; identical issued flat probability. Use q_new=.5+(w_reversal-w_continuation)*(q-.5) for exact neutral ties. The earlier v2 probability-sum roundoff, results and source bytes are preserved. Two index-parity lanes per asset/cohort. Update only from earlier same-lane non-flat outcomes actually available by issue. Retain late pending feedback, consume each exactly once. No forgetting, window, learned hyperparameters or per-cohort model choice.",
        "historical": "January-July forecast anchors in 2024,2025,2026; existing saved monthly out-of-sample probabilities. Preserve original 2024/25 July31 23:55 targets ending August1 00:01; these are not protected dates. ALL targets must end before the protected 2026-08-01 boundary, including 2026. Issue at anchor (conservative), label availability assumed at target end anchor+360. Publication timing remains conditional. Equal initial priors per asset/year; all coldstart cases retained. Earlier registration stopped on an overly strict year-local target guard before scoring, and is preserved separately.",
        "live": "Immutable checkpoint captured 2026-10-30 14:57:45UTC. Actual issued/ack/outcome receipt chronology. OFFLINE POST-HOC REPLAY informed by already inspected outcomes, not prospectively emitted new predictions or independent confirmation. Pending outcomes remain unscored.",
        "controls": "unchanged baseline, price_control, positions; neutral q=.5 and static mirrored q as diagnostic controls only. Neutral ties conventionally predict up but do not constitute a directional signal. Mirrored is not promoted from hindsight.",
        "evaluation": "All eligible moving cases: direction numerator/denominator, conditional Brier and natural logloss. All realized cases: full three-class sum Brier and natural logloss. Original selection max(pdown,pup)>=.52 reported with flat errors and coverage. Paired comparison also on the fixed baseline selection, identical cases. No threshold search or dropped cohort.",
        "success": "Empirical stable improvement requires strictly better all-moving direction and all-case Brier/logloss than both baseline and price_control in every one of nine historical cohorts. Report all controls/negative results. Even a pass is sequential retrospective evidence, not prospective success or profitability. Live replay success cannot establish a general solution.",
        "protected": "No protected historical data from 2026-08-01 onward; only named previously explored saved historical arrays and the explicitly authorized new prospective checkpoint. No changes to pilot runtime/model/config/service/env/credentials or existing frozen artifacts.",
        "code": {name: digest(ROOT / name) for name in CODE},
        "inputs": {str(p.relative_to(ROOT)): digest(p) for p in inputs()},
    }
    write(root / "protocol.json", protocol)
    write(
        root / "protocol.sha256.json", {"protocol.json": digest(root / "protocol.json")}
    )
    for name in CODE:
        target = root / "frozen-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write((ROOT / name).read_bytes())


def scores(p, label, mask):
    take = mask & (label >= 0)
    y, pp = label[take], p[take]
    moving = y != 1
    q = pp[:, 2] / (pp[:, 0] + pp[:, 2])
    correct = (q >= 0.5) == (y == 2)
    result = {
        "n": int(len(y)),
        "moving_n": int(moving.sum()),
        "flat_n": int((~moving).sum()),
    }
    if not len(y):
        return result
    result.update(
        brier=float(np.square(pp - np.eye(3)[y]).sum(1).mean()),
        log_loss=float(-np.log(pp[np.arange(len(y)), y]).mean()),
        direction_correct=int(correct[moving].sum()),
        direction_accuracy=float(correct[moving].mean()) if moving.any() else None,
        direction_brier=float(np.square(q[moving] - (y[moving] == 2)).mean())
        if moving.any()
        else None,
        direction_log_loss=float(
            -np.log(np.where(y[moving] == 2, q[moving], 1 - q[moving])).mean()
        )
        if moving.any()
        else None,
    )
    return result


def historical(year, symbol):
    with np.load(BASE / f"{year}-{symbol}-predictions.npz", allow_pickle=False) as f:
        d = {k: f[k] for k in ("anchor", "baseline", "price_control", "positions")}
    with np.load(outcome_path(year, symbol), allow_pickle=False) as f:
        anchor, end, label = f["anchor"], f["end"], f["label"]
    lo = int(np.datetime64(f"{year}-01-01", "s").astype(np.int64))
    hi = int(np.datetime64(f"{year}-08-01", "s").astype(np.int64))
    if not (
        np.array_equal(d["anchor"], anchor)
        and np.all(end == anchor + 360)
        and np.all(anchor >= lo)
        and np.all(anchor < hi)
        and np.all(end < BOUNDARY)
    ):
        raise ValueError("historical membership/target/protected boundary differs")
    return {
        **d,
        "issue": anchor.astype(float),
        "label_at": end.astype(float),
        "label": label,
    }


def live():
    c = sqlite3.connect((CHECKPOINT / "events.sqlite").as_uri() + "?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    c.execute("BEGIN")
    forecasts, acks, outcomes = {}, {}, {}
    for e in c.execute(
        "SELECT * FROM events WHERE kind IN ('forecast','forecast_commit','outcome') ORDER BY seq"
    ):
        p = json.loads(e["payload"])
        key = p["anchor"], p["symbol"]
        if e["kind"] == "forecast":
            forecasts[key] = p
        elif e["kind"] == "forecast_commit":
            acks[key] = p
        else:
            outcomes[key] = (p, e["received_at"])
    c.close()
    result = {}
    for symbol in SYMBOLS:
        keys = sorted(
            k for k in forecasts if k[1] == symbol and k in acks and acks[k]["eligible"]
        )
        result[symbol] = {
            "anchor": np.array([k[0] for k in keys], dtype=np.int64),
            "issue": np.array([forecasts[k]["issued_at"] for k in keys]),
            "label_at": np.array(
                [outcomes[k][1] if k in outcomes else np.nan for k in keys]
            ),
            "label": np.array(
                [outcomes[k][0]["label"] if k in outcomes else -1 for k in keys],
                dtype=np.int64,
            ),
            **{
                arm: np.array([forecasts[k]["models"][arm] for k in keys])
                for arm in ARMS[:3]
            },
        }
    return result


def run(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if (
        digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["protocol.json"]
    ):
        raise ValueError("protocol changed")
    for section in ("code", "inputs"):
        for name, expected in protocol[section].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"frozen {section} changed: {name}")
    if (root / "results.json").exists():
        raise FileExistsError("results already exist")
    start = time.time()
    cohorts = {}
    data = {f"{y}-{s}": historical(y, s) for y in (2024, 2025, 2026) for s in SYMBOLS}
    data.update({f"live-{s}": d for s, d in live().items()})
    for key, d in data.items():
        derived = replay(d["baseline"], d["issue"], d["label_at"], d["label"])
        d.update(derived)
        with (root / f"{key}-predictions.npz").open("xb") as handle:
            np.savez_compressed(handle, **d)
        scored = d["label"] >= 0
        fixed = scored & (np.maximum(d["baseline"][:, 0], d["baseline"][:, 2]) >= 0.52)
        arms = {}
        for arm in ARMS:
            p = d[arm]
            own = scored & (np.maximum(p[:, 0], p[:, 2]) >= 0.52)
            selected = scores(p, d["label"], own)
            if selected["n"]:
                selected["original_accuracy_flat_errors"] = float(
                    (p[own].argmax(1) == d["label"][own]).mean()
                )
            selected["coverage"] = float(own.sum() / scored.sum())
            arms[arm] = {
                "all": scores(p, d["label"], scored),
                "own_selection": selected,
                "fixed_baseline_selection": scores(p, d["label"], fixed),
            }
        gains = {}
        for control in ("baseline", "price_control", "positions", "neutral"):
            base, new = arms[control]["all"], arms["adaptive"]["all"]
            gains[control] = {
                "accuracy_change": new["direction_accuracy"]
                - base["direction_accuracy"],
                "brier_gain": base["brier"] - new["brier"],
                "logloss_gain": base["log_loss"] - new["log_loss"],
            }
        cohorts[key] = {
            "scored": int(scored.sum()),
            "pending": int((~scored).sum()),
            "models": arms,
            "adaptive_gains": gains,
            "last_weights": d["weights"][-1].tolist(),
            "strict_point_improvement": all(
                all(v > 0 for v in gains[c].values())
                for c in ("baseline", "price_control")
            ),
        }
    historical_keys = [k for k in cohorts if not k.startswith("live-")]
    results = {
        "cohorts": cohorts,
        "historical_stable_point_improvement": all(
            cohorts[k]["strict_point_improvement"] for k in historical_keys
        ),
        "historical_pass_count": sum(
            cohorts[k]["strict_point_improvement"] for k in historical_keys
        ),
        "live_replay_is_prospective_evidence": False,
        "seconds": time.time() - start,
    }
    write(root / "results.json", results)
    write(
        root / "manifest.json",
        {p.name: digest(p) for p in root.iterdir() if p.is_file()},
    )
    print(
        json.dumps(
            {
                "historical_pass_count": results["historical_pass_count"],
                "stable_improvement": results["historical_stable_point_improvement"],
                "seconds": results["seconds"],
                "cohorts": {k: v["adaptive_gains"] for k, v in cohorts.items()},
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["register", "run"])
    p.add_argument("--root", type=Path, default=DEFAULT)
    a = p.parse_args()
    register(a.root.resolve()) if a.command == "register" else run(a.root.resolve())


if __name__ == "__main__":
    main()
