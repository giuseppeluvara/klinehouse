#!/usr/bin/env python3
"""Register, aggregate existing seconds, and run the fixed direction experiment."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_microdirection as model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CODE = (
    "research/timestral_microdirection.py",
    "scripts/run_timestral_microdirection.py",
)
ARCHIVES = (
    "timestral-adaptation-20261025/development",
    "timestral-adaptation-20261025/test",
    "timestral-long-horizon-20261025",
)
PLAN = {
    "hypothesis": "Minute and last-ten-second taker flow predict short-horizon direction beyond price/activity regimes.",
    "source": "Existing BTCUSDT one-second OHLCV, 2025-09-01 through 2026-07-31; never order-book events.",
    "evaluation": "2026-01-01 through 2026-07-31, monthly past-90-day fitting; label end <= fit cutoff.",
    "status": "Historical falsification on previously inspected price periods; no pristine or prospective holdout claim.",
    "target": "Three classes: negative, exactly zero, positive terminal return; retain all ties.",
    "horizons_minutes": [1, 5, 15],
    "delays_minutes": [0, 1],
    "primary": "seconds model, 5-minute horizon, 1-minute delay; all other runs diagnostic",
    "features": "Current return/past 60-minute RMS bins [-1,-.25,.25,1]; trades/past60 mean bins [.5,1.5]; minute taker imbalance bins [-.6,-.2,.2,.6]; last10second imbalance bins [-.3,.3].",
    "models": "Smoothed global class frequency; 15 price/activity cells; 75 price/activity/flow cells; 225 cells including last10second flow. Hierarchical Dirichlet prior weights 100,100,150; no selection.",
    "sampling": "UTC anchors every horizon minutes; complete 62-minute context and full target path; no overlap within each run; no trade/executable-price claim.",
    "gate": "Primary Brier improves >=1% vs both frequency and price; paired 7-day block bootstrap 95% lower bound >0; no log-loss degradation; positive gain in >=5 of 7 months for both comparators.",
    "bootstrap": "2000 noncircular moving blocks, seed 271027; conditional on saved fitted models, not whole fitting process",
    "storage": "Reuse existing archives and environment; estimated outputs <200MiB, no downloads or installations.",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, separators=(",", ":"), allow_nan=False)
        f.write("\n")


def epoch(text):
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


def register(root):
    files = []
    for archive in ARCHIVES:
        folder = ROOT / "research/out" / archive / "archive"
        catalog = json.loads((folder / "catalog.json").read_text())
        for entry in catalog["files"]:
            if (
                epoch(entry["start"]) < epoch("2025-09-01T00:00:00Z")
                or epoch(entry["end"]) > model.BOUNDARY
            ):
                raise ValueError("source outside permitted interval")
            if entry["symbol"] != "BTCUSDT" or not entry["complete"] or entry["gaps"]:
                raise ValueError("source must be complete BTCUSDT days")
            files.append(
                {
                    "path": str((folder / entry["path"]).relative_to(ROOT)),
                    "sha256": entry["sha256"],
                    "start": entry["start"],
                    "end": entry["end"],
                }
            )
    files.sort(key=lambda f: f["start"])
    p = {
        "at": datetime.now(timezone.utc).isoformat(),
        "plan": PLAN,
        "code": {n: digest(ROOT / n) for n in CODE},
        "files": files,
        "numpy": np.__version__,
    }
    write(root / "protocol.json", p)
    write(root / "protocol.sha256.json", {"sha256": digest(root / "protocol.json")})


def validate(root):
    p = json.loads((root / "protocol.json").read_text())
    if (
        p["plan"] != PLAN
        or p["code"] != {n: digest(ROOT / n) for n in CODE}
        or digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
    ):
        raise ValueError("frozen protocol or code changed")
    return p


def prepare(root):
    p = validate(root)
    parts = {k: [] for k in ("time", "close", "volume", "taker", "trades", "tail_flow")}
    started = time.perf_counter()
    cols = ["open_time", "close_time", "close", "volume", "taker_buy_base", "trades"]
    for number, entry in enumerate(p["files"]):
        path = ROOT / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError("raw source hash mismatch")
        table = pq.read_table(path, columns=cols)
        t = table["open_time"].to_numpy().astype("datetime64[us]").astype(np.int64)
        ct = table["close_time"].to_numpy().astype("datetime64[us]").astype(np.int64)
        start = epoch(entry["start"])
        if (
            len(t) != 86400
            or not np.array_equal(t, (start + np.arange(86400)) * 1000000)
            or np.any(ct < t)
            or np.any(ct >= t + 1000000)
        ):
            raise ValueError("incomplete or unclosed source day")
        values = {k: table[k].to_numpy().reshape(1440, 60) for k in cols[2:]}
        if not all(np.isfinite(v).all() for v in values.values()) or np.any(
            values["close"] <= 0
        ):
            raise ValueError("invalid source values")
        if any(np.any(values[k] < 0) for k in cols[3:]) or np.any(
            values["taker_buy_base"] > values["volume"] + 1e-10
        ):
            raise ValueError("invalid raw flow")
        vol = values["volume"][:, -10:].sum(axis=1)
        taker = values["taker_buy_base"][:, -10:].sum(axis=1)
        parts["time"].append((t.reshape(1440, 60)[:, 0] // 1000000).astype(np.int64))
        parts["close"].append(values["close"][:, -1])
        for name, col in (
            ("volume", "volume"),
            ("taker", "taker_buy_base"),
            ("trades", "trades"),
        ):
            parts[name].append(values[col].sum(axis=1))
        parts["tail_flow"].append(
            np.divide(2 * taker - vol, vol, out=np.zeros(1440), where=vol > 0)
        )
        if number % 60 == 0:
            print(
                f"verified and aggregated {number + 1}/{len(p['files'])} days",
                flush=True,
            )
    data = {k: np.concatenate(v) for k, v in parts.items()}
    if np.any(np.diff(data["time"]) != 60) or data["time"][-1] >= model.BOUNDARY:
        raise ValueError("source gaps or protected data")
    with (root / "minutes.npz").open("xb") as f:
        np.savez_compressed(f, **data)
    write(
        root / "source.json",
        {
            "sha256": digest(root / "minutes.npz"),
            "minutes": len(data["time"]),
            "seconds": time.perf_counter() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )


def run(root):
    validate(root)
    if (
        digest(root / "minutes.npz")
        != json.loads((root / "source.json").read_text())["sha256"]
    ):
        raise ValueError("derived source changed")
    write(root / "opened.json", {"at": datetime.now(timezone.utc).isoformat()})
    data = dict(np.load(root / "minutes.npz"))
    results, fits = {}, {}
    started = time.perf_counter()
    for horizon in (1, 5, 15):
        for delay in (0, 1):
            key = f"h{horizon}_d{delay}"
            s = model.samples(data, horizon, delay)
            predictions, selected = [], []
            for month in range(1, 8):
                cutoff = epoch(f"2026-{month:02d}-01T00:00:00Z")
                stop = epoch(f"2026-{month + 1:02d}-01T00:00:00Z")
                train = (s["anchor"] >= cutoff - 90 * 86400) & (s["end"] <= cutoff)
                test = (s["anchor"] >= cutoff) & (s["anchor"] < stop)
                fit = model.fit(
                    s["cells"][train], s["label"][train], s["end"][train], cutoff
                )
                fit["training_anchor_sha256"] = hashlib.sha256(
                    s["anchor"][train].astype("<i8").tobytes()
                ).hexdigest()
                fits[f"{key}_{month}"] = fit
                predictions.append(
                    model.predict(fit, s["cells"][test], s["anchor"][test])
                )
                selected.append(np.flatnonzero(test))
            ids = np.concatenate(selected)
            pred = np.concatenate(predictions)
            # Separate files preserve forecasts independently from their outcomes.
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(
                    f, anchor=s["anchor"][ids], cells=s["cells"][ids], probability=pred
                )
            with (root / f"{key}-outcomes.npz").open("xb") as f:
                np.savez_compressed(
                    f,
                    anchor=s["anchor"][ids],
                    end=s["end"][ids],
                    label=s["label"][ids],
                    returns=s["return"][ids],
                )
            results[key] = model.scores(s["label"][ids], pred, s["anchor"][ids])
            print(key, json.dumps(results[key]["comparisons"]), flush=True)
    write(root / "fits.json", fits)
    write(
        root / "results.json",
        {
            "runs": results,
            "primary_gate": results["h5_d1"]["gate"],
            "seconds": time.perf_counter() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )
    write(
        root / "manifest.json",
        {
            p.name: digest(p)
            for p in sorted(root.iterdir())
            if p.suffix in (".json", ".npz") and p.name != "protected-before.json"
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {"register": register, "prepare": prepare, "run": run}[args.action](args.output)
