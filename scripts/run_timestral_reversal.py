#!/usr/bin/env python3
"""Freeze and execute a selective, sign-symmetric direction follow-up."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_microdirection as micro  # noqa: E402
from research import timestral_reversal as reversal  # noqa: E402
from scripts.run_timestral_microdirection import ROOT, digest, epoch, write  # noqa: E402

CODE = (
    "research/timestral_reversal.py",
    "scripts/run_timestral_reversal.py",
    "research/timestral_microdirection.py",
    "scripts/run_timestral_microdirection.py",
)
SOURCE = ROOT / "research/out/timestral-microdirection-20261027"
PLAN = {
    "reason": "Follow-up after the microdirection results: flow hurts, small price signal remains. These historical outcomes were already inspected; this is explicitly exploratory.",
    "target": "Down, exactly flat, up over 1,5,15 minutes after a one-minute delay; primary five minutes.",
    "method": "Pool positive and negative input returns by sign symmetry. Model relative reversal/flat/continuation frequencies in 12 fixed cells: abs(current return / past60 RMS) bins [.5,1,2] x trades/past60 mean bins [.5,1.5]. Shrink each cell by 100 pseudo-observations toward pooled relative frequencies. Add-one global priors. No optimization.",
    "training": "Monthly 90-day history, only matured labels; identical samples to initial experiment delay=1.",
    "selection": "Emit directional signal only for nonzero observed move and absolute predicted down or up probability >=.52. Otherwise abstain. Selection uses only past-fitted probabilities. Score all samples and selected samples separately; retain flat outcomes as errors for directional calls.",
    "baselines": "Past unconditional frequencies and pooled reversal frequencies on identical selected samples.",
    "gate": "Primary selected coverage >=5%, accuracy >=52%, paired 7-day block bootstrap lower95 accuracy >50% and Brier gain vs unconditional >0; no selected log-loss degradation; positive selected Brier gain in >=5 months with >=100 calls each; >=100 calls in each predicted direction. No primary success claimed from secondary horizons.",
    "bootstrap": "2000 noncircular seven-day moving blocks, seed271028; fixed fitted-model uncertainty only.",
    "protected": "No data >=2026-08-01; no changes to any previous protocol or model.",
}


def register(root):
    write(
        root / "protocol.json",
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "plan": PLAN,
            "code": {n: digest(ROOT / n) for n in CODE},
            "source_sha256": digest(SOURCE / "minutes.npz"),
            "prior_results_sha256": digest(SOURCE / "results.json"),
        },
    )
    write(root / "protocol.sha256.json", {"sha256": digest(root / "protocol.json")})


def run(root):
    p = json.loads((root / "protocol.json").read_text())
    if (
        p["plan"] != PLAN
        or p["code"] != {n: digest(ROOT / n) for n in CODE}
        or p["source_sha256"] != digest(SOURCE / "minutes.npz")
        or p["prior_results_sha256"] != digest(SOURCE / "results.json")
        or digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
    ):
        raise ValueError("frozen follow-up changed")
    write(root / "opened.json", {"at": datetime.now(timezone.utc).isoformat()})
    data = dict(np.load(SOURCE / "minutes.npz"))
    fits, results = {}, {}
    for h in (1, 5, 15):
        s = micro.samples(data, h, 1)
        cells, signs = reversal.states(data, s["index"])
        predictions, selections, indices = [], [], []
        for month in range(1, 8):
            cutoff = epoch(f"2026-{month:02d}-01T00:00:00Z")
            stop = epoch(f"2026-{month + 1:02d}-01T00:00:00Z")
            train = (s["anchor"] >= cutoff - 90 * 86400) & (s["end"] <= cutoff)
            test = (s["anchor"] >= cutoff) & (s["anchor"] < stop)
            fit = reversal.fit(
                cells[train], signs[train], s["label"][train], s["end"][train], cutoff
            )
            fit["training_anchor_sha256"] = hashlib.sha256(
                s["anchor"][train].astype("<i8").tobytes()
            ).hexdigest()
            fits[f"h{h}_{month}"] = fit
            pred, sel = reversal.predict(
                fit, cells[test], signs[test], s["anchor"][test]
            )
            predictions.append(pred)
            selections.append(sel)
            indices.append(np.flatnonzero(test))
        pred, selected, ids = (
            np.concatenate(predictions),
            np.concatenate(selections),
            np.concatenate(indices),
        )
        with (root / f"h{h}-predictions.npz").open("xb") as f:
            np.savez_compressed(
                f,
                anchor=s["anchor"][ids],
                probability=pred,
                selected=selected,
                cells=cells[ids],
                signs=signs[ids],
            )
        results[f"h{h}"] = reversal.evaluate(
            s["label"][ids], pred, s["anchor"][ids], selected
        )
        print(h, json.dumps(results[f"h{h}"]), flush=True)
    write(root / "fits.json", fits)
    write(
        root / "results.json", {"runs": results, "primary_gate": results["h5"]["gate"]}
    )
    write(
        root / "manifest.json",
        {
            p.name: digest(p)
            for p in sorted(root.iterdir())
            if p.suffix in (".json", ".npz")
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {"register": register, "run": run}[args.action](args.output)
