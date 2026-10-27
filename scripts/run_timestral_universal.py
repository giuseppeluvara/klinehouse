#!/usr/bin/env python3
"""Register and compare local, shared, cross-market and unseen-asset models."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_universal as model  # noqa: E402
from scripts.run_timestral_microdirection import ROOT, digest, epoch, write  # noqa: E402

PRIOR = ROOT / "reports/timestral-microdirection-20261027"
SOURCES = {
    "BTCUSDT": PRIOR / "micro/minutes.npz",
    **{s: PRIOR / f"transfer/{s}-minutes.npz" for s in model.SYMBOLS[1:]},
}
BASELINES = {
    "BTCUSDT": PRIOR / "reversal/h5-predictions.npz",
    **{s: PRIOR / f"transfer/{s}-predictions.npz" for s in model.SYMBOLS[1:]},
}
CODE = (
    "research/timestral_universal.py",
    "scripts/run_timestral_universal.py",
    "scripts/run_timestral_microdirection.py",
)
PLAN = {
    "purpose": "One common direction rule across BTC, ETH and SOL; asset-general does not mean every market or horizon.",
    "history_status": "All January-July 2026 prices already inspected; historical research, never pristine validation. No source or target >=2026-08-01 is opened.",
    "target": "Three-class direction, five-minute close return starting one minute after anchor; UTC 5m grid, no dropped flats.",
    "features": list(model.FEATURES),
    "odd_features": list(model.ODD),
    "normalization": "Own returns divided by past60 RMS and sqrt(horizon), clipped to +-8; other markets represented by their normalized returns. Past-only train standardization and +-8 standardized clipping. No asset identity feature.",
    "training": "Monthly trailing90 days with matured labels, all three assets aligned; reflection augmentation flips odd features and swaps down/up labels. Same regularized multinomial logistic model C=.1, lbfgs max500 tol1e-8.",
    "families": {
        "local": "13 own features, per-asset fit",
        "shared": "13 own features, one fit across all assets",
        "cross": "23 features including other-market context, one fit across all assets; PRIMARY",
        "leave_one_out": "23 features, target asset labels excluded from fitting; transfer diagnostic",
    },
    "evaluation": "January-July2026; all families and assets must be evaluated, no variant selection.",
    "selection": "Same probability threshold52% for up/down on every asset; abstain otherwise.",
    "baselines": "Per-asset past frequency; prior frozen selective-reversal probabilities, both scored on the new model's exact selected cases.",
    "primary_gate": "Cross family must pass on ALL three assets: coverage>=5%, accuracy>=52%, simultaneous98.333% lower accuracy>50% and Brier gain vs frequency>0, no log-loss degradation, >=100 predictions in each direction, positive gain in >=5 months with >=100 calls each. Previous asset-specific criteria are not relaxed.",
    "bootstrap": "4000 paired seven-day moving-block replicates, seed271029, same sampled days across assets; Bonferroni across3 asset checks. Not adjusted for previous exploratory research or refitting uncertainty.",
    "storage": "Existing compressed sources/environment reused; estimated new outputs <100MiB; no downloads/installations.",
}


def register(root):
    write(
        root / "protocol.json",
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "plan": PLAN,
            "code": {n: digest(ROOT / n) for n in CODE},
            "inputs": {
                str(p.relative_to(ROOT)): digest(p)
                for p in [*SOURCES.values(), *BASELINES.values()]
            },
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    )
    write(root / "protocol.sha256.json", {"sha256": digest(root / "protocol.json")})


def run(root):
    p = json.loads((root / "protocol.json").read_text())
    if (
        p["plan"] != PLAN
        or p["code"] != {n: digest(ROOT / n) for n in CODE}
        or any(digest(ROOT / n) != h for n, h in p["inputs"].items())
        or digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
    ):
        raise ValueError("frozen experiment changed")
    write(root / "opened.json", {"at": datetime.now(timezone.utc).isoformat()})
    start = time.perf_counter()
    s = model.examples({name: dict(np.load(path)) for name, path in SOURCES.items()})
    fits = {}
    output = {f: {a: [] for a in range(3)} for f in PLAN["families"]}
    frequency = {a: [] for a in range(3)}
    selected_indices = []
    for month in range(1, 8):
        cutoff = epoch(f"2026-{month:02d}-01T00:00:00Z")
        stop = epoch(f"2026-{month + 1:02d}-01T00:00:00Z")
        train = (s["anchor"] >= cutoff - 90 * 86400) & (s["end"] <= cutoff)
        test = (s["anchor"] >= cutoff) & (s["anchor"] < stop)
        selected_indices.append(np.flatnonzero(test))
        for a in range(3):
            count = np.bincount(s["y"][a, train], minlength=3)
            frequency[a].append(
                np.tile((count + 1) / (train.sum() + 3), (test.sum(), 1))
            )
        for family in PLAN["families"]:
            groups = (
                [(a, [a]) for a in range(3)]
                if family == "local"
                else [(a, [b for b in range(3) if b != a]) for a in range(3)]
                if family == "leave_one_out"
                else [("all", [0, 1, 2])]
            )
            columns = 13 if family in ("local", "shared") else 23
            for target, assets in groups:
                x = np.concatenate([s["x"][a, train, :columns] for a in assets])
                y = np.concatenate([s["y"][a, train] for a in assets])
                ends = np.tile(s["end"][train], len(assets))
                f = model.fit(x, y, ends, cutoff)
                ids = np.concatenate(
                    [
                        np.column_stack((np.full(train.sum(), a), s["anchor"][train]))
                        for a in assets
                    ]
                ).astype("<i8")
                f.update(
                    assets=assets,
                    columns=columns,
                    training_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                )
                key = f"{family}_{month}_{target}"
                fits[key] = f
                for a in range(3) if target == "all" else [target]:
                    output[family][a].append(
                        model.predict(f, s["x"][a, test, :columns], s["anchor"][test])
                    )
        print(f"completed monthly fits {month}/7", flush=True)
    ids = np.concatenate(selected_indices)
    results = {}
    for a, symbol in enumerate(model.SYMBOLS):
        prior = dict(np.load(BASELINES[symbol]))
        if not np.array_equal(prior["anchor"], s["anchor"][ids]):
            raise ValueError("baseline sample mismatch")
        baseline = np.concatenate(frequency[a])
        preds = {
            family: np.concatenate(output[family][a]) for family in PLAN["families"]
        }
        with (root / f"{symbol}-predictions.npz").open("xb") as f:
            np.savez_compressed(
                f,
                anchor=s["anchor"][ids],
                frequency=baseline,
                prior_reversal=prior["probability"][:, 2],
                **preds,
            )
        with (root / f"{symbol}-outcomes.npz").open("xb") as f:
            np.savez_compressed(
                f, anchor=s["anchor"][ids], end=s["end"][ids], label=s["y"][a, ids]
            )
        results[symbol] = {
            family: model.evaluate(
                s["y"][a, ids],
                q,
                baseline,
                prior["probability"][:, 2],
                s["anchor"][ids],
            )
            for family, q in preds.items()
        }
        print(
            symbol,
            json.dumps(
                {
                    k: {n: v[n] for n in ("coverage", "accuracy", "gate") if n in v}
                    for k, v in results[symbol].items()
                }
            ),
            flush=True,
        )
    write(root / "fits.json", fits)
    write(
        root / "results.json",
        {
            "assets": results,
            "primary_gate": all(v["cross"]["gate"] for v in results.values()),
            "seconds": time.perf_counter() - start,
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
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {"register": register, "run": run}[args.action](args.output)
