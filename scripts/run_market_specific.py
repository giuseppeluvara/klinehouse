#!/usr/bin/env python3
"""Registered fixed agreement and one-coefficient asset-aware direction study."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_market_specific as model  # noqa: E402
from research import timestral_simple_direction as scalar  # noqa: E402
from research import timestral_uncertain_market as shrink  # noqa: E402
from research.timestral_market_calibration import reliability  # noqa: E402
from research import timestral_universal as shared  # noqa: E402
from research.timestral_futures_flow import combine, comparison  # noqa: E402
from scripts import run_uncertain_market as previous  # noqa: E402
from scripts.run_futures_flow import (
    ROOT,
    digest,
    write,
    epoch,
    sources,
    prior_path,
    examples,
)  # noqa: E402

DEFAULT = ROOT / "reports/timestral-market-specific-20261027"
BASE = previous.DEFAULT
FUTURES = previous.FUTURES
ARMS = ("consensus", "blend")
PAIRS = (
    ("consensus", "common"),
    ("consensus", "blend"),
    ("consensus", "neutral"),
    ("consensus", "flow"),
    ("blend", "common"),
    ("blend", "neutral"),
    ("blend", "flow"),
)
CODE = (
    "research/timestral_market_specific.py",
    "scripts/run_market_specific.py",
    "scripts/verify_market_specific.py",
    "tests/test_timestral_market_specific.py",
    *previous.CODE,
)
PLAN = {
    "purpose": "Diagnose the three original-gate failures without assuming all are binary-direction failures; test fixed asset agreement and one-coefficient local/common blending on all nine cohorts.",
    "status": "Retrospective Jan-Jul2024/2025/2026 BTC/ETH/SOL. Prior failure identities, flat/direction decomposition and sign-disagreement diagnostics informed this hypothesis. Not untouched or prospective validation; protocol before new candidate scores.",
    "diagnosis": "For the frozen adaptive common model, record observed/predicted flats, fixed original52% selected coverage and binary versus all-class accuracy. Split every cohort by sign agreement/disagreement/zero of common and own normalized hourly returns, reporting counts and common/own accuracies on nonflat cases and the original selected subset. Descriptive only, no significance/causal claim and no optimized threshold.",
    "signals": "common=clip(-(own_return60+2*other_return60)/3,-3,3), own=clip(-own_return60,-3,3), blend=(common+own)/2. Frozen causal RMS-normalized returns, spot bars through anchor. Equal mixing fixed in advance; no asset IDs or regime fitting.",
    "primary": "Consensus retains frozen adaptive common up probability only if common*own>0; otherwise returns.5. All anchors retained, no outcome-driven filtering and no new fitted parameter. Same flat probabilities. General criterion requires all9 original consensus gates AND all9 conditional information gates vs common, neutral and flow; compare with blend diagnostically.",
    "secondary": "Blend uses one shared nonnegative scalar fit on pooled3asset matured90day nonflat labels, equal UTC-day weights, same futures availability membership as previous controls. Reuse frozen scalar MLE and HAC uncertainty shrinkage algorithms unchanged.21 new scalar fits/adjustments, no mixture-weight/horizon/penalty sweep. Original gates and pairwise comparisons reported on all9 cohorts without promoting the secondary after scores.",
    "scope": "Same550362 original evaluation anchors and five-minute spot target anchor+60sec to anchor+360sec. Flat probabilities stay unchanged to isolate direction. Ties/consensus abstention are reported; scoring assigns up at.5, which is not evidence of directional ability. Original52% selection excludes neutral predictions.",
    "thresholds": "Unchanged original thresholds. Conditional: accuracy lower99.444%>50%, Brier/logloss gain lower99.444%>0 against named control AND historical frequency,>=5 positive monthly Brier gains,>=100 each direction.4000 paired7day blocks seed271032;9cohort correction per endpoint, not jointly over arms/endpoints/sequential research. No future-flat identification assumed.",
    "protected": "No source/outcome from2026-08-01 onward. Earlier frozen models/data/results and user files preserved. Only reused arrays, no downloads/installations/orders or profitability claim.",
}


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    paths = [
        p
        for year in (2024, 2025, 2026)
        for p in [
            *sources(year).values(),
            *[prior_path(year, s) for s in shared.SYMBOLS],
            *[
                FUTURES / f"{year}-{s}-{suffix}.npz"
                for s in shared.SYMBOLS
                for suffix in ("futures", "predictions")
            ],
            *[BASE / f"{year}-{s}-predictions.npz" for s in shared.SYMBOLS],
        ]
    ]
    write(
        root / "protocol.json",
        dict(
            at=datetime.now(timezone.utc).isoformat(),
            plan=PLAN,
            code={p: digest(ROOT / p) for p in CODE},
            inputs={str(p.relative_to(ROOT)): digest(p) for p in paths},
        ),
    )
    write(root / "protocol.sha256.json", dict(sha256=digest(root / "protocol.json")))


def validate(root):
    p = json.loads((root / "protocol.json").read_text())
    if (
        p["plan"] != PLAN
        or digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
        or any(digest(ROOT / n) != h for n, h in {**p["code"], **p["inputs"]}.items())
    ):
        raise ValueError("frozen experiment changed")


def run(root):
    validate(root)
    write(root / "opened.json", dict(at=datetime.now(timezone.utc).isoformat()))
    fits, results, diagnosis = {}, {}, {}
    for year in (2024, 2025, 2026):
        sample = examples(FUTURES, year)
        t, end, y = sample["anchor"], sample["end"], sample["y"]
        common, own, z = model.signals(sample["x"])
        pieces, indices = [], []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (t >= cutoff - 90 * 86400) & (end <= cutoff) & sample["available"]
            labels = y[:, train].reshape(-1)
            moving = labels != 1
            yy = (labels[moving] == 2).astype(int)
            zz = z[:, train].reshape(-1)[moving]
            tt = np.tile(t[train], 3)[moving]
            ee = np.tile(end[train], 3)[moving]
            raw = scalar.fit(zz, yy, tt, ee, cutoff, 0)
            adjusted = shrink.adjust(raw, zz, yy, tt, ee, cutoff)
            ids = np.concatenate(
                [np.column_stack((np.full(train.sum(), a), t[train])) for a in range(3)]
            ).astype("<i8")[moving]
            for state in (raw, adjusted):
                state["training_ids_sha256"] = hashlib.sha256(ids.tobytes()).hexdigest()
                state["training_labels_sha256"] = hashlib.sha256(
                    yy.astype("<i8").tobytes()
                ).hexdigest()
            fits[f"{year}-{month}"] = dict(raw=raw, adjusted=adjusted)
            test = (t >= cutoff) & (t < epoch(year, month + 1))
            pieces.append(
                np.stack(
                    [shrink.predict(adjusted, z[a, test], t[test]) for a in range(3)]
                )
            )
            indices.append(np.flatnonzero(test))
        indices = np.concatenate(indices)
        blend = np.concatenate(pieces, axis=1)
        for a, s in enumerate(shared.SYMBOLS):
            key = f"{year}-{s}"
            prior = dict(np.load(prior_path(year, s)))
            old = dict(np.load(BASE / f"{key}-predictions.npz"))
            flow = dict(np.load(FUTURES / f"{key}-predictions.npz"))
            for d in (prior, old, flow):
                if not np.array_equal(d["anchor"], t[indices]):
                    raise ValueError("evaluation anchors changed")
            c, o = common[a, indices], own[a, indices]
            q = dict(
                common=old["adaptive_up"],
                blend=blend[a],
                consensus=model.consensus(old["adaptive_up"], c, o),
                flow=flow["flow_up"],
                neutral=np.full(len(indices), 0.5),
            )
            labels, anchors = y[a, indices], t[indices]
            flat = prior["cross" if year == 2026 else "probability"][:, 1]
            frequency = prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1])
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(
                    f, anchor=anchors, **{name + "_up": q[name] for name in ARMS}
                )
            diagnosis[key] = model.diagnose(labels, q["common"], flat, c, o)
            r = dict(
                comparisons={
                    f"{candidate}_vs_{control}": comparison(
                        labels, q[candidate], q[control], frequency, anchors
                    )
                    for candidate, control in PAIRS
                },
                calibration={name: reliability(labels, p) for name, p in q.items()},
                direction_changes={
                    name: int(np.sum((q[name] >= 0.5) != (q["common"] >= 0.5)))
                    for name in ARMS
                },
                consensus_abstentions=int(np.sum(c * o <= 0)),
            )
            for name in ARMS:
                r[name] = shared.evaluate(
                    labels,
                    combine(q[name], flat),
                    prior["frequency"],
                    prior["prior_reversal"],
                    anchors,
                )
            results[key] = r
            print(
                key,
                {name: r[name]["gate"] for name in ARMS},
                {name: v["information_gate"] for name, v in r["comparisons"].items()},
                flush=True,
            )
    write(root / "fits.json", fits)
    write(root / "diagnosis.json", diagnosis)
    information = all(
        r["comparisons"][f"consensus_vs_{c}"]["information_gate"]
        for r in results.values()
        for c in ("common", "neutral", "flow")
    )
    direction = all(r["consensus"]["gate"] for r in results.values())
    write(
        root / "results.json",
        dict(
            cohorts=results,
            all_information_gates=information,
            all_direction_gates=direction,
            general_solution=information and direction,
        ),
    )
    write(
        root / "manifest.json",
        {p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"register": register, "run": run}[args.action](args.root)
