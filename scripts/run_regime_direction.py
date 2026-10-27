#!/usr/bin/env python3
"""Preregistered long-context and recurring-regime direction comparison."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_regime_direction as model  # noqa: E402
from research import timestral_universal as shared  # noqa: E402
from scripts import run_futures_flow as previous  # noqa: E402
from scripts.run_futures_flow import ROOT, digest, write, epoch, sources, prior_path  # noqa: E402

DEFAULT = ROOT / "reports/timestral-regime-direction-20261027"
FUTURES = ROOT / "reports/timestral-futures-flow-20261027"
CODE = (
    "research/timestral_regime_direction.py",
    "scripts/run_regime_direction.py",
    *previous.CODE,
)
PLAN = {
    "purpose": "Distinguish changing regime frequencies from changing within-regime fixed-signal relationships; test whether causal long context and conditional slopes generalize up/down direction.",
    "status": "Exploratory retrospective Jan-Jul2024/2025/2026 BTC/ETH/SOL; outcomes previously explored, no pristine or prospective validation claim.",
    "sources": "Existing frozen pre-protected spot and USD-M futures minute arrays only; no downloads or protected outcomes. All original evaluation anchors retained.",
    "target": "Sign of five-minute spot return from anchor+1min to anchor+6min. Non-flat labels for binary fitting; unchanged frozen flat probabilities recombined for original unconditional gates.",
    "context": "Spot closes through anchor, futures through anchor-60sec. Require1501 complete spot minutes. Signed4h/24h log return divided by sqrt(sum squared1m log returns over that horizon), denominator floor sqrt(h*1e-16), clipped+-8. Append own4h/24h and other-two-asset means4h/24h. No source later than2026-07-31 except already admitted final minute targets2024/25.",
    "regimes": "Fixed four states: abs(normalized4h return)>=1 versus<1; mean trade count last60minutes divided by max(mean preceding1440minutes excluding those60,1)>=1 versus<1. State=2*trend+active; no label-driven thresholds.",
    "models": "Flow35; long-context39; regime71=long39 plus eight signed core features times four centered state indicators(onehot-.25). Core indices0,1,4,27,28,13,35,36. Shared coefficients and regularized regime deviations; no per-asset ID.",
    "fit": "Same monthly trailing90days matured pooled3asset samples for all arms, same sign-reflected logistic C=.1 max500 tol1e-8, train-only mean/std and+-8 clipping. All added columns23+ odd.63fits. No sweep after scores.",
    "missing": "Common eligible training contexts. On missing evaluation context long/regime fall back to flow; all original anchors retained. No filling of market data.",
    "comparisons": ["long_vs_flow", "regime_vs_flow", "regime_vs_long"],
    "primary": "All9 cohorts must pass BOTH regime-vs-flow and regime-vs-long conditional information gates, plus all9 original unconditional regime direction gates. Conditional accuracy lower99.444%>50%; Brier/logloss gain lower99.444%>0 against named control AND historical frequency; >=5 positive monthly Brier gains; >=100 forecasts each direction. Frozen scoring functions unchanged.",
    "uncertainty": "Conditional comparisons use4000 paired7day moving blocks seed271032,9-cohort correction per endpoint, no refitting and no joint correction over endpoints/comparisons/history. Original gate unchanged. No tradable-coverage inference from excluding flats.",
    "diagnosis": "Fixed rules: opposite sign of spot1m return and sign of futures1m aggressor flow, neither refit. On non-flat outcomes and nonzero signal, report rates/weights per fixed state,95% marginal seven-day block CIs seed271033. Describe2025/2026 minus2024 change as sum((w_new-w_old)*accuracy_old) plus sum(w_new*(accuracy_new-accuracy_old)). Exact descriptive decomposition, not causality or a selection criterion; no post-result regime changes. Diagnostic CIs are not multiplicity corrected.",
    "protected": "No market sources/outcomes from2026-08-01 onward. Preserve prior frozen bundles and user files. No trading/profitability claim.",
}


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    paths = [
        p
        for year in (2024, 2025, 2026)
        for p in [
            *sources(year).values(),
            *[prior_path(year, s) for s in shared.SYMBOLS],
            *[FUTURES / f"{year}-{s}-futures.npz" for s in shared.SYMBOLS],
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
        or any(
            digest(ROOT / name) != h for name, h in {**p["code"], **p["inputs"]}.items()
        )
    ):
        raise ValueError("frozen experiment changed")
    return p


def examples(root, year):
    sample = previous.examples(FUTURES, year)
    data = {}
    for s, p in sources(year).items():
        d = dict(np.load(p))
        take = (d["time"] >= epoch(year - 1, 10)) & (d["time"] < epoch(year, 8) + 360)
        data[s] = {k: v[take] for k, v in d.items()}
    valid, extra, state = model.context(data, sample["anchor"])
    additions = np.zeros((3, len(valid), 4))
    additions[:, valid] = extra
    states = np.zeros((3, len(valid)), dtype=np.int8)
    states[:, valid] = state
    extended = np.concatenate((sample["x"], additions), axis=2)
    return {
        **sample,
        "x": model.interact(extended, states),
        "state": states,
        "available": sample["available"] & valid,
    }


def run(root):
    validate(root)
    write(root / "opened.json", {"at": datetime.now(timezone.utc).isoformat()})
    fits, results, diagnostics = {}, {}, {}
    for year in (2024, 2025, 2026):
        sample = examples(root, year)
        predictions = {name: [] for name, _ in model.ARMS}
        ids = []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (sample["anchor"] >= cutoff - 90 * 86400) & (
                sample["end"] <= cutoff
            )
            train &= sample["available"]
            test = (sample["anchor"] >= cutoff) & (
                sample["anchor"] < epoch(year, month + 1)
            )
            y = np.concatenate([sample["y"][a, train] for a in range(3)])
            ends = np.tile(sample["end"][train], 3)
            membership = np.concatenate(
                [
                    np.column_stack((np.full(train.sum(), a), sample["anchor"][train]))
                    for a in range(3)
                ]
            ).astype("<i8")
            for name, columns in model.ARMS:
                x = np.concatenate([sample["x"][a, train, :columns] for a in range(3)])
                moving = y != 1
                state = model.fit(
                    x[moving], (y[moving] == 2).astype(int), ends[moving], cutoff
                )
                state["training_ids_sha256"] = hashlib.sha256(
                    membership[moving].tobytes()
                ).hexdigest()
                state["training_labels_sha256"] = hashlib.sha256(
                    (y[moving] == 2).astype("<i8").tobytes()
                ).hexdigest()
                fits[f"{year}-{month}-{name}"] = state
                predictions[name].append(
                    np.stack(
                        [
                            model.predict(
                                state,
                                sample["x"][a, test, :columns],
                                sample["anchor"][test],
                            )
                            for a in range(3)
                        ]
                    )
                )
                if name != "flow":
                    missing = ~sample["available"][test]
                    predictions[name][-1][:, missing] = predictions["flow"][-1][
                        :, missing
                    ]
            ids.append(np.flatnonzero(test))
            print("fitted", year, month, flush=True)
        ids = np.concatenate(ids)
        q = {k: np.concatenate(v, axis=1) for k, v in predictions.items()}
        for a, s in enumerate(shared.SYMBOLS):
            key = f"{year}-{s}"
            prior = dict(np.load(prior_path(year, s)))
            t, y = sample["anchor"][ids], sample["y"][a, ids]
            if not np.array_equal(t, prior["anchor"]):
                raise ValueError("evaluation anchors changed")
            raw = prior["cross" if year == 2026 else "probability"]
            flat = raw[:, 1]
            frequency = prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1])
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(
                    f,
                    anchor=t,
                    flow_up=q["flow"][a],
                    long_up=q["long"][a],
                    regime_up=q["regime"][a],
                    available=sample["available"][ids],
                    regime=sample["state"][a, ids],
                )
            results[key] = {
                "missing_contexts": int((~sample["available"][ids]).sum()),
                "comparisons": {
                    f"{candidate}_vs_{control}": model.comparison(
                        y, q[candidate][a], q[control][a], frequency, t
                    )
                    for candidate, control in model.COMPARISONS
                },
            }
            for name, _ in model.ARMS:
                results[key][name] = shared.evaluate(
                    y,
                    model.combine(q[name][a], flat),
                    prior["frequency"],
                    prior["prior_reversal"],
                    t,
                )
            diagnostics[key] = model.diagnose(
                y, t, sample["state"][a, ids], sample["x"][a, ids]
            )
            print(
                key,
                {
                    k: v["information_gate"]
                    for k, v in results[key]["comparisons"].items()
                },
                "direction",
                results[key]["regime"]["gate"],
                flush=True,
            )
    write(root / "fits.json", fits)
    write(
        root / "diagnosis.json",
        {"cohorts": diagnostics, "decomposition": model.drift(diagnostics)},
    )
    write(
        root / "results.json",
        {
            "cohorts": results,
            "all_information_gates": all(
                r["comparisons"][c]["information_gate"]
                for r in results.values()
                for c in ("regime_vs_flow", "regime_vs_long")
            ),
            "all_direction_gates": all(r["regime"]["gate"] for r in results.values()),
        },
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
