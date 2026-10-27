#!/usr/bin/env python3
"""Registered price-path and volume-center experiment on existing sources."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_volume_center as model  # noqa: E402
from research import timestral_simple_direction as scalar  # noqa: E402
from research import timestral_uncertain_market as shrink  # noqa: E402
from research.timestral_market_calibration import reliability  # noqa: E402
from research import timestral_universal as shared  # noqa: E402
from research.timestral_futures_flow import combine, comparison  # noqa: E402
from scripts import run_market_specific as previous  # noqa: E402
from scripts.run_futures_flow import (
    ROOT,
    digest,
    write,
    epoch,
    sources,
    prior_path,
    examples,
)  # noqa: E402

DEFAULT = ROOT / "reports/timestral-volume-center-20261027"
BASE = previous.BASE
FUTURES = previous.FUTURES
ARMS = ("uniform", "volume")
PAIRS = (
    ("volume", "common"),
    ("volume", "uniform"),
    ("volume", "neutral"),
    ("volume", "flow"),
    ("uniform", "common"),
)
CODE = (
    "research/timestral_volume_center.py",
    "scripts/run_volume_center.py",
    "scripts/verify_volume_center.py",
    "tests/test_timestral_volume_center.py",
    *previous.CODE,
)
PLAN = {
    "purpose": "Test information in the intra-hour price path and its volume concentration where own and common hourly movement agree. Volume must improve both the endpoint-return control and a uniform-price-center ablation.",
    "status": "Sequential retrospective exploration of already explored Jan-Jul2024/2025/2026 BTC/ETH/SOL outcomes. Previous failures motivated hypothesis; no new candidate scores inspected before registration. Not independent/prospective confirmation.",
    "centers": "At each anchor use last60 closed spot minute log closes and base volumes. Uniform center=mean(logclose); volume center=sum(volume*logclose)/sum(volume), uniform fallback if zero volume. These are geometric centers of candle closes, NOT execution VWAP or order book. Subtract source initial logclose for numerical stability. No new downloads.",
    "signals": "Each asset center displacement=-2*(current_logclose-center)/(causal prior60minute RMS*sqrt60), clipped+-8. Mean3asset displacement clipped+-3. Factor2 is fixed to approximately align a uniform linear path with the endpoint hourly displacement. No fitted weights or threshold search. Common/own sign agreement uses frozen hourly return definition. For disagreement/zero use old common signal; for agreement use respective market center signal.",
    "fit": "Each arm has ONE shared nonnegative coefficient [0,8], monthly90day matured nonflat pooled3asset training with equal UTC-day weights and frozen futures availability membership. Use unchanged scalar MLE and six-lag Bartlett uncertainty shrink.42 raw fits and42 adjustments total. No extra coefficients, intercepts, asset IDs or learned regimes. Coefficient is reestimated for each representation, so disagreement probabilities may change too.",
    "scope": "All550362 original evaluation anchors, unchanged five-minute spot target anchor+60 toanchor+360sec, identical frozen flat probabilities. Evaluate all9 cohorts and agreement subset selected solely from past inputs. Disagreement stays in full evaluation, no outcome-based row removal. Retain both original gates and fixed-bin calibration.",
    "primary": "General solution requires all9 volume original gates AND all9 volume_vs_common and volume_vs_uniform conditional gates BOTH on all anchors and agreement subset. Volume_vs_neutral/flow and uniform_vs_common full comparisons are diagnostics. Do not promote uniform after seeing results.",
    "uncertainty": "Unchanged thresholds. Conditional4000 paired7day bootstrap seed271032,99.444% intervals; accuracyLCB>50%, Brier/logloss gainLCB>0 vs named control AND frequency,>=5positive months,>=100each direction. Original52% selection and98.333% intervals unchanged. No joint correction across comparisons/subsets/endpoints/sequential studies; low dimensionality cannot prove absence of overfitting.",
    "protected": "No market source/outcome from2026-08-01 onward. Preserve all older files, frozen artifacts, credentials, holdouts and user work. No trading/profitability claim.",
}


def spot_data(year):
    result = {}
    for s, path in sources(year).items():
        d = dict(np.load(path))
        mask = (d["time"] >= epoch(year - 1, 10)) & (d["time"] < epoch(year, 8))
        result[s] = {k: v[mask] for k, v in d.items()}
    return result


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


def run(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if (
        protocol["plan"] != PLAN
        or digest(root / "protocol.json")
        != json.loads((root / "protocol.sha256.json").read_text())["sha256"]
        or any(
            digest(ROOT / p) != h
            for p, h in {**protocol["code"], **protocol["inputs"]}.items()
        )
    ):
        raise ValueError("frozen experiment changed")
    write(root / "opened.json", dict(at=datetime.now(timezone.utc).isoformat()))
    fits, results = {}, {}
    for year in (2024, 2025, 2026):
        sample = examples(FUTURES, year)
        t, end, y = sample["anchor"], sample["end"], sample["y"]
        z, agree = model.signals(sample["x"], model.centers(spot_data(year), t))
        pieces = {name: [] for name in ARMS}
        indices = []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (t >= cutoff - 90 * 86400) & (end <= cutoff) & sample["available"]
            labels = y[:, train].reshape(-1)
            moving = labels != 1
            yy = (labels[moving] == 2).astype(int)
            tt = np.tile(t[train], 3)[moving]
            ee = np.tile(end[train], 3)[moving]
            ids = np.concatenate(
                [np.column_stack((np.full(train.sum(), a), t[train])) for a in range(3)]
            ).astype("<i8")[moving]
            test = (t >= cutoff) & (t < epoch(year, month + 1))
            for name in ARMS:
                zz = z[name][:, train].reshape(-1)[moving]
                raw = scalar.fit(zz, yy, tt, ee, cutoff, 0)
                adjusted = shrink.adjust(raw, zz, yy, tt, ee, cutoff)
                for state in (raw, adjusted):
                    state["training_ids_sha256"] = hashlib.sha256(
                        ids.tobytes()
                    ).hexdigest()
                    state["training_labels_sha256"] = hashlib.sha256(
                        yy.astype("<i8").tobytes()
                    ).hexdigest()
                fits[f"{year}-{month}-{name}"] = dict(raw=raw, adjusted=adjusted)
                pieces[name].append(
                    np.stack(
                        [
                            shrink.predict(adjusted, z[name][a, test], t[test])
                            for a in range(3)
                        ]
                    )
                )
            indices.append(np.flatnonzero(test))
        indices = np.concatenate(indices)
        predictions = {name: np.concatenate(v, axis=1) for name, v in pieces.items()}
        for a, s in enumerate(shared.SYMBOLS):
            key = f"{year}-{s}"
            prior = dict(np.load(prior_path(year, s)))
            old = dict(np.load(BASE / f"{key}-predictions.npz"))
            flow = dict(np.load(FUTURES / f"{key}-predictions.npz"))
            for d in (prior, old, flow):
                if not np.array_equal(d["anchor"], t[indices]):
                    raise ValueError("anchors changed")
            q = {name: predictions[name][a] for name in ARMS}
            q.update(
                common=old["adaptive_up"],
                flow=flow["flow_up"],
                neutral=np.full(len(indices), 0.5),
            )
            labels, anchors = y[a, indices], t[indices]
            flat = prior["cross" if year == 2026 else "probability"][:, 1]
            frequency = prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1])
            agreement = agree[a, indices]
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(
                    f,
                    anchor=anchors,
                    agreement=agreement,
                    **{name + "_up": q[name] for name in ARMS},
                )
            r = dict(
                comparisons={
                    f"{candidate}_vs_{control}": comparison(
                        labels, q[candidate], q[control], frequency, anchors
                    )
                    for candidate, control in PAIRS
                },
                agreement=dict(
                    n=int(agreement.sum()),
                    comparisons={
                        f"volume_vs_{control}": comparison(
                            labels[agreement],
                            q["volume"][agreement],
                            q[control][agreement],
                            frequency[agreement],
                            anchors[agreement],
                        )
                        for control in ("common", "uniform")
                    },
                ),
                calibration={name: reliability(labels, p) for name, p in q.items()},
                direction_changes={
                    name: int(np.sum((q[name] >= 0.5) != (q["common"] >= 0.5)))
                    for name in ARMS
                },
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
                {p: v["information_gate"] for p, v in r["comparisons"].items()},
                flush=True,
            )
    information = all(
        r["comparisons"][f"volume_vs_{c}"]["information_gate"]
        and r["agreement"]["comparisons"][f"volume_vs_{c}"]["information_gate"]
        for r in results.values()
        for c in ("common", "uniform")
    )
    direction = all(r["volume"]["gate"] for r in results.values())
    write(root / "fits.json", fits)
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
    (register if args.action == "register" else run)(args.root)
