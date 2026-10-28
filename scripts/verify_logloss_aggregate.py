#!/usr/bin/env python3
"""Independently audit delayed aggregation through factorized evidence odds."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_universal_direction import (  # noqa: E402
    ROOT,
    ASSETS,
    check,
    same,
    digest,
    verify_metrics,
)
from scripts.verify_futures_flow import SOURCE, epoch  # noqa: E402
from scripts.verify_flat_repair import frequency_audit, paired_check  # noqa: E402
from scripts.verify_premium_direction import conditional  # noqa: E402

COMMON = ROOT / "reports/timestral-uncertain-market-20261027"
CENTER = ROOT / "reports/timestral-volume-center-20261027"
EXPERTS = ("common_old", "common_mixed", "uniform_old", "uniform_mixed")
CONTROLS = (*EXPERTS, "equal")
BOUNDARY = 1785542400


def components(q, f):
    return np.column_stack(((1 - f) * (1 - q), f, (1 - f) * q))


def component_losses(y, q, f):
    moving = y != 1
    direction = np.zeros(len(y))
    direction[moving] = -np.log(np.where(y[moving] == 2, q[moving], 1 - q[moving]))
    flat = -np.log(np.where(moving, 1 - f, f))
    return direction, flat


def factorized_replay(t, end, y, common, uniform, old, mixed):
    """Rebuild odds independently, with two binary experts per component/lane."""
    n = len(t)
    check(
        n >= 2
        and all(np.shape(a) == (n,) for a in (end, y, common, uniform, old, mixed))
        and np.all(np.diff(t) >= 300),
        "aligned five-minute-or-longer forecast spacing",
    )
    same(end, t + 360)
    check(np.all(end[:-2] <= t[2:]), "same-lane outcome must already be mature")
    check(np.all(np.isin(y, (0, 1, 2))), "valid three-class outcomes")
    for p in (common, uniform, old, mixed):
        check(
            np.all(np.isfinite(p) & (p > 0) & (p < 1)), "interior expert probabilities"
        )
    common_loss, old_loss = component_losses(y, common, old)
    uniform_loss, mixed_loss = component_losses(y, uniform, mixed)
    direction_weight = np.zeros(n)
    flat_weight = np.zeros(n)
    for lane in (0, 1):
        positions = np.arange(lane, n, 2)
        for delta, weights in (
            (common_loss - uniform_loss, direction_weight),
            (old_loss - mixed_loss, flat_weight),
        ):
            # Strict prefix: current and opposite-lane outcomes never update odds.
            odds = np.zeros(len(positions))
            odds[1:] = np.cumsum(delta[positions[:-1]], dtype=np.longdouble)
            weights[positions] = np.exp(-np.logaddexp(0.0, -odds))
    joint = np.column_stack(
        (
            (1 - direction_weight) * (1 - flat_weight),
            (1 - direction_weight) * flat_weight,
            direction_weight * (1 - flat_weight),
            direction_weight * flat_weight,
        )
    )
    q = (1 - direction_weight) * common + direction_weight * uniform
    f = (1 - flat_weight) * old + flat_weight * mixed
    return components(q, f), joint


def evidence_check(saved, y, adaptive, expert, q, f):
    n = len(y)
    idx = np.arange(n)
    losses = np.column_stack([-np.log(expert[name][idx, y]) for name in EXPERTS])
    total = losses.sum(axis=0)
    own_total = float(-np.log(adaptive[idx, y]).sum())
    normalizer = 0.0
    for lane in (0, 1):
        lane_totals = np.sum(losses[lane::2], axis=0)
        normalizer -= float(np.logaddexp.reduce(-lane_totals - np.log(4)))
    own_q = adaptive[:, 2] / (1 - adaptive[:, 1])
    own_direction, own_flat = component_losses(y, own_q, adaptive[:, 1])
    directional_totals = [component_losses(y, p, f[0])[0].sum() for p in q]
    flat_totals = [component_losses(y, q[0], p)[1].sum() for p in f]
    expected = dict(
        expert_logloss=total.tolist(),
        adaptive_logloss=own_total,
        best_expert_logloss=float(total.min()),
        regret=float(own_total - total.min()),
        regret_bound=float(2 * np.log(4)),
        direction_regret=float(own_direction.sum() - min(directional_totals)),
        flat_regret=float(own_flat.sum() - min(flat_totals)),
        component_bound=float(2 * np.log(2)),
        telescoping_gap=float(own_total - normalizer),
    )
    check(set(saved) == set(expected), "complete mathematical evidence")
    for name, value in expected.items():
        same(saved[name], value)
    check(abs(expected["telescoping_gap"]) <= 1e-8, "log-evidence identity")
    check(expected["regret"] <= expected["regret_bound"] + 1e-8, "joint regret bound")
    check(
        max(expected["direction_regret"], expected["flat_regret"])
        <= expected["component_bound"] + 1e-8,
        "separate direction and flat regret bounds",
    )
    same(own_direction.sum() + own_flat.sum(), own_total)
    return expected


def verify(root):
    protocol = json.loads((root / "protocol.json").read_text())
    check(
        digest(root / "protocol.json")
        == json.loads((root / "protocol.sha256.json").read_text())["sha256"],
        "protocol hash",
    )
    check(
        "scripts/verify_logloss_aggregate.py" in protocol["code"],
        "independent auditor frozen before scores",
    )
    manifest = json.loads((root / "manifest.json").read_text())
    required = {
        "protocol.json",
        "protocol.sha256.json",
        "opened.json",
        "results.json",
        "evidence.json",
    }
    required.update(
        f"{year}-{s}-predictions.npz" for year in (2024, 2025, 2026) for s in ASSETS
    )
    check(required <= set(manifest), "complete execution manifest")
    for name, h in manifest.items():
        check(digest(root / name) == h, f"artifact hash {name}")
    for name, h in {**protocol["code"], **protocol["inputs"]}.items():
        check(digest(ROOT / name) == h, f"code/input hash {name}")
    check(
        protocol["at"] < json.loads((root / "opened.json").read_text())["at"],
        "registration before evaluation",
    )
    results = json.loads((root / "results.json").read_text())
    evidence = json.loads((root / "evidence.json").read_text())
    keys = {f"{year}-{s}" for year in (2024, 2025, 2026) for s in ASSETS}
    check(set(results["cohorts"]) == set(evidence) == keys, "all nine cohorts")
    cases = 0
    max_gap = 0.0
    max_excess = -np.inf
    for year in (2024, 2025, 2026):
        for s in ASSETS:
            key = f"{year}-{s}"
            folder, prefix = (SOURCE, s) if year == 2026 else (SOURCE / "history", key)
            prior = dict(np.load(folder / f"{prefix}-predictions.npz"))
            outcome = dict(np.load(folder / f"{prefix}-outcomes.npz"))
            common = dict(np.load(COMMON / f"{key}-predictions.npz"))
            center = dict(np.load(CENTER / f"{key}-predictions.npz"))
            saved = dict(np.load(root / f"{key}-predictions.npz"))
            check(
                set(saved) == {"anchor", "adaptive", "weights"},
                "saved probability schema",
            )
            for d in (outcome, common, center, saved):
                same(prior["anchor"], d["anchor"])
            t, y = prior["anchor"], outcome["label"]
            check(
                t.min() >= epoch(year, 1)
                and t.max() < epoch(year, 8)
                and outcome["end"].max() <= BOUNDARY,
                "admitted anchor period and protected outcomes",
            )
            frequency_audit(year, s, prior, outcome)
            q = (common["adaptive_up"], center["uniform_up"])
            old = prior["cross" if year == 2026 else "probability"][:, 1]
            f = (old, (old + prior["frequency"][:, 1]) / 2)
            expert = {
                name: components(q[j // 2], f[j % 2]) for j, name in enumerate(EXPERTS)
            }
            adaptive, weights = factorized_replay(t, outcome["end"], y, *q, *f)
            same(saved["adaptive"], adaptive)
            same(saved["weights"], weights)
            same(weights.sum(axis=1), np.ones(len(t)))
            same(weights[:2], np.full((2, 4), 0.25))
            same(
                adaptive,
                sum(
                    weights[:, j, None] * expert[name] for j, name in enumerate(EXPERTS)
                ),
            )
            info = evidence_check(evidence[key], y, adaptive, expert, q, f)
            max_gap = max(max_gap, abs(info["telescoping_gap"]))
            max_excess = max(max_excess, info["regret"] - info["regret_bound"])
            p = dict(
                expert, adaptive=adaptive, equal=np.mean(list(expert.values()), axis=0)
            )
            r = results["cohorts"][key]
            for name, probability in p.items():
                verify_metrics(
                    r[name],
                    y,
                    probability,
                    prior["frequency"],
                    prior["prior_reversal"],
                    t,
                )
            for scope, mask in (
                ("all", np.ones(len(y), dtype=bool)),
                (
                    "frozen_selected",
                    np.maximum(p["uniform_mixed"][:, 0], p["uniform_mixed"][:, 2])
                    >= 0.52,
                ),
            ):
                check(set(r["paired"][scope]) == set(CONTROLS), "all paired controls")
                for name in CONTROLS:
                    paired_check(
                        r["paired"][scope][name], y, adaptive, p[name], t, mask
                    )
            own_q = adaptive[:, 2] / (1 - adaptive[:, 1])
            check(
                set(r["conditional"]) == {"common", "uniform"},
                "both direction controls",
            )
            for name, comparator in zip(("common", "uniform"), q):
                conditional(
                    {
                        k.replace("vs_control", "vs_spot"): v
                        for k, v in r["conditional"][name].items()
                    },
                    y,
                    own_q,
                    comparator,
                    prior["frequency"][:, 2] / (1 - prior["frequency"][:, 1]),
                    t,
                )
            cases += len(y)
    check(cases == 550362, "all original forecast anchors")
    original = all(r["adaptive"]["gate"] for r in results["cohorts"].values())
    paired = all(
        r["paired"][scope][name]["improvement_gate"]
        for r in results["cohorts"].values()
        for scope in ("all", "frozen_selected")
        for name in CONTROLS
    )
    direction = all(
        r["conditional"][name]["information_gate"]
        for r in results["cohorts"].values()
        for name in ("common", "uniform")
    )
    check(
        results["all_original_gates"] == original
        and results["all_paired_gates"] == paired
        and results["all_direction_information_gates"] == direction
        and results["general_solution"] == (original and paired and direction),
        "registered aggregate criteria",
    )
    return dict(
        verified=True,
        cases=cases,
        cohorts=9,
        past_frequency_windows=63,
        lanes=18,
        paired_comparisons=90,
        direction_comparisons=18,
        maximum_absolute_telescoping_gap=max_gap,
        maximum_regret_bound_excess=max_excess,
        protected_data_opened=False,
        all_original_gates=original,
        all_paired_gates=paired,
        all_direction_information_gates=direction,
        general_solution=results["general_solution"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "reports/timestral-logloss-aggregate-20261028",
    )
    print(json.dumps(verify(parser.parse_args().root), sort_keys=True))
