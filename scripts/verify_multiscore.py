#!/usr/bin/env python3
"""Independent causal-potential and score audit for two simultaneous losses."""

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
LOGLOSS = ROOT / "reports/timestral-logloss-aggregate-20261028"
EXPERTS = (
    "common_old",
    "common_mixed",
    "uniform_old",
    "uniform_mixed",
    "equal",
    "logloss_adaptive",
)
SCORES = ("logloss", "brier")
BOUNDARY = 1785542400


def probabilities(up, flat):
    return np.column_stack(((1 - flat) * (1 - up), flat, (1 - flat) * up))


def outcome_losses(p):
    """Evaluate every possible outcome, independently of the optimizer formula."""
    one = np.eye(3)
    return np.stack(
        (
            -np.log(p),
            np.square(p[..., None, :] - one).sum(axis=-1),
        ),
        axis=1,
    )


def certificate_check(saved, diagnostics, experts, learner, slack, y, anchor, end):
    n = len(anchor)
    check(
        experts.shape == (n, 6, 3)
        and learner.shape == (n, 3)
        and slack.shape == y.shape == end.shape == (n,)
        and n >= 2
        and np.all(np.diff(anchor) >= 300)
        and np.all(anchor % 300 == 0)
        and np.all(np.isin(y, (0, 1, 2))),
        "aligned forecast sequence",
    )
    same(end, anchor + 360)
    check(np.all(end[:-2] <= anchor[2:]), "strictly mature same-lane labels")
    for p in (experts, learner):
        check(np.all(np.isfinite(p) & (p > 0) & (p < 1)), "interior probabilities")
        np.testing.assert_allclose(p.sum(axis=-1), 1, atol=1e-12, rtol=0)
    check(
        np.all(np.isfinite(slack) & (slack >= 0)), "finite nonnegative numerical budget"
    )

    # Shapes: forecast, scoring rule, expert, possible outcome.
    expert_losses = outcome_losses(experts)
    learner_losses = outcome_losses(learner)
    delta = learner_losses[:, :, None, :] - expert_losses
    increments = np.take_along_axis(delta, y[:, None, None, None], axis=3)[..., 0]
    history = np.zeros_like(increments)
    for lane in (0, 1):
        ids = np.arange(lane, n, 2)
        # Prefix excludes current, future and every opposite-lane label.
        history[ids[1:]] = np.cumsum(increments[ids[:-1]], axis=0, dtype=np.longdouble)
    raw_weights = np.exp(history - history.max(axis=(1, 2), keepdims=True))
    weights = raw_weights / raw_weights.sum(axis=(1, 2), keepdims=True)
    same(weights[:2], np.full((2, 2, 6), 1 / 12))
    # Direct twelve-term calculation; no production A/B reductions or solver.
    ratios = (weights[..., None] * np.exp(delta)).sum(axis=(1, 2))
    check(np.all(np.isfinite(ratios)), "finite outcome-wise potential")
    largest = ratios.max(axis=1)
    check(float(largest.max()) <= 1 + 1e-10, "defensive condition for every outcome")
    actual_slack = np.maximum(0, np.log(largest))
    np.testing.assert_allclose(slack, actual_slack, atol=5e-12, rtol=1e-7)
    same(slack.sum(), actual_slack.sum())

    expert_realized = np.take_along_axis(expert_losses, y[:, None, None, None], axis=3)[
        ..., 0
    ]
    learner_realized = np.take_along_axis(learner_losses, y[:, None, None], axis=2)[
        ..., 0
    ]
    totals = expert_realized.sum(axis=0)
    own_totals = learner_realized.sum(axis=0)
    bound = float(2 * np.log(12))
    slack_total = float(actual_slack.sum())
    cumulative = np.cumsum(increments, axis=0, dtype=np.longdouble)
    allowances = bound + np.cumsum(actual_slack, dtype=np.longdouble)
    maximum_excess = float(np.max(cumulative - allowances[:, None, None]))
    expected = dict(
        expert_totals={name: totals[j].tolist() for j, name in enumerate(SCORES)},
        learner_totals={name: float(own_totals[j]) for j, name in enumerate(SCORES)},
        regrets={
            name: float(own_totals[j] - totals[j].min())
            for j, name in enumerate(SCORES)
        },
        base_bound=bound,
        numeric_slack=slack_total,
        effective_bound=bound + slack_total,
        maximum_prefix_excess=maximum_excess,
        bound_pass=bool(maximum_excess <= 1e-8),
    )
    check(set(saved) == set(expected), "complete two-score certificate")
    for name in ("expert_totals", "learner_totals", "regrets"):
        check(set(saved[name]) == set(SCORES), "both proper scoring rules")
        for score in SCORES:
            same(saved[name][score], expected[name][score])
    for name in (
        "base_bound",
        "numeric_slack",
        "effective_bound",
        "maximum_prefix_excess",
    ):
        same(saved[name], expected[name])
    check(
        saved["bound_pass"] is expected["bound_pass"] and expected["bound_pass"],
        "all-prefix regret bounds",
    )
    check(
        max(expected["regrets"].values()) <= expected["effective_bound"] + 1e-8,
        "both final score bounds",
    )
    if "maximum_potential_ratio" in diagnostics:
        same(diagnostics["maximum_potential_ratio"], largest.max())
    if "per_lane_slack" in diagnostics:
        same(
            diagnostics["per_lane_slack"],
            [actual_slack[lane::2].sum() for lane in (0, 1)],
        )
    return expected, float(largest.max())


def verify(root):
    protocol = json.loads((root / "protocol.json").read_text())
    check(
        digest(root / "protocol.json")
        == json.loads((root / "protocol.sha256.json").read_text())["sha256"],
        "protocol hash",
    )
    check(
        "scripts/verify_multiscore.py" in protocol["code"], "audit frozen before scores"
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
        "registration chronology",
    )
    results = json.loads((root / "results.json").read_text())
    evidence = json.loads((root / "evidence.json").read_text())
    keys = {f"{year}-{s}" for year in (2024, 2025, 2026) for s in ASSETS}
    check(set(results["cohorts"]) == set(evidence) == keys, "all nine cohorts")
    cases = 0
    largest_ratio = 0.0
    largest_slack = 0.0
    largest_excess = -np.inf
    certificates = []
    for year in (2024, 2025, 2026):
        for s in ASSETS:
            key = f"{year}-{s}"
            folder, prefix = (SOURCE, s) if year == 2026 else (SOURCE / "history", key)
            prior = dict(np.load(folder / f"{prefix}-predictions.npz"))
            outcome = dict(np.load(folder / f"{prefix}-outcomes.npz"))
            common = dict(np.load(COMMON / f"{key}-predictions.npz"))
            center = dict(np.load(CENTER / f"{key}-predictions.npz"))
            logloss = dict(np.load(LOGLOSS / f"{key}-predictions.npz"))
            saved = dict(np.load(root / f"{key}-predictions.npz"))
            check(
                set(saved) == {"anchor", "defensive", "log_excess"},
                "saved output schema",
            )
            for d in (outcome, common, center, logloss, saved):
                same(prior["anchor"], d["anchor"])
            t, y = prior["anchor"], outcome["label"]
            check(
                t.min() >= epoch(year, 1)
                and t.max() < epoch(year, 8)
                and outcome["end"].max() <= BOUNDARY,
                "admitted period and protected boundary",
            )
            frequency_audit(year, s, prior, outcome)
            q = (common["adaptive_up"], center["uniform_up"])
            old = prior["cross" if year == 2026 else "probability"][:, 1]
            f = (old, (old + prior["frequency"][:, 1]) / 2)
            p = {
                name: probabilities(q[j // 2], f[j % 2])
                for j, name in enumerate(EXPERTS[:4])
            }
            p["equal"] = np.mean(list(p.values()), axis=0)
            p["logloss_adaptive"] = logloss["adaptive"]
            experts = np.stack([p[name] for name in EXPERTS], axis=1)
            defensive = saved["defensive"]
            cert, ratio = certificate_check(
                evidence[key]["certificate"],
                evidence[key]["solver"],
                experts,
                defensive,
                saved["log_excess"],
                y,
                t,
                outcome["end"],
            )
            certificates.append(cert["bound_pass"])
            largest_ratio = max(largest_ratio, ratio)
            largest_slack = max(largest_slack, cert["numeric_slack"])
            largest_excess = max(largest_excess, cert["maximum_prefix_excess"])
            p["defensive"] = defensive
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
                check(
                    set(r["paired"][scope]) == set(EXPERTS), "all six paired controls"
                )
                for name in EXPERTS:
                    paired_check(
                        r["paired"][scope][name], y, defensive, p[name], t, mask
                    )
            own_q = defensive[:, 2] / (1 - defensive[:, 1])
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
    check(cases == 550362, "all original anchors")
    original = all(r["defensive"]["gate"] for r in results["cohorts"].values())
    paired = all(
        r["paired"][scope][name]["improvement_gate"]
        for r in results["cohorts"].values()
        for scope in ("all", "frozen_selected")
        for name in EXPERTS
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
        and results["all_certificates"] == all(certificates)
        and results["general_solution"] == (original and paired and direction),
        "registered aggregate criteria and separate certificates",
    )
    return dict(
        verified=True,
        cases=cases,
        cohorts=9,
        past_frequency_windows=63,
        lanes=18,
        original_evaluations=63,
        paired_comparisons=108,
        direction_comparisons=18,
        maximum_potential_ratio=largest_ratio,
        maximum_numeric_slack=largest_slack,
        maximum_prefix_bound_excess=largest_excess,
        all_certificates=all(certificates),
        protected_data_opened=False,
        all_original_gates=original,
        all_paired_gates=paired,
        all_direction_information_gates=direction,
        general_solution=results["general_solution"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "reports/timestral-multiscore-20261028"
    )
    print(json.dumps(verify(parser.parse_args().root), sort_keys=True))
