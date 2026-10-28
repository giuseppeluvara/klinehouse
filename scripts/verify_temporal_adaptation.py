#!/usr/bin/env python3
"""Independent temporal-state, local-potential and paired-score audit."""

import argparse
from functools import lru_cache
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
from scripts.verify_futures_flow import SOURCE, MICRO, epoch  # noqa: E402
from scripts.verify_flat_repair import frequency_audit  # noqa: E402

COMMON = ROOT / "reports/timestral-uncertain-market-20261027"
CENTER = ROOT / "reports/timestral-volume-center-20261027"
LOGLOSS = ROOT / "reports/timestral-logloss-aggregate-20261028"
CUMULATIVE = ROOT / "reports/timestral-multiscore-20261028"
METHODS = ("share", "discount", "monthly")
CONTROLS = ("cumulative", "uniform_mixed")
EXPERTS = (
    "common_old",
    "common_mixed",
    "uniform_old",
    "uniform_mixed",
    "equal",
    "logloss_adaptive",
)
BOUNDARY = 1785542400
PERIOD = 30 * 86400
REPLICATES = 12000
QUANTILES = (1 / 1080, 1079 / 1080)


def probabilities(up, flat):
    return np.column_stack(((1 - flat) * (1 - up), flat, (1 - flat) * up))


def outcome_losses(p):
    """Compute both scores against each possible class without solver algebra."""
    return np.stack(
        (-np.log(p), np.square(p[..., None, :] - np.eye(3)).sum(axis=-1)),
        axis=1,
    )


def local_check(saved, method, experts, learner, slack, y, anchor, end):
    """Replay normalized probability weights, independently of production logs."""
    n = len(anchor)
    check(
        method in METHODS
        and experts.shape == (n, 6, 3)
        and learner.shape == (n, 3)
        and slack.shape == y.shape == end.shape == (n,)
        and n >= 2
        and np.all(np.diff(anchor) >= 300)
        and np.all(anchor % 300 == 0)
        and np.all(np.isin(y, (0, 1, 2))),
        "aligned temporal forecast sequence",
    )
    same(end, anchor + 360)
    check(np.all(end[:-2] <= anchor[2:]), "mature same-lane feedback")
    for p in (experts, learner):
        check(np.all(np.isfinite(p) & (p > 0) & (p < 1)), "interior probabilities")
        np.testing.assert_allclose(p.sum(axis=-1), 1, atol=1e-12, rtol=0)
    check(np.all(np.isfinite(slack) & (slack >= 0)), "nonnegative saved slack")

    # Indices are forecast, scoring rule (log then Brier), expert, outcome.
    delta = outcome_losses(learner)[:, :, None, :] - outcome_losses(experts)
    increments = np.take_along_axis(delta, y[:, None, None, None], axis=3)[..., 0]
    months = anchor.astype("datetime64[s]").astype("datetime64[M]")
    states = np.full((2, 2, 6), np.longdouble(1) / 12, dtype=np.longdouble)
    weights = np.empty((n, 2, 6), dtype=np.longdouble)
    updates = resets = 0
    for i in range(n):
        lane = i % 2
        if i >= 2:
            j = i - 2
            if method == "monthly" and months[i] != months[j]:
                # Segment by forecast month: discard the previous month's label.
                states[lane].fill(np.longdouble(1) / 12)
                resets += 1
            else:
                updated = states[lane] * np.exp(increments[j].astype(np.longdouble))
                elapsed = np.longdouble(int(anchor[i] - anchor[j])) / PERIOD
                if method == "share":
                    retain = np.exp(-elapsed)
                    updated = (
                        retain * updated
                        + (1 - retain) * updated.sum(axis=1, keepdims=True) / 6
                    )
                elif method == "discount":
                    updated = np.power(updated, np.power(np.longdouble(0.5), elapsed))
                states[lane] = updated / updated.sum()
                updates += 1
        weights[i] = states[lane]
    check(np.all(np.isfinite(weights) & (weights > 0)), "finite causal weights")
    np.testing.assert_allclose(weights.sum(axis=(1, 2)), 1, atol=1e-15, rtol=0)
    cold = np.ones(n, dtype=bool)
    cold[2:] = (method == "monthly") & (months[2:] != months[:-2])
    np.testing.assert_allclose(weights[cold], 1 / 12, atol=1e-15, rtol=0)

    # All three outcomes, including outcomes that did not occur, must be safe.
    ratios = (weights[..., None] * np.exp(delta.astype(np.longdouble))).sum(axis=(1, 2))
    check(np.all(np.isfinite(ratios)), "finite outcome-wise potential")
    largest = np.asarray(ratios.max(axis=1), dtype=float)
    maximum = float(largest.max())
    check(maximum <= 1 + 1e-10, "local potential valid for every possible outcome")
    actual_slack = np.maximum(0, np.log(largest))
    np.testing.assert_allclose(slack, actual_slack, atol=5e-12, rtol=1e-7)
    same(slack.sum(), actual_slack.sum())
    check(
        set(saved)
        == {"maximum_potential_ratio", "numeric_slack", "local_valid", "diagnostics"},
        "local evidence schema",
    )
    same(saved["maximum_potential_ratio"], maximum)
    same(saved["numeric_slack"], actual_slack.sum())
    check(saved["local_valid"] is True, "local certificate flag")
    diagnostics = saved["diagnostics"]
    check(
        diagnostics["method"] == method
        and diagnostics["updates"] == updates
        and diagnostics["resets"] == resets
        and diagnostics["skipped_feedback"] == resets
        and updates + resets == n - 2,
        "feedback, skipped old-month labels and reset counts",
    )
    same(diagnostics["maximum_potential_ratio"], maximum)
    same(
        diagnostics["per_lane_slack"],
        [actual_slack[lane::2].sum() for lane in (0, 1)],
    )
    return dict(
        maximum_potential_ratio=maximum,
        numeric_slack=float(actual_slack.sum()),
        updates=updates,
        resets=resets,
    )


@lru_cache(maxsize=4)
def sampled_day_counts(number):
    """Independent count-matrix implementation of seven-day block bootstrap."""
    check(number >= 7, "at least one complete bootstrap block")
    starts = np.random.default_rng(271032).integers(
        0, number - 6, (REPLICATES, (number + 6) // 7)
    )
    days = (starts[:, :, None] + np.arange(7)).reshape(REPLICATES, -1)[:, :number]
    counts = np.zeros((REPLICATES, number), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(REPLICATES), number), days.ravel()), 1)
    check(np.all(counts.sum(axis=1) == number), "full bootstrap day counts")
    return counts


def paired_check(saved, y, own, control, t, mask, directional=False):
    """Recompute score means, paired intervals and every monthly criterion."""
    check(mask.shape == y.shape == t.shape and np.any(mask), "nonempty paired scope")
    if directional:
        check(np.array_equal(mask, y != 1), "moving outcomes only")
        names = ("conditional_brier", "conditional_logloss", "accuracy")
        target = y == 2

        def scores(p):
            q = p[:, 2] / (1 - p[:, 1])
            return np.column_stack(
                (
                    np.square(q - target),
                    -np.log(np.where(target, q, 1 - q)),
                    (q >= 0.5) == target,
                )
            )

        signs = np.array((1, 1, -1))
    else:
        names = ("brier", "logloss")

        def scores(p):
            return np.column_stack(
                (
                    np.square(p - np.eye(3)[y]).sum(axis=1),
                    -np.log(p[np.arange(len(y)), y]),
                )
            )

        signs = np.ones(2)
    own_scores, control_scores = scores(own), scores(control)
    gains = (control_scores - own_scores) * signs
    days = np.arange(t.min() // 86400, t.max() // 86400 + 1)
    counts = np.array([np.sum(mask & (t // 86400 == d)) for d in days])
    totals = np.array([gains[mask & (t // 86400 == d)].sum(axis=0) for d in days])
    weights = sampled_day_counts(len(days))
    denominator = weights @ counts
    check(np.all(denominator > 0), "nonempty bootstrap samples")
    boot = (weights @ totals) / denominator[:, None]
    bounds = np.quantile(boot, QUANTILES, axis=0)
    months = t.astype("datetime64[s]").astype("datetime64[M]")
    monthly = {}
    for month in np.unique(months):
        selected = mask & (months == month)
        check(np.any(selected), "every month represented")
        monthly[str(month)] = gains[selected].mean(axis=0)
    positive = (np.array(list(monthly.values())) > 0).sum(axis=0)
    check(
        set(saved)
        == {
            "n",
            "own",
            "control",
            "gain",
            "interval",
            "monthly",
            "positive_months",
            "improvement_gate",
        },
        "complete paired metric schema",
    )
    check(saved["n"] == int(mask.sum()), "paired case count")
    for field in ("own", "control", "gain", "interval", "positive_months"):
        check(set(saved[field]) == set(names), "all registered paired metrics")
    check(set(saved["monthly"]) == set(monthly), "all seven paired months")
    for month in monthly:
        check(set(saved["monthly"][month]) == set(names), "all monthly metrics")
    for j, name in enumerate(names):
        same(saved["own"][name], own_scores[mask, j].mean())
        same(saved["control"][name], control_scores[mask, j].mean())
        same(saved["gain"][name], gains[mask, j].mean())
        same(saved["interval"][name], bounds[:, j])
        check(saved["positive_months"][name] == int(positive[j]), "positive months")
        for month, values in monthly.items():
            same(saved["monthly"][month][name], values[j])
    expected = bool(np.all(bounds[0] > 0) and np.all(positive >= 5))
    check(saved["improvement_gate"] is expected, "all-metric paired improvement gate")


def verify(root):
    protocol = json.loads((root / "protocol.json").read_text())
    check(
        digest(root / "protocol.json")
        == json.loads((root / "protocol.sha256.json").read_text())["sha256"],
        "protocol hash",
    )
    check(
        {
            "scripts/verify_temporal_adaptation.py",
            "scripts/run_temporal_adaptation.py",
            "research/timestral_temporal_adaptation.py",
            "research/timestral_multiscore.py",
        }
        <= set(protocol["code"]),
        "audit, runner and frozen solver registered before scores",
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

    def frozen_input(path):
        check(str(path.relative_to(ROOT)) in protocol["inputs"], "registered input")
        return dict(np.load(path))

    results = json.loads((root / "results.json").read_text())
    evidence = json.loads((root / "evidence.json").read_text())
    keys = {f"{year}-{s}" for year in (2024, 2025, 2026) for s in ASSETS}
    check(set(results["cohorts"]) == set(evidence) == keys, "all nine cohorts")
    cases = 0
    largest_ratio = 0.0
    largest_slack = 0.0
    resets = {method: 0 for method in METHODS}
    for year in (2024, 2025, 2026):
        for s in ASSETS:
            key = f"{year}-{s}"
            folder, prefix = (SOURCE, s) if year == 2026 else (SOURCE / "history", key)
            prior = frozen_input(folder / f"{prefix}-predictions.npz")
            outcome = frozen_input(folder / f"{prefix}-outcomes.npz")
            common = frozen_input(COMMON / f"{key}-predictions.npz")
            center = frozen_input(CENTER / f"{key}-predictions.npz")
            logloss = frozen_input(LOGLOSS / f"{key}-predictions.npz")
            cumulative = frozen_input(CUMULATIVE / f"{key}-predictions.npz")
            saved = dict(np.load(root / f"{key}-predictions.npz"))
            check(
                set(saved)
                == {
                    "anchor",
                    *METHODS,
                    *(method + "_log_excess" for method in METHODS),
                },
                "saved temporal prediction schema",
            )
            for d in (outcome, common, center, logloss, cumulative, saved):
                same(prior["anchor"], d["anchor"])
            t, y = prior["anchor"], outcome["label"]
            check(
                t.min() >= epoch(year, 1)
                and t.max() < epoch(year, 8)
                and outcome["end"].max() <= BOUNDARY,
                "admitted period and protected boundary",
            )
            source_path = (
                SOURCE / "history" / f"{s}-minutes.npz"
                if year < 2026
                else MICRO
                / (
                    "micro/minutes.npz"
                    if s == "BTCUSDT"
                    else f"transfer/{s}-minutes.npz"
                )
            )
            check(
                str(source_path.relative_to(ROOT)) in protocol["inputs"],
                "frequency source frozen",
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
            controls = {
                "cumulative": cumulative["defensive"],
                "uniform_mixed": p["uniform_mixed"],
            }
            for control in controls.values():
                check(control.shape == (len(y), 3), "aligned frozen control")
                check(
                    np.all(np.isfinite(control) & (control > 0) & (control < 1)),
                    "finite frozen control",
                )
                np.testing.assert_allclose(control.sum(1), 1, atol=1e-12, rtol=0)
            selected = (
                np.maximum(p["uniform_mixed"][:, 0], p["uniform_mixed"][:, 2]) >= 0.52
            )
            cohort = results["cohorts"][key]
            check(set(cohort) == set(evidence[key]) == set(METHODS), "all three arms")
            for method in METHODS:
                prediction = saved[method]
                local = local_check(
                    evidence[key][method],
                    method,
                    experts,
                    prediction,
                    saved[method + "_log_excess"],
                    y,
                    t,
                    outcome["end"],
                )
                largest_ratio = max(largest_ratio, local["maximum_potential_ratio"])
                largest_slack = max(largest_slack, local["numeric_slack"])
                resets[method] += local["resets"]
                r = cohort[method]
                check(set(r) == {"original", "paired", "direction"}, "arm metrics")
                verify_metrics(
                    r["original"],
                    y,
                    prediction,
                    prior["frequency"],
                    prior["prior_reversal"],
                    t,
                )
                check(set(r["paired"]) == {"all", "frozen_selected"}, "both scopes")
                for scope, mask in (
                    ("all", np.ones(len(y), dtype=bool)),
                    ("frozen_selected", selected),
                ):
                    check(
                        set(r["paired"][scope]) == set(CONTROLS), "both paired controls"
                    )
                    for name in CONTROLS:
                        paired_check(
                            r["paired"][scope][name],
                            y,
                            prediction,
                            controls[name],
                            t,
                            mask,
                        )
                check(set(r["direction"]) == set(CONTROLS), "both directional controls")
                for name in CONTROLS:
                    paired_check(
                        r["direction"][name],
                        y,
                        prediction,
                        controls[name],
                        t,
                        y != 1,
                        directional=True,
                    )
            cases += len(y)
    check(cases == 550362, "all original anchors retained")
    check(set(results["summary"]) == set(METHODS), "all prespecified summaries")
    summary = {}
    for method in METHODS:
        arms = [r[method] for r in results["cohorts"].values()]
        original = sum(r["original"]["gate"] for r in arms)
        paired = sum(
            r["paired"][scope][name]["improvement_gate"]
            for r in arms
            for scope in ("all", "frozen_selected")
            for name in CONTROLS
        )
        direction = sum(
            r["direction"][name]["improvement_gate"] for r in arms for name in CONTROLS
        )
        summary[method] = dict(
            original_passes=original,
            paired_passes=paired,
            direction_passes=direction,
            stable_improvement=(original == 9 and paired == 36 and direction == 18),
        )
        check(
            results["summary"][method] == summary[method], "strict all-cohort success"
        )
    check(
        results["primary_method"] == "share"
        and results["primary_stable_improvement"]
        is summary["share"]["stable_improvement"],
        "prespecified primary arm",
    )
    return dict(
        verified=True,
        cases=cases,
        cohorts=9,
        arm_forecasts=cases * len(METHODS),
        past_frequency_windows=63,
        original_evaluations=27,
        paired_comparisons=108,
        direction_comparisons=54,
        bootstrap_replicates=REPLICATES,
        maximum_potential_ratio=largest_ratio,
        maximum_numeric_slack=largest_slack,
        monthly_resets=resets,
        all_local_potentials_valid=True,
        full_cohort_regret_bound_claimed=False,
        protected_data_opened=False,
        summary=summary,
        primary_stable_improvement=summary["share"]["stable_improvement"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "reports/timestral-temporal-adaptation-20261028",
    )
    print(json.dumps(verify(parser.parse_args().root), sort_keys=True))
