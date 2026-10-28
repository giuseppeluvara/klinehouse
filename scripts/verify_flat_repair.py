#!/usr/bin/env python3
"""Independent past-frequency, fixed-mixture and paired-score reconstruction."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_universal_direction import (
    ROOT,
    ASSETS,
    check,
    same,
    digest,
    verify_metrics,
)  # noqa: E402
from scripts.verify_futures_flow import epoch, SOURCE, MICRO  # noqa: E402

CENTER = ROOT / "reports/timestral-volume-center-20261027"


def frequency_audit(year, symbol, prior, outcome):
    path = (
        SOURCE / "history" / f"{symbol}-minutes.npz"
        if year < 2026
        else MICRO
        / (
            "micro/minutes.npz"
            if symbol == "BTCUSDT"
            else f"transfer/{symbol}-minutes.npz"
        )
    )
    source = dict(np.load(path))
    check(source["time"].max() < 1785542400, "protected source")
    use = (source["time"] >= epoch(year - 1, 10)) & (
        source["time"] < epoch(year, 8) + 360
    )
    t = source["time"][use]
    price = source["close"][use]
    check(
        np.all(np.diff(t) == 60) and np.all(np.isfinite(price)) and np.all(price > 0),
        "complete finite source",
    )
    i = np.flatnonzero(((t + 60) // 60) % 5 == 0)
    i = i[(i >= 61) & (i + 6 < len(t))]
    anchor = t[i] + 60
    end = t[i + 6] + 60
    labels = (
        (price[i + 6] > price[i + 1]).astype(int)
        - (price[i + 6] < price[i + 1]).astype(int)
        + 1
    )
    total = []
    for month in range(1, 8):
        cutoff = epoch(year, month)
        train = (anchor >= cutoff - 90 * 86400) & (end <= cutoff)
        test = (anchor >= cutoff) & (anchor < epoch(year, month + 1))
        counts = np.array([np.sum(labels[train] == k) for k in range(3)])
        expected = (counts + 1) / (train.sum() + 3)
        saved = (prior["anchor"] >= cutoff) & (prior["anchor"] < epoch(year, month + 1))
        same(prior["anchor"][saved], anchor[test])
        same(prior["frequency"][saved], np.broadcast_to(expected, (saved.sum(), 3)))
        total.extend(np.flatnonzero(test))
    total = np.array(total)
    same(outcome["anchor"], anchor[total])
    same(outcome["end"], end[total])
    same(outcome["label"], labels[total])
    check(end[total].max() <= 1785542400, "protected target")


def losses(y, p):
    one = np.eye(3)[y]
    flat = y == 1
    return np.column_stack(
        (
            np.sum((one - p) ** 2, axis=1),
            -np.log(np.take_along_axis(p, y[:, None], axis=1)[:, 0]),
            (flat - p[:, 1]) ** 2,
            -np.log(np.where(flat, p[:, 1], 1 - p[:, 1])),
        )
    )


def paired_check(saved, y, p, old, t, mask):
    delta = losses(y, old) - losses(y, p)
    days = np.arange(t.min() // 86400, t.max() // 86400 + 1)
    counts = np.array([np.sum(mask & (t // 86400 == d)) for d in days])
    sums = np.array([delta[mask & (t // 86400 == d)].sum(axis=0) for d in days])
    starts = np.random.default_rng(271032).integers(
        0, len(days) - 6, (4000, (len(days) + 6) // 7)
    )
    # Aggregate via counts of sampled days, independently of production indexing.
    ids = ((starts[:, :, None] + np.arange(7)).reshape(4000, -1))[:, : len(days)]
    weights = np.zeros((4000, len(days)), dtype=int)
    np.add.at(weights, (np.repeat(np.arange(4000), len(days)), ids.ravel()), 1)
    boot = (weights @ sums) / (weights @ counts)[:, None]
    bounds = np.quantile(boot, [1 / 360, 359 / 360], axis=0)
    names = ("brier_gain", "logloss_gain", "flat_brier_gain", "flat_logloss_gain")
    for j, name in enumerate(names):
        same(saved[name], delta[mask, j].mean())
        same(saved[name + "_ci99_444"], bounds[:, j])
    months = t.astype("datetime64[s]").astype("datetime64[M]")
    monthly = {
        str(m): float(delta[mask & (months == m), 0].mean()) for m in np.unique(months)
    }
    check(set(saved["monthly_brier_gain"]) == set(monthly), "all monthly comparisons")
    for m, v in monthly.items():
        same(saved["monthly_brier_gain"][m], v)
    check(
        saved["n"] == int(mask.sum())
        and saved["flats"] == int(np.sum(mask & (y == 1))),
        "paired membership",
    )
    check(
        saved["improvement_gate"]
        == bool(
            np.all(bounds[0, :2] > 0) and sum(v > 0 for v in monthly.values()) >= 5
        ),
        "paired gate",
    )


def decomposition_check(saved, y, p, baseline, mask):
    flat = y == 1
    parts = []
    for probability in (baseline, p):
        flat_loss = np.where(
            flat, -np.log(probability[:, 1]), -np.log1p(-probability[:, 1])
        )
        direction = np.zeros(len(y))
        direction[~flat] = -np.log(
            probability[~flat][np.arange((~flat).sum()), y[~flat]]
            / (1 - probability[~flat, 1])
        )
        parts.append(
            np.array([flat_loss[mask].sum(), direction[mask].sum()]) / mask.sum()
        )
    gain = parts[0] - parts[1]
    same(saved["flat_gain"], gain[0])
    same(saved["direction_gain"], gain[1])
    same(saved["total_gain"], gain.sum())
    same(
        saved["total_gain"],
        (losses(y, baseline)[mask, 1] - losses(y, p)[mask, 1]).mean(),
    )
    check(
        saved["selected"] == int(mask.sum())
        and saved["selected_flats"] == int(np.sum(mask & flat)),
        "decomposition membership",
    )


def verify(root):
    protocol = json.loads((root / "protocol.json").read_text())
    check(
        digest(root / "protocol.json")
        == json.loads((root / "protocol.sha256.json").read_text())["sha256"],
        "protocol hash",
    )
    for n, h in json.loads((root / "manifest.json").read_text()).items():
        check(digest(root / n) == h, f"artifact hash {n}")
    for n, h in {**protocol["code"], **protocol["inputs"]}.items():
        check(digest(ROOT / n) == h, f"input/code hash {n}")
    check(
        protocol["at"] < json.loads((root / "opened.json").read_text())["at"],
        "registration chronology",
    )
    results = json.loads((root / "results.json").read_text())
    previous = json.loads((CENTER / "results.json").read_text())["cohorts"]
    cases = 0
    for year in (2024, 2025, 2026):
        for s in ASSETS:
            key = f"{year}-{s}"
            folder = SOURCE if year == 2026 else SOURCE / "history"
            prefix = s if year == 2026 else key
            prior = dict(np.load(folder / f"{prefix}-predictions.npz"))
            outcome = dict(np.load(folder / f"{prefix}-outcomes.npz"))
            center = dict(np.load(CENTER / f"{key}-predictions.npz"))
            saved = dict(np.load(root / f"{key}-predictions.npz"))
            for d in (outcome, center, saved):
                same(prior["anchor"], d["anchor"])
            frequency_audit(year, s, prior, outcome)
            flat = prior["cross" if year == 2026 else "probability"][:, 1]
            freq = prior["frequency"][:, 1]
            up = center["uniform_up"]
            y = outcome["label"]
            t = prior["anchor"]
            fs = {"old": flat, "mixed": 0.5 * flat + 0.5 * freq, "frequency": freq}
            p = {
                name: np.column_stack(((1 - f) * (1 - up), f, (1 - f) * up))
                for name, f in fs.items()
            }
            r = results["cohorts"][key]
            for name in ("mixed", "frequency"):
                same(saved[name + "_flat"], fs[name])
            for name, prob in p.items():
                same(prob[:, 2] / (1 - prob[:, 1]), up)
                check(
                    np.array_equal(prob[:, 2] >= prob[:, 0], up >= 0.5),
                    "conditional direction changed",
                )
                verify_metrics(
                    r[name], y, prob, prior["frequency"], prior["prior_reversal"], t
                )
                mask = (prob[:, 0] >= 0.52) | (prob[:, 2] >= 0.52)
                decomposition_check(
                    r["decomposition"][name], y, prob, prior["frequency"], mask
                )
            check(
                r["old"]["gate"] == previous[key]["uniform"]["gate"]
                and r["direction_changes"] == 0,
                "old gate and preserved direction",
            )
            for scope, mask in (
                ("all", np.ones(len(y), dtype=bool)),
                (
                    "frozen_selected",
                    (p["old"][:, 0] >= 0.52) | (p["old"][:, 2] >= 0.52),
                ),
            ):
                for name in ("mixed", "frequency"):
                    paired_check(
                        r["paired"][scope][name], y, p[name], p["old"], t, mask
                    )
            cases += len(y)
    check(cases == 550362 and len(results["cohorts"]) == 9, "all cohorts/cases")
    repair = {
        a: all(r[a]["gate"] for r in results["cohorts"].values())
        for a in ("mixed", "frequency")
    }
    calibration = {
        a: all(
            r["paired"][scope][a]["improvement_gate"]
            for r in results["cohorts"].values()
            for scope in ("all", "frozen_selected")
        )
        for a in ("mixed", "frequency")
    }
    directional = all(
        r["comparisons"]["uniform_vs_common"]["information_gate"]
        for r in previous.values()
    )
    check(
        results["repair_success"] == repair
        and results["calibration_success"] == calibration
        and results["prior_direction_information_success"] == directional
        and results["general_solution"]
        == (repair["mixed"] and calibration["mixed"] and directional),
        "registered aggregate gates",
    )
    return dict(
        verified=True,
        cases=cases,
        cohorts=9,
        past_frequency_windows=63,
        new_fits=0,
        protected_data_opened=False,
        repair_success=repair,
        calibration_success=calibration,
        general_solution=results["general_solution"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "reports/timestral-flat-repair-20261028"
    )
    print(json.dumps(verify(parser.parse_args().root), sort_keys=True))
