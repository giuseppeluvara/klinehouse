#!/usr/bin/env python3
"""Verify saved forecasts and produce fixed paired comparisons and research plots."""

from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_experiment as shared  # noqa: E402

CONTRASTS = (
    ("timestral_price", "unchanged"),
    ("timestral_covariates", "unchanged"),
    ("timestral_price", "local_drift"),
    ("timestral_covariates", "local_drift"),
    ("timestral_covariates", "timestral_price"),
)
LABELS = {
    "unchanged": "Prezzo invariato", "local_drift": "Drift locale",
    "timestral_price": "Timestral prezzo", "timestral_covariates": "Timestral + covariate",
}


def read_records(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verified_pairs(run):
    run = Path(run)
    protocol = json.loads((run / "protocol.json").read_text())
    results = json.loads((run / "results.json").read_text())
    if shared.digest_file(run / "protocol.json") != results["protocol_sha256"]:
        raise ValueError("protocol hash mismatch")
    if shared.digest_file(__file__) != protocol["code_sha256"]["scripts/analyze_timestral_long_horizon.py"]:
        raise ValueError("analysis code changed after preregistration")
    for name, expected in results["output_sha256"].items():
        if Path(name).name != name or shared.digest_file(run / name) != expected:
            raise ValueError(f"saved output hash mismatch: {name}")
    predictions, outcomes = read_records(run / "predictions.jsonl"), read_records(run / "outcomes.jsonl")
    pred = {record["sample_id"]: record for record in predictions}
    actual = {record["sample_id"]: record for record in outcomes}
    if len(pred) != len(predictions) or len(actual) != len(outcomes) or pred.keys() != actual.keys():
        raise ValueError("duplicate or unmatched prediction/outcome IDs")
    pairs = {}
    for horizon in protocol["config"]["horizons"]:
        selected = sorted([(record, actual[key]) for key, record in pred.items()
                           if record["horizon_hours"] == horizon and actual[key]["status"] == "scored"],
                          key=lambda pair: pair[0]["anchor"])
        if [record["sample_id"] for record, _ in selected] != results["horizons"][str(horizon)]["sample_ids"]:
            raise ValueError("scored IDs differ from results")
        previous = None
        for record, outcome in selected:
            anchor = shared.utc(record["anchor"])
            if record["label_end"] != shared.stamp(anchor + timedelta(hours=horizon)):
                raise ValueError("incorrect target endpoint")
            if previous is not None and anchor < previous + timedelta(hours=horizon):
                raise ValueError("overlapping labels within horizon")
            previous = anchor
            if any(record[key] != outcome[key] for key in ("anchor", "label_end", "horizon_hours")):
                raise ValueError("prediction and outcome metadata differ")
            if set(record["predictions"]) != set(shared.MODELS):
                raise ValueError("models do not share identical samples")
        recalculated = shared.metric_summary([(record, outcome["return"]) for record, outcome in selected])
        for model, metrics in recalculated.items():
            expected = results["horizons"][str(horizon)]["metrics"][model]
            for name in ("return_mae", "return_rmse", "direction_accuracy", "pinball_mean", "coverage90"):
                if not np.isclose(metrics[name], expected[name], rtol=1e-12, atol=1e-15):
                    raise ValueError(f"metric mismatch: {horizon}h {model} {name}")
        pairs[horizon] = selected
    return protocol, results, pairs


def paired_block_comparison(pairs, evaluation_start, draws=2000, seed=7):
    """Resample paired seven-day calendar clusters, never individual anchors."""
    start = shared.utc(evaluation_start)
    block_ids = np.array([int((shared.utc(record["anchor"]) - start).total_seconds() // (7 * 86400))
                          for record, _ in pairs])
    blocks = [np.flatnonzero(block_ids == block) for block in sorted(set(block_ids))]
    if len(blocks) < 2:
        raise ValueError("at least two calendar blocks are required")
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(blocks), size=(draws, len(blocks)))
    sizes = np.array([len(block) for block in blocks])
    actual = np.array([outcome["return"] for _, outcome in pairs])
    losses = {name: np.abs(np.array([record["predictions"][name]["point"] for record, _ in pairs]) - actual)
              for name in shared.MODELS}
    contrasts = {}
    for candidate, reference in CONTRASTS:
        delta = losses[candidate] - losses[reference]
        sums = np.array([delta[block].sum() for block in blocks])
        bootstrap = sums[choices].sum(axis=1) / sizes[choices].sum(axis=1)
        contrasts[f"{candidate}_minus_{reference}"] = {
            "mean_mae_difference": float(delta.mean()),
            "percentile95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "fraction_calendar_blocks_lower_mae": float(np.mean([delta[block].mean() < 0 for block in blocks])),
        }
    return {"method": "paired nonoverlapping seven-day calendar-block bootstrap from evaluation start",
            "draws": draws, "seed": seed, "blocks": len(blocks), "block_sample_counts": sizes.tolist(),
            "block_origin_utc": shared.stamp(start), "sample_ids": [record["sample_id"] for record, _ in pairs],
            "negative_difference_favors_candidate": True,
            "limitation": "descriptive; few blocks, inter-week dependence, multiple comparisons and pretraining contamination remain",
            "contrasts": contrasts}


def export_csv(pairs, output):
    rows = []
    for horizon, group in pairs.items():
        for record, outcome in group:
            for name, forecast in record["predictions"].items():
                rows.append({
                    "sample_id": record["sample_id"], "anchor_utc": record["anchor"],
                    "target_end_utc": record["label_end"], "horizon_hours": horizon, "model": name,
                    "reference_price_usdt": record["reference_price"],
                    "predicted_price_usdt": forecast["point_price"], "actual_price_usdt": outcome["actual_price"],
                    "predicted_return_percent": forecast["point_return_percent"], "actual_return_percent": outcome["return_percent"],
                    "predicted_log_return": forecast["point"], "actual_log_return": outcome["return"],
                    "lower90_price_usdt": forecast["price_quantiles"][0],
                    "upper90_price_usdt": forecast["price_quantiles"][2],
                    "calibration_samples": record["calibration"]["count"],
                })
    with (output / "forecasts.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def plot_comparison(pairs, results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [2.3, 1]})
    for row, (horizon, group) in enumerate(pairs.items()):
        times = [shared.utc(record["anchor"]) for record, _ in group]
        actual = [outcome["return_percent"] for _, outcome in group]
        ax = axes[row, 0]
        ax.plot(times, actual, color="#263238", linewidth=.9, alpha=.8, label="Movimento reale")
        for name, color in (("timestral_price", "#0072b2"), ("timestral_covariates", "#d55e00")):
            ax.plot(times, [record["predictions"][name]["point_return_percent"] for record, _ in group],
                    color=color, linewidth=1, alpha=.85, label=LABELS[name])
        ax.axhline(0, color="#78909c", linewidth=.7)
        ax.set_title(f"Orizzonte {horizon}h — {len(group)} previsioni storiche")
        ax.set_ylabel("Variazione del prezzo (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=times[0].tzinfo))
        ax.grid(alpha=.17)
        if row == 0:
            ax.legend(fontsize=8, loc="upper left", ncol=3)
        scores = results["horizons"][str(horizon)]["metrics"]
        values = [scores[name]["return_mae"] * 10000 for name in shared.MODELS]
        bars = axes[row, 1].barh([LABELS[name] for name in shared.MODELS], values,
                                color=["#90a4ae", "#78909c", "#0072b2", "#d55e00"])
        axes[row, 1].invert_yaxis()
        axes[row, 1].bar_label(bars, fmt="%.1f", padding=4)
        axes[row, 1].set_xlim(0, max(values) * 1.22)
        axes[row, 1].set_xlabel("MAE in punti base di log-rendimento")
        axes[row, 1].set_title("Errore medio — minore è meglio")
    fig.suptitle("BTCUSDT · secondi → ore · movimenti a 4h e 1 giorno", fontsize=17)
    fig.text(.5, .015, "Maggio–luglio 2026 · 256 ore di contesto · zero-shot · storico esplorativo, nessuna misura di profitto",
             ha="center", fontsize=10, color="#455a64")
    fig.tight_layout(rect=[0, .04, 1, .94])
    fig.savefig(output / "forecast-comparison.png", dpi=160, facecolor="white")
    plt.close(fig)


def analyze(run, output):
    protocol, results, pairs = verified_pairs(run)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    analysis = {"source_run": str(Path(run).resolve()), "protocol_sha256": results["protocol_sha256"],
                "analysis_code_sha256": shared.digest_file(__file__),
                "horizons": {str(h): paired_block_comparison(group, protocol["config"]["evaluation_start"])
                             for h, group in pairs.items()}}
    analysis["csv_rows"] = export_csv(pairs, output)
    plot_comparison(pairs, results, output)
    shared.exclusive_json(output / "analysis.json", analysis)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.output)
    print(json.dumps({"csv_rows": result["csv_rows"], "horizons": result["horizons"]}, indent=2))


if __name__ == "__main__":
    main()
