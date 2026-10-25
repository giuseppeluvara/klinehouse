#!/usr/bin/env python3
"""Verify a sealed historical test and evaluate the frozen adaptation rule."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

MODELS = ("unchanged", "timestral_price", "timestral_covariates", "adapted")
LEVELS = np.array([.05, .5, .95])
LABELS = {"unchanged": "Prezzo invariato", "timestral_price": "Timestral prezzo",
          "timestral_covariates": "Timestral + covariate", "adapted": "Adattamento selezionato"}
ANALYSIS_KEY = "scripts/analyze_timestral_adaptation.py"
CONTRASTS = (("adapted", "unchanged"), ("timestral_price", "unchanged"),
             ("timestral_covariates", "unchanged"), ("adapted", "timestral_price"))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError("timestamps must have an explicit UTC offset")
    return result


def read_records(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique_records(records):
    indexed = {record["sample_id"]: record for record in records}
    if len(indexed) != len(records) or any(not isinstance(key, str) or not key for key in indexed):
        raise ValueError("duplicate or invalid sample IDs")
    return indexed


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError("non-finite or invalid numeric forecast value")
    return float(value)


def validate_records(records, protocol):
    """Require identical model coverage, causal grids and consistent price targets."""
    start, end = utc(protocol["test_start"]), utc(protocol["test_end"])
    if start >= end or any((value.hour, value.minute, value.second, value.microsecond) != (0, 0, 0, 0)
                           for value in (start, end)):
        raise ValueError("invalid UTC test boundaries")
    unique_records(records)
    groups = {h: [] for h in (4, 24)}
    previous = {}
    for record in records:
        horizon = record["horizon_hours"]
        if isinstance(horizon, bool) or horizon not in groups:
            raise ValueError("unsupported horizon")
        anchor, label_end = utc(record["anchor"]), utc(record["label_end"])
        if (anchor.minute or anchor.second or anchor.microsecond or anchor.hour % horizon
                or label_end != anchor + timedelta(hours=horizon)
                or not start <= anchor < label_end <= end):
            raise ValueError("incorrect, unaligned or out-of-period target endpoint")
        if horizon in previous and anchor < previous[horizon] + timedelta(hours=horizon):
            raise ValueError("duplicate, unordered or overlapping anchors")
        previous[horizon] = anchor
        reference, actual = number(record["reference_price"]), number(record["actual_price"])
        outcome = number(record["actual_log_return"])
        if reference <= 0 or actual <= 0 or not np.isclose(np.log(actual / reference), outcome, atol=1e-12, rtol=1e-10):
            raise ValueError("actual price and log return disagree")
        if set(record["predictions"]) != set(MODELS):
            raise ValueError("models must share identical test samples")
        for name, forecast in record["predictions"].items():
            point = number(forecast["point"])
            if len(forecast["quantiles"]) != 3:
                raise ValueError("expected .05, .50 and .95 return quantiles")
            quantiles = [number(value) for value in forecast["quantiles"]]
            with np.errstate(over="ignore", invalid="ignore"):
                prices = reference * np.exp([point, *quantiles])
            if not np.isfinite(prices).all() or np.any(prices <= 0):
                raise ValueError("non-finite or invalid implied forecast price")
            if name == "unchanged" and point != 0:
                raise ValueError("unchanged forecast must be zero")
        groups[horizon].append(record)
    if any(not group for group in groups.values()):
        raise ValueError("both horizons require scored test observations")
    return groups


def verified_records(run):
    run = Path(run)
    protocol = json.loads((run / "protocol.json").read_text())
    results = json.loads((run / "results.json").read_text())
    if digest(run / "protocol.json") != results["protocol_sha256"]:
        raise ValueError("protocol hash mismatch")
    if protocol["code_sha256"].get(ANALYSIS_KEY) != digest(__file__):
        raise ValueError("analysis code changed after preregistration")
    required = {"predictions.jsonl", "outcomes.jsonl", "evaluation.jsonl"}
    if not required <= results["output_sha256"].keys():
        raise ValueError("required output checksum missing")
    for name, expected in results["output_sha256"].items():
        if Path(name).name != name or digest(run / name) != expected:
            raise ValueError(f"saved output hash mismatch: {name}")
    predictions = unique_records(read_records(run / "predictions.jsonl"))
    outcomes = unique_records(read_records(run / "outcomes.jsonl"))
    if predictions.keys() != outcomes.keys():
        raise ValueError("unmatched prediction/outcome IDs")
    records = read_records(run / "evaluation.jsonl")
    for record in records:
        key = record["sample_id"]
        if key not in predictions:
            raise ValueError("evaluation contains an unknown sample ID")
        prediction, outcome = predictions[key], outcomes[key]
        if any(record[field] != other[field] for other in (prediction, outcome)
               for field in ("anchor", "label_end", "horizon_hours")):
            raise ValueError("prediction/outcome/evaluation metadata disagree")
        if outcome.get("status", "scored") != "scored":
            raise ValueError("invalid outcome included in evaluation")
        if record["reference_price"] != prediction["reference_price"]:
            raise ValueError("evaluation reference price differs from prediction")
        for name in MODELS:
            if any(record["predictions"][name][field] != prediction["predictions"][name][field]
                   for field in ("point", "quantiles")):
                raise ValueError("evaluation forecasts differ from saved predictions")
        outcome_return = outcome.get("actual_log_return", outcome.get("return"))
        if record["actual_log_return"] != outcome_return or record["actual_price"] != outcome["actual_price"]:
            raise ValueError("evaluation targets differ from saved outcomes")
    expected_scored = {key for key, row in outcomes.items() if row.get("status", "scored") == "scored"}
    if {row["sample_id"] for row in records} != expected_scored:
        raise ValueError("evaluation silently omits scored outcomes")
    return protocol, results, validate_records(records, protocol)


def metrics(records):
    actual = np.array([row["actual_log_return"] for row in records])
    reference = np.array([row["reference_price"] for row in records])
    actual_price = np.array([row["actual_price"] for row in records])
    scores = {}
    for name in MODELS:
        point = np.array([row["predictions"][name]["point"] for row in records])
        quantiles = np.array([row["predictions"][name]["quantiles"] for row in records])
        errors = point - actual
        residuals = actual[:, None] - quantiles
        called = point != 0
        correct = np.sign(point) == np.sign(actual)
        price_errors = reference * np.exp(point) - actual_price
        scores[name] = {
            "samples": len(records), "return_mae": float(np.abs(errors).mean()),
            "return_rmse": float(np.sqrt(np.square(errors).mean())),
            "price_mae_usdt": float(np.abs(price_errors).mean()),
            "price_rmse_usdt": float(np.sqrt(np.square(price_errors).mean())),
            "direction_accuracy": float(correct.mean()), "direction_calls": int(called.sum()),
            "direction_call_fraction": float(called.mean()),
            "direction_accuracy_when_called": float(correct[called].mean()) if called.any() else None,
            "pinball_mean": float(np.maximum(LEVELS * residuals, (LEVELS - 1) * residuals).mean()),
            "coverage90": float(((quantiles[:, 0] <= actual) & (actual <= quantiles[:, 2])).mean()),
            "interval_width90": float((quantiles[:, 2] - quantiles[:, 0]).mean()),
            "quantile_crossing": float((np.diff(quantiles, axis=1) < 0).any(axis=1).mean()),
        }
    baseline = scores["unchanged"]["return_mae"]
    for score in scores.values():
        score["mae_skill_vs_unchanged"] = 1 - score["return_mae"] / baseline if baseline else None
    return scores


def paired_comparison(records, start, end):
    """Keep all fixed calendar blocks and weight each sampled anchor equally."""
    start, end = utc(start), utc(end)
    block_count = int(np.ceil((end - start).total_seconds() / (7 * 86400)))
    if block_count < 2:
        raise ValueError("at least two calendar blocks are required")
    block_ids = np.array([int((utc(row["anchor"]) - start).total_seconds() // (7 * 86400)) for row in records])
    if np.any(block_ids < 0) or np.any(block_ids >= block_count):
        raise ValueError("record outside bootstrap calendar")
    sizes = np.bincount(block_ids, minlength=block_count)
    choices = np.random.default_rng(17).integers(0, block_count, size=(10000, block_count))
    sampled_sizes = sizes[choices].sum(axis=1)
    if np.any(sampled_sizes == 0):
        raise ValueError("empty resampled calendar; insufficient test coverage")
    actual = np.array([row["actual_log_return"] for row in records])
    losses = {name: np.abs(np.array([row["predictions"][name]["point"] for row in records]) - actual)
              for name in MODELS}
    contrasts = {}
    for candidate, reference in CONTRASTS:
        delta = losses[candidate] - losses[reference]
        sums = np.bincount(block_ids, weights=delta, minlength=block_count)
        bootstrap = sums[choices].sum(axis=1) / sampled_sizes
        contrast = {"mean_mae_difference": float(delta.mean()),
                    "percentile95": np.quantile(bootstrap, [.025, .975]).tolist(),
                    "fraction_nonempty_blocks_lower_mae": float(np.mean((sums[sizes > 0] / sizes[sizes > 0]) < 0))}
        if (candidate, reference) == ("adapted", "unchanged"):
            contrast["percentile97_5"] = np.quantile(bootstrap, [.0125, .9875]).tolist()
        contrasts[f"{candidate}_minus_{reference}"] = contrast
    return {"method": "paired nonoverlapping seven-day calendar-block percentile bootstrap",
            "seed": 17, "draws": 10000, "blocks": block_count, "block_sample_counts": sizes.tolist(),
            "block_origin_utc": start.isoformat().replace("+00:00", "Z"),
            "negative_difference_favors_candidate": True, "contrasts": contrasts,
            "limitation": "approximate descriptive intervals; few blocks, inter-week dependence, historical design and possible pretraining contamination remain; two-sided 97.5% intervals are Bonferroni-adjusted for the two primary horizon comparisons only"}


def summarize(records, horizon, protocol):
    scores = metrics(records)
    months = sorted({row["anchor"][:7] for row in records})
    monthly = {month: metrics([row for row in records if row["anchor"][:7] == month]) for month in months}
    comparison = paired_comparison(records, protocol["test_start"], protocol["test_end"])
    baseline, adapted = scores["unchanged"], scores["adapted"]
    monthly_wins = sum(score["adapted"]["return_mae"] < score["unchanged"]["return_mae"] for score in monthly.values())
    criteria = {
        "minimum_samples": len(records) >= {4: 480, 24: 80}[horizon],
        "strict_mae_improvement": adapted["return_mae"] < baseline["return_mae"],
        "adjusted_upper_ci_below_zero": comparison["contrasts"]["adapted_minus_unchanged"]["percentile97_5"][1] < 0,
        "at_least_two_of_three_months_improve": len(months) == 3 and monthly_wins >= 2,
        "rmse_no_worse": adapted["return_rmse"] <= baseline["return_rmse"],
    }
    planned = int((utc(protocol["test_end"]) - utc(protocol["test_start"])).total_seconds() / (horizon * 3600))
    return {"samples": len(records), "expected_samples": planned, "excluded_samples": planned - len(records),
            "sample_ids": [row["sample_id"] for row in records], "selected_candidate": protocol["selected_candidates"][str(horizon)],
            "metrics": scores, "monthly": monthly, "monthly_mae_wins": monthly_wins,
            "paired_comparison": comparison, "gate": {"passed": all(criteria.values()), "criteria": criteria,
            "meaning": "predefined historical evidence gate; does not establish prospective or profitable performance"}}


def export_csv(groups, output):
    rows = []
    for horizon, group in groups.items():
        for row in group:
            for name in MODELS:
                forecast = row["predictions"][name]
                prices = row["reference_price"] * np.exp([forecast["point"], *forecast["quantiles"]])
                rows.append({"sample_id": row["sample_id"], "anchor_utc": row["anchor"],
                             "target_end_utc": row["label_end"], "horizon_hours": horizon, "model": name,
                             "reference_price_usdt": row["reference_price"], "predicted_price_usdt": prices[0],
                             "actual_price_usdt": row["actual_price"], "predicted_log_return": forecast["point"],
                             "actual_log_return": row["actual_log_return"],
                             "predicted_return_percent": float(np.expm1(forecast["point"]) * 100),
                             "actual_return_percent": float(np.expm1(row["actual_log_return"]) * 100),
                             "lower90_price_usdt": prices[1], "upper90_price_usdt": prices[3]})
    with (output / "forecasts.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def plot_comparison(groups, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [2.3, 1]})
    colors = {"unchanged": "#90a4ae", "timestral_price": "#0072b2", "timestral_covariates": "#d55e00", "adapted": "#009e73"}
    for row_index, (horizon, records) in enumerate(groups.items()):
        times = [utc(row["anchor"]) for row in records]
        ax = axes[row_index, 0]
        ax.plot(times, [np.expm1(row["actual_log_return"]) * 100 for row in records], color="#263238", linewidth=.9, label="Movimento reale")
        for name in ("timestral_price", "adapted"):
            ax.plot(times, [np.expm1(row["predictions"][name]["point"]) * 100 for row in records],
                    color=colors[name], linewidth=1, label=LABELS[name])
        ax.axhline(0, color="#78909c", linewidth=.6)
        ax.set(title=f"Orizzonte {horizon}h — {len(records)} previsioni storiche", ylabel="Variazione del prezzo (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=timezone.utc))
        ax.grid(alpha=.17)
        if row_index == 0:
            ax.legend(fontsize=8, ncol=3, loc="upper left")
        values = [summary[str(horizon)]["metrics"][name]["return_mae"] * 10000 for name in MODELS]
        ax = axes[row_index, 1]
        bars = ax.barh([LABELS[name] for name in MODELS], values, color=[colors[name] for name in MODELS])
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set(xlim=(0, max(values) * 1.25 if max(values) else 1), xlabel="MAE in punti base di log-rendimento", title="Errore medio — minore è meglio")
    fig.suptitle("BTCUSDT · adattamento congelato · verifica storica separata", fontsize=16)
    fig.text(.5, .015, "Gennaio–marzo 2026 · selezione a dicembre 2025 · disegno informato da risultati storici successivi · nessuna misura di profitto", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .94])
    fig.savefig(output / "forecast-comparison.png", dpi=160, facecolor="white")
    plt.close(fig)


def analyze(run, output):
    protocol, results, groups = verified_records(run)
    horizons = {str(h): summarize(records, h, protocol) for h, records in groups.items()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    analysis = {"schema": 1, "source_run": str(Path(run).resolve()),
                "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "historical": True, "protocol_sha256": results["protocol_sha256"],
                "input_sha256": results["output_sha256"], "analysis_code_sha256": digest(__file__),
                "horizons": horizons, "all_horizons_pass": all(row["gate"]["passed"] for row in horizons.values())}
    analysis["csv_rows"] = export_csv(groups, output)
    plot_comparison(groups, horizons, output)
    analysis["artifact_sha256"] = {name: digest(output / name) for name in ("forecasts.csv", "forecast-comparison.png")}
    with (output / "analysis.json").open("x") as handle:
        json.dump(analysis, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.output)
    print(json.dumps({"csv_rows": result["csv_rows"], "all_horizons_pass": result["all_horizons_pass"]}, indent=2))


if __name__ == "__main__":
    main()
