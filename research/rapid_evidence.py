"""Frozen rapid-evidence ablations and a fit-free hourly shadow command."""
# ruff: noqa: E402, E701, E702

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import pickle
import random
import time
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

BASE = Path(__file__).resolve().parent.parent
ROOT = Path(__file__).resolve().parent / "out" / "sequence"
OUT = ROOT / "rapid"
BUNDLE = ROOT / "bundle"
PROTOCOL = BASE / "RAPID_EVIDENCE.md"
PROTOCOL_SHA = "34616e0edd3ab5d99f2ca3f71b480d9f371fe743f96cb7e1f795e85a9b540df4"
P0 = pd.Timestamp("2026-08-10T10:00:00Z")
COST, DECISION_THRESHOLD = 0.002, 0.002
SHUFFLE_SEED, SEEDS = 20260810, (7, 17, 29)
SYMBOLS, LOOKBACK, HORIZON, MINUTE_MS, CLASS_THRESHOLD = (
    ("BTCUSDT", "ETHUSDT", "SOLUSDT"), 120, 60, 60_000, 0.002,
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def verify_bundle() -> dict:
    if sha(PROTOCOL) != PROTOCOL_SHA: raise RuntimeError("rapid protocol changed")
    freeze = json.loads((BUNDLE / "freeze.json").read_text())
    for name, expected in freeze["files"].items():
        if sha(BUNDLE / name) != expected: raise RuntimeError(f"frozen file changed: {name}")
    if sha(ROOT / "validation" / "oof.parquet") != freeze["validation_oof_hash"]:
        raise RuntimeError("frozen OOF changed")
    if sha(ROOT / "validation" / "report.json") != freeze["validation_report_hash"]:
        raise RuntimeError("frozen validation report changed")
    return freeze


def shuffled_steps(x: np.ndarray, sample_ids: np.ndarray, seed: int = SHUFFLE_SEED) -> np.ndarray:
    """Apply the protocol's deterministic, sample-specific temporal shuffle."""
    result = np.empty_like(x)
    for pos, sample_id in enumerate(np.asarray(sample_ids, np.uint64)):
        words = [seed, int(sample_id & np.uint64(0xFFFFFFFF)), int(sample_id >> np.uint64(32))]
        order = np.random.default_rng(np.random.SeedSequence(words)).permutation(x.shape[1])
        result[pos] = x[pos, order]
    return result


def aggregate_48(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, np.float64); t = np.arange(values.shape[1], dtype=np.float64)
    centered = t - t.mean(); denominator = np.square(centered).sum()
    slope = np.einsum("ntc,t->nc", values, centered) / denominator
    return np.concatenate((values[:, -1], values.mean(1), values.std(1), slope), axis=1).astype(np.float32)


def ensemble(probabilities: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(probabilities), axis=0)


def _torch():
    import torch
    from torch import nn
    return torch, nn


def _set_seed(seed: int, deterministic: bool) -> None:
    torch, _ = _torch(); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(deterministic)


def make_model(kind: str, channels: int):
    torch, nn = _torch()
    if kind == "mlp":
        return nn.Sequential(nn.Linear(48, 128), nn.GELU(), nn.Dropout(0.1),
                             nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 3))
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__(); d = 32; self.project = nn.Linear(channels, d)
            position = torch.arange(LOOKBACK).unsqueeze(1)
            scale = torch.exp(torch.arange(0, d, 2) * (-math.log(10_000) / d))
            pe = torch.zeros(LOOKBACK, d); pe[:, 0::2] = torch.sin(position * scale); pe[:, 1::2] = torch.cos(position * scale)
            self.register_buffer("position", pe)
            layer = nn.TransformerEncoderLayer(d, 4, 64, 0.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False); self.head = nn.Linear(d, 3)
        def forward(self, value): return self.head(self.encoder(self.project(value) + self.position).mean(1))
    return Transformer()


def fit_neural(kind: str, x, y, mean, std, seed: int, device: str, batch: int = 256):
    torch, nn = _torch(); _set_seed(seed, device == "cpu"); model = make_model(kind, x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    for _ in range(10):
        model.train()
        for start in range(0, len(x), batch):
            xb = ((np.asarray(x[start:start + batch]) - mean) / std).astype(np.float32)
            xb = torch.from_numpy(xb).to(device); yb = torch.from_numpy(np.asarray(y[start:start + batch])).to(device)
            optimizer.zero_grad(); loss = nn.functional.cross_entropy(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    return model


def predict(model, x, mean, std, device: str, batch: int = 256) -> np.ndarray:
    torch, _ = _torch(); model.eval(); output = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = ((np.asarray(x[start:start + batch]) - mean) / std).astype(np.float32)
            output.append(torch.softmax(model(torch.from_numpy(xb).to(device)), 1).cpu().numpy())
    return np.concatenate(output)


def scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = (0, 1) if x.ndim == 3 else 0
    mean = np.asarray(x).mean(axis=axis, dtype=np.float64); std = np.asarray(x).std(axis=axis, dtype=np.float64)
    std[std == 0] = 1; return mean.astype(np.float32), std.astype(np.float32)


def ids_to_rows(all_ids: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    rows = np.searchsorted(all_ids, wanted)
    if np.any(rows == len(all_ids)) or not np.array_equal(all_ids[rows], wanted):
        raise RuntimeError("sample ID mismatch")
    return rows


def continuous_returns(anchor: np.ndarray) -> np.ndarray:
    from research import sequence_experiment
    source = json.loads((ROOT / "source.json").read_text()); frame = sequence_experiment._read_archives("BTCUSDT", source)
    times = frame["time"].to_numpy(np.int64); opened = frame["open"].to_numpy(float); closed = frame["close"].to_numpy(float)
    rows = np.searchsorted(times, anchor); ends = rows + HORIZON - 1
    if np.any(ends >= len(times)) or not np.array_equal(times[rows], anchor) or not np.array_equal(times[ends], anchor + 59 * MINUTE_MS):
        raise RuntimeError("continuous return has incomplete future")
    return np.log(closed[ends] / opened[rows])


def block_ci(frame: pd.DataFrame, left: str, right: str, draws: int = 10_000) -> list[float]:
    rng = np.random.default_rng(7); grouped = []
    for _, fold in frame.groupby("fold", sort=True):
        week = fold["sample_id"].to_numpy(np.int64) // (7 * 86_400_000)
        grouped.append([g[left].to_numpy() - g[right].to_numpy() for _, g in fold.assign(week=week).groupby("week")])
    estimates = [np.concatenate([groups[i] for groups in grouped for i in rng.integers(0, len(groups), len(groups))]).mean()
                 for _ in range(draws)]
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    predicted = p.argmax(1)
    return {"log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
            "brier": float(np.square(p - np.eye(3)[y]).sum(1).mean()),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted))}


def sample_loss(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(probability[np.arange(len(y)), y], 1e-15, 1))


def diagnostics(oof: pd.DataFrame, returns: np.ndarray) -> dict:
    y = oof["y"].to_numpy(); result = {}
    for name in ("prior", "logistic", "transformer"):
        p = oof[[f"p_{name}_{i}" for i in range(3)]].to_numpy(); loss = sample_loss(y, p)
        confidence, correct = p.max(1), p.argmax(1) == y; bins = np.minimum((confidence * 10).astype(int), 9)
        calibration, ece = [], 0.0
        for number in range(10):
            take = bins == number; count = int(take.sum())
            acc = float(correct[take].mean()) if count else None; conf = float(confidence[take].mean()) if count else None
            if count: ece += count / len(y) * abs(acc - conf)
            calibration.append({"bin": number, "count": count, "confidence": conf, "accuracy": acc,
                                "future_return": float(returns[take].mean()) if count else None})
        monthly = pd.DataFrame({"month": oof["anchor"].dt.strftime("%Y-%m"), "loss": loss}).groupby("month").loss.mean().to_dict()
        by_class = {str(cls): float(loss[y == cls].mean()) for cls in range(3)}
        by_fold = {str(f): float(loss[oof["fold"].to_numpy() == f].mean()) for f in sorted(oof["fold"].unique())}
        result[name] = {**probability_metrics(y, p), "ece_10": ece, "log_loss_by_fold": by_fold,
                        "log_loss_by_month": monthly, "log_loss_by_class": by_class,
                        "confidence_quantiles": np.quantile(confidence, [0, .1, .25, .5, .75, .9, 1]).tolist(),
                        "calibration": calibration}
    lt = oof["loss_transformer"].to_numpy() - oof["loss_logistic"].to_numpy()
    weeks = pd.DataFrame({"fold": oof["fold"], "week": oof["sample_id"] // (7 * 86_400_000), "delta": lt})
    result["transformer_weeks_beating_logistic_pct"] = float(100 * (weeks.groupby(["fold", "week"]).delta.mean() < 0).mean())
    return result


def fold_class_means(y: np.ndarray, returns: np.ndarray, train: np.ndarray) -> np.ndarray:
    return np.array([returns[train][y[train] == cls].mean() for cls in range(3)])


def trading_stats(decision: np.ndarray, returns: np.ndarray) -> dict:
    active = decision != 0; gross = decision[active] * returns[active]; net = gross - COST
    curve = np.r_[0.0, np.cumsum(net)]; drawdown = curve - np.maximum.accumulate(curve)
    return {"trades": int(active.sum()), "long": int((decision > 0).sum()), "short": int((decision < 0).sum()),
            "turnover": int(2 * active.sum()), "gross_return": float(gross.sum()), "net_return": float(net.sum()),
            "mean_net": float(net.mean()) if len(net) else None, "median_net": float(np.median(net)) if len(net) else None,
            "hit_rate": float((net > 0).mean()) if len(net) else None, "max_drawdown": float(drawdown.min()),
            "break_even_cost": float(gross.mean()) if len(gross) else None}


def economics(oof: pd.DataFrame, all_y: np.ndarray, all_returns: np.ndarray, all_ids: np.ndarray) -> dict:
    output = {"cost_roundtrip": COST, "decision_threshold": DECISION_THRESHOLD, "models": {}}
    oof_returns = all_returns[ids_to_rows(all_ids, oof["sample_id"].to_numpy(np.int64))]
    for name in ("logistic", "transformer"):
        decisions = np.zeros(len(oof), np.int8); fold_stats = []
        for fold in (1, 2, 3):
            ids = np.load(BUNDLE / f"fold-{fold}-ids.npz")["train"]
            mus = fold_class_means(all_y, all_returns, ids_to_rows(all_ids, ids)); take = oof["fold"].to_numpy() == fold
            p = oof.loc[take, [f"p_{name}_{i}" for i in range(3)]].to_numpy(); expected = p @ mus
            decisions[take] = np.where(expected > DECISION_THRESHOLD, 1, np.where(expected < -DECISION_THRESHOLD, -1, 0))
            fold_stats.append({"fold": fold, "class_means": mus.tolist(), **trading_stats(decisions[take], oof_returns[take])})
        output["models"][name] = {**trading_stats(decisions, oof_returns), "folds": fold_stats}
    return output


def run() -> dict:
    from research import sequence_experiment
    freeze = verify_bundle(); OUT.mkdir(parents=True, exist_ok=True); report_path = OUT / "report.json"
    if report_path.exists(): raise FileExistsError("rapid evidence already executed")
    data = sequence_experiment.load_dataset(); x, all_ids, y = (data[k] for k in ("x", "anchor", "y"))
    oof = pd.read_parquet(ROOT / "validation" / "oof.parquet").sort_values("sample_id").reset_index(drop=True)
    if len(oof) != len(set(oof.sample_id)) or not np.array_equal(oof.sample_id.to_numpy(), np.sort(oof.sample_id.to_numpy())):
        raise RuntimeError("invalid frozen OOF IDs")
    shuffled = shuffled_steps(x, all_ids); flat = aggregate_48(x); outputs = []
    definitions = (("shuffle", "transformer", shuffled), ("mlp", "mlp", flat), ("btc_only", "transformer", x[:, :, :4]))
    started = time.monotonic(); device, batch = freeze["backend"], int(freeze["batch"])
    for fold in (1, 2, 3):
        saved = np.load(BUNDLE / f"fold-{fold}-ids.npz"); tr = ids_to_rows(all_ids, saved["train"]); va = ids_to_rows(all_ids, saved["validation"])
        expected = oof.index[oof.fold == fold].to_numpy()
        if not np.array_equal(all_ids[va], oof.loc[expected, "sample_id"].to_numpy()): raise RuntimeError("OOF validation identity changed")
        for name, kind, values in definitions:
            mean, std = scaler(values[tr]); seed_probabilities, seed_metrics = [], {}
            for seed in SEEDS:
                print(f"fold={fold} ablation={name} seed={seed}", flush=True)
                model = fit_neural(kind, values[tr], y[tr], mean, std, seed, device, batch)
                probability = predict(model, values[va], mean, std, device, batch); seed_probabilities.append(probability)
                seed_metrics[str(seed)] = probability_metrics(np.asarray(y[va]), probability)
                del model
            probability = ensemble(seed_probabilities)
            for cls in range(3): oof.loc[expected, f"p_{name}_{cls}"] = probability[:, cls]
            oof.loc[expected, f"loss_{name}"] = sample_loss(np.asarray(y[va]), probability)
            outputs.append({"fold": fold, "ablation": name, "ensemble": probability_metrics(np.asarray(y[va]), probability),
                            "seeds": seed_metrics})
    returns = continuous_returns(np.asarray(all_ids)); oof.to_parquet(OUT / "ablation_oof.parquet", index=False)
    comparisons = {}
    for name, _, _ in definitions:
        delta = oof[f"loss_{name}"] - oof["loss_transformer"]
        comparisons[name] = {"metrics": probability_metrics(oof.y.to_numpy(), oof[[f"p_{name}_{i}" for i in range(3)]].to_numpy()),
                             "mean_log_loss_minus_transformer": float(delta.mean()),
                             "paired_weekly_bootstrap_95": block_ci(oof, f"loss_{name}", "loss_transformer")}
    final_ids = np.load(BUNDLE / "sample_ids.npy"); final_rows = ids_to_rows(all_ids, final_ids)
    final_mus = fold_class_means(np.asarray(y), returns, final_rows)
    result = {"created": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
              "protocol_sha256": PROTOCOL_SHA, "freeze_sha256": sha(BUNDLE / "freeze.json"), "samples": len(oof),
              "frozen_git_commit": freeze["git_commit"], "dataset_hash": freeze["dataset_hash"],
              "validation_oof_hash": freeze["validation_oof_hash"], "bundle_files": freeze["files"],
              "device": device, "batch": batch, "comparisons": comparisons, "folds": outputs,
              "diagnostics": diagnostics(oof, returns[ids_to_rows(all_ids, oof.sample_id.to_numpy(np.int64))]),
              "economics": economics(oof, np.asarray(y), returns, np.asarray(all_ids))}
    _write_json(OUT / "shadow-config.json", {"protocol_sha256": PROTOCOL_SHA, "freeze_sha256": result["freeze_sha256"],
                "class_means": final_mus.tolist(), "p0": P0.isoformat(), "cost_roundtrip": COST,
                "decision_threshold": DECISION_THRESHOLD})
    (OUT / "report.md").write_text(render_report(result))
    _write_json(report_path, result, exclusive=True)
    return result


def _write_json(path: Path, value: dict, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream: json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")


def render_report(report: dict) -> str:
    rows = ["# Rapid evidence — risultati", "", "| Modello | Log-loss | Δ vs Transformer | CI 95% |",
            "| --- | ---: | ---: | --- |"]
    base = json.loads((ROOT / "validation" / "report.json").read_text())["aggregate"]["transformer"]["log_loss"]
    rows.append(f"| Transformer originale | {base:.6f} | 0.000000 | riferimento |")
    for name, value in report["comparisons"].items():
        ci = value["paired_weekly_bootstrap_95"]
        rows.append(f"| {name} | {value['metrics']['log_loss']:.6f} | {value['mean_log_loss_minus_transformer']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |")
    interpretations = []
    labels = {"shuffle": "ordine temporale", "mlp": "rappresentazione sequenziale", "btc_only": "contesto ETH/SOL"}
    for name, value in report["comparisons"].items():
        low, high = value["paired_weekly_bootstrap_95"]
        if low > 0: statement = "contributo supportato: l'ablation è peggiore"
        elif high < 0: statement = "l'ablation è migliore; l'ipotesi originale non è supportata"
        else: statement = "differenza non risolta dal CI"
        interpretations.append(f"- {labels[name]}: {statement}.")
    rows += ["", "## Interpretazione congelata", "", *interpretations, "", "## Economia OOF esplorativa", "",
             "| Modello | Trade | Long | Short | Gross | Netto | Max DD | Break-even bps |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, stats in report["economics"]["models"].items():
        break_even = "n/a" if stats["break_even_cost"] is None else f"{stats['break_even_cost'] * 10_000:.2f}"
        rows.append(f"| {name} | {stats['trades']} | {stats['long']} | {stats['short']} | {stats['gross_return']:.4f} | "
                    f"{stats['net_return']:.4f} | {stats['max_drawdown']:.4f} | {break_even} |")
    diagnostic = report["diagnostics"]
    rows += ["", "## Diagnostica sintetica", "",
             "| Modello | Brier | ECE-10 | Macro-F1 | Balanced accuracy |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in ("prior", "logistic", "transformer"):
        value = diagnostic[name]
        rows.append(f"| {name} | {value['brier']:.6f} | {value['ece_10']:.6f} | {value['macro_f1']:.6f} | {value['balanced_accuracy']:.6f} |")
    rows += ["", f"Il Transformer batte la logistic nel {diagnostic['transformer_weeks_beating_logistic_pct']:.1f}% delle settimane OOF.", "",
             "Valutazione descrittiva; non è prova di profittabilità o robustezza multi-regime.", ""]
    return "\n".join(rows)


def append_unique(path: Path, record: dict, key: str = "sample_id") -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX); stream.seek(0)
        if any(json.loads(line)[key] == record[key] for line in stream if line.strip()): return False
        stream.seek(0, os.SEEK_END); stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    return True


def feature_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    times = frame["time"].to_numpy(np.int64); close = frame["close"].to_numpy(float)
    contiguous = np.r_[False, np.diff(times) == MINUTE_MS]; returns = np.full(len(frame), np.nan, np.float32)
    returns[contiguous] = np.log(close[contiguous] / close[np.flatnonzero(contiguous) - 1])
    opened = frame["open"].to_numpy(float); volume = frame["volume"].to_numpy(float)
    features = np.column_stack((returns, (frame["high"].to_numpy(float) - frame["low"].to_numpy(float)) / opened,
        np.log1p(frame["quote_volume"].to_numpy(float)), np.divide(2 * frame["taker_buy_base"].to_numpy(float) - volume,
        volume, out=np.zeros(len(frame)), where=volume > 0))).astype(np.float32)
    return times, features


def live_sequence(anchor: pd.Timestamp) -> np.ndarray:
    from research.db import connect, load_klines
    frames = {}; connection = connect(read_only=True)
    try:
        for symbol in SYMBOLS:
            raw = load_klines(symbol, (anchor - pd.Timedelta(minutes=121)).isoformat(), anchor.isoformat(), connection)
            raw["minute"] = raw.open_time.dt.floor("min"); rows = []
            for minute, group in raw.groupby("minute", sort=True):
                milliseconds = group.open_time.astype("int64").to_numpy() // 10**6
                if len(group) != 60 or milliseconds[0] != int(minute.timestamp() * 1000) or not np.all(np.diff(milliseconds) == 1000): continue
                rows.append({"time": milliseconds[0], "open": group.open.iloc[0], "high": group.high.max(), "low": group.low.min(),
                             "close": group.close.iloc[-1], "volume": group.volume.sum(), "quote_volume": group.quote_volume.sum(),
                             "taker_buy_base": group.taker_buy_base.sum()})
            frames[symbol] = pd.DataFrame(rows)
    finally: connection.close()
    times_features = [feature_arrays(frames[symbol]) for symbol in SYMBOLS]
    times = times_features[0][0]
    if len(times) != 121 or any(not np.array_equal(times, item[0]) for item in times_features[1:]):
        raise RuntimeError("incomplete or unaligned live sequence")
    value = np.column_stack([item[1] for item in times_features])[-120:]
    if not np.isfinite(value).all(): raise RuntimeError("non-finite live sequence")
    return value[None].astype(np.float32)


def live_return(anchor: pd.Timestamp) -> float:
    from research.db import connect, load_klines
    connection = connect(read_only=True)
    try: frame = load_klines("BTCUSDT", anchor.isoformat(), (anchor + pd.Timedelta(hours=1)).isoformat(), connection)
    finally: connection.close()
    milliseconds = frame.open_time.astype("int64").to_numpy() // 10**6
    if len(frame) != 3600 or milliseconds[0] != int(anchor.timestamp() * 1000) or not np.all(np.diff(milliseconds) == 1000):
        raise RuntimeError("shadow label is not mature and complete")
    return float(np.log(frame.close.iloc[-1] / frame.open.iloc[0]))


def decision(expected: float) -> int:
    return 1 if expected > DECISION_THRESHOLD else -1 if expected < -DECISION_THRESHOLD else 0


def mature_shadow(now: pd.Timestamp, return_loader=live_return) -> int:
    events, labels = OUT / "events.jsonl", OUT / "maturations.jsonl"
    if not events.exists(): return 0
    known = set()
    if labels.exists():
        with labels.open() as stream: known = {json.loads(line)["sample_id"] for line in stream if line.strip()}
    count = 0
    with events.open() as stream: records = [json.loads(line) for line in stream if line.strip()]
    for event in records:
        anchor = pd.Timestamp(event["anchor"])
        if event["sample_id"] in known or anchor + pd.Timedelta(hours=1) > now: continue
        value = return_loader(anchor); label = 0 if value <= -CLASS_THRESHOLD else 2 if value >= CLASS_THRESHOLD else 1
        record = {"sample_id": event["sample_id"], "label_matured_at": now.isoformat(), "future_return": value, "y": label}
        count += int(append_unique(labels, record)); known.add(event["sample_id"])
    return count


def shadow_once(now: pd.Timestamp | None = None, sequence_loader=live_sequence) -> dict:
    freeze = verify_bundle(); config = json.loads((OUT / "shadow-config.json").read_text())
    if config["freeze_sha256"] != sha(BUNDLE / "freeze.json"): raise RuntimeError("shadow config/bundle mismatch")
    now = now or pd.Timestamp.now(tz="UTC"); now = now.tz_convert("UTC") if now.tzinfo else now.tz_localize("UTC")
    anchor = now.floor("h")
    if anchor < P0: raise RuntimeError(f"first prospective anchor is {P0.isoformat()}")
    x = sequence_loader(anchor); scaler_file = np.load(BUNDLE / "scaler.npz"); mean, std = scaler_file["mean"], scaler_file["std"]
    prior = np.load(BUNDLE / "class_prior.npy")
    with (BUNDLE / "logistic.pkl").open("rb") as stream: logistic = pickle.load(stream)
    logistic_p = logistic.predict_proba(((x - mean) / std).reshape(len(x), -1))[0]
    torch, _ = _torch(); transformer_p = []
    for seed in SEEDS:
        model = make_model("transformer", 12); model.load_state_dict(torch.load(BUNDLE / f"transformer-{seed}.pt", map_location="cpu", weights_only=True))
        transformer_p.append(predict(model, x, mean, std, "cpu", freeze["batch"])[0])
    probabilities = {"prior": prior, "logistic": logistic_p, "transformer": ensemble(transformer_p)}
    mus = np.asarray(config["class_means"]); models = {}
    for name, p in probabilities.items():
        expected = float(p @ mus); models[name] = {"probabilities": p.tolist(), "expected_return": expected, "decision": decision(expected)}
    record = {"sample_id": int(anchor.timestamp() * 1000), "anchor": anchor.isoformat(), "created": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": PROTOCOL_SHA, "freeze_sha256": config["freeze_sha256"], "models": models}
    written = append_unique(OUT / "events.jsonl", record); matured = mature_shadow(now)
    return {"sample_id": record["sample_id"], "written": written, "matured": matured, "models": models}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("run", "shadow-once")); args = parser.parse_args()
    action = run if args.command == "run" else shadow_once
    print(json.dumps(action(), indent=2, default=str)); return 0


if __name__ == "__main__": raise SystemExit(main())
