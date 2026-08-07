"""Frozen BTC experiment: class prior vs logistic vs a two-layer Transformer."""
# ruff: noqa: E402, E701, E702

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import io
import json
import math
import os
import pickle
import random
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import httpx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SEEDS = (7, 17, 29)
MINUTE_MS, LOOKBACK, HORIZON, THRESHOLD = 60_000, 120, 60, 0.002
SOURCE_START = pd.Timestamp("2020-08-01", tz="UTC")
ROOT = Path(__file__).resolve().parent / "out" / "sequence"
SPEC = {
    "target": "BTCUSDT", "symbols": SYMBOLS, "frequency": "1m", "lookback": LOOKBACK,
    "horizon": HORIZON, "anchor_minutes": 60, "threshold": THRESHOLD,
    "features": ("log_return", "range", "log_quote_volume", "taker_imbalance"),
    "logistic": {"C": 1.0, "solver": "lbfgs", "class_weight": None,
                 "max_iter": 2000, "tol": 1e-4, "random_state": 7},
    "transformer": {"d_model": 32, "heads": 4, "layers": 2, "feedforward": 64,
                    "dropout": 0.1, "epochs": 10, "batch": 256, "lr": 3e-4,
                    "weight_decay": 1e-2, "seeds": SEEDS},
}


def _json(path: Path, value: dict, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str); stream.write("\n")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _archive_specs(cutoff: pd.Timestamp) -> list[tuple[str, str, str]]:
    specs, month_end = [], cutoff.floor("D").replace(day=1)
    months = pd.period_range(SOURCE_START.tz_localize(None),
                             month_end.tz_localize(None) - pd.offsets.MonthBegin(), freq="M")
    days = pd.date_range(month_end, cutoff.floor("D") - pd.Timedelta(days=1), freq="D")
    for symbol in SYMBOLS:
        for period, folder in [(months, "monthly"), (days, "daily")]:
            for value in period:
                suffix = str(value) if folder == "monthly" else f"{value:%Y-%m-%d}"
                name = f"{symbol}-1m-{suffix}.zip"
                url = f"https://data.binance.vision/data/spot/{folder}/klines/{symbol}/1m/{name}"
                specs.append((symbol, name, url))
    return specs


def _fetch(client: httpx.Client, url: str) -> bytes:
    for attempt in range(5):
        try: response = client.get(url)
        except httpx.TransportError:
            if attempt == 4: raise
            time.sleep(2**attempt); continue
        if response.status_code == 200: return response.content
        if response.status_code != 429 and response.status_code < 500: response.raise_for_status()
        time.sleep(2**attempt)
    raise RuntimeError(f"download failed after retries: {url}")


def prepare_sources(cutoff: str) -> dict:
    end, source_dir = _utc(cutoff), ROOT / "sources"; source_dir.mkdir(parents=True, exist_ok=True)
    specs, records = _archive_specs(end), []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        def download(spec):
            symbol, name, url = spec
            path = source_dir / symbol / name; path.parent.mkdir(parents=True, exist_ok=True)
            checksum = _fetch(client, url + ".CHECKSUM").decode().split()[0]
            if not path.exists() or _sha(path) != checksum:
                payload = _fetch(client, url)
                if hashlib.sha256(payload).hexdigest() != checksum: raise RuntimeError(f"checksum mismatch: {name}")
                part = path.with_suffix(".part"); part.write_bytes(payload); part.replace(path)
            return {"symbol": symbol, "name": name, "url": url, "sha256": checksum}
        with ThreadPoolExecutor(6) as pool:
            for number, record in enumerate(pool.map(download, specs), 1):
                records.append(record)
                if number % 25 == 0: print(f"sources {number}/{len(specs)}", flush=True)
    result = {"cutoff": end.isoformat(), "archives": records, "spec": SPEC}
    _json(ROOT / "source.json", result); return result


def _read_archives(symbol: str, source: dict) -> pd.DataFrame:
    paths = [ROOT / "sources" / symbol / row["name"] for row in source["archives"] if row["symbol"] == symbol]
    if not paths: raise FileNotFoundError(f"no archives for {symbol}; run prepare")
    frames, names = [], ["time", "open", "high", "low", "close", "volume", "close_time",
                         "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    for path in paths:
        if _sha(path) != next(row["sha256"] for row in source["archives"] if row["name"] == path.name):
            raise RuntimeError(f"source checksum changed: {path.name}")
        with zipfile.ZipFile(path) as archive:
            frames.append(pd.read_csv(io.BytesIO(archive.read(archive.namelist()[0])), header=None,
                                      names=names, usecols=[0, 1, 2, 3, 4, 5, 7, 9]))
    frame = pd.concat(frames, ignore_index=True); raw_time = frame.pop("time").to_numpy(np.int64)
    frame.insert(0, "time", np.where(raw_time > 100_000_000_000_000, raw_time // 1000, raw_time))
    frame = frame.sort_values("time").reset_index(drop=True)
    if frame["time"].duplicated().any(): raise RuntimeError(f"duplicate minute in {symbol}")
    return frame


def feature_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    times, close = frame["time"].to_numpy(np.int64), frame["close"].to_numpy(float)
    previous_ok = np.r_[False, np.diff(times) == MINUTE_MS]; returns = np.full(len(frame), np.nan, np.float32)
    returns[previous_ok] = np.log(close[previous_ok] / close[np.flatnonzero(previous_ok) - 1])
    opened, volume = frame["open"].to_numpy(float), frame["volume"].to_numpy(float)
    features = np.column_stack([returns, ((frame["high"].to_numpy(float) - frame["low"].to_numpy(float)) / opened),
        np.log1p(frame["quote_volume"].to_numpy(float)), np.divide(2 * frame["taker_buy_base"].to_numpy(float) - volume,
        volume, out=np.zeros(len(frame)), where=volume > 0)]).astype(np.float32)
    return times, features


def build_arrays(frames: dict[str, pd.DataFrame], cutoff: str) -> dict[str, np.ndarray]:
    cutoff_ms, times, features = int(_utc(cutoff).timestamp() * 1000), {}, {}
    for symbol in SYMBOLS: times[symbol], features[symbol] = feature_arrays(frames[symbol])
    common = times[SYMBOLS[0]]
    for symbol in SYMBOLS[1:]: common = np.intersect1d(common, times[symbol], assume_unique=True)
    aligned = np.column_stack([features[s][np.searchsorted(times[s], common)] for s in SYMBOLS]).astype(np.float32)
    btc = frames["BTCUSDT"]; bt = btc["time"].to_numpy(np.int64)
    bo, bc, result = btc["open"].to_numpy(float), btc["close"].to_numpy(float), ([], [], [], [])
    xs, anchors, ys, ends = result
    for t in common[common % (60 * MINUTE_MS) == 0]:
        end = int(np.searchsorted(common, t)); support = common[max(0, end - LOOKBACK - 1):end]
        if t + HORIZON * MINUTE_MS > cutoff_ms or len(support) != LOOKBACK + 1: continue
        if support[0] != t - (LOOKBACK + 1) * MINUTE_MS or support[-1] != t - MINUTE_MS: continue
        j = int(np.searchsorted(bt, t)); k = j + HORIZON - 1
        if j >= len(bt) or k >= len(bt) or bt[j] != t or bt[k] != t + (HORIZON - 1) * MINUTE_MS: continue
        seq = aligned[end - LOOKBACK:end]
        if not np.isfinite(seq).all(): continue
        ret = math.log(bc[k] / bo[j]); xs.append(seq); anchors.append(t); ends.append(t + HORIZON * MINUTE_MS)
        ys.append(0 if ret <= -THRESHOLD else 2 if ret >= THRESHOLD else 1)
    return {"x": np.asarray(xs, np.float32), "anchor": np.asarray(anchors, np.int64),
            "y": np.asarray(ys, np.int64), "label_end": np.asarray(ends, np.int64)}


def build_dataset() -> dict:
    source = json.loads((ROOT / "source.json").read_text())
    data = build_arrays({symbol: _read_archives(symbol, source) for symbol in SYMBOLS}, source["cutoff"])
    folder = ROOT / "dataset"; folder.mkdir(parents=True, exist_ok=True)
    for key, value in data.items(): np.save(folder / f"{key}.npy", value, allow_pickle=False)
    summary = {"samples": len(data["y"]), "start": pd.to_datetime(data["anchor"].min(), unit="ms", utc=True),
        "end": pd.to_datetime(data["label_end"].max(), unit="ms", utc=True),
        "counts": np.bincount(data["y"], minlength=3).tolist(),
        "dataset_hash": _array_hash(data["x"], data["anchor"], data["y"], data["label_end"]),
        "source_hash": _sha(ROOT / "source.json")}
    _json(folder / "dataset.json", summary); return summary


def load_dataset() -> dict[str, np.ndarray]:
    return {key: np.load(ROOT / "dataset" / f"{key}.npy", mmap_mode="r")
            for key in ("x", "anchor", "y", "label_end")}


@dataclass(frozen=True)
class Fold:
    number: int; train: np.ndarray; validation: np.ndarray; protected_start: int; protected_end: int


def temporal_folds(anchor: np.ndarray, label_end: np.ndarray, cutoff: str) -> list[Fold]:
    end, folds = _utc(cutoff), []
    for number in range(3):
        v0 = end - pd.DateOffset(months=18 - 6 * number); v1 = v0 + pd.DateOffset(months=6)
        v0ms, v1ms = int(v0.timestamp() * 1000), int(v1.timestamp() * 1000)
        validation = np.flatnonzero((anchor >= v0ms) & (anchor < v1ms))
        if not len(validation): raise RuntimeError(f"empty validation fold {number + 1}")
        p0, p1 = int(anchor[validation].min() - LOOKBACK * MINUTE_MS), int(label_end[validation].max() + LOOKBACK * MINUTE_MS)
        support0, support1 = anchor - LOOKBACK * MINUTE_MS, label_end
        train = np.flatnonzero((anchor < v0ms) & ((support1 <= p0) | (support0 >= p1)))
        if not len(train) or anchor[train].min() > int((v0 - pd.DateOffset(months=24)).timestamp() * 1000):
            raise RuntimeError("less than 24 months of initial training")
        folds.append(Fold(number + 1, train, validation, p0, p1))
    return folds


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x); mean = values.mean(axis=(0, 1), dtype=np.float64)
    std = values.std(axis=(0, 1), dtype=np.float64); std[std == 0] = 1
    return mean.astype(np.float32), std.astype(np.float32)


def prior_probabilities(y: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    prior = np.bincount(y, minlength=3).astype(float); prior /= prior.sum()
    return prior, np.tile(prior, (size, 1))


def fit_logistic(x, y, mean, std):
    if set(np.unique(y)) != {0, 1, 2}: raise RuntimeError("training split does not contain all classes")
    return LogisticRegression(**SPEC["logistic"]).fit(((np.asarray(x) - mean) / std).reshape(len(x), -1), y)


def logistic_predict(model, x, mean, std) -> np.ndarray:
    return model.predict_proba(((np.asarray(x) - mean) / std).reshape(len(x), -1))


def _torch():
    try: import torch; from torch import nn
    except ModuleNotFoundError as exc: raise RuntimeError("install research/requirements.txt") from exc
    return torch, nn


def make_transformer():
    torch, nn = _torch()
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); d = 32; self.project = nn.Linear(12, d)
            position = torch.arange(LOOKBACK).unsqueeze(1); scale = torch.exp(torch.arange(0, d, 2) * (-math.log(10_000) / d))
            pe = torch.zeros(LOOKBACK, d); pe[:, 0::2], pe[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
            self.register_buffer("position", pe); layer = nn.TransformerEncoderLayer(d, 4, 64, 0.1, batch_first=True, norm_first=True)
            self.encoder, self.head = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False), nn.Linear(d, 3)
        def forward(self, value): return self.head(self.encoder(self.project(value) + self.position).mean(dim=1))
    return Model()


def set_seed(seed: int, deterministic: bool = True) -> None:
    torch, _ = _torch(); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(deterministic)


def train_transformer(x, y, mean, std, seed: int, device: str, batch: int = 256, epochs: int = 10):
    torch, nn = _torch(); set_seed(seed, device == "cpu"); model = make_transformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    for _ in range(epochs):
        model.train()
        for start in range(0, len(x), batch):
            xb = ((np.asarray(x[start:start + batch]) - mean) / std).astype(np.float32)
            xb, yb = torch.from_numpy(xb).to(device), torch.from_numpy(np.asarray(y[start:start + batch])).to(device)
            optimizer.zero_grad(); loss = nn.functional.cross_entropy(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    return model


def transformer_predict(model, x, mean, std, device: str, batch: int = 256) -> np.ndarray:
    torch, _ = _torch(); model.eval(); out = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = ((np.asarray(x[start:start + batch]) - mean) / std).astype(np.float32)
            out.append(torch.softmax(model(torch.from_numpy(xb).to(device)), 1).cpu().numpy())
    return np.concatenate(out)


def sample_loss(y, probability) -> np.ndarray:
    return -np.log(np.clip(probability[np.arange(len(y)), y], 1e-15, 1))


def metrics(y, p) -> dict:
    predicted = p.argmax(1)
    return {"log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted))}


def _train_torch(x, y, mean, std, seed, device, batch, epochs=10):
    while True:
        try: return train_transformer(x, y, mean, std, seed, device, batch, epochs), batch
        except RuntimeError as exc:
            if "memory" not in str(exc).lower() or batch == 64: raise
            batch //= 2


def choose_backend(x, y, mean, std, workload: int) -> tuple[str, dict]:
    count = min(len(x), 512); started = time.monotonic()
    _train_torch(x[:count], y[:count], mean, std, 7, "cpu", 256, 1)
    elapsed = time.monotonic() - started; estimate = elapsed * workload / count
    torch, _ = _torch(); mps = bool(torch.backends.mps.is_available())
    return ("mps" if estimate > 7200 and mps else "cpu"), {"sample_seconds": elapsed, "samples": count,
        "estimated_total_seconds": estimate, "mps_available": mps, "validation_read": False}


def paired_bootstrap(frame: pd.DataFrame, draws: int = 10_000) -> list[float]:
    rng, estimates = np.random.default_rng(7), []
    groups = [[g["loss_transformer"].to_numpy() - g["loss_logistic"].to_numpy()
               for _, g in fold.groupby(fold["anchor"].astype("int64") // (7 * 86_400 * 10**9))]
              for _, fold in frame.groupby("fold", sort=True)]
    for _ in range(draws):
        values = [fold[index] for fold in groups for index in rng.integers(0, len(fold), len(fold))]
        estimates.append(float(np.concatenate(values).mean()))
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def run_validation_and_freeze() -> dict:
    bundle = ROOT / "bundle"
    if (bundle / "freeze.json").exists(): raise FileExistsError("final training already frozen")
    data, source = load_dataset(), json.loads((ROOT / "source.json").read_text())
    x, anchor, y, label_end = (data[k] for k in ("x", "anchor", "y", "label_end"))
    folds = temporal_folds(anchor, label_end, source["cutoff"]); validation_dir = ROOT / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    workload = 30 * (sum(len(f.train) for f in folds) + len(y)); mean0, std0 = fit_scaler(x[folds[0].train])
    device, benchmark = choose_backend(x[folds[0].train], y[folds[0].train], mean0, std0, workload)
    rows, fold_reports, batch = [], [], 256
    for fold in folds:
        tr, va = fold.train, fold.validation; mean, std = fit_scaler(x[tr]); prior, pp = prior_probabilities(y[tr], len(va))
        lp = logistic_predict(fit_logistic(x[tr], y[tr], mean, std), x[va], mean, std); seeds = []
        for seed in SEEDS:
            model, batch = _train_torch(x[tr], y[tr], mean, std, seed, device, batch)
            seeds.append(transformer_predict(model, x[va], mean, std, device, batch))
        ep = np.mean(seeds, axis=0); report = {"fold": fold.number, "train": len(tr), "validation": len(va),
            "prior": prior.tolist(), "prior_metrics": metrics(y[va], pp), "logistic_metrics": metrics(y[va], lp),
            "transformer_metrics": metrics(y[va], ep), "seed_metrics": {str(s): metrics(y[va], p) for s, p in zip(SEEDS, seeds)}}
        fold_reports.append(report); np.savez(validation_dir / f"fold-{fold.number}-ids.npz",
            train=anchor[tr], validation=anchor[va], protected=np.array([fold.protected_start, fold.protected_end]))
        named = [("prior", pp), ("logistic", lp), ("transformer", ep)]
        named_losses = {name: sample_loss(y[va], probabilities) for name, probabilities in named}
        for pos, idx in enumerate(va):
            row = {"sample_id": int(anchor[idx]), "anchor": pd.to_datetime(anchor[idx], unit="ms", utc=True),
                   "fold": fold.number, "y": int(y[idx])}
            for name, probabilities in named:
                for cls, value in enumerate(probabilities[pos]): row[f"p_{name}_{cls}"] = float(value)
                row[f"loss_{name}"] = float(named_losses[name][pos])
            for seed, probabilities in zip(SEEDS, seeds):
                for cls, value in enumerate(probabilities[pos]): row[f"p_seed_{seed}_{cls}"] = float(value)
            rows.append(row)
    oof = pd.DataFrame(rows).sort_values(["anchor", "sample_id"], kind="stable").reset_index(drop=True)
    oof.to_parquet(validation_dir / "oof.parquet", index=False)
    probs = {name: oof[[f"p_{name}_{i}" for i in range(3)]].to_numpy() for name in ("prior", "logistic", "transformer")}
    losses = {name: sample_loss(oof["y"].to_numpy(), value) for name, value in probs.items()}
    fold_signs = sum(item["transformer_metrics"]["log_loss"] < item["logistic_metrics"]["log_loss"] for item in fold_reports)
    ci = paired_bootstrap(oof); validation = {"backend": device, "batch": batch, "benchmark": benchmark,
        "samples": len(oof), "folds": fold_reports, "aggregate": {name: metrics(oof["y"], value) for name, value in probs.items()},
        "mean_loss_differences": {"transformer_minus_logistic": float((losses["transformer"] - losses["logistic"]).mean()),
            "logistic_minus_prior": float((losses["logistic"] - losses["prior"]).mean()),
            "transformer_minus_prior": float((losses["transformer"] - losses["prior"]).mean())},
        "paired_weekly_bootstrap_95": ci, "favorable_folds": fold_signs,
        "transformer_superior": bool((losses["transformer"] - losses["logistic"]).mean() < 0 and ci[1] < 0 and fold_signs >= 2)}
    _json(validation_dir / "report.json", validation)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    if status: raise RuntimeError("freeze requires a clean Git worktree")
    bundle.mkdir(parents=True, exist_ok=False); cutoff_ms = int(_utc(source["cutoff"]).timestamp() * 1000)
    final = np.flatnonzero((anchor < cutoff_ms) & (label_end <= cutoff_ms)); np.save(bundle / "sample_ids.npy", anchor[final])
    mean, std = fit_scaler(x[final]); np.savez(bundle / "scaler.npz", mean=mean, std=std)
    prior, _ = prior_probabilities(y[final], 0); np.save(bundle / "class_prior.npy", prior)
    with (bundle / "logistic.pkl").open("wb") as stream: pickle.dump(fit_logistic(x[final], y[final], mean, std), stream)
    torch, _ = _torch()
    for seed in SEEDS:
        model, batch = _train_torch(x[final], y[final], mean, std, seed, device, batch)
        torch.save(model.state_dict(), bundle / f"transformer-{seed}.pt")
    for source_path, name in [(ROOT / "source.json", "source.json"), (ROOT / "dataset" / "dataset.json", "dataset.json")]:
        (bundle / name).write_bytes(source_path.read_bytes())
    for fold in folds:
        source_path = validation_dir / f"fold-{fold.number}-ids.npz"
        (bundle / source_path.name).write_bytes(source_path.read_bytes())
    files = sorted(path for path in bundle.iterdir() if path.is_file()); now = datetime.now(timezone.utc)
    t0 = pd.Timestamp(now).ceil("D"); t1 = t0 + pd.Timedelta(days=180)
    freeze = {"created": now.isoformat(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "spec": SPEC, "source_cutoff": source["cutoff"], "dataset_hash": json.loads((bundle / "dataset.json").read_text())["dataset_hash"],
        "final_samples": len(final), "sample_ids_hash": _sha(bundle / "sample_ids.npy"), "backend": device, "batch": batch,
        "versions": {name: metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "pyarrow", "torch")},
        "files": {path.name: _sha(path) for path in files}, "validation_oof_hash": _sha(validation_dir / "oof.parquet"),
        "validation_report_hash": _sha(validation_dir / "report.json"), "t0": t0.isoformat(), "t1": t1.isoformat(), "final_test": "pending"}
    _json(bundle / "freeze.json", freeze, exclusive=True); return {"validation": validation, "freeze": freeze}


def final_test() -> dict:
    bundle, output, predictions = ROOT / "bundle", ROOT / "final-test.json", ROOT / "final-test.parquet"
    if output.exists() or predictions.exists(): raise FileExistsError("final test was already opened")
    freeze = json.loads((bundle / "freeze.json").read_text())
    for name, digest in freeze["files"].items():
        if _sha(bundle / name) != digest: raise RuntimeError(f"frozen file changed: {name}")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != freeze["git_commit"]: raise RuntimeError("Git commit differs from freeze")
    data, source = load_dataset(), json.loads((ROOT / "source.json").read_text()); t0, t1 = _utc(freeze["t0"]), _utc(freeze["t1"])
    if _utc(source["cutoff"]) < t1 + pd.Timedelta(minutes=HORIZON): raise RuntimeError("holdout is not complete")
    ids = np.flatnonzero((data["anchor"] >= int(t0.timestamp() * 1000)) & (data["anchor"] < int(t1.timestamp() * 1000)) &
                         (data["label_end"] <= int(t1.timestamp() * 1000)))
    scaler = np.load(bundle / "scaler.npz"); mean, std = scaler["mean"], scaler["std"]
    prior = np.load(bundle / "class_prior.npy"); pp = np.tile(prior, (len(ids), 1))
    with (bundle / "logistic.pkl").open("rb") as stream: logistic = pickle.load(stream)
    lp = logistic_predict(logistic, data["x"][ids], mean, std); torch, _ = _torch(); seeds = []
    for seed in SEEDS:
        model = make_transformer(); model.load_state_dict(torch.load(bundle / f"transformer-{seed}.pt", map_location=freeze["backend"], weights_only=True))
        model.to(freeze["backend"]); seeds.append(transformer_predict(model, data["x"][ids], mean, std, freeze["backend"], freeze["batch"]))
    ep = np.mean(seeds, axis=0); table = pd.DataFrame({"sample_id": np.asarray(data["anchor"][ids]),
        "anchor": pd.to_datetime(data["anchor"][ids], unit="ms", utc=True), "y": np.asarray(data["y"][ids])})
    for name, probability in [("prior", pp), ("logistic", lp), ("transformer", ep)]:
        for cls in range(3): table[f"p_{name}_{cls}"] = probability[:, cls]
    for seed, probability in zip(SEEDS, seeds):
        for cls in range(3): table[f"p_seed_{seed}_{cls}"] = probability[:, cls]
    table.to_parquet(predictions, index=False)
    result = {"opened": datetime.now(timezone.utc).isoformat(), "samples": len(ids),
        "sample_ids_hash": _array_hash(np.asarray(data["anchor"][ids])), "prior": metrics(data["y"][ids], pp),
        "logistic": metrics(data["y"][ids], lp), "transformer": metrics(data["y"][ids], ep),
        "seed_metrics": {str(s): metrics(data["y"][ids], p) for s, p in zip(SEEDS, seeds)},
        "predictions_hash": _sha(predictions)}
    _json(output, result, exclusive=True); return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare"); prepare.add_argument("--cutoff", required=True)
    for command in ("build", "run", "final"): sub.add_parser(command)
    args = parser.parse_args(); actions = {"prepare": lambda: prepare_sources(args.cutoff), "build": build_dataset,
                                          "run": run_validation_and_freeze, "final": final_test}
    print(json.dumps(actions[args.command](), indent=2, default=str)); return 0


if __name__ == "__main__": raise SystemExit(main())
