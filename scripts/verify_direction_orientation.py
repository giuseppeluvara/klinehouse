#!/usr/bin/env python3
"""Independently audit saved orientation forecasts and their source membership.

No forecaster, runner, learner or production scoring module is imported.  The
posterior is reconstructed from receipt-sorted likelihood-ratio prefix sums;
this differs from the learner's pending-list updates.  Saved live observations
are an explicitly post-hoc replay, never new prospective emissions.
"""

import argparse
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "reports/direction-orientation-20261030-v3"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
ARMS = ("baseline", "price_control", "positions", "neutral", "mirrored", "adaptive")
BOUNDARY = 1785542400
SOURCE_KEYS = {"anchor", "issue", "label_at", "label", *ARMS[:3]}
OUTPUT_KEYS = SOURCE_KEYS | {
    "neutral",
    "mirrored",
    "adaptive",
    "weights",
    "feedback_count",
}
ATOL = 2e-9


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def object_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def probabilities(value, n):
    p = np.asarray(value, dtype=float)
    require(
        p.shape == (n, 3)
        and np.isfinite(p).all()
        and np.all((p > 0) & (p < 1))
        and np.allclose(p.sum(axis=1), 1, atol=1e-9, rtol=0),
        "invalid probability matrix",
    )
    return p


def independent_orientation(probability, issue, label_at, label):
    """Reconstruct each prior using all available same-lane likelihood ratios."""
    issue = np.asarray(issue, dtype=float)
    label_at = np.asarray(label_at, dtype=float)
    label = np.asarray(label)
    n = len(issue)
    p = probabilities(probability, n)
    known = label >= 0
    require(
        n > 0
        and issue.shape == label_at.shape == label.shape == (n,)
        and np.isfinite(issue).all()
        and np.all(np.diff(issue) > 0)
        and np.isin(label, [-1, 0, 1, 2]).all()
        and np.isfinite(label_at[known]).all()
        and np.all(label_at[known] > issue[known])
        and np.isnan(label_at[~known]).all(),
        "invalid outcome availability chronology",
    )
    q = p[:, 2] / (p[:, 0] + p[:, 2])
    weights = np.full((n, 3), 1 / 3)
    counts = np.zeros(n, dtype=np.int64)
    moving = known & (label != 1)
    for lane in (0, 1):
        emission_ids = np.arange(lane, n, 2)
        feedback_ids = emission_ids[moving[emission_ids]]
        feedback_ids = feedback_ids[np.argsort(label_at[feedback_ids], kind="stable")]
        realized = np.where(
            label[feedback_ids] == 2, q[feedback_ids], 1 - q[feedback_ids]
        )
        # Divide each expert likelihood by neutral's 1/2 before accumulation.
        # This avoids subtracting two large common log-loss totals afterwards.
        ratios = np.column_stack(
            (np.log(2 * realized), np.log(2 * (1 - realized)), np.zeros(len(realized)))
        )
        prefix = np.vstack((np.zeros((1, 3)), np.cumsum(ratios, axis=0)))
        available = np.searchsorted(
            label_at[feedback_ids], issue[emission_ids], side="right"
        )
        state = prefix[available]
        mass = np.exp(state - np.max(state, axis=1, keepdims=True))
        weights[emission_ids] = mass / mass.sum(axis=1, keepdims=True)
        counts[emission_ids] = available
    adaptive_q = 0.5 + (weights[:, 0] - weights[:, 1]) * (q - 0.5)
    adaptive = np.column_stack(
        ((1 - p[:, 1]) * (1 - adaptive_q), p[:, 1], (1 - p[:, 1]) * adaptive_q)
    )
    neutral = np.column_stack(((1 - p[:, 1]) / 2, p[:, 1], (1 - p[:, 1]) / 2))
    return dict(
        adaptive=adaptive,
        neutral=neutral,
        mirrored=p[:, [2, 1, 0]],
        weights=weights,
        feedback_count=counts,
    )


def audit_prediction_data(saved, source):
    """Check exact source rows before checking derived probabilities and weights."""
    require(set(saved) == OUTPUT_KEYS, "saved array schema changed")
    require(set(source) == SOURCE_KEYS, "source array schema changed")
    for name in SOURCE_KEYS:
        require(
            np.array_equal(saved[name], source[name], equal_nan=True),
            f"source differs: {name}",
        )
    anchor = np.asarray(saved["anchor"])
    n = len(anchor)
    require(
        anchor.shape == (n,)
        and np.issubdtype(anchor.dtype, np.integer)
        and np.all(anchor % 300 == 0)
        and np.all(np.diff(anchor) > 0)
        and np.all(saved["issue"] >= anchor),
        "invalid saved anchors or issue times",
    )
    known = saved["label"] >= 0
    require(
        np.all(saved["label_at"][known] >= anchor[known] + 360),
        "outcome precedes target end",
    )
    for arm in ARMS:
        probabilities(saved[arm], n)
        require(
            np.array_equal(saved[arm][:, 1], saved["baseline"][:, 1]),
            "flat probability changed",
        )
    expected = independent_orientation(
        saved["baseline"], saved["issue"], saved["label_at"], saved["label"]
    )
    errors = {}
    for name in ("adaptive", "neutral", "mirrored", "weights"):
        actual = np.asarray(saved[name])
        require(actual.shape == expected[name].shape, f"wrong shape: {name}")
        require(np.isfinite(actual).all(), f"nonfinite values: {name}")
        error = float(np.max(np.abs(actual - expected[name])))
        require(error <= ATOL, f"independent {name} reconstruction differs: {error}")
        errors[name] = error
    require(
        np.array_equal(saved["feedback_count"], expected["feedback_count"]),
        "feedback count differs",
    )
    require(
        np.issubdtype(saved["feedback_count"].dtype, np.integer),
        "feedback count must be integral",
    )
    return errors


def independent_scores(p, labels, mask):
    y = np.asarray(labels)[mask & (labels >= 0)]
    values = p[mask & (labels >= 0)]
    moving = y != 1
    result = dict(
        n=int(len(y)), moving_n=int(moving.sum()), flat_n=int((~moving).sum())
    )
    if not len(y):
        return result
    q = values[:, 2] / (values[:, 0] + values[:, 2])
    target = y[moving] == 2
    correct = (q[moving] >= 0.5) == target
    result.update(
        brier=float(
            math.fsum(
                np.sum((values - (np.arange(3)[None, :] == y[:, None])) ** 2, axis=1)
            )
            / len(y)
        ),
        log_loss=float(math.fsum(-np.log(values[np.arange(len(y)), y])) / len(y)),
        direction_correct=int(correct.sum()),
        direction_accuracy=float(correct.sum() / len(correct))
        if len(correct)
        else None,
        direction_brier=float(math.fsum((q[moving] - target) ** 2) / len(correct))
        if len(correct)
        else None,
        direction_log_loss=float(
            math.fsum(-np.log(np.where(target, q[moving], 1 - q[moving])))
            / len(correct)
        )
        if len(correct)
        else None,
    )
    return result


def independent_report(data):
    labels = data["label"]
    scored = labels >= 0
    baseline_selected = scored & (np.max(data["baseline"][:, [0, 2]], axis=1) >= 0.52)
    models = {}
    for arm in ARMS:
        p = data[arm]
        selected = scored & (np.max(p[:, [0, 2]], axis=1) >= 0.52)
        own = independent_scores(p, labels, selected)
        if own["n"]:
            own["original_accuracy_flat_errors"] = float(
                np.count_nonzero(np.argmax(p[selected], axis=1) == labels[selected])
                / own["n"]
            )
        own["coverage"] = float(own["n"] / np.count_nonzero(scored))
        models[arm] = dict(
            all=independent_scores(p, labels, scored),
            own_selection=own,
            fixed_baseline_selection=independent_scores(p, labels, baseline_selected),
        )
    gains = {}
    candidate = models["adaptive"]["all"]
    for arm in ("baseline", "price_control", "positions", "neutral"):
        control = models[arm]["all"]
        gains[arm] = dict(
            accuracy_change=candidate["direction_accuracy"]
            - control["direction_accuracy"],
            brier_gain=control["brier"] - candidate["brier"],
            logloss_gain=control["log_loss"] - candidate["log_loss"],
        )
    return dict(
        scored=int(scored.sum()),
        pending=int((~scored).sum()),
        models=models,
        adaptive_gains=gains,
        last_weights=data["weights"][-1].tolist(),
        strict_point_improvement=all(
            all(value > 0 for value in gains[control].values())
            for control in ("baseline", "price_control")
        ),
    )


def compare_report(actual, expected, path="report"):
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and actual.keys() == expected.keys(),
            f"report schema differs: {path}",
        )
        for key in expected:
            compare_report(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        require(
            isinstance(actual, list) and len(actual) == len(expected),
            f"report list differs: {path}",
        )
        for index, value in enumerate(expected):
            compare_report(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, (bool, int)) or expected is None:
        require(
            actual == expected and type(actual) is type(expected),
            f"report value differs: {path}",
        )
    else:
        require(
            isinstance(actual, (int, float))
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-12),
            f"report number differs: {path}",
        )


def historical_source(prediction_path, outcome_path, year):
    lo = int(np.datetime64(f"{year}-01-01", "s").astype(np.int64))
    hi = int(np.datetime64(f"{year}-08-01", "s").astype(np.int64))
    with (
        np.load(prediction_path, allow_pickle=False) as p,
        np.load(outcome_path, allow_pickle=False) as y,
    ):
        anchor, outcome_anchor, end = p["anchor"], y["anchor"], y["end"]
        require(
            np.array_equal(anchor, outcome_anchor) and np.all(end == anchor + 360),
            "historical target membership differs",
        )
        # Inspect timestamps before opening any probability or outcome array.
        require(
            np.all((anchor >= lo) & (anchor < hi)) and np.all(end < BOUNDARY),
            "protected or out-of-period historical target",
        )
        return dict(
            anchor=anchor,
            issue=anchor.astype(float),
            label_at=end.astype(float),
            label=y["label"],
            **{arm: p[arm] for arm in ARMS[:3]},
        )


def live_sources(checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text())
    audit = json.loads((checkpoint / "general-audit.json").read_text())
    require(
        config.get("require_commit_ack") is True,
        "live protocol does not require commit acknowledgements",
    )
    require(
        digest(checkpoint / "model.json") == config["snapshot_sha256"],
        "live snapshot changed",
    )
    forecasts, acks, outcomes, receipts = {}, {}, {}, {}
    with closing(
        sqlite3.connect(
            (checkpoint / "events.sqlite").resolve().as_uri() + "?mode=ro", uri=True
        )
    ) as c:
        c.row_factory = sqlite3.Row
        c.execute("BEGIN")
        require(
            c.execute("PRAGMA integrity_check").fetchone()[0] == "ok",
            "live checkpoint integrity failed",
        )
        meta = dict(c.execute("SELECT key,value FROM meta"))
        require(
            json.loads(meta["config"]) == config
            and meta["config_sha256"] == object_hash(config),
            "live checkpoint protocol differs",
        )
        previous, seq = meta["config_sha256"], 0
        for seq, event in enumerate(c.execute("SELECT * FROM events ORDER BY seq"), 1):
            payload = json.loads(event["payload"])
            require(
                event["seq"] == seq and event["prev_hash"] == previous,
                "live checkpoint chain is discontinuous",
            )
            require(
                event["payload"] == canonical(payload), "live payload is not canonical"
            )
            expected_hash = object_hash(
                dict(
                    seq=seq,
                    kind=event["kind"],
                    received_ns=event["received_ns"],
                    payload=payload,
                    prev_hash=previous,
                )
            )
            require(expected_hash == event["digest"], "live checkpoint digest differs")
            previous = expected_hash
            received = event["received_ns"] / 1e9
            require(
                abs(received - event["received_at"]) <= 1e-6
                and received >= config["start_at"],
                "live receipt time differs",
            )
            require(
                event["anchor"] == payload.get("anchor")
                and event["symbol"] == payload.get("symbol"),
                "live event index differs",
            )
            kind = event["kind"]
            if kind == "receipt":
                receipts[seq] = payload
                continue
            if kind not in ("forecast", "forecast_commit", "outcome"):
                continue
            key = payload["anchor"], payload["symbol"]
            anchor, symbol = key
            require(
                symbol in SYMBOLS
                and anchor % 300 == 0
                and config["start_at"] <= anchor
                and anchor + 360 <= config["end_at"],
                "live target bounds differ",
            )
            if kind == "forecast":
                require(
                    key not in forecasts
                    and anchor <= payload["issued_at"] <= anchor + 30
                    and payload["issued_at"] <= received < anchor + 60,
                    "invalid live forecast chronology",
                )
                require(
                    payload["snapshot_sha256"] == config["snapshot_sha256"],
                    "live forecast model changed",
                )
                forecasts[key] = payload, seq, expected_hash, received
            elif kind == "forecast_commit":
                require(
                    key in forecasts and key not in acks,
                    "invalid live acknowledgement membership",
                )
                _, original_seq, original_hash, original_time = forecasts[key]
                require(
                    payload["forecast_seq"] == original_seq
                    and payload["forecast_hash"] == original_hash,
                    "live acknowledgement references another forecast",
                )
                require(
                    original_time <= payload["committed_at"] <= received,
                    "live acknowledgement backdates its commit",
                )
                require(
                    type(payload["clock_healthy"]) is bool,
                    "live acknowledgement clock health is not boolean",
                )
                valid = bool(
                    payload["clock_healthy"] and payload["committed_at"] <= anchor + 30
                )
                require(
                    type(payload["eligible"]) is bool and payload["eligible"] == valid,
                    "live acknowledgement eligibility differs",
                )
                acks[key] = payload
            else:
                require(
                    key in forecasts
                    and key in acks
                    and acks[key]["eligible"]
                    and key not in outcomes,
                    "outcome lacks unique eligible live forecast",
                )
                require(
                    received >= anchor + 360
                    and payload["return_start"] == anchor + 60
                    and payload["return_end"] == anchor + 360,
                    "live outcome is early or horizon differs",
                )
                first, last = (
                    Decimal(payload["entry_price"]),
                    Decimal(payload["exit_price"]),
                )
                require(
                    first.is_finite() and last.is_finite() and first > 0 and last > 0,
                    "invalid live target prices",
                )
                require(
                    payload["label"]
                    == (2 if last > first else 0 if last < first else 1),
                    "live label differs from target prices",
                )
                for field, minute, price in (
                    ("entry_receipt_id", anchor, first),
                    ("exit_receipt_id", anchor + 300, last),
                ):
                    reference = payload[field]
                    require(
                        reference in receipts,
                        "live outcome cites future/missing source",
                    )
                    source = receipts[reference]
                    require(
                        source["source"] == "spot_klines"
                        and source["symbol"] == symbol
                        and any(
                            row[0] == minute * 1000 and Decimal(row[4]) == price
                            for row in source["body"]
                        ),
                        "live outcome price absent from cited receipt",
                    )
                outcomes[key] = payload, received
        require(
            audit.get("verified") is True
            and audit["event_count"] == seq
            and audit["chain_tip"] == previous,
            "checkpoint differs from retained general audit",
        )
    result = {}
    for symbol in SYMBOLS:
        keys = sorted(
            key
            for key in forecasts
            if key[1] == symbol and key in acks and acks[key]["eligible"]
        )
        require(keys, "live asset has no eligible emissions")
        result[symbol] = dict(
            anchor=np.array([key[0] for key in keys], dtype=np.int64),
            issue=np.array([forecasts[key][0]["issued_at"] for key in keys]),
            label_at=np.array(
                [outcomes[key][1] if key in outcomes else np.nan for key in keys]
            ),
            label=np.array(
                [outcomes[key][0]["label"] if key in outcomes else -1 for key in keys],
                dtype=np.int64,
            ),
            **{
                arm: np.array([forecasts[key][0]["models"][arm] for key in keys])
                for arm in ARMS[:3]
            },
        )
    return result


def verify(root=DEFAULT, *, workspace=ROOT):
    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    hashes = json.loads((root / "protocol.sha256.json").read_text())
    require(
        hashes == {"protocol.json": digest(root / "protocol.json")},
        "registered protocol hash differs",
    )
    checkpoint = workspace / "research/out/direction-diagnosis-20261030/live-checkpoint"
    base = workspace / "reports/timestral-position-information-20261029"
    old = workspace / "reports/timestral-universal-20261027"
    historical_paths = {}
    for year in (2024, 2025, 2026):
        for symbol in SYMBOLS:
            outcome = old / (
                f"{symbol}-outcomes.npz"
                if year == 2026
                else f"history/{year}-{symbol}-outcomes.npz"
            )
            historical_paths[f"{year}-{symbol}"] = (
                base / f"{year}-{symbol}-predictions.npz",
                outcome,
                year,
            )
    expected_inputs = {
        str(p.relative_to(workspace))
        for pair in historical_paths.values()
        for p in pair[:2]
    } | {
        str((checkpoint / name).relative_to(workspace))
        for name in ("events.sqlite", "config.json", "model.json", "general-audit.json")
    }
    require(
        set(protocol["inputs"]) == expected_inputs,
        "registered source allowlist changed",
    )
    require(
        set(protocol["code"])
        == {
            "research/direction_orientation.py",
            "scripts/run_direction_orientation.py",
            "tests/test_direction_orientation.py",
        },
        "registered code allowlist changed",
    )
    for section in ("code", "inputs"):
        for name, value in protocol[section].items():
            frozen = root / "frozen-source" / name
            path = (
                frozen if section == "code" and frozen.is_file() else workspace / name
            )
            require(digest(path) == value, f"registered {section} hash differs: {name}")
    results = json.loads((root / "results.json").read_text())
    cohorts = set(historical_paths) | {f"live-{s}" for s in SYMBOLS}
    require(set(results["cohorts"]) == cohorts, "cohort membership differs")
    manifest = json.loads((root / "manifest.json").read_text())
    expected_artifacts = {f"{key}-predictions.npz" for key in cohorts} | {
        "protocol.json",
        "protocol.sha256.json",
        "results.json",
    }
    require(
        set(manifest) == expected_artifacts, "execution manifest membership differs"
    )
    for name, value in manifest.items():
        require(
            digest(root / name) == value, f"execution artifact hash differs: {name}"
        )
    live = live_sources(checkpoint)
    evidence, maximum = (
        {},
        {name: 0.0 for name in ("weights", "adaptive", "neutral", "mirrored")},
    )
    for key in sorted(cohorts):
        source = (
            live[key[5:]]
            if key.startswith("live-")
            else historical_source(*historical_paths[key])
        )
        with np.load(root / f"{key}-predictions.npz", allow_pickle=False) as archive:
            saved = {name: archive[name] for name in archive.files}
        errors = audit_prediction_data(saved, source)
        for name, value in errors.items():
            maximum[name] = max(maximum[name], value)
        reconstructed = independent_report(saved)
        compare_report(results["cohorts"][key], reconstructed, key)
        evidence[key] = dict(
            emissions=len(saved["anchor"]),
            scored=reconstructed["scored"],
            pending=reconstructed["pending"],
            strict_point_improvement=reconstructed["strict_point_improvement"],
            exactly_neutral_adaptive=int(
                np.count_nonzero(saved["adaptive"][:, 0] == saved["adaptive"][:, 2])
            ),
        )
    passes = sum(
        value["strict_point_improvement"]
        for key, value in evidence.items()
        if not key.startswith("live-")
    )
    require(
        type(results["historical_pass_count"]) is int
        and results["historical_pass_count"] == passes,
        "historical pass count differs",
    )
    require(
        results["historical_stable_point_improvement"] is (passes == 9),
        "historical success flag differs",
    )
    require(
        results["live_replay_is_prospective_evidence"] is False,
        "live replay mislabeled as prospective evidence",
    )
    return dict(
        verified=True,
        verified_at=datetime.now(timezone.utc).isoformat(),
        scope="Independent frozen-source membership, live acknowledgement/receipt chronology, likelihood-ratio posterior reconstruction and all saved scores/selections. No learner imports or inference rerun.",
        protocol_sha256=digest(root / "protocol.json"),
        manifest_sha256=digest(root / "manifest.json"),
        verifier_sha256=digest(Path(__file__)),
        historical_cohorts=9,
        live_replay_cohorts=3,
        historical_pass_count=passes,
        historical_stable_point_improvement=passes == 9,
        live_replay_is_prospective_evidence=False,
        maximum_absolute_reconstruction_errors=maximum,
        probability_tolerance=ATOL,
        cohorts=evidence,
        bound_claimed=False,
        protected_historical_outcomes_opened=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    output = args.output or args.root / "verification.json"
    with output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
