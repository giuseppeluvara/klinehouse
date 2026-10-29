#!/usr/bin/env python3
"""Simple OI/position correction after a separate field-specific timing audit."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import timestral_universal as shared  # noqa: E402
from research import timestral_volume_center as center  # noqa: E402
from research import timestral_lead_lag as price_model  # noqa: E402
from research.timestral_futures_flow import combine  # noqa: E402
from scripts import run_volume_center as previous  # noqa: E402
from scripts import run_lead_lag as prior_study  # noqa: E402
from scripts.run_futures_flow import (
    ROOT,
    digest,
    write,
    epoch,
    sources,
    prior_path,
    examples,
)  # noqa: E402
from scripts.run_flat_repair import outcome_path  # noqa: E402
from scripts.run_temporal_adaptation import paired_metrics  # noqa: E402

DEFAULT = ROOT / "reports/timestral-position-information-20261029"
ARMS = ("price_control", "positions")
PAIRS = (
    ("price_control", "baseline"),
    ("positions", "baseline"),
    ("positions", "price_control"),
)
CODE = tuple(
    dict.fromkeys(
        (
            "research/timestral_position_information.py",
            "scripts/run_position_information.py",
            "scripts/prepare_position_information.py",
            "scripts/verify_position_information.py",
            "scripts/audit_open_interest_sample.py",
            "tests/test_timestral_position_information.py",
            *prior_study.CODE,
        )
    )
)
PLAN = {
    "purpose": "Test new open-interest quantity and top-position-ratio archive information in a simple two-coefficient correction, after source and field-specific timestamp diagnostics. Do not transfer a flow alignment to stock fields or assume a uniformJune25change.",
    "status": "Sequential retrospective research on already explored Jan-Jul2024/2025/2026 BTC/ETH/SOL. Conditional exploratory replay, NOT historically availability-certified forecasting. Record empirical score success separately from verified success. Protected holdouts unchanged.",
    "timing": "Rawcreate_time retained. Fixed assumedlag15minutes; latest required label EXACT anchor-900sec. Thirteen unique valid labels spaced300sec cover exact1h. No shift fitted from outcomes and no retiming OI/positions according to taker flow. Publication/revision vintages and position measurement times are not certified; TIMING_CERTIFIED=False regardless of favorable scores. No timing/horizon/penalty sweep.",
    "sources": "New2730daily metrics ZIPs, Octpreviousyear throughJuly30 for three years and assets, officialSHA256; July31 files excluded to avoid plausibleprotectedAug1snapshots. Preserve rawlabel/fileday/row identity and duplicates; duplicates all invalid, offgrid invalid, OIqty andpositionratio must be finitepositive. Other unused fields can be missing. No forward filling. Source timestamp audit and independent OI probe precede registration and scoring.",
    "features": "Positions arm feature1=clip(current normalized ownspot60minreturn,-3,3)*[logOIqty(latest)-logOIqty(latest-3600)]. Feature2=imbalance(latest)-imbalance(latest-3600), imbalance=(R-1)/(R+1), R=archive sum_toptrader_long_short_ratio. Rawclip+-8. OIquantity, not notionalvalue; archive field semantics are documented as a proxy where historical mapping remains unresolved. Current spotreturn uses the frozen previous study definition with candleopenanchor-120sec and prior60return RMS excludinglatestreturn.",
    "control": "Matched price_control uses normalized ownspotreturns5and60minutes, same two coefficients and same training membership. Baseline retains exactly the monthly uncertainty-shrunk uniform-center coefficient and mixedflat=(cross/probabilityflat+pastfrequencyflat)/2. Both corrections keep flat unchanged. Every original anchor remains scored; missingcontext in anyasset means baselinefallback forbotharms onallassets.",
    "fit": "Frozen lead-lag offsetfit/predict functions, fixedridge.25, nointercept orassetID, monthly90daymaturednonflat pooled3asset training, equalUTCdayweights, trainweightedRMS floor1e-8 andclip3;42fits. Identical training mask oldfuturesavailability ANDnewcontextavailability. >=80observedtrainingdays required; failvisibly if inadequate, never disguise nofit as a newmodelpass. Baselineoffset uses current monthlypast-trained coefficient onpast rows, not an out-of-fold stack.",
    "evaluation": "All550362 unchanged forecastanchors, same5minspottargetanchor+60to+360. Bothcorrections plusbaseline originalgates. Positions vsbaseline andmatchedpricecontrol primary; pricecontrolvsbaseline diagnostic. Full3classBrier/logloss and conditionalmovingbinaryBrier/logloss/accuracy(tiesup); fixedbaseline52percentselectedscoresdiagnostic. No future-flat oracle for operational selection.",
    "uncertainty": "12000 paired7daymovingblocksseed271032; quantiles1/1080and1079/1080, conservative27comparisoncorrection relative18primarycohortpairs perendpoint/scope. Not jointlycorrected acrossendpoints/priorsequentialstudies. No bootstraprefit. Gate requiresALLmetricgainlowerbounds>0 AND>=5positivemonths foreachmetric.",
    "success": "Empiricalstableimprovement requiresall9positionsoriginalgates plusall18full andall18directiongates versusbothcontrols. Verifiedstableimprovement also requires verified historicalfieldtiming/publication; that remainsFALSE in thisconditionalstudy. Neverpromote matchedcontrol, retune lag, omitfailedcohorts or callunknownavailabilityverified afterscores.",
    "protected": "No market source/outcome from2026-08-01 onward. Reject any rawlabel withplausiblecontenttime label+300>=boundary beforefeatureconstruction. No orders/credentials/installations; previousfrozenartifacts anduserfiles preserved. Source downloadsunderregisteredstoragebudget.",
    "references": [
        "https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data",
        "https://github.com/binance/binance-public-data/issues/491",
        "https://github.com/binance/binance-public-data/issues/513",
        "https://docs.tardis.dev/historical-data-details/binance-futures",
    ],
}


def input_paths(root):
    paths = [previous.DEFAULT / "fits.json", previous.FUTURES / "sources.json"]
    paths += [
        root / name
        for name in (
            "source-protocol.json",
            "sources.json",
            "timestamp-audit.json",
            "oi-timestamp-audit.json",
        )
    ]
    for year in (2024, 2025, 2026):
        paths.extend(sources(year).values())
        for symbol in shared.SYMBOLS:
            paths.extend(
                [
                    prior_path(year, symbol),
                    outcome_path(year, symbol),
                    previous.FUTURES / f"{year}-{symbol}-futures.npz",
                    previous.DEFAULT / f"{year}-{symbol}-predictions.npz",
                    root / f"{year}-{symbol}-metrics.npz",
                ]
            )
    return paths


def register(root):
    audit = json.loads((root / "timestamp-audit.json").read_text())
    oi = json.loads((root / "oi-timestamp-audit.json").read_text())
    if (
        audit["historical_publication_verified"]
        or audit["observation_time_verified_for_positions"]
    ):
        raise ValueError("conditional protocol must not overstate source availability")
    if oi.get("protected_data_opened") is not False:
        raise ValueError("completed protected-boundary OI audit required")
    write(
        root / "protocol.json",
        dict(
            at=datetime.now(timezone.utc).isoformat(),
            plan=PLAN,
            code={p: digest(ROOT / p) for p in CODE},
            inputs={str(p.relative_to(ROOT)): digest(p) for p in input_paths(root)},
        ),
    )
    write(root / "protocol.sha256.json", dict(sha256=digest(root / "protocol.json")))


def run(root):
    from research import timestral_position_information as model

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
    oldfits = json.loads((previous.DEFAULT / "fits.json").read_text())
    fits, results = {}, {}
    for year in (2024, 2025, 2026):
        sample = examples(previous.FUTURES, year)
        t, end, y = sample["anchor"], sample["end"], sample["y"]
        spot = previous.spot_data(year)
        signals, _ = center.signals(sample["x"], center.centers(spot, t))
        zs = signals["uniform"]
        price, positions, available = [], [], []
        for symbol in shared.SYMBOLS:
            price_features, _, price_valid = price_model.features(
                spot[symbol], spot[symbol], t
            )
            stock_features, stock_valid = model.metrics_features(
                dict(np.load(root / f"{year}-{symbol}-metrics.npz")),
                t,
                price_features[:, 1],
            )
            price.append(price_features)
            positions.append(stock_features)
            available.append(price_valid & stock_valid)
        valid = np.all(available, axis=0)
        features = dict(price_control=np.stack(price), positions=np.stack(positions))
        q = {name: [] for name in (*ARMS, "baseline")}
        indices = []
        for month in range(1, 8):
            cutoff = epoch(year, month)
            train = (
                (t >= cutoff - 90 * 86400)
                & (end <= cutoff)
                & sample["available"]
                & valid
            )
            test = (t >= cutoff) & (t < epoch(year, month + 1))
            labels = y[:, train].reshape(-1)
            moving = labels != 1
            yy = (labels[moving] == 2).astype(int)
            tt, ee = np.tile(t[train], 3)[moving], np.tile(end[train], 3)[moving]
            ids = np.concatenate(
                [np.column_stack((np.full(train.sum(), a), t[train])) for a in range(3)]
            ).astype("<i8")[moving]
            baseline = oldfits[f"{year}-{month}-uniform"]["adjusted"]
            offset = baseline["beta"] * zs
            q["baseline"].append(expit(offset[:, test]))
            for name in ARMS:
                state = model.offsetfit(
                    features[name][:, train].reshape(-1, 2)[moving],
                    yy,
                    offset[:, train].reshape(-1)[moving],
                    tt,
                    ee,
                    cutoff,
                )
                state["training_ids_sha256"] = hashlib.sha256(ids.tobytes()).hexdigest()
                state["training_labels_sha256"] = hashlib.sha256(
                    yy.astype("<i8").tobytes()
                ).hexdigest()
                state["baseline_beta"] = baseline["beta"]
                fits[f"{year}-{month}-{name}"] = state
                pred = np.stack(
                    [
                        model.predict(
                            state, features[name][a, test], offset[a, test], t[test]
                        )
                        for a in range(3)
                    ]
                )
                pred[:, ~valid[test]] = q["baseline"][-1][:, ~valid[test]]
                q[name].append(pred)
            indices.append(np.flatnonzero(test))
            print("fitted", year, month, flush=True)
        ids = np.concatenate(indices)
        q = {k: np.concatenate(v, axis=1) for k, v in q.items()}
        for a, s in enumerate(shared.SYMBOLS):
            key = f"{year}-{s}"
            prior = dict(np.load(prior_path(year, s)))
            old = dict(np.load(previous.DEFAULT / f"{key}-predictions.npz"))
            outcome = dict(np.load(outcome_path(year, s)))
            anchors, labels = t[ids], y[a, ids]
            if (
                not np.array_equal(anchors, prior["anchor"])
                or not np.array_equal(anchors, old["anchor"])
                or not np.array_equal(anchors, outcome["anchor"])
                or not np.array_equal(labels, outcome["label"])
                or outcome["end"].max() > shared.BOUNDARY
            ):
                raise ValueError("changed evaluation membership/labels")
            np.testing.assert_allclose(
                q["baseline"][a], old["uniform_up"], atol=1e-14, rtol=0
            )
            raw = prior["cross" if year == 2026 else "probability"]
            flat = (raw[:, 1] + prior["frequency"][:, 1]) / 2
            p = {name: combine(qq[a], flat) for name, qq in q.items()}
            selected = np.maximum(p["baseline"][:, 0], p["baseline"][:, 2]) >= 0.52
            with (root / f"{key}-predictions.npz").open("xb") as f:
                np.savez_compressed(
                    f,
                    anchor=anchors,
                    available=valid[ids],
                    **{name: pp for name, pp in p.items()},
                )
            results[key] = dict(
                n=len(ids),
                missing_contexts=int((~valid[ids]).sum()),
                original={},
                comparisons={},
            )
            for name, pp in p.items():
                results[key]["original"][name] = shared.evaluate(
                    labels, pp, prior["frequency"], prior["prior_reversal"], anchors
                )
            for name, control in PAIRS:
                results[key]["comparisons"][f"{name}_vs_{control}"] = dict(
                    full=paired_metrics(
                        labels, p[name], p[control], anchors, np.ones(len(ids), bool)
                    ),
                    selected=paired_metrics(
                        labels, p[name], p[control], anchors, selected
                    ),
                    direction=paired_metrics(
                        labels,
                        p[name],
                        p[control],
                        anchors,
                        np.ones(len(ids), bool),
                        direction=True,
                    ),
                )
            print("scored", key, flush=True)
    write(root / "fits.json", fits)
    write(root / "results.json", results)
    original = {
        name: sum(r["original"][name]["gate"] for r in results.values())
        for name in (*ARMS, "baseline")
    }
    counts = {
        pair: {
            scope: sum(
                r["comparisons"][pair][scope]["improvement_gate"]
                for r in results.values()
            )
            for scope in ("full", "selected", "direction")
        }
        for pair in results[next(iter(results))]["comparisons"]
    }
    success = original["positions"] == 9 and all(
        counts[f"positions_vs_{control}"][scope] == 9
        for control in ("baseline", "price_control")
        for scope in ("full", "direction")
    )
    write(
        root / "summary.json",
        dict(
            original_passes=original,
            paired_gates=counts,
            empirical_stable_improvement=success,
            timing_certified=False,
            stable_improvement=False,
            anchors=sum(r["n"] for r in results.values()),
        ),
    )
    write(
        root / "manifest.json",
        {p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run"))
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.command](args.root.resolve())
