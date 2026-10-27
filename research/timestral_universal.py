"""Shared, sign-symmetric multiclass direction models across three assets."""

import warnings

import numpy as np
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

BOUNDARY = 1785542400
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
ODD = tuple(range(7)) + tuple(range(13, 21))
FEATURES = (
    "return1",
    "return5",
    "return15",
    "return60",
    "flow1",
    "flow5",
    "return1_activity",
    "abs_return1",
    "log_trades",
    "zero_fraction60",
    "log_activity",
    "log_rms60",
    "abs_flow1",
    "other_return1",
    "other_return5",
    "other_return15",
    "other_return60",
    "other_lag1_return1",
    "other_lag1_return5",
    "signed_residual_square",
    "return1_cross_dispersion",
    "other_zero_fraction60",
    "abs_other_return1",
)


def examples(datasets):
    if tuple(datasets) != SYMBOLS:
        raise ValueError("ordered registered assets required")
    t = datasets[SYMBOLS[0]]["time"]
    if len(t) < 70 or t[-1] >= BOUNDARY or np.any(np.diff(t) != 60) or t[0] % 60:
        raise ValueError("complete, aligned, pre-protected minute source required")
    for d in datasets.values():
        if not np.array_equal(d["time"], t) or not all(
            np.isfinite(a).all() for a in d.values()
        ):
            raise ValueError("assets must share complete finite minute inputs")
        if (
            np.any(d["close"] <= 0)
            or np.any(d["volume"] < 0)
            or np.any(d["trades"] < 0)
            or np.any(d["taker"] < 0)
            or np.any(d["taker"] > d["volume"] + 1e-8)
        ):
            raise ValueError("invalid price or activity")
    i = np.arange(61, len(t) - 6)
    i = i[((t[i] + 60) // 60) % 5 == 0]
    own, lagged, targets = [], [], []
    for d in datasets.values():
        log = np.log(d["close"])
        r = np.r_[0.0, np.diff(log)]
        sq = np.r_[0.0, np.cumsum(r * r)]
        tr = np.r_[0.0, np.cumsum(d["trades"])]
        zero = np.r_[0, np.cumsum(r == 0)]
        vol = np.r_[0.0, np.cumsum(d["volume"])]
        buy = np.r_[0.0, np.cumsum(d["taker"])]
        scale = np.sqrt(np.maximum((sq[i] - sq[i - 60]) / 60, 1e-16))
        prev_scale = np.sqrt(np.maximum((sq[i - 1] - sq[i - 61]) / 60, 1e-16))
        returns = np.column_stack(
            [(log[i] - log[i - h]) / scale / np.sqrt(h) for h in (1, 5, 15, 60)]
        )
        returns = np.clip(returns, -8, 8)
        lag = np.column_stack(
            [(log[i - 1] - log[i - 1 - h]) / prev_scale / np.sqrt(h) for h in (1, 5)]
        )
        activity = np.log(
            np.maximum(
                d["trades"][i] / np.maximum((tr[i] - tr[i - 60]) / 60, 1.0), 1e-6
            )
        )
        v = d["volume"][i]
        f1 = np.divide(2 * d["taker"][i] - v, v, out=np.zeros(len(i)), where=v > 0)
        v5 = vol[i + 1] - vol[i - 4]
        f5 = np.divide(
            2 * (buy[i + 1] - buy[i - 4]) - v5, v5, out=np.zeros(len(i)), where=v5 > 0
        )
        own.append(
            np.column_stack(
                (
                    returns,
                    f1,
                    f5,
                    returns[:, 0] * np.clip(activity, -4, 4),
                    abs(returns[:, 0]),
                    np.log1p(d["trades"][i]),
                    (zero[i + 1] - zero[i - 59]) / 60,
                    activity,
                    np.log(scale),
                    abs(f1),
                )
            )
        )
        lagged.append(np.clip(lag, -8, 8))
        targets.append(np.sign(d["close"][i + 6] - d["close"][i + 1]).astype(int) + 1)
    x = []
    for a in range(3):
        others = [b for b in range(3) if b != a]
        market = np.mean([own[b][:, :4] for b in others], axis=0)
        lag = np.mean([lagged[b] for b in others], axis=0)
        residual = np.clip(own[a][:, 0] - market[:, 0], -8, 8)
        dispersion = np.std([own[b][:, 0] for b in others], axis=0)
        x.append(
            np.column_stack(
                (
                    own[a],
                    market,
                    lag,
                    residual * abs(residual) / 8,
                    own[a][:, 0] * dispersion,
                    np.mean([own[b][:, 9] for b in others], axis=0),
                    abs(market[:, 0]),
                )
            )
        )
    return {
        "anchor": t[i] + 60,
        "end": t[i + 6] + 60,
        "x": np.stack(x),
        "y": np.stack(targets),
    }


def augment(x, y):
    reflected = x.copy()
    reflected[:, [j for j in ODD if j < x.shape[1]]] *= -1
    return np.concatenate((x, reflected)), np.concatenate((y, 2 - y))


def fit(x, y, ends, cutoff):
    if len(y) < 100 or np.any(ends > cutoff) or not np.isfinite(x).all():
        raise ValueError("finite matured training data required")
    a, b = augment(x, y)
    mean = a.mean(axis=0)
    mean[[j for j in ODD if j < x.shape[1]]] = 0
    scale = np.maximum(a.std(axis=0), 1e-8)
    z = np.clip((a - mean) / scale, -8, 8)
    estimator = LogisticRegression(C=0.1, solver="lbfgs", max_iter=500, tol=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(z, b)
    if not np.array_equal(estimator.classes_, [0, 1, 2]):
        raise ValueError("all three classes required")
    return {
        "cutoff": int(cutoff),
        "n": len(y),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": estimator.coef_.tolist(),
        "intercept": estimator.intercept_.tolist(),
        "iterations": int(estimator.n_iter_[0]),
    }


def predict(fit, x, anchors):
    if (
        np.any(anchors < fit["cutoff"])
        or np.any(anchors + 360 > BOUNDARY)
        or not np.isfinite(x).all()
    ):
        raise ValueError("future fit, protected target or invalid input")
    z = np.clip((x - fit["mean"]) / fit["scale"], -8, 8)
    return softmax(z @ np.asarray(fit["coef"]).T + fit["intercept"], axis=1)


def evaluate(y, p, base, prior, anchors):
    selected = np.maximum(p[:, 0], p[:, 2]) >= 0.52
    target = np.eye(3)[y]
    losses = np.column_stack(
        [np.square(q - target).sum(axis=1) for q in (base, prior, p)]
    )
    correct = p.argmax(axis=1) == y
    result = {
        "n": len(y),
        "selected": int(selected.sum()),
        "coverage": float(selected.mean()),
        "all_brier": dict(
            zip(
                ("frequency", "prior_reversal", "shared_model"),
                map(float, losses.mean(axis=0)),
            )
        ),
    }
    if not selected.any():
        return {**result, "gate": False, "reason": "no selected forecasts"}
    result.update(
        accuracy=float(correct[selected].mean()),
        predicted_direction_counts={
            str(k): int(np.sum(selected & (p.argmax(axis=1) == k))) for k in (0, 2)
        },
        selected_brier=dict(
            zip(
                ("frequency", "prior_reversal", "shared_model"),
                map(float, losses[selected].mean(axis=0)),
            )
        ),
        log_loss=float(
            -np.log(p[selected][np.arange(selected.sum()), y[selected]]).mean()
        ),
        baseline_log_loss=float(
            -np.log(base[selected][np.arange(selected.sum()), y[selected]]).mean()
        ),
    )
    days, inv = np.unique(anchors // 86400, return_inverse=True)
    if np.any(np.diff(days) != 1):
        raise ValueError("noncontiguous evaluation days")
    counts = np.bincount(inv, weights=selected)
    sums = np.column_stack(
        [np.bincount(inv, weights=selected * losses[:, j]) for j in range(3)]
        + [np.bincount(inv, weights=selected * correct)]
    )
    starts = np.random.default_rng(271029).integers(
        0, len(days) - 6, size=(4000, int(np.ceil(len(days) / 7)))
    )
    ids = (starts[:, :, None] + np.arange(7)).reshape(4000, -1)[:, : len(days)]
    den = counts[ids].sum(axis=1)
    if np.any(den == 0):
        return {**result, "gate": False, "reason": "empty bootstrap block"}
    boot = sums[ids].sum(axis=1) / den[:, None]
    # Bonferroni intervals across the three primary asset evaluations.
    q = [0.05 / 6, 1 - 0.05 / 6]
    result["accuracy_ci98_333"] = np.quantile(boot[:, 3], q).tolist()
    result["gain_ci98_333_pct"] = np.quantile(
        100 * (boot[:, 0] - boot[:, 2]) / boot[:, 0], q
    ).tolist()
    result["gain_vs_prior_ci98_333_pct"] = np.quantile(
        100 * (boot[:, 1] - boot[:, 2]) / boot[:, 1], q
    ).tolist()
    month = anchors.astype("datetime64[s]").astype("datetime64[M]")
    result["monthly"] = {}
    for m in np.unique(month):
        mask = selected & (month == m)
        result["monthly"][str(m)] = {
            "n": int(mask.sum()),
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
            "gain_pct": float(
                100
                * (losses[mask, 0].mean() - losses[mask, 2].mean())
                / losses[mask, 0].mean()
            )
            if mask.any()
            else None,
        }
    result["gate"] = bool(
        result["coverage"] >= 0.05
        and result["accuracy"] >= 0.52
        and result["accuracy_ci98_333"][0] > 0.5
        and result["gain_ci98_333_pct"][0] > 0
        and result["log_loss"] <= result["baseline_log_loss"]
        and min(result["predicted_direction_counts"].values()) >= 100
        and sum(v["n"] >= 100 and v["gain_pct"] > 0 for v in result["monthly"].values())
        >= 5
    )
    return result
