"""Short-horizon volatility forecasting.

Unlike the *direction* of returns (barely predictable, and too small to beat
fees), the *magnitude* of returns is strongly autocorrelated — volatility comes
in clusters. That makes near-term volatility genuinely forecastable, which is
useful for position sizing, risk limits, stop placement and "don't-trade-now"
gating, regardless of any directional view.

The forecaster is a deliberately simple, transparent EWMA of squared returns
(the RiskMetrics idea). The point of this module is not a fancy model but an
*honest out-of-sample evaluation* of how forecastable volatility actually is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol_forecast(df: pd.DataFrame, halflife: float = 60.0) -> pd.Series:
    """Causal EWMA volatility (per-second sigma, in return units).

    sigma_t is the square-root of an exponentially weighted mean of past
    squared returns, shifted by one so it uses only information up to ``t-1``.
    """
    if "logret" not in df:
        raise ValueError("call features.add_returns first")
    r2 = pd.Series(df["logret"].to_numpy() ** 2)
    var = r2.ewm(halflife=halflife, adjust=False).mean().shift(1)
    return np.sqrt(var)


def evaluate_vol_forecast(df: pd.DataFrame, halflife: float = 60.0,
                          horizon: int = 60) -> dict:
    """Out-of-sample skill of the EWMA forecast.

    Compares the forecast made at ``t`` against the *realised* volatility over
    the next ``horizon`` seconds. Reports rank correlation (does it order calm
    vs stormy periods correctly?) and the variance ratio of a naive constant
    forecast vs the EWMA (QLIKE-style sanity check).
    """
    sigma = ewma_vol_forecast(df, halflife=halflife).to_numpy()
    ret = df["logret"].to_numpy()
    n = len(ret)
    # realised vol over the forward window, per-second equivalent
    fwd_rv = np.full(n, np.nan)
    r2 = ret ** 2
    csum = np.concatenate([[0.0], np.cumsum(r2)])
    last = n - horizon
    if last > 0:
        fwd_rv[:last] = np.sqrt((csum[horizon:horizon + last] - csum[:last]) / horizon)

    d = pd.DataFrame({"f": sigma, "a": fwd_rv}).dropna()
    d = d[d["f"] > 0]
    rank_ic = d["f"].corr(d["a"], method="spearman")
    # skill vs a flat (unconditional) forecast under MSE on variance
    mse_ewma = np.mean((d["f"] ** 2 - d["a"] ** 2) ** 2)
    flat = d["a"].pow(2).mean()
    mse_flat = np.mean((flat - d["a"] ** 2) ** 2)
    skill = 1 - mse_ewma / mse_flat if mse_flat else np.nan
    return {
        "halflife_s": halflife,
        "horizon_s": horizon,
        "rank_ic": float(rank_ic),
        "mse_skill_vs_flat": float(skill),
        "n": int(len(d)),
    }
