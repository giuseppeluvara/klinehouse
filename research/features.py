"""Microstructure feature engineering on 1s OHLCV + order-flow candles.

The single most valuable column the collector stores beyond plain OHLCV is
``taker_buy_base`` — the share of each candle's volume that came from
*aggressive buyers* (market orders lifting the ask). Together with ``volume``
it reconstructs **signed order flow**, the only consistently price-predictive
microstructure quantity available from public candle data.

All features are *causal*: a feature at time ``t`` uses only information known
by the end of second ``t``. Forward returns (the prediction target) are kept
strictly separate in :func:`add_forward_returns`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """1-second log return of the close price (causal)."""
    df = df.copy()
    logpx = np.log(df["close"].to_numpy(dtype=float))
    df["logpx"] = logpx
    df["logret"] = np.diff(logpx, prepend=logpx[0])
    return df


def add_order_flow(df: pd.DataFrame, windows=(5, 10, 30, 60)) -> pd.DataFrame:
    """Order-flow imbalance features.

    - ``signed_vol``   : aggressive-buy minus aggressive-sell base volume
                         (``2*taker_buy_base - volume``).
    - ``taker_ratio``  : fraction of volume that was aggressive buying
                         (0.5 = balanced; NaN seconds -> 0.5).
    - ``ofi_n{w}``     : rolling signed volume over ``w`` seconds, normalised by
                         rolling total volume -> a bounded [-1, 1] imbalance.
    - ``ofi_z{w}``     : rolling signed volume as a z-score (regime-relative).
    """
    df = df.copy()
    vol = df["volume"].to_numpy(dtype=float)
    tbb = df["taker_buy_base"].to_numpy(dtype=float)

    df["signed_vol"] = 2 * tbb - vol
    with np.errstate(invalid="ignore", divide="ignore"):
        df["taker_ratio"] = np.where(vol > 0, tbb / vol, 0.5)

    sv = pd.Series(df["signed_vol"].to_numpy())
    vv = pd.Series(vol)
    for w in windows:
        roll_sv = sv.rolling(w).sum()
        roll_v = vv.rolling(w).sum().replace(0, np.nan)
        df[f"ofi_n{w}"] = (roll_sv / roll_v).fillna(0).to_numpy()
        std = roll_sv.std()
        df[f"ofi_z{w}"] = ((roll_sv - roll_sv.mean()) / std).fillna(0).to_numpy() if std else 0.0
    return df


def add_activity(df: pd.DataFrame, windows=(30, 60)) -> pd.DataFrame:
    """Liquidity / activity features: trade intensity and realized volatility."""
    df = df.copy()
    if "logret" not in df:
        df = add_returns(df)
    trades = pd.Series(df["trades"].to_numpy(dtype=float))
    ret2 = pd.Series(df["logret"].to_numpy() ** 2)
    for w in windows:
        df[f"trade_rate{w}"] = trades.rolling(w).mean().fillna(0).to_numpy()
        # realized volatility over the trailing window, in bps
        df[f"rv{w}_bps"] = (np.sqrt(ret2.rolling(w).sum()) * 1e4).fillna(0).to_numpy()
    df["dollar_vol"] = df["quote_volume"]
    return df


def add_forward_returns(df: pd.DataFrame, horizons=(1, 2, 3, 5, 10, 30, 60),
                        exec_lag: int = 1) -> pd.DataFrame:
    """Prediction targets: forward log return realisable from a signal at ``t``.

    With ``exec_lag=1`` (the honest default) a signal observed at the end of
    second ``t`` is acted on at ``close[t+1]`` and the position is held ``k``
    seconds, so the target is ``log(close[t+1+k]) - log(close[t+1])``. This
    removes any contemporaneous (same-second) leakage between signal and target.
    """
    df = df.copy()
    logpx = np.log(df["close"].to_numpy(dtype=float))
    n = len(df)
    for k in horizons:
        fr = np.full(n, np.nan)
        last = n - exec_lag - k
        if last > 0:
            entry = logpx[exec_lag:exec_lag + last]
            exit_ = logpx[exec_lag + k:exec_lag + k + last]
            fr[:last] = exit_ - entry
        df[f"fwd{k}"] = fr
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience pipeline: returns + order flow + activity (no targets)."""
    df = add_returns(df)
    df = add_order_flow(df)
    df = add_activity(df)
    return df
