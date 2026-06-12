"""Honest signal evaluation: information coefficient, decile monotonicity, and
a cost-aware backtester that *cannot* hide transaction costs.

Design principle: every profitability number is reported **net of fees**, and
the breakeven fee is always shown next to the real exchange fee. A signal can
be statistically excellent and economically worthless at the same time — this
module is built to surface exactly that, rather than flatter the signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365 * 24 * 3600


def information_coefficient(
    df: pd.DataFrame,
    signal: str,
    horizons=(1, 2, 3, 5, 10, 30, 60),
    method: str = "spearman",
    nonoverlap: bool = True,
) -> pd.DataFrame:
    """IC = rank/linear correlation between ``signal[t]`` and ``fwd{k}``.

    For ``k > 1`` consecutive forward returns overlap, which inflates naive
    significance. With ``nonoverlap=True`` the IC for horizon ``k`` is computed
    on a ``k``-strided sub-sample so the t-statistic is honest.
    """
    rows = []
    for k in horizons:
        col = f"fwd{k}"
        if col not in df:
            continue
        d = df[[signal, col]].dropna()
        d = d[d[signal] != 0]  # ignore seconds with no order flow
        if nonoverlap and k > 1:
            d = d.iloc[::k]
        n = len(d)
        if n < 100:
            rows.append({"horizon_s": k, "ic": np.nan, "t_stat": np.nan, "n": n})
            continue
        ic = d[signal].corr(d[col], method=method)
        # t-stat of a correlation coefficient
        t = ic * np.sqrt((n - 2) / max(1e-12, 1 - ic**2))
        rows.append({"horizon_s": k, "ic": ic, "t_stat": t, "n": n})
    return pd.DataFrame(rows)


def decile_table(df: pd.DataFrame, signal: str, horizon: int, deciles: int = 10) -> pd.DataFrame:
    """Mean forward return (bps) per signal decile. Monotonic, well-separated
    deciles are the visual fingerprint of a real (if small) signal."""
    col = f"fwd{horizon}"
    d = df[[signal, col]].dropna()
    d = d[d[signal] != 0].copy()
    d["bucket"] = pd.qcut(d[signal], deciles, labels=False, duplicates="drop")
    tab = d.groupby("bucket")[col].agg(["mean", "std", "count"])
    tab["mean_bps"] = tab["mean"] * 1e4
    return tab[["mean_bps", "count"]]


def backtest(
    df: pd.DataFrame,
    signal: str,
    horizon: int,
    thr_q: float = 0.20,
    fee_bps_side: float = 5.0,
    exec_lag: int = 1,
) -> dict | None:
    """Event-driven long/short backtest with non-overlapping trades.

    Go long when ``signal`` is in its top ``thr_q`` quantile, short in the
    bottom ``thr_q``. Enter at ``close[t+exec_lag]``, exit ``horizon`` seconds
    later, pay ``fee_bps_side`` on entry *and* exit. Trades never overlap.

    Returns per-trade and annualised metrics, all net of fees.
    """
    logpx = np.log(df["close"].to_numpy(dtype=float))
    s = df[signal].to_numpy()
    n = len(df)
    valid = s[s != 0]
    if len(valid) < 100:
        return None
    hi = np.quantile(valid, 1 - thr_q)
    lo = np.quantile(valid, thr_q)

    rets = []
    t = 0
    limit = n - exec_lag - horizon - 1
    while t < limit:
        pos = 1 if s[t] >= hi else (-1 if s[t] <= lo else 0)
        if pos:
            entry = logpx[t + exec_lag]
            exit_ = logpx[t + exec_lag + horizon]
            net = pos * (exit_ - entry) * 1e4 - 2 * fee_bps_side
            rets.append(net)
            t += horizon + exec_lag
        else:
            t += 1
    rets = np.asarray(rets)
    if rets.size == 0:
        return None

    dur = (df["open_time"].max() - df["open_time"].min()).total_seconds()
    trades_per_year = rets.size / dur * SECONDS_PER_YEAR if dur else 0.0
    sharpe_trade = rets.mean() / rets.std() if rets.std() > 0 else 0.0
    return {
        "n_trades": int(rets.size),
        "gross_bps": float(rets.mean() + 2 * fee_bps_side),
        "net_bps": float(rets.mean()),
        "hit_rate": float((rets > 0).mean()),
        "total_net_pct": float(rets.sum() / 100),
        "sharpe_annual": float(sharpe_trade * np.sqrt(trades_per_year)),
        "trades_per_year": float(trades_per_year),
        "fee_bps_side": fee_bps_side,
    }


def breakeven_fee_bps(df: pd.DataFrame, signal: str, horizon: int,
                      thr_q: float = 0.20, exec_lag: int = 1) -> float | None:
    """Per-side fee (bps) at which the frictionless edge is exactly consumed."""
    r0 = backtest(df, signal, horizon, thr_q, fee_bps_side=0.0, exec_lag=exec_lag)
    return None if r0 is None else r0["gross_bps"] / 2


def fee_sweep(df: pd.DataFrame, signal: str, horizon: int,
              fees=(0.0, 1.0, 2.0, 5.0, 7.5, 10.0), thr_q: float = 0.20) -> pd.DataFrame:
    """Backtest across a range of per-side fees — the reality-check table."""
    rows = []
    for fee in fees:
        r = backtest(df, signal, horizon, thr_q=thr_q, fee_bps_side=fee)
        if r:
            rows.append(r)
    return pd.DataFrame(rows)
