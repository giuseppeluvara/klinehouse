"""Execution alpha: turning the order-flow signal into money *without* betting
on direction.

The signal's per-trade edge (sub-basis-point) is far too small to overcome
round-trip fees in a speculative strategy. But there is a setting where you pay
the fee **anyway**: when you already need to trade — dollar-cost-averaging,
rebalancing, building or unwinding a position you'll hold for hours or days.

In that setting the signal is free money: by *timing* the child orders of a
parent order toward favourable seconds, you lower your average fill price (when
buying) or raise it (when selling). Same quantity, same fees — only the timing
changes.

NOTE — this module weights each second by *that same second's* order flow, i.e.
it is **contemporaneous** (it implicitly assumes you can act on a second's flow
within that same second). That is an optimistic **upper bound** (~+0.6 bps), not
an achievable result. For the realistic, causal (1-second-lag) engine and its
honest sizing (~+0.1 bps for short windows), see ``execution_engine.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def execution_savings(
    df: pd.DataFrame,
    signal: str,
    side: str = "buy",
    window_s: int = 300,
    beta: float = 1.0,
    max_participation: float = 3.0,
) -> dict:
    """Simulate signal-tilted execution vs TWAP over fixed-size parent orders.

    The segment is cut into consecutive windows of ``window_s`` seconds. In each
    window a parent order must be fully executed. Two schedules are compared:

    - **TWAP**: equal weight on every second (the cost-neutral baseline).
    - **Tilted**: child weights proportional to ``exp(beta * z)`` where ``z`` is
      the standardised, sign-adjusted signal (buy more when the signal predicts
      weakness if buying; the reverse if selling), capped at
      ``max_participation`` times the TWAP weight to stay realistic.

    Fills are valued at the candle ``close``. Returns the average improvement in
    basis points (positive = cheaper buys / richer sells) with a t-statistic
    across windows. The schedule is causal: weights use only ``signal[t]``.
    """
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    sign = -1.0 if side == "buy" else 1.0  # buyers favour low signal

    px = df["close"].to_numpy(dtype=float)
    s = df[signal].to_numpy(dtype=float)
    n = len(df)

    improvements = []
    for start in range(0, n - window_s, window_s):
        sl = slice(start, start + window_s)
        p = px[sl]
        sig = s[sl]
        if np.allclose(sig, 0):
            continue
        z = (sig - sig.mean()) / (sig.std() + 1e-12)
        w = np.exp(beta * sign * z)
        w = np.minimum(w, max_participation * w.mean())  # cap participation
        w /= w.sum()
        twap = p.mean()
        tilted = float(np.dot(w, p))
        # positive improvement = better execution for the given side
        if side == "buy":
            imp = (twap - tilted) / twap * 1e4
        else:
            imp = (tilted - twap) / twap * 1e4
        improvements.append(imp)

    arr = np.asarray(improvements)
    if arr.size == 0:
        return {"side": side, "n_windows": 0}
    t = arr.mean() / (arr.std() / np.sqrt(arr.size)) if arr.std() > 0 else 0.0
    return {
        "side": side,
        "window_s": window_s,
        "n_windows": int(arr.size),
        "mean_improvement_bps": float(arr.mean()),
        "median_improvement_bps": float(np.median(arr)),
        "win_rate": float((arr > 0).mean()),
        "t_stat": float(t),
    }
