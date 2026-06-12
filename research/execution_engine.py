"""Execution-alpha engine (paper / advisory) — the one bankable use of the
order-flow signal, sized honestly.

A speculative bet on the signal loses to fees (the per-trade edge is
sub-basis-point; see ``research/README.md``). But when you *already* have to
trade — DCA, rebalancing, building/unwinding a multi-hour position — you pay the
fee regardless, so timing the child orders of that parent order toward
favourable seconds is free money.

This engine is **causal and predictive**: at each slice it sizes the next child
order from the *most recent completed* second's order-flow imbalance (a 1-second
reaction lag — no intra-second peeking). A buyer front-loads when recent flow is
positive (price likely to tick up, so buy before it does); a seller does the
reverse. A uniform baseline avoids the trend-chasing bias of a naive
``remaining/remaining`` schedule, and the final slice trues up so the order
always completes.

Honest sizing (validated on the collected data, 1s-lag, vs TWAP):

    window   edge vs TWAP      window   edge vs TWAP
      30s    ~+0.10 bps          120s   ~+0.11 bps
      60s    ~+0.12 bps          300s   ~0.00 bps  (edge gone)

The signal predicts only a few seconds ahead, so the benefit is real and
statistically significant for **short windows (≤ ~2 min)** and vanishes beyond
that. ~+0.1 bps is ~€10 per €1M traded — small, but free and scaling with size.
(An *intra-second look-ahead* schedule shows ~+0.6 bps, but that is not
achievable in practice; this engine reports only the causal number.)

CLI
---
    # Validate on recent history (with a window-size sweep) + example schedule
    python -m research.execution_engine --symbol BTCUSDT --side buy --qty 10 --window 60s

    # Real-time advisor: prints the recommended child size every second
    python -m research.execution_engine --symbol BTCUSDT --side buy --qty 10 --window 60s --live
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import db, features

SIGNAL = "ofi_n10"  # bounded [-1, 1] order-flow imbalance


def parse_window(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("m"):
        return int(float(s[:-1]) * 60)
    if s.endswith("h"):
        return int(float(s[:-1]) * 3600)
    return int(s.rstrip("s"))


def adaptive_execution(prices: np.ndarray, signal: np.ndarray, side: str,
                       total_qty: float, beta: float = 2.0,
                       max_participation: float = 3.0) -> dict:
    """Causal, predictive child-order scheduler over one parent order.

    ``signal[t]`` is the order-flow imbalance known at the *start* of slice ``t``
    (i.e. already lagged for live use) and bounded in [-1, 1]. Child size is a
    uniform baseline ``Q/N`` tilted by ``exp(beta · direction · signal)`` —
    a buyer (direction +1) buys more when the signal is positive, a seller
    (direction −1) sells more when it is negative — capped at
    ``max_participation × baseline``. The final slice clears any remainder.
    """
    n = len(prices)
    direction = 1.0 if side == "buy" else -1.0
    baseline = total_qty / n
    q_rem = total_qty
    schedule = np.zeros(n)
    for t in range(n):
        fav = float(np.clip(direction * signal[t], -1.0, 1.0))
        qty = baseline * np.exp(beta * fav)
        qty = min(max(qty, 0.0), q_rem, max_participation * baseline)
        if t == n - 1:
            qty = q_rem
        schedule[t] = qty
        q_rem -= qty
    filled = schedule.sum()
    avg_price = float(np.dot(schedule, prices) / filled)
    twap = float(prices.mean())
    imp = ((twap - avg_price) if side == "buy" else (avg_price - twap)) / twap * 1e4
    return {"schedule": schedule, "avg_price": avg_price, "twap": twap,
            "improvement_bps": imp}


def _lag(arr: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        return arr.copy()
    out = np.zeros_like(arr)
    out[lag:] = arr[:-lag]
    return out


def _improvements(prices: np.ndarray, sig: np.ndarray, raw_sig: np.ndarray,
                  side: str, window_s: int, beta: float, max_participation: float) -> np.ndarray:
    n = len(prices)
    res = []
    for start in range(0, n - window_s, window_s):
        sl = slice(start, start + window_s)
        if np.allclose(raw_sig[sl], 0):
            continue
        res.append(adaptive_execution(prices[sl], sig[sl], side, 1.0, beta, max_participation)["improvement_bps"])
    return np.asarray(res)


def _stats(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0}
    t = arr.mean() / (arr.std() / np.sqrt(arr.size)) if arr.std() > 0 else 0.0
    return {"n": int(arr.size), "mean_bps": float(arr.mean()),
            "median_bps": float(np.median(arr)), "win_rate": float((arr > 0).mean()),
            "t_stat": float(t)}


def validate(df: pd.DataFrame, side: str, window_s: int, beta: float = 2.0,
             max_participation: float = 3.0) -> dict:
    """Replay the causal engine over every non-overlapping window. Also reports
    the (non-achievable) intra-second look-ahead bound for context."""
    df = features.add_order_flow(features.add_returns(df))
    prices = df["close"].to_numpy(dtype=float)
    raw = df[SIGNAL].to_numpy(dtype=float)
    causal = _improvements(prices, _lag(raw, 1), raw, side, window_s, beta, max_participation)
    lookahead = _improvements(prices, raw, raw, side, window_s, beta, max_participation)
    return {"side": side, "window_s": window_s,
            "causal": _stats(causal), "lookahead": _stats(lookahead)}


def window_sweep(df: pd.DataFrame, side: str, beta: float = 2.0,
                 windows=(30, 60, 120, 300)) -> pd.DataFrame:
    df = features.add_order_flow(features.add_returns(df))
    prices = df["close"].to_numpy(dtype=float)
    raw = df[SIGNAL].to_numpy(dtype=float)
    sig = _lag(raw, 1)
    rows = []
    for w in windows:
        st = _stats(_improvements(prices, sig, raw, side, w, beta, 3.0))
        st["window_s"] = w
        rows.append(st)
    return pd.DataFrame(rows)[["window_s", "n", "mean_bps", "median_bps", "win_rate", "t_stat"]]


def example_schedule(df: pd.DataFrame, side: str, qty: float, window_s: int,
                     beta: float, max_participation: float):
    df = features.add_order_flow(features.add_returns(df)).tail(window_s).reset_index(drop=True)
    prices = df["close"].to_numpy(dtype=float)
    sig = _lag(df[SIGNAL].to_numpy(dtype=float), 1)
    res = adaptive_execution(prices, sig, side, qty, beta, max_participation)
    out = pd.DataFrame({
        "time": df["open_time"],
        "price": prices,
        "ofi_signal": df[SIGNAL].to_numpy(),
        "child_qty": res["schedule"].round(6),
    })
    out["vs_twap_x"] = (out["child_qty"] / (qty / window_s)).round(2)
    return out, res


def live_advisor(symbol: str, side: str, qty: float, window_s: int,
                 beta: float, max_participation: float) -> None:
    """Real-time advisory loop: every second recommend the child size for the
    current second from the latest order-flow. Advisory only — places no orders."""
    print(f"[live] {side.upper()} {qty} {symbol} over {window_s}s — advisory only, no orders placed")
    direction = 1.0 if side == "buy" else -1.0
    q_rem = qty
    deadline = time.time() + window_s
    conn = db.connect()
    try:
        while q_rem > 1e-12 and time.time() < deadline:
            g = features.add_order_flow(features.add_returns(db.load_klines(symbol, conn=conn).tail(60)))
            sig = float(g[SIGNAL].iloc[-1])      # last completed second -> causal
            price = float(g["close"].iloc[-1])
            m = max(1, int(deadline - time.time()))
            baseline = q_rem / m
            fav = float(np.clip(direction * sig, -1.0, 1.0))
            child = min(max(baseline * np.exp(beta * fav), 0.0), q_rem, max_participation * baseline)
            tag = "▲ act now" if fav > 0.15 else ("▼ ease off" if fav < -0.15 else "· neutral")
            print(f"[live] t-{m:>4}s  px={price:<10.2f} ofi={sig:+.2f}  "
                  f"-> child {child:.6f} ({child/baseline:.2f}× pace)  {tag}", flush=True)
            q_rem -= child
            time.sleep(1)
        print(f"[live] done, residual qty={q_rem:.6f}")
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Execution-alpha engine (paper/advisory)")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--side", choices=["buy", "sell"], default="buy")
    ap.add_argument("--qty", type=float, default=10.0)
    ap.add_argument("--window", default="60s", help="parent-order window; edge concentrates ≤2m")
    ap.add_argument("--beta", type=float, default=2.0, help="tilt aggressiveness")
    ap.add_argument("--max-participation", type=float, default=3.0)
    ap.add_argument("--hours", type=float, default=0.0,
                    help="history window to validate on (0 = all available continuous data)")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    window_s = parse_window(args.window)

    if args.live:
        live_advisor(args.symbol, args.side, args.qty, window_s, args.beta, args.max_participation)
        return

    if args.csv:
        df = db.load_csv(args.csv)
        df = df[df.symbol == args.symbol].reset_index(drop=True)
    else:
        df = db.load_klines(args.symbol)
    df = db.add_segments(df)
    segs = db.largest_segments(df, min_rows=window_s * 5)
    if segs:
        seg = db.segment_summary(df[df.seg.isin(segs)]).sort_values("end").iloc[-1]["seg"]
        df = df[df.seg == seg].reset_index(drop=True)
    if args.hours > 0:
        df = df.tail(int(args.hours * 3600)).reset_index(drop=True)

    res = validate(df, args.side, window_s, args.beta, args.max_participation)
    c, la = res["causal"], res["lookahead"]
    print(f"\nExecution-alpha validation — {args.side.upper()} {args.symbol}, "
          f"{window_s}s windows, last {len(df)/3600:.1f}h ({c.get('n', 0)} windows)\n")
    if c.get("n"):
        print(f"  causal engine (1s lag, real):  mean {c['mean_bps']:+.2f} bps, "
              f"median {c['median_bps']:+.2f}, win {c['win_rate']:.0%}, t = {c['t_stat']:.1f}")
        print(f"  look-ahead bound (not real):   mean {la['mean_bps']:+.2f} bps "
              "(intra-second; shown only as a ceiling)")
        print(f"\n  → On €1,000,000 of {args.side} flow: ~€{c['mean_bps']*100:.0f} saved "
              f"per execution vs TWAP, zero extra fees.")

    print("\n  Edge by window size (causal):")
    sweep = window_sweep(df, args.side, args.beta)
    for _, r in sweep.iterrows():
        flag = "  <- edge gone" if abs(r.t_stat) < 2 else ""
        print(f"    {int(r.window_s):>4}s : {r.mean_bps:+.3f} bps  t={r.t_stat:+.1f}  "
              f"win {r.win_rate:.0%}{flag}")

    sched, single = example_schedule(df, args.side, args.qty, window_s, args.beta, args.max_participation)
    print(f"\nExample schedule — {args.side} {args.qty} {args.symbol}, last {window_s}s: "
          f"avg fill {single['avg_price']:.2f} vs TWAP {single['twap']:.2f} "
          f"→ {single['improvement_bps']:+.2f} bps")
    show = sched.iloc[::max(1, window_s // 8)].copy()
    show["time"] = show["time"].dt.strftime("%H:%M:%S")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
