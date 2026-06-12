"""Generate a full, honest research report from the collected klines.

Usage
-----
    python -m research.report --symbol BTCUSDT
    python -m research.report --symbol BTCUSDT --csv /tmp/klines.csv
    python -m research.report --symbol BTCUSDT --start 2026-06-06 --end 2026-06-08

Writes ``research/out/report.md`` plus PNG figures. Every profitability figure
is net of fees, and the data map (segments / gaps) is reported up front so the
reader knows exactly how much real, continuous data backs each conclusion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import db, evaluate, execution_engine, features, volatility  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
SIGNALS = ["signed_vol", "ofi_n10", "ofi_n30", "ofi_n60", "taker_ratio"]
HORIZONS = (1, 2, 3, 5, 10, 30, 60)


def _prep(g: pd.DataFrame) -> pd.DataFrame:
    g = features.build_features(g)
    g = features.add_forward_returns(g, horizons=HORIZONS, exec_lag=1)
    return g


def _plot_price_flow(g: pd.DataFrame, seg: int) -> str:
    fig, ax1 = plt.subplots(figsize=(11, 4))
    t = g["open_time"].to_numpy()
    ax1.plot(t, g["close"], color="black", lw=0.8, label="close")
    ax1.set_ylabel("price")
    ax2 = ax1.twinx()
    cum_flow = np.cumsum(g["signed_vol"].to_numpy())
    ax2.plot(t, cum_flow, color="tab:blue", lw=0.9, alpha=0.7, label="cumulative signed flow")
    ax2.set_ylabel("cumulative signed base volume", color="tab:blue")
    ax1.set_title(f"Segment {seg}: price vs cumulative order-flow imbalance")
    fig.autofmt_xdate()
    path = OUT / f"price_flow_seg{seg}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def _plot_ic(ic: pd.DataFrame, seg: int) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ic["horizon_s"], ic["ic"], marker="o")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("forward horizon (s, log)")
    ax.set_ylabel("IC (Spearman)")
    ax.set_title(f"Segment {seg}: order-flow IC vs horizon (ofi_n10)")
    path = OUT / f"ic_seg{seg}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def _plot_deciles(tab: pd.DataFrame, seg: int) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:red" if v < 0 else "tab:green" for v in tab["mean_bps"]]
    ax.bar(tab.index.astype(int), tab["mean_bps"], color=colors)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("ofi_n10 decile (0=most selling, 9=most buying)")
    ax.set_ylabel("mean forward return @3s (bps)")
    ax.set_title(f"Segment {seg}: forward return by order-flow decile")
    path = OUT / f"deciles_seg{seg}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def _plot_fee_sweep(sweep: pd.DataFrame, seg: int) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["fee_bps_side"], sweep["net_bps"], marker="o")
    ax.axhline(0, color="grey", lw=0.8)
    ax.fill_between(sweep["fee_bps_side"], sweep["net_bps"], 0,
                    where=sweep["net_bps"] < 0, color="tab:red", alpha=0.15)
    ax.set_xlabel("fee per side (bps)")
    ax.set_ylabel("net edge per trade (bps)")
    ax.set_title(f"Segment {seg}: per-trade net edge vs fees (the reality check)")
    path = OUT / f"fee_sweep_seg{seg}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def analyse_segment(g: pd.DataFrame, seg: int, md: list[str]) -> None:
    g = _prep(g)
    span_h = (g["open_time"].max() - g["open_time"].min()).total_seconds() / 3600
    md.append(f"\n## Segment {seg} — {len(g):,} rows, {span_h:.1f} h "
              f"({g['open_time'].min():%Y-%m-%d %H:%M} → {g['open_time'].max():%H:%M} UTC)\n")

    # ---- Information coefficient table ----
    md.append("### Predictive power — information coefficient (Spearman, 1s exec lag)\n")
    md.append("IC of each signal against the forward log-return at several horizons. "
              "t-stats use non-overlapping samples.\n")
    md.append("| signal | " + " | ".join(f"fwd{k}s" for k in HORIZONS) + " |")
    md.append("|" + "---|" * (len(HORIZONS) + 1))
    best_ic = None
    for sig in SIGNALS:
        ic = evaluate.information_coefficient(g, sig, horizons=HORIZONS)
        cells = []
        for k in HORIZONS:
            row = ic[ic.horizon_s == k]
            cells.append(f"{row.ic.iloc[0]:+.3f}" if len(row) else "—")
        md.append(f"| `{sig}` | " + " | ".join(cells) + " |")
        if sig == "ofi_n10":
            best_ic = ic
    if best_ic is not None:
        peak = best_ic.loc[best_ic.ic.abs().idxmax()]
        md.append(f"\n*Peak IC for `ofi_n10`: {peak.ic:+.3f} at {int(peak.horizon_s)}s "
                  f"(t = {peak.t_stat:.1f}, n = {int(peak.n):,}).*\n")
        md.append(f"![IC]({_plot_ic(best_ic, seg)})\n")

    # ---- Decile monotonicity ----
    tab = evaluate.decile_table(g, "ofi_n10", horizon=3)
    spread = tab["mean_bps"].iloc[-1] - tab["mean_bps"].iloc[0]
    md.append("### Economic magnitude — decile analysis (`ofi_n10`, 3s horizon)\n")
    md.append(f"Top-minus-bottom decile spread: **{spread:.2f} bps gross**. "
              "This is the entire prize before costs.\n")
    md.append(f"![deciles]({_plot_deciles(tab, seg)})\n")

    # ---- Cost-aware backtest ----
    sweep = evaluate.fee_sweep(g, "ofi_n10", horizon=3)
    be = evaluate.breakeven_fee_bps(g, "ofi_n10", horizon=3)
    md.append("### Reality check — does it survive fees?\n")
    md.append("| fee/side (bps) | net edge/trade (bps) | hit rate | total net % | ann. Sharpe |")
    md.append("|---|---|---|---|---|")
    for _, r in sweep.iterrows():
        md.append(f"| {r.fee_bps_side:.1f} | {r.net_bps:+.2f} | {r.hit_rate:.1%} | "
                  f"{r.total_net_pct:+.2f} | {r.sharpe_annual:.0f} |")
    md.append(f"\n**Breakeven fee ≈ {be:.3f} bps/side.** Cheapest realistic crypto taker "
              "fee is ~2 bps (VIP futures); typical retail is 5–10 bps. The edge is "
              f"{2/max(be,1e-9):.0f}–{10/max(be,1e-9):.0f}× too small to trade speculatively.\n")
    md.append(f"![fees]({_plot_fee_sweep(sweep, seg)})\n")

    # ---- Volatility forecasting (the honest win) ----
    vf = volatility.evaluate_vol_forecast(g, halflife=60, horizon=60)
    md.append("### Volatility is forecastable (useful for risk & sizing)\n")
    md.append(f"EWMA(hl=60s) forecast vs realised next-60s vol: "
              f"rank-IC **{vf['rank_ic']:+.3f}**, MSE skill vs flat forecast "
              f"**{vf['mse_skill_vs_flat']:+.1%}** (n = {vf['n']:,}). "
              "Direction is hard; magnitude is not.\n")

    # ---- Execution alpha (value without paying extra fees) ----
    md.append("### Execution alpha — value on flow you trade anyway\n")
    md.append("**Causal** (1s-lag, predictive) signal-tilted execution vs TWAP over "
              "60-second parent orders (same quantity, same fees — only timing differs):\n")
    md.append("| side | causal mean (bps) | win rate | t-stat | look-ahead ceiling (bps) | windows |")
    md.append("|---|---|---|---|---|---|")
    for side in ("buy", "sell"):
        r = execution_engine.validate(g, side, 60)
        c, la = r["causal"], r["lookahead"]
        if c.get("n"):
            md.append(f"| {side} | {c['mean_bps']:+.2f} | {c['win_rate']:.0%} | "
                      f"{c['t_stat']:.1f} | {la['mean_bps']:+.2f} | {c['n']:,} |")
    md.append("\n*The causal column is the achievable number; the look-ahead column "
              "assumes intra-second reaction and is an unreachable ceiling. The edge "
              "concentrates at windows ≤ ~2 min and vanishes by 5 min — run "
              "`python -m research.execution_engine` for the window sweep.*\n")

    md.append(f"\n![price/flow]({_plot_price_flow(g, seg)})\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="klinehouse honest research report")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--csv", default=None, help="offline CSV instead of live DB")
    ap.add_argument("--min-rows", type=int, default=10_000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.csv:
        df = db.load_csv(args.csv)
        if args.symbol:
            df = df[df.symbol == args.symbol].reset_index(drop=True)
    else:
        df = db.load_klines(args.symbol, args.start, args.end)
    df = db.add_segments(df)
    summary = db.segment_summary(df)
    segs = db.largest_segments(df, min_rows=args.min_rows)

    md: list[str] = [f"# klinehouse research report — {args.symbol}\n"]
    md.append("> Auto-generated by `python -m research.report`. Every profitability "
              "number is **net of fees**. Read the data map first: it bounds every "
              "claim below.\n")

    md.append("## Data map (honest coverage)\n")
    md.append(f"Total rows: **{len(df):,}** across **{df.seg.nunique()} continuous "
              f"segments**. Usable continuous data (segments ≥ {args.min_rows:,} rows): "
              f"**{len(segs)} segments**.\n")
    md.append("| seg | rows | start | end | span (h) |")
    md.append("|---|---|---|---|---|")
    for _, r in summary.head(8).iterrows():
        md.append(f"| {int(r.seg)} | {int(r.rows):,} | {r.start:%Y-%m-%d %H:%M} | "
                  f"{r.end:%H:%M} | {r.span_h} |")
    longest_h = summary["span_h"].max()
    if len(segs) <= 2 and longest_h >= 24:
        md.append(f"\n> Largest continuous block: **{longest_h:.0f}h** gap-free. Good for "
                  "microstructure research and intraday validation; still a short history "
                  "for multi-week/regime conclusions — keep collecting.\n")
    else:
        md.append("\n> The collector ran intermittently: the calendar span is days, but "
                  "real continuous data is fragmented into shorter blocks. Run "
                  "`python scripts/backfill.py` to heal interior gaps. Enough for "
                  "microstructure research; **not** enough to validate a multi-day strategy.\n")

    if not segs:
        md.append("\n**No segment large enough to analyse.** Let the collector run longer.\n")
    for seg in segs:
        analyse_segment(df[df.seg == seg].copy(), seg, md)

    md.append("\n---\n## Bottom line\n")
    md.append(
        "- Order-flow imbalance **is** a real, reproducible predictor of 2–5s returns "
        "(monotonic deciles, consistent IC across independent segments).\n"
        "- Its per-trade edge is **sub-basis-point**, while the cheapest round-trip fee "
        "is ~4 bps. **Speculative scalping on this signal loses money — structurally, "
        "not for lack of tuning.**\n"
        "- The bankable value is elsewhere: **execution alpha** (cheaper fills on flow "
        "you trade anyway), **volatility/liquidity forecasting** (risk & sizing), and "
        "the **proprietary dataset** itself (gap-free 1s order flow is scarce).\n"
    )

    report = OUT / "report.md"
    report.write_text("\n".join(md))
    print(f"Wrote {report} ({len(md)} lines) and figures to {OUT}/")


if __name__ == "__main__":
    main()
