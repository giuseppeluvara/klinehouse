"""Exploratory public funding scanner, not a validated strategy or forward yield.

Current rates use configured settlement intervals and historical rates use
elapsed time. Fee-only projections omit basis PnL, capital, slippage and
liquidation. Historical persistence does not establish future persistence.
No keys or orders. CLI: python -m research.funding_scanner --top 10
"""
from __future__ import annotations

import argparse
import json
import math
import ssl
import time
import urllib.request

SPOT = "https://api.binance.com"
FUT = "https://fapi.binance.com"


def _ctx() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_SSL = _ctx()


def _get(url: str):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30, context=_SSL) as r:
                return json.load(r)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.0 * (attempt + 1))


def fetch_universe() -> dict[str, dict]:
    """Build a per-symbol view: funding, basis, spot & perp 24h liquidity."""
    prem = {x["symbol"]: x for x in _get(f"{FUT}/fapi/v1/premiumIndex")
            if x["symbol"].endswith("USDT")}
    spot_vol = {x["symbol"]: float(x["quoteVolume"])
                for x in _get(f"{SPOT}/api/v3/ticker/24hr") if x["symbol"].endswith("USDT")}
    perp_vol = {x["symbol"]: float(x["quoteVolume"])
                for x in _get(f"{FUT}/fapi/v1/ticker/24hr") if x["symbol"].endswith("USDT")}
    intervals = {x['symbol']: float(x['fundingIntervalHours'])
                 for x in _get(f"{FUT}/fapi/v1/fundingInfo")}
    uni = {}
    for s, p in prem.items():
        if s not in spot_vol:
            continue  # no spot market -> cannot build the long leg cleanly
        try:
            fr = float(p["lastFundingRate"])
            mark = float(p["markPrice"])
            index = float(p["indexPrice"])
        except (KeyError, ValueError):
            continue
        uni[s] = {
            "funding_8h": fr,
            "funding_interval_hours": intervals.get(s, 8.0),
            "funding_ann": fr * (24 / intervals.get(s, 8.0)) * 365 * 100,
            "basis_pct": (mark / index - 1) * 100 if index else 0.0,
            "spot_vol": spot_vol.get(s, 0.0),
            "perp_vol": perp_vol.get(s, 0.0),
        }
    return uni


def funding_history(symbol: str, limit: int = 90) -> list[tuple[int, float]]:
    rows = _get(f"{FUT}/fapi/v1/fundingRate?symbol={symbol}&limit={limit}")
    return [(int(r['fundingTime']), float(r['fundingRate'])) for r in rows]


def historical_summary(history: list[tuple[int, float]]) -> dict[str, float | int]:
    """Time-weighted annualization over (first receipt, last receipt].

    Exclude the first rate: its accrual interval is outside the observed span.
    Missing history is not replaced with the current projected rate.
    """
    if len(history) < 2:
        return dict(hist_ann=0.0, pct_positive=0.0, n_hist=0)
    times = [t for t, _ in history]
    if times != sorted(set(times)):
        raise ValueError('funding history must be unique and ordered')
    if any(type(t) is not int or t < 0 or not math.isfinite(r) for t, r in history):
        raise ValueError('invalid funding timestamp or rate')
    rates = [r for _, r in history[1:]]
    days = (times[-1] - times[0]) / 86400000
    return dict(hist_ann=sum(rates) / days * 365 * 100,
                pct_positive=100 * sum(r > 0 for r in rates) / len(rates),
                n_hist=len(rates))


def enrich_persistence(symbol: str, d: dict) -> dict:
    """Add mean/%-positive/stability of funding from history (the persistence test)."""
    h = funding_history(symbol)
    d.update(historical_summary(h))
    return d


def net_yield(hist_ann: float, hold_days: float, fee_roundtrip_pct: float) -> float:
    """Annualised net carry: funding earned continuously minus the one-off
    round-trip cost amortised over the holding period."""
    return hist_ann - fee_roundtrip_pct * (365.0 / hold_days)


def breakeven_days(hist_ann: float, fee_roundtrip_pct: float) -> float:
    return fee_roundtrip_pct * 365.0 / hist_ann if hist_ann > 0 else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser(description="Delta-neutral funding-carry scanner")
    ap.add_argument("--min-spot-vol", type=float, default=2e6, help="min spot 24h quote volume ($)")
    ap.add_argument("--min-perp-vol", type=float, default=5e6, help="min perp 24h quote volume ($)")
    ap.add_argument("--hold", type=float, default=30.0, help="assumed holding period (days)")
    ap.add_argument("--fee-roundtrip", type=float, default=0.30,
                    help="assumed round-trip fee %% (spot+perp, in+out); default 0.30%%")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--include-illiquid", action="store_true",
                    help="also show high-funding but thin/risky alts")
    args = ap.parse_args()
    if args.hold <= 0 or args.top <= 0 or args.fee_roundtrip < 0:
        ap.error('hold/top must be positive; costs must be nonnegative')

    print("Scanning Binance funding × liquidity …", flush=True)
    uni = fetch_universe()

    # candidates: positive funding + liquid enough to actually trade
    cands = {s: d for s, d in uni.items()
             if d["funding_ann"] > 0
             and d["perp_vol"] >= args.min_perp_vol
             and (args.include_illiquid or d["spot_vol"] >= args.min_spot_vol)}
    # take the richest by current funding, then enrich with history (limits API calls)
    top_by_now = sorted(cands.items(), key=lambda kv: -kv[1]["funding_ann"])[:max(args.top * 2, 30)]
    enriched = []
    for s, d in top_by_now:
        enrich_persistence(s, d)
        if d['n_hist'] == 0:
            continue
        d["net_ann"] = net_yield(d["hist_ann"], args.hold, args.fee_roundtrip)
        d["be_days"] = breakeven_days(d["hist_ann"], args.fee_roundtrip)
        enriched.append((s, d))
        time.sleep(0.05)

    # risk-adjusted rank: net carry weighted by how often funding is actually positive
    enriched.sort(key=lambda kv: -(kv[1]["net_ann"] * kv[1]["pct_positive"] / 100.0))

    print(f"\nDelta-neutral funding carry  (long spot + short perp), hold={args.hold:.0f}d, "
          f"round-trip cost={args.fee_roundtrip:.2f}%\n")
    print(f"  {'symbol':<13}{'fund now':>9}{'fund hist':>9}{'%pos':>6}{'fee-net':>8}"
          f"{'b/e days':>9}{'spot$M':>8}{'perp$M':>8}  flags")
    shown = 0
    for s, d in enriched:
        if shown >= args.top:
            break
        flags = []
        if d["spot_vol"] < 5e6:
            flags.append("thin-spot")
        if d["pct_positive"] < 70:
            flags.append("unstable")
        if d["basis_pct"] < 0:
            flags.append("neg-basis")
        if d["net_ann"] <= 0:
            flags.append("cost>carry")
        print(f"  {s:<13}{d['funding_ann']:>+8.0f}%{d['hist_ann']:>+8.0f}%{d['pct_positive']:>5.0f}%"
              f"{d['net_ann']:>+7.0f}%{d['be_days']:>9.0f}{d['spot_vol']/1e6:>8.1f}{d['perp_vol']/1e6:>8.1f}"
              f"  {','.join(flags) if flags else 'OK'}")
        shown += 1

    print("\nExploratory funding-only projection: excludes basis, capital and liquidation.")
    print("History is annualized by elapsed time; the displayed window varies by settlement frequency.")
    print("\nCarry structure (requires separate execution and margin validation):")
    print("  1. Buy the asset on SPOT for notional N.")
    print("  2. Match the spot BASE QUANTITY with a short linear perpetual; allocate margin separately.")
    print("  3. Funding settles at the symbol's configured interval and can change sign; unwind when funding")
    print("     decays below your cost hurdle or '%pos' / basis turns against you.")
    print("\n⚠️  Risks: a sharp pump can liquidate an over-levered short before funding pays; thin")
    print("    spot raises entry/exit slippage; funding can flip negative. Size for survival, keep")
    print("    the perp margin well above maintenance. 'OK' only means no scanner flag, not trade approval.")


if __name__ == "__main__":
    main()
