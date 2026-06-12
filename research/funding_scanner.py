"""Funding-carry scanner — the one genuinely positive-expectancy, market-neutral,
retail-accessible money-maker this project's research could actually defend.

There is no directional or stat-arb edge in BTC/ETH/SOL second-scale data that
survives fees (see ``research/README.md``): those markets are too efficient.
What *does* pay is harvesting a **structural risk premium** — the perpetual-swap
*funding rate*. When traders crowd into leveraged longs, longs pay shorts a
funding fee every 8h. You can collect it **delta-neutrally**:

    long the spot asset  +  short the same-size perpetual   →  no price exposure,
    you just earn (positive) funding for as long as you hold.

This is not free money — it is payment for taking the *short side of leverage
demand*. The risks are real (a violent pump can liquidate the perp short before
funding compensates; thin spot makes entry/exit costly; funding can flip). So
this scanner ranks opportunities by **risk-adjusted net carry**, not headline
funding: it requires a liquid spot market (to build the long leg cleanly),
measures funding **persistence** from history (not just the current print), and
subtracts realistic round-trip fees.

CLI
---
    python -m research.funding_scanner                      # liquid, carryable, ranked
    python -m research.funding_scanner --min-spot-vol 5e6 --hold 30 --top 25
    python -m research.funding_scanner --include-illiquid   # also show risky high-yield alts

Data: Binance public REST (spot + USDⓈ-M futures). No API key, no orders placed.
"""
from __future__ import annotations

import argparse
import json
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
            "funding_ann": fr * 3 * 365 * 100,
            "basis_pct": (mark / index - 1) * 100 if index else 0.0,
            "spot_vol": spot_vol.get(s, 0.0),
            "perp_vol": perp_vol.get(s, 0.0),
        }
    return uni


def funding_history(symbol: str, limit: int = 90) -> list[float]:
    rows = _get(f"{FUT}/fapi/v1/fundingRate?symbol={symbol}&limit={limit}")
    return [float(r["fundingRate"]) for r in rows]


def enrich_persistence(symbol: str, d: dict) -> dict:
    """Add mean/%-positive/stability of funding from history (the persistence test)."""
    h = funding_history(symbol)
    if not h:
        d.update(hist_ann=d["funding_ann"], pct_positive=0.0, n_hist=0)
        return d
    mean = sum(h) / len(h)
    d["hist_ann"] = mean * 3 * 365 * 100
    d["pct_positive"] = 100 * sum(1 for r in h if r > 0) / len(h)
    d["n_hist"] = len(h)
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
    ap.add_argument("--fee-roundtrip", type=float, default=0.24,
                    help="round-trip cost %% (spot+perp, in+out); default 0.24%% with fee discounts")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--include-illiquid", action="store_true",
                    help="also show high-funding but thin/risky alts")
    args = ap.parse_args()

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
        d["net_ann"] = net_yield(d["hist_ann"], args.hold, args.fee_roundtrip)
        d["be_days"] = breakeven_days(d["hist_ann"], args.fee_roundtrip)
        enriched.append((s, d))
        time.sleep(0.05)

    # risk-adjusted rank: net carry weighted by how often funding is actually positive
    enriched.sort(key=lambda kv: -(kv[1]["net_ann"] * kv[1]["pct_positive"] / 100.0))

    print(f"\nDelta-neutral funding carry  (long spot + short perp), hold={args.hold:.0f}d, "
          f"round-trip cost={args.fee_roundtrip:.2f}%\n")
    print(f"  {'symbol':<13}{'fund now':>9}{'fund 30d':>9}{'%pos':>6}{'net/yr':>8}"
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

    print("\nHow to harvest one line above (delta-neutral, no price view):")
    print("  1. Buy the asset on SPOT for notional N.")
    print("  2. Open a SHORT PERP of the same notional N (1× — keep low leverage for liquidation buffer).")
    print("  3. Collect funding every 8h while it stays positive; unwind both legs when funding")
    print("     decays below your cost hurdle or '%pos' / basis turns against you.")
    print("\n⚠️  Risks: a sharp pump can liquidate an over-levered short before funding pays; thin")
    print("    spot raises entry/exit slippage; funding can flip negative. Size for survival, keep")
    print("    the perp margin well above maintenance, and prefer rows flagged 'OK'. Not advice.")


if __name__ == "__main__":
    main()
