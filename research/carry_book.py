"""Paper carry book — forward, out-of-sample validation of the funding-harvest
strategy, plus an instant historical what-if from real funding settlements.

A delta-neutral carry position (long spot + short perp, equal notional) has no
first-order price exposure, so its PnL is essentially:

    funding collected at each 8h settlement   −   round-trip trading cost
    (+ small basis convergence, ignored here as second-order)

This tool tracks paper positions using the **actual funding payments** Binance
publishes, so over days it shows real money accruing — honestly, with no price
guessing. ``simulate`` does the same over a past window for an immediate demo.

State persists to ``research/out/carry_book.json``. No API key; no real orders.

CLI
---
    python -m research.carry_book simulate DEXEUSDT --days 14 --notional 10000
    python -m research.carry_book open DEXEUSDT --notional 10000
    python -m research.carry_book status
    python -m research.carry_book close DEXEUSDT
"""
from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.request
from pathlib import Path

FUT = "https://fapi.binance.com"
BOOK = Path(__file__).resolve().parent / "out" / "carry_book.json"


def _ctx() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_SSL = _ctx()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30, context=_SSL) as r:
        return json.load(r)


def funding_since(symbol: str, start_ms: int) -> list[tuple[int, float]]:
    """All funding settlements for symbol at/after start_ms: [(time_ms, rate)]."""
    out: list[tuple[int, float]] = []
    cursor = start_ms
    while True:
        rows = _get(f"{FUT}/fapi/v1/fundingRate?symbol={symbol}&startTime={cursor}&limit=1000")
        if not rows:
            break
        for r in rows:
            out.append((int(r["fundingTime"]), float(r["fundingRate"])))
        if len(rows) < 1000:
            break
        cursor = rows[-1]["fundingTime"] + 1
        time.sleep(0.1)
    return out


def carry_pnl(symbol: str, notional: float, start_ms: int, end_ms: int | None,
              fee_roundtrip_pct: float) -> dict:
    """Net carry of a delta-neutral position over [start, end] from real funding.

    A *short* perp receives funding when the rate is positive, so the cash
    collected per settlement is ``+rate * notional`` (short pays when negative).
    """
    sett = [(t, r) for t, r in funding_since(symbol, start_ms)
            if end_ms is None or t <= end_ms]
    gross = sum(r * notional for _, r in sett)          # short perp collects +rate
    cost = fee_roundtrip_pct / 100.0 * notional          # entry + exit, both legs
    net = gross - cost
    days = ((end_ms or int(time.time() * 1000)) - start_ms) / 86_400_000
    # only annualise once there is a meaningful holding period and real settlements
    ann = (net / notional) / days * 365 * 100 if (days >= 0.5 and sett) else None
    return {
        "symbol": symbol, "notional": notional, "settlements": len(sett),
        "gross_funding": gross, "cost": cost, "net": net, "days": days,
        "net_ann_pct": ann, "pct_positive": 100 * sum(1 for _, r in sett if r > 0) / len(sett) if sett else 0.0,
    }


def _load() -> dict:
    if BOOK.exists():
        return json.loads(BOOK.read_text())
    return {"positions": []}


def _save(book: dict) -> None:
    BOOK.parent.mkdir(parents=True, exist_ok=True)
    BOOK.write_text(json.dumps(book, indent=2))


def _fmt(r: dict) -> str:
    ann = f"{r['net_ann_pct']:+.1f}%/yr, {r['pct_positive']:.0f}% pos" if r["net_ann_pct"] is not None else "accruing…"
    return (f"  {r['symbol']:<13} notional ${r['notional']:,.0f}  {r['days']:.1f}d  "
            f"{r['settlements']} settle  funding ${r['gross_funding']:+,.2f}  "
            f"cost ${r['cost']:,.2f}  -> NET ${r['net']:+,.2f}  ({ann})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper carry book (delta-neutral funding harvest)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap.add_argument("--fee-roundtrip", type=float, default=0.24, help="round-trip cost %%")

    s = sub.add_parser("simulate", help="historical what-if from real funding")
    s.add_argument("symbol")
    s.add_argument("--days", type=float, default=14)
    s.add_argument("--notional", type=float, default=10000)

    o = sub.add_parser("open", help="open a paper position now")
    o.add_argument("symbol")
    o.add_argument("--notional", type=float, default=10000)

    sub.add_parser("status", help="mark all open positions with real funding to date")

    c = sub.add_parser("close", help="close a paper position")
    c.add_argument("symbol")

    args = ap.parse_args()

    if args.cmd == "simulate":
        start = int(time.time() * 1000) - int(args.days * 86_400_000)
        r = carry_pnl(args.symbol, args.notional, start, None, args.fee_roundtrip)
        print(f"\nHistorical delta-neutral carry — {args.symbol}, last {args.days:.0f}d, "
              f"${args.notional:,.0f} notional (real funding settlements):")
        print(_fmt(r))
        print("  (long spot + short perp; price-neutral. Past funding ≠ future.)")
        return

    book = _load()
    if args.cmd == "open":
        book["positions"] = [p for p in book["positions"] if p["symbol"] != args.symbol]
        book["positions"].append({"symbol": args.symbol, "notional": args.notional,
                                  "open_ms": int(time.time() * 1000), "closed": False})
        _save(book)
        print(f"opened paper carry: {args.symbol} ${args.notional:,.0f} (long spot + short perp)")
    elif args.cmd == "close":
        for p in book["positions"]:
            if p["symbol"] == args.symbol and not p["closed"]:
                p["closed"] = True
                p["close_ms"] = int(time.time() * 1000)
        _save(book)
        print(f"closed paper carry: {args.symbol}")
    elif args.cmd == "status":
        if not book["positions"]:
            print("carry book empty. Open one: python -m research.carry_book open DEXEUSDT --notional 10000")
            return
        total = 0.0
        print("\nPaper carry book — marked with real funding settlements:")
        for p in book["positions"]:
            r = carry_pnl(p["symbol"], p["notional"], p["open_ms"],
                          p.get("close_ms"), args.fee_roundtrip)
            tag = "" if not p["closed"] else " [closed]"
            print(_fmt(r) + tag)
            total += r["net"]
        print(f"\n  TOTAL net carry: ${total:+,.2f}")


if __name__ == "__main__":
    main()
