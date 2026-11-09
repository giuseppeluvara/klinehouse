"""Self-financing fractional spot portfolio with adverse execution costs."""
from __future__ import annotations

import math


def rebalance(cash: float, units: list[float], prices: list[float],
              weights: list[float], fee: float, slippage: float
              ) -> tuple[float, list[float], float]:
    if not len(units) == len(prices) == len(weights) or not units:
        raise ValueError('portfolio dimensions differ')
    if not all(math.isfinite(x) for x in [cash, fee, slippage, *units, *prices, *weights]):
        raise ValueError('nonfinite portfolio')
    if cash < -1e-8 or min(units) < 0 or min(prices) <= 0 or min(weights) < 0:
        raise ValueError('negative cash, holdings, prices or weights')
    if sum(weights) > 1 + 1e-12 or not 0 <= fee < 1 or not 0 <= slippage < 1:
        raise ValueError('leverage or invalid costs')
    nav = cash + sum(u*p for u, p in zip(units, prices))
    buy_cost = (1 + slippage) * (1 + fee) - 1
    sell_cost = 1 - (1 - slippage) * (1 - fee)

    def cost_at(value: float) -> float:
        changes = [w*value-u*p for w, u, p in zip(weights, units, prices)]
        return sum(abs(d)*(buy_cost if d > 0 else sell_cost) for d in changes)

    low, high = 0.0, nav
    for _ in range(80):
        mid = (low + high) / 2
        if mid + cost_at(mid) > nav:
            high = mid
        else:
            low = mid
    after = (low + high) / 2
    new_units = [w*after/p for w, p in zip(weights, prices)]
    new_cash = after * (1 - sum(weights))
    cost = cost_at(after)
    if abs(after + cost - nav) > max(1e-8, nav * 1e-12):
        raise ValueError('portfolio failed self-financing identity')
    return new_cash, new_units, cost
