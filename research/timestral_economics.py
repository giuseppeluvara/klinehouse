"""Economic bridge from saved Timestral quantiles to delayed spot trades.

Historical research only. No model fitting, parameter search or order endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Trade:
    entry_ms: int
    exit_ms: int
    entry_price: float
    exit_price: float
    exposure: float

    def __post_init__(self) -> None:
        if not 0 <= self.entry_ms < self.exit_ms:
            raise ValueError('invalid holding interval')
        if not all(math.isfinite(x) for x in
                   (self.entry_price, self.exit_price, self.exposure)):
            raise ValueError('nonfinite trade')
        if min(self.entry_price, self.exit_price) <= 0 or not 0 <= self.exposure <= 1:
            raise ValueError('invalid price or exposure')


def net_return(trade: Trade, fee: float, slippage: float) -> float:
    """Exposure is cash spent inclusive of entry fee, preventing hidden leverage.

    Adverse slippage applies to price; fees apply to the slipped notional.
    Both legs execute as takers. Zero exposure incurs no cost.
    """
    if not all(math.isfinite(x) and 0 <= x < 1 for x in (fee, slippage)):
        raise ValueError('invalid costs')
    ratio = trade.exit_price / trade.entry_price
    factor = (1 - slippage) * (1 - fee) / ((1 + slippage) * (1 + fee))
    return trade.exposure * (ratio * factor - 1)


def exposure(policy: str, levels: list[float], quantiles: list[float], horizon: int,
             fee: float, slippage: float) -> float:
    if len(levels) != len(quantiles) or not levels:
        raise ValueError('quantile dimensions differ')
    if levels != sorted(set(levels)) or any(not 0 < x < 1 for x in levels):
        raise ValueError('invalid quantile levels')
    if not all(math.isfinite(x) for x in quantiles) or quantiles != sorted(quantiles):
        raise ValueError('nonfinite or crossing quantiles')
    if horizon not in (4, 24):
        raise ValueError('unregistered horizon')
    if not all(math.isfinite(x) and 0 <= x < 1 for x in (fee, slippage)):
        raise ValueError('invalid costs')
    q = dict(zip(levels, quantiles))
    if not {0.05, 0.25, 0.5, 0.95} <= q.keys():
        raise ValueError('missing required quantiles')
    if policy == 'median_positive':
        return float(q[0.5] > 0)
    if policy == 'lower_quartile_above_cost':
        hurdle = math.log((1 + slippage) * (1 + fee) /
                          ((1 - slippage) * (1 - fee)))
        return float(q[0.25] > hurdle)
    if policy == 'median_positive_vol_target':
        if q[0.5] <= 0:
            return 0.0
        sigma = (q[0.95] - q[0.05]) / 3.2897072539
        if sigma <= 0:
            raise ValueError('degenerate predictive spread')
        return min(1.0, 0.10 * math.sqrt(horizon / 8760) / sigma)
    raise ValueError('unregistered policy')


def wealth(returns: list[float]) -> tuple[float, float]:
    capital = peak = 1.0
    drawdown = 0.0
    for value in returns:
        if not math.isfinite(value) or value <= -1:
            raise ValueError('invalid portfolio return')
        capital *= 1 + value
        peak = max(peak, capital)
        drawdown = max(drawdown, 1 - capital / peak)
    return capital - 1, drawdown
