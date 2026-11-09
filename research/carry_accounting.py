"""Explicit accounting for a fixed-quantity long spot / short linear perp.

Prices are executed prices (or explicitly labelled historical proxies).
Funding at entry is excluded; funding at exit is excluded conservatively.
This is accounting, not an execution or liquidation simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable


def valid(value: Decimal, *, positive: bool = False) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError("finite Decimal required")
    if positive and value <= 0:
        raise ValueError("positive value required")


@dataclass(frozen=True)
class Prices:
    time_ms: int
    spot: Decimal
    perp: Decimal

    def __post_init__(self) -> None:
        if type(self.time_ms) is not int or self.time_ms < 0:
            raise ValueError("nonnegative integer timestamp required")
        valid(self.spot, positive=True)
        valid(self.perp, positive=True)


@dataclass(frozen=True)
class Settlement:
    time_ms: int
    rate: Decimal
    mark: Decimal

    def __post_init__(self) -> None:
        if type(self.time_ms) is not int or self.time_ms < 0:
            raise ValueError("nonnegative integer timestamp required")
        valid(self.rate)
        valid(self.mark, positive=True)


@dataclass(frozen=True)
class Costs:
    spot_fee: Decimal
    perp_fee: Decimal
    slippage_bps: Decimal
    annual_capital_rate: Decimal

    def __post_init__(self) -> None:
        for value in (self.spot_fee, self.perp_fee, self.slippage_bps,
                      self.annual_capital_rate):
            valid(value)
            if value < 0:
                raise ValueError("negative cost")


def account(entry: Prices, exit: Prices, quantity: Decimal,
            settlements: Iterable[Settlement], costs: Costs,
            capital: Decimal) -> dict[str, Decimal | int]:
    valid(quantity, positive=True)
    valid(capital, positive=True)
    if exit.time_ms <= entry.time_ms:
        raise ValueError("exit must follow entry")
    if capital < quantity * entry.spot:
        raise ValueError("capital must at least fund the spot leg")
    events = list(settlements)
    timestamps = [s.time_ms for s in events]
    if timestamps != sorted(set(timestamps)):
        raise ValueError("funding must be unique and ordered")
    included = [s for s in events if entry.time_ms < s.time_ms < exit.time_ms]
    funding = sum((quantity * s.mark * s.rate for s in included), Decimal(0))
    spot_pnl = quantity * (exit.spot - entry.spot)
    perp_pnl = quantity * (entry.perp - exit.perp)
    spot_turnover = quantity * (entry.spot + exit.spot)
    perp_turnover = quantity * (entry.perp + exit.perp)
    fees = spot_turnover * costs.spot_fee + perp_turnover * costs.perp_fee
    slippage = (spot_turnover + perp_turnover) * costs.slippage_bps / Decimal(10000)
    days = Decimal(exit.time_ms - entry.time_ms) / Decimal(86400000)
    capital_cost = capital * costs.annual_capital_rate * days / Decimal(365)
    net = spot_pnl + perp_pnl + funding - fees - slippage - capital_cost
    return dict(spot_pnl=spot_pnl, perp_pnl=perp_pnl,
                basis_pnl=spot_pnl + perp_pnl, funding=funding,
                fees=fees, slippage=slippage, capital_cost=capital_cost,
                net=net, return_on_capital=net / capital,
                capital=capital, days=days, settlements=len(included))
