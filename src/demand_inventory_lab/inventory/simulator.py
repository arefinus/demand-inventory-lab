"""Period-by-period replenishment simulation.

Event order inside period t:

1. review: if t is a review period, compute the inventory position
   (on hand + on order - backorders) and place an order due at t + L;
2. receive: all orders with due date <= t arrive (a lead time of 0 therefore
   arrives in the same period it was placed);
3. clear backorders from available stock (backorder mode only);
4. demand: fill from on-hand stock; unmet demand is lost or backordered;
5. account: holding cost on end-of-period on-hand stock.

Balance identity checked by tests for every row:
``on_hand_end == on_hand_open + received - backorders_cleared - filled``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import ConfigError
from .policy import CostAssumptions, LeadTime, Policy, Shock

LEDGER_COLUMNS = [
    "period", "demand", "on_hand_open", "ordered_qty", "lead_time", "received",
    "backorders_cleared", "filled", "unmet", "on_hand_end", "on_order_end", "backorders_end",
]


@dataclass(frozen=True)
class SimulationResult:
    ledger: pd.DataFrame
    units_demanded: int
    units_filled: int
    fill_rate: float | None
    stockout_periods: int
    period_service_level: float
    order_count: int
    holding_cost: float
    ordering_cost: float
    stockout_cost: float

    @property
    def total_cost(self) -> float:
        return self.holding_cost + self.ordering_cost + self.stockout_cost

    def summary(self) -> dict:
        return {
            "units_demanded": self.units_demanded,
            "units_filled": self.units_filled,
            "fill_rate": self.fill_rate,
            "stockout_periods": self.stockout_periods,
            "period_service_level": self.period_service_level,
            "order_count": self.order_count,
            "holding_cost": round(self.holding_cost, 4),
            "ordering_cost": round(self.ordering_cost, 4),
            "stockout_cost": round(self.stockout_cost, 4),
            "total_cost": round(self.total_cost, 4),
        }


def simulate(
    demand,
    policy: Policy,
    lead_time: LeadTime,
    costs: CostAssumptions,
    seed: int = 42,
    initial_on_hand: int | None = None,
    shock: Shock | None = None,
) -> SimulationResult:
    d = np.asarray(demand)
    if d.ndim != 1 or d.size == 0:
        raise ConfigError("demand must be a non-empty 1-D array")
    if (d < 0).any():
        raise ConfigError("demand must be non-negative")
    d = d.astype(int)
    if shock is not None:
        if shock.end > d.size:
            raise ConfigError(f"shock window end {shock.end} exceeds horizon {d.size}")
        d = shock.apply(d)

    rng = np.random.default_rng(seed)
    n = d.size
    on_hand = int(policy.initial_on_hand() if initial_on_hand is None else initial_on_hand)
    backorders = 0
    pipeline: list[tuple[int, int]] = []  # (due_period, qty)
    rows: list[dict] = []
    order_count = 0
    holding_cost = 0.0
    stockout_cost = 0.0

    for t in range(n):
        on_hand_open = on_hand
        ordered = 0
        lt = -1
        if t % policy.review_period == 0:
            on_order = sum(q for _, q in pipeline)
            position = on_hand + on_order - backorders
            ordered = policy.order_size(position)
            if ordered > 0:
                lt = lead_time.sample(rng)
                pipeline.append((t + lt, ordered))
                order_count += 1
        received = sum(q for due, q in pipeline if due <= t)
        pipeline = [(due, q) for due, q in pipeline if due > t]
        available = on_hand + received
        cleared = 0
        if costs.unmet_demand == "backorder" and backorders > 0:
            cleared = min(backorders, available)
            backorders -= cleared
            available -= cleared
        filled = min(int(d[t]), available)
        unmet = int(d[t]) - filled
        on_hand = available - filled
        if costs.unmet_demand == "backorder":
            backorders += unmet
        holding_cost += on_hand * costs.holding_cost_per_unit_period
        stockout_cost += unmet * costs.stockout_penalty_per_unit
        rows.append(
            {
                "period": t,
                "demand": int(d[t]),
                "on_hand_open": on_hand_open,
                "ordered_qty": ordered,
                "lead_time": lt,
                "received": received,
                "backorders_cleared": cleared,
                "filled": filled,
                "unmet": unmet,
                "on_hand_end": on_hand,
                "on_order_end": sum(q for _, q in pipeline),
                "backorders_end": backorders,
            }
        )

    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    demanded = int(ledger["demand"].sum())
    filled_total = int(ledger["filled"].sum())
    stockouts = int((ledger["unmet"] > 0).sum())
    return SimulationResult(
        ledger=ledger,
        units_demanded=demanded,
        units_filled=filled_total,
        fill_rate=(filled_total / demanded if demanded > 0 else None),
        stockout_periods=stockouts,
        period_service_level=1.0 - stockouts / n,
        order_count=order_count,
        holding_cost=float(holding_cost),
        ordering_cost=float(order_count * costs.order_cost),
        stockout_cost=float(stockout_cost),
    )
