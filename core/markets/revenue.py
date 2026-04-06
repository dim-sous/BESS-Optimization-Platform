"""Regulation revenue, penalty, and delivery score accounting.

Tracks three revenue streams at each PI step (4s):
  - Capacity payment:  price_reg * P_reg_committed * dt  (for committing)
  - Delivery payment:  price_activation * |P_delivered| * dt  (for following signal)
  - Non-delivery penalty: penalty_mult * price_reg * |P_missed| * dt

Usage
-----
    accounting = RegulationAccounting()
    cap, dlv, pen, ok = compute_step_revenue(...)
    accounting.update(cap, dlv, pen, ok, activation != 0.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.config.parameters import RegulationParams


@dataclass
class RegulationAccounting:
    """Accumulates regulation revenue, penalties, and delivery statistics."""

    capacity_revenue: float = 0.0
    delivery_revenue: float = 0.0
    penalty_cost: float = 0.0
    total_demanded_kwh: float = 0.0
    total_delivered_kwh: float = 0.0
    total_missed_kwh: float = 0.0
    delivery_steps: int = 0
    total_active_steps: int = 0

    @property
    def delivery_score(self) -> float:
        """Fraction of active steps with successful delivery."""
        if self.total_active_steps == 0:
            return 1.0
        return self.delivery_steps / self.total_active_steps

    @property
    def net_regulation_profit(self) -> float:
        """Net regulation profit: capacity + delivery - penalty."""
        return self.capacity_revenue + self.delivery_revenue - self.penalty_cost

    def update(
        self,
        cap_rev: float,
        del_rev: float,
        penalty: float,
        is_delivered: bool,
        is_active: bool,
    ) -> None:
        """Update accumulators with one step's results."""
        self.capacity_revenue += cap_rev
        self.delivery_revenue += del_rev
        self.penalty_cost += penalty
        if is_active:
            self.total_active_steps += 1
            if is_delivered:
                self.delivery_steps += 1


def compute_step_revenue(
    P_reg_committed: float,
    activation_signal: float,
    P_delivered: float,
    price_reg_capacity: float,
    reg_params: RegulationParams,
    dt: float,
) -> tuple[float, float, float, bool]:
    """Compute single-step regulation revenue components.

    Parameters
    ----------
    P_reg_committed : float
        Committed regulation capacity [kW].
    activation_signal : float
        Grid activation signal in [-1, +1].
    P_delivered : float
        Actual regulation power delivered [kW] (signed, same direction as demand).
    price_reg_capacity : float
        Regulation capacity price [$/kW/h].
    reg_params : RegulationParams
        Revenue/penalty configuration.
    dt : float
        Time step [s].

    Returns
    -------
    cap_rev : float
        Capacity payment [$].
    del_rev : float
        Delivery payment [$].
    penalty : float
        Non-delivery penalty [$].
    is_delivered : bool
        Whether delivery was within tolerance.
    """
    dt_h = dt / 3600.0

    # Capacity payment: always earned for committing
    cap_rev = price_reg_capacity * P_reg_committed * dt_h

    # What the grid demanded
    P_demanded = abs(activation_signal * P_reg_committed)

    # What was actually delivered (absolute)
    P_delivered_abs = abs(P_delivered)

    # Delivery payment: proportional to actual delivery
    del_rev = reg_params.price_activation * P_delivered_abs * dt_h

    # Missed power
    P_missed = max(0.0, P_demanded - P_delivered_abs)

    # Penalty for non-delivery
    penalty = reg_params.penalty_mult * price_reg_capacity * P_missed * dt_h

    # Delivery success check
    if P_demanded < 1e-3:
        is_delivered = True
    else:
        is_delivered = (P_missed / P_demanded) <= reg_params.delivery_tolerance

    return cap_rev, del_rev, penalty, is_delivered
