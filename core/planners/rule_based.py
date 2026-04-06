"""Rule-based planner — naive price-sorted dispatch.

This is the cheapest possible "dispatcher": no optimisation, no
forecasts beyond the deterministic-mean shape, no regulation. It
charges during the lowest-price hours and discharges during the
highest, sized so the schedule is physically feasible. Useful as a
strict lower bound for the comparison harness — every other strategy
should beat it convincingly.

Implements the same `solve()` signature as the LP and stochastic EMS
planners (see `core/planners/base.py`), so the simulator can swap them
via the strategy recipe.
"""

from __future__ import annotations

import numpy as np

from core.config.parameters import BatteryParams


class RuleBasedPlanner:
    """Price-sorted dispatch — no optimisation, no regulation."""

    def __init__(self, bp: BatteryParams) -> None:
        self._bp = bp

    def solve(
        self,
        soc_init: float,
        soh_init: float,
        t_init: float,
        energy_scenarios: np.ndarray,
        reg_scenarios: np.ndarray,
        probabilities: np.ndarray,
        vrc1_init: float = 0.0,
        vrc2_init: float = 0.0,
    ) -> dict:
        """Sort the forecast-mean energy prices and dispatch the cheap/expensive
        hours. The realized prices are never seen — accounting handles those.
        """
        bp = self._bp
        n_hours = energy_scenarios.shape[1]
        # Forecast mean (no realized leak): probability-weighted across scenarios
        forecast_mean_e = energy_scenarios.T @ probabilities

        prices = forecast_mean_e[:n_hours]
        order = np.argsort(prices)

        p_chg = np.zeros(n_hours)
        p_dis = np.zeros(n_hours)

        # Size schedule to physically deliverable energy
        usable_kwh = (bp.SOC_max - bp.SOC_min) * bp.E_nom_kwh
        power = bp.P_max_kw * 0.8
        n_hours_needed = int(np.ceil(usable_kwh / power))
        n_charge = min(n_hours_needed, n_hours // 3)
        n_discharge = min(n_hours_needed, n_hours // 3)

        charge_hours = order[:n_charge]
        discharge_hours = order[-n_discharge:]
        # Only commit if there's a profitable spread on the forecast
        if prices[discharge_hours[-1]] > prices[charge_hours[-1]]:
            p_chg[charge_hours] = power
            p_dis[discharge_hours] = power

        return {
            "P_chg_ref": p_chg,
            "P_dis_ref": p_dis,
            "P_reg_ref": np.zeros(n_hours),
            "SOC_ref": np.full(n_hours + 1, soc_init),
            "SOH_ref": np.full(n_hours + 1, soh_init),
            "TEMP_ref": np.full(n_hours + 1, t_init),
            "VRC1_ref": np.zeros(n_hours + 1),
            "VRC2_ref": np.zeros(n_hours + 1),
            "expected_profit": 0.0,
        }
