"""Plan — uniform setpoint container produced by every planner.

The simulator's linear loop reads from a `Plan` to find the current
hour's setpoint. Planners return their internal dict (the legacy
shape with `P_chg_ref / P_dis_ref / P_reg_ref / SOC_ref / ...`); the
simulator wraps that dict into a `Plan` once per re-solve.

Power is normalised here to the **signed** convention `P_net > 0 = discharge`
so the rest of the simulator (PI controller, plant) can use a single
value instead of the legacy (chg, dis) tuple. Wash trades are impossible
to express in this representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Plan:
    """One hourly dispatch plan, valid from sim_step `start_step` onward.

    Fields are hourly arrays. Use `setpoint_at(sim_step, steps_per_hour)`
    to read the (P_net, P_reg) values active at a particular sim step.
    """
    p_net_hourly: np.ndarray          # (n_hours,)   [kW, signed]
    p_reg_hourly: np.ndarray          # (n_hours,)   [kW, >= 0]
    soc_ref_hourly: np.ndarray        # (n_hours+1,) [0..1]
    start_step: int                   # sim_step at which hour 0 begins
    expected_profit: float            # planner's forecast-evaluated profit

    @classmethod
    def from_planner_dict(cls, d: dict, start_step: int) -> "Plan":
        """Wrap a planner's output dict into a Plan.

        Converts (P_chg_ref, P_dis_ref) into a single signed P_net using
        ``P_net = P_dis - P_chg``. Both must be non-negative in the
        input dict (planners produce non-negative chg/dis).
        """
        p_chg = np.asarray(d["P_chg_ref"], dtype=float)
        p_dis = np.asarray(d["P_dis_ref"], dtype=float)
        p_net = p_dis - p_chg
        p_reg = np.asarray(d["P_reg_ref"], dtype=float)
        soc_ref = np.asarray(d["SOC_ref"], dtype=float)
        return cls(
            p_net_hourly=p_net,
            p_reg_hourly=p_reg,
            soc_ref_hourly=soc_ref,
            start_step=start_step,
            expected_profit=float(d.get("expected_profit", 0.0)),
        )

    def setpoint_at(self, sim_step: int, steps_per_hour: int) -> tuple[float, float]:
        """Return (P_net, P_reg) for the given sim_step (ZOH within hour)."""
        h = (sim_step - self.start_step) // steps_per_hour
        h = max(0, min(h, len(self.p_net_hourly) - 1))
        return float(self.p_net_hourly[h]), float(self.p_reg_hourly[h])

    def soc_anchor_at(self, sim_step: int, steps_per_hour: int) -> float:
        """End-of-current-hour SOC target (the EMS strategic anchor)."""
        h = (sim_step - self.start_step) // steps_per_hour + 1
        h = max(0, min(h, len(self.soc_ref_hourly) - 1))
        return float(self.soc_ref_hourly[h])
