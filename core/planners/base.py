"""Planner protocol — uniform interface for the linear simulator.

Every planner (rule-based, deterministic LP, stochastic EMS) implements
the same `solve()` signature so the simulator can swap them via the
strategy recipe without any conditional logic.

The planner returns a dict with the legacy keys (P_chg_ref, P_dis_ref,
P_reg_ref, SOC_ref, SOH_ref, TEMP_ref, expected_profit). The simulator
wraps the dict into a `Plan` (see `core/planners/plan.py`) before
handing it to the rest of the loop.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Planner(Protocol):
    """Planner interface contract.

    Every planner solves an hourly dispatch problem given the current
    state estimate and a forecast scenario set. It MUST NOT see the
    realized prices — those are reserved for the accounting layer.
    """

    def solve(
        self,
        soc_init: float,
        soh_init: float,
        t_init: float,
        energy_scenarios: np.ndarray,        # (n_scenarios, n_hours) [$/kWh]
        reg_scenarios: np.ndarray,           # (n_scenarios, n_hours) [$/kW/h]
        probabilities: np.ndarray,           # (n_scenarios,)
        vrc1_init: float = 0.0,              # ignored by most planners
        vrc2_init: float = 0.0,              # ignored by most planners
    ) -> dict:
        """Return a dict with hourly setpoints + state references.

        Required keys:
            P_chg_ref       (n_hours,)   [kW, >= 0]
            P_dis_ref       (n_hours,)   [kW, >= 0]
            P_reg_ref       (n_hours,)   [kW, >= 0]
            SOC_ref         (n_hours+1,) [0..1]
            SOH_ref         (n_hours+1,) [0..1]
            TEMP_ref        (n_hours+1,) [degC]
            expected_profit float        [$]
        """
        ...
