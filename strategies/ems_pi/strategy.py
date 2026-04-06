"""EMS_PI — sanity check: stochastic EMS planner + PI controller, no MPC.

NOT pitch-visible. This isolates the value of the PI feedback layer alone
(without MPC trajectory management). If FULL_TRACKING and ECONOMIC_MPC don't
meaningfully beat EMS_PI, the MPC layer isn't adding value beyond what
PI feedback already gets.
"""

from __future__ import annotations

from core.config.parameters import (
    BatteryParams,
    ElectricalParams,
    EMSParams,
    RegControllerParams,
    ThermalParams,
    TimeParams,
)
from core.pi.regulation import RegulationController
from core.planners.stochastic_ems import EconomicEMS
from core.simulator.strategy import Strategy


def make_strategy(
    bp: BatteryParams,
    tp: TimeParams,
    ep: EMSParams,
    thp: ThermalParams,
    elp: ElectricalParams,
    reg_ctrl_p: RegControllerParams,
    **_unused,
) -> Strategy:
    return Strategy(
        name="ems_pi",
        planner=EconomicEMS(bp, tp, ep, thp, elp),
        mpc=None,
        pi=RegulationController(bp, reg_ctrl_p, tp.dt_pi),
        metadata={
            "label": "EMS + PI (sanity)",
            "pitch_visible": False,
            "description": "Stochastic EMS + PI feedback (sanity check, no MPC).",
        },
    )
