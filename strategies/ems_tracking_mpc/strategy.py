"""EMS + Tracking MPC — controlled-experiment baseline. Stochastic EMS + tracking MPC.

NOT pitch-visible. Exists as a clean control point for the
``ems_economic_mpc`` comparison: same prediction model, same exogenous P_reg
handling, same endurance constraint, same fallback. The two strategies
differ in exactly one place — the cost function. Tracking MPC tracks
the EMS plan in (SOC, P_chg, P_dis); economic MPC replaces those
tracking terms with arbitrage profit, degradation cost, and a soft
SOC anchor.

If ems_economic_mpc and ems_tracking_mpc produce similar profits, the economic
formulation isn't pulling its weight. If ems_economic_mpc clearly wins, the
extra complexity is justified.
"""

from __future__ import annotations

from core.config.parameters import (
    BatteryParams,
    ElectricalParams,
    EMSParams,
    MPCParams,
    ThermalParams,
    TimeParams,
)
from core.mpc.adapters import TrackingMPCAdapter
from core.mpc.tracking import TrackingMPC
from core.planners.stochastic_ems import EconomicEMS
from core.simulator.strategy import Strategy


def make_strategy(
    bp: BatteryParams,
    tp: TimeParams,
    ep: EMSParams,
    mp: MPCParams,
    thp: ThermalParams,
    elp: ElectricalParams,
    **_unused,
) -> Strategy:
    return Strategy(
        name="ems_tracking_mpc",
        planner=EconomicEMS(bp, tp, ep, thp, elp),
        mpc=TrackingMPCAdapter(TrackingMPC(bp, tp, mp, thp, elp)),
        metadata={
            "label": "EMS + Tracking MPC",
            "pitch_visible": False,
            "description": "EMS + Tracking MPC. Controlled-experiment baseline for EMS + Economic MPC.",
        },
    )
