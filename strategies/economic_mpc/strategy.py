"""ECONOMIC_MPC — the v5 product. Stochastic EMS + economic MPC.

This is the production v5 strategy. Stochastic EMS planner solves the
two-stage scenario program hourly; the economic MPC re-solves a
deterministic 1-hour-horizon optimization every minute against the
live EKF state estimate, anchored to the EMS plan's SOC reference.

Activation tracking lives in the plant (RF1, 2026-04-15). The
2026-04-15 cleanup deleted the strategy-layer PI controller because
it had become ceremonial post-RF1.
"""

from __future__ import annotations

from core.config.parameters import (
    BatteryParams,
    ElectricalParams,
    EMSParams,
    MPCParams,
    RegulationParams,
    ThermalParams,
    TimeParams,
)
from core.mpc.adapters import EconomicMPCAdapter
from core.mpc.economic import EconomicMPC
from core.planners.stochastic_ems import EconomicEMS
from core.simulator.strategy import Strategy


def make_strategy(
    bp: BatteryParams,
    tp: TimeParams,
    ep: EMSParams,
    mp: MPCParams,
    thp: ThermalParams,
    elp: ElectricalParams,
    reg_p: RegulationParams,
    **_unused,
) -> Strategy:
    return Strategy(
        name="economic_mpc",
        planner=EconomicEMS(bp, tp, ep, thp, elp),
        mpc=EconomicMPCAdapter(EconomicMPC(bp, tp, mp, thp, elp, ep, reg_p)),
        metadata={
            "label": "Economic MPC (v5)",
            "pitch_visible": True,
            "description": "Stochastic EMS + economic MPC. v5 product.",
        },
    )
