"""EMS + Economic MPC — the v5 product. Stochastic EMS + economic MPC.

Stochastic EMS planner solves the two-stage scenario program hourly;
the economic MPC re-solves a deterministic 1-hour-horizon NLP every
minute against the live EKF state estimate, anchored to the EMS plan's
end-of-hour SOC via a terminal cost term.

Activation tracking lives in the plant. The strategy-layer PI controller
was deleted in the 2026-04-15 cleanup as ceremonial post-RF1.
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
    **_unused,
) -> Strategy:
    return Strategy(
        name="ems_economic_mpc",
        planner=EconomicEMS(bp, tp, ep, thp, elp),
        mpc=EconomicMPCAdapter(EconomicMPC(bp, tp, mp, thp, elp, ep)),
        metadata={
            "label": "EMS + Economic MPC",
            "pitch_visible": True,
            "description": "EMS + Economic MPC. v5 production strategy.",
        },
    )
