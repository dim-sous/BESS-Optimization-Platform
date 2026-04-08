"""TRACKING_MPC — sanity control. Stochastic EMS + tracking MPC.

NOT pitch-visible. The "old v5 stack". Demonstrates how a tracking-only
MPC compares to the new economic MPC — if they're equal, the economic
formulation isn't doing useful work.

Has a known F2 bug from the MPC pipeline audit (dead `P_reg` decision
variable that the adapter discards). Kept as an empirical control;
the bug is intentionally not fixed because the strategy's only role
is "be a baseline that economic_mpc must demonstrably beat."
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
        name="tracking_mpc",
        planner=EconomicEMS(bp, tp, ep, thp, elp),
        mpc=TrackingMPCAdapter(TrackingMPC(bp, tp, mp, thp, elp)),
        metadata={
            "label": "Tracking MPC (sanity)",
            "pitch_visible": False,
            "description": "EMS + tracking MPC. Sanity control, not pitch.",
        },
    )
