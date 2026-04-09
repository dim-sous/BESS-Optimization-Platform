"""ems — canonical "EMS alone" strategy.

Stochastic EMS planner solving the two-stage scenario program hourly,
plus a trivial passthrough dispatch (the plant handles activation
internally post-RF1, so "open-loop dispatch" is literally a no-op
forward of the EMS hourly setpoint).

Renamed from `ems_clamps` on 2026-04-09: there are no clamps anywhere
in this strategy and the old name was a historical artifact. This is
the canonical "EMS alone, no MPC layer" baseline. The current pitch
hypothesis is that ems_economic_mpc must be strictly >= ems on every
metric to justify the MPC layer's existence.
"""

from __future__ import annotations

from core.config.parameters import (
    BatteryParams,
    ElectricalParams,
    EMSParams,
    ThermalParams,
    TimeParams,
)
from core.planners.stochastic_ems import EconomicEMS
from core.simulator.strategy import Strategy


def make_strategy(
    bp: BatteryParams,
    tp: TimeParams,
    ep: EMSParams,
    thp: ThermalParams,
    elp: ElectricalParams,
    **_unused,
) -> Strategy:
    return Strategy(
        name="ems",
        planner=EconomicEMS(bp, tp, ep, thp, elp),
        mpc=None,
        metadata={
            "label": "Stochastic EMS (alone)",
            "pitch_visible": True,
            "description": "Stochastic EMS planner + trivial dispatch. Canonical 'EMS alone' baseline.",
        },
    )
