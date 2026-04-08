"""EMS_CLAMPS — canonical "EMS alone" strategy.

Stochastic EMS planner solving the two-stage scenario program hourly,
plus a trivial passthrough dispatch (the plant handles activation
internally post-RF1, so "open-loop dispatch" is literally a no-op
forward of the EMS hourly setpoint).

Renamed semantically by the 2026-04-15 cleanup: previously called a
"sanity check" against deterministic_lp, this is now the canonical
"EMS alone, no MPC layer" strategy. The current pitch hypothesis is
that economic_mpc must be strictly >= ems_clamps on every metric to
justify the MPC layer's existence.
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
        name="ems_clamps",
        planner=EconomicEMS(bp, tp, ep, thp, elp),
        mpc=None,
        metadata={
            "label": "Stochastic EMS (alone)",
            "pitch_visible": True,
            "description": "Stochastic EMS planner + trivial dispatch. Canonical 'EMS alone' baseline.",
        },
    )
