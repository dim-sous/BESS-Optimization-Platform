"""EMS_CLAMPS — sanity check: stochastic EMS planner + open-loop dispatch.

NOT pitch-visible. This isolates the value of the stochastic EMS planner
on its own (vs the deterministic LP) by running it through the same
open-loop execution path as DETERMINISTIC_LP. If EMS_CLAMPS doesn't beat
DETERMINISTIC_LP, the stochastic formulation isn't adding value beyond
what perfect-foresight-mean already gets.
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
        pi=None,
        metadata={
            "label": "Stochastic EMS (sanity)",
            "pitch_visible": False,
            "description": "Stochastic EMS planner + open-loop dispatch (sanity check).",
        },
    )
