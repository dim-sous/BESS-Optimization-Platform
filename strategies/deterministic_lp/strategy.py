"""Deterministic LP strategy: commercial-baseline planner, no MPC.

The honest "what every commercial BESS EMS vendor ships" baseline.
A rolling-horizon LP over the forecast-mean prices, no closed-loop
controller. Beating this is the v5 product's job.
"""

from __future__ import annotations

from core.config.parameters import (
    BatteryParams,
    EMSParams,
    ThermalParams,
    TimeParams,
)
from core.planners.deterministic_lp import DeterministicLP
from core.simulator.strategy import Strategy


def make_strategy(
    bp: BatteryParams,
    tp: TimeParams,
    ep: EMSParams,
    thp: ThermalParams,
    **_unused,
) -> Strategy:
    return Strategy(
        name="deterministic_lp",
        planner=DeterministicLP(bp, tp, ep, thp),
        mpc=None,
        metadata={
            "label": "Commercial Baseline (LP)",
            "pitch_visible": True,
            "description": "Rolling-horizon LP over forecast-mean prices, no closed-loop controller.",
        },
    )
