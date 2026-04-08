"""Rule-based strategy: naive price-sorted dispatch, no regulation, no MPC.

This is the strict lower bound for the comparison. It demonstrates what
a buyer would get from a basic in-house dispatcher with no optimization
and no FCR participation. Every other strategy should beat it convincingly.
"""

from __future__ import annotations

from core.config.parameters import BatteryParams
from core.planners.rule_based import RuleBasedPlanner
from core.simulator.strategy import Strategy


def make_strategy(bp: BatteryParams, **_unused) -> Strategy:
    """Build the rule_based Strategy recipe.

    Extra kwargs are accepted (and ignored) so the comparison harness
    can pass the same params to every strategy factory uniformly.
    """
    return Strategy(
        name="rule_based",
        planner=RuleBasedPlanner(bp),
        mpc=None,
        metadata={
            "label": "Rule-Based",
            "pitch_visible": True,
            "description": "Naive price-sorted schedule, no FCR, no MPC.",
        },
    )
