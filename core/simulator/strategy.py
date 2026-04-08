"""Strategy — composition recipe for the linear simulator.

A Strategy is a frozen dataclass that names which planner and (optional)
MPC the simulator should use. The simulator's main loop has zero
strategy-specific branches: it just calls `strategy.planner.solve(...)`
and optionally `strategy.mpc.solve(...)`.

Activation tracking lives in the plant (RF1, 2026-04-15). There is no
strategy-layer PI controller — the cleanup commit on 2026-04-15 deleted
the PI class, the `pi_enabled` flag, and all `_no_pi` strategy variants
because empirically they were ceremonial post-RF1 (PI on/off deltas
≤ $0.03/day across all regimes).

Adding a new strategy means writing a new file under `strategies/<name>/`
that returns a `Strategy(...)` instance. No simulator changes required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np


class _PlannerLike(Protocol):
    def solve(
        self,
        soc_init: float,
        soh_init: float,
        t_init: float,
        energy_scenarios: np.ndarray,
        reg_scenarios: np.ndarray,
        probabilities: np.ndarray,
        vrc1_init: float = 0.0,
        vrc2_init: float = 0.0,
    ) -> dict: ...


class _MPCLike(Protocol):
    """Loose protocol — MPC implementations have varied signatures.
    The simulator handles the dispatch via duck-typing on attribute names.
    Required: a callable named `solve` returning a 3-vector
    [P_chg, P_dis, P_reg].
    Optional: `last_solve_failed` flag.
    """
    def solve(self, *args, **kwargs) -> np.ndarray: ...


@dataclass(frozen=True)
class Strategy:
    """A composition of (planner, mpc) with metadata.

    `mpc` can be `None`. Strategies that omit `mpc` use the planner's
    hourly setpoint directly at every plant step (the simulator
    dispatches them via the `_open_loop_dispatch` helper, which is now
    a trivial passthrough since the plant handles activation
    internally).
    """
    name: str
    planner: _PlannerLike
    mpc: Optional[_MPCLike] = None
    metadata: dict[str, Any] = field(default_factory=dict)
