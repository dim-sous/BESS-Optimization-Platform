"""Fast regulation controller for FCR activation signal tracking.

Runs at dt_pi = 4s.  Receives the MPC base power setpoint and the grid's
activation signal, computes the actual power command for the plant.

The activation signal in [-1, +1] is scaled by the committed regulation
capacity to produce a demanded regulation power.  This controller
modifies the base charge/discharge setpoints to deliver this power,
subject to SOC safety clamping and power limits.

Implementation: direct feedforward (no PI dynamics needed since power
adjustment is instantaneous — there is no plant lag at this level).

SOC safety zones
----------------
  - Below soc_safety_low (0.15): linearly reduce up-regulation
    (discharging) to zero at soc_cutoff_low (0.12).
  - Above soc_safety_high (0.85): linearly reduce down-regulation
    (charging) to zero at soc_cutoff_high (0.88).

Usage
-----
    pi = RegulationPI(bp, pi_params, dt=4.0)
    u_actual, P_delivered = pi.compute(P_chg_base, P_dis_base,
                                        P_reg_committed, activation, soc)
"""

from __future__ import annotations

import numpy as np

from config.parameters import BatteryParams, PIParams


class RegulationPI:
    """Fast regulation controller for FCR activation delivery."""

    def __init__(
        self,
        bp: BatteryParams,
        pi_params: PIParams,
        dt: float = 4.0,
    ) -> None:
        self._bp = bp
        self._pp = pi_params
        self._dt = dt

    def compute(
        self,
        P_chg_base: float,
        P_dis_base: float,
        P_reg_committed: float,
        activation_signal: float,
        soc_current: float,
    ) -> tuple[np.ndarray, float]:
        """Compute actual power command and delivered regulation power.

        Parameters
        ----------
        P_chg_base : float
            MPC charge setpoint [kW] (>= 0).
        P_dis_base : float
            MPC discharge setpoint [kW] (>= 0).
        P_reg_committed : float
            Committed regulation capacity [kW] (>= 0).
        activation_signal : float
            Grid activation in [-1, +1].
        soc_current : float
            Current SOC estimate for safety clamping.

        Returns
        -------
        u_actual : ndarray (3,)
            [P_chg_actual, P_dis_actual, P_reg_committed].
        P_delivered : float
            Actual regulation power delivered [kW] (signed).
            Positive = discharge (up-regulation), negative = charge (down-regulation).
        """
        bp = self._bp
        P_max = bp.P_max_kw

        # Demanded regulation power (signed)
        P_reg_demand = activation_signal * P_reg_committed

        # SOC safety clamping
        P_reg_demand = self._soc_clamp(P_reg_demand, soc_current)

        # Apply regulation to base setpoints via direct feedforward
        P_chg = P_chg_base
        P_dis = P_dis_base

        if P_reg_demand >= 0:
            # Up-regulation: discharge more (or charge less)
            # First try increasing discharge
            P_dis = P_dis + P_reg_demand
            if P_dis > P_max:
                # Can't discharge more — reduce charging instead
                overshoot = P_dis - P_max
                P_dis = P_max
                P_chg = max(0.0, P_chg - overshoot)
        else:
            # Down-regulation: charge more (or discharge less)
            # First try increasing charge
            P_chg = P_chg + abs(P_reg_demand)
            if P_chg > P_max:
                # Can't charge more — reduce discharge instead
                overshoot = P_chg - P_max
                P_chg = P_max
                P_dis = max(0.0, P_dis - overshoot)

        # Enforce power limits
        P_chg = float(np.clip(P_chg, 0.0, P_max))
        P_dis = float(np.clip(P_dis, 0.0, P_max))

        # Actual delivered regulation: net power change from base
        P_delivered = (P_dis - P_dis_base) - (P_chg - P_chg_base)

        u_actual = np.array([P_chg, P_dis, P_reg_committed])
        return u_actual, P_delivered

    def _soc_clamp(self, P_reg_demand: float, soc: float) -> float:
        """Apply SOC-based safety scaling to regulation demand.

        Up-regulation (discharge, P_reg_demand > 0): limited by low SOC.
        Down-regulation (charge, P_reg_demand < 0): limited by high SOC.
        """
        pp = self._pp

        if P_reg_demand > 0:
            # Up-regulation (discharge): limited by low SOC
            if soc <= pp.soc_cutoff_low:
                return 0.0
            elif soc < pp.soc_safety_low:
                scale = (soc - pp.soc_cutoff_low) / (pp.soc_safety_low - pp.soc_cutoff_low)
                return P_reg_demand * scale
        elif P_reg_demand < 0:
            # Down-regulation (charge): limited by high SOC
            if soc >= pp.soc_cutoff_high:
                return 0.0
            elif soc > pp.soc_safety_high:
                scale = (pp.soc_cutoff_high - soc) / (pp.soc_cutoff_high - pp.soc_safety_high)
                return P_reg_demand * scale

        return P_reg_demand

    def reset(self) -> None:
        """Reset controller state (no-op for feedforward)."""
        pass
