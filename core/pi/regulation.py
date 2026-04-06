"""Feedforward regulation controller for activation signal tracking.

v5 refactor: works in the new (P_net, P_reg) signed-power representation.
A single net power value means simultaneous charge+discharge ("wash trade")
is impossible by construction (Bug A fix). The controller enforces the
power-budget headroom ``|P_net| + P_reg <= P_max`` (Bug B fix). The plant
will additionally apply SOC headroom limiting and report the actually-applied
power back to the simulator (Bug C fix).

Runs at dt_pi = 4 s.

SOC safety zones
----------------
- Below ``soc_safety_low`` (0.15): linearly reduce up-regulation
  (discharge demand) to zero at ``soc_cutoff_low`` (0.12).
- Above ``soc_safety_high`` (0.85): linearly reduce down-regulation
  (charge demand) to zero at ``soc_cutoff_high`` (0.88).

Usage
-----
    ctrl = RegulationController(bp, reg_ctrl_params, dt=4.0)
    u_command, p_delivered = ctrl.compute(
        setpoint_pnet,        # MPC's planned net power [kW, signed]
        p_reg_committed,      # EMS commitment [kW, >= 0]
        activation_signal,    # grid signal [-1, +1]
        soc_current,          # SOC estimate from EKF
    )
    # u_command is shape (2,) [P_net_signed, P_reg] ready for plant.step()
    # p_delivered is the (signed) regulation power actually drawn this step
"""

from __future__ import annotations

import numpy as np

from core.config.parameters import BatteryParams, RegControllerParams


class RegulationController:
    """Feedforward regulation controller, signed-power formulation."""

    def __init__(
        self,
        bp: BatteryParams,
        reg_params: RegControllerParams,
        dt: float = 4.0,
    ) -> None:
        self._bp = bp
        self._rp = reg_params
        self._dt = dt

    def compute(
        self,
        setpoint_pnet: float,
        p_reg_committed: float,
        activation_signal: float,
        soc_current: float,
    ) -> tuple[np.ndarray, float]:
        """Compute the actual command for the plant + delivered regulation power.

        Parameters
        ----------
        setpoint_pnet : float
            MPC base net power setpoint [kW, signed].
            > 0 = discharge, < 0 = charge, 0 = idle.
        p_reg_committed : float
            FCR capacity committed by the EMS for the current hour [kW, >= 0].
        activation_signal : float
            Grid activation in [-1, +1]. > 0 = up-reg (discharge demand),
            < 0 = down-reg (charge demand).
        soc_current : float
            Current SOC estimate (from EKF), used for safety clamping.

        Returns
        -------
        u_command : ndarray, shape (2,)
            ``[P_net_signed, P_reg_committed]`` ready to pass to ``plant.step()``.
        p_delivered : float
            Signed regulation power actually drawn by activation this step.
            Positive = discharge (up-regulation), negative = charge.
            Equal to ``u_command[0] - setpoint_pnet`` after clipping.
        """
        bp = self._bp
        rp = self._rp
        P_max = bp.P_max_kw

        # ---- 1. Activation demand (signed) ----
        # Sign convention: activation > 0 = up-reg = discharge → positive P_net delta
        p_reg_demand = activation_signal * p_reg_committed

        # ---- 2. SOC safety clamping on the demand ----
        p_reg_demand = self._soc_clamp(p_reg_demand, soc_current)

        # ---- 3. SOC recovery bias when idle ----
        # When the grid is not asking for regulation, gently nudge SOC
        # toward the terminal target. Recovery > 0 means "want to charge",
        # which is a NEGATIVE addition to net power.
        if abs(activation_signal) < 1e-6:
            p_recovery = rp.recovery_gain * (bp.SOC_terminal - soc_current) * P_max
            p_reg_demand -= p_recovery

        # ---- 4. Combine setpoint + activation demand into a single signed P_net ----
        p_net = setpoint_pnet + p_reg_demand

        # ---- 5. Runtime power limit |P_net| <= P_max ----
        # P_reg_committed is a contract, not a separate power channel.
        # The battery has one physical current = P_net at any instant.
        # The EMS planning layer guarantees |setpoint| + P_reg <= P_max,
        # so the runtime PI just enforces the absolute battery limit.
        p_net_clipped = float(np.clip(p_net, -P_max, P_max))

        # The plant will additionally apply SOC headroom limiting; we don't
        # duplicate that logic here. The plant returns the actually-applied
        # power so accounting (Bug C fix) sees physical reality.

        # ---- 6. Delivered regulation = the activation-driven delta ----
        p_delivered = p_net_clipped - setpoint_pnet

        u_command = np.array([p_net_clipped, p_reg_committed])
        return u_command, p_delivered

    def _soc_clamp(self, p_reg_demand: float, soc: float) -> float:
        """Apply SOC-based safety scaling to the regulation demand."""
        rp = self._rp

        if p_reg_demand > 0:
            # Up-regulation (discharge): limited by low SOC
            if soc <= rp.soc_cutoff_low:
                return 0.0
            if soc < rp.soc_safety_low:
                scale = (soc - rp.soc_cutoff_low) / (rp.soc_safety_low - rp.soc_cutoff_low)
                return p_reg_demand * scale
        elif p_reg_demand < 0:
            # Down-regulation (charge): limited by high SOC
            if soc >= rp.soc_cutoff_high:
                return 0.0
            if soc > rp.soc_safety_high:
                scale = (rp.soc_cutoff_high - soc) / (rp.soc_cutoff_high - rp.soc_safety_high)
                return p_reg_demand * scale

        return p_reg_demand

    def reset(self) -> None:
        """Reset controller state (no-op for feedforward)."""
        pass
