"""Discrete-time battery energy storage system model for closed-loop simulation.

This module implements the 'plant' — a high-fidelity battery model with
asymmetric charge/discharge efficiency and SOC-limited power saturation.
It is intentionally kept separate from the simplified prediction models
used inside the optimizers so that the simulation captures real-world
model mismatch.
"""

import numpy as np

from config import BatteryParams


class BatteryModel:
    """High-fidelity discrete-time BESS plant model.

    Sign convention
    ---------------
    - ``power > 0`` → discharge (energy flows to grid)
    - ``power < 0`` → charge   (energy flows from grid)

    Parameters
    ----------
    params : BatteryParams
        Battery configuration.
    """

    def __init__(self, params: BatteryParams) -> None:
        self.params = params
        self.soc: float = params.SOC_init

    def step(self, power_cmd: float) -> tuple[float, float]:
        """Apply a power command and advance one discrete time step.

        If the commanded power would violate SOC limits, the power is
        saturated and the actually-delivered power is returned.

        Parameters
        ----------
        power_cmd : float
            Commanded power [kW].

        Returns
        -------
        power_actual : float
            Power actually delivered after SOC saturation [kW].
        soc_new : float
            Updated state of charge [-].
        """
        p = self.params
        power = float(np.clip(power_cmd, -p.P_max_kw, p.P_max_kw))

        # --- Compute unconstrained SOC change (asymmetric efficiency) ---
        if power >= 0.0:
            # Discharge: battery must supply P/eta internally
            delta_soc = -power * p.dt_hours / (p.eta_discharge * p.E_max_kwh)
        else:
            # Charge: only eta fraction of grid power is stored
            delta_soc = -power * p.eta_charge * p.dt_hours / p.E_max_kwh

        soc_new = self.soc + delta_soc

        # --- Enforce SOC limits; back-compute actual delivered power ---
        if soc_new < p.SOC_min:
            soc_new = p.SOC_min
            # Maximum discharge power given remaining energy
            power = (
                (self.soc - p.SOC_min)
                * p.eta_discharge
                * p.E_max_kwh
                / p.dt_hours
            )
        elif soc_new > p.SOC_max:
            soc_new = p.SOC_max
            # Maximum charge power given remaining capacity
            power = -(
                (p.SOC_max - self.soc)
                * p.E_max_kwh
                / (p.eta_charge * p.dt_hours)
            )

        self.soc = soc_new
        return power, soc_new

    def reset(self, soc: float | None = None) -> None:
        """Reset SOC to a specified value (default: ``SOC_init``)."""
        self.soc = soc if soc is not None else self.params.SOC_init
