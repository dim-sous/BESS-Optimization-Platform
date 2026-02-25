"""Closed-loop simulation of the hierarchical BESS control system.

Runs the tracking MPC controller in a receding-horizon loop against the
high-fidelity battery plant model, while the plant tracks the economic
schedule produced by the EMS layer.

The simulation faithfully captures:
  - Model mismatch between the MPC prediction model and the real plant.
  - SOC saturation effects (the plant clips power at SOC limits).
  - Warm-starting dynamics of the MPC solver across time steps.
"""

import logging

import numpy as np

from config import BatteryParams, MPCParams, SimParams
from control.mpc_controller import MPCController
from models.battery_model import BatteryModel

logger = logging.getLogger(__name__)


def run_simulation(
    bp: BatteryParams,
    mp: MPCParams,
    sp: SimParams,
    ems_result: dict,
    prices: np.ndarray,
) -> dict:
    """Execute the closed-loop MPC simulation.

    Parameters
    ----------
    bp : BatteryParams
        Battery parameters (shared by plant and controller).
    mp : MPCParams
        MPC tuning parameters.
    sp : SimParams
        Simulation length and settings.
    ems_result : dict
        Output of :class:`EMSOptimizer.solve` — must contain
        ``P_ref`` and ``SOC_ref``.
    prices : np.ndarray
        Electricity spot prices [$/kWh] for profit accounting
        (length ≥ ``n_steps``).

    Returns
    -------
    dict
        ``time``              – time vector [hours], length N+1
        ``soc``               – SOC trajectory [-], length N+1
        ``power``             – applied power [kW], length N
        ``cumulative_profit`` – running profit [$], length N
        ``profit``            – total profit [$]
    """
    battery = BatteryModel(bp)
    mpc = MPCController(bp, mp)

    N_sim  = sp.n_steps
    P_ref  = ems_result["P_ref"]
    SOC_ref = ems_result["SOC_ref"]

    # Pre-allocate log arrays
    soc_log    = np.zeros(N_sim + 1)
    power_log  = np.zeros(N_sim)
    profit_log = np.zeros(N_sim)

    soc_log[0] = battery.soc

    for k in range(N_sim):
        # --- Reference window for the MPC horizon ---
        soc_ref_k = SOC_ref[k:]
        p_ref_k   = P_ref[k:]

        # --- Solve MPC (receding horizon) ---
        p_cmd = mpc.solve(battery.soc, soc_ref_k, p_ref_k)

        # --- Apply command to the real plant ---
        p_actual, soc_new = battery.step(p_cmd)

        # --- Log results ---
        soc_log[k + 1] = soc_new
        power_log[k]   = p_actual
        profit_log[k]  = prices[k] * p_actual * bp.dt_hours

        logger.debug(
            "k=%2d | SOC=%.3f | P_cmd=%+7.1f kW | P_act=%+7.1f kW | Δ$=%+.2f",
            k, soc_new, p_cmd, p_actual, profit_log[k],
        )

    total_profit = float(np.sum(profit_log))
    logger.info("Simulation complete — total profit: $%.2f", total_profit)

    return {
        "time":              np.arange(N_sim + 1) * bp.dt_hours,
        "soc":               soc_log,
        "power":             power_log,
        "cumulative_profit": np.cumsum(profit_log),
        "profit":            total_profit,
    }
