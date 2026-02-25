"""Result visualisation for the Battery Optimisation Platform.

Generates a four-panel summary figure:
  1. State of charge — MPC actual vs. EMS reference
  2. Battery power   — MPC actual vs. EMS reference
  3. Electricity spot price
  4. Cumulative arbitrage profit
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from config import BatteryParams


def plot_results(
    sim: dict,
    ems: dict,
    prices: np.ndarray,
    bp: BatteryParams,
    save_path: str = "results.png",
) -> None:
    """Generate and save the four-panel summary figure.

    Parameters
    ----------
    sim : dict
        Closed-loop simulation results (from ``run_simulation``).
    ems : dict
        EMS optimiser output (from ``EMSOptimizer.solve``).
    prices : np.ndarray
        Electricity prices used in the simulation [$/kWh].
    bp : BatteryParams
        Battery parameters (for axis limits and labels).
    save_path : str
        Output file path for the saved figure.
    """
    t_sim  = sim["time"]
    t_ctrl = t_sim[:-1]                                  # power time axis
    t_price = np.arange(len(prices)) * bp.dt_hours

    N_ems = len(ems["P_ref"])
    t_ems_ctrl = np.arange(N_ems) * bp.dt_hours
    t_ems_soc  = np.arange(len(ems["SOC_ref"])) * bp.dt_hours

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(
        "Battery Energy Storage Optimisation — Hierarchical Control Results",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # ---- Panel 1: State of Charge ----
    ax = axes[0]
    ax.plot(t_sim, sim["soc"], "b-", lw=2, label="MPC (actual)")
    ax.plot(t_ems_soc, ems["SOC_ref"], "r--", lw=1.5, label="EMS (reference)")
    ax.axhline(bp.SOC_min, color="0.5", ls=":", lw=0.8, label="SOC limits")
    ax.axhline(bp.SOC_max, color="0.5", ls=":", lw=0.8)
    ax.fill_between(t_sim, bp.SOC_min, bp.SOC_max, alpha=0.05, color="green")
    ax.set_ylabel("SOC [–]")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("State of Charge", fontsize=10)

    # ---- Panel 2: Power ----
    ax = axes[1]
    ax.step(t_ctrl, sim["power"], "b-", lw=2, where="post",
            label="MPC (actual)")
    ax.step(t_ems_ctrl, ems["P_ref"], "r--", lw=1.5, where="post",
            label="EMS (reference)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Power [kW]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Battery Power  (positive = discharge to grid)", fontsize=10)

    # ---- Panel 3: Electricity Price ----
    ax = axes[2]
    ax.step(t_price, prices, color="seagreen", lw=1.8, where="post")
    ax.set_ylabel("Price [$/kWh]")
    ax.set_title("Electricity Spot Price", fontsize=10)

    # ---- Panel 4: Cumulative Profit ----
    ax = axes[3]
    ax.plot(t_ctrl, sim["cumulative_profit"], "m-", lw=2)
    ax.set_ylabel("Cumulative Profit [$]")
    ax.set_xlabel("Time [hours]")
    ax.set_title(
        f"Cumulative Profit  (total: ${sim['profit']:.2f})", fontsize=10,
    )

    # ---- Formatting (all panels) ----
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {save_path}")
    plt.close(fig)
