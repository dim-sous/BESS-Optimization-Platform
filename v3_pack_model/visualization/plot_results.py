"""Four-panel result visualisation for the v3 pack model.

Panel layout (2 rows x 2 columns)
----------------------------------
  [0,0] Pack SOC + cell spread         |  [0,1] Cell temperatures + imbalance
  [1,0] SOH + per-cell variation       |  [1,1] Power dispatch + price
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from config.parameters import BatteryParams, PackParams, ThermalParams


# ---------------------------------------------------------------------------
#  Global style constants
# ---------------------------------------------------------------------------
_TITLE_SIZE = 15
_SUPTITLE_SIZE = 17
_LABEL_SIZE = 12
_TICK_SIZE = 11
_LEGEND_SIZE = 9
_LW_TRUE = 2.2
_LW_EST = 1.6
_LW_CELL = 1.0
_LW_MPC = 1.0
_ALPHA_CELL = 0.7


def _stepify(
    t: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand (t, y) into step-plot coordinates for use with fill_between."""
    n = len(t)
    dt = float(t[1] - t[0]) if n > 1 else 1.0
    t_s = np.empty(2 * n)
    y_s = np.empty(2 * n)
    t_s[0::2] = t
    t_s[1::2] = np.append(t[1:], t[-1] + dt)
    y_s[0::2] = y
    y_s[1::2] = y
    return t_s, y_s


def plot_results(
    sim: dict,
    bp: BatteryParams,
    thp: ThermalParams,
    pp: PackParams | None = None,
    save_path: str = "results.png",
) -> None:
    """Generate the four-panel summary figure."""
    plt.rcParams.update({
        "font.size": _TICK_SIZE,
        "axes.titlesize": _TITLE_SIZE,
        "axes.labelsize": _LABEL_SIZE,
        "xtick.labelsize": _TICK_SIZE,
        "ytick.labelsize": _TICK_SIZE,
        "legend.fontsize": _LEGEND_SIZE,
    })

    n_cells = sim.get("n_cells", 1)
    has_cell_data = "cell_socs" in sim and n_cells > 1

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    cell_tag = f" ({n_cells} cells)" if has_cell_data else ""
    fig.suptitle(
        f"BESS Digital Twin — v3 Pack Model{cell_tag} — 24 h",
        fontsize=_SUPTITLE_SIZE, fontweight="bold", y=0.995,
    )

    t_sim_h = sim["time_sim"] / 3600.0
    t_mpc_h = sim["time_mpc"] / 3600.0
    dt_mpc_s = float(sim["time_mpc"][1] - sim["time_mpc"][0]) if len(sim["time_mpc"]) > 1 else 60.0

    # ==================================================================
    #  Panel [0,0] — Pack SOC + individual cell SOCs
    # ==================================================================
    ax = axes[0, 0]
    ax.plot(t_sim_h, sim["soc_true"], color="0.25", linewidth=_LW_TRUE,
            label="Pack SOC (true)")
    ax.plot(t_mpc_h, sim["soc_ekf"], color="tab:blue", linewidth=_LW_EST,
            label="EKF")
    if np.any(sim["soc_mhe"] != 0):
        ax.plot(t_mpc_h, sim["soc_mhe"], color="tab:red", linewidth=_LW_EST,
                linestyle="--", label="MHE")
    if has_cell_data:
        cell_colors = plt.cm.Set2(np.linspace(0, 1, n_cells))
        for i in range(n_cells):
            ax.plot(t_sim_h, sim["cell_socs"][i], color=cell_colors[i],
                    linewidth=_LW_CELL, alpha=_ALPHA_CELL,
                    label=f"Cell {i+1}" if i < 4 else None)
    ax.axhspan(bp.SOC_min, bp.SOC_max, alpha=0.08, color="green",
               label=f"Limits [{bp.SOC_min:.0%}–{bp.SOC_max:.0%}]")
    ax.set_ylabel("SOC [-]")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("Pack & Cell SOC")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [0,1] — Cell Temperatures + SOC imbalance  (v3 hero)
    # ==================================================================
    ax = axes[0, 1]
    if has_cell_data:
        for i in range(n_cells):
            ax.plot(t_sim_h, sim["cell_temps"][i], color=cell_colors[i],
                    linewidth=_LW_CELL + 0.3, alpha=0.85,
                    label=f"Cell {i+1}" if i < 4 else None)
    else:
        ax.plot(t_sim_h, sim["temp_true"], color="0.25", linewidth=_LW_TRUE,
                label="Temperature")
    ax.axhline(thp.T_max, color="red", linewidth=1.2, linestyle=":",
               label=f"T_max = {thp.T_max:.0f} °C")
    ax.axhline(thp.T_ambient, color="green", linewidth=1.0, linestyle=":",
               alpha=0.6, label=f"T_amb = {thp.T_ambient:.0f} °C")
    ax.set_ylabel("Temperature [°C]")
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("Cell Temperatures")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # SOC imbalance on twin axis
    if has_cell_data:
        ax2 = ax.twinx()
        soc_imb = sim["soc_imbalance"]
        ax2.plot(t_sim_h, soc_imb * 100, color="k", linewidth=1.4,
                 linestyle="--", alpha=0.5, label="SOC spread")
        ax2.set_ylabel("SOC Spread [%]", alpha=0.6)
        ax2.tick_params(axis="y", labelsize=_TICK_SIZE)

    # ==================================================================
    #  Panel [1,0] — SOH (pack + per-cell)
    # ==================================================================
    ax = axes[1, 0]
    ax.plot(t_sim_h, sim["soh_true"], color="0.25", linewidth=_LW_TRUE,
            label="Pack SOH (true)")
    ax.plot(t_mpc_h, sim["soh_ekf"], color="tab:blue", linewidth=_LW_EST,
            label="EKF")
    if np.any(sim["soh_mhe"] != 0):
        ax.plot(t_mpc_h, sim["soh_mhe"], color="tab:red", linewidth=_LW_EST,
                linestyle="--", label="MHE")
    if has_cell_data and "cell_sohs" in sim:
        for i in range(n_cells):
            ax.plot(t_sim_h, sim["cell_sohs"][i], color=cell_colors[i],
                    linewidth=_LW_CELL, alpha=_ALPHA_CELL,
                    label=f"Cell {i+1}" if i < 4 else None)
    ax.set_ylabel("SOH [-]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="lower left", ncol=2)
    ax.set_title("State of Health")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [1,1] — Power Dispatch + Price
    # ==================================================================
    ax = axes[1, 1]
    n_mpc = len(sim["power_applied"])
    t_pow = np.arange(n_mpc) * dt_mpc_s / 3600.0
    net_grid_power = sim["power_applied"][:, 1] - sim["power_applied"][:, 0]

    ts, ys = _stepify(t_pow, net_grid_power)
    ax.fill_between(ts, ys, 0, where=(ys >= 0),
                    color="tab:green", alpha=0.25, label="Selling",
                    interpolate=True)
    ax.fill_between(ts, ys, 0, where=(ys < 0),
                    color="tab:red", alpha=0.25, label="Buying",
                    interpolate=True)
    ax.plot(ts, ys, color="k", linewidth=0.9, label="Net grid power")

    total = sim.get("total_profit", 0.0)
    ax.annotate(f"Net profit: ${total:.2f}", xy=(0.02, 0.95),
                xycoords="axes fraction", fontsize=_LEGEND_SIZE + 1,
                fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7))

    ax.axhline(0, color="k", linewidth=0.6, linestyle=":")
    ax.set_ylabel("Power [kW]  (+ sell / − buy)")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="upper right")
    ax.set_title("Grid Power Dispatch")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    prices_e = sim.get("prices_energy", None)
    if prices_e is not None:
        n_price_hours = min(len(prices_e), int(t_sim_h[-1]) + 1)
        t_price = np.arange(n_price_hours)
        ax2 = ax.twinx()
        ax2.step(t_price, prices_e[:n_price_hours], where="post",
                 color="tab:purple", linewidth=1.0, alpha=0.4, linestyle="-.")
        ax2.set_ylabel("Price [$/kWh]", color="tab:purple", alpha=0.6)
        ax2.tick_params(axis="y", colors="tab:purple", labelsize=_TICK_SIZE)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Results saved to {save_path}")
