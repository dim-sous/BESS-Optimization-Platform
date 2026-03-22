"""Four-panel result visualisation for the v4 electrical RC model.

Panel layout (2 rows x 2 columns)
----------------------------------
  [0,0] Pack SOC: true vs EKF vs MHE  |  [0,1] Terminal voltage + OCV
  [1,0] SOH + RC voltages             |  [1,1] Power dispatch + price
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from config.parameters import BatteryParams, ElectricalParams, PackParams, ThermalParams
from models.battery_model import ocv_pack_numpy


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
_LW_MPC = 1.0


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
    elp: ElectricalParams,
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
        f"BESS Digital Twin — v4 Electrical RC{cell_tag} — 24 h",
        fontsize=_SUPTITLE_SIZE, fontweight="bold", y=0.995,
    )

    t_sim_h = sim["time_sim"] / 3600.0
    t_mpc_h = sim["time_mpc"] / 3600.0
    dt_mpc_s = float(sim["time_mpc"][1] - sim["time_mpc"][0]) if len(sim["time_mpc"]) > 1 else 60.0

    # ==================================================================
    #  Panel [0,0] — Pack SOC: true vs EKF vs MHE
    # ==================================================================
    ax = axes[0, 0]
    ax.plot(t_sim_h, sim["soc_true"], color="0.25", linewidth=_LW_TRUE,
            label="Pack SOC (true)")
    ax.plot(t_mpc_h, sim["soc_ekf"], color="tab:blue", linewidth=_LW_EST,
            label="EKF")
    ax.plot(t_mpc_h, sim["soc_mhe"], color="tab:red", linewidth=_LW_EST,
            linestyle="--", label="MHE")
    ax.axhspan(bp.SOC_min, bp.SOC_max, alpha=0.08, color="green",
               label=f"Limits [{bp.SOC_min:.0%}–{bp.SOC_max:.0%}]")
    ax.set_ylabel("SOC [-]")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right")
    ax.set_title("State of Charge")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [0,1] — Terminal Voltage + OCV + RC voltages  (v4 hero)
    # ==================================================================
    ax = axes[0, 1]
    vterm = sim.get("vterm_true", None)
    if vterm is not None:
        ax.plot(t_sim_h, vterm, color="tab:blue", linewidth=_LW_TRUE,
                label="V_term (true)")
        ocv_vals = ocv_pack_numpy(sim["soc_true"], elp)
        ax.plot(t_sim_h, ocv_vals, color="tab:green", linewidth=1.2,
                alpha=0.7, linestyle="--", label="OCV(SOC)")
        ax.axhline(elp.V_min_pack, color="red", linewidth=1.2, linestyle=":",
                   label=f"V_min = {elp.V_min_pack:.0f} V")
        ax.axhline(elp.V_max_pack, color="red", linewidth=1.2, linestyle=":",
                   label=f"V_max = {elp.V_max_pack:.0f} V")
    ax.set_ylabel("Voltage [V]")
    ax.legend(loc="upper left")
    ax.set_title("Pack Terminal Voltage & OCV")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # RC voltages on twin axis (small scale, right side)
    vrc1 = sim.get("vrc1_true", None)
    vrc2 = sim.get("vrc2_true", None)
    if vrc1 is not None or vrc2 is not None:
        ax2 = ax.twinx()
        if vrc1 is not None:
            ax2.plot(t_sim_h, vrc1, color="tab:cyan", linewidth=1.0,
                     alpha=0.5, label=f"V_rc1 (τ={elp.tau_1:.0f}s)")
        if vrc2 is not None:
            ax2.plot(t_sim_h, vrc2, color="tab:orange", linewidth=1.0,
                     alpha=0.5, label=f"V_rc2 (τ={elp.tau_2:.0f}s)")
        ax2.set_ylabel("RC Voltage [V]", alpha=0.6)
        ax2.tick_params(axis="y", labelsize=_TICK_SIZE)
        ax2.legend(loc="upper right", fontsize=_LEGEND_SIZE)

    # ==================================================================
    #  Panel [1,0] — SOH (clean, no twin axis)
    # ==================================================================
    ax = axes[1, 0]
    ax.plot(t_sim_h, sim["soh_true"], color="0.25", linewidth=_LW_TRUE,
            label="SOH (true)")
    ax.plot(t_mpc_h, sim["soh_ekf"], color="tab:blue", linewidth=_LW_EST,
            label="EKF")
    ax.plot(t_mpc_h, sim["soh_mhe"], color="tab:red", linewidth=_LW_EST,
            linestyle="--", label="MHE")
    ax.set_ylabel("SOH [-]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="lower left")
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
