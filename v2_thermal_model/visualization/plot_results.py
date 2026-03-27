"""Four-panel result visualisation for the v2 thermal model.

Panel layout (2 rows x 2 columns)
----------------------------------
  [0,0] SOC: true vs EKF vs MHE       |  [0,1] Temperature: true vs EKF/MHE
  [1,0] SOH: true vs EKF vs MHE       |  [1,1] Power dispatch + price
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from config.parameters import BatteryParams, ThermalParams


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
_ALPHA_REF = 0.45


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
    save_path: str = "results.png",
) -> None:
    """Generate the four-panel summary figure.

    Parameters
    ----------
    sim : dict
        Output from ``MultiRateSimulator.run()``.
    bp  : BatteryParams
    thp : ThermalParams
    save_path : str
    """
    plt.rcParams.update({
        "font.size": _TICK_SIZE,
        "axes.titlesize": _TITLE_SIZE,
        "axes.labelsize": _LABEL_SIZE,
        "xtick.labelsize": _TICK_SIZE,
        "ytick.labelsize": _TICK_SIZE,
        "legend.fontsize": _LEGEND_SIZE,
    })

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(
        "BESS Digital Twin — v2 Thermal Model — 24 h",
        fontsize=_SUPTITLE_SIZE, fontweight="bold", y=0.995,
    )

    t_sim_h = sim["time_sim"] / 3600.0
    t_mpc_h = sim["time_mpc"] / 3600.0
    dt_mpc_s = (sim["time_mpc"][1] - sim["time_mpc"][0]
                if len(sim["time_mpc"]) > 1 else 60.0)

    # ==================================================================
    #  Panel [0,0] — SOC: true vs EKF vs MHE
    # ==================================================================
    ax = axes[0, 0]
    ax.plot(t_sim_h, sim["soc_true"], color="0.25", linewidth=_LW_TRUE,
            label="True SOC")
    ax.plot(t_mpc_h, sim["soc_ekf"], color="tab:blue", linewidth=_LW_EST,
            label="EKF")
    if np.any(sim["soc_mhe"] != 0):
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
    #  Panel [0,1] — Temperature: true vs EKF vs MHE  (v2 hero)
    # ==================================================================
    ax = axes[0, 1]
    ax.plot(t_sim_h, sim["temp_true"], color="0.25", linewidth=_LW_TRUE,
            label="True T")
    ax.plot(t_mpc_h, sim["temp_ekf"], color="tab:blue", linewidth=_LW_EST,
            label="EKF T")
    if np.any(sim["temp_mhe"] != 0):
        ax.plot(t_mpc_h, sim["temp_mhe"], color="tab:red", linewidth=_LW_EST,
                linestyle="--", label="MHE T")
    ax.axhline(thp.T_max, color="red", linewidth=1.2, linestyle=":",
               label=f"T_max = {thp.T_max:.0f} °C")
    ax.axhline(thp.T_ambient, color="green", linewidth=1.0, linestyle=":",
               alpha=0.6, label=f"T_amb = {thp.T_ambient:.0f} °C")
    ax.set_ylabel("Temperature [°C]")
    ax.legend(loc="upper right")
    ax.set_title("Cell Temperature Estimation")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [1,0] — SOH: true vs EKF vs MHE
    # ==================================================================
    ax = axes[1, 0]
    ax.plot(t_sim_h, sim["soh_true"], color="0.25", linewidth=_LW_TRUE,
            label="True SOH")
    ax.plot(t_mpc_h, sim["soh_ekf"], color="tab:blue", linewidth=_LW_EST,
            label="EKF")
    if np.any(sim["soh_mhe"] != 0):
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

    # Profit annotation
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
