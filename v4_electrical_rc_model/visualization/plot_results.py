"""Ten-panel result visualisation for the v4 electrical RC model.

Panel layout (5 rows x 2 columns)
----------------------------------
  [0,0] Pack SOC: true vs EKF vs MHE   |  [0,1] Pack SOH: true vs EKF vs MHE
  [1,0] Individual cell SOCs           |  [1,1] Individual cell temperatures
  [2,0] Terminal voltage + OCV         |  [2,1] RC voltages (V_rc1, V_rc2)
  [3,0] SOC imbalance + balancing pwr  |  [3,1] Net grid power + price
  [4,0] Solver timing                  |  [4,1] Cumulative profit breakdown
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
_TITLE_SIZE = 16
_SUPTITLE_SIZE = 18
_LABEL_SIZE = 13
_TICK_SIZE = 11
_LEGEND_SIZE = 10
_LW_TRUE = 2.0
_LW_EST = 1.5
_LW_REF = 2.5
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
    elp: ElectricalParams,
    pp: PackParams | None = None,
    save_path: str = "results.png",
) -> None:
    """Generate the ten-panel summary figure."""
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

    fig, axes = plt.subplots(5, 2, figsize=(20, 28))
    suptitle = (
        f"Hierarchical BESS Control \u2014 v4 Electrical RC Model "
        f"({n_cells} cells) \u2014 24 h Simulation"
    )
    fig.suptitle(suptitle, fontsize=_SUPTITLE_SIZE, fontweight="bold", y=0.995)

    t_sim_h = sim["time_sim"] / 3600.0
    t_mpc_h = sim["time_mpc"] / 3600.0
    dt_mpc_s = float(sim["time_mpc"][1] - sim["time_mpc"][0]) if len(sim["time_mpc"]) > 1 else 60.0

    # ==================================================================
    #  Panel [0,0] — Pack SOC: true vs EKF vs MHE
    # ==================================================================
    ax = axes[0, 0]
    ax.plot(t_sim_h, sim["soc_true"], color="0.35", linewidth=_LW_TRUE,
            label="Pack SOC (true)")
    ax.plot(t_mpc_h, sim["soc_ekf"], color="tab:blue", linewidth=_LW_EST,
            label="EKF estimate")
    ax.plot(t_mpc_h, sim["soc_mhe"], color="tab:red", linewidth=_LW_EST,
            linestyle="--", label="MHE estimate")
    ax.axhspan(bp.SOC_min, bp.SOC_max, alpha=0.08, color="green",
               label=f"SOC limits [{bp.SOC_min:.0%}\u2013{bp.SOC_max:.0%}]")
    ax.set_ylabel("SOC [-]")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right")
    ax.set_title("Pack State of Charge Estimation")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [0,1] — Pack SOH: true vs EKF vs MHE
    # ==================================================================
    ax = axes[0, 1]
    ax.plot(t_sim_h, sim["soh_true"], color="0.35", linewidth=_LW_TRUE,
            label="Pack SOH (true)")
    ax.plot(t_mpc_h, sim["soh_ekf"], color="tab:blue", linewidth=_LW_EST,
            label="EKF estimate")
    ax.plot(t_mpc_h, sim["soh_mhe"], color="tab:red", linewidth=_LW_EST,
            linestyle="--", label="MHE estimate")
    ax.set_ylabel("SOH [-]")
    ax.legend(loc="lower left")
    ax.set_title("Pack State of Health Estimation")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [1,0] — Individual Cell SOCs
    # ==================================================================
    ax = axes[1, 0]
    if has_cell_data:
        cell_colors = plt.cm.Set2(np.linspace(0, 1, n_cells))
        for i in range(n_cells):
            ax.plot(t_sim_h, sim["cell_socs"][i], color=cell_colors[i],
                    linewidth=1.0, alpha=0.8, label=f"Cell {i+1}")
        ax.plot(t_sim_h, sim["soc_true"], color="k", linewidth=_LW_TRUE,
                linestyle="--", alpha=0.6, label="Pack avg")
    ax.axhspan(bp.SOC_min, bp.SOC_max, alpha=0.06, color="green")
    ax.set_ylabel("SOC [-]")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", ncol=3)
    ax.set_title("Individual Cell SOCs")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [1,1] — Individual Cell Temperatures
    # ==================================================================
    ax = axes[1, 1]
    if has_cell_data:
        for i in range(n_cells):
            ax.plot(t_sim_h, sim["cell_temps"][i], color=cell_colors[i],
                    linewidth=1.0, alpha=0.8, label=f"Cell {i+1}")
    else:
        ax.plot(t_sim_h, sim["temp_true"], color="0.35", linewidth=_LW_TRUE,
                label="Temperature")
    ax.axhline(thp.T_max, color="red", linewidth=1.0, linestyle=":",
               label=f"T_max = {thp.T_max:.0f} \u00b0C")
    ax.axhline(thp.T_ambient, color="green", linewidth=0.8, linestyle=":",
               alpha=0.5, label=f"T_ambient = {thp.T_ambient:.0f} \u00b0C")
    ax.set_ylabel("Temperature [\u00b0C]")
    ax.legend(loc="upper right", ncol=3)
    ax.set_title("Individual Cell Temperatures")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [2,0] — Terminal Voltage + OCV (v4)
    # ==================================================================
    ax = axes[2, 0]
    vterm = sim.get("vterm_true", None)
    if vterm is not None:
        ax.plot(t_sim_h, vterm, color="tab:blue", linewidth=_LW_TRUE,
                label="V_term (true)")
        # OCV from SOC
        ocv_vals = ocv_pack_numpy(sim["soc_true"], elp)
        ax.plot(t_sim_h, ocv_vals, color="tab:green", linewidth=1.0,
                alpha=0.6, linestyle="--", label="OCV(SOC)")
        # Voltage limits
        ax.axhline(elp.V_min_pack, color="red", linewidth=1.0, linestyle=":",
                   label=f"V_min = {elp.V_min_pack:.0f} V")
        ax.axhline(elp.V_max_pack, color="red", linewidth=1.0, linestyle=":")
    ax.set_ylabel("Voltage [V]")
    ax.legend(loc="upper right")
    ax.set_title("Pack Terminal Voltage")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [2,1] — RC Voltages (v4)
    # ==================================================================
    ax = axes[2, 1]
    vrc1 = sim.get("vrc1_true", None)
    vrc2 = sim.get("vrc2_true", None)
    if vrc1 is not None:
        ax.plot(t_sim_h, vrc1, color="tab:blue", linewidth=_LW_TRUE,
                label=f"V_rc1 (\u03c4={elp.tau_1:.0f}s)")
    if vrc2 is not None:
        ax.plot(t_sim_h, vrc2, color="tab:orange", linewidth=_LW_TRUE,
                label=f"V_rc2 (\u03c4={elp.tau_2:.0f}s)")
    ax.axhline(0, color="k", linewidth=0.4, linestyle=":")
    ax.set_ylabel("RC Voltage [V]")
    ax.legend(loc="upper right")
    ax.set_title("RC Circuit Transient Voltages")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [3,0] — SOC Imbalance + Balancing Power
    # ==================================================================
    ax = axes[3, 0]
    if has_cell_data:
        soc_imb = sim["soc_imbalance"]
        ax.plot(t_sim_h, soc_imb * 100, color="tab:red", linewidth=_LW_TRUE,
                label="SOC spread (max\u2212min)")
        ax.set_ylabel("SOC Spread [%]")
        ax.legend(loc="upper left")

        bal_pow = sim.get("balancing_power")
        if bal_pow is not None and bal_pow.shape[1] > 0:
            ax2 = ax.twinx()
            n_bal = bal_pow.shape[1]
            t_bal_h = np.arange(n_bal) * dt_mpc_s / 3600.0
            for i in range(n_cells):
                ax2.plot(t_bal_h, bal_pow[i], color=cell_colors[i],
                         linewidth=0.7, alpha=0.5)
            ax2.set_ylabel("Balancing Power [kW]", alpha=0.7)
    else:
        ax.plot(t_sim_h, sim["temp_true"], color="tab:red", linewidth=_LW_TRUE)
        ax.set_ylabel("Temperature [\u00b0C]")
    ax.set_title("Cell SOC Imbalance and Balancing Power")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [3,1] — Grid Power + Price
    # ==================================================================
    ax = axes[3, 1]
    n_mpc = len(sim["power_applied"])
    t_pow = np.arange(n_mpc) * dt_mpc_s / 3600.0
    net_grid_power = sim["power_applied"][:, 1] - sim["power_applied"][:, 0]
    reg_power = sim["power_applied"][:, 2]

    ts, ys = _stepify(t_pow, net_grid_power)
    ax.fill_between(ts, ys, 0, where=(ys >= 0),
                    color="tab:green", alpha=0.25, label="Selling",
                    interpolate=True)
    ax.fill_between(ts, ys, 0, where=(ys < 0),
                    color="tab:red", alpha=0.25, label="Buying",
                    interpolate=True)
    ax.plot(ts, ys, color="k", linewidth=1.0, label="Net grid power")
    ax.step(t_pow, reg_power, where="post",
            color="tab:orange", linewidth=_LW_MPC, label="P_reg")

    ax.axhline(0, color="k", linewidth=0.6, linestyle=":")
    ax.set_ylabel("Power [kW]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Grid Power Exchange and Regulation Reserve")
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

    # ==================================================================
    #  Panel [4,0] — Solver Timing
    # ==================================================================
    ax = axes[4, 0]
    mpc_times = sim.get("mpc_solve_times", np.array([]))
    est_times = sim.get("est_solve_times", np.array([]))
    if len(mpc_times) > 0:
        t_timing = np.arange(len(mpc_times)) * dt_mpc_s / 3600.0
        ax.plot(t_timing, mpc_times * 1000, color="tab:blue",
                linewidth=0.8, alpha=0.6, label="MPC solve")
        ax.axhline(np.mean(mpc_times) * 1000, color="tab:blue",
                    linewidth=1.5, linestyle="--",
                    label=f"MPC mean = {np.mean(mpc_times)*1000:.1f} ms")
    if len(est_times) > 0:
        t_timing_e = np.arange(len(est_times)) * dt_mpc_s / 3600.0
        ax.plot(t_timing_e, est_times * 1000, color="tab:orange",
                linewidth=0.8, alpha=0.6, label="Estimator solve")
        ax.axhline(np.mean(est_times) * 1000, color="tab:orange",
                    linewidth=1.5, linestyle="--",
                    label=f"Est mean = {np.mean(est_times)*1000:.1f} ms")
    ax.set_ylabel("Solve time [ms]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="upper right")
    ax.set_title("Solver Timing")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [4,1] — Cumulative profit breakdown
    # ==================================================================
    ax = axes[4, 1]
    n_prof = len(sim["cumulative_profit"])
    t_prof = np.arange(n_prof) * dt_mpc_s / 3600.0

    cum_energy = np.cumsum(sim["energy_profit"])
    cum_reg = np.cumsum(sim["reg_profit"])
    cum_deg = np.cumsum(sim["deg_cost"])

    ax.plot(t_prof, cum_energy, color="tab:blue", linewidth=_LW_EST,
            label="Energy arbitrage")
    ax.plot(t_prof, cum_reg, color="tab:orange", linewidth=_LW_EST,
            label="Regulation revenue")
    ax.plot(t_prof, -cum_deg, color="tab:red", linewidth=_LW_EST,
            label="Degradation cost")
    ax.plot(t_prof, sim["cumulative_profit"], color="k", linewidth=_LW_TRUE,
            label=f"Net profit (${sim['total_profit']:.2f})")
    ax.axhline(0, color="k", linewidth=0.4, linestyle=":")
    ax.set_ylabel("Cumulative [$]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="upper left")
    ax.set_title("Revenue Breakdown")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Results saved to {save_path}")
