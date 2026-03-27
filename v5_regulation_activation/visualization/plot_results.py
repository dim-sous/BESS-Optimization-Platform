"""Fourteen-panel result visualisation for the v5 regulation activation model.

Panel layout (7 rows x 2 columns)
----------------------------------
  [0,0] Pack SOC: true vs EKF vs MHE   |  [0,1] Pack SOH: true vs EKF vs MHE
  [1,0] Individual cell SOCs           |  [1,1] Individual cell temperatures
  [2,0] Terminal voltage + OCV         |  [2,1] RC voltages (V_rc1, V_rc2)
  [3,0] Activation signal (v5)         |  [3,1] Regulation delivery (v5)
  [4,0] SOC imbalance + balancing pwr  |  [4,1] Net grid power + price
  [5,0] Solver timing                  |  [5,1] Delivery score (v5)
  [6,0] Revenue breakdown (v5)         |  [6,1] Profit decomposition (v5)
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


def _downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by taking every factor-th element."""
    return arr[::factor]


def plot_results(
    sim: dict,
    bp: BatteryParams,
    thp: ThermalParams,
    elp: ElectricalParams,
    pp: PackParams | None = None,
    save_path: str = "results.png",
) -> None:
    """Generate the fourteen-panel summary figure."""
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
    has_activation = "activation_signal" in sim
    strategy = sim.get("strategy", "full")

    fig, axes = plt.subplots(7, 2, figsize=(20, 40))
    suptitle = (
        f"Hierarchical BESS Control \u2014 v5 Regulation Activation "
        f"({n_cells} cells, {strategy}) \u2014 "
        f"{sim['time_sim'][-1]/3600:.0f} h Simulation"
    )
    fig.suptitle(suptitle, fontsize=_SUPTITLE_SIZE, fontweight="bold", y=0.995)

    t_sim_h = sim["time_sim"] / 3600.0
    t_mpc_h = sim["time_mpc"] / 3600.0
    dt_sim_s = float(sim["time_sim"][1] - sim["time_sim"][0]) if len(sim["time_sim"]) > 1 else 4.0
    dt_mpc_s = float(sim["time_mpc"][1] - sim["time_mpc"][0]) if len(sim["time_mpc"]) > 1 else 60.0

    # Downsample factor for dt_sim arrays (4s -> plot every 15th = 60s)
    ds = max(1, int(dt_mpc_s / dt_sim_s))

    # ==================================================================
    #  Panel [0,0] — Pack SOC: true vs EKF vs MHE
    # ==================================================================
    ax = axes[0, 0]
    ax.plot(t_sim_h, sim["soc_true"], color="0.35", linewidth=_LW_TRUE,
            label="Pack SOC (true)")
    ax.plot(t_mpc_h, sim["soc_ekf"], color="tab:blue", linewidth=_LW_EST,
            label="EKF estimate")
    if np.any(sim["soc_mhe"] != 0):
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
    if np.any(sim["soh_mhe"] != 0):
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
    #  Panel [2,0] — Terminal Voltage + OCV
    # ==================================================================
    ax = axes[2, 0]
    vterm = sim.get("vterm_true", None)
    if vterm is not None:
        ax.plot(t_sim_h, vterm, color="tab:blue", linewidth=_LW_TRUE,
                label="V_term (true)")
        ocv_vals = ocv_pack_numpy(sim["soc_true"], elp)
        ax.plot(t_sim_h, ocv_vals, color="tab:green", linewidth=1.0,
                alpha=0.6, linestyle="--", label="OCV(SOC)")
        ax.axhline(elp.V_min_pack, color="red", linewidth=1.0, linestyle=":",
                   label=f"V_min = {elp.V_min_pack:.0f} V")
        ax.axhline(elp.V_max_pack, color="red", linewidth=1.0, linestyle=":")
    ax.set_ylabel("Voltage [V]")
    ax.legend(loc="upper right")
    ax.set_title("Pack Terminal Voltage")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [2,1] — RC Voltages
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
    #  Panel [3,0] — Activation Signal (v5)
    # ==================================================================
    ax = axes[3, 0]
    if has_activation:
        act = sim["activation_signal"]
        t_act_h = np.arange(len(act)) * dt_sim_s / 3600.0
        # Downsample for plotting
        act_ds = _downsample(act, ds)
        t_act_ds = _downsample(t_act_h, ds)

        ax.fill_between(t_act_ds, act_ds, 0, where=(act_ds > 0),
                        color="tab:green", alpha=0.4, label="Up-reg (discharge)")
        ax.fill_between(t_act_ds, act_ds, 0, where=(act_ds < 0),
                        color="tab:red", alpha=0.4, label="Down-reg (charge)")
        ax.plot(t_act_ds, act_ds, color="k", linewidth=0.3, alpha=0.5)

        n_active = np.sum(np.abs(act) > 1e-6)
        pct_active = n_active / len(act) * 100
        ax.set_title(f"FCR Activation Signal ({pct_active:.0f}% active)")
    else:
        ax.set_title("FCR Activation Signal (N/A)")
    ax.set_ylabel("Activation [-1, +1]")
    ax.set_ylim(-1.15, 1.15)
    ax.axhline(0, color="k", linewidth=0.4, linestyle=":")
    ax.legend(loc="upper right")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [3,1] — Regulation Delivery: demanded vs delivered (v5)
    # ==================================================================
    ax = axes[3, 1]
    if has_activation:
        act = sim["activation_signal"]
        delivered = sim["power_delivered"]
        # Reconstruct demanded power (need P_reg_committed from power_applied)
        P_reg_arr = sim["power_applied"][:, 2]
        demanded = act * P_reg_arr

        t_reg_h = np.arange(len(act)) * dt_sim_s / 3600.0
        dem_ds = _downsample(demanded, ds)
        del_ds = _downsample(delivered, ds)
        t_ds = _downsample(t_reg_h, ds)

        ax.plot(t_ds, dem_ds, color="tab:blue", linewidth=0.8, alpha=0.6,
                label="Demanded")
        ax.plot(t_ds, del_ds, color="tab:orange", linewidth=0.8, alpha=0.8,
                label="Delivered")
        ax.axhline(0, color="k", linewidth=0.4, linestyle=":")

        delivery_score = sim.get("delivery_score", 0.0)
        ax.set_title(f"Regulation Delivery (score: {delivery_score*100:.1f}%)")
    else:
        ax.set_title("Regulation Delivery (N/A)")
    ax.set_ylabel("Power [kW]")
    ax.legend(loc="upper right")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [4,0] — SOC Imbalance + Balancing Power
    # ==================================================================
    ax = axes[4, 0]
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
    #  Panel [4,1] — Net Grid Power + Price
    # ==================================================================
    ax = axes[4, 1]
    pow_arr = sim.get("power_mpc_base", sim["power_applied"])
    n_pow = len(pow_arr)
    pow_dt = dt_mpc_s if "power_mpc_base" in sim else dt_sim_s
    t_pow = np.arange(n_pow) * pow_dt / 3600.0
    net_grid_power = pow_arr[:, 1] - pow_arr[:, 0]
    reg_power = pow_arr[:, 2]

    ts, ys = _stepify(t_pow, net_grid_power)
    ax.fill_between(ts, ys, 0, where=(ys >= 0),
                    color="tab:green", alpha=0.25, label="Selling",
                    interpolate=True)
    ax.fill_between(ts, ys, 0, where=(ys < 0),
                    color="tab:red", alpha=0.25, label="Buying",
                    interpolate=True)
    ax.plot(ts, ys, color="k", linewidth=1.0, label="Net grid power")
    ax.step(t_pow, reg_power, where="post",
            color="tab:orange", linewidth=_LW_MPC, label="P_reg committed")
    ax.axhline(0, color="k", linewidth=0.6, linestyle=":")
    ax.set_ylabel("Power [kW]")
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
    #  Panel [5,0] — Solver Timing
    # ==================================================================
    ax = axes[5, 0]
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
    #  Panel [5,1] — Delivery Score over time (v5)
    # ==================================================================
    ax = axes[5, 1]
    if has_activation:
        reg_acc = sim["reg_accounting"]  # (N_sim, 4): cap, del, pen, is_ok
        is_ok = reg_acc[:, 3]
        act = sim["activation_signal"]
        active_mask = np.abs(act) > 1e-6

        # Rolling delivery score (window = 15 min = 225 steps at 4s)
        window = min(225, len(is_ok) // 4)
        if window > 0:
            ok_cumsum = np.cumsum(is_ok * active_mask)
            active_cumsum = np.cumsum(active_mask.astype(float))
            # Rolling score at each point
            roll_ok = ok_cumsum[window:] - ok_cumsum[:-window]
            roll_active = active_cumsum[window:] - active_cumsum[:-window]
            roll_score = np.where(roll_active > 0, roll_ok / roll_active, 1.0)
            t_roll = np.arange(len(roll_score)) * dt_sim_s / 3600.0

            ax.plot(t_roll, roll_score * 100, color="tab:blue", linewidth=1.0)
            ax.axhline(sim.get("delivery_score", 0) * 100, color="tab:red",
                        linewidth=1.5, linestyle="--",
                        label=f"Overall: {sim.get('delivery_score', 0)*100:.1f}%")
            ax.axhline(90, color="tab:green", linewidth=1.0, linestyle=":",
                        alpha=0.5, label="90% target")
        ax.set_ylim(-5, 105)
        ax.set_title(f"Rolling Delivery Score (15-min window)")
    else:
        ax.set_title("Delivery Score (N/A)")
    ax.set_ylabel("Delivery Score [%]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="lower right")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [6,0] — Cumulative regulation revenue breakdown (v5)
    # ==================================================================
    ax = axes[6, 0]
    if has_activation:
        reg_acc = sim["reg_accounting"]
        cap_cum = np.cumsum(reg_acc[:, 0])
        del_cum = np.cumsum(reg_acc[:, 1])
        pen_cum = np.cumsum(reg_acc[:, 2])
        net_cum = cap_cum + del_cum - pen_cum

        t_reg_h = np.arange(len(reg_acc)) * dt_sim_s / 3600.0
        t_ds = _downsample(t_reg_h, ds)

        ax.plot(t_ds, _downsample(cap_cum, ds), color="tab:blue",
                linewidth=_LW_EST, label=f"Capacity (${cap_cum[-1]:.2f})")
        ax.plot(t_ds, _downsample(del_cum, ds), color="tab:green",
                linewidth=_LW_EST, label=f"Delivery (${del_cum[-1]:.2f})")
        ax.plot(t_ds, -_downsample(pen_cum, ds), color="tab:red",
                linewidth=_LW_EST, label=f"Penalty (-${pen_cum[-1]:.2f})")
        ax.plot(t_ds, _downsample(net_cum, ds), color="k",
                linewidth=_LW_TRUE, label=f"Net (${net_cum[-1]:.2f})")
        ax.axhline(0, color="k", linewidth=0.4, linestyle=":")
    ax.set_ylabel("Cumulative [$]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="upper left")
    ax.set_title("Regulation Revenue Breakdown")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [6,1] — Total profit decomposition (v5)
    # ==================================================================
    ax = axes[6, 1]
    energy_profit = sim.get("energy_profit", np.zeros(1))
    deg_cost = sim.get("deg_cost", np.zeros(1))
    n_prof = len(energy_profit)
    t_prof = np.arange(n_prof) * dt_mpc_s / 3600.0

    cum_energy = np.cumsum(energy_profit)
    cum_deg = np.cumsum(deg_cost)

    ax.plot(t_prof, cum_energy, color="tab:blue", linewidth=_LW_EST,
            label=f"Energy (${cum_energy[-1]:.2f})")
    ax.plot(t_prof, -cum_deg, color="tab:red", linewidth=_LW_EST,
            label=f"Degradation (-${cum_deg[-1]:.2f})")

    if has_activation:
        reg_acc = sim["reg_accounting"]
        net_reg_cum = np.cumsum(reg_acc[:, 0] + reg_acc[:, 1] - reg_acc[:, 2])
        # Resample to MPC resolution for aligned plotting
        net_reg_at_mpc = _downsample(net_reg_cum, int(dt_mpc_s / dt_sim_s))[:n_prof]
        if len(net_reg_at_mpc) > 0:
            t_reg_mpc = np.arange(len(net_reg_at_mpc)) * dt_mpc_s / 3600.0
            ax.plot(t_reg_mpc, net_reg_at_mpc, color="tab:orange",
                    linewidth=_LW_EST,
                    label=f"Reg net (${net_reg_cum[-1]:.2f})")
            # Total = energy + reg_net - deg
            total_cum = cum_energy[:len(net_reg_at_mpc)] + net_reg_at_mpc - cum_deg[:len(net_reg_at_mpc)]
            ax.plot(t_reg_mpc, total_cum, color="k", linewidth=_LW_TRUE,
                    label=f"Total (${sim['total_profit']:.2f})")
    else:
        ax.plot(t_prof, cum_energy - cum_deg, color="k", linewidth=_LW_TRUE,
                label=f"Total (${sim['total_profit']:.2f})")

    ax.axhline(0, color="k", linewidth=0.4, linestyle=":")
    ax.set_ylabel("Cumulative [$]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="upper left")
    ax.set_title("Total Profit Decomposition")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Results saved to {save_path}")
