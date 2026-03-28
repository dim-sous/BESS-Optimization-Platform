"""Six-panel result visualisation for v5 regulation activation.

Panel layout (3 x 2)
---------------------
  [0,0] SOC + estimation               |  [0,1] SOH + estimation
  [1,0] Activation + delivery (v5 hero)|  [1,1] Temperature
  [2,0] Power dispatch + P_reg + price |  [2,1] Profit decomposition

Trace conventions (shared across all versions):
  True   — solid black
  EKF    — dashed blue
  MHE    — dotted red
  EMS ref — step gray, semi-transparent
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from config.parameters import BatteryParams, ElectricalParams, PackParams, ThermalParams


# ---------------------------------------------------------------------------
#  Style — shared conventions across all versions
# ---------------------------------------------------------------------------
_TITLE = 14
_SUPTITLE = 16
_LABEL = 12
_TICK = 10
_LEGEND = 9
_LW = 2.0
_LW_EST = 1.5

_TRUE_KW = dict(color="k", lw=_LW, ls="-")
_EKF_KW = dict(color="tab:blue", lw=_LW_EST, ls="--")
_MHE_KW = dict(color="tab:red", lw=_LW_EST, ls=":")
_EMS_KW = dict(color="0.5", lw=1.8, ls="-", alpha=0.4, drawstyle="steps-pre")


def _step(t: np.ndarray, y: np.ndarray):
    n = len(t)
    dt = float(t[1] - t[0]) if n > 1 else 1.0
    ts = np.empty(2 * n); ys = np.empty(2 * n)
    ts[0::2] = t; ts[1::2] = np.append(t[1:], t[-1] + dt)
    ys[0::2] = y; ys[1::2] = y
    return ts, ys


def _ds(arr: np.ndarray, factor: int) -> np.ndarray:
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
    """Generate the six-panel summary figure."""
    plt.rcParams.update({
        "font.size": _TICK, "axes.titlesize": _TITLE,
        "axes.labelsize": _LABEL, "xtick.labelsize": _TICK,
        "ytick.labelsize": _TICK, "legend.fontsize": _LEGEND,
    })

    n_cells = sim.get("n_cells", 1)
    has_act = "activation_signal" in sim
    has_cells = "cell_temps" in sim and n_cells > 1
    strategy = sim.get("strategy", "full")
    dt_sim_s = float(sim["time_sim"][1] - sim["time_sim"][0]) if len(sim["time_sim"]) > 1 else 4.0
    dt_mpc_s = float(sim["time_mpc"][1] - sim["time_mpc"][0]) if len(sim["time_mpc"]) > 1 else 60.0
    ds_factor = max(1, int(dt_mpc_s / dt_sim_s))

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        f"v5 Regulation Activation ({n_cells} cells, {strategy}) — "
        f"{sim['time_sim'][-1]/3600:.0f} h Simulation",
        fontsize=_SUPTITLE, fontweight="bold", y=0.995,
    )

    t_sim = sim["time_sim"] / 3600.0
    t_mpc = sim["time_mpc"] / 3600.0
    dt_mpc_h = float(t_mpc[1] - t_mpc[0]) if len(t_mpc) > 1 else 60.0 / 3600.0
    ems_soc_refs = sim.get("ems_soc_refs")

    # ── SOC ──────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t_sim, sim["soc_true"], **_TRUE_KW, label="True")
    ax.plot(t_mpc, sim["soc_ekf"], **_EKF_KW, label="EKF")
    if np.any(sim["soc_mhe"] != 0):
        ax.plot(t_mpc, sim["soc_mhe"], **_MHE_KW, label="MHE")
    if ems_soc_refs is not None and len(ems_soc_refs) > 0:
        soc_plan = np.array([ref[1] for ref in ems_soc_refs])
        t_ems = np.arange(1, len(soc_plan) + 1)
        ax.plot(t_ems, soc_plan, **_EMS_KW, label="EMS plan")
    ax.axhspan(bp.SOC_min, bp.SOC_max, alpha=0.07, color="green",
               label=f"Limits [{bp.SOC_min:.0%}–{bp.SOC_max:.0%}]")
    ax.set(ylabel="SOC [-]", ylim=(-0.02, 1.02))
    ax.set_title("State of Charge")
    ax.legend(loc="upper right")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.25)

    # ── SOH ──────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t_sim, sim["soh_true"], **_TRUE_KW, label="True")
    ax.plot(t_mpc, sim["soh_ekf"], **_EKF_KW, label="EKF")
    if np.any(sim["soh_mhe"] != 0):
        ax.plot(t_mpc, sim["soh_mhe"], **_MHE_KW, label="MHE")
    ax.set(ylabel="SOH [-]")
    ax.set_title("State of Health")
    ax.legend(loc="lower left")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.25)

    # ── Activation + delivery (hero) ─────────────────────────────────────
    ax = axes[1, 0]
    if has_act:
        act = sim["activation_signal"]
        delivered = sim["power_delivered"]
        P_reg_arr = sim["power_applied"][:, 2]
        demanded = act * P_reg_arr
        t_act = np.arange(len(act)) * dt_sim_s / 3600.0

        t_d = _ds(t_act, ds_factor)
        dem_d = _ds(demanded, ds_factor)
        del_d = _ds(delivered, ds_factor)

        ax.plot(t_d, dem_d, color="tab:blue", lw=0.7, alpha=0.5,
                label="Demanded")
        ax.plot(t_d, del_d, color="tab:orange", lw=0.7, alpha=0.7,
                label="Delivered")
        ax.axhline(0, color="k", lw=0.4, ls=":")

        score = sim.get("delivery_score", 0.0)
        n_active = np.sum(np.abs(act) > 1e-6)
        pct = n_active / len(act) * 100
        ax.set_title(f"FCR Regulation (v5 upgrade) — {pct:.0f}% active, {score*100:.1f}% delivered")
    else:
        ax.set_title("FCR Regulation (N/A)")
    ax.set(ylabel="Power [kW]")
    ax.legend(loc="upper right")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.25)

    # ── Temperature ──────────────────────────────────────────────────────
    ax = axes[1, 1]
    if has_cells:
        cc = plt.cm.Set2(np.linspace(0, 1, n_cells))
        for i in range(n_cells):
            ax.plot(t_sim, sim["cell_temps"][i], color=cc[i], lw=0.9,
                    alpha=0.8, label=f"Cell {i+1}" if i < 4 else None)
    else:
        ax.plot(t_sim, sim["temp_true"], **_TRUE_KW, label="Temp")
    ax.axhline(thp.T_max, color="red", lw=1.0, ls=":",
               label=f"T_max = {thp.T_max:.0f} °C")
    ax.axhline(thp.T_ambient, color="green", lw=0.8, ls=":", alpha=0.5,
               label=f"T_amb = {thp.T_ambient:.0f} °C")
    ax.set(ylabel="Temperature [°C]")
    ax.set_title("Cell Temperatures")
    ax.legend(loc="upper right")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.25)

    # ── Power dispatch + P_reg + price ───────────────────────────────────
    ax = axes[2, 0]
    pow_arr = sim.get("power_mpc_base", sim["power_applied"])
    n_pow = len(pow_arr)
    pow_dt_h = dt_mpc_h if "power_mpc_base" in sim else dt_sim_s / 3600.0
    t_pow = np.arange(n_pow) * pow_dt_h
    net = pow_arr[:, 1] - pow_arr[:, 0]
    reg = pow_arr[:, 2]
    ts, ys = _step(t_pow, net)
    ax.fill_between(ts, ys, 0, where=(ys >= 0), color="tab:green",
                    alpha=0.25, label="Sell", interpolate=True)
    ax.fill_between(ts, ys, 0, where=(ys < 0), color="tab:red",
                    alpha=0.25, label="Buy", interpolate=True)
    ax.plot(ts, ys, color="k", lw=0.8)
    ax.step(t_pow, reg, where="post", color="tab:orange", lw=1.0,
            label="P_reg committed")
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set(ylabel="Power [kW]  (+ sell / − buy)", xlabel="Time [h]")
    ax.set_title("Grid Power Exchange")
    ax.legend(loc="upper left", ncol=2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.25)

    prices_e = sim.get("prices_energy")
    if prices_e is not None:
        n_ph = min(len(prices_e), int(t_sim[-1]) + 1)
        ax2 = ax.twinx()
        ax2.step(np.arange(n_ph), prices_e[:n_ph], where="post",
                 color="tab:purple", lw=1.3, alpha=0.7, ls="-.")
        ax2.set_ylabel("E[Price] [$/kWh]", color="tab:purple", alpha=0.8)
        ax2.tick_params(axis="y", colors="tab:purple")

    # ── Profit decomposition ─────────────────────────────────────────────
    ax = axes[2, 1]
    energy_profit = sim.get("energy_profit", np.zeros(1))
    deg_cost = sim.get("deg_cost", np.zeros(1))
    n_prof = len(energy_profit)
    t_prof = np.arange(n_prof) * dt_mpc_h

    cum_energy = np.cumsum(energy_profit)
    cum_deg = np.cumsum(deg_cost)
    ax.plot(t_prof, cum_energy, color="tab:blue", lw=_LW_EST,
            label=f"Energy (${cum_energy[-1]:.2f})")
    ax.plot(t_prof, -cum_deg, color="tab:red", lw=_LW_EST,
            label=f"Degradation (-${cum_deg[-1]:.2f})")

    if has_act:
        reg_acc = sim["reg_accounting"]
        net_reg_cum = np.cumsum(reg_acc[:, 0] + reg_acc[:, 1] - reg_acc[:, 2])
        reg_at_mpc = _ds(net_reg_cum, ds_factor)[:n_prof]
        if len(reg_at_mpc) > 0:
            t_rm = np.arange(len(reg_at_mpc)) * dt_mpc_h
            ax.plot(t_rm, reg_at_mpc, color="tab:orange", lw=_LW_EST,
                    label=f"Regulation (${net_reg_cum[-1]:.2f})")
            total_cum = cum_energy[:len(reg_at_mpc)] + reg_at_mpc - cum_deg[:len(reg_at_mpc)]
            ax.plot(t_rm, total_cum, color="k", lw=_LW,
                    label=f"Total (${sim['total_profit']:.2f})")
    else:
        ax.plot(t_prof, cum_energy - cum_deg, color="k", lw=_LW,
                label=f"Total (${sim['total_profit']:.2f})")

    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set(ylabel="Cumulative [$]", xlabel="Time [h]")
    ax.set_title("Revenue Breakdown")
    ax.legend(loc="upper left")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Results saved to {save_path}")
