"""Six-panel result visualisation for v3 pack model.

Panel layout (3 x 2)
---------------------
  [0,0] SOC + cell spread (hero)       |  [0,1] SOH + cell spread
  [1,0] Cell temperatures (hero)       |  [1,1] Power dispatch + P_reg + price
  [2,0] empty (reserved)               |  [2,1] Cumulative profit breakdown

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

from config.parameters import BatteryParams, PackParams, ThermalParams


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
_LW_CELL = 0.9

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


def plot_results(
    sim: dict,
    bp: BatteryParams,
    thp: ThermalParams,
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
    has_cells = "cell_socs" in sim and n_cells > 1
    cc = plt.cm.Set2(np.linspace(0, 1, n_cells)) if has_cells else None

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    tag = f" ({n_cells} cells)" if has_cells else ""
    fig.suptitle(f"v3 Pack Model{tag} — 24 h Simulation",
                 fontsize=_SUPTITLE, fontweight="bold", y=0.995)

    t_sim = sim["time_sim"] / 3600.0
    t_mpc = sim["time_mpc"] / 3600.0
    dt_mpc = float(t_mpc[1] - t_mpc[0]) if len(t_mpc) > 1 else 60.0 / 3600.0
    ems_soc_refs = sim.get("ems_soc_refs")

    # ── SOC + cell spread ────────────────────────────────────────────────
    ax = axes[0, 0]
    if has_cells:
        cell_min = np.min(sim["cell_socs"], axis=0)
        cell_max = np.max(sim["cell_socs"], axis=0)
        ax.fill_between(t_sim, cell_min, cell_max, color="tab:cyan",
                        alpha=0.15, label="Cell spread")
    ax.plot(t_sim, sim["soc_true"], **_TRUE_KW, label="Pack (true)")
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
    ax.set_title("Pack & Cell SOC (v3 upgrade)")
    ax.legend(loc="upper right")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.25)

    # ── SOH + cell spread ────────────────────────────────────────────────
    ax = axes[0, 1]
    if has_cells and "cell_sohs" in sim:
        soh_min = np.min(sim["cell_sohs"], axis=0)
        soh_max = np.max(sim["cell_sohs"], axis=0)
        ax.fill_between(t_sim, soh_min, soh_max, color="tab:cyan",
                        alpha=0.15, label="Cell spread")
    ax.plot(t_sim, sim["soh_true"], **_TRUE_KW, label="Pack (true)")
    ax.plot(t_mpc, sim["soh_ekf"], **_EKF_KW, label="EKF")
    if np.any(sim["soh_mhe"] != 0):
        ax.plot(t_mpc, sim["soh_mhe"], **_MHE_KW, label="MHE")
    ax.set(ylabel="SOH [-]")
    ax.set_title("State of Health")
    ax.legend(loc="lower left")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.25)

    # ── Cell temperatures ────────────────────────────────────────────────
    ax = axes[1, 0]
    if has_cells:
        for i in range(n_cells):
            ax.plot(t_sim, sim["cell_temps"][i], color=cc[i], lw=_LW_CELL,
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
    ax = axes[1, 1]
    n_mpc = len(sim["power_applied"])
    t_pow = np.arange(n_mpc) * dt_mpc
    net = sim["power_applied"][:, 1] - sim["power_applied"][:, 0]
    reg = sim["power_applied"][:, 2]
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

    # ── [2,0] empty — hide axes ─────────────────────────────────────────
    axes[2, 0].set_visible(False)

    # ── Profit breakdown ─────────────────────────────────────────────────
    ax = axes[2, 1]
    n_prof = len(sim["cumulative_profit"])
    t_prof = np.arange(n_prof) * dt_mpc
    ax.plot(t_prof, np.cumsum(sim["energy_profit"]), color="tab:blue",
            lw=_LW_EST, label="Energy arb.")
    ax.plot(t_prof, np.cumsum(sim["reg_profit"]), color="tab:orange",
            lw=_LW_EST, label="Regulation")
    ax.plot(t_prof, -np.cumsum(sim["deg_cost"]), color="tab:red",
            lw=_LW_EST, label="Degradation")
    ax.plot(t_prof, sim["cumulative_profit"], color="k", lw=_LW,
            label=f"Net ${sim['total_profit']:.2f}")
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
