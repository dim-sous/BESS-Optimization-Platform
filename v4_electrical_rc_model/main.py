"""v4_electrical_rc_model — 2RC equivalent circuit with NMC OCV polynomial.

Extends the 3-state thermal model to 5 states:
    x = [SOC, SOH, T, V_rc1, V_rc2]

Terminal voltage V_term = OCV(SOC) - V_rc1 - V_rc2 - I*R0 is now modeled
and measured, providing a new observation channel for state estimation.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import numpy as np

VERSION_TAG = "v4_electrical_rc_model"

# Ensure version folder root is importable (and ONLY this folder)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.parameters import (
    BatteryParams,
    EKFParams,
    ElectricalParams,
    EMSParams,
    MHEParams,
    MPCParams,
    PackParams,
    ThermalParams,
    TimeParams,
)
from data.price_generator import PriceGenerator
from simulation.simulator import MultiRateSimulator
from visualization.plot_results import plot_results

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the hierarchical BESS control simulation with 2RC model."""
    parser = argparse.ArgumentParser(description=f"Run {VERSION_TAG} simulation")
    parser.add_argument("--mhe", action="store_true", default=False,
                        help="Enable MHE estimator (default: OFF)")
    args = parser.parse_args()

    # ---- Configuration ----
    bp = BatteryParams()
    tp = TimeParams()
    ep = EMSParams()
    mp = MPCParams()
    ekf_p = EKFParams()
    mhe_p = MHEParams()
    thp = ThermalParams()
    elp = ElectricalParams()
    pp = PackParams()

    logger.info("=" * 62)
    logger.info("  BESS HIERARCHICAL CONTROL PLATFORM  [%s]", VERSION_TAG)
    logger.info("=" * 62)
    logger.info("  Battery:    %d kWh / %d kW", bp.E_nom_kwh, bp.P_max_kw)
    logger.info("  SOC range:  [%.2f, %.2f]", bp.SOC_min, bp.SOC_max)
    logger.info("  SOH init:   %.4f", bp.SOH_init)
    logger.info("  alpha_deg:  %.2e  [1/(kW*s)]", bp.alpha_deg)
    logger.info("  --- Electrical (2RC) ---")
    logger.info("  R0:         %.4f Ohm", elp.R0)
    logger.info("  R1:         %.4f Ohm  (tau_1=%.0f s)", elp.R1, elp.tau_1)
    logger.info("  R2:         %.4f Ohm  (tau_2=%.0f s)", elp.R2, elp.tau_2)
    logger.info("  R_total:    %.4f Ohm  (R0+R1+R2)", elp.R_total_dc)
    logger.info("  V_min_pack: %.0f V", elp.V_min_pack)
    logger.info("  V_max_pack: %.0f V", elp.V_max_pack)
    logger.info("  n_series:   %d cells/module", elp.n_series_cells)
    logger.info("  --- Thermal ---")
    logger.info("  C_thermal:  %.0f J/K", thp.C_thermal)
    logger.info("  h_cool:     %.0f W/K", thp.h_cool)
    logger.info("  T_ambient:  %.1f degC", thp.T_ambient)
    logger.info("  T_max:      %.1f degC", thp.T_max)
    logger.info("  V_nominal:  %.0f V", thp.V_nominal)
    logger.info("  --- Pack ---")
    logger.info("  n_cells:    %d", pp.n_cells)
    logger.info("  cap spread: +/-%.1f%%", pp.capacity_spread * 100)
    logger.info("  R spread:   +/-%.1f%%", pp.resistance_spread * 100)
    logger.info("  deg spread: +/-%.1f%%", pp.degradation_spread * 100)
    logger.info("  balancing:  %s (gain=%.0f, max=%.1f kW)",
                "ON" if pp.balancing_enabled else "OFF",
                pp.balancing_gain, pp.max_balancing_power)
    logger.info("  --- Timing ---")
    logger.info("  dt_ems:     %d s", tp.dt_ems)
    logger.info("  dt_mpc:     %d s", tp.dt_mpc)
    logger.info("  dt_sim:     %d s", tp.dt_sim)
    logger.info("  Sim hours:  %d h", tp.sim_hours)
    logger.info("  EMS:  N=%d  scenarios=%d", ep.N_ems, ep.n_scenarios)
    logger.info("  MPC:  N=%d  Nc=%d", mp.N_mpc, mp.Nc_mpc)
    logger.info("  MHE:  N=%d", mhe_p.N_mhe)
    logger.info("=" * 62)

    # ---- Price scenarios ----
    n_hours_total = int(tp.sim_hours) + ep.N_ems
    price_gen = PriceGenerator(seed=42)
    energy_scen, reg_scen, probs = price_gen.generate_scenarios(
        n_hours=n_hours_total,
        n_scenarios=ep.n_scenarios,
    )
    logger.info(
        "Price scenarios generated: %d scenarios x %d hours",
        energy_scen.shape[0],
        energy_scen.shape[1],
    )

    # ---- Multi-rate simulation ----
    simulator = MultiRateSimulator(bp, tp, ep, mp, ekf_p, mhe_p, thp, elp, pp,
                                    run_mhe=args.mhe)
    results = simulator.run(energy_scen, reg_scen, probs)

    # ---- Visualisation ----
    results_dir = PROJECT_ROOT.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = str(results_dir / f"{VERSION_TAG}_results.png")
    plot_results(results, bp, thp, elp, pp, save_path=plot_path)

    # ---- Save raw results for comparison pipeline ----
    array_data = {}
    scalar_data = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            array_data[k] = v
        elif isinstance(v, (int, float, str, bool)):
            scalar_data[k] = v

    npz_path = results_dir / f"{VERSION_TAG}_results.npz"
    np.savez(npz_path, **array_data)

    scalars_path = results_dir / f"{VERSION_TAG}_scalars.json"
    with open(scalars_path, "w") as f:
        json.dump(scalar_data, f, indent=2)

    logger.info("Results saved to %s", npz_path)

    # ---- Summary ----
    print()
    print("=" * 62)
    print(f"  RESULTS SUMMARY  [{VERSION_TAG}]")
    print("=" * 62)
    print(f"  Battery:          {bp.E_nom_kwh:.0f} kWh / {bp.P_max_kw:.0f} kW")
    print(f"  Pack:             {pp.n_cells} cells in series")
    print(f"  Electrical:       2RC (R0={elp.R0}, R1={elp.R1}, R2={elp.R2})")
    print(f"  Simulation:       {tp.sim_hours:.0f} hours")
    print(f"  Total profit:     ${results['total_profit']:.2f}")
    print(f"  SOH degradation:  {results['soh_degradation']*100:.4f}%")
    print(f"  Final SOC:        {results['soc_true'][-1]:.4f}")
    print(f"  Final SOH:        {results['soh_true'][-1]:.6f}")
    print(f"  Final Temp:       {results['temp_true'][-1]:.2f} degC")
    print(f"  Max  Temp:        {np.max(results['temp_true']):.2f} degC")
    print(f"  --- Voltage ---")
    vterm = results.get("vterm_true")
    if vterm is not None:
        print(f"  V_term min:       {np.min(vterm[1:]):.1f} V")
        print(f"  V_term max:       {np.max(vterm[1:]):.1f} V")
        print(f"  V_term mean:      {np.mean(vterm[1:]):.1f} V")
        print(f"  V_rc1 max |val|:  {np.max(np.abs(results['vrc1_true'])):.3f} V")
        print(f"  V_rc2 max |val|:  {np.max(np.abs(results['vrc2_true'])):.3f} V")

    # Pack-specific summary
    if "cell_socs" in results:
        soc_imb = results["soc_imbalance"]
        print(f"  --- Cell-Level ---")
        print(f"  Max SOC imbalance:   {np.max(soc_imb)*100:.3f}%")
        print(f"  Avg SOC imbalance:   {np.mean(soc_imb)*100:.3f}%")
        cell_sohs_end = results["cell_sohs"][:, -1]
        print(f"  SOH spread (final):  {(np.max(cell_sohs_end) - np.min(cell_sohs_end))*100:.4f}%")
        cell_temps = results["cell_temps"]
        print(f"  Max cell temp:       {np.max(cell_temps):.2f} degC")
        bal_pow = results.get("balancing_power")
        if bal_pow is not None:
            bal_energy = np.sum(np.abs(bal_pow)) * tp.dt_mpc / 3600.0
            print(f"  Balancing energy:    {bal_energy:.3f} kWh")

    print(f"  --- Estimation ---")
    print(f"  EKF final SOC:    {results['soc_ekf'][-1]:.4f}")
    print(f"  EKF final SOH:    {results['soh_ekf'][-1]:.6f}")
    print(f"  EKF final T:      {results['temp_ekf'][-1]:.2f} degC")
    print(f"  MHE final SOC:    {results['soc_mhe'][-1]:.4f}")
    print(f"  MHE final SOH:    {results['soh_mhe'][-1]:.6f}")
    print(f"  MHE final T:      {results['temp_mhe'][-1]:.2f} degC")
    mpc_t = results.get("mpc_solve_times")
    if mpc_t is not None and len(mpc_t) > 0:
        print(f"  Avg MPC solve:    {np.mean(mpc_t)*1000:.1f} ms")
        print(f"  Max MPC solve:    {np.max(mpc_t)*1000:.1f} ms")
    est_t = results.get("est_solve_times")
    if est_t is not None and len(est_t) > 0:
        print(f"  Avg Est solve:    {np.mean(est_t)*1000:.1f} ms")
    print("=" * 62)


if __name__ == "__main__":
    main()
