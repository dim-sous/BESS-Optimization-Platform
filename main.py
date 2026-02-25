"""Battery Energy Storage Optimisation Platform — entry point.

Runs the full hierarchical control pipeline:
  1. Load (or generate) electricity price data
  2. Solve the day-ahead EMS economic optimisation
  3. Run closed-loop MPC simulation tracking the EMS schedule
  4. Visualise and report results
"""

import logging
import pathlib
import sys

import numpy as np

# Ensure project root is importable regardless of working directory
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BatteryParams, EMSParams, MPCParams, SimParams
from optimization.ems_optimizer import EMSOptimizer
from simulation.simulate import run_simulation
from visualization.plot_results import plot_results

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------
def load_prices(csv_path: pathlib.Path, n_hours: int = 48) -> np.ndarray:
    """Load electricity prices from CSV, falling back to synthetic data.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the prices CSV (columns: ``hour, price_usd_per_kwh``).
    n_hours : int
        Number of hours to generate if the file is missing.

    Returns
    -------
    np.ndarray
        Price vector [$/kWh].
    """
    if csv_path.exists():
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        prices = data[:, 1]
        logger.info("Loaded %d price samples from %s", len(prices), csv_path)
        return prices

    logger.info("Price file not found — generating synthetic prices")
    return _generate_synthetic_prices(n_hours)


def _generate_synthetic_prices(n_hours: int) -> np.ndarray:
    """Generate realistic day-ahead electricity spot prices [$/kWh].

    The synthetic profile combines:
      - a base cost
      - a sinusoidal daily cycle (solar depression at night)
      - Gaussian morning and evening demand peaks
      - small random perturbations
    """
    rng = np.random.default_rng(seed=42)
    t = np.arange(n_hours, dtype=float)

    base    = 0.050
    daily   = 0.025 * np.sin(2.0 * np.pi * (t - 6.0) / 24.0)
    evening = 0.040 * np.exp(-0.5 * ((t % 24 - 18.0) / 2.0) ** 2)
    morning = 0.015 * np.exp(-0.5 * ((t % 24 - 8.0) / 1.5) ** 2)
    noise   = rng.normal(0.0, 0.004, n_hours)

    return np.maximum(base + daily + evening + morning + noise, 0.005)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- Configuration ----
    bp = BatteryParams()
    ep = EMSParams()
    mp = MPCParams()
    sp = SimParams()

    logger.info(
        "Battery: %.0f kWh / %.0f kW  |  EMS horizon: %d h  |  "
        "MPC horizon: %d steps",
        bp.E_max_kwh, bp.P_max_kw, ep.horizon, mp.horizon,
    )

    # ---- Load prices ----
    price_path = PROJECT_ROOT / "data" / "prices.csv"
    prices = load_prices(price_path, n_hours=ep.horizon + 24)

    # ---- EMS economic optimisation ----
    logger.info("Solving EMS economic optimisation …")
    ems = EMSOptimizer(bp, ep)
    ems_result = ems.solve(prices[:ep.horizon], bp.SOC_init)
    logger.info("EMS expected profit: $%.2f", ems_result["profit"])

    # ---- Closed-loop MPC simulation ----
    logger.info("Running closed-loop MPC simulation …")
    sim_result = run_simulation(bp, mp, sp, ems_result, prices)
    logger.info("Actual simulation profit: $%.2f", sim_result["profit"])

    # ---- Visualisation ----
    fig_path = str(PROJECT_ROOT / "results.png")
    plot_results(sim_result, ems_result, prices[:sp.n_steps], bp,
                 save_path=fig_path)

    # ---- Summary ----
    print()
    print("=" * 62)
    print("  BATTERY ENERGY STORAGE OPTIMISATION — RESULTS SUMMARY")
    print("=" * 62)
    print(f"  Battery capacity:          {bp.E_max_kwh:.0f} kWh / {bp.P_max_kw:.0f} kW")
    print(f"  Simulation horizon:        {sp.n_steps} hours")
    print(f"  EMS expected profit:       ${ems_result['profit']:>8.2f}")
    print(f"  MPC simulation profit:     ${sim_result['profit']:>8.2f}")
    gap = abs(ems_result["profit"] - sim_result["profit"])
    print(f"  Tracking gap:              ${gap:>8.2f}")
    print(f"  Final SOC:                 {sim_result['soc'][-1]:.3f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
