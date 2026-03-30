"""Fair strategy comparison with Monte Carlo forecast sampling.

Runs four control strategies through identical simulation loops on real
German day-ahead + FCR prices (Q1 2024):

  1. Full optimizer  (EMS + MPC + EKF)
  2. EMS + estimator (EMS + EKF, no MPC)
  3. EMS only        (EMS, true SOC, no estimation)
  4. Rule-based      (price-sorted schedule, no optimization)

All strategies share:
  - Identical BatteryPack plant (same initial conditions)
  - Single profit accounting formula (energy + regulation - degradation - penalty)
  - Same time resolution (dt_sim, dt_mpc)

Monte Carlo: each day is simulated N_MC times with different forecast
scenario draws, giving confidence intervals on profit.

Usage:
    uv run python -m comparison.run_comparison
    uv run python comparison/run_comparison.py
"""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing
import pathlib
import sys
import time
from abc import ABC, abstractmethod

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
V4_ROOT = REPO_ROOT / "v4_electrical_rc_model"
if str(V4_ROOT) not in sys.path:
    sys.path.insert(0, str(V4_ROOT))

from config.parameters import (  # noqa: E402
    BatteryParams, EKFParams, ElectricalParams, EMSParams,
    MPCParams, PackParams, ThermalParams, TimeParams,
)
from data.real_price_loader import RealPriceLoader  # noqa: E402
from ems.economic_ems import EconomicEMS  # noqa: E402
from estimation.ekf import ExtendedKalmanFilter  # noqa: E402
from models.battery_model import BatteryPack  # noqa: E402
from mpc.tracking_mpc import TrackingMPC  # noqa: E402
from simulation.simulator import interpolate_ems_to_mpc  # noqa: E402

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout,
)

ENERGY_CSV = V4_ROOT / "data" / "real_prices" / "de_day_ahead_2024_q1.csv"
REG_CSV = V4_ROOT / "data" / "real_prices" / "de_fcr_regulation_2024_q1.csv"
RESULTS_DIR = REPO_ROOT / "results"

# ---------------------------------------------------------------------------
# Strategy names (ordered from simplest to most complex)
# ---------------------------------------------------------------------------
STRAT_NAMES = ["rule_based", "ems_only", "ems_est", "optimizer"]


# ===========================================================================
#  Strategy base class
# ===========================================================================

class ControlStrategy(ABC):
    """Base class — all strategies return u = [P_chg, P_dis, P_reg]."""

    @abstractmethod
    def get_command(
        self,
        sim_step: int,
        x_true: np.ndarray,
        y_meas: np.ndarray,
        u_prev: np.ndarray,
        steps_per_ems: int,
        steps_per_mpc: int,
        energy_scenarios: np.ndarray,
        reg_scenarios: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray: ...


# ===========================================================================
#  Strategy 1: Full Optimizer (EMS + MPC + EKF)
# ===========================================================================

class FullOptimizerStrategy(ControlStrategy):
    """Complete control stack. MHE removed — its output was unused."""

    def __init__(self, bp, tp, ep, mp, ekf_p, thp, elp, pp):
        self.bp, self.tp, self.ep = bp, tp, ep
        self.ems = EconomicEMS(bp, tp, ep, thp, elp)
        self.mpc = TrackingMPC(bp, tp, mp, thp, elp)
        self.ekf = ExtendedKalmanFilter(bp, tp, ekf_p, thp, elp)
        self.mp = mp
        self._mpc_refs = None
        self._mpc_ref_base = 0
        self._u_current = np.zeros(3)

    def get_command(self, sim_step, x_true, y_meas, u_prev,
                    steps_per_ems, steps_per_mpc,
                    energy_scenarios, reg_scenarios, probabilities):
        bp, tp, ep = self.bp, self.tp, self.ep

        # --- EMS update (every dt_ems) ---
        if sim_step % steps_per_ems == 0:
            x_est = self.ekf.get_estimate()
            ems_hour = sim_step // steps_per_ems
            remaining = max(min(ep.N_ems, energy_scenarios.shape[1] - ems_hour), 1)

            e_scen = energy_scenarios[:, ems_hour:ems_hour + remaining]
            r_scen = reg_scenarios[:, ems_hour:ems_hour + remaining]
            if e_scen.shape[1] < ep.N_ems:
                pad = ep.N_ems - e_scen.shape[1]
                e_scen = np.pad(e_scen, ((0, 0), (0, pad)), mode="edge")
                r_scen = np.pad(r_scen, ((0, 0), (0, pad)), mode="edge")

            ems_result = self.ems.solve(
                soc_init=x_est[0], soh_init=x_est[1], t_init=x_est[2],
                vrc1_init=x_est[3], vrc2_init=x_est[4],
                energy_scenarios=e_scen, reg_scenarios=r_scen,
                probabilities=probabilities,
            )
            self._mpc_refs = interpolate_ems_to_mpc(ems_result, tp.dt_ems, tp.dt_mpc)
            self._mpc_ref_base = 0

        # --- MPC + EKF update (every dt_mpc) ---
        if sim_step % steps_per_mpc == 0 and self._mpc_refs is not None:
            if sim_step > 0:
                ekf_est = self.ekf.step(self._u_current, y_meas)
            else:
                ekf_est = self.ekf.get_estimate()

            refs = self._mpc_refs
            N = self.mp.N_mpc
            b = self._mpc_ref_base

            def _pad(arr, target):
                return arr[:target] if len(arr) >= target else np.pad(
                    arr, (0, target - len(arr)), mode="edge")

            pc_win = _pad(refs["P_chg_ref_mpc"][b:b + N], N)
            pd_win = _pad(refs["P_dis_ref_mpc"][b:b + N], N)
            pr_win = _pad(refs["P_reg_ref_mpc"][b:b + N], N)
            soc_win = _pad(refs["SOC_ref_mpc"][b:b + N + 1], N + 1)
            soh_win = _pad(refs["SOH_ref_mpc"][b:b + N + 1], N + 1)
            temp_win = _pad(refs["TEMP_ref_mpc"][b:b + N + 1], N + 1)

            try:
                self._u_current = self.mpc.solve(
                    x_est=ekf_est, soc_ref=soc_win, soh_ref=soh_win,
                    temp_ref=temp_win, p_chg_ref=pc_win, p_dis_ref=pd_win,
                    p_reg_ref=pr_win, u_prev=self._u_current,
                )
            except Exception:
                self._u_current = np.zeros(3)

            self._mpc_ref_base += 1

        return self._u_current.copy()


# ===========================================================================
#  Strategy 2: EMS + Estimator (EKF only, no MPC)
# ===========================================================================

class EMSEstimatorStrategy(ControlStrategy):
    """EMS plans hourly, EKF provides state feedback. No MPC tracking."""

    def __init__(self, bp, tp, ep, ekf_p, thp, elp):
        self.bp, self.tp, self.ep = bp, tp, ep
        self.ems = EconomicEMS(bp, tp, ep, thp, elp)
        self.ekf = ExtendedKalmanFilter(bp, tp, ekf_p, thp, elp)
        self._hourly_cmd = np.zeros(3)
        self._u_current = np.zeros(3)

    def get_command(self, sim_step, x_true, y_meas, u_prev,
                    steps_per_ems, steps_per_mpc,
                    energy_scenarios, reg_scenarios, probabilities):
        ep = self.ep

        # EKF update at MPC rate
        if sim_step % steps_per_mpc == 0 and sim_step > 0:
            self.ekf.step(self._u_current, y_meas)

        # EMS update (every dt_ems)
        if sim_step % steps_per_ems == 0:
            x_est = self.ekf.get_estimate()
            ems_hour = sim_step // steps_per_ems
            remaining = max(min(ep.N_ems, energy_scenarios.shape[1] - ems_hour), 1)

            e_scen = energy_scenarios[:, ems_hour:ems_hour + remaining]
            r_scen = reg_scenarios[:, ems_hour:ems_hour + remaining]
            if e_scen.shape[1] < ep.N_ems:
                pad = ep.N_ems - e_scen.shape[1]
                e_scen = np.pad(e_scen, ((0, 0), (0, pad)), mode="edge")
                r_scen = np.pad(r_scen, ((0, 0), (0, pad)), mode="edge")

            ems_result = self.ems.solve(
                soc_init=x_est[0], soh_init=x_est[1], t_init=x_est[2],
                vrc1_init=x_est[3], vrc2_init=x_est[4],
                energy_scenarios=e_scen, reg_scenarios=r_scen,
                probabilities=probabilities,
            )
            self._hourly_cmd = np.array([
                float(ems_result["P_chg_ref"][0]),
                float(ems_result["P_dis_ref"][0]),
                float(ems_result["P_reg_ref"][0]),
            ])

        self._u_current = self._hourly_cmd.copy()
        return self._u_current.copy()


# ===========================================================================
#  Strategy 3: EMS Only (true plant SOC, no estimation)
# ===========================================================================

class EMSOnlyStrategy(ControlStrategy):
    def __init__(self, bp, tp, ep, thp, elp):
        self.bp, self.tp, self.ep = bp, tp, ep
        self.ems = EconomicEMS(bp, tp, ep, thp, elp)
        self._hourly_cmd = np.zeros(3)

    def get_command(self, sim_step, x_true, y_meas, u_prev,
                    steps_per_ems, steps_per_mpc,
                    energy_scenarios, reg_scenarios, probabilities):
        ep = self.ep

        if sim_step % steps_per_ems == 0:
            ems_hour = sim_step // steps_per_ems
            remaining = max(min(ep.N_ems, energy_scenarios.shape[1] - ems_hour), 1)

            e_scen = energy_scenarios[:, ems_hour:ems_hour + remaining]
            r_scen = reg_scenarios[:, ems_hour:ems_hour + remaining]
            if e_scen.shape[1] < ep.N_ems:
                pad = ep.N_ems - e_scen.shape[1]
                e_scen = np.pad(e_scen, ((0, 0), (0, pad)), mode="edge")
                r_scen = np.pad(r_scen, ((0, 0), (0, pad)), mode="edge")

            ems_result = self.ems.solve(
                soc_init=float(x_true[0]),
                soh_init=float(x_true[1]),
                t_init=float(x_true[2]),
                vrc1_init=float(x_true[3]),
                vrc2_init=float(x_true[4]),
                energy_scenarios=e_scen, reg_scenarios=r_scen,
                probabilities=probabilities,
            )
            self._hourly_cmd = np.array([
                float(ems_result["P_chg_ref"][0]),
                float(ems_result["P_dis_ref"][0]),
                float(ems_result["P_reg_ref"][0]),
            ])

        return self._hourly_cmd.copy()


# ===========================================================================
#  Strategy 4: Rule-Based (price-sorted, fixed schedule)
# ===========================================================================

class RuleBasedStrategy(ControlStrategy):
    def __init__(self, bp, ep):
        self.bp = bp
        self.reg_fraction = ep.regulation_fraction
        self._schedule = None

    def plan(self, forecast_scenarios: np.ndarray, probabilities: np.ndarray) -> None:
        """Build a fixed 24h schedule from probability-weighted forecast."""
        bp = self.bp
        expected_prices = np.average(
            forecast_scenarios[:, :24], axis=0, weights=probabilities)

        sorted_hours = np.argsort(expected_prices)
        usable = (bp.SOC_max - bp.SOC_min) * bp.E_nom_kwh
        n_ch = int(np.ceil(usable / bp.P_max_kw))
        charge_hours = set(sorted_hours[:n_ch])
        discharge_hours = set(sorted_hours[-n_ch:])
        overlap = charge_hours & discharge_hours
        charge_hours -= overlap
        discharge_hours -= overlap

        self._schedule = np.zeros((24, 3))
        for h in range(24):
            if h in charge_hours:
                self._schedule[h, 0] = bp.P_max_kw
            elif h in discharge_hours:
                self._schedule[h, 1] = bp.P_max_kw
            self._schedule[h, 2] = bp.P_max_kw * self.reg_fraction

    def get_command(self, sim_step, x_true, y_meas, u_prev,
                    steps_per_ems, steps_per_mpc,
                    energy_scenarios, reg_scenarios, probabilities):
        hour = min(sim_step // steps_per_ems, 23)
        return self._schedule[hour].copy()


# ===========================================================================
#  Unified simulation loop
# ===========================================================================

def unified_sim_loop(
    strategy: ControlStrategy,
    plant: BatteryPack,
    bp: BatteryParams,
    tp: TimeParams,
    ep: EMSParams,
    realized_energy_prices: np.ndarray,
    realized_reg_prices: np.ndarray,
    forecast_energy_scen: np.ndarray,
    forecast_reg_scen: np.ndarray,
    forecast_probs: np.ndarray,
    store_timeseries: bool = False,
) -> dict:
    """Run one 24h simulation. Returns profit components and optional time-series.

    Profit formula (identical for all strategies):
        energy  = price_e * (P_dis - P_chg) * dt_h
        reg     = price_r * P_reg * dt_h          (capacity payment)
        deg     = deg_cost * alpha_deg * (P_chg + P_dis + P_reg) * dt_mpc
        penalty = (1 + mult) * reg  when SOC near limits and P_reg > 0
        net     = energy + reg - deg - penalty
    """
    total_seconds = int(tp.sim_hours * 3600)
    steps_per_mpc = int(tp.dt_mpc / tp.dt_sim)
    steps_per_ems = int(tp.dt_ems / tp.dt_sim)
    N_sim_steps = int(total_seconds / tp.dt_sim)
    N_mpc_steps = N_sim_steps // steps_per_mpc
    dt_h = tp.dt_mpc / 3600.0

    # Profit components (cumulative)
    cum_energy = 0.0
    cum_reg = 0.0
    cum_deg = 0.0
    cum_penalty = 0.0

    u_current = np.zeros(3)
    soh_init = plant.get_state()[1]

    # Time-series at dt_mpc resolution
    if store_timeseries:
        ts_soc = np.zeros(N_mpc_steps)
        ts_soh = np.zeros(N_mpc_steps)
        ts_temp = np.zeros(N_mpc_steps)
        ts_power = np.zeros((N_mpc_steps, 3))
        ts_cum_profit = np.zeros(N_mpc_steps)

    mpc_idx = 0
    for sim_step in range(N_sim_steps):
        # Query plant and update command only at dt_mpc boundaries
        if sim_step % steps_per_mpc == 0:
            x_true = plant.get_state()
            y_meas = plant.get_measurement()

            u_current = strategy.get_command(
                sim_step=sim_step,
                x_true=x_true,
                y_meas=y_meas,
                u_prev=u_current,
                steps_per_ems=steps_per_ems,
                steps_per_mpc=steps_per_mpc,
                energy_scenarios=forecast_energy_scen,
                reg_scenarios=forecast_reg_scen,
                probabilities=forecast_probs,
            )

            # --- Profit accounting ---
            ems_hour = min(sim_step // steps_per_ems,
                          len(realized_energy_prices) - 1)
            price_e = float(realized_energy_prices[ems_hour])
            price_r = float(realized_reg_prices[ems_hour])

            e_profit = price_e * (u_current[1] - u_current[0]) * dt_h
            reg_gross = price_r * u_current[2] * dt_h
            d_cost = (ep.degradation_cost * bp.alpha_deg
                      * (u_current[0] + u_current[1] + u_current[2])
                      * tp.dt_mpc)

            # Regulation delivery penalty
            penalty = 0.0
            soc = x_true[0]
            if u_current[2] > 0.1:
                can_deliver = (soc > bp.SOC_min + ep.reg_soc_margin
                               and soc < bp.SOC_max - ep.reg_soc_margin)
                if not can_deliver:
                    penalty = (1.0 + ep.reg_penalty_mult) * reg_gross

            cum_energy += e_profit
            cum_reg += reg_gross
            cum_deg += d_cost
            cum_penalty += penalty

            # Time-series
            if store_timeseries and mpc_idx < N_mpc_steps:
                ts_soc[mpc_idx] = x_true[0]
                ts_soh[mpc_idx] = x_true[1]
                ts_temp[mpc_idx] = x_true[2]
                ts_power[mpc_idx] = u_current
                ts_cum_profit[mpc_idx] = cum_energy + cum_reg - cum_deg - cum_penalty

            mpc_idx += 1

        # Step the plant
        plant.step(u_current)

    x_final = plant.get_state()
    net_profit = cum_energy + cum_reg - cum_deg - cum_penalty

    result = {
        "profit": float(net_profit),
        "energy_profit": float(cum_energy),
        "reg_profit": float(cum_reg),
        "deg_cost": float(cum_deg),
        "reg_penalty": float(cum_penalty),
        "soh_degradation": float(soh_init - x_final[1]),
        "final_soc": float(x_final[0]),
        "final_soh": float(x_final[1]),
    }

    if store_timeseries:
        result["ts_soc"] = ts_soc
        result["ts_soh"] = ts_soh
        result["ts_temp"] = ts_temp
        result["ts_power"] = ts_power
        result["ts_cum_profit"] = ts_cum_profit

    return result


# ===========================================================================
#  Per-job worker  (one job = one day × one MC draw)
# ===========================================================================

def _run_single_job(args: tuple) -> dict:
    """Run all 4 strategies for one (day, mc_draw). Designed for Pool.map."""
    (day_idx, mc_idx, store_ts,
     realized_energy_24, realized_reg_24,
     forecast_energy, forecast_reg, forecast_probs,
     bp, tp, ep, mp, ekf_p, thp, elp, pp) = args

    logging.disable(logging.WARNING)

    results = {"day_idx": day_idx, "mc_idx": mc_idx}

    strategies = {
        "rule_based": lambda: _make_rule_based(bp, ep, forecast_energy, forecast_probs),
        "ems_only": lambda: EMSOnlyStrategy(bp, tp, ep, thp, elp),
        "ems_est": lambda: EMSEstimatorStrategy(bp, tp, ep, ekf_p, thp, elp),
        "optimizer": lambda: FullOptimizerStrategy(bp, tp, ep, mp, ekf_p, thp, elp, pp),
    }

    n_hours_needed = int(tp.sim_hours) + ep.N_ems
    re_padded = np.pad(realized_energy_24,
                       (0, max(0, n_hours_needed - len(realized_energy_24))),
                       mode="edge")
    rr_padded = np.pad(realized_reg_24,
                       (0, max(0, n_hours_needed - len(realized_reg_24))),
                       mode="edge")

    for name, make_strategy in strategies.items():
        t0 = time.perf_counter()
        plant = BatteryPack(bp, tp, thp, elp, pp)
        strategy = make_strategy()

        res = unified_sim_loop(
            strategy=strategy,
            plant=plant,
            bp=bp, tp=tp, ep=ep,
            realized_energy_prices=re_padded,
            realized_reg_prices=rr_padded,
            forecast_energy_scen=forecast_energy,
            forecast_reg_scen=forecast_reg,
            forecast_probs=forecast_probs,
            store_timeseries=store_ts,
        )
        res["wall_time"] = time.perf_counter() - t0
        results[name] = res

    return results


def _make_rule_based(bp, ep, forecast_energy, forecast_probs):
    strategy = RuleBasedStrategy(bp, ep)
    strategy.plan(forecast_energy, forecast_probs)
    return strategy


# ===========================================================================
#  Scenario generation (realized prices NOT in forecast set)
# ===========================================================================

def build_day_scenarios(
    loader: RealPriceLoader,
    day_idx: int,
    n_forecast_scenarios: int,
    rng: np.random.Generator,
    n_hours: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build forecast scenarios that do NOT contain the realized day.

    Returns (realized_energy_24, realized_reg_24,
             forecast_energy, forecast_reg, forecast_probs).
    """
    realized_energy_24 = loader.get_day(day_idx)

    if loader.has_real_regulation:
        realized_reg_24 = loader._daily_reg[day_idx].copy()
    else:
        realized_reg_24 = 0.4 * realized_energy_24 + 0.006

    other_days = [i for i in range(loader.n_days) if i != day_idx]
    chosen = rng.choice(other_days, size=n_forecast_scenarios, replace=False)

    forecast_energy = np.zeros((n_forecast_scenarios, n_hours))
    forecast_reg = np.zeros((n_forecast_scenarios, n_hours))

    for s_idx, d_idx in enumerate(chosen):
        e48, r48 = loader._build_48h(d_idx)
        forecast_energy[s_idx, :n_hours] = e48[:n_hours]
        forecast_reg[s_idx, :n_hours] = r48[:n_hours]

    forecast_energy = np.maximum(forecast_energy, 0.001)
    forecast_reg = np.maximum(forecast_reg, 0.0)
    forecast_probs = np.ones(n_forecast_scenarios) / n_forecast_scenarios

    return (realized_energy_24, realized_reg_24,
            forecast_energy, forecast_reg, forecast_probs)


# ===========================================================================
#  Printing
# ===========================================================================

def _print_row(label: str, values: list, fmt: str = "14.2f") -> None:
    print(f"  {label:22s}", end="")
    for v in values:
        if isinstance(v, int):
            print(f"  {v:14d}", end="")
        else:
            print(f"  {v:{fmt}}", end="")
    print()


def _print_sep() -> None:
    print(f"  {'─' * 22}", end="")
    for _ in STRAT_NAMES:
        print(f"  {'─' * 14}", end="")
    print()


def _print_header() -> None:
    print(f"  {'':22s}", end="")
    for s in STRAT_NAMES:
        print(f"  {s:>14s}", end="")
    print()
    _print_sep()


def print_results(agg: dict, n_days: int, n_mc: int) -> None:
    """Print revenue breakdown, profit statistics, and wall time."""
    EUR = 1 / 1.08

    # --- Revenue Breakdown ---
    print(f"\n  Revenue Breakdown (mean EUR/day, {n_days} days x {n_mc} MC):")
    _print_header()

    for label, key, sign in [
        ("Energy revenue", "energy_profits", 1),
        ("Regulation revenue", "reg_profits", 1),
        ("Degradation cost", "deg_costs", -1),
        ("Regulation penalty", "reg_penalties", -1),
    ]:
        vals = [sign * np.mean(agg[s][key]) * EUR for s in STRAT_NAMES]
        _print_row(label, vals)

    _print_sep()
    vals = [np.mean(agg[s]["profits"]) * EUR for s in STRAT_NAMES]
    _print_row("Net profit", vals)

    # --- Profit Distribution ---
    print(f"\n  Profit Distribution (EUR/day):")
    _print_header()

    # Per-day means (average across MC draws for each day, then stats across days)
    day_means = {}
    for s in STRAT_NAMES:
        profits = np.array(agg[s]["profits"])  # shape (n_days * n_mc,)
        day_means[s] = profits.reshape(n_days, n_mc).mean(axis=1) * EUR

    mc_stds = {}
    for s in STRAT_NAMES:
        profits = np.array(agg[s]["profits"]).reshape(n_days, n_mc) * EUR
        mc_stds[s] = profits.std(axis=1).mean()

    for label, fn in [
        ("Mean", lambda dm: np.mean(dm)),
        ("Median", lambda dm: np.median(dm)),
        ("Std (day-to-day)", lambda dm: np.std(dm)),
        ("Mean MC std", None),
        ("P5", lambda dm: np.percentile(dm, 5)),
        ("P95", lambda dm: np.percentile(dm, 95)),
        ("Worst day", lambda dm: np.min(dm)),
        ("Best day", lambda dm: np.max(dm)),
    ]:
        if label == "Mean MC std":
            vals = [mc_stds[s] for s in STRAT_NAMES]
        else:
            vals = [fn(day_means[s]) for s in STRAT_NAMES]
        _print_row(label, vals)

    _print_row("Loss days",
               [int(np.sum(day_means[s] < 0)) for s in STRAT_NAMES])
    _print_row("SOH %/day",
               [np.mean(agg[s]["soh_degs"]) * 100 for s in STRAT_NAMES])

    # --- Optimizer vs Rule-Based ---
    opt = day_means["optimizer"]
    rb = day_means["rule_based"]
    adv = opt - rb
    print(f"\n  Optimizer vs Rule-Based:")
    print(f"    Advantage:  EUR {adv.mean():.2f}/day  "
          f"({(adv > 0).mean() * 100:.0f}% win rate)")
    print(f"    Annual (200kWh): EUR {adv.mean() * 365:.0f}")
    print(f"    Annual (10MWh):  EUR {adv.mean() * 365 * 50:,.0f}")
    print(f"    Annual (50MWh):  EUR {adv.mean() * 365 * 250:,.0f}")

    # --- Wall Time ---
    print(f"\n  Wall Time (mean s/day):")
    _print_header()
    _print_row("Mean",
               [np.mean(agg[s]["wall_times"]) for s in STRAT_NAMES])
    _print_row("Max",
               [np.max(agg[s]["wall_times"]) for s in STRAT_NAMES])


# ===========================================================================
#  Main
# ===========================================================================

def main() -> None:
    # ---- Configuration (adjust here) ----
    N_DAYS = 30
    N_MC = 3          # Monte Carlo draws per day (forecast scenario variation)
    N_FORECAST = 5    # forecast scenarios per draw

    # ---- Calibrated parameters ----
    bp = dataclasses.replace(BatteryParams(), alpha_deg=4.76e-11)
    tp = dataclasses.replace(TimeParams(), dt_mpc=300.0, dt_estimator=300.0, dt_sim=60.0)
    ep = dataclasses.replace(EMSParams(), degradation_cost=36_500.0)
    mp = dataclasses.replace(MPCParams(), N_mpc=12, Nc_mpc=4)
    ekf_p = EKFParams()
    thp = dataclasses.replace(ThermalParams(), R_internal=0.072, h_cool=150.0,
                              C_thermal=300_000.0)
    elp = dataclasses.replace(ElectricalParams(), R0=0.0324, R1=0.0216, R2=0.0180)
    pp = PackParams()

    loader = RealPriceLoader(ENERGY_CSV, reg_csv=REG_CSV, seed=42)
    rng = np.random.default_rng(seed=777)
    n_hours_total = int(tp.sim_hours) + ep.N_ems
    n_days = min(N_DAYS, loader.n_days)

    print("=" * 70)
    print("  STRATEGY COMPARISON [v4_electrical_rc_model]")
    print("=" * 70)
    stats = loader.price_stats
    print(f"  Data:       German day-ahead + FCR, Q1 2024 ({stats['n_days']} days)")
    print(f"  Strategies: {', '.join(STRAT_NAMES)}")
    print(f"  Days:       {n_days}  |  MC draws/day: {N_MC}"
          f"  |  Total sims: {n_days * N_MC * len(STRAT_NAMES)}")
    print(f"  Forecasts:  {N_FORECAST} other days per draw (realized NOT in set)")
    print(f"  Reg penalty: {ep.reg_penalty_mult}x capacity price when SOC near limits")
    print("=" * 70)
    print()

    # ---- Build jobs: one per (day, mc_draw) ----
    jobs = []
    for day_idx in range(n_days):
        for mc_idx in range(N_MC):
            re24, rr24, fe, fr, fp = build_day_scenarios(
                loader, day_idx, n_forecast_scenarios=N_FORECAST,
                rng=rng, n_hours=n_hours_total,
            )
            store_ts = (mc_idx == 0)
            jobs.append((day_idx, mc_idx, store_ts,
                         re24, rr24, fe, fr, fp,
                         bp, tp, ep, mp, ekf_p, thp, elp, pp))

    # ---- Run ----
    n_workers = min(len(jobs), multiprocessing.cpu_count(), 1)
    print(f"  Running {len(jobs)} jobs ({n_days} days x {N_MC} MC x "
          f"{len(STRAT_NAMES)} strategies) across {n_workers} workers...\n")
    t0 = time.perf_counter()

    with multiprocessing.Pool(n_workers) as pool:
        all_results = []
        n_jobs = len(jobs)
        for i, result in enumerate(pool.imap(_run_single_job, jobs, chunksize=1), 1):
            all_results.append(result)
            elapsed = time.perf_counter() - t0
            eta = elapsed / i * (n_jobs - i)
            print(f"\r  Progress: {i}/{n_jobs} jobs "
                  f"({i * 100 / n_jobs:.0f}%) "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
                  end="", flush=True)

    wall = time.perf_counter() - t0
    print(f"\n\n  Done in {wall:.0f}s ({wall / n_days:.1f}s/day, "
          f"{wall / len(jobs):.1f}s/job)\n")

    # ---- Aggregate ----
    # Sort results by (day_idx, mc_idx) for consistent reshape
    all_results.sort(key=lambda r: (r["day_idx"], r["mc_idx"]))

    agg = {s: {
        "profits": [], "energy_profits": [], "reg_profits": [],
        "deg_costs": [], "reg_penalties": [],
        "soh_degs": [], "wall_times": [],
    } for s in STRAT_NAMES}

    per_day_summary = []
    for r in all_results:
        for s in STRAT_NAMES:
            sr = r[s]
            agg[s]["profits"].append(sr["profit"])
            agg[s]["energy_profits"].append(sr["energy_profit"])
            agg[s]["reg_profits"].append(sr["reg_profit"])
            agg[s]["deg_costs"].append(sr["deg_cost"])
            agg[s]["reg_penalties"].append(sr["reg_penalty"])
            agg[s]["soh_degs"].append(sr["soh_degradation"])
            agg[s]["wall_times"].append(sr["wall_time"])

    # Print
    print_results(agg, n_days=n_days, n_mc=N_MC)

    # ---- Save JSON (scalars only, no time-series) ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build per-day aggregated summary
    json_days = []
    for d in range(n_days):
        day_entry = {"day_idx": d}
        for s in STRAT_NAMES:
            start = d * N_MC
            end = start + N_MC
            profits = [agg[s]["profits"][i] for i in range(start, end)]
            day_entry[s] = {
                "mean_profit": float(np.mean(profits)),
                "std_profit": float(np.std(profits)),
                "mean_energy_profit": float(np.mean(
                    agg[s]["energy_profits"][start:end])),
                "mean_reg_profit": float(np.mean(
                    agg[s]["reg_profits"][start:end])),
                "mean_deg_cost": float(np.mean(
                    agg[s]["deg_costs"][start:end])),
                "mean_reg_penalty": float(np.mean(
                    agg[s]["reg_penalties"][start:end])),
                "mean_soh_degradation": float(np.mean(
                    agg[s]["soh_degs"][start:end])),
                "mean_wall_time": float(np.mean(
                    agg[s]["wall_times"][start:end])),
            }
        json_days.append(day_entry)

    summary = {
        "config": {
            "n_days": n_days,
            "n_mc": N_MC,
            "n_forecast_scenarios": N_FORECAST,
            "version": "v4_electrical_rc_model",
            "dt_sim": tp.dt_sim,
            "dt_mpc": tp.dt_mpc,
        },
        "per_day": json_days,
    }

    out_path = RESULTS_DIR / "v4_comparison.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
