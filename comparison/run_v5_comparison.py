"""Strategy comparison for v5 on real German market data.

Runs the six configured strategies through the linear simulator core
on real EPEX SPOT day-ahead + SMARD FCR prices (Q1 2024).

Pitch-visible (B2B):
  1. rule_based       — naive price-sorted dispatch, no FCR
  2. deterministic_lp — commercial-baseline rolling-horizon LP
  3. economic_mpc     — v5 product (stochastic EMS + economic MPC + PI)

Internal sanity checks (NOT in pitch deck):
  4. ems_clamps       — stochastic EMS + open-loop dispatch
  5. ems_pi           — stochastic EMS + PI (no MPC)
  6. tracking_mpc     — stochastic EMS + tracking MPC + PI (old v5 stack)

All strategies share identical conditions per day: same realized prices,
same forecast scenarios (realized day held out), same activation seed.

Saves structured results to results/v5_comparison.json.

Usage:
    uv run python comparison/run_v5_comparison.py            # 1 day (quick)
    uv run python comparison/run_v5_comparison.py --full     # 84 days
    uv run python comparison/run_v5_comparison.py -n 10      # custom day count
    uv run python comparison/run_v5_comparison.py --days 0,3,41
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing
import pathlib
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config.parameters import (  # noqa: E402
    BatteryParams,
    EKFParams,
    ElectricalParams,
    EMSParams,
    MHEParams,
    MPCParams,
    PackParams,
    RegControllerParams,
    RegulationParams,
    ThermalParams,
    TimeParams,
)
from core.markets.price_loader import RealPriceLoader  # noqa: E402
from core.simulator.core import run_simulation  # noqa: E402

# Strategy recipes — each module exposes `make_strategy(**params) -> Strategy`
from strategies.deterministic_lp.strategy import make_strategy as _ms_lp  # noqa: E402
from strategies.economic_mpc.strategy import make_strategy as _ms_econ  # noqa: E402
from strategies.ems_clamps.strategy import make_strategy as _ms_clamps  # noqa: E402
from strategies.ems_pi.strategy import make_strategy as _ms_pi  # noqa: E402
from strategies.rule_based.strategy import make_strategy as _ms_rb  # noqa: E402
from strategies.tracking_mpc.strategy import make_strategy as _ms_track  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# Data paths (real prices live in v4's data directory)
ENERGY_CSV = REPO_ROOT / "archive" / "v4_electrical_rc_model" / "data" / "real_prices" / "de_day_ahead_2024_q1.csv"
REG_CSV = REPO_ROOT / "archive" / "v4_electrical_rc_model" / "data" / "real_prices" / "de_fcr_regulation_2024_q1.csv"
RESULTS_DIR = REPO_ROOT / "results"

# Strategy registry. The order is also the print/plot order.
# Pitch deck renders only the "pitch_visible" subset.
STRATEGY_FACTORIES = [
    ("rule_based",       _ms_rb),
    ("ems_clamps",       _ms_clamps),
    ("ems_pi",           _ms_pi),
    ("deterministic_lp", _ms_lp),
    ("tracking_mpc",     _ms_track),
    ("economic_mpc",     _ms_econ),
]
STRAT_NAMES = [name for name, _ in STRATEGY_FACTORIES]

STRATEGY_LABELS = {
    "rule_based":       "Rule-Based",
    "ems_clamps":       "Stochastic EMS (sanity)",
    "ems_pi":           "EMS + PI (sanity)",
    "deterministic_lp": "Commercial Baseline",
    "tracking_mpc":     "Tracking MPC (sanity)",
    "economic_mpc":     "Economic MPC (v5)",
}

# Strategies shown in the B2B pitch deck.
PITCH_VISIBLE = {"rule_based", "deterministic_lp", "economic_mpc"}


# =========================================================================
#  Per-day worker (runs all 4 strategies for one day)
# =========================================================================

def _run_single_day(args: tuple) -> dict:
    """Run all configured strategies for one day. Designed for Pool.map."""
    (day_idx, forecast_e, forecast_r, probabilities,
     realized_e_prices, realized_r_prices,
     bp, tp, ep, mp, ekf_p, mhe_p, thp, elp,
     reg_ctrl_p, reg_p, pp) = args

    logging.disable(logging.WARNING)

    params = dict(
        bp=bp, tp=tp, ep=ep, mp=mp, ekf_p=ekf_p, mhe_p=mhe_p,
        thp=thp, elp=elp, reg_ctrl_p=reg_ctrl_p, reg_p=reg_p, pp=pp,
    )

    day_result = {"day_idx": day_idx}

    for name, factory in STRATEGY_FACTORIES:
        t0 = time.perf_counter()
        strategy = factory(**params)

        results = run_simulation(
            strategy=strategy,
            forecast_e=forecast_e,
            forecast_r=forecast_r,
            probabilities=probabilities,
            realized_e_prices=realized_e_prices,
            realized_r_prices=realized_r_prices,
            **params,
        )
        del strategy
        gc.collect()

        mpc_t = results.get("mpc_solve_times", np.array([]))
        day_result[name] = {
            "total_profit": float(results["total_profit"]),
            "energy_profit": float(results["energy_profit_total"]),
            "capacity_revenue": float(results["capacity_revenue"]),
            "delivery_revenue": float(results["delivery_revenue"]),
            "penalty_cost": float(results["penalty_cost"]),
            "net_regulation_profit": float(results["net_regulation_profit"]),
            "deg_cost": float(results["deg_cost_total"]),
            "delivery_score": float(results["delivery_score"]),
            "soh_degradation": float(results["soh_degradation"]),
            "final_soc": float(results["soc_true"][-1]),
            "final_soh": float(results["soh_true"][-1]),
            "mpc_solver_failures": int(results.get("mpc_solver_failures", 0)),
            "avg_mpc_solve_time_s": float(np.mean(mpc_t)) if len(mpc_t) > 0 else 0.0,
            "max_mpc_solve_time_s": float(np.max(mpc_t)) if len(mpc_t) > 0 else 0.0,
            "max_temp_degC": float(np.max(results["temp_true"])),
            "wall_time": time.perf_counter() - t0,
        }

    return day_result


# =========================================================================
#  Results aggregation and printing
# =========================================================================

def aggregate_results(all_days: list[dict]) -> dict[str, dict]:
    """Aggregate per-day results into arrays keyed by strategy."""
    agg = {s: {
        "profits": [], "energy_profits": [], "capacity_revenues": [],
        "delivery_revenues": [], "penalty_costs": [], "reg_net_profits": [],
        "deg_costs": [], "delivery_scores": [], "soh_degs": [],
        "wall_times": [], "mpc_failures": [],
    } for s in STRAT_NAMES}

    for day in all_days:
        for s in STRAT_NAMES:
            d = day[s]
            agg[s]["profits"].append(d["total_profit"])
            agg[s]["energy_profits"].append(d["energy_profit"])
            agg[s]["capacity_revenues"].append(d["capacity_revenue"])
            agg[s]["delivery_revenues"].append(d["delivery_revenue"])
            agg[s]["penalty_costs"].append(d["penalty_cost"])
            agg[s]["reg_net_profits"].append(d["net_regulation_profit"])
            agg[s]["deg_costs"].append(d["deg_cost"])
            agg[s]["delivery_scores"].append(d["delivery_score"])
            agg[s]["soh_degs"].append(d["soh_degradation"])
            agg[s]["wall_times"].append(d["wall_time"])
            agg[s]["mpc_failures"].append(d["mpc_solver_failures"])

    return agg


def print_results(agg: dict[str, dict], n_days: int) -> None:
    """Print revenue breakdown, profit statistics, and timing."""
    def _row(label: str, values: list, fmt: str = "14.2f") -> None:
        print(f"  {label:22s}", end="")
        for v in values:
            if isinstance(v, int):
                print(f"  {v:14d}", end="")
            elif isinstance(v, str):
                print(f"  {v:>14s}", end="")
            else:
                print(f"  {v:{fmt}}", end="")
        print()

    def _sep() -> None:
        print(f"  {'─' * 22}" + f"  {'─' * 14}" * len(STRAT_NAMES))

    def _header() -> None:
        print(f"  {'':22s}", end="")
        for s in STRAT_NAMES:
            print(f"  {STRATEGY_LABELS[s]:>14s}", end="")
        print()
        _sep()

    # Revenue breakdown (mean $/day)
    print(f"\n  Revenue Breakdown (mean $/day, {n_days} days):")
    _header()
    for label, key, sign in [
        ("Energy revenue", "energy_profits", 1),
        ("Capacity revenue", "capacity_revenues", 1),
        ("Delivery revenue", "delivery_revenues", 1),
        ("Penalty cost", "penalty_costs", -1),
        ("Degradation cost", "deg_costs", -1),
    ]:
        _row(label, [sign * np.mean(agg[s][key]) for s in STRAT_NAMES])
    _sep()
    _row("Net profit", [np.mean(agg[s]["profits"]) for s in STRAT_NAMES])

    # Profit distribution
    print(f"\n  Profit Distribution ($/day):")
    _header()
    for label, fn in [
        ("Mean", np.mean),
        ("Median", np.median),
        ("Std (day-to-day)", np.std),
        ("P5", lambda a: np.percentile(a, 5)),
        ("P95", lambda a: np.percentile(a, 95)),
        ("Worst day", np.min),
        ("Best day", np.max),
    ]:
        _row(label, [fn(agg[s]["profits"]) for s in STRAT_NAMES])
    _row("Loss days", [int(np.sum(np.array(agg[s]["profits"]) < 0)) for s in STRAT_NAMES])

    # Delivery & degradation
    print(f"\n  Delivery & Degradation:")
    _header()
    _row("Avg delivery score",
         [f"{np.mean(agg[s]['delivery_scores'])*100:.1f}%" for s in STRAT_NAMES])
    _row("SOH %/day",
         [np.mean(agg[s]["soh_degs"]) * 100 for s in STRAT_NAMES], "14.5f")
    _row("MPC failures (total)",
         [int(np.sum(agg[s]["mpc_failures"])) for s in STRAT_NAMES])

    # ---- Pitch comparison: economic_mpc vs rule_based and deterministic_lp ----
    if "economic_mpc" in agg and "deterministic_lp" in agg:
        econ = np.array(agg["economic_mpc"]["profits"])
        rb = np.array(agg["rule_based"]["profits"])
        lp = np.array(agg["deterministic_lp"]["profits"])

        print(f"\n  [PITCH] Economic MPC vs Rule-Based:")
        adv = econ - rb
        print(f"    Advantage:  ${adv.mean():.2f}/day  "
              f"({(adv > 0).mean() * 100:.0f}% win rate)")
        print(f"    Annual (200 kWh): ${adv.mean() * 365:.0f}")
        print(f"    Annual (50 MWh):  ${adv.mean() * 365 * 250:,.0f}")

        print(f"\n  [PITCH] Economic MPC vs Commercial Baseline (LP):")
        adv = econ - lp
        pct = adv.mean() / max(abs(lp.mean()), 1e-6) * 100
        print(f"    Advantage:  ${adv.mean():.2f}/day "
              f"({pct:+.1f}% vs LP, {(adv > 0).mean() * 100:.0f}% win rate)")
        print(f"    Annual (50 MWh):  ${adv.mean() * 365 * 250:,.0f}")

    # ---- Sanity comparison: economic vs tracking MPC ----
    if "tracking_mpc" in agg and "economic_mpc" in agg:
        econ = np.array(agg["economic_mpc"]["profits"])
        trk = np.array(agg["tracking_mpc"]["profits"])
        print(f"\n  [SANITY] Economic MPC vs Tracking MPC:")
        adv = econ - trk
        print(f"    Advantage:  ${adv.mean():.2f}/day  "
              f"({(adv > 0).mean() * 100:.0f}% win rate)")

    # Wall time
    print(f"\n  Wall Time (mean s/day):")
    _header()
    _row("Mean", [np.mean(agg[s]["wall_times"]) for s in STRAT_NAMES])
    _row("Max", [np.max(agg[s]["wall_times"]) for s in STRAT_NAMES])


# =========================================================================
#  Main
# =========================================================================

def main() -> None:
    """Run strategy comparison on real German market data."""
    parser = argparse.ArgumentParser(description="v5 strategy comparison")
    parser.add_argument("--full", action="store_true", help="Run all 84 days")
    parser.add_argument("-n", type=int, default=None, help="Number of contiguous days from day 0")
    parser.add_argument(
        "--days", type=str, default=None,
        help="Comma-separated specific day indices (e.g. '3,41,86'). Overrides -n/--full.",
    )
    args = parser.parse_args()

    if args.days is not None:
        DAY_INDICES = [int(d) for d in args.days.split(",")]
        N_DAYS = len(DAY_INDICES)
    else:
        DAY_INDICES = None
        if args.n is not None:
            N_DAYS = args.n
        elif args.full:
            N_DAYS = 84
        else:
            N_DAYS = 1

    N_FORECAST = 5     # Forecast scenarios per day (realized excluded)

    # Parameters
    bp = BatteryParams()
    tp = TimeParams()
    ep = EMSParams()
    mp = MPCParams()
    ekf_p = EKFParams()
    mhe_p = MHEParams()
    thp = ThermalParams()
    elp = ElectricalParams()
    pp = PackParams()
    reg_ctrl_p = RegControllerParams()
    reg_p = RegulationParams()

    # Load real prices
    loader = RealPriceLoader(ENERGY_CSV, reg_csv=REG_CSV, seed=42)
    if DAY_INDICES is None:
        DAY_INDICES = list(range(min(N_DAYS, loader.n_days)))
    n_days = len(DAY_INDICES)
    n_hours_total = int(tp.sim_hours) + ep.N_ems

    stats = loader.price_stats
    print("=" * 78)
    print("  V5 STRATEGY COMPARISON — Real German Market Data")
    print("=" * 78)
    print(f"  Data:       EPEX SPOT DE-LU + SMARD FCR, Q1 2024 ({stats['n_days']} days)")
    print(f"  Battery:    {bp.E_nom_kwh:.0f} kWh / {bp.P_max_kw:.0f} kW")
    print(f"  Days:       {n_days}")
    print(f"  Forecasts:  {N_FORECAST} other days per run "
          f"(realized day held out — fixed v5 info-leak)")
    print(f"  Strategies: {', '.join(STRATEGY_LABELS[name] for name, _ in STRATEGY_FACTORIES)}")
    print(f"  Total sims: {n_days * len(STRATEGY_FACTORIES)}")
    print("=" * 78)
    print()

    # Build jobs: one per requested day
    jobs = []
    for day_idx in DAY_INDICES:
        forecast_e, forecast_r, probs, realized_e, realized_r = (
            loader.generate_scenarios_for_day(
                day_idx, n_hours=n_hours_total, n_scenarios=N_FORECAST,
            )
        )
        jobs.append((
            day_idx, forecast_e, forecast_r, probs,
            realized_e, realized_r,
            bp, tp, ep, mp, ekf_p, mhe_p, thp, elp,
            reg_ctrl_p, reg_p, pp,
        ))

    # Run with multiprocessing
    n_workers = min(len(jobs), max(1, multiprocessing.cpu_count() - 1), 2)
    print(f"  Running {len(jobs)} days x {len(STRATEGY_FACTORIES)} strategies "
          f"across {n_workers} workers...")
    print(f"  Estimated runtime: ~{n_days * 6 / n_workers / 60:.0f} hours "
          f"(MPC strategies dominate at ~3 min/day with JIT)\n")
    t0 = time.perf_counter()

    all_days: list[dict] = []
    with multiprocessing.Pool(n_workers) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_run_single_day, jobs, chunksize=1), 1
        ):
            all_days.append(result)
            elapsed = time.perf_counter() - t0
            eta = elapsed / i * (len(jobs) - i)
            day_idx = result["day_idx"]
            headline_key = "economic_mpc" if "economic_mpc" in result else "tracking_mpc"
            head_profit = result[headline_key]["total_profit"]
            head_score = result[headline_key]["delivery_score"]
            print(
                f"  [{i:3d}/{len(jobs)}] "
                f"Day {day_idx:2d} done  "
                f"{headline_key}: profit=${head_profit:6.2f}  "
                f"delivery={head_score*100:5.1f}%  "
                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
                flush=True,
            )

    wall = time.perf_counter() - t0
    print(f"\n\n  Done in {wall:.0f}s ({wall / 60:.1f} min, "
          f"{wall / n_days:.1f}s/day)\n")

    # Sort by day index for consistent output
    all_days.sort(key=lambda d: d["day_idx"])

    # Aggregate
    agg = aggregate_results(all_days)
    print_results(agg, n_days)

    # Build per-day JSON summary
    per_day_json = []
    for day in all_days:
        entry = {"day_idx": day["day_idx"]}
        for s in STRAT_NAMES:
            entry[s] = day[s]
        per_day_json.append(entry)

    # Mean scalars across all days (for presentation)
    mean_scalars = {}
    for s in STRAT_NAMES:
        mean_scalars[s] = {
            "total_profit": float(np.mean(agg[s]["profits"])),
            "energy_profit": float(np.mean(agg[s]["energy_profits"])),
            "capacity_revenue": float(np.mean(agg[s]["capacity_revenues"])),
            "delivery_revenue": float(np.mean(agg[s]["delivery_revenues"])),
            "penalty_cost": float(np.mean(agg[s]["penalty_costs"])),
            "net_regulation_profit": float(np.mean(agg[s]["reg_net_profits"])),
            "deg_cost": float(np.mean(agg[s]["deg_costs"])),
            "delivery_score": float(np.mean(agg[s]["delivery_scores"])),
            "soh_degradation": float(np.mean(agg[s]["soh_degs"])),
            "mpc_solver_failures": int(np.sum(agg[s]["mpc_failures"])),
            "avg_mpc_solve_time_s": float(np.mean(agg[s]["wall_times"])),
            "loss_days": int(np.sum(np.array(agg[s]["profits"]) < 0)),
            "win_rate_vs_rule_based": float(
                np.mean(np.array(agg[s]["profits"]) > np.array(agg["rule_based"]["profits"]))
            ) if s != "rule_based" else None,
        }

    # Per-day profit arrays (for charts)
    daily_profits = {s: [d[s]["total_profit"] for d in all_days] for s in STRAT_NAMES}
    daily_delivery = {s: [d[s]["delivery_score"] for d in all_days] for s in STRAT_NAMES}

    comparison = {
        "meta": {
            "version": "v5_core_refactor",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_source": "EPEX SPOT DE-LU + SMARD FCR, Q1 2024",
            "n_days": n_days,
            "n_forecast_scenarios": N_FORECAST,
            "E_nom_kwh": bp.E_nom_kwh,
            "P_max_kw": bp.P_max_kw,
            "sim_hours": tp.sim_hours,
            "dt_ems_s": tp.dt_ems,
            "dt_mpc_s": tp.dt_mpc,
            "dt_pi_s": tp.dt_pi,
            "n_cells": pp.n_cells,
            "SOC_min": bp.SOC_min,
            "SOC_max": bp.SOC_max,
            "strategy_labels": STRATEGY_LABELS,
            "pitch_visible": sorted(PITCH_VISIBLE),
            "wall_time_total_s": round(wall, 1),
        },
        "strategies": mean_scalars,
        "daily_profits": daily_profits,
        "daily_delivery": daily_delivery,
        "per_day": per_day_json,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "v5_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, default=float)

    print(f"\n  Saved: {out_path}")
    print(f"  Size:  {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
