"""Metrics computation for BESS platform version comparison.

Computes all mandatory performance metrics from a simulation results dict.
Shared across all version folders.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np


def compute_all_metrics(
    results: dict[str, Any],
    version_tag: str,
    dt_sim: float = 1.0,
    dt_mpc: float = 60.0,
) -> dict[str, float | str]:
    """Compute all comparison metrics from simulation results.

    Parameters
    ----------
    results : dict
        Output from ``MultiRateSimulator.run()``.
    version_tag : str
        Version identifier (e.g. ``"v1_baseline"``).
    dt_sim : float
        Plant simulation time step [s].
    dt_mpc : float
        MPC / estimator time step [s].

    Returns
    -------
    dict
        Flat dictionary with all metric values.
    """
    steps_per_mpc = int(dt_mpc / dt_sim)

    metrics: dict[str, float | str] = {"version": version_tag}

    # ------------------------------------------------------------------
    #  Control performance
    # ------------------------------------------------------------------
    soc_ref = results.get("soc_ref_at_mpc")
    if soc_ref is not None and len(soc_ref) > 0:
        n = len(soc_ref)
        soc_true_at_mpc = results["soc_true"][::steps_per_mpc][:n]
        metrics["rmse_soc_tracking"] = float(
            np.sqrt(np.mean((soc_true_at_mpc - soc_ref) ** 2))
        )
    else:
        metrics["rmse_soc_tracking"] = float("nan")

    power_ref = results.get("power_ref_at_mpc")
    power_applied = results.get("power_applied")
    if power_ref is not None and power_applied is not None and len(power_ref) > 0:
        n = min(len(power_ref), len(power_applied))
        diff = power_applied[:n] - power_ref[:n]
        metrics["rmse_power_tracking"] = float(np.sqrt(np.mean(diff ** 2)))
    else:
        metrics["rmse_power_tracking"] = float("nan")

    # ------------------------------------------------------------------
    #  Estimation performance
    # ------------------------------------------------------------------
    soc_true_at_est = results["soc_true"][::steps_per_mpc]
    soh_true_at_est = results["soh_true"][::steps_per_mpc]

    for tag, key in [("ekf", "soc_ekf"), ("mhe", "soc_mhe")]:
        est = results.get(key)
        if est is not None and len(est) > 0:
            n = min(len(est), len(soc_true_at_est))
            metrics[f"rmse_soc_{tag}"] = float(
                np.sqrt(np.mean((est[:n] - soc_true_at_est[:n]) ** 2))
            )
        else:
            metrics[f"rmse_soc_{tag}"] = float("nan")

    for tag, key in [("ekf", "soh_ekf"), ("mhe", "soh_mhe")]:
        est = results.get(key)
        if est is not None and len(est) > 0:
            n = min(len(est), len(soh_true_at_est))
            metrics[f"rmse_soh_{tag}"] = float(
                np.sqrt(np.mean((est[:n] - soh_true_at_est[:n]) ** 2))
            )
        else:
            metrics[f"rmse_soh_{tag}"] = float("nan")

    # Temperature estimation (if available)
    temp_true = results.get("temp_true")
    if temp_true is not None:
        temp_true_at_est = temp_true[::steps_per_mpc]
        for tag, key in [("ekf", "temp_ekf"), ("mhe", "temp_mhe")]:
            est = results.get(key)
            if est is not None and len(est) > 0:
                n = min(len(est), len(temp_true_at_est))
                metrics[f"rmse_temp_{tag}"] = float(
                    np.sqrt(np.mean((est[:n] - temp_true_at_est[:n]) ** 2))
                )

    # ------------------------------------------------------------------
    #  Economic performance
    # ------------------------------------------------------------------
    metrics["total_profit_usd"] = float(results.get("total_profit", 0.0))
    deg_cost = results.get("deg_cost")
    metrics["total_degradation_cost_usd"] = float(np.sum(deg_cost)) if deg_cost is not None else 0.0

    # ------------------------------------------------------------------
    #  Computational performance
    # ------------------------------------------------------------------
    mpc_times = results.get("mpc_solve_times")
    if mpc_times is not None and len(mpc_times) > 0:
        metrics["avg_mpc_solve_time_s"] = float(np.mean(mpc_times))
        metrics["max_mpc_solve_time_s"] = float(np.max(mpc_times))
    else:
        metrics["avg_mpc_solve_time_s"] = float("nan")
        metrics["max_mpc_solve_time_s"] = float("nan")

    est_times = results.get("est_solve_times")
    if est_times is not None and len(est_times) > 0:
        metrics["avg_estimator_solve_time_s"] = float(np.mean(est_times))
        metrics["max_estimator_solve_time_s"] = float(np.max(est_times))
    else:
        metrics["avg_estimator_solve_time_s"] = float("nan")
        metrics["max_estimator_solve_time_s"] = float("nan")

    # ------------------------------------------------------------------
    #  Summary scalars
    # ------------------------------------------------------------------
    metrics["final_soc"] = float(results["soc_true"][-1])
    metrics["final_soh"] = float(results["soh_true"][-1])
    metrics["soh_degradation_pct"] = float(
        (results["soh_true"][0] - results["soh_true"][-1]) * 100.0
    )

    return metrics


def save_metrics(metrics: dict, path: str | pathlib.Path) -> None:
    """Save metrics dict to JSON."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")
