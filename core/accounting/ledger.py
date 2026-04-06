"""Pure-function accounting ledger.

Reads `SimTraces` (which only contains physically-applied power values
from the plant) and computes the full result dict: revenues, profits,
SOH degradation, delivery score, etc.

This is a **pure function** of (traces, realized_prices, params). No
state mutation, no I/O — easy to unit-test in isolation. Every dollar
in the result dict is traceable back to a numbered slot in `traces`.

Backward compatibility: the result dict shape matches the legacy
`MultiRateSimulator.run()` output exactly, except:
  - `power_applied` is derived from the new (P_net, P_reg) trace and
    re-exposed as the legacy (N, 3) [P_chg, P_dis, P_reg] shape so
    downstream visualization works without changes. Wash trades are
    impossible — exactly one of (P_chg, P_dis) is nonzero per row.
"""

from __future__ import annotations

import numpy as np

from core.config.parameters import (
    BatteryParams,
    EMSParams,
    RegulationParams,
    TimeParams,
)
from core.simulator.traces import SimTraces


def compute_ledger(
    traces: SimTraces,
    realized_e_prices: np.ndarray,        # (n_hours,) [$/kWh]
    realized_r_prices: np.ndarray,        # (n_hours,) [$/kW/h]
    bp: BatteryParams,
    tp: TimeParams,
    ep: EMSParams,
    reg_p: RegulationParams,
    strategy_name: str,
    strategy_metadata: dict,
) -> dict:
    """Compute the full result dict from a finished simulation trace."""
    N_sim = traces.n_sim_steps
    dt_pi = tp.dt_pi
    dt_pi_h = dt_pi / 3600.0
    steps_per_hour = int(3600 / dt_pi)

    # Trace columns
    p_net = traces.power_applied[:, 0]                  # (N_sim,) signed
    p_reg_committed = traces.p_reg_committed             # (N_sim,) committed FCR
    p_delivered = traces.p_delivered                     # (N_sim,) signed
    activation = traces.activation                        # (N_sim,)

    # Hour index for each PI step (clipped to realized array length)
    hour_idx = np.minimum(np.arange(N_sim) // steps_per_hour, len(realized_e_prices) - 1)
    realized_e_at = realized_e_prices[hour_idx]
    realized_r_at = realized_r_prices[hour_idx]

    # ---- 1. Energy revenue (signed) ----
    # P_net > 0 = discharge = revenue, P_net < 0 = charge = cost
    energy_profit_arr_pi = p_net * realized_e_at * dt_pi_h
    energy_profit_total = float(np.sum(energy_profit_arr_pi))

    # ---- 2. Capacity revenue ----
    # Always earned for committing P_reg, regardless of delivery
    capacity_revenue_arr_pi = p_reg_committed * realized_r_at * dt_pi_h
    capacity_revenue = float(np.sum(capacity_revenue_arr_pi))

    # ---- 3. Delivery revenue and non-delivery penalty ----
    p_demanded = np.abs(activation * p_reg_committed)
    p_delivered_abs = np.abs(p_delivered)
    p_missed = np.maximum(0.0, p_demanded - p_delivered_abs)

    delivery_revenue_arr_pi = reg_p.price_activation * p_delivered_abs * dt_pi_h
    delivery_revenue = float(np.sum(delivery_revenue_arr_pi))

    penalty_arr_pi = reg_p.penalty_mult * realized_r_at * p_missed * dt_pi_h
    penalty_cost = float(np.sum(penalty_arr_pi))

    net_regulation_profit = capacity_revenue + delivery_revenue - penalty_cost

    # ---- 4. Delivery score: fraction of active PI steps with successful delivery ----
    is_active = p_demanded > 1e-3
    n_active = int(np.sum(is_active))
    # Per-step "delivered within tolerance" indicator (True for inactive steps too)
    within_tol_full = np.ones(N_sim, dtype=bool)
    if n_active > 0:
        within_tol_full[is_active] = (
            p_missed[is_active] / p_demanded[is_active]
        ) <= reg_p.delivery_tolerance
    if n_active == 0:
        delivery_score = 1.0
    else:
        delivery_score = float(np.sum(within_tol_full[is_active]) / n_active)

    # ---- 5. Degradation cost ----
    # alpha_deg applies to arbitrage throughput |P_net|, alpha_deg_reg
    # applies to committed reg power |P_reg|.
    deg_cost_arr_pi = (
        ep.degradation_cost
        * (bp.alpha_deg * np.abs(p_net) + bp.alpha_deg_reg * np.abs(p_reg_committed))
        * dt_pi
    )
    deg_cost_total = float(np.sum(deg_cost_arr_pi))

    # ---- 6. Total profit ----
    total_profit = energy_profit_total + net_regulation_profit - deg_cost_total

    # ---- 7. Per-MPC-step aggregations (for visualization continuity) ----
    steps_per_mpc = int(tp.dt_mpc / tp.dt_pi)
    N_mpc = traces.n_mpc_steps
    energy_profit_per_mpc = np.zeros(N_mpc)
    deg_cost_per_mpc = np.zeros(N_mpc)
    for m in range(N_mpc):
        s, e = m * steps_per_mpc, (m + 1) * steps_per_mpc
        energy_profit_per_mpc[m] = float(np.sum(energy_profit_arr_pi[s:e]))
        deg_cost_per_mpc[m] = float(np.sum(deg_cost_arr_pi[s:e]))

    # ---- 8. Re-expose power_applied in legacy (N, 3) [chg, dis, reg] shape ----
    # The trace stores (N, 2) [P_net, P_reg]. Decompose so visualization
    # and the comparison harness see the familiar shape. Wash trades are
    # impossible — exactly one of chg/dis is nonzero per row.
    p_chg_legacy = np.where(p_net < 0, -p_net, 0.0)
    p_dis_legacy = np.where(p_net > 0, p_net, 0.0)
    power_applied_legacy = np.column_stack([p_chg_legacy, p_dis_legacy, p_reg_committed])

    # power_mpc_base in legacy (N_mpc, 3) shape, derived from setpoint_at_mpc
    pnet_mpc = traces.setpoint_at_mpc[:, 0]
    preg_mpc = traces.setpoint_at_mpc[:, 1]
    pchg_mpc = np.where(pnet_mpc < 0, -pnet_mpc, 0.0)
    pdis_mpc = np.where(pnet_mpc > 0, pnet_mpc, 0.0)
    power_mpc_base_legacy = np.column_stack([pchg_mpc, pdis_mpc, preg_mpc])
    power_ref_at_mpc_legacy = power_mpc_base_legacy.copy()

    # reg_accounting array (N_sim, 4): [cap, del, pen, delivered_ok]
    is_ok_arr = within_tol_full.astype(float)
    reg_accounting = np.column_stack([
        capacity_revenue_arr_pi,
        delivery_revenue_arr_pi,
        penalty_arr_pi,
        is_ok_arr,
    ])

    # ---- 9. Build result dict (matches legacy shape for backward compat) ----
    result = {
        "strategy": strategy_name,
        "strategy_metadata": strategy_metadata,
        # Time axes
        "time_sim": np.arange(N_sim + 1) * tp.dt_sim,
        "time_mpc": np.arange(N_mpc + 1) * tp.dt_mpc,
        # Plant trajectories
        "soc_true": traces.soc_true,
        "soh_true": traces.soh_true,
        "temp_true": traces.temp_true,
        "vrc1_true": traces.vrc1_true,
        "vrc2_true": traces.vrc2_true,
        "vterm_true": traces.vterm_true,
        # Estimator trajectories
        "soc_ekf": traces.soc_ekf,
        "soh_ekf": traces.soh_ekf,
        "temp_ekf": traces.temp_ekf,
        "vrc1_ekf": traces.vrc1_ekf,
        "vrc2_ekf": traces.vrc2_ekf,
        # MHE not used in the new simulator (could be added later)
        "soc_mhe": np.zeros(N_mpc + 1),
        "soh_mhe": np.zeros(N_mpc + 1),
        "temp_mhe": np.zeros(N_mpc + 1),
        "vrc1_mhe": np.zeros(N_mpc + 1),
        "vrc2_mhe": np.zeros(N_mpc + 1),
        # Power (legacy 3-vector shape for backward compat)
        "power_applied": power_applied_legacy,
        "power_mpc_base": power_mpc_base_legacy,
        # NEW: signed power for audit/debug. Wash-trade-free by construction.
        "power_applied_signed": traces.power_applied,  # (N, 2) [P_net, P_reg]
        # Activation and regulation
        "activation_signal": activation,
        "power_delivered": p_delivered,
        "reg_accounting": reg_accounting,
        "delivery_score": delivery_score,
        "capacity_revenue": capacity_revenue,
        "delivery_revenue": delivery_revenue,
        "penalty_cost": penalty_cost,
        "net_regulation_profit": net_regulation_profit,
        # Energy & degradation profit
        "energy_profit_total": energy_profit_total,
        "energy_profit": energy_profit_per_mpc,
        "deg_cost_total": deg_cost_total,
        "deg_cost": deg_cost_per_mpc,
        "total_profit": total_profit,
        "soh_degradation": float(traces.soh_true[0] - traces.soh_true[-1]),
        # EMS reference history
        "ems_soc_refs": traces.ems_soc_refs,
        # Forecast prices (for plotting context)
        "prices_energy": realized_e_prices,
        "prices_reg": realized_r_prices,
        # Timing
        "mpc_solve_times": traces.mpc_solve_times,
        "est_solve_times": traces.est_solve_times,
        "mpc_solver_failures": traces.mpc_solver_failures,
        # References at MPC resolution
        "soc_ref_at_mpc": traces.soc_ref_at_mpc,
        "power_ref_at_mpc": power_ref_at_mpc_legacy,
        # Multi-cell pack data
        "cell_socs": traces.cell_socs,
        "cell_sohs": traces.cell_sohs,
        "cell_temps": traces.cell_temps,
        "cell_vrc1s": traces.cell_vrc1s,
        "cell_vrc2s": traces.cell_vrc2s,
        "n_cells": traces.n_cells,
    }

    return result
