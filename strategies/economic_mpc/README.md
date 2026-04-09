# economic_mpc — the v5 product

**Pitch-visible:** yes (the production strategy)
**Composition:** `EconomicEMS` (planner) + `EconomicMPC` (closed-loop MPC layer)

## What it does

Two-layer hierarchical control. Activation tracking lives inside the
plant (post-RF1, 2026-04-15), so there is no strategy-layer PI controller
or activation-aware MPC term — both were removed because the previous
"recent activation" input was effectively a ground-truth peek that real
hardware cannot do.

1. **EMS (hourly)** — stochastic NLP over 5 forecast scenarios with
   non-anticipativity. Plans hourly P_chg/P_dis setpoints and FCR
   commitment under price uncertainty. Uses 3-state CasADi dynamics
   (SOC, SOH, T) with `expected_activation_frac` accounting for the
   round-trip efficiency drain from FCR cycling.

2. **Economic MPC (per minute)** — re-solves a 60-minute deterministic
   NLP every minute against the live EKF state estimate. `P_reg` is
   exogenous (set hourly by the EMS, not a decision variable here). The
   MPC plans P_chg/P_dis under:
   - Forecast-mean energy prices over the 1-hour horizon
   - A soft anchor on the EMS strategic SOC plan
   - Split degradation cost (arbitrage vs reg)
   - Power-budget headroom against the EMS-committed `P_reg`,
     enforced over the **full** prediction horizon (the F18 audit fix
     extended this from the unblocked control window to the entire N
     horizon)

The MPC plans against the **expected** activation realisation
(E[signed activation] = 0, since the OU process is symmetric around
zero). It does not see the activation signal directly. Its information
edge over `ems_clamps` is closed-loop reaction to the EKF state estimate,
not visibility into the FCR signal.

## Information edges over `deterministic_lp` and `ems_clamps`

| Edge | LP | EMS clamps | Economic MPC |
|---|:---:|:---:|:---:|
| Probability-weighted forecast scenarios | ✗ (mean only) | ✓ | ✓ |
| Closed-loop EKF state feedback at MPC cadence | ✗ (open-loop) | ✗ (open-loop within hour) | ✓ |
| Stochastic non-anticipativity at planning | ✗ | ✓ | ✓ |
| Per-minute economic re-optimization | ✗ | ✗ | ✓ |

## Empirical status (post-audit)

The big-experiment results (3 subsets × 5 days × 5 strategies × 2 plant
configurations) show that on the current data pipeline (hourly day-ahead
prices, no intraday signals, no real-time imbalance prices), the
per-minute economic re-optimization layer **does not produce a
positive return** relative to the EMS-only baseline. `economic_mpc`
loses to `ems_clamps` in 28 of 30 days across all three subsets,
typically by $0.06 - $0.85 per day. The loss is largest in the volatile
subset, suggesting the failure mode is myopia: the 60-min MPC horizon
cannot see the full-day arbitrage shape that the EMS captures, and any
deviation from the EMS plan trades global optimality for local
optimality.

The MPC's economic term (`-w_e × price × (P_dis - P_chg) × dt_h`) has
no intra-hour gradient because the price is constant within an hour
(day-ahead resolution). This means the MPC's per-minute decisions are
driven entirely by the SOC anchor, not by economic signal — and any
deviation from the EMS plan is a noise correction without a revenue
gain to offset its inefficiency cost.

The MPC layer would only earn its compute cost if the data pipeline
fed it intraday price information (real-time / imbalance prices,
sub-hour FCR signals) that the EMS does not see at hourly cadence.
This is a data-pipeline gap, not an MPC implementation bug.
