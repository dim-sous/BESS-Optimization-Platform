# tracking_mpc — controlled-experiment baseline for `economic_mpc`

**Pitch-visible:** no (controlled-experiment baseline, not a competitor)
**Composition:** `EconomicEMS` (planner) + `TrackingMPC` (closed-loop MPC layer)

## What it does

Same `EconomicEMS` planner as `ems_clamps` and `economic_mpc`. On top of
the hourly EMS plan, runs a per-minute MPC that **tracks** the plan in
SOC and (P_chg, P_dis) space. `P_reg` is exogenous to the MPC — it is
set hourly by the EMS and ZOH-expanded across the MPC horizon, exactly
as in `economic_mpc`.

The MPC's cost terms:

- `Q_soc * (SOC − soc_ref)²` — closed-loop SOC tracking against the EMS plan
- `R_power * (P_chg − p_chg_ref)² + (P_dis − p_dis_ref)²` — power-reference tracking
- `R_delta * ΔP²` — rate-of-change smoothness on the unblocked control window
- `Q_terminal * (SOC[N] − soc_ref[N])²` — terminal SOC anchor
- `slack_penalty * (eps² + eps_temp² + eps_endurance²)` — soft constraint slacks

The MPC also enforces a short-horizon endurance constraint
(`MPCParams.endurance_hours_mpc`, default 5 minutes) so the predicted SOC
keeps enough headroom to sustain the EMS-committed `P_reg` in either
direction.

## Why it exists

`tracking_mpc` and `economic_mpc` differ in **exactly one place**: the cost
function. They share the same prediction model, the same exogenous P_reg
handling, the same endurance constraint, the same fallback. That makes
the two-strategy comparison a controlled experiment instead of a strawman.

The audit's restated proposition is: if `tracking_mpc` ≥ `economic_mpc`
across regimes, then the economic formulation isn't pulling its weight
and the production strategy should be reconsidered. Empirically (post
audit big experiment) this is what the data shows: `tracking_mpc` is
either tied with or slightly better than `economic_mpc` on every tested
subset.

## History

Pre-2026-04-15 this file described a "Q_soc=1e4 dominated old v5 stack"
with a fictitious P_reg decision variable that the adapter discarded —
a strawman baseline that made the economic_mpc vs tracking_mpc
comparison meaningless. The audit redesigned `TrackingMPC` to drop the
fictitious decision variable, add the endurance constraint, and use
the same exogenous P_reg handling as `economic_mpc`. After the
redesign, the comparison is honest.
