# ems_clamps — canonical "EMS alone" baseline

**Pitch-visible:** yes (canonical "EMS alone" pitch baseline)
**Composition:** `EconomicEMS` (stochastic NLP planner) + open-loop dispatch (no MPC)

## What it does

Runs the stochastic EMS planner (5 forecast scenarios with non-anticipativity)
once per hour. Holds the planner's hourly setpoint constant for the next
hour, dispatched through the plant. Activation tracking lives inside the
plant (post-RF1), so there is no strategy-layer reaction to the FCR signal —
the plant clips and serves activation against the held setpoint.

## Why it exists

The canonical **"EMS alone, no MPC layer"** strategy. The current pitch
hypothesis is that `economic_mpc` must be strictly ≥ `ems_clamps` on every
metric to justify the MPC layer's existence at all. Empirically (post-audit)
this hypothesis does not hold on day 0 nor on the calm/volatile/stressed
big-experiment subsets — `ems_clamps` is competitive with or beats both
MPC strategies in every regime tested.

## Information visibility

- **Sees**: probability-weighted forecast scenarios (5 alt days from Q1 2024,
  realized day held out)
- **Does not see**: realized prices, sub-minute activation samples
- **State feedback**: the EKF runs every minute regardless of strategy, so
  the EMS uses the latest EKF state estimate at the top of each hourly
  re-solve. Within an hour the held setpoint is dispatched open-loop.

## Notes

- "Clamps" refers to the hard SOC bounds the EMS NLP enforces; there is
  no separate clamp layer beyond the EMS's own constraints and the plant's
  physical clipping.
- This is the same `EconomicEMS` planner used by `tracking_mpc` and
  `economic_mpc`; the only difference between those strategies and
  `ems_clamps` is whether a per-minute MPC re-optimization layer sits on
  top of the EMS plan.
