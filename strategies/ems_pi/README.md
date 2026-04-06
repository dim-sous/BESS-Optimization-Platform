# ems_pi — internal sanity check, NOT in pitch deck

**Pitch-visible:** no (sanity-check only)
**Composition:** `EconomicEMS` planner + `RegulationController` (PI), no MPC

## What it does

Stochastic EMS plans hourly setpoints; PI feedback handles activation
delivery in the (P_net, P_reg) signed-power form. No MPC trajectory layer.

## Why it exists

Verifies the value of the PI feedback layer in isolation. If a strategy
that adds an MPC on top doesn't meaningfully beat `ems_pi`, the MPC is
just expensive noise.

## Why it's not in the pitch deck

Same reason as `ems_clamps`: uses our own EMS, not a realistic competitor.
