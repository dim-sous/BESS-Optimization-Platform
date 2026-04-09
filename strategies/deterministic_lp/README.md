# deterministic_lp

**Pitch-visible:** yes
**Composition:** `DeterministicLP` planner (no MPC, no PI)

## What it does

Solves a rolling-horizon linear program every hour over the forecast-mean
energy and FCR-capacity prices. Includes endurance constraints, terminal
SOC anchor, and the same degradation cost as the stochastic EMS — so the
LP and stochastic EMS optimise the same physical objective and the
comparison is apples-to-apples.

The LP outputs hourly setpoints that the simulator dispatches open-loop.
Activation tracking lives inside the plant (post-RF1), so the FCR signal
is served against the held LP setpoint with no strategy-layer feedback.
This matches how most commercial BESS EMS products actually operate.

## Why it exists

The **honest commercial baseline** for the B2B pitch. Every commercial BESS
EMS vendor on the market today ships some form of deterministic LP / MILP
dispatch over forecast prices, with no closed-loop second-stage controller.
Beating this is the v5 product's value proposition.

## Information visibility

- **Sees**: probability-weighted mean of the forecast scenarios
- **Does not see**: realized prices (held out for accounting only)

## Notes

- The LP is *charitably* given the forecast mean, not a noisy forecast.
  Real vendors use noisier forecasts and would do worse.
- Solver: `scipy.optimize.linprog` with HiGHS backend.
- ~6 s/day runtime — negligible vs the MPC strategies.
