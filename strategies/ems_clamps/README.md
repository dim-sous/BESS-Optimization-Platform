# ems_clamps — internal sanity check, NOT in pitch deck

**Pitch-visible:** no (sanity-check only)
**Composition:** `EconomicEMS` (stochastic NLP planner) + open-loop dispatch (no PI, no MPC)

## What it does

Runs the stochastic EMS planner (5 forecast scenarios with non-anticipativity)
and dispatches its hourly setpoints open-loop. Activation modulation is
applied directly without PI feedback.

## Why it exists

To isolate the value of the **stochastic formulation** vs the deterministic LP.
If `ems_clamps` doesn't beat `deterministic_lp`, the scenario hedging isn't
worth the extra solver time — and we should consider dropping the stochastic
EMS in favour of the LP everywhere.

## Why it's not in the pitch deck

It uses our own EMS planner with a vendor-style execution layer — not a
realistic competitor. It would be misleading to call it "the industry baseline".
The honest commercial baseline is `deterministic_lp`.
