# rule_based

**Pitch-visible:** yes
**Composition:** `RuleBasedPlanner` (no MPC, no PI)

## What it does

Sorts the day's forecast-mean energy prices, charges during the cheapest
hours and discharges during the most expensive. No FCR commitment, no
optimisation, no closed-loop control. The execution layer applies activation
modulation in an open-loop way (which is a no-op here since `P_reg = 0`).

## Why it exists

Strict lower bound for the comparison harness. Demonstrates what a buyer
would get from a basic in-house dispatcher with zero optimisation and no
participation in frequency regulation markets. Every other strategy should
beat it by a wide margin — that's the whole point.

## Information visibility

- **Sees**: probability-weighted mean of the forecast scenarios
- **Does not see**: realized prices (held out for accounting only)

## Tunable parameters

- Schedule sized to the battery's usable energy capacity
- Charges/discharges at `0.8 * P_max_kw`
- Active for `n_hours_needed = ceil(usable_kwh / power)` hours per direction
