# economic_mpc — the v5 product

**Pitch-visible:** yes (the production strategy)
**Composition:** `EconomicEMS` + `EconomicMPC` (activation-aware) + `RegulationController` (PI)

## What it does

Three-layer hierarchical control:

1. **EMS (hourly)** — stochastic NLP over forecast scenarios with
   non-anticipativity. Plans hourly setpoints and FCR commitments
   under price uncertainty.

2. **Economic MPC (per minute)** — refines the hourly plan at minute
   resolution using:
   - Forecast-mean energy prices over a 1-hour horizon
   - The current observed activation signal (the MPC's information edge)
   - OU persistence forecast: future activation = obs · exp(−k·dt/τ_OU)
   - A soft anchor on the EMS strategic SOC plan
   - Split degradation cost (arbitrage vs reg)

3. **PI controller (per 4 s)** — closes the loop on activation delivery
   in the signed (P_net, P_reg) form. Net dispatch only — wash trades
   impossible by construction. Power budget enforced.

## Information edges over `deterministic_lp`

| Edge | LP can use? | Economic MPC uses? |
|---|---|---|
| Forecast-mean energy prices | ✓ | ✓ |
| Real-time activation reaction | ✗ (open-loop) | ✓ |
| OU activation persistence | ✗ | ✓ |
| Closed-loop state feedback | ✗ | ✓ |

## Pitch story

> Same forecast information as the commercial baseline; better execution
> because the closed-loop layer reacts to the actual grid signal as it
> happens, and the MPC plans SOC headroom proactively for upcoming
> regulation dispatch. The result is fewer delivery failures, lower
> battery wear, and (modest) profit improvement on volatile days.
