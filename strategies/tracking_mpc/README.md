# tracking_mpc — Tracking NLP MPC (controlled-experiment baseline)

**Pitch-visible:** no (controlled-experiment baseline for `economic_mpc`)
**Composition:** `EconomicEMS` planner (hourly, see `strategies/ems_clamps`) + `TrackingMPC` (per-minute closed-loop)
**Solver:** CasADi `Opti` + IPOPT (MUMPS linear solver)

## What it does (plain words)

Same prediction model and exogenous-$P_{\text{reg}}$ handling as
`economic_mpc`. Differs in **exactly one place**: the cost function
**tracks** the EMS plan in (SOC, $P_{\text{chg}}$, $P_{\text{dis}}$)
space instead of optimizing arbitrage. Also includes a short-horizon
FCR endurance constraint that `economic_mpc` does not have.

## Optimal Control Problem

This is a **deterministic NLP solved at 60 s cadence** with a 1-hour
prediction horizon and control-horizon blocking after $N_c$ steps —
structurally identical to `economic_mpc`, with a tracking objective
substituted for the economic one and a soft endurance constraint added.

### Notation

Same notation as `economic_mpc/README.md`, plus:

| Symbol | Meaning | Default |
|---|---|---|
| $Q_{\text{soc}}$ | per-step SOC tracking weight | $10^4$ |
| $R_{\text{power}}$ | power-reference tracking weight | $1$ |
| $R^{\Delta}$ | rate-of-change penalty | $10$ |
| $Q_{\text{terminal}}$ | terminal SOC anchor | $10^5$ |
| $T^{\text{mpc}}_{\text{end}}$ | MPC short-horizon endurance | $5/60$ h $\approx 0.083$ h |
| $\lambda_{\text{soc}}, \lambda_{\text{temp}}$ | slack penalties | $10^6$, $10^7$ |
| $p^{\text{ref}}_{\text{chg},k}, p^{\text{ref}}_{\text{dis},k}$ | EMS power references at MPC cadence | — |
| $\bar{P}_{\text{reg},k}$ | committed FCR power, ZOH from EMS plan | — |

### Decision variables

$$
\begin{aligned}
P_{\text{chg},j}, \, P_{\text{dis},j} &\in [0, P_{\max}], &\quad j &= 0, \dots, N_c - 1 \\
\text{SOC}_k, \, T_k &\in \mathbb{R}, &\quad k &= 0, \dots, N \\
\varepsilon_k, \, \varepsilon^{\text{temp}}_k, \, \varepsilon^{\text{end}}_k &\geq 0, &\quad k &= 0, \dots, N
\end{aligned}
$$

For $k \geq N_c$, the control is blocked: same $P_{\text{chg}, N_c-1}$
and $P_{\text{dis}, N_c-1}$ are reused. **$P_{\text{reg}}$ is not a
decision variable** — it is the parameter $\bar{P}_{\text{reg},k}$
from the EMS plan.

### Prediction model

Identical to `economic_mpc`: 2-state RK4 with frozen SOH and the
EMS-committed $\bar{P}_{\text{reg},k}$ entering the thermal Joule
heating term:

$$
(\text{SOC}_{k+1}, T_{k+1}) = F\!\big( (\text{SOC}_k, T_k), \, (P_{\text{chg},j(k)}, P_{\text{dis},j(k)}, \bar{P}_{\text{reg},k}); \, \overline{\text{soh}} \big)
$$

### Objective

$$
\begin{aligned}
\min \quad &\sum_{k=0}^{N-1} \Big[
  \underbrace{Q_{\text{soc}} \big( \text{SOC}_k - \text{soc}^{\text{ref}}_k \big)^{2}}_{\text{SOC tracking}}
  \;+\; \underbrace{R_{\text{power}} \Big( (P_{\text{chg},j(k)} - p^{\text{ref}}_{\text{chg},k})^{2} + (P_{\text{dis},j(k)} - p^{\text{ref}}_{\text{dis},k})^{2} \Big)}_{\text{power-reference tracking}}
\Big] \\[0.6em]
&+\; R^{\Delta} \Big( (P_{\text{chg},0} - P^{\text{prev}}_{\text{chg}})^{2} + (P_{\text{dis},0} - P^{\text{prev}}_{\text{dis}})^{2} \Big) \\[0.2em]
&+\; R^{\Delta} \sum_{k=1}^{N_c - 1} \Big( (P_{\text{chg},k} - P_{\text{chg},k-1})^{2} + (P_{\text{dis},k} - P_{\text{dis},k-1})^{2} \Big) \\[0.4em]
&+\; Q_{\text{terminal}} \big( \text{SOC}_N - \text{soc}^{\text{ref}}_N \big)^{2} \\[0.2em]
&+\; \lambda_{\text{soc}} \sum_{k=0}^{N} \varepsilon_k^{2}
\;+\; \lambda_{\text{temp}} \sum_{k=0}^{N} (\varepsilon^{\text{temp}}_k)^{2}
\;+\; \lambda_{\text{soc}} \sum_{k=0}^{N} (\varepsilon^{\text{end}}_k)^{2}
\end{aligned}
$$

There is **no economic term** (no `−price·(P_dis − P_chg)·dt_h`) and
**no degradation term** in the cost. Both are subsumed into the
tracking of the EMS plan, which itself optimised against energy
prices and degradation cost at hourly cadence.

### Constraints

**Initial conditions, dynamics, SOC bounds, temperature bounds, and
power-budget headroom over the full horizon** are identical to
`economic_mpc`. See `strategies/economic_mpc/README.md` for the full
expressions. The one addition is the endurance constraint:

**Endurance (soft, at every predicted step):**

$$
\text{SOC}_k + \frac{T^{\text{mpc}}_{\text{end}} \, \eta_c}{E_{\text{nom}}} \, \bar{P}_{\text{reg},k_p} \;\leq\; \text{SOC}_{\max} + \varepsilon^{\text{end}}_k
$$

$$
\text{SOC}_k - \frac{T^{\text{mpc}}_{\text{end}}}{E_{\text{nom}} \, \eta_d} \, \bar{P}_{\text{reg},k_p} \;\geq\; \text{SOC}_{\min} - \varepsilon^{\text{end}}_k
$$

for $k = 0, \dots, N$, with $k_p = \min(k, N - 1)$ (the reg-power
index, since $\bar{P}_{\text{reg}}$ has length $N$).

The horizon $T^{\text{mpc}}_{\text{end}} = 5$ minutes is deliberately
shorter than the EMS's own $T_{\text{end}} = 30$ minutes — the EMS
already enforces 30-minute strategic headroom; the MPC adds a tactical
5-minute cushion on top, scaled to typical OU activation persistence.

### Difference vs `economic_mpc` (single-knob comparison)

| Element | `tracking_mpc` | `economic_mpc` |
|---|---|---|
| SOC tracking weight | $Q_{\text{soc}} = 10^4$ (dominant) | $Q^{\text{soc}}_{\text{anchor}} = 10$ (soft) |
| Power-reference tracking | $R_{\text{power}} \, \lVert u - u^{\text{ref}} \rVert^{2}$ | (none) |
| Energy-arbitrage term | (none) | $-\,w_e \hat{p}^e (P_{\text{dis}} - P_{\text{chg}}) \Delta t_h$ |
| Degradation in objective | (none) | $w_{\text{deg}} c_{\text{deg}} \alpha_{\text{deg}} (P_{\text{chg}} + P_{\text{dis}}) \Delta t_s$ |
| Endurance constraint | $T^{\text{mpc}}_{\text{end}} = 5$ min (soft) | (none) |
| Terminal anchor | $Q_{\text{terminal}} = 10^5$ | $Q^{\text{term}}_{\text{econ}} = 10^3$ |
| Rate-of-change penalty | $R^{\Delta} = 10$ | $R^{\Delta}_{\text{econ}} = 10^{-2}$ |

The two strategies share the **same prediction model**, the **same
exogenous $P_{\text{reg}}$ handling**, the **same fallback path**, and
the **same data inputs**. The only product-relevant degree of freedom
is the cost function. This makes the two-strategy comparison a
controlled experiment.

### What this MPC does NOT have

- **No $P_{\text{reg}}$ decision variable.** Reg power is exogenous (parameter from the EMS).
- **No stochasticity.**
- **No SOH state.** Frozen as a parameter.
- **No V_rc transient states.**
- **No multi-cell pack model.**
- **Per-step SOC anchor uses the end-of-hour reference at all $k$** in the current implementation — same audit finding F33 as `economic_mpc`.

## History

Pre-2026-04-15 this file described a "$Q_{\text{soc}} = 10^{4}$
dominated old v5 stack" with a fictitious $P_{\text{reg}}$ decision
variable that the adapter discarded — a strawman baseline that made
the `economic_mpc` vs `tracking_mpc` comparison meaningless. The audit
redesigned `TrackingMPC` to drop the fictitious decision variable,
add the endurance constraint, and use the same exogenous
$P_{\text{reg}}$ handling as `economic_mpc`. After the redesign, the
comparison is honest.
