# ems_clamps — Stochastic NLP (EMS)

**Pitch-visible:** yes (canonical "EMS alone" pitch baseline)
**Composition:** `EconomicEMS` (stochastic NLP planner) + open-loop dispatch (no MPC)
**Solver:** CasADi `Opti` + IPOPT (MUMPS linear solver)

## What it does (plain words)

Solves a scenario-based two-stage stochastic NLP every hour over $S$
forecast price scenarios with non-anticipativity. Uses a 3-state
nonlinear ODE (SOC, SOH, T) with Arrhenius-coupled degradation and a
linearised current/thermal coupling. Plans hourly arbitrage and FCR
commitment hedged across price uncertainty. Holds the planner's hourly
setpoint constant for the next hour and dispatches it open-loop —
there is no MPC layer.

## Optimal Control Problem

### Notation

| Symbol | Meaning | Units |
|---|---|---|
| $N$ | planning horizon | hours (= 24) |
| $S$ | number of price scenarios | (= 5) |
| $\pi_s$ | probability of scenario $s$, $\sum_s \pi_s = 1$ | — |
| $\Delta t_h, \Delta t_s$ | step size in hours, seconds | h (= 1), s (= 3600) |
| $k$ | time index, $k = 0, \dots, N-1$ | — |
| $s$ | scenario index, $s = 1, \dots, S$ | — |
| $p^e_{s,k}, p^r_{s,k}$ | per-scenario forecast prices | \$/kWh, \$/kW/h |
| $E_{\text{nom}}$ | nominal pack capacity | kWh |
| $n_m$ | modules in pack (the `n_modules` parameter) | (= 4) |
| $\eta_c, \eta_d$ | charge / discharge efficiency | — |
| $P_{\max}$ | max power | kW |
| $\text{SOC}_{\min/\max}, T_{\min/\max}$ | state bounds | — |
| $\text{SOC}_0, \text{SOH}_0, T_0$ | initial state (from EKF) | — |
| $\text{SOC}_{\text{term}}$ | terminal SOC target | — |
| $\alpha_{\text{deg}}, \alpha_{\text{reg}}$ | degradation rates | $1/(\text{kW}\cdot\text{s})$ |
| $c_{\text{deg}}$ | degradation cost | \$/SOH |
| $\bar{a}$ | $\mathbb{E}[\lvert \text{activation} \rvert]$ | — (= 0.04) |
| $T_{\text{end}}$ | endurance horizon | h (= 0.5) |
| $\kappa(T)$ | Arrhenius factor (defined below) | — |
| $Q^{\text{soc}}_{\text{term}}, Q^{\text{soh}}_{\text{term}}$ | terminal SOC/SOH penalties | (each = $10^4$) |
| $\lambda_{\text{soc}}, \lambda_{\text{end}}$ | soft-constraint slack penalties | (each = $10^5$) |

### Decision variables (per scenario $s$)

$$
\begin{aligned}
P^{s}_{\text{chg},k}, \, P^{s}_{\text{dis},k}, \, P^{s}_{\text{reg},k} &\in [0, P_{\max}], \quad k = 0, \dots, N-1 \\
\text{SOC}^{s}_k &\in \mathbb{R}, \quad k = 0, \dots, N \\
\text{SOH}^{s}_k &\in [0.5, \, 1.001], \quad k = 0, \dots, N \\
T^{s}_k &\in [T_{\min}, T_{\max}], \quad k = 0, \dots, N \\
\varepsilon^{\text{soc}, s}_k &\geq 0, \quad k = 0, \dots, N \\
\varepsilon^{\text{end}, s}_k &\geq 0, \quad k = 0, \dots, N-1
\end{aligned}
$$

### Continuous-time dynamics (3-state ODE)

$$
\begin{aligned}
\frac{d\,\text{SOC}}{dt} &= \frac{\eta_c P_{\text{chg}} - P_{\text{dis}}/\eta_d}{\text{SOH} \cdot E_{\text{nom}} \cdot 3600}
\;+\; \underbrace{\frac{(\eta_c - 1/\eta_d)\,\bar{a}\,P_{\text{reg}}}{\text{SOH} \cdot E_{\text{nom}} \cdot 3600}}_{\text{expected FCR efficiency drain}} \\[1.4em]
\frac{d\,\text{SOH}}{dt} &= -\frac{\kappa(T)}{n_m} \Big( \alpha_{\text{deg}}(P_{\text{chg}} + P_{\text{dis}}) + \alpha_{\text{reg}} |P_{\text{reg}}| \Big) \\[1em]
\frac{dT}{dt} &= \frac{Q_{\text{joule}}(P_{\text{net}}, T) - h_{\text{cool}}(T - T_{\text{amb}})}{C_{\text{thermal}}}
\end{aligned}
$$

where $P_{\text{net}} = P_{\text{dis}} - P_{\text{chg}}$, the Arrhenius
acceleration factor is

$$
\kappa(T) = \exp\!\left[\frac{E_a}{R_{\text{gas}}}\left(\frac{1}{T_{\text{ref}}^K} - \frac{1}{T^K}\right)\right]
$$

with $T^K = T + 273.15$, and the Joule heating $Q_{\text{joule}}$ uses
the linearised current $|P_{\text{net}}| \cdot 1000 / V_{\text{oc}}$
through the total DC resistance $R_0 + R_1 + R_2$.

**Discrete-time:**

$$
x^{s}_{k+1} = F\!\big(x^{s}_k,\, u^{s}_k\big), \qquad x = (\text{SOC}, \text{SOH}, T), \quad u = (P_{\text{chg}}, P_{\text{dis}}, P_{\text{reg}})
$$

where $F$ is one explicit RK4 step at $\Delta t = 3600\text{ s}$.

### Objective (expected cost across scenarios)

$$
\min \quad \sum_{s=1}^{S} \pi_s \, J^{s}
$$

with the per-scenario cost

$$
\begin{aligned}
J^{s} \;=\; -\sum_{k=0}^{N-1} \Big[ \,
  &\underbrace{p^e_{s,k}\,(P^{s}_{\text{dis},k} - P^{s}_{\text{chg},k})\,\Delta t_h}_{\text{energy revenue}}
  \;+\; \underbrace{p^r_{s,k}\,P^{s}_{\text{reg},k}\,\Delta t_h}_{\text{capacity revenue}} \\
  &-\; \underbrace{c_{\text{deg}}\,\Delta t_s\,\big( \alpha_{\text{deg}}(P^{s}_{\text{chg},k} + P^{s}_{\text{dis},k}) + \alpha_{\text{reg}} P^{s}_{\text{reg},k} \big)}_{\text{degradation cost}}
\Big] \\[0.6em]
&+\; Q^{\text{soc}}_{\text{term}} (\text{SOC}^{s}_N - \text{SOC}_{\text{term}})^2
\;+\; Q^{\text{soh}}_{\text{term}} (\text{SOH}^{s}_N - \text{SOH}_0)^2 \\[0.4em]
&+\; \lambda_{\text{soc}} \sum_{k=0}^{N} (\varepsilon^{\text{soc},s}_k)^2
\;+\; \lambda_{\text{end}} \sum_{k=0}^{N-1} (\varepsilon^{\text{end},s}_k)^2
\end{aligned}
$$

### Constraints (per scenario $s$)

**Initial conditions:**

$$
\text{SOC}^s_0 = \text{SOC}_0, \quad \text{SOH}^s_0 = \text{SOH}_0, \quad T^s_0 = T_0
$$

**Dynamics:**

$$
x^{s}_{k+1} = F\!\big(x^{s}_k, \, u^{s}_k\big), \quad k = 0, \dots, N-1
$$

**SOC bounds (soft):**

$$
\text{SOC}_{\min} - \varepsilon^{\text{soc},s}_k \;\leq\; \text{SOC}^{s}_k \;\leq\; \text{SOC}_{\max} + \varepsilon^{\text{soc},s}_k, \quad k = 0, \dots, N
$$

**Power budget:**

$$
P^{s}_{\text{chg},k} + P^{s}_{\text{reg},k} \;\leq\; P_{\max}, \quad
P^{s}_{\text{dis},k} + P^{s}_{\text{reg},k} \;\leq\; P_{\max}
$$

**Endurance (soft):**

$$
\text{SOC}^{s}_{k+1} + \frac{T_{\text{end}} \, \eta_c}{E_{\text{nom}}} P^{s}_{\text{reg},k} \;\leq\; \text{SOC}_{\max} + \varepsilon^{\text{end},s}_k
$$

$$
\text{SOC}^{s}_{k+1} - \frac{T_{\text{end}}}{E_{\text{nom}} \, \eta_d} P^{s}_{\text{reg},k} \;\geq\; \text{SOC}_{\min} - \varepsilon^{\text{end},s}_k
$$

### Non-anticipativity (cross-scenario coupling)

The first-stage decisions — i.e. the action that gets executed *now*,
**before** the realised scenario is known — must agree across all
scenarios:

$$
P^{s}_{\text{chg},0} = P^{1}_{\text{chg},0}, \quad
P^{s}_{\text{dis},0} = P^{1}_{\text{dis},0}, \quad
P^{s}_{\text{reg},0} = P^{1}_{\text{reg},0}, \quad \forall s = 2, \dots, S
$$

This is what makes it a true two-stage stochastic program rather than
$S$ independent deterministic problems averaged after the fact. Without
non-anticipativity, the planner would "see" the realised scenario from
$k=0$ and the comparison vs `deterministic_lp` would be unfair.

### What this NLP does NOT model

- **V_rc transient dynamics** (charge-transfer + diffusion modes). Time constants $\leq 400\text{ s} \ll \Delta t_{\text{ems}} = 3600\text{ s}$, so they decay within one EMS step.
- **Multi-cell pack effects.** Plans against pack-mean SOC; cell-level imbalance is left to the plant's balancer.
- **Sub-hour activation realisation.** Uses only the expected magnitude $\bar{a} = 0.04$ in the SOC drain term — does not condition on the actual activation sample.
- **Closed-loop feedback inside the hour.** The hourly setpoint is held open-loop until the next EMS solve.
