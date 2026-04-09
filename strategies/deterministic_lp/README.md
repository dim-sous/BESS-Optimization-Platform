# deterministic_lp — Linear Program (commercial baseline)

**Pitch-visible:** yes
**Composition:** `DeterministicLP` planner (no MPC)
**Solver:** `scipy.optimize.linprog` with the HiGHS backend

## What it does (plain words)

Collapses the forecast scenarios to a single deterministic mean and
solves a 24-hour linear program for the hourly arbitrage and FCR
commitment plan, accounting for power-budget headroom, FCR endurance,
degradation cost, and a soft terminal SOC anchor. Mean substitution =
no scenario hedging. The honest commercial baseline: every commercial
BESS EMS today ships some form of this.

## Optimal Control Problem

### Notation

| Symbol | Meaning | Units |
|---|---|---|
| $N$ | planning horizon | hours (= 24) |
| $\Delta t_h$ | step size in hours | h (= 1) |
| $\Delta t_s$ | step size in seconds | s (= 3600) |
| $k$ | time index, $k = 0, \dots, N-1$ | — |
| $\bar{p}^e_k = \sum_s \pi_s p^e_{s,k}$ | forecast-mean energy price | \$/kWh |
| $\bar{p}^r_k = \sum_s \pi_s p^r_{s,k}$ | forecast-mean reg-capacity price | \$/kW/h |
| $E_{\text{nom}}$ | nominal pack capacity | kWh |
| $\eta_c, \eta_d$ | charge / discharge efficiency | — |
| $P_{\max}$ | max power | kW |
| $\text{SOC}_{\min}, \text{SOC}_{\max}$ | SOC bounds | — |
| $\text{SOC}_0$ | initial SOC (from EKF) | — |
| $\text{SOC}_{\text{term}}$ | terminal SOC target | — |
| $\alpha_{\text{deg}}$ | arbitrage-throughput degradation rate | $1/(\text{kW}\cdot\text{s})$ |
| $\alpha_{\text{reg}}$ | reg-cycling degradation rate | $1/(\text{kW}\cdot\text{s})$ |
| $c_{\text{deg}}$ | degradation cost | \$/SOH lost |
| $T_{\text{end}}$ | endurance horizon | h (= 0.5) |
| $W_{\text{term}}$ | terminal-anchor L1 weight | \$/SOC-unit (= 500) |

### Decision variables

$$
\begin{aligned}
P_{\text{chg},k} &\in [0, P_{\max}], &\quad k &= 0, \dots, N-1 \\
P_{\text{dis},k} &\in [0, P_{\max}], &\quad k &= 0, \dots, N-1 \\
P_{\text{reg},k} &\in [0, P_{\max}], &\quad k &= 0, \dots, N-1 \\
z^+, \, z^- &\geq 0 & &\text{(L1 slacks for the terminal anchor)}
\end{aligned}
$$

The state $\text{SOC}_k$ is **not** a free variable; it is reconstructed
from the inputs via the linear recursion below.

### State recursion (linear, embedded in constraints)

$$
\text{SOC}_k = \text{SOC}_0 + \frac{\Delta t_h}{E_{\text{nom}}} \sum_{j=0}^{k-1} \Big[ \eta_c \, P_{\text{chg},j} - \frac{P_{\text{dis},j}}{\eta_d} \Big]
$$

### Objective

The LP minimizes the negation of expected profit:

$$
\min_{P_{\text{chg}}, P_{\text{dis}}, P_{\text{reg}}, z^+, z^-} \quad
\sum_{k=0}^{N-1} \Big[
  \underbrace{\bar{p}^e_k P_{\text{chg},k} \Delta t_h}_{\text{charge cost}}
  - \underbrace{\bar{p}^e_k P_{\text{dis},k} \Delta t_h}_{\text{discharge revenue}}
  - \underbrace{\bar{p}^r_k P_{\text{reg},k} \Delta t_h}_{\text{capacity revenue}}
  + \underbrace{c_{\text{deg}} \Delta t_s \big( \alpha_{\text{deg}}(P_{\text{chg},k} + P_{\text{dis},k}) + \alpha_{\text{reg}} P_{\text{reg},k} \big)}_{\text{degradation cost}}
\Big]
+ W_{\text{term}}(z^+ + z^-)
$$

All terms are **linear** in the decision variables — that is what makes
this an LP and not an NLP. Note in particular that the SOC dynamics are
linear in $(P_{\text{chg}}, P_{\text{dis}})$ (no nonlinear OCV-coupled
current solve), and the terminal anchor uses an $L_1$ rather than $L_2$
penalty so it can be encoded with two non-negative slack variables.

### Constraints

**Power budget** (committed reg power must leave headroom for the
planned chg/dis dispatch):

$$
P_{\text{chg},k} + P_{\text{reg},k} \leq P_{\max}, \quad
P_{\text{dis},k} + P_{\text{reg},k} \leq P_{\max}, \quad k = 0, \dots, N-1
$$

**SOC bounds with FCR endurance margin** (the most-recently-committed
reg power $P_{\text{reg},k-1}$ must be sustainable for $T_{\text{end}}$
hours in either direction without leaving the SOC envelope):

$$
\text{SOC}_k + \frac{T_{\text{end}} \, \eta_c}{E_{\text{nom}}} P_{\text{reg},k-1} \;\leq\; \text{SOC}_{\max}, \quad k = 1, \dots, N
$$

$$
\text{SOC}_k - \frac{T_{\text{end}}}{E_{\text{nom}} \, \eta_d} P_{\text{reg},k-1} \;\geq\; \text{SOC}_{\min}, \quad k = 1, \dots, N
$$

**Terminal SOC anchor** (encoded as a linear equality with non-negative
slacks; the objective penalty $W_{\text{term}}(z^+ + z^-)$ then
implements an $L_1$ penalty $W_{\text{term}} |\text{SOC}_N - \text{SOC}_{\text{term}}|$):

$$
\text{SOC}_N - \text{SOC}_{\text{term}} = z^+ - z^-
$$

### What this LP does NOT model

- **Thermal dynamics.** No temperature state and no thermal constraint. The plant runs hot whenever the LP plans aggressive dispatch; the ledger does not bill thermal violations.
- **OCV nonlinearity.** SOC dynamics are linear in $(P_{\text{chg}}, P_{\text{dis}})$; no voltage-coupled current solve.
- **V_rc transient dynamics.** No charge-transfer or diffusion modes.
- **Multi-cell pack effects.** Plans against pack-level energy.
- **Stochastic forecast uncertainty.** Scenarios are collapsed to the mean before solving — the LP ignores variance.
- **Closed-loop state feedback within the hour.** The LP solves once per hour against the most recent EKF estimate; within the hour the held setpoint is dispatched open-loop.
