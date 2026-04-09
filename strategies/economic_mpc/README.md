# economic_mpc — Economic NLP MPC (per minute)

**Pitch-visible:** yes (the production strategy)
**Composition:** `EconomicEMS` planner (hourly, see `strategies/ems_clamps`) + `EconomicMPC` (per-minute closed-loop)
**Solver:** CasADi `Opti` + IPOPT (MUMPS linear solver)

## What it does (plain words)

Re-solves a 60-minute deterministic NLP every minute against the live
EKF state estimate. Plans charge/discharge under a soft anchor on the
EMS strategic SOC plan plus an economic term on energy arbitrage. The
committed FCR power is **exogenous** — set hourly by the EMS, treated
as a parameter (not a decision variable) inside this layer. Activation
tracking lives in the plant; the MPC does not see the FCR signal
directly.

## Optimal Control Problem

This is a **deterministic NLP solved at 60 s cadence** with a 1-hour
horizon and control-horizon blocking after $N_c = 20$ steps.

### Notation

| Symbol | Meaning | Units |
|---|---|---|
| $N$ | prediction horizon | (= 60 MPC steps = 60 min) |
| $N_c$ | control horizon (decisions are blocked beyond $N_c$) | (= 20 MPC steps = 20 min) |
| $\Delta t_h$ | step size in hours | h (= $1/60$) |
| $\Delta t_s$ | step size in seconds | s (= 60) |
| $k$ | prediction step index, $k = 0, \dots, N-1$ | — |
| $j(k) = \min(k, N_c - 1)$ | control-blocking index | — |
| $\text{soc}_0, \text{temp}_0$ | initial state from EKF (set per solve) | — |
| $\overline{\text{soh}}$ | EKF SOH estimate, **frozen** as a parameter | — |
| $\hat{p}^e_k$ | forecast-mean energy price for step $k$ | \$/kWh |
| $\bar{P}_{\text{reg},k}$ | committed FCR power, ZOH-expanded from EMS plan | kW |
| $\text{soc}^{\text{ref}}_k$ | EMS SOC reference (current implementation: end-of-hour value, repeated) | — |
| $u^{\text{prev}} = (P^{\text{prev}}_{\text{chg}}, P^{\text{prev}}_{\text{dis}})$ | last applied controls | kW |

| Symbol | Weight | Default value |
|---|---|---|
| $w_e$ | energy-arbitrage weight | $1$ |
| $w_{\text{deg}}$ | degradation-cost weight | $1$ |
| $Q^{\text{soc}}_{\text{anchor}}$ | per-step SOC anchor | $10$ |
| $Q^{\text{term}}_{\text{econ}}$ | terminal SOC anchor | $10^3$ |
| $R^{\Delta}_{\text{econ}}$ | rate-of-change penalty | $10^{-2}$ |
| $\lambda_{\text{soc}}$ | SOC slack penalty | $10^6$ |
| $\lambda_{\text{temp}}$ | temperature slack penalty | $10^7$ |

### Decision variables

$$
\begin{aligned}
P_{\text{chg},j}, \, P_{\text{dis},j} &\in [0, P_{\max}], &\quad j &= 0, \dots, N_c - 1 \\
\text{SOC}_k, \, T_k &\in \mathbb{R}, &\quad k &= 0, \dots, N \\
\varepsilon_k &\geq 0, &\quad k &= 0, \dots, N \quad \text{(SOC slack)} \\
\varepsilon^{\text{temp}}_k &\geq 0, &\quad k &= 0, \dots, N \quad \text{(temperature slack)}
\end{aligned}
$$

For $k \geq N_c$, the control is **blocked**: the same
$P_{\text{chg}, N_c-1}$ and $P_{\text{dis}, N_c-1}$ are reused. SOH is
frozen at $\overline{\text{soh}}$, not a state. **$P_{\text{reg}}$ is
not a decision variable** — it is the parameter $\bar{P}_{\text{reg},k}$.

### Prediction model (2-state, frozen SOH)

$$
\begin{aligned}
\frac{d\,\text{SOC}}{dt} &= \frac{\eta_c P_{\text{chg}} - P_{\text{dis}}/\eta_d}{\overline{\text{soh}} \cdot E_{\text{nom}} \cdot 3600} \\[1em]
\frac{dT}{dt} &= \frac{Q_{\text{joule}}(P_{\text{net}}, T) - h_{\text{cool}}(T - T_{\text{amb}})}{C_{\text{thermal}}}
\end{aligned}
$$

where $P_{\text{net}} = P_{\text{dis}} - P_{\text{chg}}$ and the thermal
input $u^{\text{eff}}_k = (P_{\text{chg},j(k)}, P_{\text{dis},j(k)}, \bar{P}_{\text{reg},k})$
includes the committed reg power so the predicted Joule heating
accounts for it. Discrete-time:

$$
(\text{SOC}_{k+1}, T_{k+1}) = F\!\big( (\text{SOC}_k, T_k), \, u^{\text{eff}}_k; \, \overline{\text{soh}} \big)
$$

with $F$ one explicit RK4 step at $\Delta t = 60\text{ s}$.

### Objective

$$
\begin{aligned}
\min \quad &\sum_{k=0}^{N-1} \Big[
  \underbrace{-\,w_e\,\hat{p}^e_k \big( P_{\text{dis},j(k)} - P_{\text{chg},j(k)} \big) \Delta t_h}_{\text{energy revenue (negated)}}
  \;+\; \underbrace{w_{\text{deg}}\,c_{\text{deg}}\,\alpha_{\text{deg}}\,\big( P_{\text{chg},j(k)} + P_{\text{dis},j(k)} \big) \Delta t_s}_{\text{arbitrage degradation}}
  \;+\; \underbrace{Q^{\text{soc}}_{\text{anchor}} \big( \text{SOC}_k - \text{soc}^{\text{ref}}_k \big)^{2}}_{\text{soft EMS SOC anchor}}
\Big] \\[0.6em]
&+\; R^{\Delta}_{\text{econ}} \Big( (P_{\text{chg},0} - P^{\text{prev}}_{\text{chg}})^{2} + (P_{\text{dis},0} - P^{\text{prev}}_{\text{dis}})^{2} \Big) \\[0.2em]
&+\; R^{\Delta}_{\text{econ}} \sum_{k=1}^{N_c - 1} \Big( (P_{\text{chg},k} - P_{\text{chg},k-1})^{2} + (P_{\text{dis},k} - P_{\text{dis},k-1})^{2} \Big) \\[0.4em]
&+\; Q^{\text{term}}_{\text{econ}} \big( \text{SOC}_N - \text{soc}^{\text{ref}}_N \big)^{2} \\[0.2em]
&+\; \lambda_{\text{soc}} \sum_{k=0}^{N} \varepsilon_k^{2}
\;+\; \lambda_{\text{temp}} \sum_{k=0}^{N} (\varepsilon^{\text{temp}}_k)^{2}
\end{aligned}
$$

The reg-cycling degradation term $c_{\text{deg}} \alpha_{\text{reg}} \bar{P}_{\text{reg},k}$
is **constant** with respect to the decision variables (since $\bar{P}_{\text{reg},k}$
is exogenous), so it has zero gradient and is omitted from the
objective. The simulator's ledger bills it post-hoc.

### Constraints

**Initial conditions:**

$$
\text{SOC}_0 = \text{soc}_0, \quad T_0 = \text{temp}_0
$$

**Dynamics:**

$$
(\text{SOC}_{k+1}, T_{k+1}) = F\!\big( (\text{SOC}_k, T_k), \, u^{\text{eff}}_k; \, \overline{\text{soh}} \big), \quad k = 0, \dots, N-1
$$

**SOC bounds (soft):**

$$
\text{SOC}_{\min} - \varepsilon_k \;\leq\; \text{SOC}_k \;\leq\; \text{SOC}_{\max} + \varepsilon_k, \quad k = 0, \dots, N
$$

**Temperature bounds (soft):**

$$
T_{\min} - \varepsilon^{\text{temp}}_k \;\leq\; T_k \;\leq\; T_{\max} + \varepsilon^{\text{temp}}_k, \quad k = 0, \dots, N
$$

**Power budget (full prediction horizon, post F18 fix):**

$$
P_{\text{chg},j(k)} + \bar{P}_{\text{reg},k} \;\leq\; P_{\max}, \quad
P_{\text{dis},j(k)} + \bar{P}_{\text{reg},k} \;\leq\; P_{\max}, \quad k = 0, \dots, N-1
$$

This constraint is enforced over the **full** horizon $k \in [0, N)$,
not just the unblocked control window $j \in [0, N_c)$, so the held
control values cannot propagate a physically infeasible plan into the
blocked region of the prediction.

### What this MPC does NOT have

- **No $P_{\text{reg}}$ decision variable.** Reg power is exogenous (parameter from the EMS).
- **No endurance constraint.** Currently differs from `tracking_mpc` in this respect.
- **No stochasticity.** Single deterministic price horizon (forecast mean), single 2-state trajectory.
- **No SOH state.** Frozen as a parameter; the slow SOH dynamics are modelled by the EMS, not the MPC.
- **No V_rc transient states.**
- **No multi-cell pack model.** Plans against pack-mean SOC.
- **Per-step SOC anchor uses the end-of-hour reference at all $k$** in the current implementation — see audit finding F33 for the structural issue this creates inside an hour.

### Empirical status (post-audit)

The post-audit big experiment (3 subsets × 5 days × 5 strategies × 2
plant configurations) shows that on the current data pipeline (hourly
day-ahead prices, no intraday or real-time signals), the per-minute
economic re-optimization layer **does not produce a positive return**
relative to the EMS-only baseline (`ems_clamps`). Specifically:

- `economic_mpc` loses to `ems_clamps` in 28 of 30 days, by $0.06–0.85$/day.
- The loss is largest in the **volatile** subset, not the stressed subset, suggesting the failure mode is **horizon myopia**: the 60-min MPC cannot see the full-day arbitrage shape the EMS captures.
- The MPC's economic term has **zero intra-hour gradient** because $\hat{p}^e_k$ is constant within an hour (day-ahead resolution).
- Per-minute decisions are therefore driven entirely by the SOC anchor — a tracking signal that does no economic work.

This is a **data-pipeline gap**, not (necessarily) an MPC formulation
bug. The MPC layer would only earn its compute cost if it received
intraday/real-time prices the EMS does not see. See the OCP write-ups
for `ems_clamps` and `deterministic_lp` and the audit
[backlog](../../backlog.md) for the full diagnosis.
