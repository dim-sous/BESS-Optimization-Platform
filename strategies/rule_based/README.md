# rule_based — heuristic dispatch (NOT an OCP)

**Pitch-visible:** yes (strict lower bound)
**Composition:** `RuleBasedPlanner` (no MPC)

## What it does (plain words)

Sorts the day's forecast-mean energy prices, charges during the cheapest
hours, and discharges during the most expensive. No FCR commitment. No
optimization problem is solved at all — this is a fixed decision rule
applied to the price array. Strict lower bound for the comparison
harness; every other strategy should beat it by a wide margin.

## Decision rule (NOT an OCP)

This strategy does not solve an optimization problem. It executes a
closed-form heuristic over the forecast prices.

### Notation

| Symbol | Meaning | Units |
|---|---|---|
| $N$ | planning horizon | hours (= 24) |
| $\bar{p}^e \in \mathbb{R}^N$ | probability-weighted mean of forecast energy prices | \$/kWh |
| $E_{\text{nom}}$ | nominal energy capacity | kWh |
| $\text{SOC}_{\min}, \text{SOC}_{\max}$ | SOC bounds | — |
| $P_{\max}$ | max charge / discharge power | kW |

### Algorithm

1. **Forecast collapse:** $\bar{p}^e_k = \sum_s \pi_s \, p^e_{s,k}$ for each hour $k$.

2. **Compute usable energy and dispatch power:**

$$
E_{\text{use}} = (\text{SOC}_{\max} - \text{SOC}_{\min}) \, E_{\text{nom}}, \qquad P = 0.8 \, P_{\max}
$$

3. **Number of active hours per direction:**

$$
n^* = \min\!\Big( \big\lceil E_{\text{use}} / P \big\rceil, \, \big\lfloor N/3 \big\rfloor \Big)
$$

4. **Sort hours by ascending forecast price:** $\sigma = \operatorname{argsort}(\bar{p}^e)$.

5. **Pick the $n^*$ cheapest hours as charge hours, the $n^*$ most expensive as discharge hours:**

$$
\mathcal{C} = \{\sigma_0, \dots, \sigma_{n^*-1}\}, \qquad \mathcal{D} = \{\sigma_{N-n^*}, \dots, \sigma_{N-1}\}
$$

6. **Profitability gate** — only commit if there is a positive spread:

$$
\bar{p}^e_{\max(\mathcal{D})} > \bar{p}^e_{\max(\mathcal{C})}
$$

7. **Output schedule:**

$$
P_{\text{chg},k} = \begin{cases} P & \text{if gate true and } k \in \mathcal{C} \\ 0 & \text{otherwise} \end{cases}, \quad
P_{\text{dis},k} = \begin{cases} P & \text{if gate true and } k \in \mathcal{D} \\ 0 & \text{otherwise} \end{cases}
$$

$$
P_{\text{reg},k} = 0 \quad \forall k
$$

### What this does NOT model

- **No optimization.** Greedy price-sorting only.
- **No state evolution.** SOC is not tracked by the rule itself.
- **No FCR participation.** $P_{\text{reg}} \equiv 0$.
- **No constraints.** SOC bounds are honored only by the magic factor `0.8` and the `n_hours_needed` heuristic; the plant clips anything that would violate physics.
- **No closed-loop feedback.** Single hourly setpoint, dispatched open-loop.
