## Current State
- **Frozen versions:** v1, v2, v3, v4 live in [archive/](archive/) and **must not be modified**.
- **Active platform:** v5 (Regulation Activation & MPC Necessity) with v5b additive extension (Greek-style multi-product MILP bidding tier). Linear simulator core; per-strategy folders; pure-function ledger.
- **Active issues / open empirical questions:** see [backlog.md](backlog.md).
- **Historical gate reports:** [archive/gate_reports.md](archive/gate_reports.md). Most recent gate report: [results/v5b_gate_report.md](results/v5b_gate_report.md).

## If context is unclear
Re-read this file top to bottom. Ask the user to confirm the current state.

You are assisting in the development of a research-grade battery digital twin, control, and optimization platform in Python.

────────────────────────────────────────
BEHAVIORAL PRINCIPLES
────────────────────────────────────────
1. **PROPOSE BEFORE IMPLEMENTING.** Before writing any code for a new upgrade, output a short proposal (what / why / risks / dependencies). For non-trivial changes, wait for confirmation. If you see a better approach or a meaningful tradeoff, surface it first.
2. **ONE UPGRADE AT A TIME.** Each upgrade is a self-contained step. Never bundle multiple upgrades. Each step must be independently runnable and reversible.
3. **FOUR-STAGE GATE — mandatory before moving to the next upgrade:**
   A. Validation — confirm the implementation is physically and mathematically consistent
   B. Evaluation — compute and log the standard metrics (see below)
   C. Comparison — generate plots comparing this version to the previous one
   D. Stress Test — test the implementation under extreme conditions
4. **PRODUCTION-GRADE CODE.** Modular architecture, type hints, docstrings, configuration files, logging, exception handling. Explicit physical units everywhere. Clarity over cleverness.
5. **AVOID OVER-ENGINEERING.** Smallest implementation that meaningfully advances the system. Flag complexity/compute tradeoffs and suggest lighter alternatives.
6. **COMPUTATIONAL AWARENESS.** For each upgrade, note the effect on simulation and solver time. Flag anything that risks making real-time operation infeasible.
7. **PLAN BEFORE IMPLEMENTING.** For any non-trivial change (NLP reformulation, architecture change, new feature), enter plan mode first. Use `EnterPlanMode` to explore the codebase, design the approach, and get user approval before writing code. Never silently start implementing.
8. **STICK WITH REALISM.** Model mismatch is real and the controllers must plan around it. Don't paper over physical inconsistencies — surface them and decide explicitly.

────────────────────────────────────────
REPOSITORY LAYOUT
────────────────────────────────────────
```
bess/
├── README.md
├── CLAUDE.md                  ← this file
├── backlog.md                 ← active issues + future work
├── pyproject.toml
├── archive/                   ← v1–v4 frozen, gate_reports.md
├── core/                      ← shared platform modules
│   ├── config/                ←   parameter dataclasses
│   ├── physics/               ←   plant, ODE, OCV
│   ├── markets/               ←   prices, activation, products, bids, clearing, imbalance
│   ├── estimators/            ←   EKF
│   ├── planners/              ←   rule_based, deterministic_lp, stochastic_ems, milp_bidding
│   ├── mpc/                   ←   tracking, economic
│   ├── accounting/            ←   pure-function ledger + Greek settlement
│   ├── simulator/             ←   strategy spec, traces, linear core loop, bidding_protocol
│   └── visualization/
├── strategies/                ← one folder per strategy (recipe + README)
│   ├── rule_based/            ← naive baseline
│   ├── deterministic_lp/      ← commercial baseline (LP, mean-substitution)
│   ├── ems/                   ← canonical "EMS alone" (stochastic NLP, no MPC)
│   ├── ems_economic_mpc/      ← EMS + Economic MPC (production v5 strategy)
│   └── greek_milp_bidding/    ← v5b additive: multi-product MILP bidding + Economic MPC
├── comparison/                ← strategy comparison harnesses
└── results/
```

────────────────────────────────────────
STANDARD METRICS — compute and store after every version
────────────────────────────────────────
- Control: RMSE_SOC_tracking, RMSE_power_tracking
- Estimation: RMSE_SOC_estimation, RMSE_SOH_estimation
- Economic: total_profit, total_degradation_cost (plus per-product Greek revenue when v5b bidding tier is active)
- Computational: avg_mpc_solve_time, max_mpc_solve_time, estimator_solve_time, mip_solve_time
- Generate comparison plots for: SOC, SOH, temperature, voltage, power, profit, solver time.

────────────────────────────────────────
UPGRADE BACKLOG
────────────────────────────────────────
Frozen (in archive/): v1 baseline · v2 thermal · v3 multi-cell pack · v4 2RC electrical
Done: v5 Regulation Activation & MPC Necessity · v5b Multi-product MILP bidding tier
Future: v6 UKF · v7 joint state/param estimation · v8 ACADOS NMPC · v9 degradation-aware MPC · v10 stochastic forecast uncertainty · v11 delays · v12 multi-battery · v13 grid inverter

────────────────────────────────────────
GOAL
────────────────────────────────────────
A well-tested, physically realistic, maintainable platform that improves measurably at each step — not one that is maximally complex.
