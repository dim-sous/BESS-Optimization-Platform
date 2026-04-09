## Current State
- **Frozen versions:** v1, v2, v3, v4 live in [archive/](archive/) and **must not be modified**.
- **Active development:** v5 (Regulation Activation & MPC Necessity) — refactored into modular `core/` (shared platform) + `strategies/` (one folder per strategy) layout. Linear simulator core; bugs A/B/C from the audit eliminated by construction.
- **Active issues:** see [backlog.md](backlog.md) (item 0 lists the open audit findings the refactor is addressing).
- **Historical gate reports:** [archive/gate_reports.md](archive/gate_reports.md).

## If context is unclear
Re-read this file top to bottom. Ask the user to confirm the current state.

You are assisting in the development of an industry-grade battery digital twin, control, and optimization platform in Python.

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
REPOSITORY LAYOUT (post-refactor)
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
│   ├── markets/               ←   prices, activation, revenue
│   ├── estimators/            ←   EKF, MHE
│   ├── planners/              ←   rule_based, deterministic_lp, stochastic_ems
│   ├── mpc/                   ←   tracking, economic
│   ├── pi/                    ←   regulation controller
│   ├── accounting/            ←   pure-function ledger
│   ├── simulator/             ←   strategy spec, traces, linear core loop
│   └── visualization/
├── strategies/                ← one folder per strategy (recipe + README)
│   ├── rule_based/            ← naive baseline
│   ├── deterministic_lp/      ← commercial baseline (LP, mean-substitution)
│   ├── ems/            ← canonical "EMS alone" (stochastic EMS, no MPC)
│   ├── ems_tracking_mpc/      ← EMS + Tracking MPC (controlled-experiment baseline, not pitch)
│   └── ems_economic_mpc/      ← EMS + Economic MPC (production v5 strategy)
├── comparison/                ← strategy comparison harness
├── presentation/              ← B2B pitch deck generator
└── results/
```

────────────────────────────────────────
STANDARD METRICS — compute and store after every version
────────────────────────────────────────
- Control: RMSE_SOC_tracking, RMSE_power_tracking
- Estimation: RMSE_SOC_estimation, RMSE_SOH_estimation
- Economic: total_profit, total_degradation_cost
- Computational: avg_mpc_solve_time, max_mpc_solve_time, estimator_solve_time
- Generate comparison plots for: SOC, SOH, temperature, voltage, power, profit, solver time.

────────────────────────────────────────
UPGRADE BACKLOG
────────────────────────────────────────
Frozen (in archive/): v1 baseline · v2 thermal · v3 multi-cell pack · v4 2RC electrical
Active: v5 Regulation Activation & MPC Necessity
Future: v6 UKF · v7 joint state/param estimation · v8 ACADOS NMPC · v9 degradation-aware MPC · v10 stochastic forecast uncertainty · v11 delays · v12 multi-battery · v13 grid inverter · v14 market bidding

The v5 refactor (`core/` + `strategies/` modularization) takes precedence over v6 work — the audit found execution-layer bugs that must be fixed by construction, not patched, before adding more capabilities.

────────────────────────────────────────
GOAL
────────────────────────────────────────
A well-tested, physically realistic, maintainable platform that improves measurably at each step — not one that is maximally complex.
