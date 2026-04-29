# BESS Optimization Platform

A Python research platform for the optimal scheduling, real-time dispatch, state estimation, and market participation of grid-connected battery energy storage systems. Built incrementally — each capability is a self-contained upgrade gated behind validation, evaluation, comparison and stress tests before being frozen in [archive/](archive/).

The codebase is a **study of the BESS control & optimization problem space**, not a deployed system. Real prices come from German EPEX SPOT + SMARD Q1 2024; a synthetic 1-day dataset drives unit tests. Limitations are listed [below](#limitations).

## What's modelled

**Plant (5-state, 4 s integration)**
- SOC, SOH, temperature, two RC voltages
- 2RC equivalent circuit with NMC OCV polynomial (fitted to single-cell data); total DC resistance and τ pairs at typical Li-ion ranges
- Lumped thermal model with Arrhenius-coupled SOH degradation
- Multi-cell pack (default 4 cells in series) with manufacturing variation and active balancing

**State estimation (1 min)**
- Extended Kalman Filter consuming noisy SOC, temperature and terminal-voltage measurements; CasADi-based Jacobians

**Planning (1 hour)**
- Greedy price-sort heuristic (`rule_based`)
- Deterministic LP via `scipy.optimize.linprog` (HiGHS) with linear SOC dynamics, FCR endurance soft constraint, and L1 terminal anchor
- Stochastic two-stage NLP with non-anticipativity, 3-state ODE (SOC, SOH, T) and Arrhenius-coupled degradation; CasADi `Opti` + IPOPT
- Multi-product MILP bidding for day-ahead, intraday and balancing markets, wash-trade-free by binary mutex; PuLP + HiGHS

**Control (1 minute)**
- Economic NLP-MPC with terminal-SOC anchor, rate-of-change penalty and soft SOC/temperature bounds; CasADi `Opti` + IPOPT
- Open-loop dispatcher (no MPC) for the EMS-only baseline

**Markets**
- Energy arbitrage with FCR capacity commitment and activation delivery (PJM-RegD-style 4 s activation generator from an OU frequency model)
- Multi-product bidding tier with `ClearingEngine` Protocol (stub `ReferencePriceClearingStub` provided), per-product awards, and dual-pricing imbalance settlement

## Architecture

```
prices ──► Planner (1 h) ──► MPC (1 min) ──► Plant (4 s, 5-state)
              │                  ▲                   │
              │                  │                   │ V_term, SOC, T
              │                  └─── EKF (1 min) ◄──┘
              │
              └─►(optional) Bidding tier:
                     BidBook → ClearingEngine → Awards
                     │
                     └─► greek_settlement block in result dict

```

The simulator's main loop has zero strategy-specific branches — strategy = `(planner, mpc, bidding_protocol)` recipe, dispatched via duck typing. Activation tracking lives inside the plant. The accounting ledger is a pure function over the trace.

## Strategies

Each strategy is one folder in [strategies/](strategies/) with its own README documenting the OCP formulation, what it models and what it doesn't.

| Strategy | Planner | MPC | Bidding tier |
|---|---|---|---|
| [`rule_based`](strategies/rule_based/README.md) | Greedy price-sort heuristic | none | no |
| [`deterministic_lp`](strategies/deterministic_lp/README.md) | LP, scenario-mean (HiGHS) | none | no |
| [`ems`](strategies/ems/README.md) | Stochastic two-stage NLP (IPOPT) | none | no |
| [`ems_economic_mpc`](strategies/ems_economic_mpc/README.md) | Stochastic NLP | Economic NLP-MPC | no |
| [`greek_milp_bidding`](strategies/greek_milp_bidding/README.md) | Multi-product MILP (HiGHS) | Economic NLP-MPC | yes |

## Tests & gating

```bash
uv run pytest                # ~70 tests, ~3 minutes wall
```

Coverage spans the simulator pipeline shape, MILP formulation invariants (wash-trade-free, MBQ enforcement, SOC dynamics consistency, terminal anchor, objective reconstruction), market clearing rules, dual-pricing imbalance truth table, end-to-end strategy runs, and adversarial stress scenarios (random adversarial price profiles, MBQ at 50 kW, LP-relaxation fallback under 1 ms time budget, 3× scenario-fan volatility, 2× activation rate, extreme initial-vs-terminal SOC).

Each version completes a 4-stage gate (validate → evaluate → compare → stress) before being frozen. The most recent gate report is [results/v5b_gate_report.md](results/v5b_gate_report.md).

## Versions

Frozen versions (v1–v4) live in [archive/](archive/) and are not modified. v5 is the active platform with the linear simulator core and modular strategy layout; v5b is an additive multi-product MILP bidding layer.

| Version | Adds |
|---|---|
| v1 | Baseline EMS + MPC + EKF (single-state SOC) |
| v2 | Thermal state, Arrhenius degradation coupling |
| v3 | 4-cell pack with active balancing |
| v4 | 2RC equivalent circuit + voltage measurement, EKF expanded to 5 states |
| v5 | FCR activation moved into the plant; refactor into linear simulator core + per-strategy folders |
| v5b | Multi-product MILP bidding tier (DAM / intraday / balancing) with stub clearing engine and dual-pricing imbalance settlement |

## Limitations

The platform is research code. Specifically:

- **Data pipeline** — real prices are German EPEX SPOT + SMARD Q1 2024 only. No live feeds, no other regions or timeframes.
- **Synthetic clearing engine** — the `ClearingEngine` Protocol is the abstraction boundary; no real exchange or TSO connector is implemented. The stub clears bids against a per-product reference price array.
- **Single-day horizon for the bidding tier** — multi-day rolling horizons and intra-day re-bidding are deferred. One DAM-style gate-closure per run.
- **Symmetric balancing capacity** — single capacity variable per balancing product; UP/DOWN asymmetry, block bids, minimum-income conditions and most jurisdiction-specific market rules are not modelled.
- **Probability-weighted-mean MILP** — the bidding tier uses scenario-weighted means rather than a true two-stage scenario MILP, matching the `deterministic_lp` baseline for fair comparison.
- **MPC is not real-time-grade** — CasADi + IPOPT, ~150 ms per solve. ACADOS / RTI is on the roadmap, not in.
- **Empirical caveats** — the per-minute economic MPC does not earn its compute cost on day-ahead-only data. The relevant diagnosis lives in [strategies/ems_economic_mpc/README.md](strategies/ems_economic_mpc/README.md) under "Empirical status". Toy synthetic-day numbers (a few dollars per day on a 100 kW pack) are not representative of any real BESS economics.
- **Activation-fraction calibration** — α_mfrr and α_aFRR are hand-set defaults, not derived from balancing-area historicals.

## Quick start

```bash
uv sync                                                            # install
uv run python comparison/run_v5_comparison.py --days 1             # 1-day strategy comparison
uv run python -m core.planners.milp_bidding --solve --plot         # MILP-only smoke + plots
uv run python comparison/run_phase4_gate.py --hours 4              # full 4-stage gate
uv run pytest                                                      # tests
```

`core/` holds shared modules; `strategies/<name>/` holds one folder per strategy recipe with its own README and tests; `comparison/` holds the multi-strategy runners; `archive/` holds frozen v1–v4; `results/` is generated output.

## Tech stack

CasADi + IPOPT for nonlinear OCP and EKF Jacobians, PuLP + HiGHS for MILP / LP, SciPy + NumPy for numerics, Matplotlib for plots, pytest for tests, uv for environment management.

## Roadmap (intent, not commitment)

| | |
|---|---|
| v6 | Unscented Kalman Filter |
| v7 | Joint state + parameter estimation |
| v8 | ACADOS NMPC (RTI, control blocking) |
| v9 | Degradation-aware MPC (SOH in MPC state) |
| v10 | Stochastic forecast uncertainty in dispatch |
| v11 | Measurement and communication delays |
| v12 | Multi-battery system with central EMS coordinator |
| v13 | Grid-connected inverter dynamics (id, iq, V_dc) |
