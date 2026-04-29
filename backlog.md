# Backlog — Active Issues & Future Work

Working tree (and git history) is the only authoritative source of truth. If
something here disagrees with the source code, the source code wins.

## Active platform

The current platform layout (post-v5 refactor):

- `core/` — shared modules (physics, estimator, planners, mpc, accounting, simulator, markets, visualization).
- `strategies/<name>/` — one folder per strategy recipe (planner + optional MPC + optional bidding protocol + README).
- `comparison/` — multi-strategy runners.
- `archive/` — frozen v1–v4.
- `results/` — generated output (gate reports, traces, plots).

The simulator's main loop ([core/simulator/core.py](core/simulator/core.py)) has zero strategy-specific branches; strategies are dispatched via duck typing.

### Strategy ladder

| # | Strategy | Role |
|---|---|---|
| 1 | `rule_based` | Naive baseline (lower bound) |
| 2 | `deterministic_lp` | Commercial baseline (LP, mean substitution) |
| 3 | `ems` | Canonical "EMS alone" (stochastic two-stage NLP) |
| 4 | `ems_economic_mpc` | EMS + per-minute economic NLP-MPC (production v5) |
| 5 | `greek_milp_bidding` | Multi-product MILP bidding tier + Economic MPC (additive v5b extension) |

The canonical pitch comparison is `ems_economic_mpc` vs `ems`. `greek_milp_bidding` is opt-in (`pitch_visible=False`) and is not in `comparison/run_v5_comparison.py`'s `STRATEGY_FACTORIES`; it has its own gate runner at [comparison/run_phase4_gate.py](comparison/run_phase4_gate.py).

### Architectural facts (verifiable in source)

- Activation tracking is performed inside `BatteryPlant.step()` — see [core/physics/plant.py](core/physics/plant.py). The strategy layer outputs `(setpoint_pnet, p_reg_committed)` only.
- There is no strategy-layer PI controller. `core/pi/` does not exist.
- The plant clips dispatch in two passes (base setpoint, then setpoint + activation), so `p_delivered` is correctly attributed to the FCR portion only.
- The bidding tier (v5b) is an additive opt-in: `Strategy.bidding_protocol` defaults to `None`. When it is `None`, the simulator's traces are bit-identical to the pre-v5b baseline. Verified by [tests/test_greek_milp_bidding_e2e.py::test_v5_strategy_not_affected_by_phase3](tests/test_greek_milp_bidding_e2e.py).

### Open empirical questions

1. The per-minute economic MPC layer does not earn its compute cost on day-ahead-only data — see the "Empirical status" section of [strategies/ems_economic_mpc/README.md](strategies/ems_economic_mpc/README.md). Whether intraday or near-real-time data closes that gap is unverified; would require a different data pipeline.
2. The MILP bidding tier (v5b) on the canonical synthetic day reports `total_greek_revenue ≈ $92.79` (single-DAM-gate model, perfect plant tracking). On real Greek market data the numerical results would differ; calibration of activation fractions and product price decompositions has not been performed against historicals.

## Future work (intent, not commitments)

In rough order of value:

- **v6** — Unscented Kalman Filter (replace EKF; better for the OCV nonlinearity)
- **v7** — Joint state and parameter estimation (online R_internal, capacity, efficiency)
- **v8** — ACADOS NMPC (replace CasADi/IPOPT online: RTI, control blocking, real-time-grade)
- **v9** — Degradation-aware MPC (SOH in MPC state, profit-vs-degradation tradeoff)
- **v10** — Disturbance forecast uncertainty (scenario-based MPC, chance constraints)
- **v11** — Measurement and communication delays
- **v12** — Multi-battery system with central EMS coordinator
- **v13** — Grid-connected inverter model (id, iq, V_dc dynamics)

Possible v5b follow-ups (additive to the bidding tier):

- Real exchange / TSO connector implementing the `ClearingEngine` Protocol (replaces the `ReferencePriceClearingStub`)
- Two-stage scenario MILP for bid + dispatch with non-anticipativity
- Asymmetric UP/DOWN balancing capacity (split each product into two variables)
- Block-bid contiguity constraints (binary infrastructure already in place; add the contiguity inequalities)
- Multi-session intraday gate-closures (currently a single DAM-style gate per run)
- Empirical calibration of activation fractions and product price decompositions against balancing-area historicals
