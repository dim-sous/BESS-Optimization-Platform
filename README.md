# Energy Storage Optimization Platform

A production-grade Python platform for **optimal scheduling, real-time dispatch, and state estimation** of grid-connected battery energy storage systems (BESS). Optimizes **energy arbitrage** and **regulation capacity bidding** while actively managing battery degradation.

Built with the same hierarchical control architecture deployed in commercial utility-scale installations. Evolves through **incremental, gated upgrades** from baseline to industry-grade digital twin.

## What It Solves

Every BESS operator faces the same question: *Given uncertain market prices, how should I charge, discharge, and bid regulation capacity over the next 24 hours -- and how do I execute that plan in real time while protecting the battery?*

This platform answers it end-to-end:

- **Economic scheduling** -- stochastic 24-hour optimization across multiple price scenarios, jointly optimizing arbitrage revenue, regulation capacity payments, and degradation costs
- **Real-time dispatch** -- nonlinear model predictive control tracking the economic schedule while enforcing SOC, thermal, and voltage constraints
- **State estimation** -- EKF reconstructing battery internals (SOC, SOH, temperature, RC voltages) from noisy sensors
- **Multi-cell pack modeling** -- per-cell parameter variation with active balancing, weakest-link SOH tracking
- **High-fidelity plant** -- 2RC equivalent circuit with NMC OCV polynomial, Arrhenius degradation, and thermal dynamics

**Current scope:** The platform optimizes regulation **capacity commitment** (how much to bid) and enforces feasibility constraints. Real-time regulation **delivery** (following stochastic activation signals from the grid) is the target of v5.

## Architecture

```
 Stochastic Prices ──► EMS (hourly)  ──► MPC  ──► Battery Plant
                       24h horizon       warm-started   multi-cell pack
                       N scenarios       constrained    2RC circuit
                                         ◄── EKF ◄──   noisy measurements
```

## Versioned Upgrades

Each version adds one major capability, passes a **4-stage gate** (validation, evaluation, comparison, stress testing), and is frozen before the next begins. See each version's `README.md` for mathematical formulations, metrics, and implementation details.

| Version | What It Adds | Status |
|---------|-------------|--------|
| **v1** Baseline | EMS + MPC + EKF, energy arbitrage + regulation capacity | Frozen |
| **v2** Thermal Model | Temperature state, Arrhenius degradation coupling | Frozen |
| **v3** Pack Model | 4-cell pack, active cell balancing | Frozen |
| **v4** Electrical RC | 2RC equivalent circuit, NMC OCV, voltage measurement | Frozen |

## Quick Start

```bash
uv sync                                       # install dependencies
uv run python v4_electrical_rc_model/main.py   # run latest version
```

Each version is independently runnable. Results go to `results/`.

## Technical Stack

CasADi + IPOPT for nonlinear optimization, NumPy for numerics, Matplotlib for visualization. All optimization models use automatic differentiation and warm-started interior-point solving.

## Roadmap

| Upcoming | Description |
|----------|------------|
| v5 | Regulation activation signals & MPC necessity demonstration |
| v6 | Unscented Kalman Filter (UKF) |
| v7 | Online parameter estimation |
| v8 | Real-time NMPC with ACADOS |
| v9 | Degradation-aware MPC |
| v10-v14 | Uncertainty, delays, multi-battery, inverter, market bidding |

---

*Each version contains its own `README.md` with full mathematical formulations and implementation details.*
