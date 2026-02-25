# Battery Energy Storage Optimisation Platform

A production-grade Python platform for optimal scheduling and real-time control of a Battery Energy Storage System (BESS) performing electricity price arbitrage.

## Architecture

The platform implements a **hierarchical control** architecture — the same pattern used in commercial BESS deployments:

```
┌─────────────────────────────────────────────────────────┐
│                    Price Forecast                        │
│                   (data/prices.csv)                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              EMS Economic Optimiser                      │
│  (optimization/ems_optimizer.py)                         │
│                                                          │
│  • Day-ahead planning horizon (24 h)                     │
│  • Maximises arbitrage profit                            │
│  • Split charge/discharge variables (exact efficiency)   │
│  • Terminal SOC constraint                               │
│  • CasADi Opti + IPOPT                                  │
│                                                          │
│  Output: P_ref(k), SOC_ref(k)                            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│            Tracking MPC Controller                       │
│  (control/mpc_controller.py)                             │
│                                                          │
│  • Short prediction horizon (6 steps)                    │
│  • Quadratic SOC + power tracking cost                   │
│  • Soft SOC constraints (guaranteed feasibility)         │
│  • Warm-started IPOPT solves                             │
│  • Simplified dynamics (feedback corrects mismatch)      │
│                                                          │
│  Output: P_cmd(k)  (applied to plant every step)         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│            Battery Plant Model                           │
│  (models/battery_model.py)                               │
│                                                          │
│  • Asymmetric charge / discharge efficiency              │
│  • SOC-limited power saturation with back-calculation    │
│  • Discrete-time dynamics                                │
└─────────────────────────────────────────────────────────┘
```

**Why two layers?**

| Layer | Horizon | Objective | Model fidelity |
|-------|---------|-----------|----------------|
| EMS   | 24 h    | Maximise profit | Full (split P_chg / P_dis, exact η) |
| MPC   | 6 steps | Track EMS reference | Simplified (single P, unit η) |

The EMS computes the economically optimal schedule once.  The MPC then tracks that schedule in closed-loop, compensating for model mismatch, disturbances, and constraint violations in real time.

## Project Structure

```
battery_optimization_platform/
├── main.py                          # Entry point
├── config.py                        # All tuneable parameters
├── pyproject.toml                   # uv project config & dependencies
├── data/
│   └── prices.csv                   # Electricity price data
├── models/
│   └── battery_model.py             # High-fidelity plant model
├── optimization/
│   └── ems_optimizer.py             # Day-ahead economic optimiser
├── control/
│   └── mpc_controller.py            # Real-time tracking MPC
├── simulation/
│   └── simulate.py                  # Closed-loop simulation loop
└── visualization/
    └── plot_results.py              # Four-panel result figure
```

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Install & Run

```bash
cd battery_optimization_platform
uv sync
uv run python main.py
```

### Expected Output

```
EMS expected profit:       $  XX.XX
MPC simulation profit:     $  XX.XX
Tracking gap:              $   X.XX
Final SOC:                 0.500
```

A `results.png` figure is saved to the project root showing:
1. **SOC** — MPC actual vs. EMS reference, with SOC limits shaded
2. **Power** — Dispatch commands (positive = discharge to grid)
3. **Price** — Electricity spot price over the horizon
4. **Profit** — Cumulative arbitrage profit

## Configuration

All parameters live in `config.py` as frozen dataclasses:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `E_max_kwh` | 200 | Battery energy capacity [kWh] |
| `P_max_kw` | 100 | Max charge/discharge power [kW] |
| `SOC_min / SOC_max` | 0.10 / 0.90 | Operational SOC window |
| `eta_charge / eta_discharge` | 0.95 / 0.95 | Round-trip efficiency |
| `EMS horizon` | 24 | Day-ahead planning window [h] |
| `MPC horizon` | 6 | Real-time prediction window [steps] |
| `Q_soc / R_power` | 1000 / 1 | MPC tracking weights |

## Technical Details

- **Solver**: All optimisation uses CasADi's `Opti` stack with the IPOPT interior-point method.
- **EMS formulation**: The charge and discharge powers are separate non-negative decision variables, which avoids piecewise or big-M formulations for asymmetric efficiency.  Simultaneous charge + discharge is automatically sub-optimal and never selected.
- **MPC feasibility**: Soft SOC constraints via slack variables with a heavy quadratic penalty ensure the MPC QP is always feasible, even under model mismatch or disturbances.
- **Warm-starting**: The MPC caches and shifts the previous solution to initialise the next solve, reducing IPOPT iterations from ~50 to ~5 in steady state.

## License

MIT
