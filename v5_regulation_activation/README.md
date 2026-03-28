# v5_regulation_activation — FCR Regulation Delivery with MPC Necessity

Extends the v4 electrical model with **real-time frequency containment reserve (FCR) regulation delivery**. The grid sends stochastic activation signals at 4-second resolution that the BESS must follow while maintaining SOC, thermal, and voltage constraints.

Adds a fast **PI controller** between MPC and the plant, creating a 4-level control hierarchy:

    EMS (3600s) → MPC (60s) → PI (4s) → Plant (4s)

## What Changed from v4

| Component | v4_electrical_rc_model | v5_regulation_activation |
|-----------|----------------------|--------------------------|
| **Control hierarchy** | EMS → MPC → Plant | EMS → MPC → PI → Plant |
| **Simulation dt** | 5 s | 4 s (matches activation signal resolution) |
| **Activation signals** | Not modeled | Stochastic ±P_reg at 4s resolution |
| **PI controller** | None | Tracks activation signal with SOC safety limits |
| **Revenue model** | Arbitrage + capacity bidding | Arbitrage + capacity + delivery revenue − penalties |
| **Strategy comparison** | Single strategy | `--strategy full\|ems_only\|no_regulation` |
| **EMS** | Arbitrage + capacity commitment | Adds SOC headroom margin for regulation delivery |
| **MPC** | 3-state (SOC, SOH, T) tracking | 2-state (SOC, T) + SOH frozen as parameter |

**Unchanged**: 5-state model (SOC, SOH, T, V_rc1, V_rc2), 2RC circuit, OCV polynomial, pack architecture (4 cells, active balancing), EKF/MHE estimation.

## Key Concept: MPC Necessity

This version demonstrates that MPC is **indispensable** for real-time grid service delivery:

- **EMS-only** dispatch cannot react to stochastic activation signals → SOC constraint violations, delivery failures, penalties
- **EMS+MPC** provides closed-loop feedback → smooth delivery, constraint satisfaction, higher net profit

Run the formal comparison with `--strategy ems_only` vs `--strategy full`.

## Regulation Delivery Model

The grid sends a normalized activation signal `a(t) ∈ [-1, 1]` at 4-second intervals. The BESS must deliver:

    P_reg(t) = a(t) × P_committed

where `P_committed` is the capacity bid from the EMS (hourly). A PI controller tracks this signal:

    P_pi(t) = Kp × e(t) + Ki × ∫e(τ)dτ

with SOC safety margins that curtail delivery near SOC limits to prevent constraint violations.

## Revenue Structure

| Component | Description |
|-----------|-------------|
| Energy arbitrage | Buy low / sell high on day-ahead prices |
| Capacity revenue | Payment for committed regulation capacity ($/kW/h) |
| Delivery revenue | Payment for energy delivered during activation ($/kWh) |
| Penalty cost | Penalty for under-delivery vs committed capacity |
| Degradation cost | Battery wear from all power flows |

## Control Strategies

| Strategy | Description |
|----------|-------------|
| `full` | EMS + MPC + PI — full hierarchical control |
| `ems_only` | EMS dispatch only, no MPC feedback — demonstrates MPC necessity |
| `no_regulation` | Arbitrage only, no regulation participation |

## Module Structure

```
v5_regulation_activation/
├── main.py                       # Entry point with --strategy and --mhe flags
├── config/
│   └── parameters.py             # All v4 params + PIParams, RegulationParams, Strategy
├── models/
│   └── battery_model.py          # 5-state CasADi + numpy dynamics (from v4)
├── ems/
│   └── economic_ems.py           # Hourly planning with SOC headroom for regulation
├── mpc/
│   └── tracking_mpc.py           # 2-state MPC (SOC, T) with SOH as parameter
├── pi/
│   └── regulation_pi.py          # Fast PI controller for activation signal tracking
├── estimation/
│   ├── ekf.py                    # 5-state, 3-measurement EKF (from v4)
│   └── mhe.py                    # 5-state MHE (optional, --mhe flag)
├── simulation/
│   └── simulator.py              # 4-level multi-rate coordinator
├── data/
│   ├── price_generator.py        # Stochastic price scenarios
│   ├── activation_generator.py   # Stochastic regulation activation signals
│   └── real_price_loader.py      # Load real market price data
├── revenue/
│   └── regulation_revenue.py     # Delivery revenue, penalty, and score calculation
├── visualization/
│   └── plot_results.py           # Results plotting with regulation-specific panels
└── stress_test.py                # Stress test suite
```

## Running

```bash
# From repository root
uv run python v5_regulation_activation/main.py                    # full strategy (default)
uv run python v5_regulation_activation/main.py --strategy ems_only
uv run python v5_regulation_activation/main.py --strategy no_regulation

# With MHE estimator (slower)
uv run python v5_regulation_activation/main.py --mhe

# Run stress tests
uv run python v5_regulation_activation/stress_test.py
```

## Status

**In development** — MPC simplified (2-state, 24% faster solves). Pending stress testing and gate review. See `backlog.md` for gate process.
