# greek_milp_bidding — Greek-market MILP bidding strategy (v5b)

A v5 strategy that adds a **multi-product MILP bidding tier** on top of
the existing EMS + Economic MPC stack. Adapts the platform to the
Greek electricity market structure as described by the METLEN Energy
Market Optimization Expert role: HEnEx Day-Ahead, HEnEx Intraday, and
ADMIE/IPTO Balancing (mFRR + aFRR), with automated bidding workflows
and dual-pricing imbalance settlement.

## What it does

```
MILPBiddingPlanner            EconomicMPC                 Plant
  ├─► six-product bid book   ├─► tactical dispatch        ├─► applies setpoints
  ├─► P_chg / P_dis / P_reg  │   against EMS plan         └─► reports actuals
  └─► binaries (mutex, MBQ)
                              ▲
                              │ same plan_dict shape v5 expects
```

Per gate-closure (hourly EMS solve), the simulator:

1. Calls `MILPBiddingPlanner.solve(...)` — returns a `plan_dict`
   extended with a `bid_book` (the six-product offer set).
2. Calls `bidding_protocol.on_gate_closure(...)` to publish the bid
   book and `clear(...)` to obtain awards from the
   `ReferencePriceClearingStub` (clearing against realised prices).
3. Stores both bids and awards into `SimTraces`.
4. After the run, the ledger calls
   `compute_greek_settlement_from_traces` to translate plant traces
   into per-bid deliveries and produce a per-product Greek settlement
   block in the result dict.

## Greek market terminology in code

| Code identifier | Greek market product |
|---|---|
| `Product.HEnEx_DAM_Energy` | HEnEx Day-Ahead Energy (uniform clearing) |
| `Product.HEnEx_IDM_Energy` | HEnEx Intraday Energy (single representative auction) |
| `Product.mFRR_Capacity` | manual Frequency Restoration Reserve capacity (€/MW/h) |
| `Product.aFRR_Capacity` | automatic FRR capacity |
| `Product.mFRR_Energy` | mFRR activation energy (€/MWh delivered) |
| `Product.aFRR_Energy` | aFRR activation energy |

Synthetic decomposition (Phase 1–3 default):
- `idm_premium = 1.05` — IDM = DAM × 1.05
- `mfrr_cap_share = 0.6, afrr_cap_share = 0.4` — split of total reg-cap price
- `mfrr_act_factor = 1.5, afrr_act_factor = 1.3` — activation premium over energy
- `α_mfrr = 0.10, α_afrr = 0.20` — expected activation fractions
- `k_dual = 1.25` — Greek dual-pricing penalty multiplier

A real HEnEx/ADMIE data pipeline would replace `decompose_prices` with
per-product historical traces; the `ClearingEngine` Protocol is the
abstraction boundary for swapping in a real connector.

## Wash-trade-free guarantee

The MILP enforces `b_dis[k] + b_chg[k] ≤ 1` for every hour, with the
power-budget constraints
`P_dam_dis + P_idm_dis ≤ P_max · b_dis` and
`P_dam_chg + P_idm_chg ≤ P_max · b_chg`.
At most one of `{b_dis, b_chg}` is 1, so at most one of
`{discharge legs, charge legs}` is non-zero per hour. Simultaneous
charge+discharge is **infeasible by construction** — the audit-bug
fix is the binary mutex itself, not a degraded objective preference.

Verified by `tests/test_milp_bidding_invariants.py::test_C1_*` for
every hour of the canonical synthetic day.

## How to run

End-to-end on the canonical synthetic day:

```bash
uv run python -m strategies.greek_milp_bidding.run --day 0
```

Standalone MILP solve + clearing/settlement (no plant):

```bash
uv run python -m core.planners.milp_bidding --solve --plot
```

Tests:

```bash
uv run pytest -v tests/test_milp_bidding_invariants.py \
                 tests/test_clearing.py \
                 tests/test_imbalance.py \
                 tests/test_greek_settlement.py \
                 tests/test_greek_milp_bidding_e2e.py
```

## Known scope cuts (deferred)

- Real HEnEx/ADMIE REST connector — Protocol designed for it; stub for now.
- Multi-session intraday — single IDM gate at hour 12 in Phase 3; multi-session is a follow-up.
- Block-bid contiguity constraints — binary infrastructure is in place; the constraints themselves are deferred.
- Scenario-aware MILP — uses probability-weighted means like `deterministic_lp` (consistent with the v5 ladder).
- Asymmetric UP/DOWN capacity split — single symmetric variable per product.
- Promotion to v5 strategy ladder — `pitch_visible=False`. Not in
  `comparison/run_v5_comparison.py`'s `STRATEGY_FACTORIES`. Promote in a follow-up.

## Default parameters

| Parameter | Default | Notes |
|---|---|---|
| `MBQ_HEnEx`, `MBQ_mFRR`, `MBQ_aFRR` | 10 kW | Real Greek MBQs are 1 MW. Demo pack is 100 kW so 10 kW lets bids actually clear. |
| `mip_rel_gap` | 1e-4 | HiGHS default. |
| `mip_time_limit_s` | 30.0 | Falls back to LP relaxation if exceeded. |
| Currency / scale | $/kWh & $/kW/h | Same numeric scale as v5 ladder; Greek-market interpretation is documented, not silently converted. |
