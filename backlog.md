# Backlog — Active Issues & Future Work

> **Frozen versions (v1–v4)** live in [archive/](archive/). They are not part
> of active development and should not be modified.
>
> **Historical gate reports** for v1–v5 are in [archive/gate_reports.md](archive/gate_reports.md).

---

## Active issues

### 0. v5 simulator: hard audit findings (2026-04-06)

A strict audit of v5 simulator + strategies turned up four real bugs in the
execution layer between the optimizers and the plant. **The relative ranking
across LP / EMS_PI / FULL / FULL_ECON is probably still meaningful** because
all suffer the bugs in roughly equal proportion, but **absolute numbers and
the rule_based baseline are wrong**.

**Audits that PASSED:**
- Information leak — planners only see forecast scenarios, never realized prices
- EKF observability — uses noisy measurements, not true state
- Multi-rate cadence — correct array lengths
- Per-strategy semantic checks — rule_based has P_reg=0 etc.

**Bug A — Wash trades from PI activation modulation.**
The PI controller in `pi/regulation_controller.py` only nets P_chg/P_dis when
their sum exceeds P_max. Otherwise both stay nonzero. The plant then computes
`dSOC = (eta_c*P_chg - P_dis/eta_d)/E_eff`, applying round-trip losses to BOTH
directions independently. Energy silently leaks to the wash trade.
*Example*: MPC plans `P_chg=21.7, P_dis=0`. Activation demands `P_dis=10`. PI
sets `[21.7, 10, P_reg]`. Plant computes `(0.95*21.7 - 10/0.95)/E_eff = 10.1/E_eff`
instead of the correct net `(0.95*11.7)/E_eff = 11.1/E_eff` — ~9% efficiency
loss on the wash portion.
*Affected*: every PI-using strategy and the EMS_CLAMPS open-loop path.

**Bug B — Power budget violations.**
The PI controller adds activation modulation on top of MPC chg/dis without
rechecking `P_chg + P_reg <= P_max` or `P_dis + P_reg <= P_max`. The plant
doesn't enforce the budget either — it clips each input to [0, P_max]
independently. The battery is silently commanded above its rated power.
*Counts on day 3*: deterministic_lp 6845 chg+reg violations / 5177 dis+reg;
EMS_CLAMPS 6958 / 5240; EMS_PI 9216 / 10247; FULL 13350 / 14639;
FULL_ECON 8363 / 9877. `P_total` used for thermal Joule heating routinely
exceeds 100 kW, inflating temperature and degradation.

**Bug C — `power_applied` records commands, not actuals.** [Critical]
The plant has pre-integration limiting (it clips P_chg/P_dis to SOC headroom),
but `MultiRateSimulator` records the unclipped commanded values into
`power_applied`. All downstream accounting bills strategies for power that
the battery never accepted.
*Smoking gun*: rule_based hour 4 commands `P_chg=80 kW` continuously. Predicted
SOC end = 1.264 (impossible). Actual = 0.900 (clamped). Strategy is credited
with ~80 kWh of cheap-price charging that physically never happened.
*Severity*: rule_based severe (SOC error 0.375); LP/EMS/MPC mild (~0.0025).

**Bug D — Multi-cell pack scaling vs alpha_deg calibration.**
`alpha_deg` was calibrated for "1 FCE/day → 1.37 %/yr at the PACK level", but
the multi-cell `BatteryPack` distributes pack power equally across `n_cells`
cells, so each cell sees `P_pack/n`. The pack reports MIN(cell_SOH). Net
effect: pack SOH/yr is reported at ~3.6× lower than the calibration target.
*Treatment*: kept as a feature per user — model mismatch is real, MPC should
plan around it. Relative SOH ranking across strategies is unchanged. Absolute
SOH numbers should be flagged in any external deck.

**Architectural root cause**
The simulator's `run()` method is a 500+ line monolith with 6 strategy
branches inlined. Bugs A/B/C live in tangled if/else chains nobody can read in
one sitting. The plant's `step()` returns only `(state, measurement)` and
never reports the actually-applied power.
**Fix:** the major refactor planned in
[/home/user/.claude/plans/radiant-crafting-cosmos.md](../../.claude/plans/radiant-crafting-cosmos.md)
— linear simulator core, plant returns u_applied, single signed `P_net`,
strategies as composition recipes, modular `core/` + `strategies/` layout.

---

## Future work

### Upcoming versions (deferred until v5 refactor lands and audits pass)

- **v6** — Unscented Kalman Filter (replace EKF)
- **v7** — Joint state and parameter estimation (online R_internal, capacity, efficiency)
- **v8** — ACADOS NMPC (replace CasADi/IPOPT, RTI, control blocking)
- **v9** — Degradation-aware MPC (SOH in MPC state, profit-vs-degradation tradeoff)
- **v10** — Disturbance forecast uncertainty (scenario-based MPC, chance constraints)
- **v11** — Measurement and communication delays
- **v12** — Multi-battery system with central EMS coordinator
- **v13** — Grid-connected inverter model (id, iq, Vdc dynamics)
- **v14** — Market bidding optimization (day-ahead, reserve, intraday)

### v5 follow-ups (after refactor)

- Add aFRR / mFRR revenue streams alongside FCR
- 84-day gate run with cleaned execution layer
- B2B pitch deck regeneration with audit-clean numbers
- Stress test sweep: tighter SOC bounds, smaller battery duration, intraday volatility
