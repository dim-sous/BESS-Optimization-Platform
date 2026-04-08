# Backlog — Active Issues & Future Work

> **Frozen versions (v1–v4)** live in [archive/](archive/). They are not part
> of active development and should not be modified.
>
> **Historical gate reports** for v1–v5 are in [archive/gate_reports.md](archive/gate_reports.md).

---

## Current state (2026-04-15, post trust-reset)

**HEAD:** see `git log --oneline | head -10`. Working tree is the only
authoritative source of truth in this project. Anything not currently in
the source code, the git history, or the empirical results listed below
should be re-derived rather than recalled.

### Trust reset (2026-04-15)

This session purged the backlog, the design docs, and the suspect
empirical results because they were generated under heavy cognitive load
and at least one audit subagent had a documented factual error
(claiming LP and EMS lacked capacity revenue in their objectives, which
was false on inspection).

**What was deleted:**
- All "Bug A-E", "Concern F-I", "F1-F6", "EMS-A/B/C" audit-derived
  findings that were logged in this file
- `docs/realism_fix_1_design.md` (RF1 implementation is now the truth)
- `docs/wow_factor_1_design.md` (speculative pre-investigation thinking)
- `D1-D5` future-work catalog (speculative)
- `Wow Factor 1` plan (speculative)
- The two session-recap memory files in
  `.claude/projects/-home-user-bess/memory/`
- `results/v5_big_experiment.json` and `results/v5_big_traces/` (moved
  to `results/_quarantine_pre_2026_04_15/` — generated on a configuration
  that no longer exists, do not use without re-running)

**What survived** (only things verifiable on the current HEAD):
- The git commit history
- The current source code
- The 5-strategy ladder (verified in
  [comparison/run_v5_comparison.py](comparison/run_v5_comparison.py))
- The fresh 5-day big experiment ran on this HEAD (2026-04-15,
  post-purge) — results below
- The user's restated proposition (below)

### New canonical baseline (2026-04-15, post trust-reset, 5 days × 3 subsets × 5 strategies, 33 min wall)

All numbers from a fresh run on the current HEAD at
[results/v5_big_experiment.json](results/v5_big_experiment.json). These
are the only empirical numbers that should be trusted going forward.

**Net profit ($/day mean):**

| Strategy | Calm | Volatile | Stressed |
|---|---|---|---|
| rule_based      |  0.09 |  3.75 |  0.22 |
| deterministic_lp| 18.61 | 31.12 | 19.79 |
| **ems_clamps**  | **18.68** | **31.39** | **20.29** |
| tracking_mpc    | 18.49 | 31.10 | 18.68 |
| economic_mpc    | 18.49 | 30.86 | 20.07 |

**Headline (economic_mpc − ems_clamps):** **−$0.19 / −$0.52 / −$0.22**
per day. **The proposition is empirically false on all three regimes.**

**Ladder deltas:**
- rule_based → LP: +$18.52 / +$27.37 / +$19.57 (huge, the value of
  optimization itself)
- LP → ems_clamps: +$0.07 / +$0.27 / +$0.50 (small but monotone with
  disturbance — the real stochastic-optimization signal)
- ems_clamps → economic_mpc: −$0.19 / −$0.52 / −$0.22 (negative — MPC
  layer costs money)
- ems_clamps → tracking_mpc: −$0.19 / −$0.29 / −$1.61 (also negative,
  craters under stress)

**Non-numeric observations:**
- Delivery scores cluster at 76–79% across strategies and regimes.
  Everyone misses some delivery; the cheat is gone.
- tracking_mpc in stressed: 276 P_max touches/day, 76.2% delivery
  (lowest), 47% higher SOH degradation than other MPCs.
- economic_mpc in stressed: 0 P_max touches, 77.9% delivery, SOH same
  as ems_clamps.
- MPC solve time: ~0.04 s/step (warm-start IPOPT), ~60 per hour ×
  24 hours ≈ 57 s/day of pure solver work, roughly matching the ~100
  s/day wall time after EMS and EKF overhead.
- Wall time ratio: rule_based 7s · LP 7s · ems_clamps 40s · tracking_mpc
  92s · economic_mpc 106s per day. economic_mpc costs 2.7× ems_clamps
  in compute for negative return.

**Biggest single-day MPC loss vs ems_clamps (for the investigation):**
volatile day 38, economic_mpc $19.63 vs ems_clamps $20.56, delta
−$0.932. Traces are not persisted for this day (harness persists
first-day-per-subset only). Next-best persisted target: stressed day
41 (−$0.31 gap, traces in
[results/v5_big_traces/stressed_{strategy}_day41.npz](results/v5_big_traces/)).

### Restated proposition (user-approved, 2026-04-15)

> EMS + MPC strategies are strictly ≥ ALL other strategies on
> every metric we care about — profit, delivery score, constraint
> satisfaction, SOH preservation, robustness under disturbance — on the
> honest RF1 baseline, at the current ledger calibration, without any
> new architectural machinery. Activation tracking lives in the plant.
> There is no strategy-layer PI.

The proposition has not been verified on the current HEAD with a
multi-day experiment. Verifying it (or empirically rejecting it) is
the next concrete deliverable.

### Strategy ladder

| # | Strategy | Role |
|---|---|---|
| 1 | `rule_based` | Naive baseline |
| 2 | `deterministic_lp` | Commercial baseline (LP, mean-substitution) |
| 3 | `ems_clamps` | Canonical "EMS alone" (stochastic two-stage program) |
| 4 | `tracking_mpc` | Sanity control (kept as broken baseline) |
| 5 | `economic_mpc` | EMS + MPC. Production v5. |

**Canonical pitch comparison:** `economic_mpc` − `ems_clamps`.

### Architectural facts (verifiable in source)

- Activation tracking is performed inside `BatteryPlant.step()` — see
  [core/physics/plant.py](core/physics/plant.py). The strategy layer
  outputs `[setpoint_pnet, p_reg_committed]` only.
- There is no strategy-layer PI controller. `core/pi/` does not exist.
- The `Plan` dataclass at [core/planners/plan.py](core/planners/plan.py)
  carries both probability-weighted averages and per-scenario
  trajectories from `EconomicEMS`. The MPC currently consumes only the
  averages (mean-substitution).
- The plant clips dispatch in two passes (base setpoint, then setpoint
  + activation), so `p_delivered` is correctly attributed to the FCR
  portion only — see [core/physics/plant.py](core/physics/plant.py).

### Pending work

1. ✅ **Ground truth re-established.** 1-day sanity + 5-day big
   experiment landed on current HEAD (2026-04-15). Results above.
2. **Investigate the value leak.** Walk stressed day 41 (persisted
   traces) hour-by-hour, comparing `economic_mpc` against `ems_clamps`,
   and find where the MPC strategy loses money. The investigation is
   a *reading*, not an experiment — output is a precise diagnosis with
   code citations. If the pattern is clear, generalize to the volatile
   regime where the gap is biggest. If not, re-run volatile day 38
   with targeted trace persistence.
3. **Decide on next direction** based on the diagnosis.

### How to resume after a session break

1. Read this section of [backlog.md](backlog.md) (the current state).
2. Run `git log --oneline | head -10` to confirm HEAD.
3. Read [comparison/run_v5_comparison.py](comparison/run_v5_comparison.py)
   `STRATEGY_FACTORIES` to confirm the 5-strategy ladder.
4. Look at `results/` for the latest empirical baseline; if it doesn't
   exist or is outdated, re-run it before trusting any conclusion that
   depends on it.
5. Do not consult any pre-2026-04-15 audit findings, design docs, or
   memory entries. They were purged because they were not trustworthy.

---

## Future work — upcoming versions (intent, not commitments)

These are *roadmap intent*, not derived from any audit. They describe
what each subsequent version *would* add, in rough order of value.

- **v6** — Unscented Kalman Filter (replace EKF)
- **v7** — Joint state and parameter estimation (online R_internal,
  capacity, efficiency)
- **v8** — ACADOS NMPC (replace CasADi/IPOPT, RTI, control blocking)
- **v9** — Degradation-aware MPC (SOH in MPC state, profit-vs-degradation
  tradeoff)
- **v10** — Disturbance forecast uncertainty (scenario-based MPC, chance
  constraints)
- **v11** — Measurement and communication delays
- **v12** — Multi-battery system with central EMS coordinator
- **v13** — Grid-connected inverter model (id, iq, Vdc dynamics)
- **v14** — Market bidding optimization (day-ahead, reserve, intraday)
