# Archive

Frozen artifacts that are preserved as audit trail / engineering history but
**should not be modified** and should be excluded from active development
searches.

## Contents

| Path | What | Status |
|---|---|---|
| [v1_baseline/](v1_baseline/) | Baseline 2-state model + simple EMS/MPC/EKF/MHE | Frozen |
| [v2_thermal_model/](v2_thermal_model/) | Adds thermal coupling + Arrhenius degradation | Frozen |
| [v3_pack_model/](v3_pack_model/) | Adds multi-cell pack with heterogeneity + balancing | Frozen |
| [v4_electrical_rc_model/](v4_electrical_rc_model/) | Adds 2RC equivalent circuit + OCV polynomial | Frozen |
| [gate_reports.md](gate_reports.md) | Four-stage gate reports for v1–v5 | Historical record |

## Why archived

- v1–v4 are frozen per [CLAUDE.md](../CLAUDE.md). They were never repaired
  for the bugs found in later audits and **should not be cited as
  physically-correct numbers** in any external context. They exist only as
  history of how the platform was built up incrementally.
- The gate reports were taking ~80 % of [backlog.md](../backlog.md) as
  historical archive that no active work references. They are preserved here
  for audit-trail purposes.

## Active development

Active code lives in:
- [core/](../core/) — shared platform modules (physics, markets, controllers, accounting, simulator)
- [strategies/](../strategies/) — one folder per strategy (rule_based, deterministic_lp, ems_clamps, ems_pi, tracking_mpc, economic_mpc)
- [comparison/](../comparison/) — strategy comparison harness
- [presentation/](../presentation/) — pitch deck generator
