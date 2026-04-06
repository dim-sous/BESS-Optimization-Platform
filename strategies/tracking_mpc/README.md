# tracking_mpc — internal sanity check, NOT in pitch deck

**Pitch-visible:** no (sanity-check only)
**Composition:** `EconomicEMS` + `TrackingMPC` + `RegulationController` (PI)

## What it does

The "old v5 stack". The tracking MPC's objective is dominated by SOC tracking
(`Q_soc=1e4`) which forces it to mirror the EMS reference plan, leaving very
little room for intra-hour optimization.

## Why it exists

To demonstrate that a tracking-only MPC formulation is **dominated** by the
new economic-MPC approach. If `tracking_mpc` and `economic_mpc` produce
similar profits, the economic formulation isn't pulling its weight and we
should reconsider the v5 product story.

## Why it's not in the pitch deck

It's a comparison point against `economic_mpc`, not a competitor.
