"""Synthetic 1-day market dataset designed to rank the v5 strategies.

Goal
----
Produce ONE 24-hour bundle of exogenous channels (energy + regulation
prices, scenarios, realized values) such that, run through the v5
simulator, the five strategies rank strictly by total profit:

    rule_based  <  deterministic_lp  <  ems  <  ems_tracking_mpc  <  ems_economic_mpc

The dataset uses only the existing channel schema — no simulator
changes, no sub-hourly price cheating. The legitimate levers are:

  1. rule_based -> LP
     Multi-trough / multi-peak structure (deep midday PV trough +
     evening peak) with a non-zero regulation-capacity stream the
     LP can commit to. rule_based ignores reg and only chains a
     single charge/discharge pair.

  2. LP -> ems
     Asymmetric scenario dispersion: a fat tail (s5) where the
     evening peak doubles. LP collapses to the mean and underweights
     the tail; the stochastic EMS hedges by entering evening with
     more SOC. The realized day is drawn from the upper part of the
     distribution so the hedge pays off.

  3. ems -> ems_tracking_mpc
     The EMS plan is built once per hour against scenarios; live FCR
     activation perturbs SOC inside the hour. Tracking MPC's 60 s
     closed loop rejects the perturbation and arrives at the next
     EMS boundary on plan, which the open-loop EMS does not.

  4. ems_tracking_mpc -> ems_economic_mpc
     Sharp adjacent-hour price steps (17->18 and 21->22). The
     economic MPC's 60-min horizon straddles the hour boundary, so
     the upcoming step enters its objective ~30 min early and it
     pre-positions intra-hour SOC for the peak. Tracking MPC follows
     the hour-averaged EMS power reference and cannot pre-position.
     Additionally: when the same FCR burst that helps tracking MPC
     beat EMS hits in the run-up to the peak, tracking MPC pays
     power to chase the stale SOC anchor back; economic MPC accepts
     the new state and re-optimizes terminal-only.

The activation signal is generated inside the simulator from the
RegulationParams seed — we expose `recommended_activation_seed`
so callers can pass it through `RegulationParams(activation_seed=…)`.

Battery context (for sizing — see core/config/parameters.py)
------------------------------------------------------------
  E_nom = 200 kWh,  P_max = 100 kW,  SOC in [0.10, 0.90]
  -> usable energy = 160 kWh, full discharge time = 1.6 h
  -> 1 kWh round-trip at the 0.19 $/kWh trough->peak spread is
     worth ~0.17 $/kWh after eta_rt = 0.9025.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Channel design constants. Tuned against the default 200 kWh / 100 kW pack.
# ---------------------------------------------------------------------------

N_HOURS = 24
N_SCENARIOS = 5

# Energy price MEAN trajectory [$/kWh], one value per hour.
# Two-peak day with a deep midday trough and a sharp evening peak.
_E_MEAN = np.array([
    0.06, 0.06, 0.06, 0.06, 0.06, 0.06,   # 00-05  overnight flat
    0.14, 0.14, 0.14,                      # 06-08  morning peak
    0.03, 0.03, 0.03, 0.03, 0.03, 0.03,    # 09-14  PV trough
    0.09, 0.09, 0.09,                      # 15-17  shoulder  (sharp step at 17->18)
    0.22, 0.22, 0.22,                      # 18-20  evening peak
    0.08, 0.08, 0.08,                      # 21-23  taper      (sharp step at 21->22)
])
assert _E_MEAN.shape == (N_HOURS,)

# Regulation-capacity price MEAN trajectory [$/kW/h]. Low midday (cheap
# to commit FCR while energy arbitrage is unattractive); peaks evening.
_R_MEAN = np.array([
    0.012, 0.012, 0.012, 0.012, 0.012, 0.012,
    0.010, 0.010, 0.010,
    0.004, 0.004, 0.004, 0.004, 0.004, 0.004,
    0.012, 0.012, 0.012,
    0.020, 0.020, 0.020,
    0.014, 0.014, 0.014,
])
assert _R_MEAN.shape == (N_HOURS,)


def _shift_evening_peak(profile: np.ndarray, shift: int) -> np.ndarray:
    """Roll the 18-20 evening peak by `shift` hours, in place-safe."""
    out = profile.copy()
    if shift == 0:
        return out
    peak_hours = (18, 19, 20)
    fill = profile[15]  # shoulder value, used to clear the original window
    peak_vals = profile[list(peak_hours)]
    for h in peak_hours:
        out[h] = fill
    for h, v in zip(peak_hours, peak_vals):
        new_h = h + shift
        if 0 <= new_h < N_HOURS:
            out[new_h] = v
    return out


def _build_energy_scenarios() -> np.ndarray:
    """5 scenarios, shape (5, 24).

    s1 mild       p=0.20  mean * 0.6
    s2 soft       p=0.20  mean * 0.85
    s3 median     p=0.20  mean
    s4 firm+shift p=0.20  mean * 1.20, evening peak shifted +1h (18-20 -> 19-21)
    s5 scarcity   p=0.20  mean * 1.30 with evening peak DOUBLED on hour 19
    """
    s1 = _E_MEAN * 0.60
    s2 = _E_MEAN * 0.85
    s3 = _E_MEAN.copy()
    s4 = _shift_evening_peak(_E_MEAN, shift=+1) * 1.20
    s5 = _E_MEAN * 1.30
    s5[18] = 0.34
    s5[19] = 0.45
    s5[20] = 0.34
    return np.stack([s1, s2, s3, s4, s5], axis=0)


def _build_reg_scenarios() -> np.ndarray:
    """Reg-capacity prices are far less dispersed than energy in real
    markets, so use a tight band around the mean."""
    factors = np.array([0.85, 0.95, 1.00, 1.05, 1.15])
    return factors[:, None] * _R_MEAN[None, :]


def _build_realized_energy() -> np.ndarray:
    """Realized day: between s4 and s5. The peak hits at hours 18-20
    (matches s3/s5 timing, NOT s4's shifted timing) with magnitude
    closer to s5 — so EMS's hedging across s4/s5 captures most of it,
    while LP (which planned for the mean) under-charges before evening.

    Sharp step 17->18 (0.10 -> 0.30) and 21->22 (0.18 -> 0.07) are the
    intra-horizon-but-cross-hour features that economic MPC exploits.
    """
    realized = np.array([
        0.06, 0.06, 0.06, 0.06, 0.06, 0.06,
        0.13, 0.15, 0.13,
        0.04, 0.03, 0.03, 0.03, 0.03, 0.04,
        0.09, 0.10, 0.10,
        0.30, 0.34, 0.28,
        0.18, 0.07, 0.06,
    ])
    assert realized.shape == (N_HOURS,)
    return realized


def _build_realized_reg() -> np.ndarray:
    """Realized reg-capacity prices: same as the mean (low variance)."""
    return _R_MEAN.copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyntheticDay:
    """Bundle of channel arrays consumed by `core.simulator.core.run_simulation`.

    Attributes
    ----------
    forecast_e         : (5, 24) energy price scenarios          [$/kWh]
    forecast_r         : (5, 24) regulation-capacity scenarios   [$/kW/h]
    probabilities      : (5,) scenario probabilities             [-]
    realized_e_prices  : (24,) realized energy prices            [$/kWh]
    realized_r_prices  : (24,) realized regulation prices        [$/kW/h]
    recommended_activation_seed : int
        RNG seed for `RegulationParams(activation_seed=…)`. The
        activation signal is generated inside the simulator from
        this seed; the value below was chosen because the OU + DFD
        realization places a sustained burst near the run-up to the
        evening peak (the situation where tracking MPC's plan-anchor
        cost differs most from economic MPC's terminal-only cost).
    """

    forecast_e: np.ndarray
    forecast_r: np.ndarray
    probabilities: np.ndarray
    realized_e_prices: np.ndarray
    realized_r_prices: np.ndarray
    recommended_activation_seed: int = 99


def make_synthetic_day() -> SyntheticDay:
    """Construct the canonical 1-day synthetic dataset."""
    forecast_e = _build_energy_scenarios()
    forecast_r = _build_reg_scenarios()
    probabilities = np.full(N_SCENARIOS, 1.0 / N_SCENARIOS)
    realized_e = _build_realized_energy()
    realized_r = _build_realized_reg()
    return SyntheticDay(
        forecast_e=forecast_e,
        forecast_r=forecast_r,
        probabilities=probabilities,
        realized_e_prices=realized_e,
        realized_r_prices=realized_r,
    )
