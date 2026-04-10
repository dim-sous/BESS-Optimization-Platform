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
# Double-cycle day: morning peak + evening peak with a short trough
# between them. The LP discharges aggressively in the morning (using
# mean prices), leaving it with low SOC entering the evening. When
# hour 18 arrives with strong positive activation (up-reg = discharge),
# the LP's high P_reg + low SOC = delivery failure.
_E_MEAN = np.array([
    0.06, 0.06, 0.06, 0.06, 0.06, 0.06,   # 00-05  overnight flat
    0.18, 0.20, 0.18,                      # 06-08  morning peak (discharge)
    0.06, 0.06, 0.06,                      # 09-11  mid-morning
    0.03, 0.03,                            # 12-13  SHORT PV trough (2h)
    0.06, 0.06, 0.06, 0.06,                # 14-17  shoulder
    0.22, 0.22, 0.22,                      # 18-20  evening peak
    0.08, 0.08, 0.08,                      # 21-23  taper
])
assert _E_MEAN.shape == (N_HOURS,)

# Regulation-capacity price MEAN trajectory [$/kW/h]. Low during
# trough (cheap to commit FCR); peaks evening.
_R_MEAN = np.array([
    0.012, 0.012, 0.012, 0.012, 0.012, 0.012,
    0.010, 0.010, 0.010,
    0.012, 0.012, 0.012,
    0.004, 0.004,
    0.012, 0.012, 0.012, 0.012,
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

    Asymmetric fan designed so LP (mean-substitution) underestimates
    the evening peak and over-estimates the shoulder, leading to
    sub-optimal charge timing.

    s1 collapse   p=0.20  evening peak crushed to shoulder level
    s2 mild       p=0.20  mean * 0.75
    s3 median     p=0.20  mean * 1.00
    s4 firm+shift p=0.20  evening peak shifted +1h, scaled 1.3×
    s5 scarcity   p=0.20  evening peak tripled

    With uniform probabilities the LP sees a mean peak of ~$0.19,
    while scenarios s4/s5 have peaks of $0.29-0.66. The EMS hedges
    by charging more aggressively during the short trough. The
    realized day matches s5 magnitude, rewarding the hedge.
    """
    s1 = _E_MEAN.copy()
    s1[18:21] = _E_MEAN[14]   # evening collapses to shoulder level
    s2 = _E_MEAN * 0.80
    s3 = _E_MEAN.copy()
    s4 = _shift_evening_peak(_E_MEAN, shift=+1) * 1.25
    s5 = _E_MEAN.copy()
    s5[17] = 0.18              # shoulder rises early
    s5[18] = 0.55
    s5[19] = 0.66
    s5[20] = 0.55
    return np.stack([s1, s2, s3, s4, s5], axis=0)


def _build_reg_scenarios() -> np.ndarray:
    """Reg-capacity prices are far less dispersed than energy in real
    markets, so use a tight band around the mean."""
    factors = np.array([0.85, 0.95, 1.00, 1.05, 1.15])
    return factors[:, None] * _R_MEAN[None, :]


def _build_realized_energy() -> np.ndarray:
    """Realized day: drawn from the upper tail of the scenario fan.

    The peak hits at hours 18-20 (matches s3/s5 timing, NOT s4's shifted
    timing) with magnitude close to s5 — so EMS's hedging across s4/s5
    captures most of it, while LP (which planned for the mean that
    includes s1's collapsed evening) under-charges before evening.

    Sharp step 17->18 (0.10 -> 0.44) and 20->21 (0.40 -> 0.14) are the
    intra-horizon features that economic MPC can exploit.
    """
    realized = np.array([
        0.06, 0.06, 0.06, 0.06, 0.06, 0.06,
        0.18, 0.21, 0.18,
        0.06, 0.06, 0.06,
        0.03, 0.03,
        0.06, 0.06, 0.06, 0.16,
        0.48, 0.58, 0.44,
        0.14, 0.07, 0.06,
    ])
    assert realized.shape == (N_HOURS,)
    return realized


def _build_realized_reg() -> np.ndarray:
    """Realized reg-capacity prices: same as the mean."""
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
    recommended_sigma_mhz_mult: float = 3.0


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
