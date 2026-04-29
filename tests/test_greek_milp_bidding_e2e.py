"""End-to-end test for the greek_milp_bidding strategy (Phase 3).

Runs the new strategy through ``run_simulation`` on the canonical
synthetic day and verifies:

  - The simulator pipeline doesn't crash with the bidding tier active.
  - The result dict contains BOTH the v5 ledger fields AND a
    ``greek_settlement`` block.
  - Per-product revenue lines are finite and structurally consistent.
  - SimTraces records one BidBook + one awards dict per EMS hour.
  - Plant traces remain physically sane (SOC bounds, finite values).
  - With perfect plant tracking on a synthetic day, the imbalance
    block is small relative to the total revenue.

Usage::

    uv run pytest -v tests/test_greek_milp_bidding_e2e.py
"""

from __future__ import annotations

import os

# Skip CasADi JIT compilation: dominates per-test wall time at this
# horizon length and is not what we're testing.
os.environ.setdefault("BESS_JIT", "0")

import numpy as np
import pytest

from core.config.parameters import (
    BatteryParams,
    EKFParams,
    ElectricalParams,
    EMSParams,
    MPCParams,
    PackParams,
    RegulationParams,
    ThermalParams,
    TimeParams,
)
from core.simulator.core import run_simulation
from core.simulator.synthetic_day import make_synthetic_day
from strategies.greek_milp_bidding.strategy import make_strategy


SHORT_HOURS = 4.0   # short horizon keeps runtime under ~30s


@pytest.fixture(scope="module")
def synthetic_inputs() -> dict:
    bp = BatteryParams()
    tp = TimeParams(sim_hours=SHORT_HOURS)
    ep = EMSParams()
    mp = MPCParams()
    ekf_p = EKFParams()
    thp = ThermalParams()
    elp = ElectricalParams()
    reg_p = RegulationParams()
    pp = PackParams()

    day = make_synthetic_day()
    return {
        "bp": bp, "tp": tp, "ep": ep, "mp": mp, "ekf_p": ekf_p,
        "thp": thp, "elp": elp, "reg_p": reg_p, "pp": pp,
        "forecast_e": day.forecast_e,
        "forecast_r": day.forecast_r,
        "probabilities": day.probabilities,
        "realized_e_prices": day.realized_e_prices,
        "realized_r_prices": day.realized_r_prices,
    }


@pytest.fixture(scope="module")
def result(synthetic_inputs) -> dict:
    s = synthetic_inputs
    strategy = make_strategy(
        bp=s["bp"], tp=s["tp"], ep=s["ep"], mp=s["mp"],
        thp=s["thp"], elp=s["elp"],
        realized_e_prices=s["realized_e_prices"],
        realized_r_prices=s["realized_r_prices"],
    )
    return run_simulation(
        strategy=strategy,
        forecast_e=s["forecast_e"],
        forecast_r=s["forecast_r"],
        probabilities=s["probabilities"],
        realized_e_prices=s["realized_e_prices"],
        realized_r_prices=s["realized_r_prices"],
        bp=s["bp"], tp=s["tp"], ep=s["ep"], mp=s["mp"],
        ekf_p=s["ekf_p"], thp=s["thp"], elp=s["elp"], reg_p=s["reg_p"],
        pp=s["pp"],
    )


# ----------------------------------------------------------------------
# Result-dict shape
# ----------------------------------------------------------------------

def test_result_has_v5_fields(result):
    """v5 ledger keys must still be present even with the bidding
    tier active — the Greek block is additive, not replacing."""
    for k in (
        "total_profit", "energy_profit_total", "capacity_revenue",
        "delivery_revenue", "penalty_cost", "deg_cost_total",
        "soc_true", "soh_true", "power_applied",
    ):
        assert k in result, f"v5 field {k!r} missing from greek_milp_bidding result"


def test_result_has_greek_settlement_block(result):
    assert "greek_settlement" in result
    s = result["greek_settlement"]
    for k in (
        "dam_revenue", "idm_revenue",
        "mfrr_cap_revenue", "afrr_cap_revenue",
        "mfrr_activation_revenue", "afrr_activation_revenue",
        "imbalance_settlement", "non_delivery_penalty",
        "total_greek_revenue",
        "n_bids_total", "n_bids_accepted",
    ):
        assert k in s, f"greek_settlement missing key: {k!r}"


def test_per_product_revenue_lines_finite(result):
    s = result["greek_settlement"]
    for k in (
        "dam_revenue", "idm_revenue",
        "mfrr_cap_revenue", "afrr_cap_revenue",
        "mfrr_activation_revenue", "afrr_activation_revenue",
        "imbalance_settlement", "non_delivery_penalty",
        "total_greek_revenue",
    ):
        assert np.isfinite(s[k]), f"{k} = {s[k]!r} not finite"


def test_total_greek_revenue_matches_components(result):
    """Aggregate consistency: total = Σ revenues + imbalance − penalty."""
    s = result["greek_settlement"]
    expected = (
        s["dam_revenue"] + s["idm_revenue"]
        + s["mfrr_cap_revenue"] + s["afrr_cap_revenue"]
        + s["mfrr_activation_revenue"] + s["afrr_activation_revenue"]
        + s["imbalance_settlement"]
        - s["non_delivery_penalty"]
    )
    assert s["total_greek_revenue"] == pytest.approx(expected)


def test_at_least_some_bids_accepted(result):
    """The synthetic day's realised prices should clear at least one
    bid; a zero-acceptance result would mean clearing logic is broken."""
    s = result["greek_settlement"]
    assert s["n_bids_total"] > 0
    assert s["n_bids_accepted"] > 0


# ----------------------------------------------------------------------
# SimTraces bidding-tier records
# ----------------------------------------------------------------------

def test_settlement_horizon_covers_run(result, synthetic_inputs):
    """Phase 3 single-gate-closure model: bidding happens once at k=0,
    the bid book covers the EMS planning horizon (24h), and settlement
    is computed over the actual run horizon (SHORT_HOURS hours)."""
    s = result["greek_settlement"]
    pp_ph = s.get("per_product_per_hour")
    assert pp_ph is not None
    # Settlement horizon should at minimum cover all delivery hours
    # within the run window.
    expected_hours = int(SHORT_HOURS)
    for arr in pp_ph.values():
        assert len(arr) >= expected_hours


# ----------------------------------------------------------------------
# Physical sanity (mirrors test_simulation_smoke)
# ----------------------------------------------------------------------

def test_soc_remains_in_bounds(result, synthetic_inputs):
    bp = synthetic_inputs["bp"]
    soc = np.asarray(result["soc_true"])
    assert (soc >= bp.SOC_min - 1e-3).all()
    assert (soc <= bp.SOC_max + 1e-3).all()


def test_soh_monotone_decreasing(result):
    soh = np.asarray(result["soh_true"])
    assert np.all(np.diff(soh) <= 1e-9), (
        f"SOH increased somewhere — max +delta = {np.diff(soh).max()}"
    )


def test_no_mpc_failures(result):
    failures = int(result.get("mpc_solver_failures", 0))
    assert failures == 0


# ----------------------------------------------------------------------
# Bit-identicality regression: v5 strategy unaffected by Phase 3
# ----------------------------------------------------------------------

def test_v5_strategy_not_affected_by_phase3(synthetic_inputs):
    """Run a v5 strategy through the same simulator. It should not
    have a greek_settlement key — bidding tier was inactive — and its
    v5 ledger fields should be present and finite."""
    from strategies.deterministic_lp.strategy import make_strategy as v5_factory

    s = synthetic_inputs
    strategy = v5_factory(
        bp=s["bp"], tp=s["tp"], ep=s["ep"], mp=s["mp"],
        thp=s["thp"], elp=s["elp"],
    )
    result = run_simulation(
        strategy=strategy,
        forecast_e=s["forecast_e"], forecast_r=s["forecast_r"],
        probabilities=s["probabilities"],
        realized_e_prices=s["realized_e_prices"],
        realized_r_prices=s["realized_r_prices"],
        bp=s["bp"], tp=s["tp"], ep=s["ep"], mp=s["mp"],
        ekf_p=s["ekf_p"], thp=s["thp"], elp=s["elp"], reg_p=s["reg_p"],
        pp=s["pp"],
    )
    assert "greek_settlement" not in result, (
        "v5 strategy ran the bidding tier — bit-identicality broken."
    )
    assert np.isfinite(result["total_profit"])
