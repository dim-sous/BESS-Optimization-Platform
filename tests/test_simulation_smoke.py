"""Smoke integration test — every strategy must run end-to-end on day 0.

Catches the class of regression that the 2026-04-15 trust reset existed
to prevent: refactors that leave the simulator's main loop, the
strategy recipes, or the result-dict shape contract in an inconsistent
state. The test runs a short (1-hour) horizon on real Q1 2024 EU
prices for every strategy registered in the comparison runner, then
asserts that:

  1. The result dict has every key the visualization and ledger expect
  2. All numeric outputs are finite
  3. SOC and SOH stay within physical bounds (with floating-point slack)
  4. The delivery score is a valid fraction
  5. The MPC solver does not fail

The test reuses ``STRATEGY_FACTORIES`` directly from the comparison
runner so it is the **single source of truth** for which strategies
should work end-to-end. Adding or removing a strategy in
``comparison/run_v5_comparison.py`` automatically updates the test
parametrization with no edits here.

Usage::

    uv run pytest -v tests/test_simulation_smoke.py
"""

from __future__ import annotations

import os

# Skip CasADi JIT compilation: it dominates the per-strategy wall time
# at this horizon length, and we are testing the simulator/ledger
# pipeline shape, not the JIT toolchain.
os.environ.setdefault("BESS_JIT", "0")

import numpy as np
import pytest

from comparison.run_v5_comparison import (
    ENERGY_CSV,
    REG_CSV,
    STRATEGY_FACTORIES,
)
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
from core.markets.price_loader import RealPriceLoader
from core.simulator.core import run_simulation


# Keys the ledger and downstream visualization rely on. If a refactor
# drops one of these the smoke test fails immediately.
REQUIRED_KEYS = frozenset({
    # Headline scalars
    "total_profit",
    "energy_profit_total",
    "capacity_revenue",
    "delivery_revenue",
    "penalty_cost",
    "net_regulation_profit",
    "deg_cost_total",
    "delivery_score",
    "soh_degradation",
    # Trace arrays
    "soc_true",
    "soh_true",
    "temp_true",
    "vterm_true",
    "power_applied",
    "power_applied_signed",
    "activation_signal",
    "power_delivered",
    "prices_energy",
    "prices_reg",
    # Solver instrumentation
    "mpc_solver_failures",
    "mpc_solve_times",
})


@pytest.fixture(scope="module")
def day0_inputs() -> dict:
    """Build a 1-hour day-0 input bundle once per test session.

    The 1-hour horizon keeps the test under a minute per strategy.
    Real EPEX SPOT + SMARD Q1 2024 prices, multi-cell pack truth model,
    standard parameter defaults — same configuration the comparison
    runner uses, just with sim_hours shrunk.
    """
    bp = BatteryParams()
    tp = TimeParams(sim_hours=1.0)         # short horizon -> fast test
    ep = EMSParams()
    mp = MPCParams()
    ekf_p = EKFParams()
    thp = ThermalParams()
    elp = ElectricalParams()
    reg_p = RegulationParams()
    pp = PackParams()

    loader = RealPriceLoader(ENERGY_CSV, reg_csv=REG_CSV, seed=42)

    # The forecast window has to cover the full sim horizon plus the
    # EMS planning horizon, same convention as run_v5_comparison.py.
    n_hours_total = int(tp.sim_hours) + ep.N_ems
    forecast_e, forecast_r, probabilities, realized_e, realized_r = (
        loader.generate_scenarios_for_day(
            day_idx=0,
            n_hours=n_hours_total,
            n_scenarios=5,
        )
    )

    return {
        "bp": bp,
        "tp": tp,
        "ep": ep,
        "mp": mp,
        "ekf_p": ekf_p,
        "thp": thp,
        "elp": elp,
        "reg_p": reg_p,
        "pp": pp,
        "forecast_e": forecast_e,
        "forecast_r": forecast_r,
        "probabilities": probabilities,
        "realized_e_prices": realized_e,
        "realized_r_prices": realized_r,
    }


@pytest.mark.parametrize(
    "name,factory",
    STRATEGY_FACTORIES,
    ids=[name for name, _ in STRATEGY_FACTORIES],
)
def test_strategy_runs_clean(name: str, factory, day0_inputs: dict) -> None:
    """Each strategy must produce a well-formed, physical result dict."""
    # Strategy factories take only the parameter dataclasses, not the
    # forecast / realized arrays.
    strat_kwargs = {
        k: day0_inputs[k]
        for k in ("bp", "tp", "ep", "mp", "ekf_p", "thp", "elp", "reg_p", "pp")
    }
    strategy = factory(**strat_kwargs)
    assert strategy.name == name, (
        f"factory for {name!r} produced strategy {strategy.name!r}"
    )

    # The simulator takes the parameters AND the forecast/realized arrays.
    sim_kwargs = {
        k: day0_inputs[k]
        for k in (
            "bp", "tp", "ep", "mp", "ekf_p", "thp", "elp", "reg_p", "pp",
            "forecast_e", "forecast_r", "probabilities",
            "realized_e_prices", "realized_r_prices",
        )
    }
    result = run_simulation(strategy=strategy, **sim_kwargs)

    # ---- Shape contract ----
    missing = REQUIRED_KEYS - set(result)
    assert not missing, f"{name}: result dict missing keys: {sorted(missing)}"

    # ---- Headline scalar finiteness ----
    for scalar_key in (
        "total_profit",
        "energy_profit_total",
        "capacity_revenue",
        "delivery_revenue",
        "penalty_cost",
        "net_regulation_profit",
        "deg_cost_total",
        "soh_degradation",
    ):
        assert np.isfinite(result[scalar_key]), (
            f"{name}: {scalar_key} = {result[scalar_key]!r} is not finite"
        )

    # ---- Trace finiteness ----
    for trace_key in ("soc_true", "soh_true", "temp_true", "vterm_true"):
        arr = np.asarray(result[trace_key])
        assert np.all(np.isfinite(arr)), (
            f"{name}: {trace_key} contains non-finite values"
        )

    # ---- Physical bounds (with small floating-point slack) ----
    soc = np.asarray(result["soc_true"])
    soh = np.asarray(result["soh_true"])
    assert (soc >= -1e-6).all() and (soc <= 1.0 + 1e-6).all(), (
        f"{name}: SOC out of [0, 1]: min={soc.min()}, max={soc.max()}"
    )
    assert (soh >= 0.5 - 1e-6).all() and (soh <= 1.0 + 1e-6).all(), (
        f"{name}: SOH out of [0.5, 1]: min={soh.min()}, max={soh.max()}"
    )

    # SOH only ever decreases (degradation is monotone)
    soh_diff = np.diff(soh)
    assert (soh_diff <= 1e-9).all(), (
        f"{name}: SOH increased somewhere — max +delta = {soh_diff.max()}"
    )

    # ---- Delivery score is a valid fraction ----
    delivery_score = float(result["delivery_score"])
    assert 0.0 <= delivery_score <= 1.0, (
        f"{name}: delivery_score = {delivery_score} not in [0, 1]"
    )

    # ---- Solver health ----
    failures = int(result.get("mpc_solver_failures", 0))
    assert failures == 0, (
        f"{name}: MPC solver failed {failures} times during smoke test"
    )
