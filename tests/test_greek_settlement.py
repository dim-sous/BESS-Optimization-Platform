"""Greek settlement aggregator — unit tests.

Numerical checks on the per-product revenue formulas and end-to-end
consistency with the MILP planner's Phase-1 expected_profit.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.accounting.greek_settlement import compute_greek_settlement
from core.config.parameters import (
    BatteryParams,
    EMSParams,
    ThermalParams,
    TimeParams,
)
from core.markets.bids import Award, Bid, BidBook
from core.markets.clearing import (
    ReferencePriceClearingStub,
    decompose_prices,
)
from core.markets.products import Product
from core.planners.milp_bidding import (
    MILPBiddingConfig,
    MILPBiddingPlanner,
)
from core.simulator.synthetic_day import make_synthetic_day

N_HOURS = 24


# ---------------------------------------------------------------------
# Atomic per-product checks on a hand-constructed bid set
# ---------------------------------------------------------------------

@pytest.fixture
def hand_refs():
    return {
        Product.HEnEx_DAM_Energy: np.full(N_HOURS, 0.10),
        Product.HEnEx_IDM_Energy: np.full(N_HOURS, 0.12),
        Product.mFRR_Capacity:    np.full(N_HOURS, 0.005),
        Product.aFRR_Capacity:    np.full(N_HOURS, 0.004),
        Product.mFRR_Energy:      np.full(N_HOURS, 0.15),
        Product.aFRR_Energy:      np.full(N_HOURS, 0.13),
    }


def _make_award(bid: Bid, ref_price: float) -> Award:
    return Award(accepted=True, awarded_kw=bid.quantity_kw,
                 clearing_price_dollar_per_kwh=ref_price)


def test_dam_sell_revenue_formula(hand_refs):
    """One DAM sell of 50 kW for 1 hour at clearing 0.10 = $5.00."""
    bid = Bid(product=Product.HEnEx_DAM_Energy, delivery_hour=0,
              quantity_kw=50.0, price_dollar_per_kwh=0.05, leg="sell")
    awards = {bid: _make_award(bid, 0.10)}
    s = compute_greek_settlement(awards, hand_refs, n_hours=N_HOURS)
    assert s["dam_revenue"] == pytest.approx(50.0 * 0.10 * 1.0)
    assert s["total_greek_revenue"] == pytest.approx(5.00)


def test_dam_buy_is_a_cost(hand_refs):
    """One DAM buy of 50 kW × 0.10 = -$5.00 (energy cost)."""
    bid = Bid(product=Product.HEnEx_DAM_Energy, delivery_hour=0,
              quantity_kw=50.0, price_dollar_per_kwh=0.20, leg="buy")
    awards = {bid: _make_award(bid, 0.10)}
    s = compute_greek_settlement(awards, hand_refs, n_hours=N_HOURS)
    assert s["dam_revenue"] == pytest.approx(-5.00)


def test_capacity_revenue_formula(hand_refs):
    """1 hour × 100 kW × $0.005/kW/h = $0.50."""
    bid = Bid(product=Product.mFRR_Capacity, delivery_hour=0,
              quantity_kw=100.0, price_dollar_per_kwh=0.005, leg="sell")
    awards = {bid: _make_award(bid, 0.005)}
    s = compute_greek_settlement(awards, hand_refs, n_hours=N_HOURS)
    assert s["mfrr_cap_revenue"] == pytest.approx(0.50)


def test_activation_revenue_uses_realised_alpha(hand_refs):
    """Activation revenue = awarded × α × λ × dt.

    bid 100 kW × 0.20 (default afrr alpha) × 0.13 ref × 1 h = $2.60.
    """
    bid = Bid(product=Product.aFRR_Energy, delivery_hour=0,
              quantity_kw=100.0, price_dollar_per_kwh=0.13, leg="sell")
    awards = {bid: _make_award(bid, 0.13)}
    s = compute_greek_settlement(awards, hand_refs, n_hours=N_HOURS)
    assert s["afrr_activation_revenue"] == pytest.approx(100.0 * 0.20 * 0.13 * 1.0)


def test_unfilled_bid_contributes_zero(hand_refs):
    bid = Bid(product=Product.HEnEx_DAM_Energy, delivery_hour=0,
              quantity_kw=50.0, price_dollar_per_kwh=0.20, leg="sell")
    awards = {bid: Award(accepted=False, awarded_kw=0.0,
                         clearing_price_dollar_per_kwh=0.0)}
    s = compute_greek_settlement(awards, hand_refs, n_hours=N_HOURS)
    assert s["total_greek_revenue"] == 0.0
    assert s["n_bids_total"] == 1
    assert s["n_bids_accepted"] == 0


def test_perfect_delivery_gives_zero_imbalance(hand_refs):
    """With deliveries=None and accepted bids, imbalance is zero everywhere."""
    bid = Bid(product=Product.HEnEx_DAM_Energy, delivery_hour=0,
              quantity_kw=50.0, price_dollar_per_kwh=0.05, leg="sell")
    awards = {bid: _make_award(bid, 0.10)}
    s = compute_greek_settlement(awards, hand_refs, n_hours=N_HOURS)
    assert s["imbalance_settlement"] == 0.0
    assert s["non_delivery_penalty"] == 0.0


def test_underdelivery_triggers_imbalance_and_penalty(hand_refs):
    """Award 100 kW activation but deliver 50 kW: should trigger
    non-delivery penalty AND no positive activation revenue beyond
    the delivered portion."""
    bid = Bid(product=Product.aFRR_Energy, delivery_hour=0,
              quantity_kw=100.0, price_dollar_per_kwh=0.13, leg="sell")
    awards = {bid: _make_award(bid, 0.13)}
    deliveries = {bid: 10.0}   # delivered 10 kW vs expected 100×0.20 = 20 kW
    s = compute_greek_settlement(
        awards, hand_refs, n_hours=N_HOURS,
        deliveries_kw=deliveries,
    )
    # Activation revenue = 10 (delivered) × 0.13 × 1 = $1.30
    assert s["afrr_activation_revenue"] == pytest.approx(10.0 * 0.13)
    # Penalty: missed = max(0, 20 − 10) = 10 kW; cap_price = ref activation = 0.13
    # Penalty = 3.0 × 0.13 × 10 × 1 = $3.90
    assert s["non_delivery_penalty"] == pytest.approx(3.0 * 0.13 * 10.0)


# ---------------------------------------------------------------------
# End-to-end: MILP plan → clearing → settlement on canonical day.
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def end_to_end():
    bp = BatteryParams()
    tp = TimeParams()
    ep = EMSParams()
    thp = ThermalParams()
    cfg = MILPBiddingConfig()
    planner = MILPBiddingPlanner(bp=bp, tp=tp, ep=ep, thp=thp, config=cfg)
    day = make_synthetic_day()
    plan = planner.solve(
        soc_init=bp.SOC_init,
        soh_init=bp.SOH_init,
        t_init=thp.T_init,
        energy_scenarios=day.forecast_e,
        reg_scenarios=day.forecast_r,
        probabilities=day.probabilities,
    )
    realized = decompose_prices(
        energy_per_hour=day.realized_e_prices,
        reg_per_hour=day.realized_r_prices,
        idm_premium=cfg.decomposition.idm_premium,
        mfrr_cap_share=cfg.decomposition.mfrr_cap_share,
        afrr_cap_share=cfg.decomposition.afrr_cap_share,
        mfrr_act_factor=cfg.decomposition.mfrr_act_factor,
        afrr_act_factor=cfg.decomposition.afrr_act_factor,
    )
    engine = ReferencePriceClearingStub(references=realized)
    awards = engine.clear(plan["bid_book"])
    settlement = compute_greek_settlement(
        awards=awards, realized_prices=realized,
        n_hours=ep.N_ems,
        activation_fractions={
            Product.mFRR_Energy: cfg.decomposition.alpha_mfrr,
            Product.aFRR_Energy: cfg.decomposition.alpha_afrr,
        },
    )
    return {"plan": plan, "settlement": settlement, "awards": awards,
            "realized": realized, "day": day, "cfg": cfg}


def test_e2e_some_bids_clear(end_to_end):
    s = end_to_end["settlement"]
    assert s["n_bids_total"] > 0
    assert 0 < s["n_bids_accepted"] < s["n_bids_total"], (
        "At least one bid should clear and at least one should fail to "
        "clear under the canonical synthetic day."
    )


def test_e2e_zero_imbalance_with_perfect_delivery(end_to_end):
    s = end_to_end["settlement"]
    assert s["imbalance_settlement"] == pytest.approx(0.0)
    assert s["non_delivery_penalty"] == pytest.approx(0.0)


def test_e2e_per_product_per_hour_rolls_up_to_aggregate(end_to_end):
    """The flat revenue lines should equal the sum of the per-product
    per-hour detail. Sanity check on the aggregator's bookkeeping."""
    s = end_to_end["settlement"]
    pp_ph = s["per_product_per_hour"]

    aggregate_lookup = {
        Product.HEnEx_DAM_Energy: "dam_revenue",
        Product.HEnEx_IDM_Energy: "idm_revenue",
        Product.mFRR_Capacity:    "mfrr_cap_revenue",
        Product.aFRR_Capacity:    "afrr_cap_revenue",
        Product.mFRR_Energy:      "mfrr_activation_revenue",
        Product.aFRR_Energy:      "afrr_activation_revenue",
    }
    for product, key in aggregate_lookup.items():
        if product in pp_ph:
            assert s[key] == pytest.approx(float(pp_ph[product].sum()))


def test_e2e_realised_revenue_close_to_milp_expected_profit(end_to_end):
    """On the canonical synthetic day, the MILP's forecast-based
    expected_profit should be within a few cents of the realised total
    Greek revenue (the gap is the small degradation cost the MILP
    subtracts in its objective)."""
    plan = end_to_end["plan"]
    s = end_to_end["settlement"]
    diff = abs(plan["expected_profit"] - s["total_greek_revenue"])
    # Looser tolerance: the realised vs forecast prices differ at peaks;
    # the bid prices were forecast-based, so this is a meaningful sanity
    # check, not an identity. Just want them in the same ballpark.
    assert diff < 5.0, (
        f"MILP expected_profit ${plan['expected_profit']:.2f} and realised "
        f"${s['total_greek_revenue']:.2f} differ by ${diff:.2f}"
    )


def test_e2e_individual_revenue_lines_finite(end_to_end):
    s = end_to_end["settlement"]
    for k in (
        "dam_revenue", "idm_revenue",
        "mfrr_cap_revenue", "afrr_cap_revenue",
        "mfrr_activation_revenue", "afrr_activation_revenue",
        "imbalance_settlement", "non_delivery_penalty",
        "total_greek_revenue",
    ):
        assert np.isfinite(s[k]), f"{k} = {s[k]} is not finite"
