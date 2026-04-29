"""ReferencePriceClearingStub — unit tests.

Verifies the bid-vs-reference clearing rule for sells and buys, the
edge cases (missing references, out-of-range hour), and the price
decomposition helper.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.markets.bids import Bid, BidBook
from core.markets.clearing import (
    ReferencePriceClearingStub,
    decompose_prices,
)
from core.markets.products import Product

N = 4


@pytest.fixture
def refs() -> dict[Product, np.ndarray]:
    """Reference prices per product per hour, length N=4."""
    return {
        Product.HEnEx_DAM_Energy: np.array([0.10, 0.20, 0.30, 0.05]),
        Product.HEnEx_IDM_Energy: np.array([0.12, 0.22, 0.32, 0.06]),
        Product.mFRR_Capacity:    np.array([0.005, 0.005, 0.010, 0.010]),
        Product.aFRR_Capacity:    np.array([0.004, 0.004, 0.008, 0.008]),
        Product.mFRR_Energy:      np.array([0.15, 0.30, 0.45, 0.07]),
        Product.aFRR_Energy:      np.array([0.13, 0.26, 0.39, 0.06]),
    }


@pytest.fixture
def engine(refs):
    return ReferencePriceClearingStub(references=refs)


def test_sell_bid_at_or_below_reference_clears(engine):
    """Sell bid at price <= reference clears at the reference price."""
    book = BidBook()
    book.add(Bid(product=Product.HEnEx_DAM_Energy,
                 delivery_hour=1, quantity_kw=20.0,
                 price_dollar_per_kwh=0.15, leg="sell"))   # 0.15 <= ref 0.20
    awards = engine.clear(book)
    bid = book.bids[0]
    a = awards[bid]
    assert a.accepted is True
    assert a.awarded_kw == pytest.approx(20.0)
    assert a.clearing_price_dollar_per_kwh == pytest.approx(0.20)


def test_sell_bid_above_reference_rejects(engine):
    book = BidBook()
    book.add(Bid(product=Product.HEnEx_DAM_Energy,
                 delivery_hour=0, quantity_kw=20.0,
                 price_dollar_per_kwh=0.15, leg="sell"))   # 0.15 > ref 0.10
    awards = engine.clear(book)
    bid = book.bids[0]
    a = awards[bid]
    assert a.accepted is False
    assert a.awarded_kw == 0.0


def test_buy_bid_at_or_above_reference_clears(engine):
    """Buy bid at price >= reference clears."""
    book = BidBook()
    book.add(Bid(product=Product.HEnEx_DAM_Energy,
                 delivery_hour=0, quantity_kw=15.0,
                 price_dollar_per_kwh=0.12, leg="buy"))    # 0.12 >= ref 0.10
    awards = engine.clear(book)
    a = awards[book.bids[0]]
    assert a.accepted is True
    assert a.awarded_kw == pytest.approx(15.0)


def test_buy_bid_below_reference_rejects(engine):
    book = BidBook()
    book.add(Bid(product=Product.HEnEx_DAM_Energy,
                 delivery_hour=2, quantity_kw=15.0,
                 price_dollar_per_kwh=0.20, leg="buy"))    # 0.20 < ref 0.30
    awards = engine.clear(book)
    a = awards[book.bids[0]]
    assert a.accepted is False
    assert a.awarded_kw == 0.0


def test_capacity_product_always_treated_as_sell(engine):
    """Capacity products are always sells (you offer reserve to TSO)."""
    book = BidBook()
    book.add(Bid(product=Product.mFRR_Capacity,
                 delivery_hour=2, quantity_kw=50.0,
                 price_dollar_per_kwh=0.005, leg="sell"))   # 0.005 <= 0.010
    awards = engine.clear(book)
    a = awards[book.bids[0]]
    assert a.accepted is True
    assert a.clearing_price_dollar_per_kwh == pytest.approx(0.010)


def test_missing_reference_for_product_rejects(engine):
    """If the engine has no reference array for a product, bids reject."""
    refs_partial = {Product.HEnEx_DAM_Energy: np.array([0.10])}
    eng2 = ReferencePriceClearingStub(references=refs_partial)
    book = BidBook()
    book.add(Bid(product=Product.aFRR_Capacity,
                 delivery_hour=0, quantity_kw=10.0,
                 price_dollar_per_kwh=0.001, leg="sell"))
    awards = eng2.clear(book)
    a = awards[book.bids[0]]
    assert a.accepted is False


def test_out_of_range_hour_rejects(engine):
    book = BidBook()
    book.add(Bid(product=Product.HEnEx_DAM_Energy,
                 delivery_hour=99, quantity_kw=10.0,
                 price_dollar_per_kwh=0.01, leg="sell"))
    awards = engine.clear(book)
    assert awards[book.bids[0]].accepted is False


def test_clearing_preserves_bid_count(engine):
    book = BidBook()
    for h in range(N):
        book.add(Bid(product=Product.HEnEx_DAM_Energy,
                     delivery_hour=h, quantity_kw=10.0,
                     price_dollar_per_kwh=0.05, leg="sell"))
    awards = engine.clear(book)
    assert len(awards) == len(book)


# ---------------------------------------------------------------------
# decompose_prices
# ---------------------------------------------------------------------

def test_decompose_prices_shapes_and_relations():
    e = np.array([0.10, 0.20, 0.30])
    r = np.array([0.01, 0.02, 0.03])
    out = decompose_prices(
        energy_per_hour=e, reg_per_hour=r,
        idm_premium=1.05, mfrr_cap_share=0.6, afrr_cap_share=0.4,
        mfrr_act_factor=1.5, afrr_act_factor=1.3,
    )
    np.testing.assert_allclose(out[Product.HEnEx_DAM_Energy], e)
    np.testing.assert_allclose(out[Product.HEnEx_IDM_Energy], e * 1.05)
    np.testing.assert_allclose(out[Product.mFRR_Capacity], r * 0.6)
    np.testing.assert_allclose(out[Product.aFRR_Capacity], r * 0.4)
    np.testing.assert_allclose(out[Product.mFRR_Energy], e * 1.5)
    np.testing.assert_allclose(out[Product.aFRR_Energy], e * 1.3)
    # Capacity shares partition reg total (aFRR + mFRR shares = 1.0)
    np.testing.assert_allclose(
        out[Product.mFRR_Capacity] + out[Product.aFRR_Capacity], r,
    )
