"""Market clearing — Bid → Award.

Defines the abstract ``ClearingEngine`` protocol so a real HEnEx /
ADMIE REST connector can replace the synthetic stub later, plus
``ReferencePriceClearingStub``: a deterministic price-merit-order
clearing rule used for offline simulation.

Stub clearing rule
------------------
Each bid is matched against a single per-hour reference price for its
product. For sells (capacity, discharge energy, activation energy) the
bid clears iff ``bid.price <= reference_price[hour]``. For buys (charge
energy) the bid clears iff ``bid.price >= reference_price[hour]``.
Awarded quantity equals bid quantity (no partial fills in Phase 2;
``min_acceptance_ratio`` is stored on the bid for future use).

This rule is intentionally simple — it lets the planner's bidding
strategy be tested against realised prices without a full auction
simulator. Real markets clear at the marginal-bid price across the
merit order; that goes in a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from core.markets.bids import Award, Bid, BidBook
from core.markets.products import Product


# ---------------------------------------------------------------------------
#  Reference-price decomposition
# ---------------------------------------------------------------------------

def decompose_prices(
    energy_per_hour: np.ndarray,
    reg_per_hour: np.ndarray,
    *,
    idm_premium: float,
    mfrr_cap_share: float,
    afrr_cap_share: float,
    mfrr_act_factor: float,
    afrr_act_factor: float,
) -> dict[Product, np.ndarray]:
    """Synthesise the six Greek-product price arrays from the legacy
    two-channel forecast / realised-price layout.

    Mirrors the recipe used by ``MILPBiddingPlanner`` so the bids the
    planner emits are clearable against the same decomposition applied
    to *realised* prices in the clearing engine.

    Returns a dict with one ``np.ndarray`` of length ``N`` per product.
    """
    e = np.asarray(energy_per_hour, dtype=float)
    r = np.asarray(reg_per_hour, dtype=float)
    return {
        Product.HEnEx_DAM_Energy: e.copy(),
        Product.HEnEx_IDM_Energy: e * idm_premium,
        Product.mFRR_Capacity: r * mfrr_cap_share,
        Product.aFRR_Capacity: r * afrr_cap_share,
        Product.mFRR_Energy: e * mfrr_act_factor,
        Product.aFRR_Energy: e * afrr_act_factor,
    }


# ---------------------------------------------------------------------------
#  Clearing engine
# ---------------------------------------------------------------------------

class ClearingEngine(Protocol):
    """Stable interface that any real or synthetic clearing implementation
    must satisfy. The simulator does not care whether bids hit ADMIE's
    REST endpoint or a stub: both produce the same ``dict[Bid, Award]``.
    """

    def clear(self, bid_book: BidBook) -> dict[Bid, Award]:
        ...


# Sell-side products: clearing requires bid.price <= reference price.
# Buy legs of energy products are detected by the bid's `leg` field, so
# this set is actually used as "bids whose `leg='sell'` semantics apply
# regardless of leg field" — capacity and activation are always sells.
_ALWAYS_SELL_PRODUCTS: frozenset[Product] = frozenset({
    Product.mFRR_Capacity,
    Product.aFRR_Capacity,
    Product.mFRR_Energy,
    Product.aFRR_Energy,
})


@dataclass
class ReferencePriceClearingStub:
    """Synthetic clearing engine: bid clears iff its price beats the
    per-hour reference price for the same product.

    Parameters
    ----------
    references : dict[Product, np.ndarray]
        One reference price array of shape ``(N,)`` per product, in the
        LP scale (``$/kWh`` for energy and activation, ``$/kW/h`` for
        capacity). Built by ``decompose_prices`` from the realised
        single-channel prices.
    """

    references: dict[Product, np.ndarray]

    def clear(self, bid_book: BidBook) -> dict[Bid, Award]:
        out: dict[Bid, Award] = {}
        for bid in bid_book:
            ref_arr = self.references.get(bid.product)
            if ref_arr is None or bid.delivery_hour >= len(ref_arr):
                # Defensive: missing reference series ⇒ unfilled bid.
                out[bid] = Award(
                    accepted=False,
                    awarded_kw=0.0,
                    clearing_price_dollar_per_kwh=0.0,
                )
                continue

            ref_price = float(ref_arr[bid.delivery_hour])

            # Energy products are signed (DAM / IDM). The MILP emits a
            # separate Bid for each leg, with `leg="buy"` or "sell" set;
            # capacity and activation products are always sells.
            if bid.leg == "buy":
                accepted = bid.price_dollar_per_kwh >= ref_price
            else:
                accepted = bid.price_dollar_per_kwh <= ref_price

            out[bid] = Award(
                accepted=accepted,
                awarded_kw=bid.quantity_kw if accepted else 0.0,
                clearing_price_dollar_per_kwh=ref_price if accepted else 0.0,
            )
        return out
