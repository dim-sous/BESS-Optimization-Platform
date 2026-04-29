"""Greek-market product taxonomy.

Single source of truth for product identifiers, market sessions, minimum
bid quantities, and direction conventions used across the new multi-market
bidding layer:

  - core/markets/bids.py        Bid / BidBook / Award dataclasses
  - core/markets/clearing.py    ClearingEngine protocol + stub (Phase 2)
  - core/markets/imbalance.py   dual-pricing imbalance settlement (Phase 2)
  - core/planners/milp_bidding.py   MILP planner that emits BidBooks

Greek market context (HEnEx / ADMIE / IPTO)
-------------------------------------------
HEnEx Day-Ahead Market (DAM)
    Hourly energy auction with D-1 12:00 EET gate closure, uniform
    clearing price.
HEnEx Intraday Market (IDM)
    Continuous + multi-auction sessions during D-day. Phase 1 models
    a single representative auction at hour 12 of D-day; multi-session
    is a follow-up.
Balancing Market (operated by ADMIE/IPTO under MARI/PICASSO)
    mFRR (manual Frequency Restoration Reserve) — capacity payment
        (EUR/MW/h) plus activation energy (EUR/MWh, UP-direction).
    aFRR (automatic FRR) — capacity payment plus activation energy
        (symmetric direction).

Numerical-scale convention
--------------------------
The LP/MILP layer matches the existing v5 channel scale ($/kWh and
$/kW/h) so comparisons against `deterministic_lp` are apples-to-apples.
The Greek-market interpretation (treat the numbers as EUR/MWh × 1e-3)
is documented in `strategies/greek_milp_bidding/README.md` and is a
unit-semantics overlay — there are no silent conversions inside this
module.

Production note: real Greek MBQs are 1 MW per product. The default
MBQ here is 10 kW so the canonical 100 kW demo pack can clear at all;
the v5 strategy README flags this and the production target.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MarketSession(str, Enum):
    """Top-level market sessions in the Greek electricity market."""

    HEnEx_DAM = "HEnEx_DAM"
    HEnEx_IDM = "HEnEx_IDM"
    Balancing = "Balancing"


class Direction(str, Enum):
    """Activation / settlement direction.

    UP        = generator-direction (battery discharges).
    DOWN      = load-direction (battery charges).
    SYMMETRIC = product is settled signed; positive quantity = net
                discharge in that hour.
    """

    UP = "UP"
    DOWN = "DOWN"
    SYMMETRIC = "SYMMETRIC"


class Product(str, Enum):
    """The seven products the multi-market layer can bid into."""

    HEnEx_DAM_Energy = "HEnEx_DAM_Energy"
    HEnEx_IDM_Energy = "HEnEx_IDM_Energy"
    mFRR_Capacity = "mFRR_Capacity"
    aFRR_Capacity = "aFRR_Capacity"
    mFRR_Energy = "mFRR_Energy"
    aFRR_Energy = "aFRR_Energy"
    Imbalance = "Imbalance"


@dataclass(frozen=True)
class ProductSpec:
    """Static market rules for one product.

    Used by:
      - MILPBiddingPlanner to build per-product bounds and MBQ floors.
      - ClearingEngine implementations to route bids by session.
      - Settlement to map awards to revenue formulas.
    """

    product: Product
    session: MarketSession
    direction: Direction
    min_bid_qty_kw: float            # MBQ floor; 0 disables
    tick_dollar_per_kwh: float       # price increment, LP scale
    gate_closure_offset_h: int       # hours before delivery hour
    is_capacity: bool                # True = $/kW/h commitment; False = energy
    label: str

    @property
    def is_energy(self) -> bool:
        return not self.is_capacity


# ---------------------------------------------------------------------------
# Default product specs — small enough to read at a glance.
# Override per-strategy by passing a dict-merge to MILPBiddingPlanner.
# ---------------------------------------------------------------------------

DEFAULT_PRODUCT_SPECS: dict[Product, ProductSpec] = {
    Product.HEnEx_DAM_Energy: ProductSpec(
        product=Product.HEnEx_DAM_Energy,
        session=MarketSession.HEnEx_DAM,
        direction=Direction.SYMMETRIC,
        min_bid_qty_kw=10.0,
        tick_dollar_per_kwh=1e-4,
        gate_closure_offset_h=12,
        is_capacity=False,
        label="HEnEx Day-Ahead Energy",
    ),
    Product.HEnEx_IDM_Energy: ProductSpec(
        product=Product.HEnEx_IDM_Energy,
        session=MarketSession.HEnEx_IDM,
        direction=Direction.SYMMETRIC,
        min_bid_qty_kw=10.0,
        tick_dollar_per_kwh=1e-4,
        gate_closure_offset_h=1,
        is_capacity=False,
        label="HEnEx Intraday Energy",
    ),
    Product.mFRR_Capacity: ProductSpec(
        product=Product.mFRR_Capacity,
        session=MarketSession.Balancing,
        direction=Direction.SYMMETRIC,
        min_bid_qty_kw=10.0,
        tick_dollar_per_kwh=1e-4,
        gate_closure_offset_h=1,
        is_capacity=True,
        label="mFRR Capacity",
    ),
    Product.aFRR_Capacity: ProductSpec(
        product=Product.aFRR_Capacity,
        session=MarketSession.Balancing,
        direction=Direction.SYMMETRIC,
        min_bid_qty_kw=10.0,
        tick_dollar_per_kwh=1e-4,
        gate_closure_offset_h=1,
        is_capacity=True,
        label="aFRR Capacity",
    ),
    Product.mFRR_Energy: ProductSpec(
        product=Product.mFRR_Energy,
        session=MarketSession.Balancing,
        direction=Direction.UP,
        min_bid_qty_kw=10.0,
        tick_dollar_per_kwh=1e-4,
        gate_closure_offset_h=1,
        is_capacity=False,
        label="mFRR Activation Energy",
    ),
    Product.aFRR_Energy: ProductSpec(
        product=Product.aFRR_Energy,
        session=MarketSession.Balancing,
        direction=Direction.SYMMETRIC,
        min_bid_qty_kw=10.0,
        tick_dollar_per_kwh=1e-4,
        gate_closure_offset_h=1,
        is_capacity=False,
        label="aFRR Activation Energy",
    ),
}


def product_spec(p: Product) -> ProductSpec:
    return DEFAULT_PRODUCT_SPECS[p]
