"""Bid / BidBook / Award — pure data, no logic.

The MILPBiddingPlanner emits a BidBook per gate-closure event; a
ClearingEngine (Phase 2) returns Awards keyed by the originating Bid.

Convention: quantity_kw is always >= 0. Direction is encoded in the
product's ProductSpec, not on the bid itself. For SYMMETRIC energy
products (DAM, IDM, aFRR_Energy) the planner emits separate bids for
the discharge and charge legs in any given hour; the clearing engine
sees them as independent offers with independent prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from core.markets.products import Product


@dataclass(frozen=True)
class Bid:
    """A single offer: one product, one delivery hour."""

    product: Product
    delivery_hour: int                          # 0..N-1
    quantity_kw: float                          # >= 0
    price_dollar_per_kwh: float                 # LP-scale numerical convention
    leg: str = "sell"                           # "sell" or "buy" (for SYMMETRIC)
    block_id: int | None = None                 # reserved for block-bid grouping
    min_acceptance_ratio: float = 1.0           # 1.0 = all-or-nothing


@dataclass(frozen=True)
class Award:
    """Clearing outcome for one bid."""

    accepted: bool
    awarded_kw: float
    clearing_price_dollar_per_kwh: float


@dataclass
class BidBook:
    """Bids submitted at one gate-closure event."""

    bids: list[Bid] = field(default_factory=list)
    submitted_at_step: int = 0

    def add(self, bid: Bid) -> None:
        if bid.quantity_kw < 0:
            raise ValueError(
                f"Bid.quantity_kw must be >= 0, got {bid.quantity_kw}"
            )
        self.bids.append(bid)

    def __iter__(self) -> Iterator[Bid]:
        return iter(self.bids)

    def __len__(self) -> int:
        return len(self.bids)

    def by_product(self, product: Product) -> list[Bid]:
        return [b for b in self.bids if b.product == product]
