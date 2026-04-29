"""Bidding protocol — gate-closure hook for the simulator.

The simulator's ``Strategy`` recipe holds an optional
``BiddingProtocol``. When set, the simulator calls
``protocol.on_gate_closure(sim_step, plan_dict)`` after each EMS solve
to obtain a ``BidBook``, then ``protocol.clear(bid_book)`` to obtain
``Awards``. Both are recorded into ``SimTraces`` and consumed by the
ledger to produce the per-product Greek settlement block.

When ``bidding_protocol`` is ``None`` (every existing v5 strategy), the
simulator's behaviour is bit-identical to the pre-Phase-3 baseline.
That property is what allows the v5 pitch ladder
(``rule_based / deterministic_lp / ems / ems_economic_mpc``) to keep
producing the exact same traces while a new ``greek_milp_bidding``
strategy lights up the bidding tier on the same input bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.markets.bids import Award, Bid, BidBook
from core.markets.clearing import ClearingEngine
from core.markets.products import Product


class BiddingProtocol(Protocol):
    """Interface every bidding implementation (stub or real) must satisfy."""

    def on_gate_closure(self, sim_step: int, plan_dict: dict) -> BidBook:
        ...

    def clear(self, bid_book: BidBook) -> dict[Bid, Award]:
        ...


@dataclass
class GreekMarketBiddingProtocol:
    """Default Greek-market protocol: pulls the bid book from the
    MILP planner's ``plan_dict`` and clears it via the wired
    ``ClearingEngine``.

    The activation-fraction map is held alongside the engine so the
    ledger can compute settlement at the same fractions the planner
    optimised against.
    """

    clearing_engine: ClearingEngine
    activation_fractions: dict[Product, float] = field(default_factory=dict)
    k_dual: float = 1.25

    def on_gate_closure(self, sim_step: int, plan_dict: dict) -> BidBook:
        book = plan_dict.get("bid_book")
        if book is None:
            # Defensive: a strategy wired to a bidding protocol should
            # use a planner that emits a bid book. Fail loudly rather
            # than silently dropping the bidding tier.
            raise ValueError(
                "Bidding protocol active but planner returned no 'bid_book'. "
                "Use MILPBiddingPlanner or another planner that emits a "
                "BidBook in its plan_dict."
            )
        book.submitted_at_step = sim_step
        return book

    def clear(self, bid_book: BidBook) -> dict[Bid, Award]:
        return self.clearing_engine.clear(bid_book)
