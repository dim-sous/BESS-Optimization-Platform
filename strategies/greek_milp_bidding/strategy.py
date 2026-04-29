"""Greek MILP Bidding strategy — the v5b production recipe.

Wires together:

  - ``MILPBiddingPlanner``       multi-product MILP bidding tier
  - ``EconomicMPC``              real-time tactical dispatch
  - ``GreekMarketBiddingProtocol`` clearing engine + activation fractions

The MILP planner emits a six-product bid book (HEnEx DAM / IDM, mFRR
capacity + activation, aFRR capacity + activation) plus the v5-shape
``P_chg_ref / P_dis_ref / P_reg_ref`` aggregates that the EconomicMPC
needs. The simulator's bidding hook (Phase 3) records bids + awards
per hour; the ledger turns those into a ``greek_settlement`` block
alongside the existing v5 revenue lines.

This factory needs the realized energy + regulation prices passed
through ``**kwargs`` because the clearing engine clears against them.
The factory is keyword-only on those two arrays so existing strategy
factories that ignore them via ``**_unused`` keep working.
"""

from __future__ import annotations

import numpy as np

from core.config.parameters import (
    BatteryParams,
    ElectricalParams,
    EMSParams,
    MPCParams,
    ThermalParams,
    TimeParams,
)
from core.markets.clearing import ReferencePriceClearingStub, decompose_prices
from core.markets.products import Product
from core.mpc.adapters import EconomicMPCAdapter
from core.mpc.economic import EconomicMPC
from core.planners.milp_bidding import (
    MarketDecomposition,
    MILPBiddingConfig,
    MILPBiddingPlanner,
)
from core.simulator.bidding_protocol import GreekMarketBiddingProtocol
from core.simulator.strategy import Strategy


def make_strategy(
    bp: BatteryParams,
    tp: TimeParams,
    ep: EMSParams,
    mp: MPCParams,
    thp: ThermalParams,
    elp: ElectricalParams,
    *,
    realized_e_prices: np.ndarray | None = None,
    realized_r_prices: np.ndarray | None = None,
    market_decomposition: MarketDecomposition | None = None,
    milp_config: MILPBiddingConfig | None = None,
    k_dual: float = 1.25,
    **_unused,
) -> Strategy:
    """Build the ``greek_milp_bidding`` strategy.

    Parameters
    ----------
    realized_e_prices, realized_r_prices : np.ndarray
        Per-hour realised price arrays, length ≥ ``ep.N_ems``. Required.
        Used to populate the clearing engine's per-product reference
        prices via ``decompose_prices``.
    market_decomposition : MarketDecomposition | None
        Override the default decomposition (idm_premium, mfrr/afrr
        shares, activation factors / fractions). Used by both the
        planner and the clearing engine so the planner's bid prices
        match what the engine clears against.
    milp_config : MILPBiddingConfig | None
        Override MIP time limit, MBQ enforcement, etc.
    k_dual : float
        Imbalance penalty multiplier for dual-pricing settlement.
    """
    if realized_e_prices is None or realized_r_prices is None:
        raise ValueError(
            "greek_milp_bidding requires realized_e_prices and "
            "realized_r_prices (passed as kwargs to make_strategy). "
            "These populate the clearing engine's per-product reference "
            "prices."
        )

    decomp = market_decomposition if market_decomposition is not None else MarketDecomposition()
    cfg = milp_config if milp_config is not None else MILPBiddingConfig(decomposition=decomp)
    # Make sure planner and engine see the same decomposition object.
    if cfg.decomposition is not decomp:
        cfg = MILPBiddingConfig(
            mip_time_limit_s=cfg.mip_time_limit_s,
            mip_rel_gap=cfg.mip_rel_gap,
            enforce_mbq=cfg.enforce_mbq,
            decomposition=decomp,
            product_specs=cfg.product_specs,
        )

    planner = MILPBiddingPlanner(bp=bp, tp=tp, ep=ep, thp=thp, config=cfg)

    # Clearing engine: realised prices decomposed via the same recipe
    # the planner uses internally for forecast prices.
    realized_per_product = decompose_prices(
        energy_per_hour=np.asarray(realized_e_prices),
        reg_per_hour=np.asarray(realized_r_prices),
        idm_premium=decomp.idm_premium,
        mfrr_cap_share=decomp.mfrr_cap_share,
        afrr_cap_share=decomp.afrr_cap_share,
        mfrr_act_factor=decomp.mfrr_act_factor,
        afrr_act_factor=decomp.afrr_act_factor,
    )
    engine = ReferencePriceClearingStub(references=realized_per_product)

    bidding_protocol = GreekMarketBiddingProtocol(
        clearing_engine=engine,
        activation_fractions={
            Product.mFRR_Energy: decomp.alpha_mfrr,
            Product.aFRR_Energy: decomp.alpha_afrr,
        },
        k_dual=k_dual,
    )

    mpc = EconomicMPCAdapter(EconomicMPC(bp, tp, mp, thp, elp, ep))

    return Strategy(
        name="greek_milp_bidding",
        planner=planner,
        mpc=mpc,
        bidding_protocol=bidding_protocol,
        metadata={
            "label": "Greek Market MILP Bidding",
            "pitch_visible": False,
            "description": (
                "MILP-driven six-product bidding (HEnEx DAM / IDM, mFRR / aFRR "
                "capacity + activation) cleared against realised prices with "
                "Greek dual-pricing imbalance settlement."
            ),
        },
    )
