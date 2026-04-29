"""Greek market settlement aggregator.

Per-product revenue lines for a 24h settlement period given:
  - the per-bid Awards from the clearing engine
  - the realised per-product reference prices for the same period
  - (optional) per-bid delivered_kw for imbalance computation
  - (optional) per-product realised activation fractions

Phase 2 (offline) operation: ``deliveries`` defaults to ``awarded_kw``
for every accepted bid (perfect delivery), so imbalance and
non-delivery penalty are zero. Phase 3 passes a ``deliveries`` map
constructed from plant traces via
``compute_greek_settlement_from_traces``, and the same function
produces non-trivial imbalance / penalty terms when the plant
deviates from the awarded plan.

Returns a structured dict matching the v5 ledger's flat-revenue style
so it can be merged into the existing ``compute_ledger`` result by the
simulator integration in Phase 3.
"""

from __future__ import annotations

import numpy as np

from core.config.parameters import BatteryParams, TimeParams
from core.markets.bids import Award, Bid
from core.markets.imbalance import settle_imbalance
from core.markets.products import Product, product_spec


def compute_greek_settlement(
    awards: dict[Bid, Award],
    realized_prices: dict[Product, np.ndarray],
    *,
    n_hours: int,
    dt_h: float = 1.0,
    deliveries_kw: dict[Bid, float] | None = None,
    expected_delivery_kw: dict[Bid, float] | None = None,
    activation_fractions: dict[Product, float] | None = None,
    system_direction: np.ndarray | None = None,
    k_dual: float = 1.25,
    bp: BatteryParams | None = None,
    non_delivery_penalty_mult: float = 3.0,
) -> dict:
    """Compute per-product Greek market revenue for one 24h settlement.

    Parameters
    ----------
    awards : dict[Bid, Award]
        Output of ``ClearingEngine.clear``.
    realized_prices : dict[Product, np.ndarray]
        Per-product realised price arrays of length ``n_hours``.
    n_hours : int
        Number of settlement hours (24 for the canonical day).
    dt_h : float
        Hours per settlement period (default 1.0 — hourly).
    deliveries_kw : dict[Bid, float] | None
        Phase 3 input: per-bid power physically delivered. ``None`` ⇒
        perfect delivery (delivered = awarded). Activation-energy
        bids' delivery is interpreted differently — see notes below.
    expected_delivery_kw : dict[Bid, float] | None
        Per-bid demanded delivery, used only for activation-energy
        non-delivery-penalty computation. If ``None``, defaults to
        ``α × awarded`` (long-run-fraction model used in Phase-2
        offline). Phase-3 trace-aware path passes
        ``realised_activation_fraction_per_hour × awarded`` via
        ``compute_greek_settlement_from_traces``, which is closer to
        the real Greek settlement convention (penalise missed
        per-hour realised demand, not long-run expectation).
    activation_fractions : dict[Product, float] | None
        Long-run activation fraction per product. Used only when
        ``expected_delivery_kw`` is ``None``. Defaults to 0.10 mFRR /
        0.20 aFRR (Phase-1 ``MarketDecomposition`` defaults).
    system_direction : np.ndarray | None
        +1/-1/0 per hour; controls dual-pricing imbalance penalty.
        ``None`` ⇒ all zeros (no directional penalty).
    k_dual : float
        Imbalance penalty multiplier. See ``markets/imbalance.py``.
    bp : BatteryParams | None
        Reserved for Phase 3 (degradation accounting links).
    non_delivery_penalty_mult : float
        Multiplier on capacity-price for missed activation delivery.
        Mirrors ``RegulationParams.penalty_mult`` (default 3.0).

    Returns
    -------
    dict
        Flat revenue lines plus per-product per-hour detail. Keys::

            dam_revenue                    [$]
            idm_revenue                    [$]
            mfrr_cap_revenue               [$]
            afrr_cap_revenue               [$]
            mfrr_activation_revenue        [$]
            afrr_activation_revenue        [$]
            imbalance_settlement           [$]   net (signed)
            non_delivery_penalty           [$]   non-negative cost
            total_greek_revenue            [$]
            per_product_per_hour           dict[Product, ndarray]
            n_bids_total                   int
            n_bids_accepted                int

    Notes on activation-energy interpretation
    -----------------------------------------
    A bid for ``mFRR_Energy`` / ``aFRR_Energy`` represents the energy
    you would deliver if fully activated. Realised delivered energy is
    ``awarded × activation_fraction`` in expectation. Phase 2 with
    ``deliveries=None`` uses this expected formula so revenue lines
    match the MILP planner's objective. Phase 3 swaps in actual
    delivered fractions from the plant.
    """
    if activation_fractions is None:
        activation_fractions = {
            Product.mFRR_Energy: 0.10,
            Product.aFRR_Energy: 0.20,
        }
    if system_direction is None:
        system_direction = np.zeros(n_hours, dtype=int)
    system_direction = np.asarray(system_direction, dtype=int)
    if system_direction.shape != (n_hours,):
        raise ValueError(
            f"system_direction shape {system_direction.shape} != ({n_hours},)"
        )

    lam_dam = realized_prices[Product.HEnEx_DAM_Energy]

    # Per-product per-hour revenue accumulators
    revenue_pp_ph: dict[Product, np.ndarray] = {
        p: np.zeros(n_hours) for p in realized_prices
    }
    imbalance_pp_ph: dict[Product, np.ndarray] = {
        p: np.zeros(n_hours) for p in realized_prices
    }
    penalty_pp_ph: dict[Product, np.ndarray] = {
        p: np.zeros(n_hours) for p in realized_prices
    }

    n_bids_total = len(awards)
    n_bids_accepted = sum(1 for a in awards.values() if a.accepted)

    for bid, award in awards.items():
        k = bid.delivery_hour
        if k >= n_hours:
            continue
        spec = product_spec(bid.product)
        prices = realized_prices[bid.product]
        clearing = award.clearing_price_dollar_per_kwh

        if not award.accepted:
            # Unfilled bids contribute nothing — no revenue, no penalty.
            continue

        # Determine delivered kW. For activation-energy products,
        # "delivered" is energy delivered after the realised fraction.
        if deliveries_kw is not None and bid in deliveries_kw:
            delivered = float(deliveries_kw[bid])
        else:
            # Perfect delivery (Phase 2)
            if bid.product in (Product.mFRR_Energy, Product.aFRR_Energy):
                alpha = float(activation_fractions.get(bid.product, 0.0))
                delivered = award.awarded_kw * alpha
            else:
                delivered = award.awarded_kw

        if spec.is_capacity:
            # Capacity payment: $/kW/h × kW × h
            revenue_pp_ph[bid.product][k] += clearing * award.awarded_kw * dt_h

        elif bid.product in (
            Product.HEnEx_DAM_Energy,
            Product.HEnEx_IDM_Energy,
        ):
            # Energy products: signed revenue based on leg direction.
            # SELL leg: revenue = +q × p; BUY leg: revenue = −q × p.
            sign = +1.0 if bid.leg == "sell" else -1.0
            # On the *awarded* (planned) energy.
            revenue_pp_ph[bid.product][k] += (
                sign * award.awarded_kw * clearing * dt_h
            )
            # Imbalance settlement on the deviation. For a sell leg,
            # delivering more than awarded is being long; for a buy leg,
            # delivering more (i.e. consuming more) is being short.
            # Translate buy-leg deltas into the surplus/short convention
            # by flipping the sign before passing to settle_imbalance.
            sell_delivered = delivered if bid.leg == "sell" else -delivered
            sell_awarded = award.awarded_kw if bid.leg == "sell" else -award.awarded_kw
            imbalance_pp_ph[bid.product][k] += settle_imbalance(
                awarded_kw=sell_awarded,
                delivered_kw=sell_delivered,
                system_direction=int(system_direction[k]),
                lambda_marginal_dollar_per_kwh=float(lam_dam[k]),
                k_dual=k_dual,
                dt_h=dt_h,
            )

        elif bid.product in (Product.mFRR_Energy, Product.aFRR_Energy):
            # Activation energy: revenue per delivered kWh.
            revenue_pp_ph[bid.product][k] += clearing * delivered * dt_h
            # Non-delivery penalty: charged against missed kWh, where
            # "missed" = max(0, expected − delivered) and "expected" is
            # the demanded delivery for the hour. Two ways to set
            # expected:
            #   - Phase 3 trace-aware: per-hour realised |activation|
            #     × awarded (passed in via expected_delivery_kw).
            #   - Phase 2 offline: long-run α × awarded.
            if expected_delivery_kw is not None and bid in expected_delivery_kw:
                expected_kw = float(expected_delivery_kw[bid])
            else:
                alpha = float(activation_fractions.get(bid.product, 0.0))
                expected_kw = award.awarded_kw * alpha
            missed = max(0.0, expected_kw - delivered)
            cap_price = float(prices[k])  # activation-price domain for penalty
            penalty_pp_ph[bid.product][k] += (
                non_delivery_penalty_mult * cap_price * missed * dt_h
            )

    # ---- Aggregate per-product totals ----
    def _sum(prod: Product, arr_dict: dict[Product, np.ndarray]) -> float:
        return float(arr_dict.get(prod, np.zeros(n_hours)).sum())

    dam_revenue = _sum(Product.HEnEx_DAM_Energy, revenue_pp_ph)
    idm_revenue = _sum(Product.HEnEx_IDM_Energy, revenue_pp_ph)
    mfrr_cap_revenue = _sum(Product.mFRR_Capacity, revenue_pp_ph)
    afrr_cap_revenue = _sum(Product.aFRR_Capacity, revenue_pp_ph)
    mfrr_act_revenue = _sum(Product.mFRR_Energy, revenue_pp_ph)
    afrr_act_revenue = _sum(Product.aFRR_Energy, revenue_pp_ph)

    imbalance_total = float(sum(arr.sum() for arr in imbalance_pp_ph.values()))
    penalty_total = float(sum(arr.sum() for arr in penalty_pp_ph.values()))

    total = (
        dam_revenue + idm_revenue
        + mfrr_cap_revenue + afrr_cap_revenue
        + mfrr_act_revenue + afrr_act_revenue
        + imbalance_total
        - penalty_total
    )

    return {
        "dam_revenue": dam_revenue,
        "idm_revenue": idm_revenue,
        "mfrr_cap_revenue": mfrr_cap_revenue,
        "afrr_cap_revenue": afrr_cap_revenue,
        "mfrr_activation_revenue": mfrr_act_revenue,
        "afrr_activation_revenue": afrr_act_revenue,
        "imbalance_settlement": imbalance_total,
        "non_delivery_penalty": penalty_total,
        "total_greek_revenue": total,
        "per_product_per_hour": revenue_pp_ph,
        "imbalance_per_product_per_hour": imbalance_pp_ph,
        "penalty_per_product_per_hour": penalty_pp_ph,
        "n_bids_total": n_bids_total,
        "n_bids_accepted": n_bids_accepted,
    }


# ---------------------------------------------------------------------------
#  Phase 3: bridge plant traces into the offline settlement pipeline.
# ---------------------------------------------------------------------------

def _bid_class(bid: Bid) -> str:
    """Group bids by their delivery channel — same channel ⇒ shared
    plant trace, hence proportional-split deliveries.
    """
    if bid.product in (Product.HEnEx_DAM_Energy, Product.HEnEx_IDM_Energy):
        return "discharge_energy" if bid.leg == "sell" else "charge_energy"
    if bid.product in (Product.mFRR_Capacity, Product.aFRR_Capacity):
        return "balancing_capacity"
    if bid.product in (Product.mFRR_Energy, Product.aFRR_Energy):
        return "activation_energy"
    return "other"


def compute_greek_settlement_from_traces(
    traces,                                   # core.simulator.traces.SimTraces
    bidding_protocol,                         # GreekMarketBiddingProtocol
    tp: TimeParams,
) -> dict:
    """Run the offline settlement on plant traces from a finished simulation.

    Per-bid deliveries are derived from PI-step plant traces by the
    proportional-split rule, indexed by **delivery hour** (not by bid
    book) so multiple bid-book emissions covering the same delivery
    hour do not cause double-counting:
      - Group all accepted bids by ``delivery_hour``.
      - For each (delivery_hour, product_class), sum awarded ⇒
        ``total_awarded_class_h``.
      - Aggregate the matching plant trace across PI steps in that
        hour ⇒ ``plant_class_total_h``.
      - Each bid's delivered_kw = ``awarded_kw × plant_total / total_awarded``.

    The rule is exact when the plant tracks the planner (delivered =
    awarded across all bids); produces non-zero imbalance when the
    plant deviates.

    Plant trace mapping
    -------------------
    discharge_energy   ⇐  max(0, p_net_applied)
    charge_energy      ⇐  max(0, -p_net_applied)
    balancing_capacity ⇐  p_reg_committed
    activation_energy  ⇐  |p_delivered|
    """
    engine = bidding_protocol.clearing_engine
    references: dict[Product, np.ndarray] = engine.references
    activation_fractions = bidding_protocol.activation_fractions
    k_dual = bidding_protocol.k_dual

    if not traces.bid_books_per_hour:
        # No bids submitted — nothing to settle.
        return compute_greek_settlement(
            awards={}, realized_prices=references, n_hours=0,
            activation_fractions=activation_fractions, k_dual=k_dual,
        )

    steps_per_hour = int(3600 / tp.dt_pi)

    # Flatten awards across all gate-closure events. With the Phase-3
    # single-gate-closure model there is exactly one event per run,
    # but the loop is robust to multiple (e.g. future IDM re-bidding).
    flat_awards: dict[Bid, Award] = {}
    for awards_h in traces.awards_per_hour:
        flat_awards.update(awards_h)

    # Per-hour realised activation fraction (mean |signal|). Used to
    # set per-bid demanded delivery for activation-energy products,
    # mirroring v5 ledger lines 71-73 (p_demanded = |activation| × p_reg).
    n_hours_signal = traces.n_sim_steps // steps_per_hour
    realised_alpha_per_hour = np.zeros(n_hours_signal)
    for h in range(n_hours_signal):
        s, e = h * steps_per_hour, (h + 1) * steps_per_hour
        if e > s:
            realised_alpha_per_hour[h] = float(np.mean(np.abs(traces.activation[s:e])))

    # Group accepted bids by their delivery_hour. Settlement is
    # delivery-hour-driven so the awards' "submission hour" is
    # irrelevant once we have the awards table.
    bids_by_delivery_hour: dict[int, list[Bid]] = {}
    for bid, award in flat_awards.items():
        if not award.accepted:
            continue
        bids_by_delivery_hour.setdefault(bid.delivery_hour, []).append(bid)

    if not bids_by_delivery_hour:
        return compute_greek_settlement(
            awards=flat_awards, realized_prices=references,
            n_hours=max(len(arr) for arr in references.values()),
            activation_fractions=activation_fractions, k_dual=k_dual,
        )

    # Settlement horizon = max delivery hour + 1, capped to the price
    # array length (which sets the simulator's horizon).
    max_delivery_h = max(bids_by_delivery_hour) + 1
    n_hours_total = max(len(arr) for arr in references.values())
    n_settlement_hours = min(max_delivery_h, n_hours_total)

    # ------------------------------------------------------------------
    # Build deliveries[bid] via proportional split per (class, delivery_hour),
    # AND build expected_delivery[bid] for activation-energy products using
    # the realised per-hour activation fraction (Phase-3 trace-aware path).
    # ------------------------------------------------------------------
    deliveries: dict[Bid, float] = {}
    expected_delivery: dict[Bid, float] = {}
    for h, bids_h in bids_by_delivery_hour.items():
        s, e = h * steps_per_hour, min((h + 1) * steps_per_hour, traces.n_sim_steps)
        if e <= s:
            continue
        p_net_window = traces.power_applied[s:e, 0]
        p_reg_window = traces.p_reg_committed[s:e]
        p_del_window = traces.p_delivered[s:e]

        plant_per_class: dict[str, float] = {
            "discharge_energy":   float(np.mean(np.maximum(p_net_window, 0.0))),
            "charge_energy":      float(np.mean(np.maximum(-p_net_window, 0.0))),
            "balancing_capacity": float(np.mean(p_reg_window)),
            "activation_energy":  float(np.mean(np.abs(p_del_window))),
        }

        awarded_per_class: dict[str, float] = {k: 0.0 for k in plant_per_class}
        bids_by_class: dict[str, list[Bid]] = {k: [] for k in plant_per_class}
        for bid in bids_h:
            cls = _bid_class(bid)
            if cls not in awarded_per_class:
                continue
            awarded_per_class[cls] += flat_awards[bid].awarded_kw
            bids_by_class[cls].append(bid)

        for cls, total_awarded in awarded_per_class.items():
            if total_awarded <= 0.0:
                continue
            plant_total = plant_per_class[cls]
            for bid in bids_by_class[cls]:
                share = flat_awards[bid].awarded_kw / total_awarded
                deliveries[bid] = plant_total * share

        # For activation-energy bids, demanded delivery for the hour =
        # realised |activation| × awarded. This is what the v5 ledger
        # uses as p_demanded; it makes "missed" = 0 if the plant tracks
        # the per-hour signal (the v5 plant does so by construction).
        if h < n_hours_signal:
            alpha_h = float(realised_alpha_per_hour[h])
            for bid in bids_by_class["activation_energy"]:
                expected_delivery[bid] = flat_awards[bid].awarded_kw * alpha_h

    return compute_greek_settlement(
        awards=flat_awards,
        realized_prices=references,
        n_hours=n_settlement_hours,
        deliveries_kw=deliveries,
        expected_delivery_kw=expected_delivery,
        activation_fractions=activation_fractions,
        k_dual=k_dual,
    )
