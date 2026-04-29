"""Multi-product MILP bidding planner — Greek market layer (Phase 1).

The drop-in MILP companion to ``DeterministicLP``. Same ``solve()``
signature, same returned dict shape (so the simulator can swap them
later in Phase 3); the extension points are:

  - **Per-product power decisions.** The single (P_chg, P_dis, P_reg)
    triple in the LP becomes 8 continuous variables per hour split
    across HEnEx DAM Energy, HEnEx IDM Energy, mFRR Capacity, aFRR
    Capacity, mFRR Activation Energy, aFRR Activation Energy.
  - **Binary variables.** Net-direction indicators (b_dis, b_chg) and
    balancing-product participation gates (b_mfrr, b_afrr) — these
    turn the LP into a MILP and are what kill the wash-trade family
    of bugs by construction (see constraint family C1 below).
  - **Greek market-rule encoding.** Minimum bid quantities (MBQ) for
    balancing capacity, gate-closure timing recorded on the emitted
    BidBook, dual-pricing imbalance settled in Phase 2.

Comparison to ``DeterministicLP``
---------------------------------
The LP is already in plan_dict-shape compatible form. The MILP keeps
that contract and additionally returns ``bid_book`` (a BidBook of the
six Greek products) and ``planner_diagnostics`` (solve time, MIP gap,
relaxation flag, n_binaries). The simulator bidding integration in
Phase 3 will consume ``bid_book``; for Phase 1 it is purely planner
output for offline inspection.

Numerical conditioning
----------------------
- **Big-M = P_max_kw** (physical bound). Tight Big-M shrinks the LP
  relaxation gap so branch-and-bound converges fast and the relaxed
  problem is well-conditioned. Generic 1e6 Big-M would blow up
  conditioning by ~10000× on this problem.
- All cost coefficients live in $/kWh / $/kW/h scale (matches the
  existing LP convention) so matrix entries are O(1e-3..1e-1).
- Solver: ``pulp.HiGHS_CMD`` first, ``pulp.PULP_CBC_CMD`` fallback.
  LP relaxation fallback if MIP exceeds ``mip_time_limit_s``.

Returns the same dict shape as ``DeterministicLP.solve`` extended with:
    bid_book               : BidBook (the six-product offer set)
    planner_diagnostics    : dict with mip_solve_time_s, mip_gap,
                             n_binaries, was_relaxed, solver_status
    P_dam_chg_ref/dis_ref  : per-hour kW arrays
    P_idm_chg_ref/dis_ref  :
    P_mfrr_cap_ref         :
    P_afrr_cap_ref         :
    P_mfrr_e_ref           :
    P_afrr_e_ref           :
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pulp

from core.config.parameters import BatteryParams, EMSParams, ThermalParams, TimeParams
from core.markets.bids import Bid, BidBook
from core.markets.products import (
    DEFAULT_PRODUCT_SPECS,
    Product,
    ProductSpec,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketDecomposition:
    """Maps the existing 2-channel forecast (energy, reg-capacity) onto
    the six Greek-market product price arrays.

    Phase 1 uses simple multiplicative / share-based transforms because
    the synthetic dataset has only two price channels. A real HEnEx /
    ADMIE pipeline would replace this with per-product historical
    traces and an empirical ``alpha_*`` activation-fraction calibration.

    Defaults are conservative midpoints from the European balancing
    literature; the README notes them as production-tunable.
    """

    idm_premium: float = 1.05              # IDM = DAM × this (small premium)
    mfrr_cap_share: float = 0.6            # mFRR cap = total reg-cap × this
    afrr_cap_share: float = 0.4            # aFRR cap = total reg-cap × this
    mfrr_act_factor: float = 1.5           # mFRR activation price = energy × this
    afrr_act_factor: float = 1.3           # aFRR activation price = energy × this
    alpha_mfrr: float = 0.10               # expected mFRR activation fraction
    alpha_afrr: float = 0.20               # expected aFRR activation fraction


@dataclass(frozen=True)
class MILPBiddingConfig:
    """Knobs that control MILP behaviour without touching the formulation."""

    mip_time_limit_s: float = 30.0
    mip_rel_gap: float = 1e-4
    enforce_mbq: bool = True               # Phase 1 enforces MBQ for capacity products
    decomposition: MarketDecomposition = field(default_factory=MarketDecomposition)
    product_specs: dict[Product, ProductSpec] = field(
        default_factory=lambda: dict(DEFAULT_PRODUCT_SPECS)
    )


class MILPBiddingPlanner:
    """Greek-market multi-product MILP planner.

    Drop-in replacement for ``DeterministicLP`` in the simulator strategy
    dispatch (signature-compatible). Emits per-product bids in the
    returned plan_dict; the v5 ``P_chg_ref / P_dis_ref / P_reg_ref``
    keys are populated from the per-product solution so existing
    simulator code works unchanged.

    Phase 1: solver-only; standalone CLI runs the planner once on a
    synthetic-day forecast and prints a bid-book summary.
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        ep: EMSParams,
        thp: ThermalParams,
        config: MILPBiddingConfig | None = None,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.ep = ep
        self.thp = thp
        self.cfg = config if config is not None else MILPBiddingConfig()

    # ------------------------------------------------------------------
    #  Public interface — matches DeterministicLP / EconomicEMS
    # ------------------------------------------------------------------

    def solve(
        self,
        soc_init: float,
        soh_init: float,
        t_init: float,
        energy_scenarios: np.ndarray,
        reg_scenarios: np.ndarray,
        probabilities: np.ndarray,
        vrc1_init: float = 0.0,
        vrc2_init: float = 0.0,
    ) -> dict:
        bp = self.bp
        ep = self.ep
        cfg = self.cfg
        dec = cfg.decomposition
        N = int(min(ep.N_ems, energy_scenarios.shape[1]))
        dt_h = self.tp.dt_ems / 3600.0      # = 1.0 for hourly EMS

        # --- Collapse scenarios to expected price (matches DeterministicLP) ---
        w = np.asarray(probabilities, dtype=float)
        e_price = np.asarray(energy_scenarios[:, :N], dtype=float).T @ w   # (N,)
        r_price = np.asarray(reg_scenarios[:, :N], dtype=float).T @ w      # (N,)

        # --- Decompose into the six product price arrays ---
        # All in $/kWh or $/kW/h numerical scale to match the v5 ladder.
        lam_dam = e_price                                   # ($/kWh)
        lam_idm = e_price * dec.idm_premium                 # ($/kWh)
        lam_mfrr_cap = r_price * dec.mfrr_cap_share         # ($/kW/h)
        lam_afrr_cap = r_price * dec.afrr_cap_share         # ($/kW/h)
        lam_mfrr_e = e_price * dec.mfrr_act_factor          # ($/kWh)
        lam_afrr_e = e_price * dec.afrr_act_factor          # ($/kWh)

        # --- Clip initial state to strictly feasible (matches LP/EMS) ---
        soc_init = float(np.clip(soc_init, bp.SOC_min + 1e-3, bp.SOC_max - 1e-3))

        # --- Build the model ---
        prob, vars_, binary_vars = self._build_problem(
            N=N,
            dt_h=dt_h,
            soc_init=soc_init,
            lam_dam=lam_dam,
            lam_idm=lam_idm,
            lam_mfrr_cap=lam_mfrr_cap,
            lam_afrr_cap=lam_afrr_cap,
            lam_mfrr_e=lam_mfrr_e,
            lam_afrr_e=lam_afrr_e,
        )

        # --- Solve (MILP first; LP relaxation fallback if it times out) ---
        was_relaxed, status, solve_time = self._solve(prob, binary_vars)

        if status != "Optimal":
            # "Not Solved", "Infeasible", "Crashed" — any non-Optimal
            # status leaves the solver with infeasible last-iterate
            # values, so we cannot trust pulp.value(...). Return the
            # all-zeros fallback (idle plan) and surface the failure
            # in the diagnostics block instead.
            logger.error("MILPBiddingPlanner failed: status=%s", status)
            fallback = self._fallback_result(N, soc_init, soh_init, t_init)
            fallback["planner_diagnostics"]["solver_status"] = status
            fallback["planner_diagnostics"]["was_relaxed"] = was_relaxed
            fallback["planner_diagnostics"]["mip_solve_time_s"] = solve_time
            return fallback

        # --- Extract solution ---
        sol = self._extract_solution(vars_, N)

        # --- Reconstruct SOC trajectory ---
        soc_ref = self._reconstruct_soc(sol, N, dt_h, soc_init)

        # --- Build the bid book ---
        bid_book = self._build_bid_book(
            sol=sol,
            N=N,
            lam_dam=lam_dam,
            lam_idm=lam_idm,
            lam_mfrr_cap=lam_mfrr_cap,
            lam_afrr_cap=lam_afrr_cap,
            lam_mfrr_e=lam_mfrr_e,
            lam_afrr_e=lam_afrr_e,
        )

        # --- Aggregate into v5-compatible (P_chg_ref, P_dis_ref, P_reg_ref) ---
        # The simulator's existing dispatch pipeline consumes these three
        # arrays. We sum the per-product decisions here so the planner is
        # a true drop-in replacement for DeterministicLP.
        p_chg_ref = sol["P_dam_chg"] + sol["P_idm_chg"]
        p_dis_ref = sol["P_dam_dis"] + sol["P_idm_dis"]
        p_reg_ref = sol["P_mfrr_cap"] + sol["P_afrr_cap"]

        objective = float(pulp.value(prob.objective))
        # We minimise -profit, so profit = -objective.
        expected_profit = -objective

        # pulp 3.x stores binaries internally as cat="Integer" with bounds
        # [0, 1], so `v.cat == pulp.LpBinary` is unreliable. Count from
        # the explicit binary list we tracked at build time.
        n_binaries = len(binary_vars)

        diagnostics = {
            "mip_solve_time_s": solve_time,
            "mip_gap": cfg.mip_rel_gap,
            "n_binaries": n_binaries,
            "was_relaxed": was_relaxed,
            "solver_status": status,
        }

        logger.info(
            "MILPBiddingPlanner solved: objective=$%.2f  |  SOC [%.3f -> %.3f]  "
            "|  N=%d  |  binaries=%d  |  solve=%.2fs  |  relaxed=%s",
            expected_profit, soc_ref[0], soc_ref[-1], N,
            n_binaries, solve_time, was_relaxed,
        )

        return {
            # v5-compatible aggregates (drop-in for DeterministicLP)
            "P_chg_ref": p_chg_ref,
            "P_dis_ref": p_dis_ref,
            "P_reg_ref": p_reg_ref,
            "SOC_ref": soc_ref,
            "SOH_ref": np.full(N + 1, soh_init),
            "TEMP_ref": np.full(N + 1, t_init),
            "VRC1_ref": np.zeros(N + 1),
            "VRC2_ref": np.zeros(N + 1),
            "expected_profit": expected_profit,
            # Greek market layer extensions
            "bid_book": bid_book,
            "planner_diagnostics": diagnostics,
            "P_dam_chg_ref": sol["P_dam_chg"],
            "P_dam_dis_ref": sol["P_dam_dis"],
            "P_idm_chg_ref": sol["P_idm_chg"],
            "P_idm_dis_ref": sol["P_idm_dis"],
            "P_mfrr_cap_ref": sol["P_mfrr_cap"],
            "P_afrr_cap_ref": sol["P_afrr_cap"],
            "P_mfrr_e_ref": sol["P_mfrr_e"],
            "P_afrr_e_ref": sol["P_afrr_e"],
        }

    # ------------------------------------------------------------------
    #  MILP construction
    # ------------------------------------------------------------------

    def _build_problem(
        self,
        N: int,
        dt_h: float,
        soc_init: float,
        lam_dam: np.ndarray,
        lam_idm: np.ndarray,
        lam_mfrr_cap: np.ndarray,
        lam_afrr_cap: np.ndarray,
        lam_mfrr_e: np.ndarray,
        lam_afrr_e: np.ndarray,
    ) -> tuple[pulp.LpProblem, dict, list[pulp.LpVariable]]:
        bp = self.bp
        ep = self.ep
        cfg = self.cfg
        dec = cfg.decomposition
        eta_c = bp.eta_charge
        eta_d = bp.eta_discharge
        E_nom = bp.E_nom_kwh
        P_max = bp.P_max_kw
        big_M = P_max          # Tight Big-M = physical bound

        # Mirrors DeterministicLP's reg_drift_coef logic, but applied to
        # the *combined* balancing capacity (mfrr_cap + afrr_cap). We use
        # an effective alpha that blends per-product activation fractions
        # weighted by the share split, so the SOC drift matches what the
        # plant will actually experience under the decomposition.
        eta_loss = eta_c - 1.0 / eta_d                              # < 0
        alpha_eff = (
            dec.mfrr_cap_share * dec.alpha_mfrr
            + dec.afrr_cap_share * dec.alpha_afrr
        )
        reg_drift_coef = eta_loss * alpha_eff * dt_h / E_nom

        # ------------------------------------------------------------------
        # Variables
        # ------------------------------------------------------------------
        prob = pulp.LpProblem("MILP_Greek_Bidding", pulp.LpMinimize)

        def cont_vec(name: str, ub: float) -> list[pulp.LpVariable]:
            return [
                pulp.LpVariable(f"{name}[{k}]", lowBound=0.0, upBound=ub)
                for k in range(N)
            ]

        def bin_vec(name: str) -> list[pulp.LpVariable]:
            return [pulp.LpVariable(f"{name}[{k}]", cat=pulp.LpBinary) for k in range(N)]

        # Per-product continuous powers (kW)
        P_dam_dis = cont_vec("P_dam_dis", P_max)
        P_dam_chg = cont_vec("P_dam_chg", P_max)
        P_idm_dis = cont_vec("P_idm_dis", P_max)
        P_idm_chg = cont_vec("P_idm_chg", P_max)
        P_mfrr_cap = cont_vec("P_mfrr_cap", P_max)
        P_afrr_cap = cont_vec("P_afrr_cap", P_max)
        P_mfrr_e = cont_vec("P_mfrr_e", P_max)
        P_afrr_e = cont_vec("P_afrr_e", P_max)

        # Binaries — also collected into a flat list so the LP-relaxation
        # fallback in _solve() can flip them reliably (pulp 3.x's `cat`
        # attribute reports "Integer" for binaries, so we cannot identify
        # them after-the-fact via cat alone).
        b_dis = bin_vec("b_dis")
        b_chg = bin_vec("b_chg")
        b_mfrr = bin_vec("b_mfrr")
        b_afrr = bin_vec("b_afrr")
        binary_vars: list[pulp.LpVariable] = [*b_dis, *b_chg, *b_mfrr, *b_afrr]

        # SOC trajectory variables (k = 0..N)
        SOC = [
            pulp.LpVariable(f"SOC[{k}]", lowBound=bp.SOC_min, upBound=bp.SOC_max)
            for k in range(N + 1)
        ]

        # Slacks
        z_plus = pulp.LpVariable("z_plus", lowBound=0.0)
        z_minus = pulp.LpVariable("z_minus", lowBound=0.0)
        eps_end = [pulp.LpVariable(f"eps_end[{k}]", lowBound=0.0) for k in range(N)]

        # ------------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------------

        # SOC[0] = soc_init (equality)
        prob += SOC[0] == soc_init, "soc_init"

        for k in range(N):
            # (C1) Wash-trade-free by construction.
            #   At most one of {discharge mode, charge mode} is active per hour.
            #   With Big-M = P_max, the bound is tight (LP relaxation = MILP).
            prob += (
                P_dam_dis[k] + P_idm_dis[k] <= big_M * b_dis[k],
                f"dis_indicator_{k}",
            )
            prob += (
                P_dam_chg[k] + P_idm_chg[k] <= big_M * b_chg[k],
                f"chg_indicator_{k}",
            )
            prob += b_dis[k] + b_chg[k] <= 1, f"mutex_{k}"

            # (C3) Power budget per direction. Reserves capacity headroom
            #      for both balancing products on top of the energy bid
            #      because aFRR/mFRR can activate in either direction.
            prob += (
                P_dam_dis[k] + P_idm_dis[k] + P_mfrr_cap[k] + P_afrr_cap[k] <= P_max,
                f"power_budget_dis_{k}",
            )
            prob += (
                P_dam_chg[k] + P_idm_chg[k] + P_mfrr_cap[k] + P_afrr_cap[k] <= P_max,
                f"power_budget_chg_{k}",
            )

            # (C4) Greek MBQ floors on balancing capacity (Phase 1).
            #      Forces P in {0} or P >= MBQ via the indicator binary.
            if cfg.enforce_mbq:
                mbq_mfrr = cfg.product_specs[Product.mFRR_Capacity].min_bid_qty_kw
                mbq_afrr = cfg.product_specs[Product.aFRR_Capacity].min_bid_qty_kw
                prob += P_mfrr_cap[k] >= mbq_mfrr * b_mfrr[k], f"mbq_mfrr_lo_{k}"
                prob += P_mfrr_cap[k] <= P_max * b_mfrr[k], f"mbq_mfrr_hi_{k}"
                prob += P_afrr_cap[k] >= mbq_afrr * b_afrr[k], f"mbq_afrr_lo_{k}"
                prob += P_afrr_cap[k] <= P_max * b_afrr[k], f"mbq_afrr_hi_{k}"

            # Activation-energy bid <= committed capacity (you can only
            # activate as much as you committed).
            prob += P_mfrr_e[k] <= P_mfrr_cap[k], f"act_lim_mfrr_{k}"
            prob += P_afrr_e[k] <= P_afrr_cap[k], f"act_lim_afrr_{k}"

            # (C5) SOC dynamics. Mirrors DeterministicLP:236-243 but with
            #      energy decisions split over DAM + IDM. P_*_e enter the
            #      SOC recursion as expected delivered energy at fraction
            #      alpha_*: aFRR activation drains/refills symmetrically
            #      (uses eta_loss); mFRR is UP-only so it's a strict
            #      discharge in expectation.
            net_chg = (
                eta_c * (P_dam_chg[k] + P_idm_chg[k])
                - (P_dam_dis[k] + P_idm_dis[k]) / eta_d
            )
            prob += (
                SOC[k + 1] == SOC[k]
                + (dt_h / E_nom) * net_chg
                + reg_drift_coef * (P_mfrr_cap[k] + P_afrr_cap[k]),
                f"soc_dynamics_{k}",
            )

            # (C6) Endurance for committed balancing capacity.
            #      Same form as DeterministicLP:232-254, applied to
            #      (P_mfrr_cap + P_afrr_cap). Soft via eps_end.
            #      SOC[k+1] >= SOC_min + endurance * P_cap / (E_nom * eta_d) - eps_end
            #      SOC[k+1] <= SOC_max - endurance * P_cap * eta_c / E_nom + eps_end
            endurance_h = ep.endurance_hours
            prob += (
                SOC[k + 1] >= bp.SOC_min
                + endurance_h * (P_mfrr_cap[k] + P_afrr_cap[k]) / (E_nom * eta_d)
                - eps_end[k],
                f"endurance_lo_{k}",
            )
            prob += (
                SOC[k + 1] <= bp.SOC_max
                - endurance_h * (P_mfrr_cap[k] + P_afrr_cap[k]) * eta_c / E_nom
                + eps_end[k],
                f"endurance_hi_{k}",
            )

        # (C7) Terminal SOC anchor (soft L1).
        prob += SOC[N] - bp.SOC_terminal == z_plus - z_minus, "terminal_anchor"

        # ------------------------------------------------------------------
        # Objective: minimise (-profit + slack penalties).
        # Profit components in $/kWh / $/kW/h scale (matches LP convention).
        # ------------------------------------------------------------------
        deg_unit = ep.degradation_cost * self.tp.dt_ems   # $/kW per LP step
        TERMINAL_W = 50.0 * E_nom
        ENDURANCE_W = 10.0 * TERMINAL_W

        revenue = pulp.LpAffineExpression()
        for k in range(N):
            # DAM / IDM energy (signed: dis - chg)
            revenue += lam_dam[k] * dt_h * (P_dam_dis[k] - P_dam_chg[k])
            revenue += lam_idm[k] * dt_h * (P_idm_dis[k] - P_idm_chg[k])
            # Capacity commitment payments
            revenue += lam_mfrr_cap[k] * dt_h * P_mfrr_cap[k]
            revenue += lam_afrr_cap[k] * dt_h * P_afrr_cap[k]
            # Activation energy (expected delivery = alpha × bid quantity)
            revenue += lam_mfrr_e[k] * dt_h * dec.alpha_mfrr * P_mfrr_e[k]
            revenue += lam_afrr_e[k] * dt_h * dec.alpha_afrr * P_afrr_e[k]

        deg_cost = pulp.LpAffineExpression()
        for k in range(N):
            throughput = (
                P_dam_chg[k] + P_dam_dis[k] + P_idm_chg[k] + P_idm_dis[k]
            )
            deg_cost += deg_unit * bp.alpha_deg * throughput
            deg_cost += deg_unit * bp.alpha_deg_reg * (P_mfrr_cap[k] + P_afrr_cap[k])

        slack_cost = TERMINAL_W * (z_plus + z_minus)
        for k in range(N):
            slack_cost += ENDURANCE_W * eps_end[k]

        prob += -revenue + deg_cost + slack_cost

        vars_ = {
            "P_dam_dis": P_dam_dis,
            "P_dam_chg": P_dam_chg,
            "P_idm_dis": P_idm_dis,
            "P_idm_chg": P_idm_chg,
            "P_mfrr_cap": P_mfrr_cap,
            "P_afrr_cap": P_afrr_cap,
            "P_mfrr_e": P_mfrr_e,
            "P_afrr_e": P_afrr_e,
            "b_dis": b_dis,
            "b_chg": b_chg,
            "b_mfrr": b_mfrr,
            "b_afrr": b_afrr,
            "SOC": SOC,
            "z_plus": z_plus,
            "z_minus": z_minus,
            "eps_end": eps_end,
        }
        return prob, vars_, binary_vars

    # ------------------------------------------------------------------
    #  Solver dispatch with LP-relaxation fallback
    # ------------------------------------------------------------------

    def _solve(
        self,
        prob: pulp.LpProblem,
        binary_vars: list[pulp.LpVariable],
    ) -> tuple[bool, str, float]:
        cfg = self.cfg
        t0 = time.perf_counter()

        solver = self._make_solver(time_limit_s=cfg.mip_time_limit_s)
        try:
            prob.solve(solver)
            status = pulp.LpStatus[prob.status]
        except Exception as exc:                         # noqa: BLE001 — defensive
            logger.warning("MILP solver crashed (%s); falling back to LP relaxation.", exc)
            status = "Crashed"

        solve_time = time.perf_counter() - t0
        was_relaxed = False

        # If MILP failed to find an optimum within budget, relax the
        # tracked binaries and re-solve. The LP relaxation is always
        # feasible if the MILP was.
        if status != "Optimal":
            logger.warning(
                "MILP non-optimal (status=%s, %.2fs). Relaxing binaries and re-solving.",
                status, solve_time,
            )
            was_relaxed = True
            for v in binary_vars:
                v.cat = pulp.LpContinuous
                v.lowBound = 0.0
                v.upBound = 1.0

            # Give the LP relaxation a generous budget. It's a pure
            # LP — no integer search — and finishes in tens of ms on
            # this problem size. Inheriting the tight MILP budget
            # would risk the relaxation timing out too, leaving the
            # solver with infeasible last-iterate values.
            relax_time_limit = max(cfg.mip_time_limit_s * 100.0, 30.0)

            t1 = time.perf_counter()
            try:
                prob.solve(self._make_solver(time_limit_s=relax_time_limit))
                status = pulp.LpStatus[prob.status]
            except Exception as exc:                     # noqa: BLE001
                logger.error("LP relaxation also crashed: %s", exc)
                status = "Crashed"
            solve_time += time.perf_counter() - t1

        return was_relaxed, status, solve_time

    @staticmethod
    def _make_solver(time_limit_s: float) -> pulp.LpSolver:
        """Prefer HiGHS; fall back to CBC if HiGHS isn't reachable."""
        # HiGHS_CMD requires the `highs` CLI on PATH; pulp's HiGHS class
        # uses highspy if installed. Try them in order.
        candidates: list[pulp.LpSolver] = []
        if hasattr(pulp, "HiGHS"):
            candidates.append(pulp.HiGHS(timeLimit=time_limit_s, msg=False))
        candidates.append(
            pulp.PULP_CBC_CMD(timeLimit=time_limit_s, msg=False)
        )
        for solver in candidates:
            if solver.available():
                return solver
        raise RuntimeError(
            "No MILP solver available. Install highspy (`pip install highspy`) "
            "or ensure CBC is on PATH."
        )

    # ------------------------------------------------------------------
    #  Solution extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_solution(vars_: dict, N: int) -> dict[str, np.ndarray]:
        def vec(name: str) -> np.ndarray:
            return np.array(
                [pulp.value(vars_[name][k]) or 0.0 for k in range(N)],
                dtype=float,
            )
        return {
            "P_dam_dis": vec("P_dam_dis"),
            "P_dam_chg": vec("P_dam_chg"),
            "P_idm_dis": vec("P_idm_dis"),
            "P_idm_chg": vec("P_idm_chg"),
            "P_mfrr_cap": vec("P_mfrr_cap"),
            "P_afrr_cap": vec("P_afrr_cap"),
            "P_mfrr_e": vec("P_mfrr_e"),
            "P_afrr_e": vec("P_afrr_e"),
            "b_dis": vec("b_dis"),
            "b_chg": vec("b_chg"),
            "b_mfrr": vec("b_mfrr"),
            "b_afrr": vec("b_afrr"),
        }

    def _reconstruct_soc(
        self,
        sol: dict[str, np.ndarray],
        N: int,
        dt_h: float,
        soc_init: float,
    ) -> np.ndarray:
        bp = self.bp
        dec = self.cfg.decomposition
        eta_c = bp.eta_charge
        eta_d = bp.eta_discharge
        E_nom = bp.E_nom_kwh
        eta_loss = eta_c - 1.0 / eta_d
        alpha_eff = (
            dec.mfrr_cap_share * dec.alpha_mfrr
            + dec.afrr_cap_share * dec.alpha_afrr
        )
        reg_drift_coef = eta_loss * alpha_eff * dt_h / E_nom

        soc_ref = np.zeros(N + 1)
        soc_ref[0] = soc_init
        for k in range(N):
            net_chg = (
                eta_c * (sol["P_dam_chg"][k] + sol["P_idm_chg"][k])
                - (sol["P_dam_dis"][k] + sol["P_idm_dis"][k]) / eta_d
            )
            soc_ref[k + 1] = (
                soc_ref[k]
                + (dt_h / E_nom) * net_chg
                + reg_drift_coef * (sol["P_mfrr_cap"][k] + sol["P_afrr_cap"][k])
            )
        return soc_ref

    # ------------------------------------------------------------------
    #  Bid-book construction
    # ------------------------------------------------------------------

    def _build_bid_book(
        self,
        sol: dict[str, np.ndarray],
        N: int,
        lam_dam: np.ndarray,
        lam_idm: np.ndarray,
        lam_mfrr_cap: np.ndarray,
        lam_afrr_cap: np.ndarray,
        lam_mfrr_e: np.ndarray,
        lam_afrr_e: np.ndarray,
    ) -> BidBook:
        """Translate the MILP solution into a BidBook for the clearing engine.

        Bid prices use the forecast (probability-weighted mean) for that
        product as the *willingness-to-trade* anchor. A real strategy would
        adjust bid prices for risk aversion / market-power considerations;
        for Phase 1 we bid at the forecast price, which yields all-or-nothing
        clearing under the reference-price stub.
        """
        book = BidBook()

        TOL = 1e-3   # kW threshold below which we consider the leg inactive

        for k in range(N):
            # DAM energy (split charge / discharge legs; only emit the
            # leg that the optimiser actually uses — guaranteed non-overlap
            # by constraint C1).
            if sol["P_dam_dis"][k] > TOL:
                book.add(Bid(
                    product=Product.HEnEx_DAM_Energy,
                    delivery_hour=k,
                    quantity_kw=float(sol["P_dam_dis"][k]),
                    price_dollar_per_kwh=float(lam_dam[k]),
                    leg="sell",
                ))
            if sol["P_dam_chg"][k] > TOL:
                book.add(Bid(
                    product=Product.HEnEx_DAM_Energy,
                    delivery_hour=k,
                    quantity_kw=float(sol["P_dam_chg"][k]),
                    price_dollar_per_kwh=float(lam_dam[k]),
                    leg="buy",
                ))
            # IDM
            if sol["P_idm_dis"][k] > TOL:
                book.add(Bid(
                    product=Product.HEnEx_IDM_Energy,
                    delivery_hour=k,
                    quantity_kw=float(sol["P_idm_dis"][k]),
                    price_dollar_per_kwh=float(lam_idm[k]),
                    leg="sell",
                ))
            if sol["P_idm_chg"][k] > TOL:
                book.add(Bid(
                    product=Product.HEnEx_IDM_Energy,
                    delivery_hour=k,
                    quantity_kw=float(sol["P_idm_chg"][k]),
                    price_dollar_per_kwh=float(lam_idm[k]),
                    leg="buy",
                ))
            # Balancing capacity
            if sol["P_mfrr_cap"][k] > TOL:
                book.add(Bid(
                    product=Product.mFRR_Capacity,
                    delivery_hour=k,
                    quantity_kw=float(sol["P_mfrr_cap"][k]),
                    price_dollar_per_kwh=float(lam_mfrr_cap[k]),
                    leg="sell",
                ))
            if sol["P_afrr_cap"][k] > TOL:
                book.add(Bid(
                    product=Product.aFRR_Capacity,
                    delivery_hour=k,
                    quantity_kw=float(sol["P_afrr_cap"][k]),
                    price_dollar_per_kwh=float(lam_afrr_cap[k]),
                    leg="sell",
                ))
            # Activation energy (UP-only for mFRR, symmetric for aFRR)
            if sol["P_mfrr_e"][k] > TOL:
                book.add(Bid(
                    product=Product.mFRR_Energy,
                    delivery_hour=k,
                    quantity_kw=float(sol["P_mfrr_e"][k]),
                    price_dollar_per_kwh=float(lam_mfrr_e[k]),
                    leg="sell",
                ))
            if sol["P_afrr_e"][k] > TOL:
                book.add(Bid(
                    product=Product.aFRR_Energy,
                    delivery_hour=k,
                    quantity_kw=float(sol["P_afrr_e"][k]),
                    price_dollar_per_kwh=float(lam_afrr_e[k]),
                    leg="sell",
                ))
        return book

    # ------------------------------------------------------------------
    #  Fallback for solver failure (mirrors DeterministicLP._fallback_result)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_result(N: int, soc_init: float, soh_init: float, t_init: float) -> dict:
        return {
            "P_chg_ref": np.zeros(N),
            "P_dis_ref": np.zeros(N),
            "P_reg_ref": np.zeros(N),
            "SOC_ref": np.full(N + 1, soc_init),
            "SOH_ref": np.full(N + 1, soh_init),
            "TEMP_ref": np.full(N + 1, t_init),
            "VRC1_ref": np.zeros(N + 1),
            "VRC2_ref": np.zeros(N + 1),
            "expected_profit": 0.0,
            "bid_book": BidBook(),
            "planner_diagnostics": {
                "mip_solve_time_s": 0.0,
                "mip_gap": float("nan"),
                "n_binaries": 0,
                "was_relaxed": False,
                "solver_status": "Failed",
            },
            "P_dam_chg_ref": np.zeros(N),
            "P_dam_dis_ref": np.zeros(N),
            "P_idm_chg_ref": np.zeros(N),
            "P_idm_dis_ref": np.zeros(N),
            "P_mfrr_cap_ref": np.zeros(N),
            "P_afrr_cap_ref": np.zeros(N),
            "P_mfrr_e_ref": np.zeros(N),
            "P_afrr_e_ref": np.zeros(N),
        }


# ---------------------------------------------------------------------------
#  CLI smoke test (Phase 1)
# ---------------------------------------------------------------------------

def _format_bid_book_summary(book: BidBook) -> str:
    """Human-readable bid-book summary by product."""
    from collections import defaultdict
    agg: dict[Product, dict] = defaultdict(
        lambda: {"n_hours": 0, "total_kw": 0.0, "total_kwh_or_kwh_eq": 0.0, "wprice_num": 0.0}
    )
    for b in book:
        a = agg[b.product]
        a["n_hours"] += 1
        a["total_kw"] += b.quantity_kw
        a["wprice_num"] += b.quantity_kw * b.price_dollar_per_kwh
    lines = [f"  {'product':<22s}  {'n_hr':>4s}  {'total_kw':>10s}  {'avg_price_$/kwh':>16s}"]
    lines.append("  " + "-" * 60)
    for p, a in sorted(agg.items(), key=lambda kv: kv[0].value):
        avg_price = (a["wprice_num"] / a["total_kw"]) if a["total_kw"] > 0 else float("nan")
        lines.append(
            f"  {p.value:<22s}  {a['n_hours']:>4d}  {a['total_kw']:>10.2f}  {avg_price:>16.5f}"
        )
    if not agg:
        lines.append("  (empty bid book)")
    return "\n".join(lines)


def _plot_phase1(
    result: dict,
    bp: BatteryParams,
    ep: EMSParams,
    cfg: MILPBiddingConfig,
    forecast_e: np.ndarray,
    forecast_r: np.ndarray,
    probabilities: np.ndarray,
    out_path: str,
) -> None:
    """Diagnostic 4-panel figure for the MILP Phase 1 solve.

    Panels:
      (a) Per-hour power allocation by Greek product (stacked bars,
          discharge positive / charge negative).
      (b) Per-hour balancing capacity commitment (mFRR + aFRR).
      (c) SOC trajectory with bounds, terminal target, and endurance
          headroom shading.
      (d) Decomposed Greek-market product prices used by the optimiser
          (DAM, IDM, mFRR_cap, aFRR_cap, mFRR_act, aFRR_act).
    """
    import matplotlib.pyplot as plt

    dec = cfg.decomposition
    N = ep.N_ems
    hours = np.arange(N)

    # Recover decomposed prices (same recipe as solve())
    w = np.asarray(probabilities, dtype=float)
    e_price = np.asarray(forecast_e[:, :N]).T @ w
    r_price = np.asarray(forecast_r[:, :N]).T @ w
    lam_dam = e_price
    lam_idm = e_price * dec.idm_premium
    lam_mfrr_cap = r_price * dec.mfrr_cap_share
    lam_afrr_cap = r_price * dec.afrr_cap_share
    lam_mfrr_e = e_price * dec.mfrr_act_factor
    lam_afrr_e = e_price * dec.afrr_act_factor

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"MILP Bidding Planner — Phase 1 smoke test  "
        f"(profit \\${result['expected_profit']:.2f}, "
        f"{result['planner_diagnostics']['mip_solve_time_s']*1000:.0f} ms, "
        f"{result['planner_diagnostics']['n_binaries']} binaries)",
        fontsize=13,
    )

    # (a) Energy product allocation: signed (discharge=+, charge=-).
    ax = axs[0, 0]
    dam_signed = result["P_dam_dis_ref"] - result["P_dam_chg_ref"]
    idm_signed = result["P_idm_dis_ref"] - result["P_idm_chg_ref"]
    width = 0.8
    ax.bar(hours, dam_signed, width=width, label="HEnEx_DAM_Energy", color="#1f77b4")
    ax.bar(hours, idm_signed, width=width, bottom=dam_signed,
           label="HEnEx_IDM_Energy", color="#ff7f0e")
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.axhline(bp.P_max_kw, color="r", linestyle=":", linewidth=0.8, label=f"+P_max={bp.P_max_kw}")
    ax.axhline(-bp.P_max_kw, color="r", linestyle=":", linewidth=0.8)
    ax.set_ylabel("kW  (+ discharge, − charge)")
    ax.set_xlabel("hour of day")
    ax.set_title("(a) Energy bids per hour — DAM + IDM")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xticks(hours[::2])
    ax.grid(alpha=0.3)

    # (b) Balancing-capacity commitment (always positive — symmetric reserve)
    ax = axs[0, 1]
    mfrr = result["P_mfrr_cap_ref"]
    afrr = result["P_afrr_cap_ref"]
    ax.bar(hours, mfrr, width=width, label="mFRR_Capacity", color="#2ca02c")
    ax.bar(hours, afrr, width=width, bottom=mfrr,
           label="aFRR_Capacity", color="#d62728")
    # Activation-energy bids overlaid as hatching
    ax.bar(hours, result["P_mfrr_e_ref"], width=width * 0.5,
           bottom=0, color="none", edgecolor="#2ca02c", hatch="///",
           label="mFRR_Energy bid")
    ax.bar(hours, result["P_afrr_e_ref"], width=width * 0.5,
           bottom=mfrr, color="none", edgecolor="#d62728", hatch="\\\\\\",
           label="aFRR_Energy bid")
    ax.axhline(bp.P_max_kw, color="r", linestyle=":", linewidth=0.8)
    if cfg.enforce_mbq:
        mbq = cfg.product_specs[Product.mFRR_Capacity].min_bid_qty_kw
        ax.axhline(mbq, color="grey", linestyle="--", linewidth=0.8,
                   label=f"MBQ={mbq:.0f} kW")
    ax.set_ylabel("kW")
    ax.set_xlabel("hour of day")
    ax.set_title("(b) Balancing capacity + activation bids")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xticks(hours[::2])
    ax.grid(alpha=0.3)

    # (c) SOC trajectory
    ax = axs[1, 0]
    soc_hours = np.arange(N + 1)
    ax.plot(soc_hours, result["SOC_ref"], "o-", color="#1f77b4",
            label="SOC trajectory")
    ax.axhline(bp.SOC_min, color="r", linestyle=":", label=f"SOC_min={bp.SOC_min}")
    ax.axhline(bp.SOC_max, color="r", linestyle=":", label=f"SOC_max={bp.SOC_max}")
    ax.axhline(bp.SOC_terminal, color="g", linestyle="--",
               label=f"SOC_terminal={bp.SOC_terminal}")
    ax.scatter([N], [bp.SOC_terminal], marker="*", color="g", s=200, zorder=5)
    ax.set_ylabel("SOC")
    ax.set_xlabel("hour")
    ax.set_title("(c) SOC trajectory + bounds + terminal anchor")
    ax.legend(loc="best", fontsize=9)
    ax.set_xticks(soc_hours[::2])
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)

    # (d) Decomposed product prices
    ax = axs[1, 1]
    ax.plot(hours, lam_dam, "-", label="λ_DAM", color="#1f77b4", linewidth=1.5)
    ax.plot(hours, lam_idm, "--", label="λ_IDM", color="#ff7f0e", linewidth=1.5)
    ax.plot(hours, lam_mfrr_e, ":", label="λ_mFRR_act", color="#2ca02c", linewidth=1.5)
    ax.plot(hours, lam_afrr_e, ":", label="λ_aFRR_act", color="#d62728", linewidth=1.5)
    ax.set_ylabel("\\$/kWh  (energy)")
    ax.set_xlabel("hour of day")
    ax.set_title("(d) Decomposed Greek-market product prices")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(hours, lam_mfrr_cap, "-.", label="λ_mFRR_cap",
             color="#2ca02c", linewidth=1.0, alpha=0.7)
    ax2.plot(hours, lam_afrr_cap, "-.", label="λ_aFRR_cap",
             color="#d62728", linewidth=1.0, alpha=0.7)
    ax2.set_ylabel("\\$/kW/h  (capacity)")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {out_path}")


def _plot_milp_vs_lp(
    milp_result: dict,
    lp_result: dict,
    bp: BatteryParams,
    out_path: str,
) -> None:
    """Side-by-side MILP-vs-DeterministicLP comparison.

    Two panels:
      (a) MILP per-product power allocation (signed) per hour.
      (b) DeterministicLP single P_chg/P_dis/P_reg breakdown per hour.

    Visual point: the MILP can express what the LP cannot.
    """
    import matplotlib.pyplot as plt

    N = len(milp_result["P_dam_dis_ref"])
    hours = np.arange(N)
    width = 0.8

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(
        f"Power allocation per hour — MILP (\\${milp_result['expected_profit']:.2f}) "
        f"vs DeterministicLP (\\${lp_result['expected_profit']:.2f})",
        fontsize=13,
    )

    # (a) MILP — six-product split
    ax = axs[0]
    dam_signed = milp_result["P_dam_dis_ref"] - milp_result["P_dam_chg_ref"]
    idm_signed = milp_result["P_idm_dis_ref"] - milp_result["P_idm_chg_ref"]
    mfrr = milp_result["P_mfrr_cap_ref"]
    afrr = milp_result["P_afrr_cap_ref"]

    ax.bar(hours, dam_signed, width=width, label="HEnEx_DAM_Energy", color="#1f77b4")
    ax.bar(hours, idm_signed, width=width, bottom=dam_signed,
           label="HEnEx_IDM_Energy", color="#ff7f0e")
    # capacity is reserved (not delivered net energy) — show above the
    # energy stack as a separate "reserved" tier
    energy_top = np.maximum(dam_signed + idm_signed, 0)
    ax.bar(hours, mfrr, width=width, bottom=energy_top,
           label="mFRR_Capacity", color="#2ca02c", alpha=0.85)
    ax.bar(hours, afrr, width=width, bottom=energy_top + mfrr,
           label="aFRR_Capacity", color="#d62728", alpha=0.85)

    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.axhline(bp.P_max_kw, color="r", linestyle=":", linewidth=0.8,
               label=f"P_max={bp.P_max_kw}")
    ax.axhline(-bp.P_max_kw, color="r", linestyle=":", linewidth=0.8)
    ax.set_ylabel("kW (energy: + dis / − chg; capacity: stacked above)")
    ax.set_xlabel("hour of day")
    ax.set_title("MILP — six Greek products")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xticks(hours[::2])
    ax.grid(alpha=0.3)

    # (b) DeterministicLP — three buckets
    ax = axs[1]
    lp_signed = lp_result["P_dis_ref"] - lp_result["P_chg_ref"]
    lp_reg = lp_result["P_reg_ref"]
    energy_top_lp = np.maximum(lp_signed, 0)

    ax.bar(hours, lp_signed, width=width, label="P_dis − P_chg (energy)",
           color="#1f77b4")
    ax.bar(hours, lp_reg, width=width, bottom=energy_top_lp,
           label="P_reg (FCR capacity)", color="#9467bd", alpha=0.85)

    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.axhline(bp.P_max_kw, color="r", linestyle=":", linewidth=0.8,
               label=f"P_max={bp.P_max_kw}")
    ax.axhline(-bp.P_max_kw, color="r", linestyle=":", linewidth=0.8)
    ax.set_xlabel("hour of day")
    ax.set_title("DeterministicLP — single energy + FCR")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xticks(hours[::2])
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {out_path}")


def _cli_main() -> int:  # pragma: no cover — manual smoke test entrypoint
    """Solve the MILP once on the canonical 1-day synthetic dataset."""
    import argparse
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="MILP bidding planner smoke test")
    parser.add_argument("--solve", action="store_true", help="Run a single MILP solve")
    parser.add_argument("--time-limit", type=float, default=30.0, help="MIP time limit [s]")
    parser.add_argument("--no-mbq", action="store_true", help="Disable MBQ floors")
    parser.add_argument("--plot", action="store_true",
                        help="Save diagnostic plots to results/")
    args = parser.parse_args()

    if not args.solve:
        parser.print_help()
        return 0

    from core.simulator.synthetic_day import make_synthetic_day

    bp = BatteryParams()
    tp = TimeParams()
    ep = EMSParams()
    thp = ThermalParams()
    cfg = MILPBiddingConfig(
        mip_time_limit_s=args.time_limit,
        enforce_mbq=not args.no_mbq,
    )

    planner = MILPBiddingPlanner(bp=bp, tp=tp, ep=ep, thp=thp, config=cfg)
    day = make_synthetic_day()

    print("=" * 72)
    print("MILP Bidding Planner — Phase 1 smoke test")
    print("=" * 72)
    print(f"Battery:  E_nom={bp.E_nom_kwh} kWh, P_max={bp.P_max_kw} kW, "
          f"SOC[0]={bp.SOC_init:.2f}, SOC_term={bp.SOC_terminal:.2f}")
    print(f"Horizon:  N={ep.N_ems} hours")
    print(f"MBQ:      {'enforced' if cfg.enforce_mbq else 'disabled'}")
    print()

    result = planner.solve(
        soc_init=bp.SOC_init,
        soh_init=bp.SOH_init,
        t_init=thp.T_init,
        energy_scenarios=day.forecast_e,
        reg_scenarios=day.forecast_r,
        probabilities=day.probabilities,
    )

    diag = result["planner_diagnostics"]
    print()
    print(f"Solver status   : {diag['solver_status']}")
    print(f"Was relaxed     : {diag['was_relaxed']}")
    print(f"MIP solve time  : {diag['mip_solve_time_s']:.3f} s")
    print(f"# binaries      : {diag['n_binaries']}")
    print(f"Expected profit : ${result['expected_profit']:.2f}")
    print(f"SOC trajectory  : {result['SOC_ref'][0]:.3f} -> {result['SOC_ref'][-1]:.3f}")
    print()
    print("Bid book summary (per product):")
    print(_format_bid_book_summary(result["bid_book"]))
    print()

    # Wash-trade sanity check (this is what the binary mutex enforces)
    soln_chg = result["P_dam_chg_ref"] + result["P_idm_chg_ref"]
    soln_dis = result["P_dam_dis_ref"] + result["P_idm_dis_ref"]
    overlap = np.minimum(soln_chg, soln_dis)
    max_overlap = float(np.max(overlap))
    print(f"Wash-trade check: max(min(chg, dis)) = {max_overlap:.6f} kW  "
          f"({'PASS' if max_overlap < 1e-3 else 'FAIL'})")

    # ----- Phase 2: clear the MILP's bid book against realised prices -----
    from core.accounting.greek_settlement import compute_greek_settlement
    from core.markets.clearing import ReferencePriceClearingStub, decompose_prices

    realized_prices = decompose_prices(
        energy_per_hour=day.realized_e_prices,
        reg_per_hour=day.realized_r_prices,
        idm_premium=cfg.decomposition.idm_premium,
        mfrr_cap_share=cfg.decomposition.mfrr_cap_share,
        afrr_cap_share=cfg.decomposition.afrr_cap_share,
        mfrr_act_factor=cfg.decomposition.mfrr_act_factor,
        afrr_act_factor=cfg.decomposition.afrr_act_factor,
    )
    engine = ReferencePriceClearingStub(references=realized_prices)
    awards = engine.clear(result["bid_book"])
    settlement = compute_greek_settlement(
        awards=awards,
        realized_prices=realized_prices,
        n_hours=ep.N_ems,
        activation_fractions={
            Product.mFRR_Energy: cfg.decomposition.alpha_mfrr,
            Product.aFRR_Energy: cfg.decomposition.alpha_afrr,
        },
    )

    print("Phase 2: clearing & settlement (perfect delivery, system_dir=0):")
    print(f"  Bids submitted   : {settlement['n_bids_total']}")
    print(f"  Bids accepted    : {settlement['n_bids_accepted']}")
    print()
    print(f"  {'revenue line':<28s}  {'amount [$]':>12s}")
    print("  " + "-" * 44)
    print(f"  {'DAM energy':<28s}  {settlement['dam_revenue']:>12.2f}")
    print(f"  {'IDM energy':<28s}  {settlement['idm_revenue']:>12.2f}")
    print(f"  {'mFRR capacity':<28s}  {settlement['mfrr_cap_revenue']:>12.2f}")
    print(f"  {'aFRR capacity':<28s}  {settlement['afrr_cap_revenue']:>12.2f}")
    print(f"  {'mFRR activation':<28s}  {settlement['mfrr_activation_revenue']:>12.2f}")
    print(f"  {'aFRR activation':<28s}  {settlement['afrr_activation_revenue']:>12.2f}")
    print(f"  {'imbalance settlement':<28s}  {settlement['imbalance_settlement']:>12.2f}")
    print(f"  {'non-delivery penalty':<28s}  {-settlement['non_delivery_penalty']:>12.2f}")
    print("  " + "-" * 44)
    print(f"  {'TOTAL realised revenue':<28s}  {settlement['total_greek_revenue']:>12.2f}")
    print()
    print(f"  vs MILP expected_profit : {result['expected_profit']:>12.2f}  "
          f"(forecast-based, before degradation)")
    print()

    if args.plot:
        from core.planners.deterministic_lp import DeterministicLP
        os.makedirs("results", exist_ok=True)
        print()
        print("Saving plots:")
        _plot_phase1(
            result=result,
            bp=bp, ep=ep, cfg=cfg,
            forecast_e=day.forecast_e,
            forecast_r=day.forecast_r,
            probabilities=day.probabilities,
            out_path="results/phase1_milp_smoke.png",
        )
        # Side-by-side comparison vs the existing v5 LP baseline.
        lp = DeterministicLP(bp=bp, tp=tp, ep=ep, thp=thp)
        lp_result = lp.solve(
            soc_init=bp.SOC_init,
            soh_init=bp.SOH_init,
            t_init=thp.T_init,
            energy_scenarios=day.forecast_e,
            reg_scenarios=day.forecast_r,
            probabilities=day.probabilities,
        )
        _plot_milp_vs_lp(
            milp_result=result,
            lp_result=lp_result,
            bp=bp,
            out_path="results/phase1_milp_vs_lp.png",
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())
