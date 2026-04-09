"""Economic MPC.

Profit-maximising NLP with a soft terminal SOC anchor against the EMS
plan. The MPC sees forecast prices (probability-weighted forecast mean,
never realized) and trades arbitrage profit against degradation cost.

Design points:

- ``P_reg`` is exogenous (parameter, not decision variable). The
  simulator passes the EMS-committed reg power ZOH-expanded across the
  horizon; the MPC plans `P_chg`/`P_dis` to keep SOC feasible under
  that commitment.
- The MPC plans expected signed activation = 0 (the OU activation
  process is symmetric around zero), so there is no SOC pre-positioning
  term and no expected delivery payment in the cost (constant w.r.t.
  the decision variables).
- No per-step SOC anchor — the EMS pins end-of-hour SOC, not an in-hour
  trajectory. Cross-hour alignment is handled exclusively by the
  terminal anchor `Q_terminal_econ · (SOC[N] − soc_ref[N])²`.
- Solver-failure fallback: a `TrackingMPC` instance, so the EMS plan is
  never lost.
"""

from __future__ import annotations

import logging

import casadi as ca
import numpy as np

from core.config.parameters import (
    BatteryParams,
    ElectricalParams,
    EMSParams,
    MPCParams,
    ThermalParams,
    TimeParams,
)
from core.mpc._common import ipopt_opts, pad_to
from core.mpc.tracking import TrackingMPC, _build_2state_integrator

logger = logging.getLogger(__name__)


class EconomicMPC:
    """Profit-maximising MPC with terminal EMS-anchor."""

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        mp: MPCParams,
        thp: ThermalParams,
        elp: ElectricalParams,
        ep: EMSParams,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.mp = mp
        self.thp = thp
        self.elp = elp
        self.ep = ep

        self.last_solve_failed = False
        self._clip_warned = False  # one-shot warn when EKF SOC is out-of-bounds

        # Warm-start caches
        self._prev_P_chg: np.ndarray | None = None
        self._prev_P_dis: np.ndarray | None = None
        self._prev_SOC: np.ndarray | None = None
        self._prev_TEMP: np.ndarray | None = None

        # Fallback: if the economic NLP fails, fall back to a tracking
        # MPC solve so we never lose the EMS plan.
        self._tracking_fallback = TrackingMPC(bp, tp, mp, thp, elp)

        self._build_problem()

    # ------------------------------------------------------------------
    #  NLP construction (called once)
    # ------------------------------------------------------------------

    def _build_problem(self) -> None:
        N = self.mp.N_mpc      # 60
        Nc = self.mp.Nc_mpc    # 20
        bp = self.bp
        mp = self.mp
        thp = self.thp
        ep = self.ep

        opti = ca.Opti()

        # ---- Decision variables (P_reg is exogenous, not optimised) ----
        P_chg = opti.variable(Nc)
        P_dis = opti.variable(Nc)
        SOC = opti.variable(N + 1)
        TEMP = opti.variable(N + 1)
        eps = opti.variable(N + 1)
        eps_temp = opti.variable(N + 1)

        # ---- Parameters (set each solve) ----
        soc_0 = opti.parameter()
        soh_p = opti.parameter()
        temp_0 = opti.parameter()
        soc_ref_p = opti.parameter(N + 1)             # only soc_ref_p[N] enters cost
        u_prev_p = opti.parameter(2)                  # last [P_chg, P_dis]
        price_e_p = opti.parameter(N)                 # forecast $/kWh per step
        p_reg_committed_p = opti.parameter(N)         # ZOH from EMS plan, magnitude ≥0

        # ---- 2-state integrator with SOH parameter ----
        F_mpc = _build_2state_integrator(
            bp, thp, self.elp, self.tp.dt_mpc, soh_p,
        )

        opti.subject_to(SOC[0] == soc_0)
        opti.subject_to(TEMP[0] == temp_0)

        dt_h = self.tp.dt_mpc / 3600.0
        dt_s = self.tp.dt_mpc

        cost = 0.0

        for k in range(N):
            j = min(k, Nc - 1)   # control horizon blocking

            # Reg input is the absolute committed reg power so thermal
            # Joule heating is correctly accounted in the prediction.
            u_k_eff = ca.vertcat(P_chg[j], P_dis[j], p_reg_committed_p[k])

            x2_k = ca.vertcat(SOC[k], TEMP[k])
            x2_next = F_mpc(x2_k, u_k_eff, soh_p)
            opti.subject_to(SOC[k + 1] == x2_next[0])
            opti.subject_to(TEMP[k + 1] == x2_next[1])

            # Energy arbitrage profit (negative cost = profit)
            cost += -price_e_p[k] * (P_dis[j] - P_chg[j]) * dt_h

            # Degradation cost on planned chg/dis throughput. Reg-power
            # degradation is constant w.r.t. our decision variables and
            # is omitted from the optimization (the simulator's value
            # report still accounts for it).
            P_arb = P_chg[j] + P_dis[j]
            cost += ep.degradation_cost * bp.alpha_deg * P_arb * dt_s

            # NO per-step SOC anchor. Intra-hour SOC trajectory is fiction
            # (the EMS pins end-of-hour SOC only). Cross-hour alignment
            # falls entirely on the terminal anchor below.

            # Rate-of-change penalty (smoothness)
            if k == 0:
                cost += mp.R_delta_econ * (
                    (P_chg[0] - u_prev_p[0]) ** 2
                    + (P_dis[0] - u_prev_p[1]) ** 2
                )
            elif k < Nc:
                cost += mp.R_delta_econ * (
                    (P_chg[k] - P_chg[k - 1]) ** 2
                    + (P_dis[k] - P_dis[k - 1]) ** 2
                )

        # Terminal anchor (cross-hour alignment with EMS plan)
        cost += mp.Q_terminal_econ * (SOC[N] - soc_ref_p[N]) ** 2

        # Slack penalties
        for k in range(N + 1):
            cost += mp.slack_penalty * eps[k] ** 2
            cost += mp.slack_penalty_temp * eps_temp[k] ** 2

        opti.minimize(cost)

        # ---- Constraints ----
        opti.subject_to(opti.bounded(0.0, P_chg, bp.P_max_kw))
        opti.subject_to(opti.bounded(0.0, P_dis, bp.P_max_kw))
        opti.subject_to(eps >= 0)
        opti.subject_to(eps_temp >= 0)

        for k in range(N + 1):
            opti.subject_to(SOC[k] >= bp.SOC_min - eps[k])
            opti.subject_to(SOC[k] <= bp.SOC_max + eps[k])
            opti.subject_to(TEMP[k] <= thp.T_max + eps_temp[k])
            opti.subject_to(TEMP[k] >= thp.T_min - eps_temp[k])

        # Power-budget headroom: planned chg/dis must leave room for the
        # committed reg power at the corresponding horizon step. Enforced
        # over the FULL prediction horizon so the blocked control region
        # cannot propagate a physically infeasible plan.
        # NOTE: assumes p_reg_committed_p ≥ 0 (magnitude of symmetric FCR
        # capacity). If reg ever becomes signed/directional, replace with
        # ca.fabs(p_reg_committed_p[k]).
        for k in range(N):
            j = min(k, Nc - 1)
            opti.subject_to(P_chg[j] + p_reg_committed_p[k] <= bp.P_max_kw)
            opti.subject_to(P_dis[j] + p_reg_committed_p[k] <= bp.P_max_kw)

        opti.solver("ipopt", ipopt_opts())

        # ---- Store handles ----
        self._opti = opti
        self._P_chg = P_chg
        self._P_dis = P_dis
        self._SOC = SOC
        self._TEMP = TEMP
        self._eps = eps
        self._eps_temp = eps_temp
        self._soc_0 = soc_0
        self._soh_p = soh_p
        self._temp_0 = temp_0
        self._soc_ref_p = soc_ref_p
        self._u_prev_p = u_prev_p
        self._price_e_p = price_e_p
        self._p_reg_committed_p = p_reg_committed_p

    # ------------------------------------------------------------------
    #  Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        x_est: np.ndarray,
        soc_ref: np.ndarray,
        p_chg_ref: np.ndarray,
        p_dis_ref: np.ndarray,
        price_e_horizon: np.ndarray,
        p_reg_committed_horizon: np.ndarray,
        u_prev: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the economic-MPC control action.

        Parameters
        ----------
        x_est                    : ndarray (>=3,) EKF estimate. Layout
                                   assumed `[SOC, SOH, T, ...]`.
        soc_ref                  : ndarray (N+1,) — only `soc_ref[N]`
                                   enters the cost (terminal anchor);
                                   the full vector seeds the SOC
                                   warm-start and is forwarded to the
                                   tracking fallback.
        p_chg_ref/p_dis_ref      : ndarray (N,) — fallback only.
        price_e_horizon          : ndarray (N,) forecast energy price ZOH [$/kWh]
        p_reg_committed_horizon  : ndarray (N,) committed FCR power ZOH
                                   from EMS plan, magnitude ≥0 [kW]
        u_prev                   : ndarray (>=2,) previous applied control.
                                   Adapter passes the full 3-vector
                                   `[P_chg, P_dis, P_reg]`; we only use
                                   the first two entries.

        Returns
        -------
        u_cmd : ndarray (3,)  [P_chg, P_dis, P_reg]  (P_reg = committed)
        """
        N = self.mp.N_mpc
        opti = self._opti
        thp = self.thp

        assert len(x_est) >= 3, f"x_est must have ≥3 entries, got {len(x_est)}"

        soc_ref = pad_to(soc_ref, N + 1)
        price_e = pad_to(price_e_horizon, N)
        p_reg_committed = pad_to(p_reg_committed_horizon, N)

        soc_raw = float(x_est[0])
        soc_0_val = float(np.clip(soc_raw, self.bp.SOC_min, self.bp.SOC_max))
        if soc_raw != soc_0_val and not self._clip_warned:
            logger.warning(
                "EconomicMPC: EKF SOC %.4f outside [%.2f, %.2f]; clipped to %.4f. "
                "Subsequent occurrences suppressed.",
                soc_raw, self.bp.SOC_min, self.bp.SOC_max, soc_0_val,
            )
            self._clip_warned = True
        soh_val = float(np.clip(x_est[1], 0.5, 1.0))
        temp_0_val = float(np.clip(x_est[2], thp.T_min - 5.0, thp.T_max + 5.0))

        if u_prev is None:
            u_prev = np.zeros(2)
        else:
            u_prev = np.asarray(u_prev, dtype=float)[:2]

        # Set parameters
        opti.set_value(self._soc_0, soc_0_val)
        opti.set_value(self._soh_p, soh_val)
        opti.set_value(self._temp_0, temp_0_val)
        opti.set_value(self._soc_ref_p, soc_ref)
        opti.set_value(self._u_prev_p, u_prev)
        opti.set_value(self._price_e_p, price_e)
        opti.set_value(self._p_reg_committed_p, p_reg_committed)

        # Warm-start
        if self._prev_P_chg is not None:
            opti.set_initial(self._P_chg, self._prev_P_chg)
            opti.set_initial(self._P_dis, self._prev_P_dis)
            opti.set_initial(self._SOC, self._prev_SOC)
            opti.set_initial(self._TEMP, self._prev_TEMP)
        else:
            opti.set_initial(self._P_chg, 0.0)
            opti.set_initial(self._P_dis, 0.0)
            opti.set_initial(self._SOC, soc_ref)
            opti.set_initial(self._TEMP, temp_0_val)
        opti.set_initial(self._eps, 0.0)
        opti.set_initial(self._eps_temp, 0.0)

        try:
            sol = opti.solve()
            self.last_solve_failed = False

            pc = np.array(sol.value(self._P_chg)).flatten()
            pd = np.array(sol.value(self._P_dis)).flatten()
            soc_opt = np.array(sol.value(self._SOC)).flatten()
            temp_opt = np.array(sol.value(self._TEMP)).flatten()

            # P_reg returned to caller is the EMS-committed value at k=0
            # (MPC does not choose it; it forwards what's committed).
            u_cmd = np.array([pc[0], pd[0], float(p_reg_committed[0])])

            self._prev_P_chg = np.append(pc[1:], pc[-1])
            self._prev_P_dis = np.append(pd[1:], pd[-1])
            self._prev_SOC = np.append(soc_opt[1:], soc_opt[-1])
            self._prev_TEMP = np.append(temp_opt[1:], temp_opt[-1])

        except RuntimeError as exc:
            logger.warning("EconomicMPC failed (%s); falling back to TrackingMPC",
                           str(exc)[:200])
            self.last_solve_failed = True
            u_cmd = self._tracking_fallback.solve(
                x_est=x_est,
                soc_ref=soc_ref,
                p_chg_ref=p_chg_ref,
                p_dis_ref=p_dis_ref,
                p_reg_committed_horizon=p_reg_committed,
                u_prev=u_prev,
            )

        return u_cmd
