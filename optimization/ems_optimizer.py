"""Day-ahead Energy Management System (EMS) optimizer.

Solves an economic dispatch problem over a planning horizon to determine the
optimal charge / discharge schedule that maximises electricity arbitrage
profit, subject to battery dynamics and operational constraints.

The charge and discharge powers are modelled as separate non-negative
decision variables so that asymmetric round-trip efficiency is handled
exactly — no piecewise or big-M formulations are needed because
simultaneous charge and discharge is automatically sub-optimal.

Solver: CasADi Opti stack → IPOPT interior-point method.
"""

import logging

import casadi as ca
import numpy as np

from config import BatteryParams, EMSParams

logger = logging.getLogger(__name__)


class EMSOptimizer:
    """Economic optimisation layer for battery energy arbitrage.

    Parameters
    ----------
    battery : BatteryParams
        Battery physical parameters.
    ems : EMSParams
        EMS planning parameters.
    """

    def __init__(self, battery: BatteryParams, ems: EMSParams) -> None:
        self.bp = battery
        self.ep = ems
        self._build_problem()

    # ------------------------------------------------------------------
    # Problem construction (called once)
    # ------------------------------------------------------------------
    def _build_problem(self) -> None:
        N = self.ep.horizon
        bp = self.bp

        opti = ca.Opti()

        # ---- Decision variables ----
        P_dis = opti.variable(N)        # Discharge power [kW]  (≥ 0)
        P_chg = opti.variable(N)        # Charge power   [kW]  (≥ 0)
        SOC   = opti.variable(N + 1)    # State of charge [-]

        # ---- Parameters (set at solve-time) ----
        price = opti.parameter(N)
        soc_0 = opti.parameter()

        # ---- Objective: maximise profit  →  minimise –profit ----
        revenue = 0.0
        for k in range(N):
            # Net power to grid = P_dis − P_chg ;  revenue = price × energy
            revenue += price[k] * (P_dis[k] - P_chg[k]) * bp.dt_hours
        opti.minimize(-revenue)

        # ---- Dynamics ----
        opti.subject_to(SOC[0] == soc_0)
        for k in range(N):
            # Asymmetric efficiency: discharge depletes more SOC,
            # charge stores less than drawn from grid.
            delta_soc = (
                -P_dis[k] / bp.eta_discharge + P_chg[k] * bp.eta_charge
            ) * bp.dt_hours / bp.E_max_kwh
            opti.subject_to(SOC[k + 1] == SOC[k] + delta_soc)

        # ---- Bounds ----
        opti.subject_to(opti.bounded(bp.SOC_min, SOC, bp.SOC_max))
        opti.subject_to(opti.bounded(0, P_dis, bp.P_max_kw))
        opti.subject_to(opti.bounded(0, P_chg, bp.P_max_kw))

        # ---- Terminal constraint: return SOC to target ----
        opti.subject_to(SOC[N] >= bp.SOC_terminal)

        # ---- Initial guesses (feasible interior point) ----
        opti.set_initial(SOC, bp.SOC_init)
        opti.set_initial(P_dis, 0.0)
        opti.set_initial(P_chg, 0.0)

        # ---- Solver options ----
        solver_opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 2000,
            "ipopt.tol": 1e-8,
            "ipopt.acceptable_tol": 1e-6,
        }
        opti.solver("ipopt", solver_opts)

        # Store handles
        self._opti  = opti
        self._P_dis = P_dis
        self._P_chg = P_chg
        self._SOC   = SOC
        self._price = price
        self._soc_0 = soc_0

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self, prices: np.ndarray, soc_init: float) -> dict:
        """Solve the day-ahead economic optimisation.

        Parameters
        ----------
        prices : np.ndarray
            Electricity price forecast [$/kWh], length ≥ ``horizon``.
        soc_init : float
            Current battery state of charge [-].

        Returns
        -------
        dict
            ``P_ref``   – net power reference [kW]  (positive = discharge)
            ``SOC_ref`` – SOC reference trajectory [-]
            ``P_dis``   – discharge power trajectory [kW]
            ``P_chg``   – charge power trajectory [kW]
            ``profit``  – expected total profit [$]

        Raises
        ------
        RuntimeError
            If IPOPT fails to converge.
        """
        N = self.ep.horizon
        if len(prices) < N:
            raise ValueError(
                f"Price vector length {len(prices)} is shorter than "
                f"the EMS horizon {N}"
            )

        self._opti.set_value(self._price, prices[:N])
        self._opti.set_value(self._soc_0, soc_init)

        try:
            sol = self._opti.solve()
        except RuntimeError as exc:
            logger.error("EMS optimisation failed: %s", exc)
            raise

        P_dis = np.asarray(sol.value(self._P_dis)).flatten()
        P_chg = np.asarray(sol.value(self._P_chg)).flatten()
        SOC   = np.asarray(sol.value(self._SOC)).flatten()
        P_ref = P_dis - P_chg
        profit = float(-sol.value(self._opti.f))

        logger.info("EMS solved — expected profit: $%.2f", profit)

        return {
            "P_ref":  P_ref,
            "SOC_ref": SOC,
            "P_dis":  P_dis,
            "P_chg":  P_chg,
            "profit": profit,
        }
