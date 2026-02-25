"""Real-time tracking Model Predictive Controller (MPC).

Implements a receding-horizon quadratic tracking controller that follows
the reference trajectories produced by the EMS economic layer.

Key design choices
------------------
* **Soft SOC constraints** — slack variables with a heavy quadratic penalty
  guarantee that the QP is always feasible, even under model mismatch.
* **Warm-starting** — the previous solution is shifted by one step and used
  as the initial guess, which drastically reduces IPOPT iteration count.
* **Simplified dynamics** — a single net-power variable with unity efficiency
  is used inside the MPC.  The feedback loop corrects for the mismatch
  between this prediction model and the real plant (which has asymmetric
  η_chg / η_dis).  This is standard practice in hierarchical BESS control.

Solver: CasADi Opti stack → IPOPT.
"""

import logging

import casadi as ca
import numpy as np

from config import BatteryParams, MPCParams

logger = logging.getLogger(__name__)


class MPCController:
    """Tracking MPC for real-time battery power dispatch.

    Parameters
    ----------
    battery : BatteryParams
        Battery configuration.
    mpc : MPCParams
        MPC tuning parameters.
    """

    def __init__(self, battery: BatteryParams, mpc: MPCParams) -> None:
        self.bp = battery
        self.mp = mpc
        self._prev_P: np.ndarray | None = None
        self._prev_SOC: np.ndarray | None = None
        self._build_problem()

    # ------------------------------------------------------------------
    # Problem construction
    # ------------------------------------------------------------------
    def _build_problem(self) -> None:
        N  = self.mp.horizon
        bp = self.bp
        mp = self.mp

        opti = ca.Opti()

        # ---- Decision variables ----
        P   = opti.variable(N)          # Net power [kW]
        SOC = opti.variable(N + 1)      # SOC trajectory [-]
        eps = opti.variable(N + 1)      # Slack for soft SOC bounds

        # ---- Parameters (set every call) ----
        soc_0   = opti.parameter()
        soc_ref = opti.parameter(N + 1)
        p_ref   = opti.parameter(N)

        # ---- Objective: weighted tracking + soft-constraint penalty ----
        cost = 0.0
        for k in range(N):
            cost += mp.Q_soc    * (SOC[k] - soc_ref[k]) ** 2
            cost += mp.R_power  * (P[k]   - p_ref[k])   ** 2
        # Terminal SOC penalty
        cost += mp.Q_terminal * (SOC[N] - soc_ref[N]) ** 2
        # Soft-constraint violation penalty (quadratic barrier)
        for k in range(N + 1):
            cost += mp.slack_penalty * eps[k] ** 2
        opti.minimize(cost)

        # ---- Simplified dynamics (unit efficiency) ----
        opti.subject_to(SOC[0] == soc_0)
        for k in range(N):
            opti.subject_to(
                SOC[k + 1] == SOC[k] - P[k] * bp.dt_hours / bp.E_max_kwh
            )

        # ---- Constraints ----
        opti.subject_to(opti.bounded(-bp.P_max_kw, P, bp.P_max_kw))
        opti.subject_to(eps >= 0)
        for k in range(N + 1):
            opti.subject_to(SOC[k] >= bp.SOC_min - eps[k])
            opti.subject_to(SOC[k] <= bp.SOC_max + eps[k])

        # ---- Solver options ----
        solver_opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 500,
            "ipopt.tol": 1e-6,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.mu_init": 1e-3,
        }
        opti.solver("ipopt", solver_opts)

        # Store handles
        self._opti    = opti
        self._P       = P
        self._SOC     = SOC
        self._eps     = eps
        self._soc_0   = soc_0
        self._soc_ref = soc_ref
        self._p_ref   = p_ref

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pad_reference(ref: np.ndarray, required_len: int) -> np.ndarray:
        """Extend a reference vector with its last value if too short."""
        if len(ref) >= required_len:
            return ref[:required_len]
        pad = np.full(required_len - len(ref), ref[-1])
        return np.concatenate([ref, pad])

    # ------------------------------------------------------------------
    # Solve (called every time step)
    # ------------------------------------------------------------------
    def solve(
        self,
        soc_current: float,
        soc_ref: np.ndarray,
        p_ref: np.ndarray,
    ) -> float:
        """Compute the MPC control action for the current time step.

        Parameters
        ----------
        soc_current : float
            Measured SOC at the current instant [-].
        soc_ref : np.ndarray
            SOC reference from the EMS (length ≥ 1; padded if shorter
            than ``horizon + 1``).
        p_ref : np.ndarray
            Power reference from the EMS (length ≥ 1; padded if shorter
            than ``horizon``).

        Returns
        -------
        float
            Optimal power command for the current step [kW].
            Falls back to ``p_ref[0]`` if the solver fails.
        """
        N = self.mp.horizon

        soc_ref_vec = self._pad_reference(
            np.asarray(soc_ref, dtype=float), N + 1
        )
        p_ref_vec = self._pad_reference(
            np.asarray(p_ref, dtype=float), N
        )

        # Set parameter values
        self._opti.set_value(self._soc_0,   soc_current)
        self._opti.set_value(self._soc_ref,  soc_ref_vec)
        self._opti.set_value(self._p_ref,    p_ref_vec)

        # Warm-start from shifted previous solution
        if self._prev_P is not None:
            self._opti.set_initial(self._P,   self._prev_P)
            self._opti.set_initial(self._SOC, self._prev_SOC)
        else:
            # First call — initialise from reference
            self._opti.set_initial(self._P, p_ref_vec)
            self._opti.set_initial(
                self._SOC,
                np.linspace(soc_current, soc_ref_vec[-1], N + 1),
            )
        self._opti.set_initial(self._eps, 0.0)

        try:
            sol = self._opti.solve()

            P_opt   = np.asarray(sol.value(self._P)).flatten()
            SOC_opt = np.asarray(sol.value(self._SOC)).flatten()

            # Cache shifted solution for next warm-start
            self._prev_P   = np.append(P_opt[1:],   P_opt[-1])
            self._prev_SOC = np.append(SOC_opt[1:], SOC_opt[-1])

            return float(P_opt[0])

        except RuntimeError:
            logger.warning(
                "MPC solve failed at SOC=%.3f — falling back to reference",
                soc_current,
            )
            return float(p_ref_vec[0])
