"""Moving Horizon Estimation for joint SOC / SOH / Temperature / V_rc estimation.

v4_electrical_rc_model: 5-state, 3-output MHE.

Performance note: The NLP is pre-built once at full window size (N_mhe) with
CasADi parameters for measurements and inputs, avoiding costly per-call NLP
reconstruction.  During the fill-up phase (first N_mhe steps), the estimate
is propagated via the dynamics model; the MHE NLP is only solved once the
buffer is full.

Formulation
-----------
  min   arrival_cost(x_0 - x_bar)
      + sum_k  w_soc_meas  * (SOC_k  - y_soc_k)^2
      + sum_k  w_temp_meas * (T_k    - y_temp_k)^2
      + sum_k  w_vterm_meas * (V_term_pred_k - y_vterm_k)^2
      + sum_k  w_process_soc  * w_soc_k^2
      +        w_process_soh  * w_soh_k^2
      +        w_process_temp * w_temp_k^2
      +        w_process_vrc1 * w_vrc1_k^2
      +        w_process_vrc2 * w_vrc2_k^2

  s.t.  x_{k+1} = f(x_k, u_k) + w_k
        SOC_min - margin  <=  SOC_k  <=  SOC_max + margin
        0.5  <=  SOH_k  <=  1.0
        -10  <=  T_k    <=  80
        -50  <=  V_rc1  <=  50
        -50  <=  V_rc2  <=  50

Estimation window:  N_mhe = 30 steps  (30 minutes at dt_estimator = 60 s).
Solved with CasADi Opti / IPOPT.
"""

from __future__ import annotations

import logging

import casadi as ca
import numpy as np

from core.config.parameters import (
    BatteryParams,
    ElectricalParams,
    MHEParams,
    ThermalParams,
    TimeParams,
)
from core.physics.plant import build_casadi_rk4_integrator, build_casadi_measurement

logger = logging.getLogger(__name__)


class MovingHorizonEstimator:
    """MHE for joint SOC / SOH / Temperature / V_rc1 / V_rc2 estimation.

    The NLP is pre-built once at construction for window size N_mhe.
    During the initial fill-up phase (< N_mhe measurements), the estimate
    is propagated via the RK4 integrator.  Once the buffer is full, the
    pre-built NLP is solved each step with updated parameter values.

    Parameters
    ----------
    bp  : BatteryParams
    tp  : TimeParams
    mp  : MHEParams
    thp : ThermalParams
    elp : ElectricalParams
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        mp: MHEParams,
        thp: ThermalParams,
        elp: ElectricalParams,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.mp = mp
        self.thp = thp
        self.elp = elp

        N = mp.N_mhe

        # Circular buffers for the estimation window
        self._u_buf: list[np.ndarray] = []
        self._y_buf: list[np.ndarray] = []    # each entry is ndarray(3,)

        # Arrival cost reference state  [SOC, SOH, T, V_rc1, V_rc2]
        self._x_arrival = np.array(
            [bp.SOC_init, bp.SOH_init, thp.T_init,
             elp.V_rc1_init, elp.V_rc2_init],
            dtype=np.float64,
        )

        # Current estimate  [SOC, SOH, T, V_rc1, V_rc2]
        self.x_hat = np.array(
            [bp.SOC_init, bp.SOH_init, thp.T_init,
             elp.V_rc1_init, elp.V_rc2_init],
            dtype=np.float64,
        )

        # CasADi integrator (for fill-up phase propagation)
        self._F_step = build_casadi_rk4_integrator(
            bp, thp, elp, tp.dt_estimator,
        )

        # Pre-build the NLP once at full window size
        self._build_problem(N)

        # Warm-start cache
        self._prev_X: np.ndarray | None = None

    # ------------------------------------------------------------------
    #  NLP construction (called once at init)
    # ------------------------------------------------------------------

    def _build_problem(self, N: int) -> None:
        """Pre-build the MHE NLP for window size N with CasADi parameters."""
        bp = self.bp
        mp = self.mp
        elp = self.elp

        F_step = self._F_step
        h_func = build_casadi_measurement(elp)

        opti = ca.Opti()

        # Decision variables
        X = opti.variable(5, N + 1)       # state trajectory
        W = opti.variable(5, N)            # process noise

        # Parameters (set each solve)
        x_arrival_p = opti.parameter(5)
        U_p = opti.parameter(3, N)         # input buffer
        Y_p = opti.parameter(3, N)         # measurement buffer

        # ----- Objective -----
        cost = 0.0

        # Arrival cost
        cost += mp.arrival_soc * (X[0, 0] - x_arrival_p[0]) ** 2
        cost += mp.arrival_soh * (X[1, 0] - x_arrival_p[1]) ** 2
        cost += mp.arrival_temp * (X[2, 0] - x_arrival_p[2]) ** 2
        cost += mp.arrival_vrc1 * (X[3, 0] - x_arrival_p[3]) ** 2
        cost += mp.arrival_vrc2 * (X[4, 0] - x_arrival_p[4]) ** 2

        # Stage costs
        for k in range(N):
            # Dynamics with additive process noise
            x_k = X[:, k]
            u_k = U_p[:, k]
            x_next_nom = F_step(x_k, u_k)

            opti.subject_to(X[0, k + 1] == x_next_nom[0] + W[0, k])
            opti.subject_to(X[1, k + 1] == x_next_nom[1] + W[1, k])
            opti.subject_to(X[2, k + 1] == x_next_nom[2] + W[2, k])
            opti.subject_to(X[3, k + 1] == x_next_nom[3] + W[3, k])
            opti.subject_to(X[4, k + 1] == x_next_nom[4] + W[4, k])

            # Measurement cost (SOC, Temperature, V_term)
            cost += mp.w_soc_meas * (X[0, k + 1] - Y_p[0, k]) ** 2
            cost += mp.w_temp_meas * (X[2, k + 1] - Y_p[1, k]) ** 2

            # V_term measurement cost -- nonlinear measurement model
            y_pred = h_func(X[:, k + 1], u_k)
            V_term_pred = y_pred[2]
            cost += mp.w_vterm_meas * (V_term_pred - Y_p[2, k]) ** 2

            # Process noise penalty
            cost += mp.w_process_soc * W[0, k] ** 2
            cost += mp.w_process_soh * W[1, k] ** 2
            cost += mp.w_process_temp * W[2, k] ** 2
            cost += mp.w_process_vrc1 * W[3, k] ** 2
            cost += mp.w_process_vrc2 * W[4, k] ** 2

        opti.minimize(cost)

        # ----- State bounds -----
        soc_margin = 0.05
        for k in range(N + 1):
            opti.subject_to(
                opti.bounded(bp.SOC_min - soc_margin, X[0, k], bp.SOC_max + soc_margin)
            )
            opti.subject_to(opti.bounded(0.5, X[1, k], 1.0))
            opti.subject_to(opti.bounded(-10.0, X[2, k], 80.0))
            opti.subject_to(opti.bounded(-50.0, X[3, k], 50.0))
            opti.subject_to(opti.bounded(-50.0, X[4, k], 50.0))

        # ----- Solver -----
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 300,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
        }
        opti.solver("ipopt", opts)

        # Store handles
        self._opti = opti
        self._X = X
        self._W = W
        self._x_arrival_p = x_arrival_p
        self._U_p = U_p
        self._Y_p = Y_p
        self._N = N

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    def step(self, u: np.ndarray, y_meas: np.ndarray) -> np.ndarray:
        """Process a new (input, measurement) pair and return the estimate.

        Parameters
        ----------
        u : ndarray, shape (3,)
            [P_chg, P_dis, P_reg] applied during the preceding interval.
        y_meas : ndarray, shape (3,)
            [SOC_meas, T_meas, V_term_meas] at end of interval.

        Returns
        -------
        x_hat : ndarray, shape (5,)
            [SOC_est, SOH_est, T_est, V_rc1_est, V_rc2_est]
        """
        N = self._N

        self._u_buf.append(u.copy())
        self._y_buf.append(y_meas.copy())

        # Trim to window
        if len(self._u_buf) > N:
            # Update arrival cost from the discarded portion
            if self._prev_X is not None and self._prev_X.shape[1] > 1:
                self._x_arrival = self._prev_X[:, 1].copy()
            self._u_buf = self._u_buf[-N:]
            self._y_buf = self._y_buf[-N:]

        M = len(self._u_buf)

        if M < N:
            # Fill-up phase: propagate with dynamics (fast, no NLP)
            x_pred = np.array(self._F_step(self.x_hat, u)).flatten()
            # Blend with measurement for basic correction
            alpha_soc = 0.1
            alpha_temp = 0.1
            x_pred[0] += alpha_soc * (y_meas[0] - x_pred[0])
            x_pred[2] += alpha_temp * (y_meas[1] - x_pred[2])
            # Clamp
            x_pred[0] = np.clip(x_pred[0], 0.0, 1.0)
            x_pred[1] = np.clip(x_pred[1], 0.5, 1.0)
            x_pred[2] = np.clip(x_pred[2], -20.0, 80.0)
            x_pred[3] = np.clip(x_pred[3], -50.0, 50.0)
            x_pred[4] = np.clip(x_pred[4], -50.0, 50.0)
            self.x_hat = x_pred
            return self.x_hat.copy()

        return self._solve_mhe()

    def get_estimate(self) -> np.ndarray:
        """Return current state estimate [SOC_est, SOH_est, T_est, V_rc1_est, V_rc2_est]."""
        return self.x_hat.copy()

    # ------------------------------------------------------------------
    #  NLP solve (reuses pre-built problem)
    # ------------------------------------------------------------------

    def _solve_mhe(self) -> np.ndarray:
        """Solve the pre-built MHE NLP with current buffer data."""
        N = self._N
        opti = self._opti

        # Set parameters
        opti.set_value(self._x_arrival_p, self._x_arrival)

        U_val = np.column_stack(self._u_buf).T.T  # (3, N)
        Y_val = np.column_stack(self._y_buf).T.T  # (3, N)
        # Ensure correct shape
        U_val = np.array([self._u_buf[k] for k in range(N)]).T  # (3, N)
        Y_val = np.array([self._y_buf[k] for k in range(N)]).T  # (3, N)

        opti.set_value(self._U_p, U_val)
        opti.set_value(self._Y_p, Y_val)

        # Warm-start
        if self._prev_X is not None and self._prev_X.shape[1] == N + 1:
            opti.set_initial(self._X, self._prev_X)
        else:
            for k in range(N + 1):
                opti.set_initial(self._X[0, k], self.x_hat[0])
                opti.set_initial(self._X[1, k], self.x_hat[1])
                opti.set_initial(self._X[2, k], self.x_hat[2])
                opti.set_initial(self._X[3, k], 0.0)
                opti.set_initial(self._X[4, k], 0.0)
        opti.set_initial(self._W, 0.0)

        try:
            sol = opti.solve()
            X_opt = np.array(sol.value(self._X))   # (5, N+1)
            self._prev_X = X_opt
            self.x_hat = X_opt[:, -1].copy()
        except RuntimeError:
            # Solver failed -- keep previous estimate, propagate with dynamics
            logger.warning("MHE solver failed; keeping previous estimate.")
            if len(self._u_buf) > 0:
                x_pred = np.array(
                    self._F_step(self.x_hat, self._u_buf[-1])
                ).flatten()
                self.x_hat = x_pred

        return self.x_hat.copy()
