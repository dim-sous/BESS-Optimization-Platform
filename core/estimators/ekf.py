"""Extended Kalman Filter for joint SOC / SOH / Temperature / V_rc estimation.

v4_electrical_rc_model: 5-state, 3-output EKF.

State:        x = [SOC, SOH, T, V_rc1, V_rc2]
Measurement:  y = [SOC_measured, T_measured, V_term_measured]

The measurement model is NONLINEAR because V_term depends on SOC via
the OCV polynomial.  Both the state transition Jacobian A(x, u) and the
measurement Jacobian H(x, u) are computed via CasADi automatic
differentiation.

Runs every ``dt_estimator`` = 60 s.
"""

from __future__ import annotations

import logging

import casadi as ca
import numpy as np

from core.config.parameters import (
    BatteryParams,
    EKFParams,
    ElectricalParams,
    ThermalParams,
    TimeParams,
)
from core.physics.plant import build_casadi_rk4_integrator, build_casadi_measurement

logger = logging.getLogger(__name__)


class ExtendedKalmanFilter:
    """EKF for joint SOC / SOH / Temperature / V_rc1 / V_rc2 estimation.

    Parameters
    ----------
    bp  : BatteryParams
    tp  : TimeParams
    ep  : EKFParams
    thp : ThermalParams
    elp : ElectricalParams
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        ep: EKFParams,
        thp: ThermalParams,
        elp: ElectricalParams,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.ep = ep
        self.thp = thp
        self.elp = elp

        # State estimate  [SOC, SOH, T, V_rc1, V_rc2]
        self.x_hat = np.array(
            [bp.SOC_init, bp.SOH_init, thp.T_init,
             elp.V_rc1_init, elp.V_rc2_init],
            dtype=np.float64,
        )

        # Error covariance  (5x5)
        self.P = np.diag([
            ep.p0_soc, ep.p0_soh, ep.p0_temp,
            ep.p0_vrc1, ep.p0_vrc2,
        ]).astype(np.float64)

        # Process noise covariance  (5x5)
        self.Q = np.diag([
            ep.q_soc, ep.q_soh, ep.q_temp,
            ep.q_vrc1, ep.q_vrc2,
        ]).astype(np.float64)

        # Measurement noise covariance  (3x3)
        self.R = np.diag([
            ep.r_soc_meas, ep.r_temp_meas, ep.r_vterm_meas,
        ]).astype(np.float64)

        # Build CasADi Jacobians
        self._build_jacobians()

    # ------------------------------------------------------------------
    #  CasADi setup
    # ------------------------------------------------------------------

    def _build_jacobians(self) -> None:
        """Compute Jacobian functions A(x, u) and H(x, u) via CasADi AD.

        A(x, u) = d(f_discrete) / dx   (5x5)  -- state transition Jacobian
        H(x, u) = d(h) / dx            (3x5)  -- measurement Jacobian
        """
        F_step = build_casadi_rk4_integrator(
            self.bp, self.thp, self.elp, self.tp.dt_estimator,
        )

        x_sym = ca.MX.sym("x", 5)
        u_sym = ca.MX.sym("u", 3)

        # State transition Jacobian (5x5)
        x_next = F_step(x_sym, u_sym)
        A_sym = ca.jacobian(x_next, x_sym)
        self._A_func = ca.Function("A_ekf", [x_sym, u_sym], [A_sym])

        # Measurement Jacobian (3x5) -- nonlinear measurement model
        h_func = build_casadi_measurement(self.elp)
        y_pred = h_func(x_sym, u_sym)
        H_sym = ca.jacobian(y_pred, x_sym)
        self._H_func = ca.Function("H_ekf", [x_sym, u_sym], [H_sym])

        # Store integrator and measurement functions for evaluation
        self._F_step = F_step
        self._h_func = h_func

    def _f_eval(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Evaluate the discrete-time state transition (numpy)."""
        return np.array(self._F_step(x, u)).flatten()

    def _A_eval(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Evaluate the state transition Jacobian (numpy, 5x5)."""
        return np.array(self._A_func(x, u))

    def _h_eval(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Evaluate the nonlinear measurement model (numpy, 3,)."""
        return np.array(self._h_func(x, u)).flatten()

    def _H_eval(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Evaluate the measurement Jacobian (numpy, 3x5)."""
        return np.array(self._H_func(x, u))

    # ------------------------------------------------------------------
    #  EKF steps
    # ------------------------------------------------------------------

    def predict(self, u: np.ndarray) -> None:
        """Prediction (time-update) step.

        Parameters
        ----------
        u : ndarray, shape (3,)
            Control input [P_chg, P_dis, P_reg] applied during this interval.
        """
        # Jacobian at *prior* estimate (before prediction update)
        A = self._A_eval(self.x_hat, u)

        # State prediction
        x_pred = self._f_eval(self.x_hat, u)
        x_pred[0] = np.clip(x_pred[0], 0.0, 1.0)
        x_pred[1] = np.clip(x_pred[1], 0.5, 1.0)
        x_pred[2] = np.clip(x_pred[2], -10.0, 80.0)
        x_pred[3] = np.clip(x_pred[3], -2.0, 2.0)
        x_pred[4] = np.clip(x_pred[4], -2.0, 2.0)

        # Covariance prediction
        self.P = A @ self.P @ A.T + self.Q
        self.x_hat = x_pred

    def update(self, y_meas: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Correction (measurement-update) step.

        Uses the Joseph form for numerical stability of the covariance
        update:  P = (I - K H) P (I - K H)^T  +  K R K^T

        The measurement model is nonlinear, so H is evaluated at the
        current state and input using CasADi auto-differentiation.

        Parameters
        ----------
        y_meas : ndarray, shape (3,)
            Noisy measurement [SOC_meas, T_meas, V_term_meas].
        u : ndarray, shape (3,)
            Control input (needed for nonlinear measurement evaluation).

        Returns
        -------
        x_hat : ndarray, shape (5,)
            Updated estimate [SOC_est, SOH_est, T_est, V_rc1_est, V_rc2_est].
        """
        # Evaluate nonlinear measurement model and its Jacobian
        y_pred = self._h_eval(self.x_hat, u)                # (3,)
        H = self._H_eval(self.x_hat, u)                     # (3, 5)

        # Innovation
        innov = y_meas - y_pred                              # (3,)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R                       # (3, 3)

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)                 # (5, 3)

        # State update
        self.x_hat = self.x_hat + (K @ innov).flatten()

        # Project state onto feasible region
        self.x_hat[0] = np.clip(self.x_hat[0], 0.0, 1.0)       # SOC in [0, 1]
        self.x_hat[1] = np.clip(self.x_hat[1], 0.5, 1.0)       # SOH in [0.5, 1]
        self.x_hat[2] = np.clip(self.x_hat[2], -10.0, 80.0)    # T in [-10, 80]
        self.x_hat[3] = np.clip(self.x_hat[3], -2.0, 2.0)       # V_rc1 physical bound
        self.x_hat[4] = np.clip(self.x_hat[4], -2.0, 2.0)       # V_rc2 physical bound

        # Covariance update -- Joseph form
        I_KH = np.eye(5) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x_hat.copy()

    def step(self, u: np.ndarray, y_meas: np.ndarray) -> np.ndarray:
        """Combined predict + update.

        Parameters
        ----------
        u : ndarray, shape (3,)
            Control input applied during the preceding interval.
        y_meas : ndarray, shape (3,)
            [SOC_meas, T_meas, V_term_meas] at end of the interval.

        Returns
        -------
        x_hat : ndarray, shape (5,)
            [SOC_est, SOH_est, T_est, V_rc1_est, V_rc2_est]
        """
        self.predict(u)
        return self.update(y_meas, u)

    def get_estimate(self) -> np.ndarray:
        """Return current state estimate [SOC_est, SOH_est, T_est, V_rc1_est, V_rc2_est]."""
        return self.x_hat.copy()
