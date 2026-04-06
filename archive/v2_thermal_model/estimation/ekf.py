"""Extended Kalman Filter for joint SOC / SOH / Temperature estimation.

v2_thermal_model: 3-state, 2-output EKF.

State:        x = [SOC, SOH, T]
Measurement:  y = [SOC_measured, T_measured]   (SOH is NOT measured)

The state transition Jacobian is computed analytically via CasADi
automatic differentiation of the shared RK4 integrator.

Runs every ``dt_estimator`` = 60 s.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from config.parameters import BatteryParams, EKFParams, ThermalParams, TimeParams
from models.battery_model import build_casadi_rk4_integrator


class ExtendedKalmanFilter:
    """EKF for joint SOC / SOH / Temperature estimation.

    Parameters
    ----------
    bp  : BatteryParams
    tp  : TimeParams
    ep  : EKFParams
    thp : ThermalParams
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        ep: EKFParams,
        thp: ThermalParams,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.ep = ep
        self.thp = thp

        # State estimate  [SOC, SOH, T]
        self.x_hat = np.array(
            [bp.SOC_init, bp.SOH_init, thp.T_init], dtype=np.float64,
        )

        # Error covariance  (3x3)
        self.P = np.diag([ep.p0_soc, ep.p0_soh, ep.p0_temp]).astype(np.float64)

        # Process noise covariance  (3x3)
        self.Q = np.diag([ep.q_soc, ep.q_soh, ep.q_temp]).astype(np.float64)

        # Measurement noise covariance  (2x2)
        self.R = np.diag([ep.r_soc_meas, ep.r_temp_meas]).astype(np.float64)

        # Measurement matrix:  y = H @ x  =>  y = [SOC, T]
        self.H = np.array([
            [1.0, 0.0, 0.0],   # SOC measured
            [0.0, 0.0, 1.0],   # Temperature measured (SOH NOT measured)
        ], dtype=np.float64)

        # Build CasADi Jacobian
        self._build_jacobian()

    # ------------------------------------------------------------------
    #  CasADi setup
    # ------------------------------------------------------------------

    def _build_jacobian(self) -> None:
        """Compute the Jacobian function  A(x, u) = df_discrete / dx."""
        F_step = build_casadi_rk4_integrator(
            self.bp, self.thp, self.tp.dt_estimator,
        )

        x_sym = ca.MX.sym("x", 3)
        u_sym = ca.MX.sym("u", 3)
        x_next = F_step(x_sym, u_sym)

        # Jacobian of the discrete-time map w.r.t. the state (3x3)
        A_sym = ca.jacobian(x_next, x_sym)

        self._A_func = ca.Function("A_ekf", [x_sym, u_sym], [A_sym])
        self._F_step = F_step

    def _f_eval(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Evaluate the discrete-time state transition (numpy)."""
        return np.array(self._F_step(x, u)).flatten()

    def _A_eval(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Evaluate the state transition Jacobian (numpy, 3x3)."""
        return np.array(self._A_func(x, u))

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
        x_pred[2] = np.clip(x_pred[2], -20.0, 80.0)

        # Covariance prediction
        self.P = A @ self.P @ A.T + self.Q
        self.x_hat = x_pred

    def update(self, y_meas: np.ndarray) -> np.ndarray:
        """Correction (measurement-update) step.

        Uses the Joseph form for numerical stability of the covariance
        update:  P = (I - K H) P (I - K H)^T  +  K R K^T

        Parameters
        ----------
        y_meas : ndarray, shape (2,)
            Noisy measurement [SOC_meas, T_meas].

        Returns
        -------
        x_hat : ndarray, shape (3,)
            Updated estimate [SOC_est, SOH_est, T_est].
        """
        # Innovation
        y_pred = self.H @ self.x_hat                      # (2,)
        innov = y_meas - y_pred                             # (2,)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R           # (2, 2)

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)          # (3, 2)

        # State update
        self.x_hat = self.x_hat + (K @ innov).flatten()

        # Project state onto feasible region
        self.x_hat[0] = np.clip(self.x_hat[0], 0.0, 1.0)     # SOC in [0, 1]
        self.x_hat[1] = np.clip(self.x_hat[1], 0.5, 1.0)     # SOH in [0.5, 1]
        self.x_hat[2] = np.clip(self.x_hat[2], -10.0, 80.0)  # T in [-10, 80]

        # Covariance update — Joseph form
        I_KH = np.eye(3) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x_hat.copy()

    def step(self, u: np.ndarray, y_meas: np.ndarray) -> np.ndarray:
        """Combined predict + update.

        Parameters
        ----------
        u : ndarray, shape (3,)
            Control input applied during the preceding interval.
        y_meas : ndarray, shape (2,)
            [SOC_meas, T_meas] at end of the interval.

        Returns
        -------
        x_hat : ndarray, shape (3,)
            [SOC_est, SOH_est, T_est]
        """
        self.predict(u)
        return self.update(y_meas)

    def get_estimate(self) -> np.ndarray:
        """Return current state estimate [SOC_est, SOH_est, T_est]."""
        return self.x_hat.copy()
