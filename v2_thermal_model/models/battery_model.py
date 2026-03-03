"""Nonlinear 3-state battery energy storage system model with thermal dynamics.

v2_thermal_model: extends the 2-state baseline with temperature.

Provides:
  1. CasADi symbolic dynamics (shared by EMS, MPC, EKF, MHE).
  2. A numpy-based high-fidelity plant for closed-loop simulation.

State vector   x = [SOC, SOH, T]
Input vector   u = [P_charge, P_discharge, P_reg]   (all >= 0, kW)
Measurement    y = [SOC_measured, T_measured]   (SOH is NOT measured)

Continuous-time dynamics
------------------------
  dSOC/dt = (eta_c * P_chg  -  P_dis / eta_d) / (SOH * E_nom * 3600)    [1/s]
  dSOH/dt = -alpha_deg * kappa(T) * (P_chg + P_dis + |P_reg|)            [1/s]
  dT/dt   = (I^2 * R_internal - h_cool * (T - T_ambient)) / C_thermal    [degC/s]

  where kappa(T) = exp(E_a / R_gas * (1/T_ref_K - 1/T_K))   (Arrhenius)
        I = (P_chg + P_dis + |P_reg|) * 1000 / V_nominal     [A]
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from config.parameters import BatteryParams, ThermalParams, TimeParams


# ---------------------------------------------------------------------------
#  CasADi symbolic dynamics  (reused by EMS, MPC, EKF, MHE)
# ---------------------------------------------------------------------------

def build_casadi_dynamics(bp: BatteryParams, thp: ThermalParams) -> ca.Function:
    """Return a CasADi Function  f(x, u) -> x_dot  (continuous-time ODE).

    Parameters
    ----------
    bp  : BatteryParams
    thp : ThermalParams

    Returns
    -------
    ca.Function  with signature  (x[3], u[3]) -> x_dot[3]
    """
    x = ca.MX.sym("x", 3)     # [SOC, SOH, T]
    u = ca.MX.sym("u", 3)     # [P_chg, P_dis, P_reg]

    SOC, SOH, T = x[0], x[1], x[2]
    P_chg, P_dis, P_reg = u[0], u[1], u[2]

    # ---- SOC dynamics (unchanged from v1) ----
    E_eff_kws = SOH * bp.E_nom_kwh * 3600.0          # effective capacity [kW*s]
    dSOC_dt = (bp.eta_charge * P_chg - P_dis / bp.eta_discharge) / E_eff_kws

    # ---- Thermally-coupled degradation ----
    T_ref_K = thp.T_ref + 273.15                      # [K]
    T_K = T + 273.15                                   # [K]
    kappa = ca.exp(thp.E_a / thp.R_gas * (1.0 / T_ref_K - 1.0 / T_K))

    P_total = P_chg + P_dis + ca.fabs(P_reg)          # total power throughput [kW]
    dSOH_dt = -bp.alpha_deg * kappa * P_total

    # ---- Thermal dynamics ----
    I = P_total * 1000.0 / thp.V_nominal              # pack current [A]
    Q_joule = I ** 2 * thp.R_internal                  # Joule heating [W]
    Q_cool = thp.h_cool * (T - thp.T_ambient)         # cooling [W]
    dT_dt = (Q_joule - Q_cool) / thp.C_thermal        # [degC/s]

    x_dot = ca.vertcat(dSOC_dt, dSOH_dt, dT_dt)
    return ca.Function("battery_ode", [x, u], [x_dot], ["x", "u"], ["x_dot"])


def build_casadi_rk4_integrator(
    bp: BatteryParams, thp: ThermalParams, dt: float,
) -> ca.Function:
    """Return a single-step RK4 integrator  F(x, u) -> x_next.

    Parameters
    ----------
    bp  : BatteryParams
    thp : ThermalParams
    dt  : float
        Integration time step [s].

    Returns
    -------
    ca.Function  with signature  (x[3], u[3]) -> x_next[3]
    """
    f = build_casadi_dynamics(bp, thp)

    x = ca.MX.sym("x", 3)
    u = ca.MX.sym("u", 3)

    k1 = f(x, u)
    k2 = f(x + dt / 2.0 * k1, u)
    k3 = f(x + dt / 2.0 * k2, u)
    k4 = f(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return ca.Function("rk4_step", [x, u], [x_next], ["x", "u"], ["x_next"])


# ---------------------------------------------------------------------------
#  Numpy plant model for simulation
# ---------------------------------------------------------------------------

class BatteryPlant:
    """High-fidelity 3-state BESS plant integrated at ``dt_sim`` resolution.

    The plant uses RK4 internally and generates noisy measurements for
    SOC and Temperature.  SOH is a hidden state — it is never directly
    measured.

    Parameters
    ----------
    bp  : BatteryParams
    tp  : TimeParams
    thp : ThermalParams
    seed : int
        Random seed for measurement noise.
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        thp: ThermalParams,
        seed: int = 42,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.thp = thp
        self._rng = np.random.default_rng(seed)

        # True state  [SOC, SOH, T]
        self._x = np.array(
            [bp.SOC_init, bp.SOH_init, thp.T_init], dtype=np.float64,
        )

        # Measurement noise standard deviations
        self._meas_std_soc = 0.01       # SOC noise  (sigma ~ sqrt(1e-4))
        self._meas_std_temp = 0.5       # Temperature noise  [degC]

    # ---- continuous-time ODE (numpy) ----
    def _ode(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        SOC, SOH, T = x[0], x[1], x[2]
        P_chg, P_dis, P_reg = u[0], u[1], u[2]
        thp = self.thp

        # SOC
        E_eff = SOH * self.bp.E_nom_kwh * 3600.0
        dSOC = (self.bp.eta_charge * P_chg - P_dis / self.bp.eta_discharge) / E_eff

        # Arrhenius factor
        T_ref_K = thp.T_ref + 273.15
        T_K = T + 273.15
        kappa = np.exp(thp.E_a / thp.R_gas * (1.0 / T_ref_K - 1.0 / T_K))

        # SOH
        P_total = P_chg + P_dis + abs(P_reg)
        dSOH = -self.bp.alpha_deg * kappa * P_total

        # Thermal
        I = P_total * 1000.0 / thp.V_nominal
        Q_joule = I ** 2 * thp.R_internal
        Q_cool = thp.h_cool * (T - thp.T_ambient)
        dT = (Q_joule - Q_cool) / thp.C_thermal

        return np.array([dSOC, dSOH, dT])

    # ---- single RK4 step (numpy) ----
    def _rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._ode(x, u)
        k2 = self._ode(x + dt / 2.0 * k1, u)
        k3 = self._ode(x + dt / 2.0 * k2, u)
        k4 = self._ode(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ---- public interface ----
    def step(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Integrate one ``dt_sim`` step and return (x_new, y_meas).

        Parameters
        ----------
        u : ndarray, shape (3,)
            [P_charge, P_discharge, P_reg]  (kW, all >= 0).

        Returns
        -------
        x_new : ndarray, shape (3,)
            Updated true state [SOC, SOH, T].
        y_meas : ndarray, shape (2,)
            Noisy measurement [SOC_meas, T_meas].
        """
        bp = self.bp

        # Clamp inputs to physical bounds
        u_clamped = np.array([
            np.clip(u[0], 0.0, bp.P_max_kw),
            np.clip(u[1], 0.0, bp.P_max_kw),
            np.clip(u[2], 0.0, bp.P_max_kw),
        ])

        x_new = self._rk4_step(self._x, u_clamped, self.tp.dt_sim)

        # --- SOC saturation with back-calculation ---
        if x_new[0] < bp.SOC_min:
            x_new[0] = bp.SOC_min
        elif x_new[0] > bp.SOC_max:
            x_new[0] = bp.SOC_max

        # SOH can only decrease; clamp to valid range
        x_new[1] = np.clip(x_new[1], 0.5, 1.0)

        # Temperature clamp to physical bounds
        x_new[2] = np.clip(x_new[2], -20.0, 80.0)

        self._x = x_new.copy()
        y_meas = self.get_measurement()
        return self._x.copy(), y_meas

    def get_measurement(self) -> np.ndarray:
        """Return noisy [SOC, Temperature] measurement (SOH is NOT measured).

        Returns
        -------
        y_meas : ndarray, shape (2,)
            [SOC_measured, T_measured]
        """
        noise_soc = self._rng.normal(0.0, self._meas_std_soc)
        noise_temp = self._rng.normal(0.0, self._meas_std_temp)
        soc_meas = float(np.clip(self._x[0] + noise_soc, 0.0, 1.0))
        t_meas = self._x[2] + noise_temp
        return np.array([soc_meas, t_meas])

    def get_state(self) -> np.ndarray:
        """Return the true state [SOC, SOH, T] (for logging only)."""
        return self._x.copy()

    def reset(
        self,
        soc: float | None = None,
        soh: float | None = None,
        temp: float | None = None,
    ) -> None:
        """Reset plant state."""
        self._x[0] = soc if soc is not None else self.bp.SOC_init
        self._x[1] = soh if soh is not None else self.bp.SOH_init
        self._x[2] = temp if temp is not None else self.thp.T_init
