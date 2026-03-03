"""Configuration parameters for the hierarchical BESS control platform.

v2_thermal_model: adds ThermalParams and temperature-related tuning
to EKF, MHE, and MPC parameter classes.

All physical units are SI-consistent and explicitly documented:
  - Energy:  kWh
  - Power:   kW
  - Time:    seconds (except sim_hours)
  - Price:   $/kWh  (energy),  $/kW/h  (regulation)
  - SOC/SOH: dimensionless  [0, 1]
  - Temperature: degC
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryParams:
    """Physical parameters of the battery energy storage system with degradation.

    Degradation model (v2: thermally coupled via Arrhenius factor)
    ---------------------------------------------------------------
    dSOH/dt = -alpha_deg * kappa(T) * (P_chg + P_dis + |P_reg|)

    kappa(T) = exp(E_a / R_gas * (1/T_ref_K - 1/T_K))

    At T_ref (25 degC): kappa = 1.0 (identical to v1 baseline).
    At 45 degC: kappa ~ 1.66 (66 % faster degradation).
    """

    E_nom_kwh: float = 200.0           # Nominal energy capacity  [kWh]
    P_max_kw: float = 100.0            # Maximum charge / discharge power  [kW]
    SOC_min: float = 0.10              # Minimum allowable state of charge  [-]
    SOC_max: float = 0.90              # Maximum allowable state of charge  [-]
    SOC_init: float = 0.50             # Initial state of charge  [-]
    SOH_init: float = 1.00             # Initial state of health  [-]
    SOC_terminal: float = 0.50         # Terminal SOC target for EMS  [-]
    eta_charge: float = 0.95           # Charging efficiency  [-]
    eta_discharge: float = 0.95        # Discharging efficiency  [-]
    alpha_deg: float = 2.78e-9         # Degradation rate  [1/(kW*s)]


@dataclass(frozen=True)
class ThermalParams:
    """Lumped-parameter thermal model for a 200 kWh / 100 kW utility-scale BESS.

    Thermal dynamics
    ----------------
    dT/dt = (I^2 * R_internal - h_cool * (T - T_ambient)) / C_thermal   [degC/s]

    Current derived from total power:
        I = P_total_kW * 1000 / V_nominal   [A]

    At 100 kW: I = 125 A, Q_joule = 156.25 W, steady-state dT = 3.1 degC.
    Thermal time constant = C_thermal / h_cool = 3000 s = 50 min.

    Arrhenius coupling
    ------------------
    kappa(T) = exp(E_a / R_gas * (1/T_ref_K - 1/T_K))

    At 25 degC: kappa = 1.00.  At 35 degC: kappa = 1.30.  At 45 degC: kappa = 1.66.
    """

    R_internal: float = 0.010          # Internal resistance  [Ohm]
    C_thermal: float = 150_000.0       # Thermal mass (heat capacity)  [J/K]
    h_cool: float = 50.0               # Cooling coefficient  [W/K]
    T_ambient: float = 25.0            # Ambient temperature  [degC]
    T_init: float = 25.0               # Initial cell temperature  [degC]
    T_max: float = 45.0                # Maximum allowable temperature  [degC]
    T_min: float = 5.0                 # Minimum allowable temperature  [degC]
    V_nominal: float = 800.0           # Nominal pack voltage  [V]
    E_a: float = 20_000.0             # Arrhenius activation energy  [J/mol]
    R_gas: float = 8.314               # Universal gas constant  [J/(mol*K)]
    T_ref: float = 25.0                # Reference temperature for Arrhenius  [degC]


@dataclass(frozen=True)
class TimeParams:
    """Time discretisation for multi-rate control.

    Every time quantity is in **seconds** except ``sim_hours``.
    """

    dt_ems: float = 3600.0             # EMS sampling period  [s]
    dt_mpc: float = 60.0               # MPC sampling period  [s]
    dt_estimator: float = 60.0         # Estimator sampling period  [s]  (= dt_mpc)
    dt_sim: float = 1.0                # Plant simulation step  [s]
    sim_hours: float = 24.0            # Total simulation duration  [hours]


@dataclass(frozen=True)
class EMSParams:
    """Parameters for the stochastic Energy Management System optimizer."""

    N_ems: int = 24                    # Planning horizon  [hours / steps at dt_ems]
    Nc_ems: int = 24                   # Control horizon  (= N_ems)
    n_scenarios: int = 5               # Number of price scenarios
    regulation_fraction: float = 0.3   # Max fraction of P_max for regulation  [-]
    degradation_cost: float = 50.0     # Cost of SOH loss  [$/unit SOH lost]
    terminal_soc_weight: float = 1e4   # Terminal SOC deviation penalty
    terminal_soh_weight: float = 1e4   # Terminal SOH deviation penalty


@dataclass(frozen=True)
class MPCParams:
    """Tuning parameters for the nonlinear tracking MPC.

    v2: adds temperature tracking and soft temperature constraints.
    """

    N_mpc: int = 60                    # Prediction horizon  [steps at dt_mpc]
    Nc_mpc: int = 20                   # Control horizon  [steps at dt_mpc]
    Q_soc: float = 1e4                 # SOC tracking weight
    Q_soh: float = 1e2                 # SOH tracking weight
    Q_temp: float = 1e2                # Temperature tracking weight  [v2]
    R_power: float = 1.0               # Power reference tracking weight (per input)
    R_delta: float = 10.0              # Control rate-of-change penalty
    Q_terminal: float = 1e5            # Terminal SOC penalty
    slack_penalty: float = 1e6         # Soft SOC constraint violation penalty
    slack_penalty_temp: float = 1e5    # Soft temperature constraint penalty  [v2]
    n_blend_steps: int = 5             # EMS boundary reference smoothing  [MPC steps]


@dataclass(frozen=True)
class EKFParams:
    """Tuning parameters for the Extended Kalman Filter.

    v2: adds temperature process/measurement noise and initial uncertainty.
    """

    # Process noise covariance  Q  (3x3 diagonal in v2)
    q_soc: float = 1e-6               # SOC process noise variance
    q_soh: float = 1e-12              # SOH process noise variance (extremely slow dynamics)
    q_temp: float = 1e-4               # Temperature process noise variance  [degC^2]  [v2]

    # Measurement noise covariance  R  (2x2 diagonal in v2)
    r_soc_meas: float = 1e-4          # SOC measurement noise variance (sigma ~ 0.01)
    r_temp_meas: float = 0.25         # Temperature measurement noise variance  [degC^2, sigma=0.5]  [v2]

    # Initial state error covariance  P_0  (3x3 diagonal in v2)
    p0_soc: float = 1e-3              # Initial SOC uncertainty
    p0_soh: float = 1e-2              # Initial SOH uncertainty (larger — unknown)
    p0_temp: float = 1.0              # Initial temperature uncertainty  [degC^2]  [v2]


@dataclass(frozen=True)
class MHEParams:
    """Tuning parameters for Moving Horizon Estimation.

    v2: adds temperature arrival cost, measurement weight, and process noise weight.
    """

    N_mhe: int = 30                    # Estimation window  [steps at dt_estimator]

    # Arrival cost weights  (inverse of prior covariance)
    arrival_soc: float = 1e3           # Weight on SOC arrival cost
    arrival_soh: float = 1e4           # Weight on SOH arrival cost (high — SOH barely observable)
    arrival_temp: float = 1e2          # Weight on temperature arrival cost  [v2]

    # Stage cost weights
    w_soc_meas: float = 1e4            # SOC measurement residual weight
    w_temp_meas: float = 1e3           # Temperature measurement residual weight  [v2]
    w_process_soc: float = 1e2         # Process disturbance weight (SOC)
    w_process_soh: float = 1e8         # Process disturbance weight (SOH, very high — penalise SOH noise)
    w_process_temp: float = 1e3        # Temperature process disturbance weight  [v2]
