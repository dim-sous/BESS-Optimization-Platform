"""Configuration parameters for the Battery Energy Storage Optimization Platform.

All physical units are SI-consistent:
  - Energy:  kWh
  - Power:   kW
  - Time:    hours
  - Price:   $/kWh
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryParams:
    """Physical parameters of the battery energy storage system."""

    E_max_kwh: float = 200.0        # Rated energy capacity [kWh]
    P_max_kw: float = 100.0         # Maximum charge / discharge power [kW]
    SOC_min: float = 0.10           # Minimum allowable state of charge [-]
    SOC_max: float = 0.90           # Maximum allowable state of charge [-]
    SOC_init: float = 0.50          # Initial state of charge [-]
    SOC_terminal: float = 0.50      # Terminal SOC target for EMS [-]
    eta_charge: float = 0.95        # Charging efficiency [-]
    eta_discharge: float = 0.95     # Discharging efficiency [-]
    dt_hours: float = 1.0           # Sampling / control period [hours]


@dataclass(frozen=True)
class EMSParams:
    """Parameters for the day-ahead Energy Management System optimizer."""

    horizon: int = 24               # Planning horizon [time steps]


@dataclass(frozen=True)
class MPCParams:
    """Tuning parameters for the tracking Model Predictive Controller."""

    horizon: int = 6                # Prediction horizon [time steps]
    Q_soc: float = 1000.0           # SOC tracking weight
    R_power: float = 1.0            # Power tracking weight
    Q_terminal: float = 5000.0      # Terminal SOC penalty
    slack_penalty: float = 1e5      # Soft-constraint violation penalty


@dataclass(frozen=True)
class SimParams:
    """Parameters for the closed-loop simulation."""

    n_steps: int = 24               # Number of simulation time steps
