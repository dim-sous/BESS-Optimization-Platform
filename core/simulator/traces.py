"""SimTraces — pre-allocated arrays for the linear simulator's recording.

The simulator stores everything that happens during the loop into a
SimTraces instance via a single `record(k, ...)` call. The accounting
ledger then reads the traces (and only the traces) to compute revenues,
SOH, etc. This makes the data flow trivially auditable: every dollar
in the result dict comes from a numbered slot in `traces`.

Power is recorded as **actually-applied** values from `plant.step()`,
NOT as commands. This is the Bug C fix: `power_applied[:, 0]` is the
post-clip net power, `power_applied[:, 1]` is the post-clip reg power.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimTraces:
    """Pre-allocated trace storage. One row per sim step.

    All arrays are pre-allocated at simulator startup so the loop never
    has to grow them. Layout matches the legacy v5 result dict for
    backward compatibility with visualization and the comparison harness.
    """
    n_sim_steps: int
    n_mpc_steps: int
    n_cells: int

    # Plant true states (n_sim_steps + 1, ...)
    soc_true: np.ndarray = field(init=False)
    soh_true: np.ndarray = field(init=False)
    temp_true: np.ndarray = field(init=False)
    vrc1_true: np.ndarray = field(init=False)
    vrc2_true: np.ndarray = field(init=False)
    vterm_true: np.ndarray = field(init=False)

    # Power actually applied to the plant (n_sim_steps, 2) [P_net, P_reg]
    # P_net > 0 = discharge, < 0 = charge.
    power_applied: np.ndarray = field(init=False)

    # Activation and PI delivery (n_sim_steps,)
    activation: np.ndarray = field(init=False)
    p_delivered: np.ndarray = field(init=False)

    # Hourly EMS plan storage (n_sim_steps,) — what was committed when
    p_reg_committed: np.ndarray = field(init=False)

    # Estimator states at MPC resolution (n_mpc_steps + 1, ...)
    soc_ekf: np.ndarray = field(init=False)
    soh_ekf: np.ndarray = field(init=False)
    temp_ekf: np.ndarray = field(init=False)
    vrc1_ekf: np.ndarray = field(init=False)
    vrc2_ekf: np.ndarray = field(init=False)

    # MPC instrumentation (n_mpc_steps,)
    mpc_solve_times: np.ndarray = field(init=False)
    est_solve_times: np.ndarray = field(init=False)
    mpc_solver_failures: int = field(init=False, default=0)

    # MPC base setpoints (n_mpc_steps, 2) [P_net, P_reg]
    setpoint_at_mpc: np.ndarray = field(init=False)
    soc_ref_at_mpc: np.ndarray = field(init=False)

    # Multi-cell pack arrays (n_cells, n_sim_steps + 1, ...)
    cell_socs: np.ndarray = field(init=False)
    cell_sohs: np.ndarray = field(init=False)
    cell_temps: np.ndarray = field(init=False)
    cell_vrc1s: np.ndarray = field(init=False)
    cell_vrc2s: np.ndarray = field(init=False)

    # EMS plan history (list of SOC reference arrays, one per EMS solve)
    ems_soc_refs: list = field(default_factory=list)

    # Phase 3 (2026-04-28): bidding tier — appended to once per EMS hour
    # ONLY when the strategy has a bidding_protocol wired. Empty for
    # every v5 strategy; keeps existing trace shape bit-identical.
    bid_books_per_hour: list = field(default_factory=list)
    awards_per_hour: list = field(default_factory=list)

    def __post_init__(self) -> None:
        N = self.n_sim_steps
        M = self.n_mpc_steps
        C = self.n_cells

        self.soc_true = np.zeros(N + 1)
        self.soh_true = np.zeros(N + 1)
        self.temp_true = np.zeros(N + 1)
        self.vrc1_true = np.zeros(N + 1)
        self.vrc2_true = np.zeros(N + 1)
        self.vterm_true = np.zeros(N + 1)

        self.power_applied = np.zeros((N, 2))
        self.activation = np.zeros(N)
        self.p_delivered = np.zeros(N)
        self.p_reg_committed = np.zeros(N)

        self.soc_ekf = np.zeros(M + 1)
        self.soh_ekf = np.zeros(M + 1)
        self.temp_ekf = np.zeros(M + 1)
        self.vrc1_ekf = np.zeros(M + 1)
        self.vrc2_ekf = np.zeros(M + 1)

        self.mpc_solve_times = np.zeros(M)
        self.est_solve_times = np.zeros(M)
        self.mpc_solver_failures = 0

        self.setpoint_at_mpc = np.zeros((M, 2))
        self.soc_ref_at_mpc = np.zeros(M)

        self.cell_socs = np.zeros((C, N + 1))
        self.cell_sohs = np.zeros((C, N + 1))
        self.cell_temps = np.zeros((C, N + 1))
        self.cell_vrc1s = np.zeros((C, N + 1))
        self.cell_vrc2s = np.zeros((C, N + 1))

    def record_initial_state(self, x0: np.ndarray, vterm: float, cells: np.ndarray | None = None) -> None:
        """Record the initial plant state at index 0 of every state array."""
        self.soc_true[0] = x0[0]
        self.soh_true[0] = x0[1]
        self.temp_true[0] = x0[2]
        self.vrc1_true[0] = x0[3]
        self.vrc2_true[0] = x0[4]
        self.vterm_true[0] = vterm
        if cells is not None:
            self.cell_socs[:, 0] = cells[:, 0]
            self.cell_sohs[:, 0] = cells[:, 1]
            self.cell_temps[:, 0] = cells[:, 2]
            self.cell_vrc1s[:, 0] = cells[:, 3]
            self.cell_vrc2s[:, 0] = cells[:, 4]

    def record_step(
        self,
        k: int,
        u_applied: np.ndarray,           # (2,) [P_net, P_reg] from plant
        p_delivered: float,
        x_new: np.ndarray,               # (5,) plant state after step
        vterm_new: float,
        activation_k: float,
        p_reg_committed_k: float,
        cells: np.ndarray | None = None,
    ) -> None:
        """Record one PI/plant sub-step."""
        self.power_applied[k] = u_applied
        self.activation[k] = activation_k
        self.p_delivered[k] = p_delivered
        self.p_reg_committed[k] = p_reg_committed_k

        self.soc_true[k + 1] = x_new[0]
        self.soh_true[k + 1] = x_new[1]
        self.temp_true[k + 1] = x_new[2]
        self.vrc1_true[k + 1] = x_new[3]
        self.vrc2_true[k + 1] = x_new[4]
        self.vterm_true[k + 1] = vterm_new

        if cells is not None:
            self.cell_socs[:, k + 1] = cells[:, 0]
            self.cell_sohs[:, k + 1] = cells[:, 1]
            self.cell_temps[:, k + 1] = cells[:, 2]
            self.cell_vrc1s[:, k + 1] = cells[:, 3]
            self.cell_vrc2s[:, k + 1] = cells[:, 4]

    def record_mpc(
        self,
        m: int,
        ekf_state: np.ndarray,
        setpoint_pnet: float,
        setpoint_preg: float,
        soc_anchor: float,
        solve_time_s: float,
        est_time_s: float,
        solver_failed: bool,
    ) -> None:
        """Record one MPC sub-step."""
        self.soc_ekf[m] = ekf_state[0]
        self.soh_ekf[m] = ekf_state[1]
        self.temp_ekf[m] = ekf_state[2]
        self.vrc1_ekf[m] = ekf_state[3]
        self.vrc2_ekf[m] = ekf_state[4]

        self.setpoint_at_mpc[m] = (setpoint_pnet, setpoint_preg)
        self.soc_ref_at_mpc[m] = soc_anchor
        self.mpc_solve_times[m] = solve_time_s
        self.est_solve_times[m] = est_time_s
        if solver_failed:
            self.mpc_solver_failures += 1
