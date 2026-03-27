"""Stochastic FCR activation signal generator using a 3-state Markov chain.

States: IDLE (0), UP (+1), DOWN (-1).  Transitions occur at each dt_pi step
(4s by default).  When in UP or DOWN, the activation magnitude is drawn
from U(0.3, 1.0) to model realistic partial activations.

The output signal is in [-1, +1] and represents the fraction of committed
regulation capacity that the grid demands at each instant.

Usage
-----
    gen = ActivationSignalGenerator(reg_params)
    signal = gen.generate(n_steps=21600)   # 24h at 4s
"""

from __future__ import annotations

import numpy as np

from config.parameters import RegulationParams

# Markov chain state indices
_IDLE = 0
_UP = 1
_DOWN = 2


class ActivationSignalGenerator:
    """Stochastic FCR activation signal via 3-state Markov chain."""

    def __init__(
        self,
        reg_params: RegulationParams,
        dt: float = 4.0,
    ) -> None:
        self._rp = reg_params
        self._dt = dt
        self._rng = np.random.default_rng(reg_params.activation_seed)
        self._state = _IDLE
        self._transition_matrix = self._build_transition_matrix()

    def _build_transition_matrix(self) -> np.ndarray:
        """Build 3x3 row-stochastic transition matrix."""
        rp = self._rp
        T = np.zeros((3, 3))

        # From IDLE
        T[_IDLE, _UP] = rp.p_idle_to_up
        T[_IDLE, _DOWN] = rp.p_idle_to_down
        T[_IDLE, _IDLE] = 1.0 - T[_IDLE, _UP] - T[_IDLE, _DOWN]

        # From UP
        T[_UP, _IDLE] = rp.p_up_to_idle
        T[_UP, _DOWN] = rp.p_up_to_down
        T[_UP, _UP] = 1.0 - T[_UP, _IDLE] - T[_UP, _DOWN]

        # From DOWN
        T[_DOWN, _IDLE] = rp.p_down_to_idle
        T[_DOWN, _UP] = rp.p_down_to_up
        T[_DOWN, _DOWN] = 1.0 - T[_DOWN, _IDLE] - T[_DOWN, _UP]

        return T

    def generate(self, n_steps: int) -> np.ndarray:
        """Generate activation signal array of shape (n_steps,), values in [-1, +1].

        Each step: transition to next state, then output magnitude based on state.
        """
        signal = np.zeros(n_steps)

        for i in range(n_steps):
            # Transition
            row = self._transition_matrix[self._state]
            self._state = self._rng.choice(3, p=row)

            # Output
            if self._state == _IDLE:
                signal[i] = 0.0
            elif self._state == _UP:
                signal[i] = self._rng.uniform(0.3, 1.0)
            else:  # DOWN
                signal[i] = -self._rng.uniform(0.3, 1.0)

        return signal

    def reset(self, seed: int | None = None) -> None:
        """Reset chain to IDLE and optionally reseed RNG."""
        self._state = _IDLE
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng(self._rp.activation_seed)

    @property
    def transition_matrix(self) -> np.ndarray:
        """Return the 3x3 transition probability matrix (rows sum to 1)."""
        return self._transition_matrix.copy()
