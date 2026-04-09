"""Shared bits between TrackingMPC and EconomicMPC.

Both NLPs use the same IPOPT options, the same JIT toggle, and the same
right-pad-with-last-value helper for horizon arrays. Kept here so the
two MPC files don't drift.
"""

from __future__ import annotations

import os

import numpy as np


def pad_to(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad (or truncate) `arr` to `target_len` by repeating the last value."""
    if len(arr) >= target_len:
        return arr[:target_len]
    pad_val = arr[-1] if len(arr) > 0 else 0.0
    return np.concatenate([arr, np.full(target_len - len(arr), pad_val)])


def ipopt_opts() -> dict:
    """Standard IPOPT options for both MPC NLPs.

    JIT codegen is enabled by default (BESS_JIT=0 to disable for A/B
    equivalence checks). HSL MA57 is intentionally NOT enabled — no
    academic license — so MUMPS remains the linear solver.
    """
    opts = {
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
        "ipopt.max_iter": 500,
        "ipopt.tol": 1e-6,
        "ipopt.acceptable_tol": 1e-4,
        "ipopt.warm_start_init_point": "yes",
        "ipopt.mu_init": 1e-3,
    }
    if os.environ.get("BESS_JIT", "1") != "0":
        opts.update({
            "jit": True,
            "compiler": "shell",
            "jit_options": {"flags": ["-O3", "-march=native"], "verbose": False},
        })
    return opts
