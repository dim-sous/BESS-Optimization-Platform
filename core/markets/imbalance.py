"""Greek dual-pricing imbalance settlement.

Pure functions that translate a per-hour delivery deviation into a
settlement amount under a directional dual-pricing penalty model. No
state, no I/O — easy to unit-test and to plug into the per-product
revenue aggregator in ``core/accounting/greek_settlement.py``.

Dual-pricing model
------------------
Every hour the system has a direction:

  system_direction = +1  : system is short — needs UP regulation
                           (extra discharge); long BSPs help.
  system_direction = -1  : system is long  — needs DOWN regulation
                           (extra charge); short BSPs help.
  system_direction =  0  : balanced       — no penalty applied.

A BSP's per-hour imbalance ``Δ = delivered − awarded`` interacts with
the system direction in four cases. Helping = ``sign(Δ) == direction``
(your imbalance reduces the system imbalance). Hurting = the opposite.

Helping is settled at the marginal domain price ``λ`` regardless of
sign — long deliveries earn ``+|Δ|·λ``, short deliveries owe ``−|Δ|·λ``.
Hurting is penalised *directionally*: a long BSP that hurts the system
gets ``λ/k_dual`` (less revenue than fair), and a short BSP that hurts
gets ``λ·k_dual`` (more cost than fair). Both directions converge to
"helping wins, hurting loses" relative to the marginal price.

This is the simplification of the real Greek IST_UP / IST_DN scheme
appropriate for a Phase-2 offline layer; a real implementation would
read the per-hour ``IST_UP / IST_DN`` directly from ADMIE feeds.
"""

from __future__ import annotations

import numpy as np


def settle_imbalance(
    awarded_kw: float,
    delivered_kw: float,
    system_direction: int,
    lambda_marginal_dollar_per_kwh: float,
    k_dual: float = 1.25,
    dt_h: float = 1.0,
) -> float:
    """Compute the settlement (revenue, signed) for one BSP-hour.

    Parameters
    ----------
    awarded_kw, delivered_kw : float
        Power scheduled at gate closure vs. power physically delivered
        over the settlement period (kW).
    system_direction : int
        +1, -1, or 0. See module docstring.
    lambda_marginal_dollar_per_kwh : float
        Reference (DAM-domain) price for the hour, LP scale.
    k_dual : float
        Penalty multiplier for hurting BSPs. Real Greek values are in
        ``[1.05, 1.5]``; default ``1.25`` is a midpoint.
    dt_h : float
        Settlement interval in hours.

    Returns
    -------
    settlement : float
        Net dollars credited to the BSP (positive = paid, negative =
        owed). Zero exactly if the BSP delivered to plan.
    """
    if k_dual <= 0:
        raise ValueError(f"k_dual must be > 0, got {k_dual}")

    delta_kwh = (delivered_kw - awarded_kw) * dt_h
    if delta_kwh == 0.0:
        return 0.0
    if system_direction == 0:
        # No directional info: settle at marginal — clean baseline.
        return float(delta_kwh * lambda_marginal_dollar_per_kwh)

    helping = (delta_kwh > 0) == (system_direction > 0)
    if helping:
        return float(delta_kwh * lambda_marginal_dollar_per_kwh)

    # Hurting: penalty applied directionally so helping always wins.
    if delta_kwh > 0:
        # Long BSP, system already long: paid less than fair.
        return float(delta_kwh * lambda_marginal_dollar_per_kwh / k_dual)
    # Short BSP, system already short: pays more than fair.
    return float(delta_kwh * lambda_marginal_dollar_per_kwh * k_dual)


def settle_imbalance_hourly(
    awarded_kw: np.ndarray,
    delivered_kw: np.ndarray,
    system_direction: np.ndarray,
    lambda_marginal_dollar_per_kwh: np.ndarray,
    k_dual: float = 1.25,
    dt_h: float = 1.0,
) -> np.ndarray:
    """Vectorised version: per-hour settlement array of length ``N``."""
    awarded_kw = np.asarray(awarded_kw, dtype=float)
    delivered_kw = np.asarray(delivered_kw, dtype=float)
    system_direction = np.asarray(system_direction, dtype=int)
    lam = np.asarray(lambda_marginal_dollar_per_kwh, dtype=float)
    return np.array([
        settle_imbalance(
            awarded_kw=float(awarded_kw[k]),
            delivered_kw=float(delivered_kw[k]),
            system_direction=int(system_direction[k]),
            lambda_marginal_dollar_per_kwh=float(lam[k]),
            k_dual=k_dual,
            dt_h=dt_h,
        )
        for k in range(len(awarded_kw))
    ], dtype=float)
