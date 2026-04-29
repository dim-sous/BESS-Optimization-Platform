"""Greek dual-pricing imbalance settlement — unit tests.

Verifies the four-quadrant truth table of (sign(delta), system_direction)
and the price-multiplier rules described in
``core/markets/imbalance.py``.

Helping = sign(Δ) == system_direction. Helping is settled at the
marginal price; hurting is penalised directionally so that helping
always nets more than hurting.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.markets.imbalance import settle_imbalance, settle_imbalance_hourly

LAM = 0.10            # $/kWh marginal price
K = 1.25              # penalty multiplier
DT = 1.0              # 1 hour


def test_zero_delta_returns_zero():
    assert settle_imbalance(50.0, 50.0, +1, LAM, K, DT) == 0.0
    assert settle_imbalance(0.0, 0.0, 0, LAM, K, DT) == 0.0


def test_long_helping_paid_at_marginal():
    """delta > 0, system needs UP (+1): you helped — paid at λ."""
    s = settle_imbalance(awarded_kw=0.0, delivered_kw=10.0,
                         system_direction=+1,
                         lambda_marginal_dollar_per_kwh=LAM,
                         k_dual=K, dt_h=DT)
    assert s == pytest.approx(10.0 * LAM)


def test_short_helping_paid_at_marginal():
    """delta < 0, system needs DOWN (-1): you helped (made system shorter
    of long state). Settled at λ — you owe |Δ|·λ."""
    s = settle_imbalance(awarded_kw=10.0, delivered_kw=0.0,
                         system_direction=-1,
                         lambda_marginal_dollar_per_kwh=LAM,
                         k_dual=K, dt_h=DT)
    assert s == pytest.approx(-10.0 * LAM)


def test_long_hurting_paid_below_marginal():
    """delta > 0, system was already long (-1): you hurt — paid at λ/k.

    Helping at λ would have given you 1.0; hurting at λ/k = 0.8 in this
    fixture — strictly less than helping."""
    s = settle_imbalance(awarded_kw=0.0, delivered_kw=10.0,
                         system_direction=-1,
                         lambda_marginal_dollar_per_kwh=LAM,
                         k_dual=K, dt_h=DT)
    assert s == pytest.approx(10.0 * LAM / K)
    # Strictly worse than helping.
    helping = 10.0 * LAM
    assert s < helping


def test_short_hurting_charged_above_marginal():
    """delta < 0, system was already short (+1): you hurt — charged at λ·k.

    Helping at λ would have cost you 1.0; hurting at λ·k = 1.25 — strictly
    more painful than helping."""
    s = settle_imbalance(awarded_kw=10.0, delivered_kw=0.0,
                         system_direction=+1,
                         lambda_marginal_dollar_per_kwh=LAM,
                         k_dual=K, dt_h=DT)
    assert s == pytest.approx(-10.0 * LAM * K)
    helping = -10.0 * LAM
    assert s < helping        # more negative ⇒ pays more


def test_balanced_system_no_penalty():
    """system_direction = 0: no dual-pricing — settled at λ."""
    long_revenue = settle_imbalance(0.0, 10.0, 0, LAM, K, DT)
    short_revenue = settle_imbalance(10.0, 0.0, 0, LAM, K, DT)
    assert long_revenue == pytest.approx(+10.0 * LAM)
    assert short_revenue == pytest.approx(-10.0 * LAM)


def test_helping_strictly_dominates_hurting():
    """For every quadrant, helping nets strictly more than hurting at
    the same |Δ| and same marginal price. This is the dual-pricing
    invariant the formula must respect."""
    for delta_kw in (+5.0, -5.0):
        for system_dir in (+1, -1):
            helping = settle_imbalance(0.0, delta_kw, system_dir,
                                       LAM, K, DT) if (delta_kw > 0) == (system_dir > 0) \
                else None
            hurting = settle_imbalance(0.0, delta_kw, system_dir,
                                       LAM, K, DT) if (delta_kw > 0) != (system_dir > 0) \
                else None
            # Either helping or hurting was assigned per the truth table;
            # find each by re-querying.
            help_dir = +1 if delta_kw > 0 else -1
            hurt_dir = -help_dir
            r_help = settle_imbalance(0.0, delta_kw, help_dir, LAM, K, DT)
            r_hurt = settle_imbalance(0.0, delta_kw, hurt_dir, LAM, K, DT)
            assert r_help > r_hurt, (
                f"delta={delta_kw}: helping={r_help}, hurting={r_hurt}"
            )


def test_dt_h_scales_settlement_linearly():
    """Doubling settlement period doubles dollars (linear)."""
    s1 = settle_imbalance(0.0, 10.0, +1, LAM, K, dt_h=1.0)
    s2 = settle_imbalance(0.0, 10.0, +1, LAM, K, dt_h=2.0)
    assert s2 == pytest.approx(2.0 * s1)


def test_invalid_k_dual_raises():
    with pytest.raises(ValueError):
        settle_imbalance(0.0, 10.0, +1, LAM, k_dual=0.0)
    with pytest.raises(ValueError):
        settle_imbalance(0.0, 10.0, +1, LAM, k_dual=-0.1)


def test_vectorised_matches_scalar():
    awarded = np.array([0.0, 10.0, 0.0, 10.0])
    delivered = np.array([10.0, 0.0, 10.0, 0.0])
    sys_dir = np.array([+1, -1, -1, +1])
    lam = np.array([LAM] * 4)
    vec = settle_imbalance_hourly(awarded, delivered, sys_dir, lam, K, DT)
    scal = np.array([
        settle_imbalance(awarded[k], delivered[k], int(sys_dir[k]),
                         float(lam[k]), K, DT)
        for k in range(4)
    ])
    np.testing.assert_allclose(vec, scal, atol=1e-12)
