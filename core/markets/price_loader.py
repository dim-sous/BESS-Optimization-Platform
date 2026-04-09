"""Load real European electricity prices and build scenario bundles.

Reads historical day-ahead prices (from Energy-Charts / ENTSO-E data) and
FCR capacity prices (from SMARD / Bundesnetzagentur) to construct stochastic
scenario bundles compatible with the EMS interface.

Data sources
------------
  Energy: Energy-Charts (Fraunhofer ISE) — DE-LU day-ahead, EPEX SPOT.
  Regulation: SMARD (Bundesnetzagentur) — FCR total system cost (filter 4998),
    converted to per-MW capacity price by dividing by Germany's FCR
    requirement (~620 MW).

Units
-----
  Energy input CSV:  EUR/MWh  → converted to $/kWh
  Regulation input:  $/kW/h   (pre-converted from SMARD data)
"""

from __future__ import annotations

import pathlib

import numpy as np

# --------------------------------------------------------------------------
#  Conversion constants
# --------------------------------------------------------------------------
EUR_TO_USD = 1.08          # Approximate EUR→USD (Q1 2024 average)
MWH_TO_KWH = 1_000.0      # 1 MWh = 1 000 kWh


class RealPriceLoader:
    """Load real prices and create EMS-compatible scenario bundles.

    Parameters
    ----------
    energy_csv : path-like
        CSV with columns ``unix_timestamp, price_eur_per_mwh``.
    reg_csv : path-like or None
        CSV with columns ``unix_timestamp, reg_price_usd_per_kw_h``.
        If None, regulation prices are synthesised (less accurate).
    seed : int
        Random seed for scenario sampling.
    eur_to_usd : float
        EUR→USD conversion rate.
    """

    def __init__(
        self,
        energy_csv: str | pathlib.Path,
        reg_csv: str | pathlib.Path | None = None,
        seed: int = 42,
        eur_to_usd: float = EUR_TO_USD,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._eur_to_usd = eur_to_usd

        raw = np.genfromtxt(str(energy_csv), delimiter=",", skip_header=1)
        self._timestamps = raw[:, 0].astype(np.int64)
        self._prices_eur_mwh = raw[:, 1]

        # Convert to $/kWh (platform unit)
        self._prices_usd_kwh = (
            self._prices_eur_mwh * self._eur_to_usd / MWH_TO_KWH
        )

        # Load real regulation prices if available
        if reg_csv is not None:
            reg_raw = np.genfromtxt(str(reg_csv), delimiter=",", skip_header=1)
            self._reg_prices_usd_kw_h = reg_raw[:, 1]  # Already in $/kW/h
            self._has_real_reg = True
        else:
            self._reg_prices_usd_kw_h = None
            self._has_real_reg = False

        # Split into complete 24-hour days (use minimum of energy/reg lengths)
        n_energy_days = len(self._prices_usd_kwh) // 24
        if self._has_real_reg:
            n_reg_days = len(self._reg_prices_usd_kw_h) // 24
            n_full_days = min(n_energy_days, n_reg_days)
            usable_reg = n_full_days * 24
            self._daily_reg = self._reg_prices_usd_kw_h[:usable_reg].reshape(
                n_full_days, 24
            )
        else:
            n_full_days = n_energy_days
            self._daily_reg = None

        usable = n_full_days * 24
        self._daily_prices = self._prices_usd_kwh[:usable].reshape(
            n_full_days, 24
        )
        self.n_days = n_full_days

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    def get_day(self, day_idx: int) -> np.ndarray:
        """Return energy prices for a single day [$/kWh], shape (24,)."""
        return self._daily_prices[day_idx].copy()

    def generate_scenarios_for_day(
        self,
        day_idx: int,
        n_hours: int = 48,
        n_scenarios: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build forecast scenarios + held-out realized prices for *day_idx*.

        Honest evaluation paradigm:
          - Forecast scenarios contain N draws from days **other than**
            day_idx. They are what every planning strategy (rule-based,
            LP, EMS, MPC) is allowed to see.
          - Realized prices contain the actual day_idx prices and are
            used **only** for profit accounting after the simulation
            has executed each strategy's plan.

        The realized day is **never** in the forecast set, eliminating
        the perfect-foresight information leak that v5 originally had.

        Returns
        -------
        forecast_e : (n_scenarios, n_hours) [$/kWh]
            Energy-price forecast scenarios from other historical days.
        forecast_r : (n_scenarios, n_hours) [$/kW/h]
            FCR-capacity-price forecast scenarios from other historical days.
        probabilities : (n_scenarios,)
            Equal weights — no forecast scenario is privileged.
        realized_e : (n_hours,) [$/kWh]
            Actual day_idx energy prices (held out, accounting only).
        realized_r : (n_hours,) [$/kW/h]
            Actual day_idx FCR-capacity prices (held out, accounting only).
        """
        # --- Realized day (held out) ---
        # Energy prices pass through unchanged INCLUDING negative hours.
        # Negative day-ahead clearing prices are real EU market reality
        # (~5 % of German hours in 2024) and they are exactly the hours
        # where battery arbitrage is most valuable — the market pays you
        # to charge. Every downstream solver (LP, EMS, EconomicMPC) and
        # the ledger handle negative energy prices correctly.
        # FCR capacity prices are clamped to >= 0; SMARD historically
        # never clears negative for capacity payments and the revenue
        # model assumes commitment payments are non-negative.
        actual_e48, actual_r48 = self._build_48h(day_idx)
        realized_e = actual_e48[:n_hours].copy()
        realized_r = np.maximum(actual_r48[:n_hours], 0.0)

        # --- Forecast scenarios: sample N OTHER historical days ---
        # day_idx and (day_idx + 1) are both excluded — the latter is the
        # realized lookahead used to build the 48h realized window, and we
        # don't want to leak it via a forecast scenario either.
        excluded = {day_idx, day_idx + 1 if day_idx + 1 < self.n_days else day_idx}
        other_days = [i for i in range(self.n_days) if i not in excluded]
        if n_scenarios > len(other_days):
            raise ValueError(
                f"Requested {n_scenarios} forecast scenarios but only "
                f"{len(other_days)} other days available."
            )
        chosen = self._rng.choice(other_days, size=n_scenarios, replace=False)

        forecast_e = np.zeros((n_scenarios, n_hours))
        forecast_r = np.zeros((n_scenarios, n_hours))
        for s_idx, d_idx in enumerate(chosen):
            alt_e48, alt_r48 = self._build_48h(int(d_idx))
            forecast_e[s_idx, :n_hours] = alt_e48[:n_hours]
            forecast_r[s_idx, :n_hours] = alt_r48[:n_hours]

        # Energy forecasts pass through unchanged (see realized note above);
        # reg forecasts clamped to >= 0.
        forecast_r = np.maximum(forecast_r, 0.0)

        # --- Equal probabilities (no scenario privileged) ---
        probs = np.full(n_scenarios, 1.0 / n_scenarios)

        return forecast_e, forecast_r, probs, realized_e, realized_r

    def sample_day_indices(self, n: int) -> np.ndarray:
        """Sample *n* day indices without replacement for Monte Carlo."""
        return self._rng.choice(self.n_days, size=min(n, self.n_days), replace=False)

    @property
    def has_real_regulation(self) -> bool:
        """Whether real regulation prices were loaded."""
        return self._has_real_reg

    @property
    def price_stats(self) -> dict:
        """Summary statistics of loaded price data."""
        p = self._prices_eur_mwh[:self.n_days * 24]
        stats = {
            "n_hours": int(self.n_days * 24),
            "n_days": self.n_days,
            "mean_eur_mwh": float(np.mean(p)),
            "median_eur_mwh": float(np.median(p)),
            "min_eur_mwh": float(np.min(p)),
            "max_eur_mwh": float(np.max(p)),
            "std_eur_mwh": float(np.std(p)),
            "pct_negative": float(np.mean(p < 0) * 100),
            "reg_data": "real FCR (SMARD)" if self._has_real_reg else "synthesised",
        }
        if self._has_real_reg:
            r = self._reg_prices_usd_kw_h[:self.n_days * 24]
            # Convert back to EUR/MW/h for display
            r_eur = r * 1000 / self._eur_to_usd
            stats["reg_mean_eur_mw_h"] = float(np.mean(r_eur))
            stats["reg_median_eur_mw_h"] = float(np.median(r_eur))
        return stats

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _build_48h(
        self, day_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build 48-hour energy + regulation price windows."""
        e_day = self._daily_prices[day_idx]
        next_idx = day_idx + 1 if day_idx + 1 < self.n_days else day_idx
        e_next = self._daily_prices[next_idx]
        energy_48 = np.concatenate([e_day, e_next])

        if self._has_real_reg:
            r_day = self._daily_reg[day_idx]
            r_next = self._daily_reg[next_idx]
            reg_48 = np.concatenate([r_day, r_next])
        else:
            # Fallback: synthesise from energy (less accurate)
            noise = self._rng.normal(0.0, 0.002, 48)
            reg_48 = 0.4 * energy_48 + 0.006 + noise
            reg_48 = np.maximum(reg_48, 0.002)

        return energy_48, reg_48
