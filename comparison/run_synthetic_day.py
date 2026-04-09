"""Run all 5 v5 strategies on the synthetic 1-day dataset and print the ranking.

Usage:
    uv run python comparison/run_synthetic_day.py
"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config.parameters import (  # noqa: E402
    BatteryParams,
    EKFParams,
    ElectricalParams,
    EMSParams,
    MPCParams,
    PackParams,
    RegulationParams,
    ThermalParams,
    TimeParams,
)
from core.simulator.core import run_simulation  # noqa: E402
from core.simulator.synthetic_day import make_synthetic_day  # noqa: E402

from strategies.deterministic_lp.strategy import make_strategy as _ms_lp  # noqa: E402
from strategies.ems.strategy import make_strategy as _ms_ems  # noqa: E402
from strategies.ems_economic_mpc.strategy import make_strategy as _ms_econ  # noqa: E402
from strategies.ems_tracking_mpc.strategy import make_strategy as _ms_track  # noqa: E402
from strategies.rule_based.strategy import make_strategy as _ms_rb  # noqa: E402

STRATEGY_FACTORIES = [
    ("rule_based",       _ms_rb),
    ("deterministic_lp", _ms_lp),
    ("ems",              _ms_ems),
    ("ems_tracking_mpc", _ms_track),
    ("ems_economic_mpc", _ms_econ),
]


def main() -> None:
    day = make_synthetic_day()

    bp = BatteryParams()
    tp = TimeParams()
    ep = EMSParams()
    mp = MPCParams()
    ekf_p = EKFParams()
    thp = ThermalParams()
    elp = ElectricalParams()
    pp = PackParams()
    reg_p = RegulationParams(activation_seed=day.recommended_activation_seed)

    params = dict(
        bp=bp, tp=tp, ep=ep, mp=mp, ekf_p=ekf_p,
        thp=thp, elp=elp, reg_p=reg_p, pp=pp,
    )

    rows = []
    for name, factory in STRATEGY_FACTORIES:
        strategy = factory(**params)
        results = run_simulation(
            strategy=strategy,
            forecast_e=day.forecast_e,
            forecast_r=day.forecast_r,
            probabilities=day.probabilities,
            realized_e_prices=day.realized_e_prices,
            realized_r_prices=day.realized_r_prices,
            **params,
        )
        rows.append({
            "name": name,
            "total":  float(results["total_profit"]),
            "energy": float(results["energy_profit_total"]),
            "cap":    float(results["capacity_revenue"]),
            "deliv":  float(results["delivery_revenue"]),
            "pen":    float(results["penalty_cost"]),
            "deg":    float(results["deg_cost_total"]),
        })

    print(f"\n{'strategy':<20} {'total':>10} {'energy':>10} {'cap':>10} "
          f"{'deliv':>10} {'pen':>10} {'deg':>10}")
    print("-" * 82)
    for r in rows:
        print(f"{r['name']:<20} {r['total']:>10.2f} {r['energy']:>10.2f} "
              f"{r['cap']:>10.2f} {r['deliv']:>10.2f} {r['pen']:>10.2f} "
              f"{r['deg']:>10.2f}")

    order = [r["name"] for r in sorted(rows, key=lambda r: r["total"])]
    expected = ["rule_based", "deterministic_lp", "ems",
                "ems_tracking_mpc", "ems_economic_mpc"]
    print(f"\nrealized order (worst -> best): {order}")
    print(f"target order                  : {expected}")
    print(f"strict ranking achieved       : {order == expected}")


if __name__ == "__main__":
    main()
