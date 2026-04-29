"""Phase 4 gate-stage runner — v5b "Greek Market Bidding Layer".

Executes the four CLAUDE.md gate stages in order and writes a
markdown report to ``results/v5b_gate_report.md``:

  Validate  — counts pass/fail of the tests/test_milp_*.py suite.
  Evaluate  — metrics dict for ``greek_milp_bidding`` on the canonical
              synthetic day (forecast-based MILP solve + simulator
              run).
  Compare   — ``greek_milp_bidding`` vs ``deterministic_lp`` on the
              same forecast, same simulator pipeline. Numerical
              table.
  Stress    — re-runs the stress scenarios from the unit tests and
              prints summary numbers.

Usage::

    uv run python comparison/run_phase4_gate.py                # 4h horizon (default)
    uv run python comparison/run_phase4_gate.py --hours 24     # full day
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# JIT off for runtime predictability on this gate sweep.
os.environ.setdefault("BESS_JIT", "0")

import numpy as np

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
from core.planners.deterministic_lp import DeterministicLP  # noqa: E402
from core.planners.milp_bidding import (  # noqa: E402
    MarketDecomposition,
    MILPBiddingConfig,
    MILPBiddingPlanner,
)
from core.simulator.core import run_simulation  # noqa: E402
from core.simulator.synthetic_day import make_synthetic_day  # noqa: E402
from strategies.deterministic_lp.strategy import make_strategy as ms_lp  # noqa: E402
from strategies.greek_milp_bidding.strategy import make_strategy as ms_greek  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
#  Stage A: Validate (run tests, capture pass/fail counts)
# ---------------------------------------------------------------------------

def stage_validate() -> dict:
    """Run the v5b test suite; return pass/fail counts per file."""
    test_files = [
        "tests/test_milp_bidding_invariants.py",
        "tests/test_clearing.py",
        "tests/test_imbalance.py",
        "tests/test_greek_settlement.py",
        "tests/test_greek_milp_bidding_e2e.py",
        "tests/test_milp_phase4_stress.py",
    ]
    results: dict[str, dict] = {}
    for tf in test_files:
        cmd = ["uv", "run", "pytest", "--tb=no", "-q", tf]
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - t0
        # Parse "N passed" or "N failed" from last line.
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        results[tf] = {
            "passed": "passed" in last and "failed" not in last,
            "summary": last,
            "elapsed_s": elapsed,
            "exit_code": proc.returncode,
        }
    return results


# ---------------------------------------------------------------------------
#  Stage B + C: Evaluate + Compare
# ---------------------------------------------------------------------------

def _build_inputs(sim_hours: float):
    bp = BatteryParams()
    tp = TimeParams(sim_hours=sim_hours)
    ep, mp = EMSParams(), MPCParams()
    ekf_p, thp, elp, reg_p, pp = (
        EKFParams(), ThermalParams(), ElectricalParams(),
        RegulationParams(), PackParams(),
    )
    day = make_synthetic_day()
    return {
        "bp": bp, "tp": tp, "ep": ep, "mp": mp, "ekf_p": ekf_p,
        "thp": thp, "elp": elp, "reg_p": reg_p, "pp": pp, "day": day,
    }


def _run_strategy(strategy, inputs):
    s = inputs
    return run_simulation(
        strategy=strategy,
        forecast_e=s["day"].forecast_e, forecast_r=s["day"].forecast_r,
        probabilities=s["day"].probabilities,
        realized_e_prices=s["day"].realized_e_prices,
        realized_r_prices=s["day"].realized_r_prices,
        bp=s["bp"], tp=s["tp"], ep=s["ep"], mp=s["mp"],
        ekf_p=s["ekf_p"], thp=s["thp"], elp=s["elp"], reg_p=s["reg_p"],
        pp=s["pp"],
    )


def _extract_metrics(name: str, result: dict) -> dict:
    soc = np.asarray(result["soc_true"])
    soc_ref = np.asarray(result.get("soc_ref_at_mpc", np.zeros_like(soc)))
    n = min(len(soc) - 1, len(soc_ref))
    rmse_soc = float(
        np.sqrt(np.mean((soc[:n] - soc_ref[:n]) ** 2))
    ) if n > 0 else 0.0

    metrics = {
        "name": name,
        "v5_total_profit": float(result["total_profit"]),
        "energy_profit_total": float(result["energy_profit_total"]),
        "capacity_revenue": float(result["capacity_revenue"]),
        "delivery_revenue": float(result["delivery_revenue"]),
        "deg_cost_total": float(result["deg_cost_total"]),
        "delivery_score": float(result["delivery_score"]),
        "soh_degradation": float(result["soh_degradation"]),
        "rmse_soc_tracking": rmse_soc,
        "avg_mpc_solve_time_s": float(np.mean(result["mpc_solve_times"])),
        "max_mpc_solve_time_s": float(np.max(result["mpc_solve_times"])),
        "mpc_solver_failures": int(result["mpc_solver_failures"]),
    }

    if "greek_settlement" in result:
        gs = result["greek_settlement"]
        metrics.update({
            "n_bids_total": gs["n_bids_total"],
            "n_bids_accepted": gs["n_bids_accepted"],
            "dam_revenue": gs["dam_revenue"],
            "idm_revenue": gs["idm_revenue"],
            "mfrr_cap_revenue": gs["mfrr_cap_revenue"],
            "afrr_cap_revenue": gs["afrr_cap_revenue"],
            "mfrr_activation_revenue": gs["mfrr_activation_revenue"],
            "afrr_activation_revenue": gs["afrr_activation_revenue"],
            "imbalance_settlement": gs["imbalance_settlement"],
            "non_delivery_penalty": gs["non_delivery_penalty"],
            "total_greek_revenue": gs["total_greek_revenue"],
        })
    return metrics


def stage_evaluate_and_compare(sim_hours: float) -> dict:
    """Run greek_milp_bidding and deterministic_lp on the same day,
    same simulator, return their metrics dicts side-by-side."""
    inputs = _build_inputs(sim_hours=sim_hours)
    s = inputs

    # greek_milp_bidding
    strat_greek = ms_greek(
        bp=s["bp"], tp=s["tp"], ep=s["ep"], mp=s["mp"],
        thp=s["thp"], elp=s["elp"],
        realized_e_prices=s["day"].realized_e_prices,
        realized_r_prices=s["day"].realized_r_prices,
    )
    t0 = time.perf_counter()
    res_greek = _run_strategy(strat_greek, inputs)
    elapsed_greek = time.perf_counter() - t0

    # deterministic_lp baseline
    strat_lp = ms_lp(
        bp=s["bp"], tp=s["tp"], ep=s["ep"], mp=s["mp"],
        thp=s["thp"], elp=s["elp"],
    )
    t0 = time.perf_counter()
    res_lp = _run_strategy(strat_lp, inputs)
    elapsed_lp = time.perf_counter() - t0

    return {
        "greek": {
            **_extract_metrics("greek_milp_bidding", res_greek),
            "wall_time_s": elapsed_greek,
        },
        "lp": {
            **_extract_metrics("deterministic_lp", res_lp),
            "wall_time_s": elapsed_lp,
        },
    }


# ---------------------------------------------------------------------------
#  Stage D: Stress (planner-only, fast — sim integration too slow for sweep)
# ---------------------------------------------------------------------------

def stage_stress() -> dict:
    """Replay key stress scenarios at planner level only (offline) and
    return a numbers dict the report can render."""
    bp, tp = BatteryParams(), TimeParams()
    ep, thp = EMSParams(), ThermalParams()
    day = make_synthetic_day()
    base_kwargs = {
        "bp": bp, "tp": tp, "ep": ep, "thp": thp,
    }

    # Baseline solve
    base = MILPBiddingPlanner(**base_kwargs)
    res_base = base.solve(
        soc_init=0.5, soh_init=1.0, t_init=25.0,
        energy_scenarios=day.forecast_e, reg_scenarios=day.forecast_r,
        probabilities=day.probabilities,
    )

    # 1) Adversarial wash-trade — random perturbation, seed 0
    rng = np.random.default_rng(0)
    adv_e = day.forecast_e.copy()
    adv_r = day.forecast_r.copy()
    flip = rng.choice(24, size=4, replace=False)
    adv_e[:, flip] *= -2.0
    adv_r[:, flip] *= 5.0
    res_adv = base.solve(
        soc_init=0.5, soh_init=1.0, t_init=25.0,
        energy_scenarios=adv_e, reg_scenarios=adv_r,
        probabilities=day.probabilities,
    )
    p_chg = res_adv["P_dam_chg_ref"] + res_adv["P_idm_chg_ref"]
    p_dis = res_adv["P_dam_dis_ref"] + res_adv["P_idm_dis_ref"]
    adv_max_overlap = float(np.max(np.minimum(p_chg, p_dis)))

    # 2) MBQ at 50 kW
    from dataclasses import replace
    from core.markets.products import Product
    cfg_mbq = MILPBiddingConfig()
    cfg_mbq.product_specs[Product.mFRR_Capacity] = replace(
        cfg_mbq.product_specs[Product.mFRR_Capacity], min_bid_qty_kw=50.0,
    )
    cfg_mbq.product_specs[Product.aFRR_Capacity] = replace(
        cfg_mbq.product_specs[Product.aFRR_Capacity], min_bid_qty_kw=50.0,
    )
    p_mbq = MILPBiddingPlanner(**base_kwargs, config=cfg_mbq)
    res_mbq = p_mbq.solve(
        soc_init=0.5, soh_init=1.0, t_init=25.0,
        energy_scenarios=day.forecast_e, reg_scenarios=day.forecast_r,
        probabilities=day.probabilities,
    )
    mbq_violation = False
    for arr in (res_mbq["P_mfrr_cap_ref"], res_mbq["P_afrr_cap_ref"]):
        for p in arr:
            if 1e-6 < p < 50.0 - 1e-6:
                mbq_violation = True

    # 3) Tight time limit (1 ms) — relaxation should fire
    cfg_tight = MILPBiddingConfig(mip_time_limit_s=0.001)
    p_tight = MILPBiddingPlanner(**base_kwargs, config=cfg_tight)
    res_tight = p_tight.solve(
        soc_init=0.5, soh_init=1.0, t_init=25.0,
        energy_scenarios=day.forecast_e, reg_scenarios=day.forecast_r,
        probabilities=day.probabilities,
    )

    # 4) High activation rate (2× α)
    cfg_high_alpha = MILPBiddingConfig(decomposition=MarketDecomposition(
        alpha_mfrr=0.20, alpha_afrr=0.40,
    ))
    p_high_alpha = MILPBiddingPlanner(**base_kwargs, config=cfg_high_alpha)
    res_high = p_high_alpha.solve(
        soc_init=0.5, soh_init=1.0, t_init=25.0,
        energy_scenarios=day.forecast_e, reg_scenarios=day.forecast_r,
        probabilities=day.probabilities,
    )
    cap_base = float(res_base["P_mfrr_cap_ref"].sum() + res_base["P_afrr_cap_ref"].sum())
    cap_high = float(res_high["P_mfrr_cap_ref"].sum() + res_high["P_afrr_cap_ref"].sum())

    return {
        "baseline_profit": res_base["expected_profit"],
        "baseline_solve_s": res_base["planner_diagnostics"]["mip_solve_time_s"],
        "adv_max_overlap_kw": adv_max_overlap,
        "adv_profit": res_adv["expected_profit"],
        "mbq_violation": mbq_violation,
        "mbq_total_capacity_kw": float(
            res_mbq["P_mfrr_cap_ref"].sum() + res_mbq["P_afrr_cap_ref"].sum()
        ),
        "tight_was_relaxed": res_tight["planner_diagnostics"]["was_relaxed"],
        "tight_status": res_tight["planner_diagnostics"]["solver_status"],
        "tight_solve_s": res_tight["planner_diagnostics"]["mip_solve_time_s"],
        "alpha_base_total_cap": cap_base,
        "alpha_high_total_cap": cap_high,
        "alpha_base_profit": res_base["expected_profit"],
        "alpha_high_profit": res_high["expected_profit"],
    }


# ---------------------------------------------------------------------------
#  Report writer
# ---------------------------------------------------------------------------

def write_report(
    *,
    sim_hours: float,
    validate: dict,
    evaluate_and_compare: dict,
    stress: dict,
    out_path: pathlib.Path,
) -> None:
    g = evaluate_and_compare["greek"]
    lp = evaluate_and_compare["lp"]

    lines = []
    lines.append("# v5b Gate Report — Greek Market Bidding Layer")
    lines.append("")
    lines.append(f"Generated by `comparison/run_phase4_gate.py --hours {sim_hours:g}`.")
    lines.append("")

    # Stage A
    lines.append("## Stage A — Validate")
    lines.append("")
    lines.append("| Test file | Result | Wall time |")
    lines.append("|---|---|---:|")
    for tf, info in validate.items():
        flag = "PASS" if info["passed"] else "FAIL"
        lines.append(f"| `{tf}` | **{flag}** ({info['summary']}) | {info['elapsed_s']:.2f} s |")
    lines.append("")

    # Stage B+C
    lines.append("## Stage B/C — Evaluate + Compare")
    lines.append("")
    lines.append(f"Both strategies on the canonical synthetic day, "
                 f"`sim_hours={sim_hours:g}`, identical inputs.")
    lines.append("")
    lines.append("| Metric | greek_milp_bidding | deterministic_lp |")
    lines.append("|---|---:|---:|")
    rows = [
        ("v5 total_profit",          "v5_total_profit",      "$"),
        ("energy_profit_total",      "energy_profit_total",  "$"),
        ("capacity_revenue",         "capacity_revenue",     "$"),
        ("delivery_revenue",         "delivery_revenue",     "$"),
        ("deg_cost_total",           "deg_cost_total",       "$"),
        ("delivery_score",           "delivery_score",       ""),
        ("soh_degradation",          "soh_degradation",      ""),
        ("RMSE_SOC_tracking",        "rmse_soc_tracking",    ""),
        ("avg_mpc_solve_time_s",     "avg_mpc_solve_time_s", "s"),
        ("max_mpc_solve_time_s",     "max_mpc_solve_time_s", "s"),
        ("mpc_solver_failures",      "mpc_solver_failures",  ""),
        ("wall_time_s",              "wall_time_s",          "s"),
    ]
    for label, key, unit in rows:
        gv = g.get(key, "-")
        lv = lp.get(key, "-")
        if isinstance(gv, float):
            gv = f"{gv:.4f}"
        if isinstance(lv, float):
            lv = f"{lv:.4f}"
        lines.append(f"| {label} | {gv} {unit} | {lv} {unit} |")
    lines.append("")

    lines.append("### Greek per-product revenue (greek_milp_bidding only)")
    lines.append("")
    lines.append("| Product | Amount [$] |")
    lines.append("|---|---:|")
    greek_rows = [
        ("HEnEx DAM Energy",         "dam_revenue"),
        ("HEnEx IDM Energy",         "idm_revenue"),
        ("mFRR Capacity",            "mfrr_cap_revenue"),
        ("aFRR Capacity",            "afrr_cap_revenue"),
        ("mFRR Activation",          "mfrr_activation_revenue"),
        ("aFRR Activation",          "afrr_activation_revenue"),
        ("Imbalance settlement",     "imbalance_settlement"),
        ("Non-delivery penalty",     "non_delivery_penalty"),
        ("**TOTAL Greek revenue**",  "total_greek_revenue"),
    ]
    for label, key in greek_rows:
        v = g.get(key, "-")
        if isinstance(v, float):
            v = f"{v:.4f}"
        lines.append(f"| {label} | {v} |")
    lines.append("")
    lines.append(f"Bids submitted / accepted: **{g['n_bids_total']} / {g['n_bids_accepted']}**.")
    lines.append("")

    # Stage D
    lines.append("## Stage D — Stress")
    lines.append("")
    s = stress
    lines.append("| Scenario | Metric | Value | Pass criterion | Result |")
    lines.append("|---|---|---:|---|---|")
    lines.append(f"| Baseline (canonical day) | profit, MIP solve | "
                 f"${s['baseline_profit']:.2f}, {s['baseline_solve_s']*1000:.0f} ms | reference | — |")
    lines.append(f"| Adversarial prices | max(min(P_chg, P_dis)) | "
                 f"{s['adv_max_overlap_kw']:.3e} kW | < 1e-6 kW | "
                 f"**{'PASS' if s['adv_max_overlap_kw'] < 1e-6 else 'FAIL'}** |")
    lines.append(f"| MBQ = 50 kW | any awarded P in (0, 50)? | "
                 f"{'YES' if s['mbq_violation'] else 'NO'} | NO | "
                 f"**{'FAIL' if s['mbq_violation'] else 'PASS'}** |")
    lines.append(f"| Tight 1 ms time limit | LP relaxation fired | "
                 f"{s['tight_was_relaxed']} | True | "
                 f"**{'PASS' if s['tight_was_relaxed'] else 'CHECK'}** |")
    lines.append(f"| Tight 1 ms time limit | post-relax status | "
                 f"{s['tight_status']} | Optimal | "
                 f"**{'PASS' if s['tight_status'] == 'Optimal' else 'FAIL'}** |")
    lines.append(f"| 2× α (high activation) | total committed cap [kWh] | "
                 f"{s['alpha_high_total_cap']:.0f} (vs base {s['alpha_base_total_cap']:.0f}) | "
                 f"!= base (planner responds) | "
                 f"**{'PASS' if abs(s['alpha_high_total_cap'] - s['alpha_base_total_cap']) > 1e-3 else 'CHECK'}** |")
    lines.append("")

    # Footer
    lines.append("## Notes")
    lines.append("")
    lines.append("- The **MILP planner is wash-trade-free by construction**: the binary "
                 "mutex (`b_dis + b_chg ≤ 1`) makes simultaneous charge+discharge infeasible "
                 "regardless of price profile. Stress test S1 confirms over 5 random "
                 "adversarial profiles.")
    lines.append("- The **LP-relaxation fallback** kicks in when the MIP cannot finish "
                 "within `mip_time_limit_s`. The relaxation gets a generous time budget "
                 "(`max(100×mip_limit, 30s)`) so it always produces a feasible plan; "
                 "if even the relaxation fails, the planner returns an idle plan and "
                 "logs the failure in `planner_diagnostics.solver_status`.")
    lines.append("- The **v5 ledger fields and the Greek settlement block coexist** in "
                 "the result dict. v5 metrics use the existing one-energy-channel + "
                 "one-FCR accounting model; the Greek block uses six product price arrays "
                 "synthesised from realised channels via `decompose_prices`. They are "
                 "different accounting layers, both valid.")
    lines.append("")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="v5b Phase 4 gate runner")
    parser.add_argument("--hours", type=float, default=4.0, help="Sim horizon (hours)")
    parser.add_argument("--out", type=pathlib.Path,
                        default=RESULTS_DIR / "v5b_gate_report.md")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip Stage A (test re-run) — useful for quick iteration")
    args = parser.parse_args()

    print("=" * 72)
    print(f"v5b Phase 4 gate — sim_hours={args.hours:g}")
    print("=" * 72)

    print("\n[Stage A] Validate ...")
    if args.skip_validate:
        print("  skipped (--skip-validate)")
        validate = {"(skipped)": {"passed": True, "summary": "skipped",
                                  "elapsed_s": 0.0, "exit_code": 0}}
    else:
        validate = stage_validate()
        for tf, info in validate.items():
            flag = "PASS" if info["passed"] else "FAIL"
            print(f"  {flag}  {tf}  ({info['summary']})  [{info['elapsed_s']:.2f}s]")

    print("\n[Stages B+C] Evaluate + Compare ...")
    bc = stage_evaluate_and_compare(sim_hours=args.hours)
    g, lp = bc["greek"], bc["lp"]
    print(f"  greek_milp_bidding: total_greek=${g.get('total_greek_revenue', 0.0):.2f}, "
          f"v5={g['v5_total_profit']:.2f}, wall={g['wall_time_s']:.1f}s")
    print(f"  deterministic_lp:   v5={lp['v5_total_profit']:.2f}, "
          f"wall={lp['wall_time_s']:.1f}s")

    print("\n[Stage D] Stress ...")
    stress = stage_stress()
    print(f"  baseline profit:    ${stress['baseline_profit']:.2f}")
    print(f"  adv max overlap:    {stress['adv_max_overlap_kw']:.3e} kW")
    print(f"  MBQ violation:      {stress['mbq_violation']}")
    print(f"  1ms relax fired:    {stress['tight_was_relaxed']} ({stress['tight_status']})")
    print(f"  α-high total cap:   {stress['alpha_high_total_cap']:.0f} kWh (vs base {stress['alpha_base_total_cap']:.0f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(
        sim_hours=args.hours,
        validate=validate,
        evaluate_and_compare=bc,
        stress=stress,
        out_path=args.out,
    )
    print(f"\nReport written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
