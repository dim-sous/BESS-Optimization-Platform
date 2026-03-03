"""Compare metrics across platform versions.

Usage (from repo root):
    python -m comparison.compare_versions

Reads all ``results/*_metrics.json`` files and produces:
  - Console table
  - ``results/version_comparison.csv``
  - ``results/version_comparison.png``
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"

METRIC_KEYS = [
    "version",
    "rmse_soc_tracking",
    "rmse_power_tracking",
    "rmse_soc_ekf",
    "rmse_soc_mhe",
    "rmse_soh_ekf",
    "rmse_soh_mhe",
    "total_profit_usd",
    "total_degradation_cost_usd",
    "avg_mpc_solve_time_s",
    "max_mpc_solve_time_s",
    "avg_estimator_solve_time_s",
    "max_estimator_solve_time_s",
    "final_soc",
    "final_soh",
    "soh_degradation_pct",
]


def _fmt(val: float | str, width: int = 12) -> str:
    """Format a metric value for display."""
    if isinstance(val, str):
        return val.ljust(width)
    if np.isnan(val):
        return "N/A".center(width)
    if abs(val) < 0.001:
        return f"{val:.2e}".rjust(width)
    return f"{val:.4f}".rjust(width)


def main() -> None:
    """Discover metrics files, print comparison, write CSV and plot."""
    metrics_files = sorted(RESULTS_DIR.glob("*_metrics.json"))
    if not metrics_files:
        print("No metrics files found in", RESULTS_DIR)
        sys.exit(1)

    all_metrics: list[dict] = []
    for mf in metrics_files:
        with open(mf) as f:
            all_metrics.append(json.load(f))

    # ---- Console table ----
    header = "  ".join(
        [f"{'Metric':<30}"]
        + [f"{m['version']:>16}" for m in all_metrics]
    )
    print("\n" + "=" * len(header))
    print("  VERSION COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for key in METRIC_KEYS:
        if key == "version":
            continue
        row = f"{key:<30}"
        for m in all_metrics:
            row += f"  {_fmt(m.get(key, float('nan')), 16)}"
        print(row)
    print("=" * len(header))

    # ---- CSV ----
    csv_path = RESULTS_DIR / "version_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_KEYS)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: m.get(k, "") for k in METRIC_KEYS})
    print(f"\nCSV saved to {csv_path}")

    # ---- Bar chart comparison ----
    plot_keys = [
        ("rmse_soc_tracking", "RMSE SOC Tracking"),
        ("rmse_soc_ekf", "RMSE SOC (EKF)"),
        ("rmse_soh_ekf", "RMSE SOH (EKF)"),
        ("total_profit_usd", "Total Profit [$]"),
        ("total_degradation_cost_usd", "Degradation Cost [$]"),
        ("avg_mpc_solve_time_s", "Avg MPC Solve [s]"),
        ("soh_degradation_pct", "SOH Degradation [%]"),
    ]

    versions = [m["version"] for m in all_metrics]
    n_metrics = len(plot_keys)
    n_versions = len(versions)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes_flat = axes.flatten()

    for i, (key, label) in enumerate(plot_keys):
        ax = axes_flat[i]
        vals = [m.get(key, float("nan")) for m in all_metrics]
        colours = plt.cm.tab10(np.linspace(0, 1, n_versions))
        bars = ax.bar(versions, vals, color=colours[:n_versions], edgecolor="k", linewidth=0.5)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.4g}",
                    ha="center", va="bottom", fontsize=7,
                )

    # Hide unused subplots
    for j in range(n_metrics, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Version Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "version_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved to {plot_path}")


if __name__ == "__main__":
    main()
