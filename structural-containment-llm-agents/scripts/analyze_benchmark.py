
"""Analyse benchmark results and generate research figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def main() -> None:
    csv_path = RESULTS / "benchmark.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {csv_path}. Run scripts/run_benchmarks.py first."
        )

    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {
        "topology",
        "policy",
        "budget",
        "seed",
        "baseline_compromise_rate",
        "intervention_compromise_rate",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark CSV is missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Benchmark CSV contains no experiment runs.")

    df["density"] = df["topology"].str.split("_").str[0]
    df["absolute_reduction"] = (
        df["baseline_compromise_rate"] - df["intervention_compromise_rate"]
    )

    # Summary table: policy performance at each containment budget.
    summary = (
        df.groupby(["policy", "budget"], as_index=False)
        .agg(
            mean_baseline_rate=("baseline_compromise_rate", "mean"),
            mean_intervention_rate=("intervention_compromise_rate", "mean"),
            mean_absolute_reduction=("absolute_reduction", "mean"),
            std_intervention_rate=("intervention_compromise_rate", "std"),
            run_count=("intervention_compromise_rate", "size"),
        )
    )
    summary.to_csv(RESULTS / "policy_summary.csv", index=False)

    # Figure 1: pooled intervention rate by containment budget.
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy, group in summary.groupby("policy"):
        group = group.sort_values("budget")
        ax.plot(
            group["budget"],
            group["mean_intervention_rate"],
            marker="o",
            label=policy.replace("_", " ").title(),
        )

    ax.set(
        title="Containment Budget vs. Compromise Rate",
        xlabel="Containment budget (agents)",
        ylabel="Mean intervention compromise rate",
    )
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "compromise_rate_by_budget.png", dpi=300)
    plt.close(fig)

    # Figure 2: mean reduction by topology density and policy.
    density_summary = (
        df.groupby(["density", "policy", "budget"], as_index=False)
        .agg(mean_reduction=("absolute_reduction", "mean"))
    )

    densities = ["sparse", "medium", "dense"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, density in zip(axes, densities, strict=True):
        subset = density_summary[density_summary["density"] == density]
        for policy, group in subset.groupby("policy"):
            group = group.sort_values("budget")
            ax.plot(
                group["budget"],
                group["mean_reduction"],
                marker="o",
                label=policy.replace("_", " ").title(),
            )

        ax.set_title(density.title())
        ax.set_xlabel("Containment budget")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean absolute reduction")
    axes[-1].legend()
    fig.suptitle("Containment Effectiveness by Network Density")
    fig.tight_layout()
    fig.savefig(FIGURES / "reduction_by_density.png", dpi=300)
    plt.close(fig)

    # Figure 3: topology-specific intervention outcomes.
    topology_summary = (
        df.groupby(["topology", "policy", "budget"], as_index=False)
        .agg(
            mean_intervention_rate=("intervention_compromise_rate", "mean"),
            mean_reduction=("absolute_reduction", "mean"),
        )
    )
    topology_summary.to_csv(RESULTS / "topology_summary.csv", index=False)

    print(f"Runs analysed: {len(df)}")
    print(f"Policies: {', '.join(sorted(df['policy'].unique()))}")
    print(f"Summary table: {RESULTS / 'policy_summary.csv'}")
    print(f"Topology table: {RESULTS / 'topology_summary.csv'}")
    print(f"Figures saved to: {FIGURES}")

    print("\nPolicy summary:")
    print(
        summary[
            ["policy", "budget", "mean_intervention_rate", "mean_absolute_reduction"]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
