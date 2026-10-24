
"""Paired robustness analysis of containment policies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int = 10_000,
) -> tuple[float, float]:
    """Return a percentile bootstrap 95% confidence interval for a mean."""
    if values.size == 0:
        raise ValueError("Cannot calculate a confidence interval without data.")

    samples = rng.choice(values, size=(n_bootstrap, values.size), replace=True)
    means = samples.mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def main() -> None:
    source = RESULTS / "benchmark.csv"
    if not source.exists():
        raise FileNotFoundError(f"Benchmark results not found: {source}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source)

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
    df["reduction"] = (
        df["baseline_compromise_rate"] - df["intervention_compromise_rate"]
    )

    # Average over simulation seeds first. Each topology is then one
    # paired observation for each policy and budget.
    per_graph = (
        df.groupby(["topology", "density", "policy", "budget"], as_index=False)
        .agg(mean_reduction=("reduction", "mean"))
    )

    paired = per_graph.pivot(
        index=["topology", "density", "budget"],
        columns="policy",
        values="mean_reduction",
    ).reset_index()

    expected_policies = {"random", "highest_out_degree"}
    if not expected_policies.issubset(paired.columns):
        raise ValueError("Results must contain both expected containment policies.")

    paired["paired_difference"] = (
        paired["highest_out_degree"] - paired["random"]
    )

    rng = np.random.default_rng(20261023)
    summary_rows = []

    # Overall comparison and comparisons within each graph-density group.
    for scope, subset in [
        ("all", paired),
        *[
            (density, paired[paired["density"] == density])
            for density in ("sparse", "medium", "dense")
        ],
    ]:
        for budget, group in subset.groupby("budget"):
            differences = group["paired_difference"].to_numpy(dtype=float)
            lower, upper = bootstrap_mean_ci(differences, rng=rng)

            summary_rows.append(
                {
                    "scope": scope,
                    "budget": int(budget),
                    "graph_count": int(differences.size),
                    "mean_highest_degree_reduction": float(
                        group["highest_out_degree"].mean()
                    ),
                    "mean_random_reduction": float(group["random"].mean()),
                    "mean_paired_difference": float(differences.mean()),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "paired_robustness_summary.csv", index=False)

    # Plot the overall paired difference and its confidence interval.
    overall = summary[summary["scope"] == "all"].sort_values("budget")
    x = overall["budget"].to_numpy()
    means = overall["mean_paired_difference"].to_numpy()
    lower_errors = means - overall["ci95_lower"].to_numpy()
    upper_errors = overall["ci95_upper"].to_numpy() - means

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x,
        means,
        yerr=np.vstack([lower_errors, upper_errors]),
        marker="o",
        capsize=5,
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set(
        title="Paired Containment Policy Difference",
        xlabel="Containment budget (agents)",
        ylabel="Reduction difference: highest-out-degree minus random",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "paired_policy_difference_ci.png", dpi=300)
    plt.close(fig)

    print(f"Graph instances analysed: {paired['topology'].nunique()}")
    print(f"Summary saved: {RESULTS / 'paired_robustness_summary.csv'}")
    print(f"Figure saved: {FIGURES / 'paired_policy_difference_ci.png'}")
    print("\nOverall paired comparisons:")
    print(
        overall[
            [
                "budget",
                "graph_count",
                "mean_paired_difference",
                "ci95_lower",
                "ci95_upper",
            ]
        ].round(4).to_string(index=False)
    )
    print(
        "\nPositive differences favour highest-out-degree containment; "
        "negative differences favour random containment."
    )


if __name__ == "__main__":
    main()
