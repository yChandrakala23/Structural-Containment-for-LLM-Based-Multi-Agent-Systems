
"""Statistically validate paired containment-policy comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

POLICY_A = "highest_out_degree"
POLICY_B = "random"
BUDGETS_TO_TEST = (1, 2, 4)
DENSITIES_TO_TEST = ("sparse", "medium", "dense")
N_PERMUTATIONS = 100_000
RANDOM_SEED = 20261023


def holm_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm's step-down correction to a family of p-values."""
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_max = 0.0

    for rank, index in enumerate(order):
        candidate = (count - rank) * p_values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)

    return adjusted.tolist()


def paired_sign_flip_test(
    differences: np.ndarray,
    *,
    rng: np.random.Generator,
    n_permutations: int = N_PERMUTATIONS,
) -> float:
    """Estimate a two-sided paired sign-flip permutation p-value.

    The null reference distribution is formed by independently flipping
    the signs of graph-level paired differences. This test assumes the
    paired differences are exchangeable under sign changes.
    """
    if differences.size == 0:
        raise ValueError("At least one paired difference is required.")

    observed = abs(float(np.mean(differences)))
    extreme = 0
    completed = 0
    batch_size = 10_000

    while completed < n_permutations:
        size = min(batch_size, n_permutations - completed)
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(size, differences.size),
        )
        permuted_means = np.mean(signs * differences, axis=1)
        extreme += int(np.count_nonzero(np.abs(permuted_means) >= observed - 1e-15))
        completed += size

    # Add-one correction avoids reporting a zero Monte Carlo p-value.
    return (extreme + 1) / (n_permutations + 1)


def main() -> None:
    source = RESULTS / "benchmark.csv"
    if not source.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {source}. Run scripts/run_benchmarks.py first."
        )

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

    # First average over simulation seeds. Graph instances, not individual
    # simulation runs, are the paired observational units.
    per_graph = (
        df.groupby(["topology", "density", "policy", "budget"], as_index=False)
        .agg(mean_reduction=("reduction", "mean"))
    )

    paired = per_graph.pivot(
        index=["topology", "density", "budget"],
        columns="policy",
        values="mean_reduction",
    ).reset_index()

    if not {POLICY_A, POLICY_B}.issubset(paired.columns):
        raise ValueError("The results must contain both expected policies.")

    paired["difference"] = paired[POLICY_A] - paired[POLICY_B]

    rng = np.random.default_rng(RANDOM_SEED)
    tests: list[dict[str, object]] = []

    for density in DENSITIES_TO_TEST:
        for budget in BUDGETS_TO_TEST:
            subset = paired[
                (paired["density"] == density)
                & (paired["budget"] == budget)
            ]
            differences = subset["difference"].to_numpy(dtype=float)

            if differences.size < 2:
                raise ValueError(
                    f"Need at least two graph instances for {density}, budget {budget}."
                )

            p_value = paired_sign_flip_test(differences, rng=rng)

            tests.append(
                {
                    "density": density,
                    "budget": budget,
                    "graph_count": int(differences.size),
                    "mean_difference": float(differences.mean()),
                    "median_difference": float(np.median(differences)),
                    "p_value": p_value,
                }
            )

    adjusted = holm_adjust([float(item["p_value"]) for item in tests])

    for item, adjusted_p in zip(tests, adjusted, strict=True):
        item["holm_adjusted_p"] = adjusted_p
        item["significant_at_0_05"] = adjusted_p < 0.05

    output = pd.DataFrame(tests)
    output["direction"] = np.where(
        output["mean_difference"] > 0,
        "highest_out_degree",
        np.where(output["mean_difference"] < 0, "random", "tie"),
    )
    output.to_csv(RESULTS / "statistical_validation.csv", index=False)

    print(f"Graph instances: {paired['topology'].nunique()}")
    print(f"Comparisons tested: {len(output)}")
    print(f"Sign-flip permutations per comparison: {N_PERMUTATIONS}")
    print(f"Results saved: {RESULTS / 'statistical_validation.csv'}")
    print("\nMultiple-comparison-adjusted results:")
    print(
        output[
            [
                "density",
                "budget",
                "graph_count",
                "mean_difference",
                "p_value",
                "holm_adjusted_p",
                "significant_at_0_05",
                "direction",
            ]
        ].round(5).to_string(index=False)
    )
    print(
        "\nPositive mean differences favour highest-out-degree containment. "
        "Interpret results cautiously: sign-flip inference assumes symmetry "
        "under the null, and the tests cover only the listed density-budget pairs."
    )


if __name__ == "__main__":
    main()
