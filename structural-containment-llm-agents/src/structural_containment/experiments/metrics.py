"""Aggregate and compare containment experiment results."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev

from structural_containment.experiments.runner import ExperimentBatch


@dataclass(frozen=True, slots=True)
class PolicyMetrics:
    """Aggregate effectiveness metrics for one policy and budget."""

    attack_strategy: str
    policy: str
    budget: int
    run_count: int
    mean_baseline_rate: float
    mean_intervention_rate: float
    mean_absolute_reduction: float
    mean_relative_reduction: float | None
    standard_deviation_reduction: float


@dataclass(frozen=True, slots=True)
class PolicyComparison:
    """Compare two policies under the same attack and budget."""

    attack_strategy: str
    budget: int
    first_policy: str
    second_policy: str
    first_mean_reduction: float
    second_mean_reduction: float
    difference_in_mean_reduction: float


def summarize_experiments(batch: ExperimentBatch) -> tuple[PolicyMetrics, ...]:
    """Aggregate runs by attack strategy, containment policy and budget."""
    if not isinstance(batch, ExperimentBatch):
        raise TypeError("batch must be an ExperimentBatch.")

    groups: dict[tuple[str, str, int], list] = {}

    for run in batch.runs:
        key = (run.attack_strategy, run.policy, run.budget)
        groups.setdefault(key, []).append(run)

    summaries: list[PolicyMetrics] = []

    for (attack_strategy, policy, budget), runs in sorted(groups.items()):
        baseline_rates = [run.result.baseline_compromise_rate for run in runs]
        intervention_rates = [run.result.intervention_compromise_rate for run in runs]
        reductions = [run.result.absolute_reduction for run in runs]
        relative_reductions = [
            run.result.relative_reduction
            for run in runs
            if run.result.relative_reduction is not None
        ]

        summaries.append(
            PolicyMetrics(
                attack_strategy=attack_strategy,
                policy=policy,
                budget=budget,
                run_count=len(runs),
                mean_baseline_rate=fmean(baseline_rates),
                mean_intervention_rate=fmean(intervention_rates),
                mean_absolute_reduction=fmean(reductions),
                mean_relative_reduction=(
                    fmean(relative_reductions) if relative_reductions else None
                ),
                standard_deviation_reduction=pstdev(reductions),
            )
        )

    return tuple(summaries)


def compare_policies(
    summaries: tuple[PolicyMetrics, ...],
    *,
    first_policy: str,
    second_policy: str,
) -> tuple[PolicyComparison, ...]:
    """Compare policy effectiveness for matching attack strategies and budgets."""
    if first_policy == second_policy:
        raise ValueError("Policies to compare must be different.")

    if not isinstance(summaries, tuple) or any(
        not isinstance(item, PolicyMetrics) for item in summaries
    ):
        raise TypeError("summaries must be a tuple of PolicyMetrics.")

    indexed: dict[tuple[str, int, str], PolicyMetrics] = {}

    for summary in summaries:
        key = (summary.attack_strategy, summary.budget, summary.policy)
        if key in indexed:
            raise ValueError(f"Duplicate policy summary for {key!r}.")
        indexed[key] = summary

    comparisons: list[PolicyComparison] = []
    settings = sorted({(item.attack_strategy, item.budget) for item in summaries})

    for attack_strategy, budget in settings:
        first = indexed.get((attack_strategy, budget, first_policy))
        second = indexed.get((attack_strategy, budget, second_policy))

        if first is None or second is None:
            continue

        comparisons.append(
            PolicyComparison(
                attack_strategy=attack_strategy,
                budget=budget,
                first_policy=first_policy,
                second_policy=second_policy,
                first_mean_reduction=first.mean_absolute_reduction,
                second_mean_reduction=second.mean_absolute_reduction,
                difference_in_mean_reduction=(
                    first.mean_absolute_reduction - second.mean_absolute_reduction
                ),
            )
        )

    return tuple(comparisons)
