import networkx as nx
import pytest

from structural_containment.experiments.config import ExperimentConfig
from structural_containment.experiments.metrics import (
    PolicyMetrics,
    compare_policies,
    summarize_experiments,
)
from structural_containment.experiments.runner import run_experiments


@pytest.fixture
def batch():
    graph = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])
    config = ExperimentConfig(
        initial_compromised=("a",),
        policies=("random", "highest_out_degree"),
        budgets=(0, 1),
        seeds=(11, 22),
    )
    return run_experiments(graph, config)


def test_summary_groups_runs_by_policy_and_budget(batch):
    summaries = summarize_experiments(batch)

    assert len(summaries) == 4
    assert all(item.run_count == 2 for item in summaries)


def test_summary_rates_are_valid(batch):
    for item in summarize_experiments(batch):
        assert 0.0 <= item.mean_baseline_rate <= 1.0
        assert 0.0 <= item.mean_intervention_rate <= 1.0
        assert item.standard_deviation_reduction >= 0.0


def test_summary_is_deterministic(batch):
    assert summarize_experiments(batch) == summarize_experiments(batch)


def test_wrong_batch_type_is_rejected():
    with pytest.raises(TypeError, match="ExperimentBatch"):
        summarize_experiments(None)


def test_policy_comparison_uses_matching_budgets():
    summaries = (
        PolicyMetrics("targeted", "random", 1, 2, 0.8, 0.6, 0.2, 0.25, 0.0),
        PolicyMetrics("targeted", "highest_out_degree", 1, 2, 0.8, 0.4, 0.4, 0.5, 0.0),
        PolicyMetrics("targeted", "random", 2, 2, 0.8, 0.5, 0.3, 0.375, 0.0),
    )

    comparisons = compare_policies(
        summaries,
        first_policy="highest_out_degree",
        second_policy="random",
    )

    assert len(comparisons) == 1
    assert comparisons[0].budget == 1
    assert comparisons[0].difference_in_mean_reduction == pytest.approx(0.2)


def test_comparison_rejects_same_policy():
    with pytest.raises(ValueError, match="different"):
        compare_policies(
            (),
            first_policy="random",
            second_policy="random",
        )


def test_comparison_rejects_duplicate_summaries():
    item = PolicyMetrics("targeted", "random", 1, 1, 0.8, 0.6, 0.2, 0.25, 0.0)

    with pytest.raises(ValueError, match="Duplicate"):
        compare_policies(
            (item, item),
            first_policy="random",
            second_policy="highest_out_degree",
        )
