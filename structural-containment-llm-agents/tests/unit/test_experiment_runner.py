import networkx as nx
import pytest

from structural_containment.experiments.config import ExperimentConfig
from structural_containment.experiments.runner import run_experiments


@pytest.fixture
def graph():
    return nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])


def test_runner_executes_every_policy_budget_seed_combination(graph):
    config = ExperimentConfig(
        initial_compromised=("a",),
        policies=("random", "highest_out_degree"),
        budgets=(0, 1),
        seeds=(11, 22),
    )

    batch = run_experiments(graph, config)

    assert batch.run_count == 8
    assert len(batch.runs) == 8


def test_runner_records_configuration_for_each_run(graph):
    config = ExperimentConfig(
        initial_compromised=("a",),
        policies=("highest_out_degree",),
        budgets=(1,),
        seeds=(42,),
    )

    batch = run_experiments(graph, config)
    run = batch.runs[0]

    assert run.attack_strategy == "targeted"
    assert run.policy == "highest_out_degree"
    assert run.budget == 1
    assert run.seed == 42
    assert run.initial_compromised == ("a",)


@pytest.mark.parametrize(
    "strategy",
    ["random", "highest_out_degree"],
)
def test_runner_supports_automated_attack_selection(graph, strategy):
    config = ExperimentConfig(
        initial_compromised=("a",),
        attack_strategy=strategy,
        policies=("random",),
        budgets=(0,),
        seeds=(42,),
    )

    batch = run_experiments(graph, config)

    assert len(batch.runs[0].initial_compromised) == 1
    assert batch.runs[0].initial_compromised[0] in graph


def test_runner_rejects_undirected_graph():
    config = ExperimentConfig(initial_compromised=("a",))

    with pytest.raises(TypeError, match="directed"):
        run_experiments(nx.Graph([("a", "b")]), config)


def test_runner_rejects_budget_exceeding_eligible_agents():
    graph = nx.DiGraph([("a", "b")])
    config = ExperimentConfig(
        initial_compromised=("a",),
        policies=("random",),
        budgets=(2,),
        seeds=(42,),
    )

    with pytest.raises(ValueError, match="eligible"):
        run_experiments(graph, config)
