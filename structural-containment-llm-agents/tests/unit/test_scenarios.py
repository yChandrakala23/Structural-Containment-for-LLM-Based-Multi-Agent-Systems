"""Tests for containment scenario evaluation."""

import networkx as nx
import pytest

from structural_containment.simulation.scenarios import (
    run_containment_scenario,
    run_repeated_scenarios,
)


@pytest.fixture
def graph():
    return nx.DiGraph(
        [
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
        ]
    )


def test_scenario_returns_valid_rates(graph):
    result = run_containment_scenario(
        graph,
        ["a"],
        policy="highest_out_degree",
        budget=1,
        propagation_probability=1.0,
    )

    assert 0.0 <= result.baseline_compromise_rate <= 1.0
    assert 0.0 <= result.intervention_compromise_rate <= 1.0
    assert len(result.contained_agents) == 1


def test_containment_reduces_spread_on_chain():
    graph = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d")])

    result = run_containment_scenario(
        graph,
        ["a"],
        policy="highest_out_degree",
        budget=1,
        propagation_probability=1.0,
    )

    assert result.baseline_compromise_rate == 1.0
    assert result.intervention_compromise_rate < 1.0


def test_relative_reduction_is_none_when_baseline_is_zero():
    graph = nx.DiGraph([("a", "b")])

    result = run_containment_scenario(
        graph,
        [],
        policy="random",
        budget=0,
    )

    assert result.baseline_compromise_rate == 0.0
    assert result.relative_reduction is None


def test_unknown_policy_is_rejected(graph):
    with pytest.raises(ValueError, match="Unsupported policy"):
        run_containment_scenario(graph, ["a"], policy="unknown")


def test_containment_excludes_initially_compromised_agents():
    graph = nx.DiGraph([("a", "b")])

    result = run_containment_scenario(
        graph,
        ["a"],
        policy="highest_out_degree",
        budget=1,
    )

    assert result.contained_agents == ("b",)

def test_budget_cannot_exceed_eligible_agents():
    graph = nx.DiGraph([("a", "b")])

    with pytest.raises(ValueError, match="eligible for containment"):
        run_containment_scenario(
            graph,
            ["a"],
            policy="random",
            budget=2,
        )


def test_repeated_scenarios_return_aggregate_metrics(graph):
    result = run_repeated_scenarios(
        graph,
        ["a"],
        policy="highest_out_degree",
        budget=1,
        propagation_probability=0.5,
        seeds=(1, 2, 3),
    )

    assert len(result.runs) == 3
    assert 0.0 <= result.mean_baseline_rate <= 1.0
    assert 0.0 <= result.mean_intervention_rate <= 1.0
    assert result.standard_deviation_reduction >= 0.0


def test_repeated_scenarios_reject_empty_seeds(graph):
    with pytest.raises(ValueError, match="at least one seed"):
        run_repeated_scenarios(graph, ["a"], seeds=())


def test_repeated_scenarios_reject_invalid_seed(graph):
    with pytest.raises(TypeError, match="every seed"):
        run_repeated_scenarios(graph, ["a"], seeds=(1, True))
