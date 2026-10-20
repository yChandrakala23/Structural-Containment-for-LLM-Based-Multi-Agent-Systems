"""Tests for synchronous compromise propagation."""

import networkx as nx
import pytest

from structural_containment.simulation.propagation import (
    TerminationReason,
    simulate_propagation,
)
from structural_containment.simulation.state import AgentStatus


def chain() -> nx.DiGraph:
    return nx.DiGraph([("a", "b"), ("b", "c")])


def test_probability_zero_prevents_spread():
    result = simulate_propagation(chain(), ["a"], propagation_probability=0.0)
    assert result.compromised_agents == {"a"}
    assert result.termination_reason == TerminationReason.STABLE


def test_probability_one_spreads_through_chain():
    result = simulate_propagation(chain(), ["a"], propagation_probability=1.0)
    assert result.compromised_agents == {"a", "b", "c"}
    assert result.final_states["c"].compromised_at == 2


def test_propagation_is_synchronous():
    result = simulate_propagation(chain(), ["a"], propagation_probability=1.0)
    step_one = result.steps[1]
    assert step_one.newly_compromised == ("b",)
    assert step_one.statuses["c"] == AgentStatus.HEALTHY


def test_contained_agent_blocks_propagation():
    result = simulate_propagation(
        chain(),
        ["a"],
        initial_contained=["b"],
        propagation_probability=1.0,
    )
    assert result.compromised_agents == {"a"}
    assert result.final_states["b"].status == AgentStatus.CONTAINED
    assert result.final_states["c"].status == AgentStatus.HEALTHY


def test_seeded_simulation_is_reproducible():
    graph = nx.relabel_nodes(
        nx.gnp_random_graph(30, 0.15, seed=12, directed=True),
        str,
    )
    first = simulate_propagation(graph, ["0"], propagation_probability=0.35, seed=99)
    second = simulate_propagation(graph, ["0"], propagation_probability=0.35, seed=99)

    assert first.compromised_agents == second.compromised_agents
    assert first.steps == second.steps
    assert first.termination_reason == second.termination_reason


def test_cycle_terminates():
    graph = nx.DiGraph([("a", "b"), ("b", "a")])
    result = simulate_propagation(graph, ["a"], propagation_probability=1.0)
    assert result.compromised_agents == {"a", "b"}
    assert result.termination_reason == TerminationReason.STABLE


def test_empty_initial_compromise_stabilizes():
    result = simulate_propagation(chain(), [])
    assert not result.compromised_agents
    assert result.termination_reason == TerminationReason.STABLE


def test_max_steps_limits_propagation():
    result = simulate_propagation(
        chain(),
        ["a"],
        propagation_probability=1.0,
        max_steps=1,
    )
    assert result.compromised_agents == {"a", "b"}
    assert result.termination_reason == TerminationReason.MAX_STEPS
    assert result.steps_executed == 1


def test_zero_max_steps_performs_no_propagation():
    result = simulate_propagation(chain(), ["a"], propagation_probability=1.0, max_steps=0)
    assert result.compromised_agents == {"a"}
    assert result.steps_executed == 0
    assert result.termination_reason == TerminationReason.MAX_STEPS


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_probability_is_rejected(probability):
    with pytest.raises((TypeError, ValueError)):
        simulate_propagation(chain(), ["a"], propagation_probability=probability)


@pytest.mark.parametrize("max_steps", [-1, 1.5, True])
def test_invalid_max_steps_is_rejected(max_steps):
    with pytest.raises((TypeError, ValueError)):
        simulate_propagation(chain(), ["a"], max_steps=max_steps)


def test_unknown_initial_agent_is_rejected():
    with pytest.raises(ValueError, match="unknown agents"):
        simulate_propagation(chain(), ["missing"])


def test_initial_sets_must_not_overlap():
    with pytest.raises(ValueError, match="cannot be initially"):
        simulate_propagation(chain(), ["a"], initial_contained=["a"])


def test_result_mappings_are_read_only():
    result = simulate_propagation(chain(), ["a"])
    with pytest.raises(TypeError):
        result.final_states["x"] = result.final_states["a"]


def test_undirected_graph_is_rejected():
    with pytest.raises(ValueError, match="directed"):
        simulate_propagation(nx.Graph(), [])


def test_string_is_rejected_as_initial_agent_iterable():
    with pytest.raises(TypeError, match="iterable of agent IDs"):
        simulate_propagation(chain(), "a")


def test_none_is_rejected_as_initial_agent_iterable():
    with pytest.raises(TypeError, match="iterable of agent IDs"):
        simulate_propagation(chain(), None)


def test_blank_initial_agent_id_is_rejected():
    with pytest.raises(ValueError, match="non-empty string IDs"):
        simulate_propagation(chain(), [" "])


def test_blank_graph_node_id_is_rejected():
    graph = nx.DiGraph()
    graph.add_nodes_from(["a", " "])

    with pytest.raises(ValueError, match="node identifiers"):
        simulate_propagation(graph, ["a"])


@pytest.mark.parametrize("seed", [True, 1.5, "99"])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(TypeError, match="seed"):
        simulate_propagation(chain(), ["a"], seed=seed)
