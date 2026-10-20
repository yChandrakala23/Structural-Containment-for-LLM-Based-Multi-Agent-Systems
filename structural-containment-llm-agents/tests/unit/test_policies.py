"""Tests for the randomized containment policy."""

import networkx as nx
import pytest

from structural_containment.containment.random_policy import random_containment


@pytest.fixture
def graph():
    return nx.DiGraph(
        [
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
            ("c", "d"),
        ]
    )


def test_selects_exact_budget(graph):
    selected = random_containment(graph, 2, seed=7)
    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert set(selected) <= set(graph.nodes)


def test_selection_is_reproducible(graph):
    first = random_containment(graph, 3, seed=42)
    second = random_containment(graph, 3, seed=42)
    assert first == second


def test_zero_budget_returns_empty_tuple(graph):
    assert random_containment(graph, 0) == ()


def test_full_budget_selects_every_agent(graph):
    selected = random_containment(graph, graph.number_of_nodes())
    assert set(selected) == set(graph.nodes)


def test_candidate_subset_is_respected(graph):
    selected = random_containment(graph, 1, candidates=["b", "d"], seed=3)
    assert len(selected) == 1
    assert selected[0] in {"b", "d"}


def test_empty_candidate_subset_with_zero_budget(graph):
    assert random_containment(graph, 0, candidates=[]) == ()


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_invalid_budget_is_rejected(graph, budget):
    with pytest.raises((TypeError, ValueError)):
        random_containment(graph, budget)


def test_budget_exceeding_eligible_agents_is_rejected(graph):
    with pytest.raises(ValueError, match="eligible agents"):
        random_containment(graph, 3, candidates=["a", "b"])


@pytest.mark.parametrize("seed", [True, 1.5, "42"])
def test_invalid_seed_is_rejected(graph, seed):
    with pytest.raises(TypeError, match="seed"):
        random_containment(graph, 1, seed=seed)


def test_unknown_candidate_is_rejected(graph):
    with pytest.raises(ValueError, match="Unknown candidate"):
        random_containment(graph, 1, candidates=["missing"])


def test_duplicate_candidates_are_rejected(graph):
    with pytest.raises(ValueError, match="duplicate"):
        random_containment(graph, 1, candidates=["a", "a"])


def test_string_is_not_accepted_as_candidate_iterable(graph):
    with pytest.raises(TypeError, match="iterable of agent IDs"):
        random_containment(graph, 1, candidates="a")


def test_blank_node_identifier_is_rejected():
    graph = nx.DiGraph()
    graph.add_node(" ")
    with pytest.raises(ValueError, match="node identifiers"):
        random_containment(graph, 0)


def test_undirected_graph_is_rejected():
    with pytest.raises(ValueError, match="directed"):
        random_containment(nx.Graph(), 0)


from structural_containment.containment.degree_policy import (
    highest_out_degree_containment,
)


def test_degree_policy_selects_highest_out_degree_agents():
    graph = nx.DiGraph(
        [
            ("a", "b"),
            ("a", "c"),
            ("a", "d"),
            ("b", "c"),
            ("b", "d"),
            ("c", "d"),
        ]
    )

    assert highest_out_degree_containment(graph, 2) == ("a", "b")


def test_degree_policy_respects_candidate_subset():
    graph = nx.DiGraph(
        [
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
        ]
    )

    selected = highest_out_degree_containment(graph, 1, candidates=["b", "c"])

    assert selected == ("b",)


def test_degree_policy_breaks_ties_deterministically():
    graph = nx.DiGraph([("z", "x"), ("a", "y")])

    assert highest_out_degree_containment(graph, 2) == ("a", "z")


def test_degree_policy_zero_budget_returns_empty_tuple():
    graph = nx.DiGraph([("a", "b")])

    assert highest_out_degree_containment(graph, 0) == ()


def test_degree_policy_rejects_invalid_budget():
    graph = nx.DiGraph([("a", "b")])

    with pytest.raises(ValueError, match="non-negative"):
        highest_out_degree_containment(graph, -1)

    with pytest.raises(TypeError, match="integer"):
        highest_out_degree_containment(graph, True)


def test_degree_policy_rejects_unknown_candidates():
    graph = nx.DiGraph([("a", "b")])

    with pytest.raises(ValueError, match="Unknown candidate"):
        highest_out_degree_containment(graph, 1, candidates=["missing"])


def test_degree_policy_rejects_duplicate_candidates():
    graph = nx.DiGraph([("a", "b")])

    with pytest.raises(ValueError, match="duplicate"):
        highest_out_degree_containment(graph, 1, candidates=["a", "a"])


def test_degree_policy_rejects_undirected_graph():
    with pytest.raises(ValueError, match="directed"):
        highest_out_degree_containment(nx.Graph(), 0)
