"""Tests for initial compromise selection strategies."""

import networkx as nx
import pytest

from structural_containment.simulation.attacks import (
    select_explicit_targets,
    select_highest_out_degree_targets,
    select_random_targets,
)


@pytest.fixture
def graph():
    return nx.DiGraph(
        [
            ("a", "b"),
            ("a", "c"),
            ("a", "d"),
            ("b", "c"),
            ("b", "d"),
            ("c", "d"),
        ]
    )


def test_random_selection_returns_requested_count(graph):
    targets = select_random_targets(graph, 2, seed=7)
    assert len(targets) == 2
    assert len(set(targets)) == 2
    assert set(targets) <= set(graph.nodes)


def test_random_selection_is_reproducible(graph):
    first = select_random_targets(graph, 3, seed=21)
    second = select_random_targets(graph, 3, seed=21)
    assert first == second


def test_random_selection_allows_zero_targets(graph):
    assert select_random_targets(graph, 0) == ()


def test_random_selection_can_select_every_agent(graph):
    targets = select_random_targets(graph, graph.number_of_nodes())
    assert set(targets) == set(graph.nodes)


def test_highest_out_degree_ranks_correctly(graph):
    targets = select_highest_out_degree_targets(graph, 2)
    assert targets == ("a", "b")


def test_highest_out_degree_ties_are_deterministic():
    graph = nx.DiGraph([("z", "x"), ("a", "y")])
    assert select_highest_out_degree_targets(graph, 2) == ("a", "z")


def test_explicit_targets_preserve_input_order(graph):
    assert select_explicit_targets(graph, ["c", "a"]) == ("c", "a")


def test_explicit_targets_allow_empty_selection(graph):
    assert select_explicit_targets(graph, []) == ()


@pytest.mark.parametrize("count", [-1, 1.5, True, 10])
def test_random_selection_rejects_invalid_count(graph, count):
    with pytest.raises((TypeError, ValueError)):
        select_random_targets(graph, count)


@pytest.mark.parametrize("count", [-1, 1.5, True, 10])
def test_degree_selection_rejects_invalid_count(graph, count):
    with pytest.raises((TypeError, ValueError)):
        select_highest_out_degree_targets(graph, count)


@pytest.mark.parametrize("seed", [True, 1.5, "7"])
def test_random_selection_rejects_invalid_seed(graph, seed):
    with pytest.raises(TypeError, match="seed"):
        select_random_targets(graph, 1, seed=seed)


def test_explicit_targets_reject_unknown_agent(graph):
    with pytest.raises(ValueError, match="unknown agents"):
        select_explicit_targets(graph, ["missing"])


def test_explicit_targets_reject_duplicates(graph):
    with pytest.raises(ValueError, match="duplicate"):
        select_explicit_targets(graph, ["a", "a"])


def test_explicit_targets_reject_string_argument(graph):
    with pytest.raises(TypeError, match="iterable of agent IDs"):
        select_explicit_targets(graph, "a")


def test_undirected_graph_is_rejected():
    with pytest.raises(ValueError, match="directed"):
        select_random_targets(nx.Graph(), 0)


def test_blank_node_identifier_is_rejected():
    graph = nx.DiGraph()
    graph.add_node(" ")
    with pytest.raises(ValueError, match="node identifiers"):
        select_random_targets(graph, 0)
