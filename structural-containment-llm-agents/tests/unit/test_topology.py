
"""Tests for graph topology operations."""

import networkx as nx
import pytest

from structural_containment.graph.topology import (
    directed_random_graph,
    find_directed_cycle,
    is_acyclic,
    strongly_connected_components,
)


def test_dag_is_acyclic():
    graph = nx.DiGraph([("a", "b"), ("b", "c")])
    assert is_acyclic(graph)


def test_cycle_is_detected():
    graph = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a")])
    assert not is_acyclic(graph)


def test_find_cycle_returns_cycle_edges():
    graph = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a")])
    cycle = find_directed_cycle(graph)

    assert cycle is not None
    assert len(cycle) >= 2
    for source, target in cycle:
        assert graph.has_edge(source, target)


def test_find_cycle_returns_none_for_dag():
    graph = nx.DiGraph([("a", "b"), ("b", "c")])
    assert find_directed_cycle(graph) is None


def test_strongly_connected_components():
    graph = nx.DiGraph(
        [
            ("a", "b"),
            ("b", "a"),
            ("b", "c"),
            ("c", "d"),
            ("d", "c"),
        ]
    )

    components = strongly_connected_components(graph)

    assert {frozenset(component) for component in components} == {
        frozenset({"a", "b"}),
        frozenset({"c", "d"}),
    }


def test_isolated_node_is_its_own_component():
    graph = nx.DiGraph()
    graph.add_node("isolated")

    assert strongly_connected_components(graph) == (frozenset({"isolated"}),)


def test_topology_rejects_undirected_graph():
    with pytest.raises(ValueError, match="directed"):
        is_acyclic(nx.Graph())


def test_topology_rejects_multidigraph():
    with pytest.raises(ValueError, match="simple"):
        is_acyclic(nx.MultiDiGraph())


@pytest.mark.parametrize("node_count", [-1, 1.5, True])
def test_random_graph_rejects_invalid_node_count(node_count):
    with pytest.raises((TypeError, ValueError)):
        directed_random_graph(node_count, 0.2)


@pytest.mark.parametrize(
    "probability",
    [-0.1, 1.1, float("nan"), float("inf"), True],
)
def test_random_graph_rejects_invalid_probability(probability):
    with pytest.raises((TypeError, ValueError)):
        directed_random_graph(5, probability)


def test_random_graph_is_reproducible():
    first = directed_random_graph(20, 0.2, seed=17)
    second = directed_random_graph(20, 0.2, seed=17)

    assert set(first.nodes) == set(second.nodes)
    assert set(first.edges) == set(second.edges)


def test_random_graph_probability_boundaries():
    empty = directed_random_graph(5, 0.0)
    complete = directed_random_graph(5, 1.0)

    assert empty.number_of_edges() == 0
    assert complete.number_of_edges() == 5 * 4
