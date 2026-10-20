
"""Contract and edge-case tests for the graph foundation."""

import networkx as nx
import pytest

from structural_containment.graph.builder import build_directed_graph
from structural_containment.graph.topology import (
    directed_random_graph,
    find_directed_cycle,
    is_acyclic,
    strongly_connected_components,
)
from structural_containment.graph.validation import validate_directed_graph


def test_empty_graph_is_supported():
    graph = build_directed_graph([], [])
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0
    assert is_acyclic(graph)
    assert strongly_connected_components(graph) == ()


def test_edge_generator_is_supported():
    edges = ((source, target) for source, target in [("a", "b"), ("b", "c")])
    graph = build_directed_graph(["a", "b", "c"], edges)
    assert set(graph.edges) == {("a", "b"), ("b", "c")}


def test_non_iterable_edges_are_rejected():
    with pytest.raises(TypeError, match="edges"):
        build_directed_graph(["a"], None)


def test_non_iterable_edge_is_rejected():
    with pytest.raises(TypeError, match="Edge"):
        build_directed_graph(["a"], [42])


def test_self_loop_is_a_directed_cycle_when_allowed():
    graph = build_directed_graph(["a"], [("a", "a")], allow_self_loops=True)
    assert not is_acyclic(graph)
    assert find_directed_cycle(graph) == (("a", "a"),)


def test_directed_graph_validator_rejects_non_graph_objects():
    with pytest.raises(TypeError, match="NetworkX"):
        validate_directed_graph(None)


def test_random_graph_with_zero_nodes_is_supported():
    graph = directed_random_graph(0, 0.5, seed=1)
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_random_graph_has_no_self_loops():
    graph = directed_random_graph(30, 1.0, seed=7)
    assert nx.number_of_selfloops(graph) == 0
