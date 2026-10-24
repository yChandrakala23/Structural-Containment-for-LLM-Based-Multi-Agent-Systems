"""Tests for the greedy FVS heuristic containment policy."""

from __future__ import annotations

import networkx as nx
import pytest

from structural_containment.containment.fvs_heuristic import fvs_heuristic_containment


@pytest.fixture
def triangle():
    """A simple directed 3-cycle: a → b → c → a."""
    return nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a")])


@pytest.fixture
def dag():
    """A directed acyclic graph."""
    return nx.DiGraph([("a", "b"), ("b", "c"), ("a", "c")])


# ── Happy-path tests ──────────────────────────────────────────────────────────


def test_acyclic_graph_returns_empty(dag):
    result = fvs_heuristic_containment(dag, budget=3)
    assert result == ()


def test_single_cycle_broken_by_one_node(triangle):
    result = fvs_heuristic_containment(triangle, budget=1)
    assert len(result) == 1
    remaining = set(triangle.nodes) - set(result)
    assert nx.is_directed_acyclic_graph(triangle.subgraph(remaining))


def test_result_length_respects_budget(triangle):
    result = fvs_heuristic_containment(triangle, budget=2)
    assert len(result) <= 2


def test_zero_budget_returns_empty(triangle):
    assert fvs_heuristic_containment(triangle, budget=0) == ()


def test_budget_larger_than_needed_stops_early(dag):
    # Already acyclic; no nodes should be selected.
    result = fvs_heuristic_containment(dag, budget=10)
    assert result == ()


def test_candidate_subset_is_respected(triangle):
    result = fvs_heuristic_containment(triangle, budget=1, candidates=["b", "c"])
    assert len(result) <= 1
    for agent in result:
        assert agent in {"b", "c"}


def test_large_cyclic_graph_is_handled():
    # Ring of 15 nodes — any single node breaks the cycle.
    n = 15
    nodes = [str(i) for i in range(n)]
    edges = [(nodes[i], nodes[(i + 1) % n]) for i in range(n)]
    graph = nx.DiGraph(edges)
    result = fvs_heuristic_containment(graph, budget=1)
    assert len(result) == 1
    remaining = set(graph.nodes) - set(result)
    assert nx.is_directed_acyclic_graph(graph.subgraph(remaining))


def test_two_independent_cycles_both_broken():
    graph = nx.DiGraph([("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")])
    result = fvs_heuristic_containment(graph, budget=2)
    remaining = set(graph.nodes) - set(result)
    assert nx.is_directed_acyclic_graph(graph.subgraph(remaining))


def test_result_contains_no_duplicates(triangle):
    result = fvs_heuristic_containment(triangle, budget=3)
    assert len(set(result)) == len(result)


def test_result_is_reproducible(triangle):
    first = fvs_heuristic_containment(triangle, budget=2)
    second = fvs_heuristic_containment(triangle, budget=2)
    assert first == second


def test_empty_graph_is_supported():
    graph = nx.DiGraph()
    assert fvs_heuristic_containment(graph, budget=0) == ()


def test_self_loop_is_broken():
    graph = nx.DiGraph()
    graph.add_node("x")
    graph.add_edge("x", "x")
    result = fvs_heuristic_containment(graph, budget=1)
    assert result == ("x",)


# ── Error-handling tests ──────────────────────────────────────────────────────


def test_undirected_graph_is_rejected():
    with pytest.raises(ValueError, match="directed"):
        fvs_heuristic_containment(nx.Graph(), budget=1)


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_budget_is_rejected(triangle, budget):
    with pytest.raises((TypeError, ValueError)):
        fvs_heuristic_containment(triangle, budget)


def test_budget_exceeding_candidates_is_rejected(triangle):
    with pytest.raises(ValueError, match="eligible agents"):
        fvs_heuristic_containment(triangle, budget=5, candidates=["a"])


def test_unknown_candidate_is_rejected(triangle):
    with pytest.raises(ValueError, match="Unknown candidate"):
        fvs_heuristic_containment(triangle, budget=1, candidates=["missing"])


def test_duplicate_candidates_are_rejected(triangle):
    with pytest.raises(ValueError, match="duplicate"):
        fvs_heuristic_containment(triangle, budget=1, candidates=["a", "a"])


def test_string_candidates_are_rejected(triangle):
    with pytest.raises(TypeError, match="iterable of agent IDs"):
        fvs_heuristic_containment(triangle, budget=1, candidates="a")


def test_blank_node_identifier_is_rejected():
    graph = nx.DiGraph()
    graph.add_node(" ")
    with pytest.raises(ValueError, match="node identifiers"):
        fvs_heuristic_containment(graph, budget=0)
