"""Tests for the exact feedback vertex set containment policy."""

from __future__ import annotations

import networkx as nx
import pytest

from structural_containment.containment.fvs_exact import (
    MAX_EXACT_NODES,
    fvs_exact_containment,
)


@pytest.fixture
def triangle():
    """A simple directed 3-cycle: a → b → c → a."""
    return nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a")])


@pytest.fixture
def dag():
    """An acyclic directed graph (DAG)."""
    return nx.DiGraph([("a", "b"), ("b", "c"), ("a", "c")])


# ── Happy-path tests ──────────────────────────────────────────────────────────


def test_acyclic_graph_returns_empty_tuple(dag):
    result = fvs_exact_containment(dag, budget=3)
    assert result == ()


def test_single_cycle_broken_by_one_node(triangle):
    result = fvs_exact_containment(triangle, budget=1)
    assert len(result) == 1
    # Removing the selected node must make the graph acyclic.
    remaining = set(triangle.nodes) - set(result)
    assert nx.is_directed_acyclic_graph(triangle.subgraph(remaining))


def test_result_is_sorted_lexicographically(triangle):
    result = fvs_exact_containment(triangle, budget=1)
    assert result == tuple(sorted(result))


def test_budget_zero_returns_empty_for_dag(dag):
    assert fvs_exact_containment(dag, budget=0) == ()


def test_budget_zero_on_cyclic_graph_returns_empty(triangle):
    # Cannot break any cycle with budget=0.
    assert fvs_exact_containment(triangle, budget=0) == ()


def test_candidate_subset_restricts_selection(triangle):
    # Only allow "b" and "c" as candidates.
    result = fvs_exact_containment(triangle, budget=1, candidates=["b", "c"])
    assert len(result) == 1
    assert result[0] in {"b", "c"}


def test_full_budget_breaks_all_cycles():
    # Two independent cycles sharing no nodes.
    graph = nx.DiGraph([("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")])
    result = fvs_exact_containment(graph, budget=2)
    remaining = set(graph.nodes) - set(result)
    assert nx.is_directed_acyclic_graph(graph.subgraph(remaining))


def test_self_loop_is_broken():
    graph = nx.DiGraph()
    graph.add_node("a")
    graph.add_edge("a", "a")
    result = fvs_exact_containment(graph, budget=1)
    assert result == ("a",)


def test_result_is_reproducible(triangle):
    first = fvs_exact_containment(triangle, budget=2)
    second = fvs_exact_containment(triangle, budget=2)
    assert first == second


def test_complex_graph_fvs_is_valid():
    # Graph with two cycles: a→b→c→a and d→e→d
    graph = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "d"), ("c", "d")])
    result = fvs_exact_containment(graph, budget=2)
    remaining = set(graph.nodes) - set(result)
    assert nx.is_directed_acyclic_graph(graph.subgraph(remaining))


# ── Error-handling tests ──────────────────────────────────────────────────────


def test_undirected_graph_is_rejected():
    with pytest.raises(ValueError, match="directed"):
        fvs_exact_containment(nx.Graph(), budget=1)


def test_multigraph_is_rejected():
    with pytest.raises(ValueError, match="simple"):
        fvs_exact_containment(nx.MultiDiGraph(), budget=0)


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_budget_is_rejected(triangle, budget):
    with pytest.raises((TypeError, ValueError)):
        fvs_exact_containment(triangle, budget)


def test_budget_exceeding_candidates_is_rejected(triangle):
    with pytest.raises(ValueError, match="eligible agents"):
        fvs_exact_containment(triangle, budget=5, candidates=["a", "b"])


def test_unknown_candidate_is_rejected(triangle):
    with pytest.raises(ValueError, match="Unknown candidate"):
        fvs_exact_containment(triangle, budget=1, candidates=["missing"])


def test_duplicate_candidates_are_rejected(triangle):
    with pytest.raises(ValueError, match="duplicate"):
        fvs_exact_containment(triangle, budget=1, candidates=["a", "a"])


def test_string_candidates_are_rejected(triangle):
    with pytest.raises(TypeError, match="iterable of agent IDs"):
        fvs_exact_containment(triangle, budget=1, candidates="a")


def test_graph_too_large_is_rejected():
    big = nx.DiGraph()
    big.add_nodes_from([str(i) for i in range(MAX_EXACT_NODES + 1)])
    with pytest.raises(ValueError, match=str(MAX_EXACT_NODES)):
        fvs_exact_containment(big, budget=1)


def test_blank_node_identifier_is_rejected():
    graph = nx.DiGraph()
    graph.add_node(" ")
    with pytest.raises(ValueError, match="node identifiers"):
        fvs_exact_containment(graph, budget=0)
