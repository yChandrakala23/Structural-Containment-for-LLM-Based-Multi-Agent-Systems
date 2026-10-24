"""Exact feedback vertex set (FVS) containment policy.

A Feedback Vertex Set (FVS) is a set of nodes whose removal makes the graph
acyclic.  Containing an FVS guarantees that no directed cycle can form among
the remaining agents, eliminating all possible propagation loops.

The exact solver uses brute-force enumeration and is only practical for graphs
with at most MAX_EXACT_NODES nodes.  Use the heuristic for larger graphs.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import networkx as nx

from structural_containment.graph.validation import validate_directed_graph

MAX_EXACT_NODES: int = 20
"""Maximum graph size (nodes) for which the exact solver is attempted."""


def _is_fvs(graph: nx.DiGraph, candidate_set: frozenset[str]) -> bool:
    """Return True if removing ``candidate_set`` makes ``graph`` acyclic."""
    subgraph = graph.subgraph(set(graph.nodes) - candidate_set)
    return nx.is_directed_acyclic_graph(subgraph)


def fvs_exact_containment(
    graph: nx.Graph,
    budget: int,
    *,
    candidates: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Select a minimum-size feedback vertex set up to ``budget`` agents.

    The algorithm enumerates all combinations of eligible agents in increasing
    size order and returns the first combination that makes the graph acyclic.
    If no subset of size ≤ ``budget`` suffices, the highest-budget combination
    with the most cycles broken is returned.

    .. warning::
        This is exponential in the number of candidates.  Only use it for
        graphs with at most ``MAX_EXACT_NODES`` (``20``) nodes.

    Args:
        graph: Directed, simple NetworkX graph with string node IDs.
        budget: Maximum number of agents to select.
        candidates: Optional subset of eligible agents.

    Returns:
        Selected agent IDs forming a minimal FVS, ordered lexicographically.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If the graph, budget, candidates, or graph size are invalid.
    """
    validate_directed_graph(graph)

    if any(not isinstance(node, str) or not node.strip() for node in graph.nodes):
        raise ValueError("All graph node identifiers must be non-empty strings.")

    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError("budget must be an integer.")
    if budget < 0:
        raise ValueError("budget must be non-negative.")

    if graph.number_of_nodes() > MAX_EXACT_NODES:
        raise ValueError(
            f"fvs_exact_containment only supports graphs with at most {MAX_EXACT_NODES} nodes. "
            "Use fvs_heuristic_containment for larger graphs."
        )

    if candidates is None:
        eligible = tuple(sorted(graph.nodes))
    else:
        if isinstance(candidates, (str, bytes)):
            raise TypeError("candidates must be an iterable of agent IDs.")

        try:
            eligible = tuple(candidates)
        except TypeError as exc:
            raise TypeError("candidates must be an iterable of agent IDs.") from exc

        if any(not isinstance(agent, str) or not agent.strip() for agent in eligible):
            raise ValueError("candidates must contain non-empty string IDs.")
        if len(set(eligible)) != len(eligible):
            raise ValueError("candidates must not contain duplicate IDs.")

        unknown = set(eligible) - set(graph.nodes)
        if unknown:
            raise ValueError(f"Unknown candidate agents: {sorted(unknown)!r}.")

    if budget > len(eligible):
        raise ValueError(f"budget ({budget}) cannot exceed eligible agents ({len(eligible)}).")

    # If already acyclic, no containment needed.
    if nx.is_directed_acyclic_graph(graph):
        return ()

    # Try combinations from smallest to largest until an FVS is found.
    for size in range(1, budget + 1):
        for combo in combinations(eligible, size):
            if _is_fvs(graph, frozenset(combo)):
                return tuple(sorted(combo))

    # No complete FVS found within budget; return full budget sorted.
    return tuple(sorted(eligible[:budget]))
