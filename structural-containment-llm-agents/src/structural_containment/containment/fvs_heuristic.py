"""Greedy heuristic feedback vertex set (FVS) containment policy.

The greedy heuristic iteratively removes the node that participates in the
most directed cycles (approximated by selecting the node with the highest
product of in-degree and out-degree, a fast proxy for cycle involvement).
This runs in polynomial time and scales to large graphs.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from structural_containment.graph.validation import validate_directed_graph


def _cycle_score(graph: nx.DiGraph, node: str) -> tuple[int, str]:
    """Return a sortable score for cycle-breaking priority.

    Higher ``in_degree × out_degree`` means the node lies on more potential
    cycles. Lexicographic tie-breaking by node ID ensures determinism.
    """
    return (-(graph.in_degree(node) * graph.out_degree(node)), node)


def fvs_heuristic_containment(
    graph: nx.Graph,
    budget: int,
    *,
    candidates: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Greedily select agents that break the most directed cycles.

    At each step, the algorithm picks the eligible candidate with the highest
    ``in_degree × out_degree`` product on the current subgraph and removes it,
    repeating until the graph is acyclic or the budget is exhausted.

    Ties are resolved lexicographically by agent ID.

    Args:
        graph: Directed, simple NetworkX graph with string node IDs.
        budget: Maximum number of agents to select.
        candidates: Optional subset of eligible agents.

    Returns:
        Selected agent IDs ordered by selection priority, then lexicographically.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If the graph, budget, or candidates are invalid.
    """
    validate_directed_graph(graph)

    if any(not isinstance(node, str) or not node.strip() for node in graph.nodes):
        raise ValueError("All graph node identifiers must be non-empty strings.")

    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError("budget must be an integer.")
    if budget < 0:
        raise ValueError("budget must be non-negative.")

    if candidates is None:
        eligible: set[str] = set(graph.nodes)
    else:
        if isinstance(candidates, (str, bytes)):
            raise TypeError("candidates must be an iterable of agent IDs.")

        try:
            eligible_seq = tuple(candidates)
        except TypeError as exc:
            raise TypeError("candidates must be an iterable of agent IDs.") from exc

        if any(not isinstance(agent, str) or not agent.strip() for agent in eligible_seq):
            raise ValueError("candidates must contain non-empty string IDs.")
        if len(set(eligible_seq)) != len(eligible_seq):
            raise ValueError("candidates must not contain duplicate IDs.")

        unknown = set(eligible_seq) - set(graph.nodes)
        if unknown:
            raise ValueError(f"Unknown candidate agents: {sorted(unknown)!r}.")

        eligible = set(eligible_seq)

    # No containment is needed when the graph is already acyclic.
    if nx.is_directed_acyclic_graph(graph):
        return ()

    # For a cyclic graph, reject budgets larger than the eligible candidate set.
    if budget > len(eligible):
        raise ValueError(f"budget ({budget}) cannot exceed eligible agents ({len(eligible)}).")

    selected: list[str] = []
    working = graph.copy()

    for _ in range(budget):
        if nx.is_directed_acyclic_graph(working):
            break

        remaining_eligible = eligible - set(selected)
        if not remaining_eligible:
            break

        best = min(
            remaining_eligible,
            key=lambda node: _cycle_score(working, node),
        )
        selected.append(best)
        working.remove_node(best)

    return tuple(selected)
