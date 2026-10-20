"""Degree-based containment policy for directed agent networks."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from structural_containment.graph.validation import validate_directed_graph


def highest_out_degree_containment(
    graph: nx.Graph,
    budget: int,
    *,
    candidates: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Select eligible agents with the highest outgoing degree.

    Ties are resolved lexicographically by agent ID.

    Args:
        graph: Directed, simple NetworkX graph with string node IDs.
        budget: Number of agents to select.
        candidates: Optional subset of eligible agents.

    Returns:
        Selected agent IDs ordered by descending out-degree, then ID.

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
        eligible = tuple(graph.nodes)
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

    ranked = sorted(
        eligible,
        key=lambda agent_id: (-graph.out_degree(agent_id), agent_id),
    )
    return tuple(ranked[:budget])
