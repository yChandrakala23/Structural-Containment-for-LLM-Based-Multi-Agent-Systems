"""Graph isolation utilities for structural containment.

Isolation is the act of removing all edges incident to a set of contained
agents so that they can neither receive nor spread compromise, regardless of
the propagation simulator's state machine.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from structural_containment.graph.validation import validate_directed_graph


def apply_isolation(
    graph: nx.DiGraph,
    agents: Iterable[str],
    *,
    in_edges: bool = True,
    out_edges: bool = True,
) -> nx.DiGraph:
    """Return a copy of *graph* with all edges to/from *agents* removed.

    The original graph is not modified.

    Args:
        graph: Directed, simple NetworkX graph with string node IDs.
        agents: Agent IDs to isolate.
        in_edges: Remove incoming edges to each isolated agent.
        out_edges: Remove outgoing edges from each isolated agent.

    Returns:
        A new DiGraph with the specified edges removed.  All nodes are
        preserved so that isolated agents remain visible in the network.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If the graph or agent IDs are invalid.
    """
    validate_directed_graph(graph)

    if any(not isinstance(node, str) or not node.strip() for node in graph.nodes):
        raise ValueError("All graph node identifiers must be non-empty strings.")

    if isinstance(agents, (str, bytes)):
        raise TypeError("agents must be an iterable of agent IDs.")

    try:
        agent_list = tuple(agents)
    except TypeError as exc:
        raise TypeError("agents must be an iterable of agent IDs.") from exc

    if any(not isinstance(agent, str) or not agent.strip() for agent in agent_list):
        raise ValueError("agents must contain non-empty string IDs.")

    if len(set(agent_list)) != len(agent_list):
        raise ValueError("agents must not contain duplicate IDs.")

    unknown = set(agent_list) - set(graph.nodes)
    if unknown:
        raise ValueError(f"Unknown agents: {sorted(unknown)!r}.")

    isolated = set(agent_list)
    result = graph.copy()

    edges_to_remove: list[tuple[str, str]] = []
    for source, target in list(result.edges()):
        if out_edges and source in isolated or in_edges and target in isolated:
            edges_to_remove.append((source, target))

    result.remove_edges_from(edges_to_remove)
    return result


def compute_isolation_set(
    graph: nx.DiGraph,
    contained_agents: Iterable[str],
) -> frozenset[tuple[str, str]]:
    """Return the set of edges that would be removed by isolating *contained_agents*.

    This is a dry-run version of :func:`apply_isolation` that returns the
    edges that would be cut without modifying the graph.

    Args:
        graph: Directed, simple NetworkX graph with string node IDs.
        contained_agents: Agent IDs whose incident edges to compute.

    Returns:
        Frozenset of ``(source, target)`` edge tuples that would be removed.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If the graph or agent IDs are invalid.
    """
    validate_directed_graph(graph)

    if any(not isinstance(node, str) or not node.strip() for node in graph.nodes):
        raise ValueError("All graph node identifiers must be non-empty strings.")

    if isinstance(contained_agents, (str, bytes)):
        raise TypeError("contained_agents must be an iterable of agent IDs.")

    try:
        agent_list = tuple(contained_agents)
    except TypeError as exc:
        raise TypeError("contained_agents must be an iterable of agent IDs.") from exc

    if any(not isinstance(agent, str) or not agent.strip() for agent in agent_list):
        raise ValueError("contained_agents must contain non-empty string IDs.")

    unknown = set(agent_list) - set(graph.nodes)
    if unknown:
        raise ValueError(f"Unknown agents: {sorted(unknown)!r}.")

    isolated = set(agent_list)
    cut: set[tuple[str, str]] = set()

    for source, target in graph.edges():
        if source in isolated or target in isolated:
            cut.add((source, target))

    return frozenset(cut)
