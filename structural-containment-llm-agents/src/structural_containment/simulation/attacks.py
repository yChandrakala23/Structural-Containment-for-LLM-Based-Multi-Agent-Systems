"""Initial compromise selection strategies for directed agent networks."""

from __future__ import annotations

import random
from collections.abc import Iterable

import networkx as nx

from structural_containment.graph.validation import validate_directed_graph


def _validate_graph_nodes(graph: nx.Graph) -> None:
    """Validate the graph structure and agent identifier contract."""
    validate_directed_graph(graph)

    if any(not isinstance(node, str) or not node.strip() for node in graph.nodes):
        raise ValueError("All graph node identifiers must be non-empty strings.")


def _validate_count(count: int, available: int) -> None:
    """Validate a requested number of initial compromised agents."""
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer.")

    if count < 0:
        raise ValueError("count must be non-negative.")

    if count > available:
        raise ValueError(f"count ({count}) cannot exceed available agents ({available}).")


def select_random_targets(
    graph: nx.Graph,
    count: int,
    *,
    seed: int | None = 42,
) -> tuple[str, ...]:
    """Select distinct initial targets uniformly at random.

    A fixed seed makes the selection reproducible for the same graph and
    node iteration order.
    """
    _validate_graph_nodes(graph)
    _validate_count(count, graph.number_of_nodes())

    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None.")

    nodes = list(graph.nodes)
    return tuple(random.Random(seed).sample(nodes, count))


def select_highest_out_degree_targets(
    graph: nx.Graph,
    count: int,
) -> tuple[str, ...]:
    """Select agents with the highest outgoing degree.

    Ties are resolved lexicographically by agent ID, making the result
    deterministic and independent of graph insertion order.
    """
    _validate_graph_nodes(graph)
    _validate_count(count, graph.number_of_nodes())

    ranked_nodes = sorted(
        graph.nodes,
        key=lambda node: (-graph.out_degree(node), node),
    )
    return tuple(ranked_nodes[:count])


def select_explicit_targets(
    graph: nx.Graph,
    targets: Iterable[str],
) -> tuple[str, ...]:
    """Validate and return explicitly selected initial compromise targets."""
    _validate_graph_nodes(graph)

    if isinstance(targets, (str, bytes)):
        raise TypeError("targets must be an iterable of agent IDs, not a string.")

    try:
        selected = tuple(targets)
    except TypeError as exc:
        raise TypeError("targets must be an iterable of agent IDs.") from exc

    if any(not isinstance(target, str) or not target.strip() for target in selected):
        raise ValueError("targets must contain non-empty string IDs.")

    if len(set(selected)) != len(selected):
        raise ValueError("targets must not contain duplicate agent IDs.")

    unknown = set(selected) - set(graph.nodes)
    if unknown:
        raise ValueError(f"targets contain unknown agents: {sorted(unknown)!r}.")

    return selected
