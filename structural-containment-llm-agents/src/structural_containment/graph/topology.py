"""Topology analysis for directed agent-interaction networks."""

from __future__ import annotations

import math

import networkx as nx


def directed_random_graph(
    node_count: int,
    edge_probability: float,
    seed: int | None = 42,
) -> nx.DiGraph:
    """Generate a reproducible directed Erdős–Rényi random graph.

    Self-loops are excluded. The same node count, probability, and integer
    seed produce the same graph under the same NetworkX implementation.
    """
    if isinstance(node_count, bool) or not isinstance(node_count, int):
        raise TypeError("node_count must be an integer.")
    if node_count < 0:
        raise ValueError("node_count must be non-negative.")

    if isinstance(edge_probability, bool) or not isinstance(edge_probability, (int, float)):
        raise TypeError("edge_probability must be a real number.")

    probability = float(edge_probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("edge_probability must be finite and between 0 and 1.")

    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None.")

    return nx.gnp_random_graph(
        n=node_count,
        p=probability,
        seed=seed,
        directed=True,
    )


def is_acyclic(graph: nx.Graph) -> bool:
    """Return whether a directed graph contains no directed cycles."""
    from .validation import validate_directed_graph

    validate_directed_graph(graph)
    return nx.is_directed_acyclic_graph(graph)


def find_directed_cycle(
    graph: nx.Graph,
) -> tuple[tuple[str, str], ...] | None:
    """Return one directed cycle as ordered edges, or None if acyclic."""
    from .validation import validate_directed_graph

    validate_directed_graph(graph)

    try:
        cycle = nx.find_cycle(graph, orientation="original")
    except nx.NetworkXNoCycle:
        return None

    return tuple((source, target) for source, target, *_ in cycle)


def strongly_connected_components(
    graph: nx.Graph,
) -> tuple[frozenset[str], ...]:
    """Return strongly connected components in deterministic order."""
    from .validation import validate_directed_graph

    validate_directed_graph(graph)

    components = (frozenset(component) for component in nx.strongly_connected_components(graph))
    return tuple(
        sorted(
            components,
            key=lambda component: tuple(sorted(component)),
        )
    )
