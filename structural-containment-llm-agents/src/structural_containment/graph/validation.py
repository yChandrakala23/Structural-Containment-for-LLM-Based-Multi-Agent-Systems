
"""Validation contracts for agent-interaction graphs."""

from __future__ import annotations

import networkx as nx


def validate_directed_graph(graph: nx.Graph) -> None:
    """Validate that an object is a directed, simple NetworkX graph.

    This function validates the graph's structural type. It does not enforce
    application-specific policies such as maximum graph size or self-loop
    restrictions; those belong to construction or simulation configuration.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph instance.")

    if not graph.is_directed():
        raise ValueError("Expected a directed graph.")

    if graph.is_multigraph():
        raise ValueError("MultiDiGraph is not supported; use a simple DiGraph.")
