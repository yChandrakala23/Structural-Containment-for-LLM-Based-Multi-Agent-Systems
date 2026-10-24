"""Construction utilities for directed agent-interaction graphs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import networkx as nx


def build_directed_graph(
    nodes: Sequence[str],
    edges: Iterable[tuple[str, str]],
    *,
    allow_self_loops: bool = False,
) -> nx.DiGraph:
    """Build a directed simple graph from node identifiers and edges.

    Args:
        nodes: Unique, non-empty string node identifiers.
        edges: Iterable of (source, target) pairs.
        allow_self_loops: Whether edges from a node to itself are permitted.

    Returns:
        A NetworkX DiGraph containing the supplied nodes and edges.

    Raises:
        TypeError: If inputs or their elements have unsupported types.
        ValueError: If identifiers or edge definitions are invalid.

    Notes:
        Duplicate edges are naturally collapsed by DiGraph. Node and edge
        insertion order is preserved for the supplied inputs.
    """
    if isinstance(nodes, (str, bytes)) or not isinstance(nodes, Sequence):
        raise TypeError("nodes must be a sequence of string identifiers.")

    if not isinstance(allow_self_loops, bool):
        raise TypeError("allow_self_loops must be a boolean.")

    validated_nodes: list[str] = []
    seen_nodes: set[str] = set()

    for node in nodes:
        if not isinstance(node, str):
            raise TypeError("Node identifiers must be strings.")
        if not node.strip():
            raise ValueError("Node identifiers must not be empty or whitespace.")
        if node in seen_nodes:
            raise ValueError(f"Duplicate node identifier: {node!r}.")

        seen_nodes.add(node)
        validated_nodes.append(node)

    graph = nx.DiGraph()
    graph.add_nodes_from(validated_nodes)

    try:
        edge_iterator = iter(edges)
    except TypeError as exc:
        raise TypeError("edges must be an iterable of endpoint pairs.") from exc

    for index, edge in enumerate(edge_iterator):
        if isinstance(edge, (str, bytes)):
            raise TypeError(f"Edge at index {index} must be an endpoint pair.")

        try:
            endpoints = tuple(edge)
        except TypeError as exc:
            raise TypeError(f"Edge at index {index} must be an iterable of two endpoints.") from exc

        if len(endpoints) != 2:
            raise ValueError(f"Edge at index {index} must contain exactly two endpoints.")

        source, target = endpoints

        if not isinstance(source, str) or not isinstance(target, str):
            raise TypeError(f"Edge at index {index} must use string identifiers.")

        if source not in seen_nodes or target not in seen_nodes:
            raise ValueError(f"Edge ({source!r}, {target!r}) references an unknown node.")

        if source == target and not allow_self_loops:
            raise ValueError(f"Self-loop at {source!r} is not allowed.")

        graph.add_edge(source, target)

    return graph
