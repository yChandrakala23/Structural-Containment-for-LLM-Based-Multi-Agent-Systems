"""Abstract protocol for containment policies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import networkx as nx


@runtime_checkable
class ContainmentPolicy(Protocol):
    """Protocol satisfied by all containment selection callables.

    Any callable that accepts a directed graph and a budget (with an optional
    ``candidates`` keyword argument) and returns a tuple of selected agent IDs
    conforms to this protocol.
    """

    def __call__(
        self,
        graph: nx.DiGraph,
        budget: int,
        *,
        candidates: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        """Select agents for containment.

        Args:
            graph: Directed simple graph with string node identifiers.
            budget: Maximum number of agents to contain.
            candidates: Optional subset of agents eligible for selection.

        Returns:
            A tuple of distinct selected agent IDs, at most ``budget`` long.
        """
        ...
