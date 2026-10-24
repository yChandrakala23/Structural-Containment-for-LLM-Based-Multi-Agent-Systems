"""Containment policies for structural agent isolation.

Available policies
------------------
- :func:`~structural_containment.containment.random_policy.random_containment`
  — Uniform random selection.
- :func:`~structural_containment.containment.degree_policy.highest_out_degree_containment`
  — Prioritise agents with the most outgoing connections.
- :func:`~structural_containment.containment.centrality.betweenness_centrality_containment`
  — Prioritise agents on critical shortest paths.
- :func:`~structural_containment.containment.fvs_heuristic.fvs_heuristic_containment`
  — Greedy feedback vertex set heuristic (any graph size).
- :func:`~structural_containment.containment.fvs_exact.fvs_exact_containment`
  — Exact feedback vertex set (graphs ≤ 20 nodes).
- :func:`~structural_containment.containment.isolation.apply_isolation`
  — Remove edges incident to contained agents.
- :func:`~structural_containment.containment.isolation.compute_isolation_set`
  — Dry-run: compute edges that would be cut.
"""

from structural_containment.containment.base import ContainmentPolicy
from structural_containment.containment.centrality import (
    betweenness_centrality_containment,
)
from structural_containment.containment.degree_policy import (
    highest_out_degree_containment,
)
from structural_containment.containment.fvs_exact import (
    MAX_EXACT_NODES,
    fvs_exact_containment,
)
from structural_containment.containment.fvs_heuristic import fvs_heuristic_containment
from structural_containment.containment.isolation import (
    apply_isolation,
    compute_isolation_set,
)
from structural_containment.containment.random_policy import random_containment

__all__ = [
    "MAX_EXACT_NODES",
    "ContainmentPolicy",
    "apply_isolation",
    "betweenness_centrality_containment",
    "compute_isolation_set",
    "fvs_exact_containment",
    "fvs_heuristic_containment",
    "highest_out_degree_containment",
    "random_containment",
]
