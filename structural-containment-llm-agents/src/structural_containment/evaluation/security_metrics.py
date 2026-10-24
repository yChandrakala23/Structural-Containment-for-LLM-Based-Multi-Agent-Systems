"""Security-oriented metrics for assessing containment effectiveness."""

from __future__ import annotations

import networkx as nx

from structural_containment.simulation.propagation import SimulationResult
from structural_containment.simulation.state import AgentStatus


def containment_coverage(
    result: SimulationResult,
    total_agents: int,
) -> float:
    """Return the fraction of agents that remained uncompromised at simulation end.

    A higher coverage means the containment strategy protected more agents.
    Contained agents count as *protected* (they are not compromised).

    Args:
        result: A completed simulation result.
        total_agents: Total number of agents in the network (denominator).

    Returns:
        A float in ``[0.0, 1.0]``.  Returns ``1.0`` for an empty network.

    Raises:
        TypeError: If ``result`` is not a ``SimulationResult``.
        ValueError: If ``total_agents`` is negative.
    """
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult.")

    if isinstance(total_agents, bool) or not isinstance(total_agents, int):
        raise TypeError("total_agents must be an integer.")
    if total_agents < 0:
        raise ValueError("total_agents must be non-negative.")
    if total_agents == 0:
        return 1.0

    compromised_count = sum(
        1 for state in result.final_states.values() if state.status == AgentStatus.COMPROMISED
    )
    return 1.0 - compromised_count / total_agents


def isolation_quality(
    contained_agents: frozenset[str] | set[str] | tuple[str, ...] | list[str],
    compromised_agents: frozenset[str] | set[str],
    total_agents: int,
) -> float:
    """Return a quality score for the chosen isolation set.

    Quality is defined as the fraction of contained agents that were *not*
    compromised at simulation end (i.e., the containment successfully
    protected them or they were never targeted).  A score of ``1.0`` means
    every contained agent was genuinely protected; ``0.0`` means all contained
    agents were compromised anyway (containment was futile).

    Returns ``1.0`` if no agents were contained (vacuous case).

    Args:
        contained_agents: Agents selected for containment.
        compromised_agents: Agents compromised at simulation end.
        total_agents: Total network size (for context; not used in score).

    Returns:
        A float in ``[0.0, 1.0]``.

    Raises:
        TypeError: If arguments have unsupported types.
        ValueError: If ``total_agents`` is negative.
    """
    if isinstance(contained_agents, (str, bytes)):
        raise TypeError("contained_agents must be a collection of agent IDs.")

    if isinstance(compromised_agents, (str, bytes)):
        raise TypeError("compromised_agents must be a collection of agent IDs.")

    if isinstance(total_agents, bool) or not isinstance(total_agents, int):
        raise TypeError("total_agents must be an integer.")
    if total_agents < 0:
        raise ValueError("total_agents must be non-negative.")

    contained_set = set(contained_agents)
    if not contained_set:
        return 1.0

    compromised_and_contained = contained_set & set(compromised_agents)
    return 1.0 - len(compromised_and_contained) / len(contained_set)


def reachability_reduction(
    graph: nx.DiGraph,
    contained_agents: frozenset[str] | set[str] | tuple[str, ...] | list[str],
) -> float:
    """Return the fraction by which the mean reachability is reduced.

    Reachability of a node is the fraction of other nodes it can reach via
    directed paths.  This metric computes the mean reachability before and
    after removing the contained agents, then returns the relative reduction.

    A higher value means containment more effectively breaks propagation paths.

    Args:
        graph: Directed, simple NetworkX graph.
        contained_agents: Agents to remove when computing post-containment
            reachability.

    Returns:
        A float in ``[0.0, 1.0]``.  Returns ``0.0`` for trivial graphs or
        when containment has no effect.

    Raises:
        TypeError: If ``graph`` is not a directed NetworkX graph.
        ValueError: If any contained agent is not in the graph.
    """
    from structural_containment.graph.validation import validate_directed_graph

    validate_directed_graph(graph)

    if isinstance(contained_agents, (str, bytes)):
        raise TypeError("contained_agents must be a collection of agent IDs.")

    agent_set = set(contained_agents)
    unknown = agent_set - set(graph.nodes)
    if unknown:
        raise ValueError(f"Unknown agents: {sorted(unknown)!r}.")

    n = graph.number_of_nodes()
    if n <= 1:
        return 0.0

    def _mean_reachability(g: nx.DiGraph) -> float:
        total = sum(len(nx.descendants(g, node)) for node in g.nodes)
        size = g.number_of_nodes()
        denominator = size * (size - 1)
        return total / denominator if denominator > 0 else 0.0

    baseline = _mean_reachability(graph)
    if baseline == 0.0:
        return 0.0

    post = graph.subgraph(set(graph.nodes) - agent_set)
    post_reach = _mean_reachability(post)

    return max(0.0, (baseline - post_reach) / baseline)
