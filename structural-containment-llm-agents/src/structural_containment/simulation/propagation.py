"""Synchronous compromise propagation over directed agent networks."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import networkx as nx

from structural_containment.graph.validation import validate_directed_graph
from structural_containment.simulation.state import (
    AgentState,
    AgentStatus,
    SimulationState,
)


class TerminationReason(StrEnum):
    """Why a propagation simulation stopped."""

    STABLE = "stable"
    MAX_STEPS = "max_steps"


@dataclass(frozen=True, slots=True)
class SimulationStep:
    """Immutable record of the state after one propagation attempt."""

    step: int
    newly_compromised: tuple[str, ...]
    statuses: Mapping[str, AgentStatus]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Immutable summary of a completed propagation simulation."""

    steps: tuple[SimulationStep, ...]
    final_states: Mapping[str, AgentState]
    termination_reason: TerminationReason

    @property
    def steps_executed(self) -> int:
        """Return the number of propagation attempts executed."""
        return max(0, len(self.steps) - 1)

    @property
    def compromised_agents(self) -> frozenset[str]:
        """Return all agents compromised when the simulation ended."""
        return frozenset(
            agent_id
            for agent_id, state in self.final_states.items()
            if state.status == AgentStatus.COMPROMISED
        )


def _validated_agent_ids(
    agent_ids: Iterable[str],
    *,
    argument_name: str,
    known_agents: set[str],
) -> tuple[str, ...]:
    """Validate and normalize an iterable of agent identifiers."""
    if isinstance(agent_ids, (str, bytes)):
        raise TypeError(f"{argument_name} must be an iterable of agent IDs.")

    try:
        identifiers = tuple(agent_ids)
    except TypeError as exc:
        raise TypeError(f"{argument_name} must be an iterable of agent IDs.") from exc

    if any(not isinstance(agent_id, str) or not agent_id.strip() for agent_id in identifiers):
        raise ValueError(f"{argument_name} must contain non-empty string IDs.")

    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{argument_name} must not contain duplicate IDs.")

    unknown = set(identifiers) - known_agents
    if unknown:
        raise ValueError(f"{argument_name} contains unknown agents: {sorted(unknown)!r}.")

    return identifiers


def _snapshot(state: SimulationState) -> Mapping[str, AgentStatus]:
    """Create an immutable snapshot of current agent statuses."""
    return MappingProxyType(
        {agent_id: agent_state.status for agent_id, agent_state in state.agents.items()}
    )


def simulate_propagation(
    graph: nx.Graph,
    initial_compromised: Iterable[str],
    *,
    initial_contained: Iterable[str] = (),
    propagation_probability: float = 0.2,
    max_steps: int = 100,
    seed: int | None = 42,
) -> SimulationResult:
    """Simulate synchronous compromise propagation on a directed graph.

    Each eligible directed edge independently attempts to compromise its
    target with the configured probability. A newly compromised agent cannot
    spread compromise until the next step.

    Args:
        graph: Directed simple NetworkX graph with string node identifiers.
        initial_compromised: Agents compromised at simulation step zero.
        initial_contained: Healthy agents isolated before propagation begins.
        propagation_probability: Per-edge probability in [0, 1].
        max_steps: Maximum number of propagation attempts.
        seed: Integer random seed, or None for non-reproducible randomness.

    Returns:
        Immutable step records, final agent states, and termination reason.

    Raises:
        TypeError: If arguments have unsupported types.
        ValueError: If arguments or initial agent sets are invalid.
    """
    validate_directed_graph(graph)

    if any(not isinstance(node, str) or not node.strip() for node in graph.nodes):
        raise ValueError("All graph node identifiers must be non-empty strings.")

    if isinstance(propagation_probability, bool) or not isinstance(
        propagation_probability, (int, float)
    ):
        raise TypeError("propagation_probability must be a real number.")

    probability = float(propagation_probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("propagation_probability must be finite and between 0 and 1.")

    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an integer.")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative.")

    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None.")

    known_agents = set(graph.nodes)
    compromised = _validated_agent_ids(
        initial_compromised,
        argument_name="initial_compromised",
        known_agents=known_agents,
    )
    contained = _validated_agent_ids(
        initial_contained,
        argument_name="initial_contained",
        known_agents=known_agents,
    )

    overlap = set(compromised) & set(contained)
    if overlap:
        raise ValueError(
            f"An agent cannot be initially compromised and contained: {sorted(overlap)!r}."
        )

    state = SimulationState(graph.nodes)
    for agent_id in contained:
        state.contain(agent_id)
    for agent_id in compromised:
        state.compromise(agent_id, step=0)

    records = [
        SimulationStep(
            step=0,
            newly_compromised=tuple(compromised),
            statuses=_snapshot(state),
        )
    ]
    rng = random.Random(seed)
    termination_reason = TerminationReason.STABLE

    for step in range(1, max_steps + 1):
        newly_compromised: set[str] = set()

        # Read only the state at the start of this step.
        for source in graph.nodes:
            if not state.is_spread_eligible(source):
                continue

            for target in graph.successors(source):
                if not state.is_vulnerable(target):
                    continue
                if target in newly_compromised:
                    continue

                if rng.random() < probability:
                    newly_compromised.add(target)

        # Apply transitions only after all propagation attempts finish.
        ordered_new = tuple(node for node in graph.nodes if node in newly_compromised)
        for agent_id in ordered_new:
            state.compromise(agent_id, step=step)

        records.append(
            SimulationStep(
                step=step,
                newly_compromised=ordered_new,
                statuses=_snapshot(state),
            )
        )

        if not ordered_new:
            termination_reason = TerminationReason.STABLE
            break
    else:
        termination_reason = TerminationReason.MAX_STEPS

    return SimulationResult(
        steps=tuple(records),
        final_states=MappingProxyType(dict(state.agents)),
        termination_reason=termination_reason,
    )
