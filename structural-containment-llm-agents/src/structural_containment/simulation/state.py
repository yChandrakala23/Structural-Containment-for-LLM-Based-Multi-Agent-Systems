"""Typed state management for agent compromise simulations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class AgentStatus(StrEnum):
    """Security status of a simulated agent."""

    HEALTHY = "healthy"
    COMPROMISED = "compromised"
    CONTAINED = "contained"


@dataclass(frozen=True, slots=True)
class AgentState:
    """Immutable state snapshot for a single agent."""

    agent_id: str
    status: AgentStatus = AgentStatus.HEALTHY
    compromised_at: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError("agent_id must be a non-empty string.")

        if not isinstance(self.status, AgentStatus):
            raise TypeError("status must be an AgentStatus value.")

        if self.compromised_at is not None and (
            isinstance(self.compromised_at, bool)
            or not isinstance(self.compromised_at, int)
            or self.compromised_at < 0
        ):
            raise ValueError("compromised_at must be a non-negative integer.")

        if self.status == AgentStatus.COMPROMISED and self.compromised_at is None:
            raise ValueError("A compromised agent must have a compromise time.")

        if self.status != AgentStatus.COMPROMISED and self.compromised_at is not None:
            raise ValueError("Only compromised agents may have a compromise timestamp.")


class SimulationState:
    """Manage agent states and their simulation-time transitions.

    Containment is reversible via release(). Compromise is permanent for a
    simulation run; a compromised agent cannot be reset to healthy.
    """

    def __init__(self, agent_ids: Iterable[str]) -> None:
        states: dict[str, AgentState] = {}

        for agent_id in agent_ids:
            if not isinstance(agent_id, str) or not agent_id.strip():
                raise ValueError("Agent identifiers must be non-empty strings.")
            if agent_id in states:
                raise ValueError(f"Duplicate agent identifier: {agent_id!r}.")
            states[agent_id] = AgentState(agent_id=agent_id)

        self._states = states

    @property
    def agents(self) -> Mapping[str, AgentState]:
        """Read-only view of all agent states."""
        return MappingProxyType(self._states)

    def get(self, agent_id: str) -> AgentState:
        """Return the state of a known agent."""
        try:
            return self._states[agent_id]
        except KeyError as exc:
            raise KeyError(f"Unknown agent: {agent_id!r}.") from exc

    def compromise(self, agent_id: str, step: int) -> bool:
        """Compromise a healthy agent; return False if it cannot transition."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer.")

        current = self.get(agent_id)
        if current.status != AgentStatus.HEALTHY:
            return False

        self._states[agent_id] = AgentState(
            agent_id=agent_id,
            status=AgentStatus.COMPROMISED,
            compromised_at=step,
        )
        return True

    def contain(self, agent_id: str) -> bool:
        """Contain a healthy agent; return False if it cannot be contained."""
        current = self.get(agent_id)
        if current.status != AgentStatus.HEALTHY:
            return False

        self._states[agent_id] = AgentState(
            agent_id=agent_id,
            status=AgentStatus.CONTAINED,
        )
        return True

    def release(self, agent_id: str) -> bool:
        """Release a contained agent back to healthy status."""
        current = self.get(agent_id)
        if current.status != AgentStatus.CONTAINED:
            return False

        self._states[agent_id] = AgentState(agent_id=agent_id)
        return True

    def is_spread_eligible(self, agent_id: str) -> bool:
        """Return whether an agent is currently allowed to spread compromise."""
        return self.get(agent_id).status == AgentStatus.COMPROMISED

    def is_vulnerable(self, agent_id: str) -> bool:
        """Return whether an agent can currently become compromised."""
        return self.get(agent_id).status == AgentStatus.HEALTHY
