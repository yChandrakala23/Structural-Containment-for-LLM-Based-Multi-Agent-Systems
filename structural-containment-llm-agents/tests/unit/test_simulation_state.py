"""Tests for simulation state management."""

import pytest

from structural_containment.simulation.state import (
    AgentState,
    AgentStatus,
    SimulationState,
)


def test_agent_status_values():
    assert AgentStatus.HEALTHY == "healthy"
    assert AgentStatus.COMPROMISED == "compromised"
    assert AgentStatus.CONTAINED == "contained"
    assert len(AgentStatus) == 3


def test_initial_agents_are_healthy():
    state = SimulationState(["a", "b"])
    assert state.get("a").status == AgentStatus.HEALTHY
    assert state.get("b").status == AgentStatus.HEALTHY


def test_compromise_records_step():
    state = SimulationState(["a"])
    assert state.compromise("a", 3)
    assert state.get("a").status == AgentStatus.COMPROMISED
    assert state.get("a").compromised_at == 3


def test_compromise_is_not_applied_twice():
    state = SimulationState(["a"])
    assert state.compromise("a", 1)
    assert not state.compromise("a", 2)
    assert state.get("a").compromised_at == 1


def test_containment_and_release():
    state = SimulationState(["a"])
    assert state.contain("a")
    assert not state.is_vulnerable("a")
    assert not state.is_spread_eligible("a")
    assert state.release("a")
    assert state.is_vulnerable("a")


def test_compromised_agent_cannot_be_contained_or_released():
    state = SimulationState(["a"])
    state.compromise("a", 0)
    assert not state.contain("a")
    assert not state.release("a")


def test_unknown_agent_is_rejected():
    state = SimulationState(["a"])
    with pytest.raises(KeyError, match="Unknown agent"):
        state.get("missing")


def test_duplicate_agent_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        SimulationState(["a", "a"])


@pytest.mark.parametrize("step", [-1, True, 1.5])
def test_invalid_compromise_step_is_rejected(step):
    state = SimulationState(["a"])
    with pytest.raises(ValueError):
        state.compromise("a", step)


@pytest.mark.parametrize("agent_id", ["", "  "])
def test_empty_agent_ids_are_rejected(agent_id):
    with pytest.raises(ValueError):
        SimulationState([agent_id])


def test_agent_state_requires_compromise_time():
    with pytest.raises(ValueError, match="compromise time"):
        AgentState("a", status=AgentStatus.COMPROMISED)


def test_agent_state_rejects_timestamp_for_healthy_agent():
    with pytest.raises(ValueError, match="Only compromised"):
        AgentState("a", compromised_at=0)


def test_agent_state_is_immutable():
    agent = AgentState("a")
    with pytest.raises(AttributeError):
        agent.status = AgentStatus.COMPROMISED


def test_agent_mapping_is_read_only():
    state = SimulationState(["a"])
    with pytest.raises(TypeError):
        state.agents["b"] = AgentState("b")


def test_negative_compromise_timestamp_is_rejected():

    with pytest.raises(ValueError, match="non-negative integer"):
        AgentState(
            agent_id="a",
            status=AgentStatus.COMPROMISED,
            compromised_at=-1,
        )


def test_boolean_compromise_timestamp_is_rejected():

    with pytest.raises(ValueError, match="non-negative integer"):
        AgentState(
            agent_id="a",
            status=AgentStatus.COMPROMISED,
            compromised_at=True,
        )


def test_healthy_agent_cannot_have_compromise_timestamp():

    with pytest.raises(ValueError, match="Only compromised agents"):
        AgentState(agent_id="a", compromised_at=0)
