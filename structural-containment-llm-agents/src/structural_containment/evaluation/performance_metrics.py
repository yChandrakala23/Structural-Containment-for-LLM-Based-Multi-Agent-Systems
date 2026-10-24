"""Performance metrics derived from propagation simulation results."""

from __future__ import annotations

from statistics import fmean

from structural_containment.simulation.propagation import SimulationResult
from structural_containment.simulation.state import AgentStatus


def time_to_first_compromise(result: SimulationResult) -> int | None:
    """Return the step at which the first new agent was compromised.

    Step 0 records the initially compromised agents. This function returns
    the step at which propagation first compromises another agent, or None
    if propagation never spreads beyond the initial set.
    """
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult.")

    for step_record in result.steps[1:]:
        if step_record.newly_compromised:
            return step_record.step

    return None


def steps_to_stable(result: SimulationResult) -> int:
    """Return the step at which propagation became stable.

    Returns the last step at which a new compromise occurred, excluding the
    final quiet step. Returns 1 if no additional agents were compromised.
    """
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult.")

    last_compromise_step = max(
        (
            state.compromised_at
            for state in result.final_states.values()
            if state.status == AgentStatus.COMPROMISED and state.compromised_at is not None
        ),
        default=0,
    )

    return max(1, last_compromise_step)


def mean_newly_compromised_per_step(result: SimulationResult) -> float:
    """Return the mean number of newly compromised agents per active step.

    Only steps where at least one new compromise occurred are included.
    Returns 0.0 if propagation never spread.
    """
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult.")

    active_steps = [
        len(step_record.newly_compromised)
        for step_record in result.steps[1:]
        if step_record.newly_compromised
    ]

    return fmean(active_steps) if active_steps else 0.0


def final_compromise_count(result: SimulationResult) -> int:
    """Return the total number of agents compromised at simulation end."""
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult.")

    return sum(
        1 for state in result.final_states.values() if state.status == AgentStatus.COMPROMISED
    )


def containment_overhead(result: SimulationResult) -> int:
    """Return the number of agents contained during the simulation."""
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult.")

    return sum(1 for state in result.final_states.values() if state.status == AgentStatus.CONTAINED)
