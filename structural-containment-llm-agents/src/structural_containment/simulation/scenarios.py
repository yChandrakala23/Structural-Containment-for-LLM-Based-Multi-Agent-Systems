"""Scenario execution and containment effectiveness evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, pstdev

import networkx as nx

from structural_containment.containment.degree_policy import (
    highest_out_degree_containment,
)
from structural_containment.containment.random_policy import random_containment
from structural_containment.simulation.propagation import (
    SimulationResult,
    simulate_propagation,
)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Outcomes from a baseline and a containment intervention."""

    baseline: SimulationResult
    intervention: SimulationResult
    contained_agents: tuple[str, ...]
    baseline_compromise_rate: float
    intervention_compromise_rate: float
    absolute_reduction: float
    relative_reduction: float | None


@dataclass(frozen=True, slots=True)
class RepeatedScenarioResult:
    """Aggregate outcomes across independent simulation seeds."""

    runs: tuple[ScenarioResult, ...]
    mean_baseline_rate: float
    mean_intervention_rate: float
    mean_absolute_reduction: float
    standard_deviation_reduction: float


def _validate_inputs(
    graph: nx.Graph,
    initial_compromised: tuple[str, ...],
    budget: int,
    propagation_probability: float,
    max_steps: int,
) -> None:
    """Validate shared scenario parameters."""
    if not isinstance(graph, nx.DiGraph) or graph.is_multigraph():
        raise ValueError("graph must be a directed, simple NetworkX graph.")

    if any(not isinstance(node, str) or not node.strip() for node in graph.nodes):
        raise ValueError("All graph node identifiers must be non-empty strings.")

    if len(set(initial_compromised)) != len(initial_compromised):
        raise ValueError("initial_compromised must not contain duplicates.")

    unknown = set(initial_compromised) - set(graph.nodes)
    if unknown:
        raise ValueError(f"Unknown initially compromised agents: {sorted(unknown)!r}.")

    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError("budget must be an integer.")
    if budget < 0 or budget > graph.number_of_nodes():
        raise ValueError("budget must be between zero and the number of agents.")

    if isinstance(propagation_probability, bool) or not isinstance(
        propagation_probability, (int, float)
    ):
        raise TypeError("propagation_probability must be a real number.")

    if not math.isfinite(propagation_probability) or not (0.0 <= propagation_probability <= 1.0):
        raise ValueError("propagation_probability must be finite and in [0, 1].")

    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an integer.")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative.")


def _compromise_rate(result: SimulationResult, total_agents: int) -> float:
    """Calculate the fraction of all agents that ended compromised."""
    if total_agents == 0:
        return 0.0
    return len(result.compromised_agents) / total_agents


def run_containment_scenario(
    graph: nx.Graph,
    initial_compromised: tuple[str, ...] | list[str],
    *,
    policy: str = "random",
    budget: int = 1,
    propagation_probability: float = 0.2,
    max_steps: int = 100,
    seed: int = 42,
) -> ScenarioResult:
    """Compare an uncontained baseline against one containment policy.

    Both runs use the same graph, initial attack, propagation probability,
    maximum steps, and simulation seed. Containment is applied before
    propagation starts.

    Policy choices are ``"random"`` and ``"highest_out_degree"``.
    """
    initial = tuple(initial_compromised)
    _validate_inputs(graph, initial, budget, propagation_probability, max_steps)

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    eligible_candidates = tuple(
        agent_id for agent_id in graph.nodes if agent_id not in set(initial)
    )

    if budget > len(eligible_candidates):
        raise ValueError(
            "budget cannot exceed the number of agents eligible for containment "
            "after excluding initially compromised agents."
        )

    if policy == "random":
        contained = random_containment(graph, budget, candidates=eligible_candidates, seed=seed)
    elif policy == "highest_out_degree":
        contained = highest_out_degree_containment(graph, budget, candidates=eligible_candidates)
    else:
        raise ValueError("Unsupported policy. Use 'random' or 'highest_out_degree'.")

    overlap = set(initial) & set(contained)
    if overlap:
        raise ValueError(
            "Containment policy selected initially compromised agents: "
            f"{sorted(overlap)!r}. Choose eligible candidates excluding the attack set."
        )

    baseline = simulate_propagation(
        graph,
        initial,
        propagation_probability=propagation_probability,
        max_steps=max_steps,
        seed=seed,
    )
    intervention = simulate_propagation(
        graph,
        initial,
        initial_contained=contained,
        propagation_probability=propagation_probability,
        max_steps=max_steps,
        seed=seed,
    )

    total_agents = graph.number_of_nodes()
    baseline_rate = _compromise_rate(baseline, total_agents)
    intervention_rate = _compromise_rate(intervention, total_agents)
    absolute_reduction = baseline_rate - intervention_rate
    relative_reduction = absolute_reduction / baseline_rate if baseline_rate > 0 else None

    return ScenarioResult(
        baseline=baseline,
        intervention=intervention,
        contained_agents=contained,
        baseline_compromise_rate=baseline_rate,
        intervention_compromise_rate=intervention_rate,
        absolute_reduction=absolute_reduction,
        relative_reduction=relative_reduction,
    )


def run_repeated_scenarios(
    graph: nx.Graph,
    initial_compromised: tuple[str, ...] | list[str],
    *,
    policy: str = "random",
    budget: int = 1,
    propagation_probability: float = 0.2,
    max_steps: int = 100,
    seeds: tuple[int, ...] = (11, 22, 33, 44, 55),
) -> RepeatedScenarioResult:
    """Repeat scenario comparisons across multiple seeds.

    For random containment, each seed determines the selected containment
    set and the propagation random stream. The same seed is used for the
    baseline and intervention within each comparison.
    """
    if not seeds:
        raise ValueError("seeds must contain at least one seed.")

    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("every seed must be an integer.")

    runs = tuple(
        run_containment_scenario(
            graph,
            initial_compromised,
            policy=policy,
            budget=budget,
            propagation_probability=propagation_probability,
            max_steps=max_steps,
            seed=seed,
        )
        for seed in seeds
    )

    reductions = [run.absolute_reduction for run in runs]

    return RepeatedScenarioResult(
        runs=runs,
        mean_baseline_rate=fmean(run.baseline_compromise_rate for run in runs),
        mean_intervention_rate=fmean(run.intervention_compromise_rate for run in runs),
        mean_absolute_reduction=fmean(reductions),
        standard_deviation_reduction=pstdev(reductions),
    )
