"""Execution engine for reproducible containment experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import networkx as nx

from structural_containment.experiments.config import ExperimentConfig
from structural_containment.simulation.attacks import (
    select_explicit_targets,
    select_highest_out_degree_targets,
    select_random_targets,
)
from structural_containment.simulation.scenarios import (
    ScenarioResult,
    run_containment_scenario,
)


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """One scenario result with the settings that produced it."""

    attack_strategy: str
    policy: str
    budget: int
    seed: int
    initial_compromised: tuple[str, ...]
    result: ScenarioResult


@dataclass(frozen=True, slots=True)
class ExperimentBatch:
    """Results from all configurations in an experiment batch."""

    runs: tuple[ExperimentRun, ...]

    @property
    def run_count(self) -> int:
        """Return the number of completed runs."""
        return len(self.runs)


def _select_attack_targets(
    graph: nx.DiGraph,
    config: ExperimentConfig,
    seed: int,
) -> tuple[str, ...]:
    """Select initial compromised agents according to the configured strategy."""
    count = len(config.initial_compromised)

    if config.attack_strategy == "targeted":
        return tuple(select_explicit_targets(graph, config.initial_compromised))
    if config.attack_strategy == "random":
        return tuple(select_random_targets(graph, count, seed=seed))
    if config.attack_strategy == "highest_out_degree":
        return tuple(select_highest_out_degree_targets(graph, count))

    raise ValueError(f"Unsupported attack strategy: {config.attack_strategy!r}.")


def run_experiments(
    graph: nx.DiGraph,
    config: ExperimentConfig,
) -> ExperimentBatch:
    """Run every policy, budget and seed combination in the configuration."""
    if not isinstance(graph, nx.DiGraph) or graph.is_multigraph():
        raise TypeError("graph must be a simple directed NetworkX graph.")

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig.")

    runs: list[ExperimentRun] = []

    for policy, budget, seed in product(
        config.policies,
        config.budgets,
        config.seeds,
    ):
        initial = _select_attack_targets(graph, config, seed)

        # Check feasibility before starting a simulation.
        if not set(initial).issubset(graph.nodes):
            raise ValueError("Initial compromised agents must exist in the graph.")

        eligible_count = graph.number_of_nodes() - len(set(initial))
        if budget > eligible_count:
            raise ValueError(
                "Containment budget exceeds the number of agents eligible "
                "after excluding initially compromised agents."
            )

        result = run_containment_scenario(
            graph,
            initial,
            policy=policy,
            budget=budget,
            propagation_probability=config.propagation_probability,
            max_steps=config.max_steps,
            seed=seed,
        )

        runs.append(
            ExperimentRun(
                attack_strategy=config.attack_strategy,
                policy=policy,
                budget=budget,
                seed=seed,
                initial_compromised=initial,
                result=result,
            )
        )

    return ExperimentBatch(runs=tuple(runs))
