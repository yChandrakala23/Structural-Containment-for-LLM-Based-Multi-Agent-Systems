
"""Run comparable containment experiments across graph topologies."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import networkx as nx

from structural_containment.experiments.config import ExperimentConfig
from structural_containment.experiments.export import batch_to_records
from structural_containment.experiments.runner import ExperimentBatch, run_experiments


@dataclass(frozen=True, slots=True)
class TopologyBenchmark:
    """Experiment results associated with one named graph topology."""

    topology: str
    batch: ExperimentBatch


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """Results from a benchmark spanning multiple graph topologies."""

    benchmarks: tuple[TopologyBenchmark, ...]

    @property
    def run_count(self) -> int:
        """Return the total number of completed experiment runs."""
        return sum(item.batch.run_count for item in self.benchmarks)

    def to_records(self) -> list[dict[str, Any]]:
        """Flatten all results and retain their topology labels."""
        return [
            {"topology": item.topology, **record}
            for item in self.benchmarks
            for record in batch_to_records(item.batch)
        ]


def run_benchmark_suite(
    graphs: Mapping[str, nx.DiGraph],
    initial_compromised_by_topology: Mapping[str, tuple[str, ...]],
    *,
    attack_strategy: str = "targeted",
    policies: tuple[str, ...] = ("random", "highest_out_degree"),
    budgets: tuple[int, ...] = (0, 1, 2),
    propagation_probability: float = 0.2,
    max_steps: int = 100,
    seeds: tuple[int, ...] = (11, 22, 33, 44, 55),
) -> BenchmarkSuite:
    """Run the same experiment settings against each supplied graph."""
    if not graphs:
        raise ValueError("graphs must contain at least one topology.")

    if set(graphs) != set(initial_compromised_by_topology):
        raise ValueError(
            "graphs and initial_compromised_by_topology must have identical keys."
        )

    benchmarks = []

    for topology, graph in graphs.items():
        config = ExperimentConfig(
            initial_compromised=initial_compromised_by_topology[topology],
            attack_strategy=attack_strategy,
            policies=policies,
            budgets=budgets,
            propagation_probability=propagation_probability,
            max_steps=max_steps,
            seeds=seeds,
        )

        batch = run_experiments(graph, config)
        benchmarks.append(TopologyBenchmark(topology=topology, batch=batch))

    return BenchmarkSuite(benchmarks=tuple(benchmarks))


def export_benchmark_csv(
    suite: BenchmarkSuite,
    path: str | Path | None = None,
) -> str:
    """Export benchmark runs to CSV and optionally write them to disk."""
    records = suite.to_records()

    if not records:
        raise ValueError("Cannot export a benchmark suite with no runs.")

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    content = buffer.getvalue()

    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="")

    return content


def export_benchmark_json(
    suite: BenchmarkSuite,
    path: str | Path | None = None,
) -> str:
    """Export benchmark metadata and results to JSON."""
    payload = {
        "topology_count": len(suite.benchmarks),
        "run_count": suite.run_count,
        "topologies": [item.topology for item in suite.benchmarks],
        "runs": suite.to_records(),
    }
    content = json.dumps(payload, indent=2)

    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content + "\n", encoding="utf-8")

    return content
