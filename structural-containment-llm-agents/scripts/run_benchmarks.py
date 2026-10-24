
"""Run a reproducible containment benchmark across graph densities."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

# Allow running this script directly from the repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structural_containment.experiments.benchmark import (
    export_benchmark_csv,
    export_benchmark_json,
    run_benchmark_suite,
)
from structural_containment.graph.topology import directed_random_graph


def main() -> None:
    node_count = 30
    densities = {
        "sparse": 0.05,
        "medium": 0.15,
        "dense": 0.30,
    }
    graph_seeds = tuple(range(101, 121))

    graphs: dict[str, nx.DiGraph] = {}
    initial_compromised: dict[str, tuple[str, ...]] = {}

    for density_name, probability in densities.items():
        for graph_seed in graph_seeds:
            name = f"{density_name}_seed_{graph_seed}"

            graph = directed_random_graph(
                node_count=node_count,
                edge_probability=probability,
                seed=graph_seed,
            )

            graph = nx.relabel_nodes(
                graph,
                {node: str(node) for node in graph.nodes},
            )

            graphs[name] = graph
            initial_compromised[name] = ("0", "1")

    suite = run_benchmark_suite(
        graphs=graphs,
        initial_compromised_by_topology=initial_compromised,
        attack_strategy="targeted",
        policies=("random", "highest_out_degree"),
        budgets=(0, 1, 2, 4),
        propagation_probability=0.2,
        max_steps=30,
        seeds=(11, 22, 33, 44, 55, 66, 77, 88, 99, 110),
    )

    output_dir = ROOT / "results"
    csv_path = output_dir / "benchmark.csv"
    json_path = output_dir / "benchmark.json"

    export_benchmark_csv(suite, csv_path)
    export_benchmark_json(suite, json_path)

    print(f"Topologies evaluated: {len(suite.benchmarks)}")
    print(f"Experiment runs: {suite.run_count}")
    print(f"CSV results: {csv_path}")
    print(f"JSON results: {json_path}")


if __name__ == "__main__":
    main()
