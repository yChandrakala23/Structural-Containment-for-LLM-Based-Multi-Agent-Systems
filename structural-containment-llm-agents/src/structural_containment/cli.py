"""Command-line interface for the structural containment simulator.

Usage examples
--------------
Run a quick smoke-test simulation::

    python -m structural_containment run --nodes 10 --prob 0.3 --budget 2

Export an experiment batch to CSV::

    python -m structural_containment export --output results/batch.csv

Generate a Markdown report::

    python -m structural_containment report --output results/report.md

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _run_command(args: argparse.Namespace) -> None:
    """Execute a containment scenario and print a summary."""
    import networkx as nx

    from structural_containment.experiments.config import ExperimentConfig
    from structural_containment.experiments.metrics import summarize_experiments
    from structural_containment.experiments.runner import run_experiments
    from structural_containment.graph.topology import directed_random_graph

    graph: nx.DiGraph
    if args.graph == "random":
        graph = directed_random_graph(args.nodes, args.prob, seed=args.seed)
    elif args.graph == "chain":
        nodes = [str(i) for i in range(args.nodes)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        graph = nx.DiGraph(edges)
    elif args.graph == "cycle":
        nodes = [str(i) for i in range(args.nodes)]
        edges = [(nodes[i], nodes[(i + 1) % args.nodes]) for i in range(args.nodes)]
        graph = nx.DiGraph(edges)
    else:
        print(f"Unknown graph type: {args.graph!r}", file=sys.stderr)
        sys.exit(1)

    print(f"Graph: {args.graph}, nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")

    node_list = list(graph.nodes)
    initial = tuple(node_list[: min(args.attack, len(node_list))])
    budgets = tuple(range(args.budget + 1))

    config = ExperimentConfig(
        initial_compromised=initial,
        policies=("random", "highest_out_degree"),
        budgets=budgets,
        propagation_probability=args.prob,
        seeds=(args.seed,),
    )

    batch = run_experiments(graph, config)
    summaries = summarize_experiments(batch)

    print(f"\nTotal runs: {batch.run_count}")
    print(f"{'Policy':<25} {'Budget':>6} {'Baseline':>10} {'Contained':>10} {'Abs Δ':>8}")
    print("-" * 65)
    for m in summaries:
        print(
            f"{m.policy:<25} {m.budget:>6} "
            f"{m.mean_baseline_rate * 100:>9.1f}% "
            f"{m.mean_intervention_rate * 100:>9.1f}% "
            f"{m.mean_absolute_reduction * 100:>7.1f}%"
        )


def _export_command(args: argparse.Namespace) -> None:
    """Run a benchmark and export results."""

    from structural_containment.experiments.config import ExperimentConfig
    from structural_containment.experiments.export import export_to_csv, export_to_json
    from structural_containment.experiments.runner import run_experiments
    from structural_containment.graph.topology import directed_random_graph

    graph = directed_random_graph(args.nodes, args.prob, seed=args.seed)
    node_list = list(graph.nodes)
    initial = tuple(node_list[: min(args.attack, len(node_list))])

    config = ExperimentConfig(
        initial_compromised=initial,
        policies=("random", "highest_out_degree"),
        budgets=tuple(range(args.budget + 1)),
        propagation_probability=args.prob,
        seeds=tuple(range(args.seed, args.seed + args.runs)),
    )

    batch = run_experiments(graph, config)
    out_path = Path(args.output)

    if out_path.suffix.lower() == ".json":
        export_to_json(batch, out_path)
    else:
        export_to_csv(batch, out_path)

    print(f"Exported {batch.run_count} runs to {out_path}")


def _report_command(args: argparse.Namespace) -> None:
    """Run a benchmark and generate a report."""

    from structural_containment.experiments.config import ExperimentConfig
    from structural_containment.experiments.metrics import (
        compare_policies,
        summarize_experiments,
    )
    from structural_containment.experiments.runner import run_experiments
    from structural_containment.graph.topology import directed_random_graph
    from structural_containment.reporting.report import (
        generate_markdown_report,
        generate_text_report,
        write_report,
    )

    graph = directed_random_graph(args.nodes, args.prob, seed=args.seed)
    node_list = list(graph.nodes)
    initial = tuple(node_list[: min(args.attack, len(node_list))])

    config = ExperimentConfig(
        initial_compromised=initial,
        policies=("random", "highest_out_degree"),
        budgets=tuple(range(args.budget + 1)),
        propagation_probability=args.prob,
        seeds=tuple(range(args.seed, args.seed + args.runs)),
    )

    batch = run_experiments(graph, config)
    summaries = summarize_experiments(batch)
    comparisons = compare_policies(
        summaries,
        first_policy="highest_out_degree",
        second_policy="random",
    )

    out_path = Path(args.output)
    if out_path.suffix.lower() == ".md":
        content = generate_markdown_report(batch, summaries, comparisons)
    else:
        content = generate_text_report(batch, summaries)

    write_report(content, out_path)
    print(f"Report written to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m structural_containment",
        description="Structural Containment for LLM-Based Multi-Agent Systems",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared graph arguments
    def _add_graph_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--nodes", type=int, default=20, help="Number of graph nodes")
        p.add_argument(
            "--prob", type=float, default=0.2, help="Edge probability / propagation probability"
        )
        p.add_argument("--seed", type=int, default=42, help="Random seed")
        p.add_argument(
            "--attack", type=int, default=1, help="Number of initially compromised agents"
        )
        p.add_argument("--budget", type=int, default=2, help="Maximum containment budget")

    # run subcommand
    run_p = subparsers.add_parser("run", help="Run a simulation and print results")
    _add_graph_args(run_p)
    run_p.add_argument(
        "--graph",
        choices=["random", "chain", "cycle"],
        default="random",
        help="Graph topology",
    )
    run_p.set_defaults(func=_run_command)

    # export subcommand
    export_p = subparsers.add_parser("export", help="Run and export results to CSV/JSON")
    _add_graph_args(export_p)
    export_p.add_argument("--runs", type=int, default=5, help="Number of seed repetitions")
    export_p.add_argument("--output", required=True, help="Output file path (.csv or .json)")
    export_p.set_defaults(func=_export_command)

    # report subcommand
    report_p = subparsers.add_parser("report", help="Run and generate a text/Markdown report")
    _add_graph_args(report_p)
    report_p.add_argument("--runs", type=int, default=5, help="Number of seed repetitions")
    report_p.add_argument("--output", required=True, help="Output file path (.txt or .md)")
    report_p.set_defaults(func=_report_command)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the structural containment CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
