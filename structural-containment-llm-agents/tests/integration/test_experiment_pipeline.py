"""Integration tests for the full experiment pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import networkx as nx
import pytest

from structural_containment.experiments.config import ExperimentConfig
from structural_containment.experiments.export import (
    batch_to_records,
    export_to_csv,
    export_to_json,
)
from structural_containment.experiments.metrics import (
    compare_policies,
    summarize_experiments,
)
from structural_containment.experiments.runner import run_experiments
from structural_containment.experiments.seeds import (
    derive_seed,
    latin_hypercube_seeds,
    make_seed_sequence,
)
from structural_containment.reporting.report import (
    generate_markdown_report,
    generate_text_report,
    write_report,
)


@pytest.fixture
def medium_graph():
    """A 10-node random-style directed graph for integration tests."""
    return nx.DiGraph(
        [
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
            ("a", "c"),
            ("b", "d"),
            ("c", "e"),
            ("e", "f"),
            ("f", "g"),
            ("g", "h"),
            ("h", "i"),
            ("i", "j"),
            ("a", "f"),
            ("b", "g"),
            ("d", "h"),
        ]
    )


@pytest.fixture
def config(medium_graph):
    return ExperimentConfig(
        initial_compromised=("a",),
        policies=("random", "highest_out_degree"),
        budgets=(0, 1, 2),
        propagation_probability=0.5,
        seeds=(11, 22, 33),
    )


@pytest.fixture
def batch(medium_graph, config):
    return run_experiments(medium_graph, config)


# ── Experiment execution ──────────────────────────────────────────────────────


def test_batch_run_count_matches_combinations(batch):
    # 2 policies × 3 budgets × 3 seeds = 18 runs
    assert batch.run_count == 18


def test_all_runs_have_valid_rates(batch):
    for run in batch.runs:
        assert 0.0 <= run.result.baseline_compromise_rate <= 1.0
        assert 0.0 <= run.result.intervention_compromise_rate <= 1.0


def test_baseline_rate_is_consistent_across_same_seed(medium_graph, config):
    """Baseline should be the same for all policies/budgets with the same seed."""
    batch = run_experiments(medium_graph, config)
    # Group by seed
    by_seed: dict[int, set[float]] = {}
    for run in batch.runs:
        by_seed.setdefault(run.seed, set()).add(round(run.result.baseline_compromise_rate, 10))
    # Each seed should produce exactly one unique baseline rate.
    for seed, rates in by_seed.items():
        assert len(rates) == 1, f"Inconsistent baseline for seed {seed}: {rates}"


def test_intervention_rate_lte_baseline_with_full_prob(medium_graph):
    """With propagation_probability=1.0, containment should never increase spread."""
    config = ExperimentConfig(
        initial_compromised=("a",),
        policies=("highest_out_degree",),
        budgets=(1, 2),
        propagation_probability=1.0,
        seeds=(42,),
    )
    batch = run_experiments(medium_graph, config)
    for run in batch.runs:
        assert run.result.intervention_compromise_rate <= run.result.baseline_compromise_rate


# ── Metrics ───────────────────────────────────────────────────────────────────


def test_summary_groups_all_combinations(batch):
    summaries = summarize_experiments(batch)
    # 2 policies × 3 budgets = 6 groups
    assert len(summaries) == 6


def test_policy_comparison_returns_comparisons(batch):
    summaries = summarize_experiments(batch)
    comparisons = compare_policies(
        summaries,
        first_policy="highest_out_degree",
        second_policy="random",
    )
    # One comparison per budget where both policies appear.
    assert len(comparisons) == 3


def test_summary_run_counts_match(batch):
    summaries = summarize_experiments(batch)
    for m in summaries:
        assert m.run_count == 3


# ── Export ────────────────────────────────────────────────────────────────────


def test_batch_to_records_has_correct_keys(batch):
    records = batch_to_records(batch)
    assert len(records) == batch.run_count
    expected_keys = {
        "attack_strategy",
        "policy",
        "budget",
        "seed",
        "initial_compromised",
        "contained_agents",
        "baseline_compromise_rate",
        "intervention_compromise_rate",
        "absolute_reduction",
        "relative_reduction",
        "baseline_steps_executed",
        "intervention_steps_executed",
        "baseline_termination",
        "intervention_termination",
    }
    for record in records:
        assert set(record.keys()) == expected_keys


def test_export_to_csv_produces_valid_csv(batch):
    csv_text = export_to_csv(batch)
    lines = csv_text.strip().split("\n")
    # Header + one row per run
    assert len(lines) == batch.run_count + 1
    header = lines[0].split(",")
    assert "policy" in header
    assert "budget" in header


def test_export_to_csv_writes_file(batch):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.csv"
        csv_text = export_to_csv(batch, path)
        assert path.exists()
        assert path.read_text(encoding="utf-8") == csv_text


def test_export_to_json_produces_valid_json(batch):
    json_text = export_to_json(batch)
    data = json.loads(json_text)
    assert data["run_count"] == batch.run_count
    assert len(data["runs"]) == batch.run_count


def test_export_to_json_writes_file(batch):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.json"
        export_to_json(batch, path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["run_count"] == batch.run_count


def test_export_to_csv_wrong_type_rejected():
    with pytest.raises(TypeError, match="ExperimentBatch"):
        export_to_csv(None)


def test_export_to_json_wrong_type_rejected():
    with pytest.raises(TypeError, match="ExperimentBatch"):
        export_to_json("bad")


# ── Reporting ─────────────────────────────────────────────────────────────────


def test_text_report_contains_policy_names(batch):
    summaries = summarize_experiments(batch)
    report = generate_text_report(batch, summaries)
    assert "random" in report
    assert "highest_out_degree" in report


def test_text_report_contains_run_count(batch):
    summaries = summarize_experiments(batch)
    report = generate_text_report(batch, summaries)
    assert str(batch.run_count) in report


def test_markdown_report_contains_table(batch):
    summaries = summarize_experiments(batch)
    report = generate_markdown_report(batch, summaries)
    assert "| `random`" in report or "| random" in report
    assert "|" in report


def test_markdown_report_with_comparisons(batch):
    summaries = summarize_experiments(batch)
    comparisons = compare_policies(
        summaries,
        first_policy="highest_out_degree",
        second_policy="random",
    )
    report = generate_markdown_report(batch, summaries, comparisons)
    assert "Policy Comparisons" in report


def test_write_report_creates_file(batch):
    summaries = summarize_experiments(batch)
    content = generate_text_report(batch, summaries)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sub" / "report.txt"
        write_report(content, path)
        assert path.exists()
        assert path.read_text(encoding="utf-8") == content


def test_write_report_wrong_type_rejected(batch):
    with pytest.raises(TypeError, match="string"):
        write_report(42, "report.txt")


def test_text_report_wrong_batch_type_rejected():
    from structural_containment.experiments.metrics import PolicyMetrics

    summaries = (PolicyMetrics("targeted", "random", 1, 1, 0.5, 0.4, 0.1, 0.2, 0.0),)
    with pytest.raises(TypeError, match="ExperimentBatch"):
        generate_text_report(None, summaries)


# ── Seeds ─────────────────────────────────────────────────────────────────────


def test_make_seed_sequence_length(batch):
    seeds = make_seed_sequence(100, 5)
    assert len(seeds) == 5


def test_make_seed_sequence_values():
    seeds = make_seed_sequence(10, 3, offset=2)
    assert seeds == (12, 13, 14)


def test_make_seed_sequence_zero_count():
    assert make_seed_sequence(42, 0) == ()


def test_derive_seed_is_deterministic():
    s1 = derive_seed(42, "policy", "random", 1)
    s2 = derive_seed(42, "policy", "random", 1)
    assert s1 == s2


def test_derive_seed_differs_by_label():
    s1 = derive_seed(42, "random")
    s2 = derive_seed(42, "highest_out_degree")
    assert s1 != s2


def test_derive_seed_is_non_negative():
    assert derive_seed(0, "x") >= 0


def test_latin_hypercube_seeds_length():
    seeds = latin_hypercube_seeds(10)
    assert len(seeds) == 10


def test_latin_hypercube_seeds_are_distinct():
    seeds = latin_hypercube_seeds(20, base_seed=7)
    assert len(set(seeds)) == 20


def test_latin_hypercube_seeds_reproducible():
    assert latin_hypercube_seeds(5, base_seed=1) == latin_hypercube_seeds(5, base_seed=1)


def test_latin_hypercube_seeds_zero_count_rejected():
    with pytest.raises(ValueError, match="positive"):
        latin_hypercube_seeds(0)
