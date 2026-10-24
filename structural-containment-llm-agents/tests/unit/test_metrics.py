"""Tests for evaluation metrics (performance, security, statistics)."""

from __future__ import annotations

import networkx as nx
import pytest

from structural_containment.evaluation.performance_metrics import (
    containment_overhead,
    final_compromise_count,
    mean_newly_compromised_per_step,
    steps_to_stable,
    time_to_first_compromise,
)
from structural_containment.evaluation.security_metrics import (
    containment_coverage,
    isolation_quality,
    reachability_reduction,
)
from structural_containment.evaluation.statistics import (
    bootstrap_confidence_interval,
    cohen_d,
    welch_t_statistic,
)
from structural_containment.simulation.propagation import simulate_propagation

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def chain():
    return nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d")])


@pytest.fixture
def full_spread_result(chain):
    """Simulation with probability=1 — all agents compromised."""
    return simulate_propagation(chain, ["a"], propagation_probability=1.0)


@pytest.fixture
def no_spread_result(chain):
    """Simulation with probability=0 — only the initial agent compromised."""
    return simulate_propagation(chain, ["a"], propagation_probability=0.0)


@pytest.fixture
def contained_result(chain):
    """Simulation with 'b' contained — only 'a' compromised."""
    return simulate_propagation(chain, ["a"], initial_contained=["b"], propagation_probability=1.0)


# ── Performance metrics ───────────────────────────────────────────────────────


class TestTimeToFirstCompromise:
    def test_full_spread_returns_step_one(self, full_spread_result):
        assert time_to_first_compromise(full_spread_result) == 1

    def test_no_spread_returns_none(self, no_spread_result):
        assert time_to_first_compromise(no_spread_result) is None

    def test_wrong_type_is_rejected(self):
        with pytest.raises(TypeError, match="SimulationResult"):
            time_to_first_compromise(None)


class TestStepsToStable:
    def test_full_chain_takes_correct_steps(self, full_spread_result):
        # chain a→b→c→d with prob=1 takes 3 steps.
        assert steps_to_stable(full_spread_result) == 3

    def test_no_spread_terminates_quickly(self, no_spread_result):
        assert steps_to_stable(no_spread_result) == 1

    def test_wrong_type_is_rejected(self):
        with pytest.raises(TypeError, match="SimulationResult"):
            steps_to_stable("bad")


class TestMeanNewlyCompromisedPerStep:
    def test_full_spread_returns_positive_rate(self, full_spread_result):
        rate = mean_newly_compromised_per_step(full_spread_result)
        assert rate > 0.0

    def test_no_spread_returns_zero(self, no_spread_result):
        assert mean_newly_compromised_per_step(no_spread_result) == 0.0

    def test_wrong_type_is_rejected(self):
        with pytest.raises(TypeError, match="SimulationResult"):
            mean_newly_compromised_per_step(42)


class TestFinalCompromiseCount:
    def test_full_spread_counts_all(self, full_spread_result):
        assert final_compromise_count(full_spread_result) == 4

    def test_no_spread_counts_one(self, no_spread_result):
        assert final_compromise_count(no_spread_result) == 1

    def test_wrong_type_is_rejected(self):
        with pytest.raises(TypeError, match="SimulationResult"):
            final_compromise_count([])


class TestContainmentOverhead:
    def test_contained_result_shows_one_overhead(self, contained_result):
        assert containment_overhead(contained_result) == 1

    def test_no_containment_shows_zero_overhead(self, full_spread_result):
        assert containment_overhead(full_spread_result) == 0

    def test_wrong_type_is_rejected(self):
        with pytest.raises(TypeError, match="SimulationResult"):
            containment_overhead(None)


# ── Security metrics ──────────────────────────────────────────────────────────


class TestContainmentCoverage:
    def test_no_compromise_gives_full_coverage(self, no_spread_result):
        # Only "a" compromised in a 4-node chain.
        coverage = containment_coverage(no_spread_result, total_agents=4)
        assert coverage == pytest.approx(0.75)

    def test_full_compromise_gives_zero_coverage(self, full_spread_result):
        coverage = containment_coverage(full_spread_result, total_agents=4)
        assert coverage == pytest.approx(0.0)

    def test_empty_network_returns_one(self, no_spread_result):
        assert containment_coverage(no_spread_result, total_agents=0) == 1.0

    def test_wrong_result_type_is_rejected(self):
        with pytest.raises(TypeError, match="SimulationResult"):
            containment_coverage(None, total_agents=4)

    def test_negative_total_agents_is_rejected(self, no_spread_result):
        with pytest.raises(ValueError, match="non-negative"):
            containment_coverage(no_spread_result, total_agents=-1)

    def test_boolean_total_agents_is_rejected(self, no_spread_result):
        with pytest.raises(TypeError, match="integer"):
            containment_coverage(no_spread_result, total_agents=True)


class TestIsolationQuality:
    def test_perfect_isolation(self):
        quality = isolation_quality(["b", "c"], frozenset(), total_agents=4)
        assert quality == 1.0

    def test_zero_quality_when_all_contained_are_compromised(self):
        quality = isolation_quality(["b"], frozenset({"b"}), total_agents=4)
        assert quality == 0.0

    def test_partial_quality(self):
        quality = isolation_quality(["b", "c"], frozenset({"b"}), total_agents=4)
        assert quality == pytest.approx(0.5)

    def test_no_containment_returns_one(self):
        assert isolation_quality([], frozenset({"a"}), total_agents=4) == 1.0

    def test_string_contained_agents_rejected(self):
        with pytest.raises(TypeError, match="collection"):
            isolation_quality("a", frozenset(), total_agents=4)

    def test_negative_total_agents_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            isolation_quality([], frozenset(), total_agents=-1)


class TestReachabilityReduction:
    def test_chain_reduction_is_positive(self, chain):
        reduction = reachability_reduction(chain, {"b"})
        assert 0.0 <= reduction <= 1.0

    def test_empty_contained_set_returns_zero(self, chain):
        # No isolation → no reduction.
        assert reachability_reduction(chain, set()) == pytest.approx(0.0)

    def test_trivial_graph_returns_zero(self):
        graph = nx.DiGraph()
        graph.add_node("a")
        assert reachability_reduction(graph, set()) == 0.0

    def test_unknown_agent_is_rejected(self, chain):
        with pytest.raises(ValueError, match="Unknown agents"):
            reachability_reduction(chain, {"missing"})

    def test_undirected_graph_is_rejected(self):
        with pytest.raises(ValueError, match="directed"):
            reachability_reduction(nx.Graph(), set())


# ── Statistical helpers ───────────────────────────────────────────────────────


class TestBootstrapCI:
    def test_interval_contains_sample_mean(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = bootstrap_confidence_interval(data, n_resamples=500, seed=1)
        mean = sum(data) / len(data)
        assert lo <= mean <= hi

    def test_lower_bound_lte_upper_bound(self):
        data = [0.1, 0.2, 0.3, 0.4]
        lo, hi = bootstrap_confidence_interval(data, seed=7)
        assert lo <= hi

    def test_median_statistic(self):
        data = [1.0, 2.0, 3.0, 100.0]
        lo, hi = bootstrap_confidence_interval(data, statistic="median", n_resamples=500, seed=0)
        assert lo <= hi

    def test_reproducible_with_same_seed(self):
        data = [0.5, 0.6, 0.7, 0.8]
        assert bootstrap_confidence_interval(data, seed=99) == bootstrap_confidence_interval(
            data, seed=99
        )

    def test_empty_data_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_confidence_interval([])

    def test_invalid_confidence_is_rejected(self):
        with pytest.raises(ValueError, match="strictly between"):
            bootstrap_confidence_interval([1.0], confidence=1.0)

    def test_unsupported_statistic_is_rejected(self):
        with pytest.raises(ValueError, match="statistic"):
            bootstrap_confidence_interval([1.0], statistic="mode")

    def test_invalid_n_resamples_is_rejected(self):
        with pytest.raises(ValueError, match="n_resamples"):
            bootstrap_confidence_interval([1.0], n_resamples=0)

    def test_string_data_is_rejected(self):
        with pytest.raises(TypeError):
            bootstrap_confidence_interval("abc")


class TestCohenD:
    def test_identical_groups_return_zero(self):
        assert cohen_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_separated_groups_return_large_effect(self):
        a = [10.0] * 10
        b = [0.0] * 10
        d = cohen_d(a, b)
        assert d > 1.0

    def test_sign_is_positive_when_a_larger(self):
        assert cohen_d([2.0, 3.0], [0.0, 1.0]) > 0

    def test_sign_is_negative_when_b_larger(self):
        assert cohen_d([0.0, 1.0], [2.0, 3.0]) < 0

    def test_empty_group_a_is_rejected(self):
        with pytest.raises(ValueError, match="group_a"):
            cohen_d([], [1.0])

    def test_empty_group_b_is_rejected(self):
        with pytest.raises(ValueError, match="group_b"):
            cohen_d([1.0], [])

    def test_string_group_is_rejected(self):
        with pytest.raises(TypeError):
            cohen_d("abc", [1.0])


class TestWelchT:
    def test_identical_groups_return_zero(self):
        assert welch_t_statistic([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_sign_reflects_direction(self):
        t = welch_t_statistic([5.0, 6.0, 7.0], [1.0, 2.0, 3.0])
        assert t > 0

    def test_small_group_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            welch_t_statistic([1.0], [2.0, 3.0])

    def test_small_group_b_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            welch_t_statistic([1.0, 2.0], [3.0])

    def test_string_group_is_rejected(self):
        with pytest.raises(TypeError):
            welch_t_statistic("abc", [1.0, 2.0])
