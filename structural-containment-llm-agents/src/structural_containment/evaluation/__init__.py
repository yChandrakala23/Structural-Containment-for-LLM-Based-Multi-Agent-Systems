"""Evaluation metrics for structural containment experiments.

Submodules
----------
- :mod:`~structural_containment.evaluation.performance_metrics` — Timing and
  spread-rate metrics derived from individual simulation runs.
- :mod:`~structural_containment.evaluation.security_metrics` — Coverage and
  isolation quality metrics.
- :mod:`~structural_containment.evaluation.statistics` — Bootstrap confidence
  intervals, Cohen's d, and Welch's t-statistic.
"""

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

__all__ = [
    "bootstrap_confidence_interval",
    "cohen_d",
    "containment_coverage",
    "containment_overhead",
    "final_compromise_count",
    "isolation_quality",
    "mean_newly_compromised_per_step",
    "reachability_reduction",
    "steps_to_stable",
    "time_to_first_compromise",
    "welch_t_statistic",
]
