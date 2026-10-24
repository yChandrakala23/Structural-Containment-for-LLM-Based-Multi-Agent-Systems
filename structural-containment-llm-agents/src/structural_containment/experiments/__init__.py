"""Experiment configuration, execution, metrics, and export.

Submodules
----------
- :mod:`~structural_containment.experiments.config` — :class:`ExperimentConfig`
- :mod:`~structural_containment.experiments.runner` — :func:`run_experiments`,
  :class:`ExperimentBatch`, :class:`ExperimentRun`
- :mod:`~structural_containment.experiments.metrics` — :func:`summarize_experiments`,
  :func:`compare_policies`, :class:`PolicyMetrics`, :class:`PolicyComparison`
- :mod:`~structural_containment.experiments.export` — CSV / JSON / DataFrame export
- :mod:`~structural_containment.experiments.seeds` — Seed generation utilities
"""

from structural_containment.experiments.config import ExperimentConfig
from structural_containment.experiments.export import (
    batch_to_records,
    export_to_csv,
    export_to_json,
)
from structural_containment.experiments.metrics import (
    PolicyComparison,
    PolicyMetrics,
    compare_policies,
    summarize_experiments,
)
from structural_containment.experiments.runner import (
    ExperimentBatch,
    ExperimentRun,
    run_experiments,
)
from structural_containment.experiments.seeds import (
    derive_seed,
    latin_hypercube_seeds,
    make_seed_sequence,
)

__all__ = [
    "ExperimentBatch",
    "ExperimentConfig",
    "ExperimentRun",
    "PolicyComparison",
    "PolicyMetrics",
    "batch_to_records",
    "compare_policies",
    "derive_seed",
    "export_to_csv",
    "export_to_json",
    "latin_hypercube_seeds",
    "make_seed_sequence",
    "run_experiments",
    "summarize_experiments",
]
