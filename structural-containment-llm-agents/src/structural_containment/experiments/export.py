"""Export experiment results to CSV and JSON formats."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from structural_containment.experiments.runner import ExperimentBatch, ExperimentRun


def _run_to_dict(run: ExperimentRun) -> dict[str, Any]:
    """Flatten one :class:`ExperimentRun` to a plain dictionary.

    Nested :class:`~structural_containment.simulation.scenarios.ScenarioResult`
    fields are flattened with dotted-key names suitable for tabular formats.
    """
    return {
        "attack_strategy": run.attack_strategy,
        "policy": run.policy,
        "budget": run.budget,
        "seed": run.seed,
        "initial_compromised": "|".join(run.initial_compromised),
        "contained_agents": "|".join(run.result.contained_agents),
        "baseline_compromise_rate": run.result.baseline_compromise_rate,
        "intervention_compromise_rate": run.result.intervention_compromise_rate,
        "absolute_reduction": run.result.absolute_reduction,
        "relative_reduction": (
            run.result.relative_reduction if run.result.relative_reduction is not None else ""
        ),
        "baseline_steps_executed": run.result.baseline.steps_executed,
        "intervention_steps_executed": run.result.intervention.steps_executed,
        "baseline_termination": str(run.result.baseline.termination_reason),
        "intervention_termination": str(run.result.intervention.termination_reason),
    }


def batch_to_records(batch: ExperimentBatch) -> list[dict[str, Any]]:
    """Convert an :class:`ExperimentBatch` to a list of flat dictionaries.

    Each dictionary represents one :class:`ExperimentRun` with all nested
    fields flattened for easy tabular display or export.

    Args:
        batch: Completed experiment batch.

    Returns:
        List of flat dictionaries, one per run.

    Raises:
        TypeError: If ``batch`` is not an ``ExperimentBatch``.
    """
    if not isinstance(batch, ExperimentBatch):
        raise TypeError("batch must be an ExperimentBatch.")

    return [_run_to_dict(run) for run in batch.runs]


def export_to_csv(
    batch: ExperimentBatch,
    path: str | Path | None = None,
) -> str:
    """Export experiment results to CSV.

    Args:
        batch: Completed experiment batch.
        path: File path to write.  If ``None``, the CSV text is returned as a
            string without writing to disk.

    Returns:
        The CSV content as a string.

    Raises:
        TypeError: If ``batch`` is not an ``ExperimentBatch``.
        OSError: If ``path`` cannot be written.
    """
    if not isinstance(batch, ExperimentBatch):
        raise TypeError("batch must be an ExperimentBatch.")

    records = batch_to_records(batch)

    if not records:
        fieldnames: list[str] = []
    else:
        fieldnames = list(records[0].keys())

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    csv_text = buf.getvalue()

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(csv_text, encoding="utf-8")

    return csv_text


def export_to_json(
    batch: ExperimentBatch,
    path: str | Path | None = None,
    *,
    indent: int = 2,
) -> str:
    """Export experiment results to JSON.

    The JSON contains a top-level ``"runs"`` array of flat record objects.

    Args:
        batch: Completed experiment batch.
        path: File path to write.  If ``None``, the JSON string is returned
            without writing to disk.
        indent: JSON indentation level.

    Returns:
        The JSON content as a string.

    Raises:
        TypeError: If ``batch`` is not an ``ExperimentBatch``.
        OSError: If ``path`` cannot be written.
    """
    if not isinstance(batch, ExperimentBatch):
        raise TypeError("batch must be an ExperimentBatch.")

    records = batch_to_records(batch)
    json_text = json.dumps({"run_count": len(records), "runs": records}, indent=indent)

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json_text, encoding="utf-8")

    return json_text


def export_to_dataframe(batch: ExperimentBatch) -> Any:
    """Convert an :class:`ExperimentBatch` to a :class:`pandas.DataFrame`.

    Requires pandas to be installed.

    Args:
        batch: Completed experiment batch.

    Returns:
        A ``pandas.DataFrame`` with one row per run.

    Raises:
        TypeError: If ``batch`` is not an ``ExperimentBatch``.
        ImportError: If pandas is not installed.
    """
    if not isinstance(batch, ExperimentBatch):
        raise TypeError("batch must be an ExperimentBatch.")

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for export_to_dataframe. Install it with: pip install pandas"
        ) from exc

    records = batch_to_records(batch)
    return pd.DataFrame(records)
