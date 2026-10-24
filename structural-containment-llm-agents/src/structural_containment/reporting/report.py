"""Text and Markdown report generation for experiment results."""

from __future__ import annotations

from pathlib import Path

from structural_containment.experiments.metrics import PolicyComparison, PolicyMetrics
from structural_containment.experiments.runner import ExperimentBatch


def _fmt_pct(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.1f}%"


def _fmt_f(value: float | None, default: str = "N/A") -> str:
    """Format an optional float to 4 decimal places."""
    if value is None:
        return default
    return f"{value:.4f}"


def generate_text_report(
    batch: ExperimentBatch,
    summaries: tuple[PolicyMetrics, ...],
    *,
    title: str = "Structural Containment — Experiment Report",
) -> str:
    """Generate a plain-text summary report of experiment results.

    Args:
        batch: Completed experiment batch.
        summaries: Aggregated policy metrics from
            :func:`~structural_containment.experiments.metrics.summarize_experiments`.
        title: Report title.

    Returns:
        A multi-line string containing the formatted report.

    Raises:
        TypeError: If arguments have unsupported types.
    """
    if not isinstance(batch, ExperimentBatch):
        raise TypeError("batch must be an ExperimentBatch.")
    if not isinstance(summaries, tuple) or any(not isinstance(s, PolicyMetrics) for s in summaries):
        raise TypeError("summaries must be a tuple of PolicyMetrics.")

    lines: list[str] = []
    sep = "=" * 72

    lines.append(sep)
    lines.append(title.center(72))
    lines.append(sep)
    lines.append("")
    lines.append(f"Total runs : {batch.run_count}")
    lines.append(f"Policies   : {sorted({r.policy for r in batch.runs})}")
    lines.append(f"Budgets    : {sorted({r.budget for r in batch.runs})}")
    lines.append(f"Seeds      : {sorted({r.seed for r in batch.runs})}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("POLICY SUMMARY")
    lines.append("-" * 72)

    col_w = [20, 8, 6, 12, 12, 12, 12]
    header = (
        f"{'Policy':<{col_w[0]}} {'Budget':>{col_w[1]}} {'Runs':>{col_w[2]}} "
        f"{'Baseline':>{col_w[3]}} {'Containment':>{col_w[4]}} "
        f"{'Abs Δ':>{col_w[5]}} {'Rel Δ':>{col_w[6]}}"
    )
    lines.append(header)
    lines.append("-" * 72)

    for m in summaries:
        row = (
            f"{m.policy:<{col_w[0]}} {m.budget:>{col_w[1]}} {m.run_count:>{col_w[2]}} "
            f"{_fmt_pct(m.mean_baseline_rate):>{col_w[3]}} "
            f"{_fmt_pct(m.mean_intervention_rate):>{col_w[4]}} "
            f"{_fmt_pct(m.mean_absolute_reduction):>{col_w[5]}} "
            f"{_fmt_pct(m.mean_relative_reduction) if m.mean_relative_reduction is not None else 'N/A':>{col_w[6]}}"
        )
        lines.append(row)

    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


def generate_markdown_report(
    batch: ExperimentBatch,
    summaries: tuple[PolicyMetrics, ...],
    comparisons: tuple[PolicyComparison, ...] = (),
    *,
    title: str = "Structural Containment — Experiment Report",
) -> str:
    """Generate a GitHub-flavoured Markdown report of experiment results.

    Args:
        batch: Completed experiment batch.
        summaries: Aggregated policy metrics.
        comparisons: Optional policy comparisons from
            :func:`~structural_containment.experiments.metrics.compare_policies`.
        title: Report heading.

    Returns:
        A Markdown string suitable for saving as ``.md`` or displaying in a
        notebook.

    Raises:
        TypeError: If arguments have unsupported types.
    """
    if not isinstance(batch, ExperimentBatch):
        raise TypeError("batch must be an ExperimentBatch.")
    if not isinstance(summaries, tuple) or any(not isinstance(s, PolicyMetrics) for s in summaries):
        raise TypeError("summaries must be a tuple of PolicyMetrics.")

    lines: list[str] = []

    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total runs | {batch.run_count} |")
    lines.append(f"| Policies | {', '.join(sorted({r.policy for r in batch.runs}))} |")
    lines.append(
        f"| Budgets | {', '.join(str(b) for b in sorted({r.budget for r in batch.runs}))} |"
    )
    lines.append(f"| Seeds | {', '.join(str(s) for s in sorted({r.seed for r in batch.runs}))} |")
    lines.append("")

    lines.append("## Policy Summary")
    lines.append("")
    lines.append("| Policy | Budget | Runs | Baseline | Containment | Abs Δ | Rel Δ | σ(Δ) |")
    lines.append("|--------|-------:|-----:|---------:|------------:|------:|------:|-----:|")

    for m in summaries:
        rel = (
            _fmt_pct(m.mean_relative_reduction) if m.mean_relative_reduction is not None else "N/A"
        )
        lines.append(
            f"| `{m.policy}` | {m.budget} | {m.run_count} "
            f"| {_fmt_pct(m.mean_baseline_rate)} "
            f"| {_fmt_pct(m.mean_intervention_rate)} "
            f"| {_fmt_pct(m.mean_absolute_reduction)} "
            f"| {rel} "
            f"| {_fmt_f(m.standard_deviation_reduction)} |"
        )

    lines.append("")

    if comparisons:
        lines.append("## Policy Comparisons")
        lines.append("")
        lines.append("| Attack | Budget | Policy A | Policy B | Δ(A − B) |")
        lines.append("|--------|-------:|----------|----------|----------:|")
        for c in comparisons:
            lines.append(
                f"| {c.attack_strategy} | {c.budget} "
                f"| `{c.first_policy}` | `{c.second_policy}` "
                f"| {_fmt_pct(c.difference_in_mean_reduction)} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_report(
    content: str,
    path: str | Path,
) -> None:
    """Write a report string to a file.

    Args:
        content: Report content to write.
        path: Destination file path.

    Raises:
        TypeError: If ``content`` is not a string.
        OSError: If the file cannot be written.
    """
    if not isinstance(content, str):
        raise TypeError("content must be a string.")

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
