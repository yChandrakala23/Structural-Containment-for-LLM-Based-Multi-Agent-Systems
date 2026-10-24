"""Report generation for structural containment experiment results.

Provides plain-text and Markdown report generators.
"""

from structural_containment.reporting.report import (
    generate_markdown_report,
    generate_text_report,
    write_report,
)

__all__ = [
    "generate_markdown_report",
    "generate_text_report",
    "write_report",
]
