"""
vacoda.hs_sync.reports

Report builders for HS-sync diagnostics.
"""

from __future__ import annotations

import pandas as pd


def build_year_hs_coverage_report_text(
    *,
    audit_info: dict | None = None,
    overall_summary: pd.DataFrame | None = None,
    count_pivot: pd.DataFrame | None = None,
    percentage_pivot: pd.DataFrame | None = None,
    summary: pd.DataFrame | None = None,
    reporter_coverage: pd.DataFrame | None = None,
    reporter_preview_rows: int = 20,
) -> str:
    """
    Build a plain-text report for HS classification coverage.

    All table arguments are optional so YAML-disabled products do not break
    report generation.
    """
    audit_info = audit_info or {}

    lines: list[str] = [
        "HS classification coverage audit",
        "=" * 35,
        "",
        "Audit overview:",
        f"  Input directory: {audit_info.get('input_dir', 'n/a')}",
        f"  File pattern: {audit_info.get('pattern', 'n/a')}",
        f"  Files scanned: {audit_info.get('n_files', 'n/a')}",
        f"  Engine: {audit_info.get('engine', 'n/a')}",
        f"  Threads: {audit_info.get('threads', 'n/a')}",
        "",
    ]

    if overall_summary is not None:
        lines.extend(
            [
                "Overall HS-system coverage:",
                overall_summary.to_string(index=False),
                "",
            ]
        )

    if summary is not None:
        lines.extend(
            [
                "Year-level mix summary:",
                summary.to_string(index=False),
                "",
            ]
        )

    if percentage_pivot is not None:
        lines.extend(
            [
                "Percentage pivot:",
                percentage_pivot.to_string(),
                "",
            ]
        )

    if count_pivot is not None:
        lines.extend(
            [
                "Count pivot:",
                count_pivot.to_string(),
                "",
            ]
        )

    if reporter_coverage is not None:
        lines.extend(
            [
                "Reporter × year × HS coverage:",
                f"  Shape: {reporter_coverage.shape[0]:,} rows × {reporter_coverage.shape[1]:,} columns",
                "",
                f"Preview first {reporter_preview_rows} rows:",
                reporter_coverage.head(reporter_preview_rows).to_string(index=False),
                "",
            ]
        )

    return "\n".join(lines)