"""
vacoda.hs_sync.reports

Report builders for HS-sync diagnostics.
"""

from __future__ import annotations

import pandas as pd


def build_year_hs_coverage_report_text(
    *,
    audit_info: dict,
    overall_summary: pd.DataFrame,
    count_pivot: pd.DataFrame,
    percentage_pivot: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    """Build a plain-text report for HS classification coverage."""
    lines = [
        "HS classification coverage audit",
        "=" * 35,
        "",
        "Audit overview:",
        f"  Input directory: {audit_info.get('input_dir')}",
        f"  File pattern: {audit_info.get('pattern')}",
        f"  Files scanned: {audit_info.get('n_files'):,}",
        f"  Engine: {audit_info.get('engine')}",
        f"  Threads: {audit_info.get('threads')}",
        "",
        "Overall HS-system coverage:",
        overall_summary.to_string(index=False),
        "",
        "Year-level mix summary:",
        summary.to_string(index=False),
        "",
        "Percentage pivot:",
        percentage_pivot.to_string(),
        "",
        "Count pivot:",
        count_pivot.to_string(),
        "",
    ]

    return "\n".join(lines)