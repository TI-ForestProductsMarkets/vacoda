"""
vacoda.hs_sync.reports

Report builders for HS-sync diagnostics.
"""

from __future__ import annotations

import pandas as pd


def build_year_hs_coverage_report_text(
    count_pivot: pd.DataFrame,
    percentage_pivot: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    """Build a plain-text report for year × HS classification coverage."""
    lines = [
        "HS classification coverage audit",
        "=" * 35,
        "",
        "Count pivot:",
        count_pivot.to_string(),
        "",
        "Percentage pivot:",
        percentage_pivot.to_string(),
        "",
        "Summary:",
        summary.to_string(index=False),
        "",
    ]

    return "\n".join(lines)