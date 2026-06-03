"""
scripts/hs_sync/06_inspect_hscodes_requiring_weights.py

Inspect source HS-code pairs that require weights for conversion.

This script reads a source-code-level coverage table, for example:

    target_h5_source_code_coverage.csv

and reports:
    - target HS system inferred from the file
    - number of source HS-code pairs requiring weights
    - distribution by source HS system
    - top source HS-code pairs by trade value
    - concentration of value among the top cases

The script is generic: it does not assume H5 as target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vacoda.config.paths import get_reports_dir
from vacoda.io.reports import write_text_report

DEFAULT_INPUT = Path(
    "/storage/vacoda/data/interim/hs_sync/target_h5_mapping_coverage/"
    "target_h5_source_code_coverage.csv")

DEFAULT_REPORT_NAME = "diagnostic_hs_sync_source_h6_target_h5.md"

def _format_value(value: float) -> str:
    """Format large values compactly for terminal readability."""
    value = float(value)
    abs_value = abs(value)

    if abs_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:.2f}"


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Input file is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect source HS-code pairs that require conversion weights."
    )

    parser.add_argument(
    "--report-name",
    default=DEFAULT_REPORT_NAME,
    help="Report filename to write under the configured reports directory.",
)

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to source-code-level coverage CSV.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top concentration rows to display.",
    )

    args = parser.parse_args()
    report_lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        report_lines.append(text)

    def emit_table(df: pd.DataFrame) -> None:
        markdown = df.to_markdown(index=False)
        emit(markdown)

    df = pd.read_csv(args.input)

    required = {
        "source_hs",
        "source_cmd",
        "target_hs",
        "first_year",
        "last_year",
        "n_target_codes",
        "target_codes",
        "relationship_original",
        "actual_mapping_category",
        "conversion_need",
        "n_rows",
        "value_total",
    }

    _require_columns(df, required)

    needs = df[df["conversion_need"] == "needs_weights"].copy()

    target_hs_values = sorted(df["target_hs"].dropna().astype(str).unique())
    source_hs_values = sorted(needs["source_hs"].dropna().astype(str).unique())

    overall_first_year = int(df["first_year"].min())
    overall_last_year = int(df["last_year"].max())

    if needs.empty:
        emit("=" * 80)
        emit("Needs-weights inspection")
        emit("=" * 80)
        emit(f"Input file: {args.input}")
        emit(f"Target HS system(s): {', '.join(target_hs_values)}")
        emit(f"Coverage period: {overall_first_year}–{overall_last_year}")
        emit()
        emit("No source HS-code pairs require weights.")
        return

    needs_first_year = int(needs["first_year"].min())
    needs_last_year = int(needs["last_year"].max())

    total_value = needs["value_total"].sum()
    total_rows = needs["n_rows"].sum()

    emit("=" * 80)
    emit("Needs-weights inspection")
    emit("=" * 80)
    emit(f"Input file: {args.input}")
    emit(f"Target HS system(s): {', '.join(target_hs_values)}")
    emit(f"Coverage period in full file: {overall_first_year}–{overall_last_year}")
    emit(f"Period covered by needs-weights cases: {needs_first_year}–{needs_last_year}")
    emit(f"Source HS systems requiring weights: {', '.join(source_hs_values)}")
    emit()
    emit(
        "Interpretation: each listed source HS-code pair maps to more than one "
        "target code in the configured target HS system. These cases cannot be "
        "converted deterministically without weights or another split rule."
    )
    emit()

    emit(f"Needs-weights source HS-code pairs: {len(needs):,}")
    emit(f"Rows affected: {int(total_rows):,}")
    emit(f"Value affected: {_format_value(total_value)}")
    emit()

    emit("By source_hs:")
    by_source = (
        needs.groupby("source_hs")
        .agg(
            n_source_hs_code_pairs=("source_cmd", "nunique"),
            n_rows=("n_rows", "sum"),
            value_total=("value_total", "sum"),
            first_year=("first_year", "min"),
            last_year=("last_year", "max"),
        )
        .reset_index()
        .sort_values("value_total", ascending=False)
    )

    by_source["value"] = by_source["value_total"].map(_format_value)
    by_source["value_share_within_needs"] = (
        by_source["value_total"] / total_value * 100
    ).map(lambda x: f"{x:.2f}%")

    by_source_display = by_source[
            [
                "source_hs",
                "n_source_hs_code_pairs",
                "n_rows",
                "value",
                "value_share_within_needs",
                "first_year",
                "last_year",
            ]
        ]
    
    emit_table(by_source_display)

    emit()
    emit(f"Top {len(needs):,} needs-weights cases by value:")

    needs_display = needs.sort_values("value_total", ascending=False).copy()
    needs_display["value"] = needs_display["value_total"].map(_format_value)
    needs_display["value_share_within_needs"] = (
        needs_display["value_total"] / total_value * 100
    ).map(lambda x: f"{x:.2f}%")

    emit(
        needs_display[
            [
                "source_hs",
                "source_cmd",
                "target_hs",
                "first_year",
                "last_year",
                "n_target_codes",
                "target_codes",
                "relationship_original",
                "actual_mapping_category",
                "n_rows",
                "value",
                "value_share_within_needs",
            ]
        ].to_string(index=False)
    )

    emit()
    emit("Concentration:")

    needs_sorted = needs.sort_values("value_total", ascending=False).copy()
    needs_sorted["cum_value"] = needs_sorted["value_total"].cumsum()
    needs_sorted["cum_value_share"] = needs_sorted["cum_value"] / total_value * 100
    needs_sorted["value"] = needs_sorted["value_total"].map(_format_value)

    emit(
        needs_sorted[
            [
                "source_hs",
                "source_cmd",
                "target_hs",
                "n_target_codes",
                "target_codes",
                "value",
                "cum_value_share",
            ]
        ]
        .head(args.top_n)
        .to_string(index=False)
    )

    top1 = needs_sorted.iloc[0]
    top_n_value_share = (
        needs_sorted.head(args.top_n)["value_total"].sum() / total_value * 100
    )

    emit()
    emit("=" * 80)
    emit("Interpretation")
    emit("=" * 80)
    emit(
        f"{len(needs):,} source HS-code pairs require weights to map into "
        f"{', '.join(target_hs_values)}."
    )
    emit(
        f"The largest case is {top1['source_hs']} {top1['source_cmd']} "
        f"→ {top1['target_hs']} targets {top1['target_codes']}, "
        f"accounting for {top1['value_total'] / total_value * 100:.2f}% "
        "of the value requiring weights."
    )
    emit(
        f"The top {min(args.top_n, len(needs))} cases account for "
        f"{top_n_value_share:.2f}% of the value requiring weights."
    )

    outputs = write_text_report(
                            "\n".join(report_lines),
                            report_dir=get_reports_dir(),
                            filename=args.report_name,
                            timestamp=True,
                            write_latest=True,
                            )

    emit()
    emit("Report written:")
    for key, path in outputs.items():
        emit(f"  {key}: {path}")    

if __name__ == "__main__":
    main()