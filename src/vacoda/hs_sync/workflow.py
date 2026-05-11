"""
vacoda.hs_sync.workflow

End-to-end HS-sync diagnostic workflows.
"""

from __future__ import annotations

from pathlib import Path

from vacoda.config.paths import get_data_dir, get_output_dir
from vacoda.hs_sync.coverage import (
    build_year_hs_counts,
    build_year_hs_mix_summary,
    iter_csv_files,
    make_year_hs_count_pivot,
    make_year_hs_percentage_pivot,
    basic_coverage
)
from vacoda.hs_sync.reports import build_year_hs_coverage_report_text
from vacoda.io.writer import write_csv, write_parquet
from vacoda.utils.logger import get_logger, rel_path


logger = get_logger("HS_SYNC.WORKFLOW")

REPORT_FILE = "audit_hs_year_coverage_report.txt"


def run_year_hs_coverage_audit(
    *,
    input_subdir: str = "api/single",
    output_subdir: str = "hs_sync",
    pattern: str = "*.csv",
    overwrite: bool = True,
) -> dict[str, Path]:
    """
    Run year × HS classification coverage audit.

    Reads raw Comtrade CSV files, builds coverage tables, saves outputs,
    and writes a plain-text report.

    Returns
    -------
    dict[str, Path]
        Mapping of output names to saved paths.
    """

    ## Setup input and output paths
    data_dir = get_data_dir("raw") / input_subdir
    out_dir = get_data_dir("interim") / output_subdir
    report_dir = get_output_dir("reports")

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning input files in: {rel_path(data_dir)}")
    files = iter_csv_files(data_dir, pattern=pattern)
    logger.info(f"Found {len(files):,} file(s).")

    ## 1. Basic coverage diagnostics    
    classification_counter, year_range, error_count = basic_coverage()

    ## Save basic coverage diagnostics to output file
    report_path = report_dir / REPORT_FILE

    ## 2. Year × HS classification coverage analysis
    year_hs_counts = build_year_hs_counts(files)
    count_pivot = make_year_hs_count_pivot(year_hs_counts)
    percentage_pivot = make_year_hs_percentage_pivot(count_pivot)
    summary = build_year_hs_mix_summary(count_pivot)

    ## Save outputs and report
    outputs: dict[str, Path] = {}

    outputs["year_hs_counts_long_csv"] = write_csv(
        year_hs_counts,
        out_dir / "year_hs_counts_long.csv",
        overwrite=overwrite,
    )
    outputs["year_hs_counts_pivot_csv"] = write_csv(
        count_pivot.reset_index(),
        out_dir / "year_hs_counts_pivot.csv",
        overwrite=overwrite,
    )
    outputs["year_hs_percentage_pivot_csv"] = write_csv(
        percentage_pivot.reset_index(),
        out_dir / "year_hs_percentage_pivot.csv",
        overwrite=overwrite,
    )
    outputs["year_hs_mix_summary_csv"] = write_csv(
        summary,
        out_dir / "year_hs_mix_summary.csv",
        overwrite=overwrite,
    )

    outputs["year_hs_counts_long_parquet"] = write_parquet(
        year_hs_counts,
        out_dir / "year_hs_counts_long.parquet",
        overwrite=overwrite,
    )
    outputs["year_hs_counts_pivot_parquet"] = write_parquet(
        count_pivot.reset_index(),
        out_dir / "year_hs_counts_pivot.parquet",
        overwrite=overwrite,
    )
    outputs["year_hs_percentage_pivot_parquet"] = write_parquet(
        percentage_pivot.reset_index(),
        out_dir / "year_hs_percentage_pivot.parquet",
        overwrite=overwrite,
    )
    outputs["year_hs_mix_summary_parquet"] = write_parquet(
        summary,
        out_dir / "year_hs_mix_summary.parquet",
        overwrite=overwrite,
    )

    report_text = build_year_hs_coverage_report_text(
        count_pivot=count_pivot,
        percentage_pivot=percentage_pivot,
        summary=summary,
    )

    report_text = (
    f"HS Classification Coverage Audit\n\n"
    f"Input directory: {rel_path(data_dir)}\n"
    f"Number of files: {len(files):,}\n"
    f"ClassificationCode counts:\n"
    + "\n".join(f"{k}: {v}" for k, v in classification_counter.most_common())
    + f"\n\nYear range: {year_range[0]} to {year_range[1]} "
      f"(n_years: {year_range[2]})\n"
    + f"\nErrors: {error_count}\n\n"
    + build_year_hs_coverage_report_text(
        count_pivot=count_pivot,
        percentage_pivot=percentage_pivot,
        summary=summary,
    ))
    report_path.write_text(report_text, encoding="utf-8")
    outputs["report"] = report_path

    logger.info(f"Saved HS coverage report to: {rel_path(report_path)}")

    return outputs