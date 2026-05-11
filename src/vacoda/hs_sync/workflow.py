"""
vacoda.hs_sync.workflow

End-to-end HS-sync diagnostic workflows.
"""

from __future__ import annotations

from pathlib import Path

from vacoda.config.hs_sync import (
    get_coverage_columns,
    get_coverage_engine_name,
    get_coverage_input_config,
    get_coverage_output_config,
    get_coverage_progress_bool,
    get_coverage_report_config,
    get_coverage_thresholds,
    get_coverage_threads,
    get_coverage_write_config,
    load_hs_sync_config,
)
from vacoda.config.paths import get_data_dir, get_output_dir
from vacoda.hs_sync.coverage import (
    build_hs_system_overall_summary,
    build_year_hs_counts,
    build_year_hs_mix_summary,
    make_year_hs_count_pivot,
    make_year_hs_percentage_pivot,
)
from vacoda.hs_sync.reports import build_year_hs_coverage_report_text
from vacoda.io.writer import write_csv, write_parquet
from vacoda.utils.logger import get_logger, rel_path
from vacoda.utils.timing import timed_step, timed_workflow


logger = get_logger("HS_SYNC.WORKFLOW")


WORKFLOW_STEPS = {
    1: "Load HS-sync config and resolve paths",
    2: "Build year × HS-system counts",
    3: "Build pivots and mix summary",
    4: "Write tables",
    5: "Build and write text report",
}

TOTAL_STEPS = len(WORKFLOW_STEPS)

def _resolve_data_stage_dir(block: dict) -> Path:
    stage = block["stage"]
    subdir = block.get("subdir", "")

    base = get_data_dir(stage)
    return base / subdir if subdir else base



def run_hs_system_coverage_audit(
    *,
    config_path: Path | None = None,
) -> dict[str, Path]:
    """
    Run HS system coverage audit.

    Reads HS-sync config, builds year × HS-system coverage diagnostics,
    saves machine-readable outputs, and writes a text report.
    """
    outputs: dict[str, Path] = {}

    
    with timed_workflow(logger, "HS system coverage audit"):
        step = 1

        with timed_step(logger, step, WORKFLOW_STEPS[step], total_steps=TOTAL_STEPS,):
            cfg = load_hs_sync_config(config_path)

            input_cfg = get_coverage_input_config(cfg)
            output_cfg = get_coverage_output_config(cfg)
            report_cfg = get_coverage_report_config(cfg)
            columns = get_coverage_columns(cfg)
            progress_enabled = get_coverage_progress_bool(cfg)
            thresholds = get_coverage_thresholds(cfg)
            write_cfg = get_coverage_write_config(cfg)

            data_dir = _resolve_data_stage_dir(input_cfg)
            out_dir = _resolve_data_stage_dir(output_cfg)

            report_kind = report_cfg.get("output_kind", "reports")
            report_filename = report_cfg.get(
                "filename",
                "audit_hs_system_coverage_report.txt",
            )
            report_dir = get_output_dir(report_kind)

            pattern = input_cfg.get("pattern", "*.csv")
            engine = get_coverage_engine_name(cfg)
            threads = get_coverage_threads(cfg)

            out_dir.mkdir(parents=True, exist_ok=True)
            report_dir.mkdir(parents=True, exist_ok=True)

            files = sorted(data_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(
                    f"No input files found in {rel_path(data_dir)} "
                    f"matching pattern {pattern!r}"
                )

            audit_info = {
                "input_dir": str(data_dir),
                "pattern": pattern,
                "n_files": len(files),
                "engine": engine,
                "threads": threads,
            }

            logger.info(f"Input directory: {rel_path(data_dir)}")
            logger.info(f"Input pattern: {pattern}")
            logger.info(f"Input files found: {len(files):,}")
            logger.info(f"Engine: {engine}, threads={threads}")

        ##########################################################################################
        step = step + 1
        with timed_step(logger, step, WORKFLOW_STEPS[step], total_steps=TOTAL_STEPS,):

            if engine in ["DuckDB", "duckdb", "polars"]:
                logger.info(f"Using {engine} engine for data processing.")
                logger.info("DuckDB query may take a while; progress is shown by step timing, not per file.")

            year_hs_counts = build_year_hs_counts(
                data_dir=data_dir,
                pattern=pattern,
                engine=engine,
                year_col=columns.get("year", "RefYear"),
                hs_col=columns.get("hs", "ClassificationCode"),
                threads=threads,
                show_progress=progress_enabled,
            )

            overall_summary = build_hs_system_overall_summary(
                year_hs_counts,
                year_col=columns.get("year", "RefYear"),
                hs_col=columns.get("hs", "ClassificationCode"),
            )

            logger.info(f"Counts shape: {year_hs_counts.shape}")
            logger.info(f"Overall summary shape: {overall_summary.shape}")

        ###########################################################################################
        step = step + 1

        with timed_step(logger, step, WORKFLOW_STEPS[step], total_steps=TOTAL_STEPS,):
            count_pivot = make_year_hs_count_pivot(
                year_hs_counts,
                year_col=columns.get("year", "RefYear"),
                hs_col=columns.get("hs", "ClassificationCode"),
            )

            percentage_pivot = make_year_hs_percentage_pivot(count_pivot)

            summary = build_year_hs_mix_summary(
                count_pivot,
                clean_threshold=float(thresholds.get("clean", 95.0)),
                mostly_clean_threshold=float(thresholds.get("mostly_clean", 90.0)),
            )

            logger.info(f"Count pivot shape: {count_pivot.shape}")
            logger.info(f"Percentage pivot shape: {percentage_pivot.shape}")
            logger.info(f"Mix summary shape: {summary.shape}")

        ###########################################################################################
        step = step + 1
        with timed_step(logger, step, WORKFLOW_STEPS[step], total_steps=TOTAL_STEPS,):
            overwrite = bool(write_cfg.get("overwrite", True))
            write_csv_enabled = bool(write_cfg.get("csv", True))
            write_parquet_enabled = bool(write_cfg.get("parquet", True))

            tables = {
                "year_hs_counts_long": year_hs_counts,
                "hs_system_overall_summary": overall_summary,
                "year_hs_counts_pivot": count_pivot.reset_index(),
                "year_hs_percentage_pivot": percentage_pivot.reset_index(),
                "year_hs_mix_summary": summary,
            }

            if write_csv_enabled:
                for name, table in tables.items():
                    outputs[f"{name}_csv"] = write_csv(
                        table,
                        out_dir / f"{name}.csv",
                        overwrite=overwrite,
                    )

            if write_parquet_enabled:
                for name, table in tables.items():
                    outputs[f"{name}_parquet"] = write_parquet(
                        table,
                        out_dir / f"{name}.parquet",
                        overwrite=overwrite,
                    )

        ###########################################################################################
        step = step + 1
        with timed_step(logger, step, WORKFLOW_STEPS[step], total_steps=TOTAL_STEPS,):
            report_text = build_year_hs_coverage_report_text(
                audit_info=audit_info,
                overall_summary=overall_summary,
                count_pivot=count_pivot,
                percentage_pivot=percentage_pivot,
                summary=summary,
            )

            report_path = report_dir / report_filename
            report_path.write_text(report_text, encoding="utf-8")
            outputs["report"] = report_path

            logger.info(f"Saved report: {rel_path(report_path)}")

    return outputs