"""
vacoda.hs_sync.workflow

End-to-end HS-sync diagnostic workflows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vacoda.config.hs_sync import (
    get_coverage_engine_name,
    get_coverage_input_config,
    get_coverage_output_config,
    get_coverage_product_columns,
    get_coverage_product_filename,
    get_coverage_product_thresholds,
    get_coverage_progress_bool,
    get_coverage_report_config,
    get_coverage_threads,
    get_coverage_write_config,
    is_coverage_product_enabled,
    load_hs_sync_config,
)
from vacoda.utils.logger import get_logger, rel_path, make_timestamped_log_path
from vacoda.config.paths import get_data_dir, get_output_dir
from vacoda.hs_sync.coverage import (
    build_hs_system_overall_summary,
    build_reporter_year_hs_coverage,
    build_year_hs_counts,
    build_year_hs_mix_summary,
    make_year_hs_count_pivot,
    make_year_hs_percentage_pivot,
)
from vacoda.hs_sync.reports import build_year_hs_coverage_report_text
from vacoda.io.writer import write_csv, write_parquet
from vacoda.utils.timing import timed_step, timed_workflow


logger = get_logger("HS_SYNC.WORKFLOW")


WORKFLOW_STEPS = {
    1: "Load HS-sync config and resolve paths",
    2: "Build year × HS-system counts",
    3: "Build derived HS-system summaries",
    4: "Build reporter × year × HS coverage",
    5: "Write machine-readable tables",
    6: "Build and save coverage report",
}

TOTAL_STEPS = len(WORKFLOW_STEPS)


def _resolve_data_stage_dir(block: dict) -> Path:
    """
    Resolve a data stage block to an absolute directory.

    Example
    -------
    block:
        {
            "stage": "raw",
            "subdir": "api/single"
        }

    returns:
        /storage/vacoda/data/raw/api/single
    """
    stage = block["stage"]
    subdir = block.get("subdir", "")

    base = get_data_dir(stage)
    return base / subdir if subdir else base


def _workflow_step(step_num: int):
    """
    Convenience wrapper for a timed HS-sync workflow step.
    """
    return timed_step(
        logger,
        step_num,
        WORKFLOW_STEPS[step_num],
        total_steps=TOTAL_STEPS,
    )


def run_hs_system_coverage_audit(
    *,
    config_path: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Path]:
    """
    Run HS system coverage audit.

    Reads HS-sync config, builds selected HS coverage products,
    saves machine-readable outputs, and optionally writes a text report.

    Returns
    -------
    dict[str, Path]
        Mapping of output names to written output paths.
    """
    outputs: dict[str, Path] = {}
    tables: dict[str, pd.DataFrame] = {}

    # Objects used later by the report builder.
    year_hs_counts: pd.DataFrame | None = None
    overall_summary: pd.DataFrame | None = None
    count_pivot: pd.DataFrame | None = None
    percentage_pivot: pd.DataFrame | None = None
    mix_summary: pd.DataFrame | None = None
    reporter_year_hs_coverage: pd.DataFrame | None = None

    with timed_workflow(logger, "HS Sync coverage audit"):

        # ------------------------------------------------------------------
        # STEP 1 — config and paths
        # ------------------------------------------------------------------
        with _workflow_step(1):
            cfg = load_hs_sync_config(config_path)

            input_cfg = get_coverage_input_config(cfg)
            output_cfg = get_coverage_output_config(cfg)
            report_cfg = get_coverage_report_config(cfg)
            write_cfg = get_coverage_write_config(cfg)

            data_dir = _resolve_data_stage_dir(input_cfg)
            out_dir = _resolve_data_stage_dir(output_cfg)

            report_kind = report_cfg.get("output_kind", "reports")
            report_filename = report_cfg.get("filename","audit_hs_system_coverage_report.txt")
            report_dir = get_output_dir(report_kind)

            pattern = input_cfg.get("pattern", "*.csv")
            engine = get_coverage_engine_name(cfg)
            threads = get_coverage_threads(cfg)
            progress_enabled = get_coverage_progress_bool(cfg)

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

            logger.info("Input directory: %s", rel_path(data_dir))
            logger.info("Input pattern: %s", pattern)
            logger.info("Input files found: %s", f"{len(files):,}")
            logger.info("Engine: %s, threads=%s", engine, threads)

        # ------------------------------------------------------------------
        # STEP 2 — base year × HS counts
        # ------------------------------------------------------------------
        with _workflow_step(2):
            if is_coverage_product_enabled("year_hs_counts", cfg, default=True):
                year_columns = get_coverage_product_columns("year_hs_counts", cfg)

                year_col = year_columns.get("year", "RefYear")
                hs_col = year_columns.get("hs", "ClassificationCode")

                engine_normalized = engine.strip().lower()
                if engine_normalized in {"duckdb", "polars"}:
                    logger.info("Using %s engine for data processing.", engine)
                    logger.info(
                        "%s query may take a while; progress is shown by step timing, "
                        "not per file.",
                        engine,
                    )

                year_hs_counts = build_year_hs_counts(
                    data_dir=data_dir,
                    pattern=pattern,
                    engine=engine,
                    year_col=year_col,
                    hs_col=hs_col,
                    threads=threads,
                    show_progress=progress_enabled,
                )

                table_name = get_coverage_product_filename(
                    "year_hs_counts",
                    cfg,
                    default="year_hs_counts_long",
                )
                tables[table_name] = year_hs_counts

                logger.info("Counts shape: %s", year_hs_counts.shape)

            else:
                logger.info("Skipping year × HS-system counts; disabled in config.")

        # ------------------------------------------------------------------
        # STEP 3 — derived year-HS summaries
        # ------------------------------------------------------------------
        with _workflow_step(3):
            if year_hs_counts is None:
                logger.info(
                    "Skipping derived HS-system summaries; "
                    "year_hs_counts was not built."
                )

            else:
                year_columns = get_coverage_product_columns("year_hs_counts", cfg)
                year_col = year_columns.get("year", "RefYear")
                hs_col = year_columns.get("hs", "ClassificationCode")

                if is_coverage_product_enabled(
                    "hs_system_overall_summary",
                    cfg,
                    default=True,
                ):
                    overall_summary = build_hs_system_overall_summary(
                        year_hs_counts,
                        year_col=year_col,
                        hs_col=hs_col,
                    )

                    table_name = get_coverage_product_filename(
                        "hs_system_overall_summary",
                        cfg,
                        default="hs_system_overall_summary",
                    )
                    tables[table_name] = overall_summary

                    logger.info(
                        "Overall summary shape: %s",
                        overall_summary.shape,
                    )
                else:
                    logger.info(
                        "Skipping HS-system overall summary; disabled in config."
                    )

                if is_coverage_product_enabled(
                    "year_hs_count_pivot",
                    cfg,
                    default=True,
                ):
                    count_pivot = make_year_hs_count_pivot(
                        year_hs_counts,
                        year_col=year_col,
                        hs_col=hs_col,
                    )

                    table_name = get_coverage_product_filename(
                        "year_hs_count_pivot",
                        cfg,
                        default="year_hs_counts_pivot",
                    )
                    tables[table_name] = count_pivot.reset_index()

                    logger.info("Count pivot shape: %s", count_pivot.shape)
                else:
                    logger.info("Skipping year-HS count pivot; disabled in config.")

                if is_coverage_product_enabled(
                    "year_hs_percentage_pivot",
                    cfg,
                    default=True,
                ):
                    if count_pivot is None:
                        raise RuntimeError(
                            "year_hs_percentage_pivot requires "
                            "year_hs_count_pivot."
                        )

                    percentage_pivot = make_year_hs_percentage_pivot(count_pivot)

                    table_name = get_coverage_product_filename(
                        "year_hs_percentage_pivot",
                        cfg,
                        default="year_hs_percentage_pivot",
                    )
                    tables[table_name] = percentage_pivot.reset_index()

                    logger.info(
                        "Percentage pivot shape: %s",
                        percentage_pivot.shape,
                    )
                else:
                    logger.info(
                        "Skipping year-HS percentage pivot; disabled in config."
                    )

                if is_coverage_product_enabled(
                    "year_hs_mix_summary",
                    cfg,
                    default=True,
                ):
                    if count_pivot is None:
                        raise RuntimeError(
                            "year_hs_mix_summary requires year_hs_count_pivot."
                        )

                    thresholds = get_coverage_product_thresholds(
                        "year_hs_mix_summary",
                        cfg,
                    )

                    mix_summary = build_year_hs_mix_summary(
                        count_pivot,
                        clean_threshold=float(thresholds.get("clean", 95.0)),
                        mostly_clean_threshold=float(
                            thresholds.get("mostly_clean", 90.0)
                        ),
                    )

                    table_name = get_coverage_product_filename(
                        "year_hs_mix_summary",
                        cfg,
                        default="year_hs_mix_summary",
                    )
                    tables[table_name] = mix_summary

                    logger.info("Mix summary shape: %s", mix_summary.shape)
                else:
                    logger.info("Skipping year-HS mix summary; disabled in config.")

        # ------------------------------------------------------------------
        # STEP 4 — reporter × year × HS coverage
        # ------------------------------------------------------------------
        with _workflow_step(4):
            if is_coverage_product_enabled(
                "reporter_year_hs_coverage",
                cfg,
                default=False,
            ):
                reporter_columns = get_coverage_product_columns(
                    "reporter_year_hs_coverage",
                    cfg,
                )

                reporter_year_hs_coverage = build_reporter_year_hs_coverage(
                    data_dir=data_dir,
                    pattern=pattern,
                    engine=engine,
                    threads=threads,
                    columns=reporter_columns,
                )

                table_name = get_coverage_product_filename(
                    "reporter_year_hs_coverage",
                    cfg,
                    default="reporter_year_hs_coverage",
                )
                tables[table_name] = reporter_year_hs_coverage

                logger.info(
                    "Reporter coverage shape: %s",
                    reporter_year_hs_coverage.shape,
                )

            else:
                logger.info(
                    "Skipping reporter × year × HS coverage; disabled in config."
                )

        # ------------------------------------------------------------------
        # STEP 5 — write machine-readable tables
        # ------------------------------------------------------------------
        with _workflow_step(5):
            overwrite             = bool(write_cfg.get("overwrite", True))
            write_csv_enabled     = bool(write_cfg.get("csv", True))
            write_parquet_enabled = bool(write_cfg.get("parquet", True))

            if not tables:
                logger.warning("No tables were built; nothing to write.")

            if write_csv_enabled:
                for name, table in tables.items():
                    outputs[f"{name}_csv"] = write_csv(table, out_dir / f"{name}.csv", overwrite=overwrite)

            if write_parquet_enabled:
                for name, table in tables.items():
                    outputs[f"{name}_parquet"] = write_parquet(table, out_dir / f"{name}.parquet",overwrite=overwrite)
    

        # ------------------------------------------------------------------
        # STEP 6 — build and save human-readable report
        # ------------------------------------------------------------------
        with _workflow_step(6):
            if bool(report_cfg.get("enabled", True)):
                report_text = build_year_hs_coverage_report_text(audit_info=audit_info,
                                                                overall_summary=overall_summary,
                                                                count_pivot=count_pivot,
                                                                percentage_pivot=percentage_pivot,
                                                                summary=mix_summary,
                                                                reporter_coverage=reporter_year_hs_coverage)

                report_base_path = report_dir / report_filename

                report_timestamp = bool(report_cfg.get("timestamp", True))
                write_latest = bool(report_cfg.get("write_latest", True))

                if report_timestamp:
                    report_path = make_timestamped_log_path(report_base_path, timestamp=run_id)
                else:
                    report_path = report_base_path

                report_path.write_text(report_text, encoding="utf-8")
                outputs["report"] = report_path

                if write_latest and report_timestamp:
                    latest_path = report_base_path.with_name( f"{report_base_path.stem}_latest{report_base_path.suffix}")
                    latest_path.write_text(report_text, encoding="utf-8")
                    outputs["report_latest"] = latest_path

                    logger.info("Saved latest report copy: %s", rel_path(latest_path))

                logger.info("Saved report: %s", rel_path(report_path))

            else:
                logger.info("Skipping text report; disabled in config.")


    outputs["output_dir"] = out_dir

    return outputs