"""
vacoda.hs_sync.workflow

End-to-end HS synchronization diagnostics workflow.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from vacoda.config.hs_sync import HsSyncConfig

from vacoda.hs_sync.coverage import (
    build_target_hs_source_code_coverage,
    build_target_hs_mapping_coverage,
    summarize_target_hs_mapping_coverage,
    get_target_hs_missing_mappings,
    build_target_hs_mapping_interpretation_report,
    build_target_hs_mapping_interpretation_display_report,
)

from vacoda.hs_sync.diagnostics import (
    build_trade_source_pair_summary,
)
from vacoda.io.writer import write_csv, write_parquet
from vacoda.utils.logger import get_logger, rel_path
from vacoda.utils.timing import timed_step, timed_workflow


logger = get_logger("HS_SYNC.WORKFLOW")


WORKFLOW_STEPS = {
    1: "Load target-HS coverage config and resolve paths",
    2: "Load trade source summary and mapping files",
    3: "Build target-HS mapping coverage products",
    4: "Write target-HS mapping coverage outputs",
}

TOTAL_STEPS = len(WORKFLOW_STEPS)


def _target_hs_workflow_step(step_num: int):
    return timed_step(
        logger,
        step_num,
        WORKFLOW_STEPS[step_num],
        total_steps=TOTAL_STEPS,
    )


def _workflow_step(step_num: int):
    return timed_step(
        logger,
        step_num,
        WORKFLOW_STEPS[step_num],
        total_steps=TOTAL_STEPS,
    )


def run_target_hs_mapping_coverage_audit(
    *,
    config_path: Path | None = None,
    run_id: str | None = None) -> dict[str, Path]:
    """
    Run target-HS mapping coverage diagnostics.

    This workflow answers:
        How many trade records can be mapped to target HS without
        conversion factors / optimized weights?

    It uses only deterministic concordance relationship categories:
        - already_target
        - 1:1
        - n:1
        - 1:n
        - n:n
        - missing_mapping

    It does not require Harvard optimized weights.
    """

    outputs: dict[str, Path] = {}
    runtime: dict[str, pd.DataFrame] = {}
    tables: dict[str, pd.DataFrame] = {}

    with timed_workflow(logger, "Target-HS mapping coverage audit"):

        with _target_hs_workflow_step(1):
            cfg = HsSyncConfig.from_yaml(config_path)
            
            ### define paths and parameters based on config
            input_dir = cfg.coverage_input_dir()
            mapping_dir = cfg.coverage_mapping_dir()
            out_dir = cfg.coverage_output_dir()
            out_dir.mkdir(parents=True, exist_ok=True)

            input_pattern = cfg.coverage.input.pattern
            target_hs = cfg.coverage.target_hs
            min_year = cfg.coverage.min_year
            max_year = cfg.coverage.max_year

            logger.info("Coverage input directory: %s", rel_path(input_dir))
            logger.info("Coverage input pattern: %s", input_pattern)
            logger.info("Mapping directory: %s", rel_path(mapping_dir))
            logger.info("Output directory: %s", rel_path(out_dir))
            logger.info("Coverage target HS: %s", target_hs)
            logger.info("Coverage year range: %s–%s", min_year, max_year)

        with _target_hs_workflow_step(2):
            source_summary = build_trade_source_pair_summary(
                    data_dir=input_dir,
                    pattern=input_pattern,
                    engine=cfg.engine.name,
                    year_col=cfg.columns.year,
                    hs_col=cfg.columns.hs,
                    cmd_col=cfg.columns.cmd,
                    value_col=cfg.columns.value,
                    min_year=min_year,
                    threads=cfg.engine.threads,
                    show_progress=cfg.progress.enabled,
            )

            # For now, filter max_year after summary if your helper only supports min_year.
            if "year" in source_summary.columns:
                source_summary = source_summary[source_summary["year"] <= max_year]

            mapping_files = cfg.coverage_mapping_files()
            
            mappings: dict[str, pd.DataFrame] = {}
            for mapping_name, path in mapping_files.items():
                mappings[mapping_name] = pd.read_csv(path)
                logger.info(
                        "Loaded mapping %s: %s shape=%s",
                        mapping_name,
                        rel_path(path),
                        mappings[mapping_name].shape,
                    )

            logger.info("Source summary shape: %s", source_summary.shape)

        with _target_hs_workflow_step(3):

            coverage = build_target_hs_mapping_coverage(
                source_summary=source_summary,
                mappings=mappings,
                target_hs=target_hs,
                mapping_columns=cfg.coverage.mappings.columns.as_dict(),
                easy_without_weights=cfg.coverage.categories.easy_without_weights,
                needs_weights=cfg.coverage.categories.needs_weights,
                missing_categories=cfg.coverage.categories.missing,
            )

            source_code_coverage = build_target_hs_source_code_coverage(coverage)
            runtime["source_code_coverage"] = source_code_coverage
            tables["source_code_coverage"] = source_code_coverage
            
            #runtime["mapping_coverage"] = coverage
            #tables["mapping_coverage"] = coverage

            summary = summarize_target_hs_mapping_coverage(coverage)
            runtime["mapping_summary"] = summary
            tables["mapping_summary"] = summary

            interpretation_report = build_target_hs_mapping_interpretation_report(source_code_coverage)

            interpretation_display_report = (build_target_hs_mapping_interpretation_display_report(interpretation_report))
            runtime["mapping_interpretation_report"] = (interpretation_display_report)
            tables["mapping_interpretation_report"] = (interpretation_display_report)

            runtime["mapping_interpretation_display_report"] = interpretation_report
            tables["mapping_interpretation_display_report"] = interpretation_report

            logger.info(
                f"{target_hs} interpretation report:\n%s",
                interpretation_report[
                    [
                        "result_class",
                        "share_rows_percent",
                        "share_value_percent",
                        "interpretation",
                    ]
                ].to_string(index=False),
            )

            missing = get_target_hs_missing_mappings(
                                            coverage,
                                            top_n=cfg.product_options("missing_mappings").get("top_n", 50),
                                        )
            
            runtime["missing_mappings"] = missing
            tables["missing_mappings"] = missing

            needs_weights = coverage[coverage["conversion_need"] == "needs_weights"].copy()
            runtime["needs_weights"] = needs_weights
            tables["needs_weights"] = needs_weights

            easy = coverage[coverage["conversion_need"] == "easy_without_weights"].copy()
            runtime["easy_without_weights"] = easy
            tables["easy_without_weights"] = easy

            for name, table in tables.items():
                logger.info("%s shape: %s", name, table.shape)

        with _target_hs_workflow_step(4):
            
            overwrite = cfg.coverage.write.overwrite
            write_csv_enabled = cfg.coverage.write.csv
            write_parquet_enabled = cfg.coverage.write.parquet

            for product_name, table in tables.items():
                
                if not cfg.product_write_enabled(product_name, default=True):
                    continue
                
                
                filename = cfg.product_filename(product_name)
                product_out_dir = cfg.product_output_dir(product_name, default_dir=out_dir,)
                product_out_dir.mkdir(parents=True, exist_ok=True)
                
                
                if write_csv_enabled:
                    outputs[f"{product_name}_csv"] = write_csv(
                                table,
                                product_out_dir / f"{filename}.csv",
                                overwrite=overwrite,
                                allow_empty=True,
                                )
                
                if write_parquet_enabled:
                    outputs[f"{product_name}_parquet"] = write_parquet(
                                table,
                                product_out_dir / f"{filename}.parquet",
                                overwrite=overwrite,
                                allow_empty=True,
                                )

    outputs["output_dir"] = out_dir
    return outputs