"""
vacoda.hs_sync.workflow

End-to-end HS synchronization diagnostics workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from vacoda.config.paths import get_data_dir, get_output_dir
from vacoda.hs_sync.concordance import load_hs_concordance

from vacoda.config.hs_sync import (
    get_cmd_col,
    get_concordance_file,
    get_concordance_stage,
    get_concordance_weight_col,
    get_engine_name,
    get_hs_col,
    get_input_config,
    get_input_pattern,
    get_min_year,
    get_output_config,
    get_product_dependency,
    get_product_filename,
    get_product_options,
    get_product_write_bool,
    get_progress_bool,
    get_target_hs,
    get_engine_threads,
    get_value_col,
    get_weight_sum_tolerance,
    get_write_csv_bool,
    get_write_overwrite_bool,
    get_write_parquet_bool,
    get_year_col,
    is_product_enabled,
    load_hs_sync_config,
    get_coverage_input_stage,
    get_coverage_input_subdir,
    get_coverage_input_pattern,
    get_coverage_mappings_stage,
    get_coverage_mappings_subdir,
    get_coverage_mapping_files,
    get_coverage_mapping_columns,
    get_coverage_target_hs,
    get_coverage_min_year,
    get_coverage_max_year,
    get_coverage_easy_without_weights_categories,
    get_coverage_needs_weights_categories,
    get_coverage_missing_categories,
    get_coverage_output_stage,
    get_coverage_output_subdir,
    get_coverage_write_csv_bool,
    get_coverage_write_parquet_bool,
    get_coverage_write_overwrite_bool,
    get_coverage_report_output_kind,
    get_coverage_report_filename,
    get_coverage_report_timestamp_bool,
    get_coverage_report_write_latest_bool,
    get_target_h5_mapping_interpretation_display_report_output_kind,
)

from vacoda.hs_sync.coverage import (
    build_hs_concordance_coverage_from_source_summary,
    get_missing_hs_mappings,
    get_one_to_many_hs_mappings,
    build_target_h5_mapping_coverage,
    summarize_target_h5_mapping_coverage,
    get_target_h5_missing_mappings,
    build_target_h5_mapping_interpretation_report,
    build_target_h5_mapping_interpretation_display_report
)

from vacoda.hs_sync.diagnostics import (
    build_trade_source_pair_summary,
    summarize_hs_concordance_coverage,
)
from vacoda.io.writer import write_csv, write_parquet
from vacoda.utils.logger import get_logger, rel_path
from vacoda.utils.timing import timed_step, timed_workflow


logger = get_logger("HS_SYNC.WORKFLOW")


WORKFLOW_STEPS = {
    1: "Load HS-sync config and resolve paths",
    2: "Load trade data and concordance",
    3: "Build HS concordance coverage products",
    4: "Write HS-sync diagnostic outputs",
}

TOTAL_STEPS = len(WORKFLOW_STEPS)


TARGET_H5_WORKFLOW_STEPS = {
    1: "Load target-H5 coverage config and resolve paths",
    2: "Load trade source summary and mapping files",
    3: "Build target-H5 mapping coverage products",
    4: "Write target-H5 mapping coverage outputs",
}

TARGET_H5_TOTAL_STEPS = len(TARGET_H5_WORKFLOW_STEPS)


def _target_h5_workflow_step(step_num: int):
    return timed_step(
        logger,
        step_num,
        TARGET_H5_WORKFLOW_STEPS[step_num],
        total_steps=TARGET_H5_TOTAL_STEPS,
    )


def _workflow_step(step_num: int):
    return timed_step(
        logger,
        step_num,
        WORKFLOW_STEPS[step_num],
        total_steps=TOTAL_STEPS,
    )


def _resolve_data_stage_dir(block: dict[str, Any]) -> Path:
    stage = block["stage"]
    subdir = block.get("subdir", "")

    base = get_data_dir(stage)
    return base / subdir if subdir else base


def _dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """Serialize a DataFrame into a simple GitHub-flavored markdown table."""
    if df.empty:
        headers = [str(col) for col in df.columns]
        if not headers:
            return ""
        header_row = "| " + " | ".join(headers) + " |"
        divider_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        return "\n".join([header_row, divider_row])

    def _sanitize(value: Any) -> str:
        text = str(value)
        return text.replace("\n", " ").replace("|", r"\|")

    headers = [str(col) for col in df.columns]
    header_row = "| " + " | ".join(headers) + " |"
    divider_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    rows: list[str] = []
    for row in df.itertuples(index=False, name=None):
        cells = [_sanitize(value) for value in row]
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_row, divider_row, *rows])


def _build_hs_concordance_coverage_product(
    *,
    runtime: dict[str, pd.DataFrame],
    source_summary: pd.DataFrame,
    concordance_df: pd.DataFrame,
    cfg: dict[str, Any]) -> pd.DataFrame:
    return build_hs_concordance_coverage_from_source_summary(
        source_summary=source_summary,
        concordance_df=concordance_df,
        target_hs=get_target_hs(cfg),
        weight_col=get_concordance_weight_col(cfg),
        weight_sum_tolerance=get_weight_sum_tolerance(cfg),
    )


def _build_hs_concordance_summary_product(
    *,
    runtime: dict[str, pd.DataFrame],
    source_summary: pd.DataFrame,
    concordance_df: pd.DataFrame,
    cfg: dict[str, Any]) -> pd.DataFrame:
    summary = summarize_hs_concordance_coverage(
        runtime["hs_concordance_coverage"]
    )
    return pd.DataFrame([summary])

def _build_missing_hs_mappings_product(
    *,
    runtime: dict[str, pd.DataFrame],
    source_summary: pd.DataFrame,
    concordance_df: pd.DataFrame,
    cfg: dict[str, Any]) -> pd.DataFrame:
    options = get_product_options("missing_hs_mappings", cfg)
    top_n = options.get("top_n")

    return get_missing_hs_mappings(
        runtime["hs_concordance_coverage"],
        top_n=top_n,
    )


def _build_one_to_many_hs_mappings_product(
    *,
    runtime: dict[str, pd.DataFrame],
    source_summary: pd.DataFrame,
    concordance_df: pd.DataFrame,
    cfg: dict[str, Any]) -> pd.DataFrame:
    return get_one_to_many_hs_mappings(
        runtime["hs_concordance_coverage"],
        only_value_splits=False,
    )


def _build_value_split_hs_mappings_product(
    *,
    runtime: dict[str, pd.DataFrame],
    source_summary: pd.DataFrame,
    concordance_df: pd.DataFrame,
    cfg: dict[str, Any]) -> pd.DataFrame:
    return get_one_to_many_hs_mappings(
        runtime["hs_concordance_coverage"],
        only_value_splits=True,
    )


PRODUCT_BUILDERS: dict[
    str,
    Callable[..., pd.DataFrame],] = {
    "hs_concordance_coverage": _build_hs_concordance_coverage_product,
    "hs_concordance_summary": _build_hs_concordance_summary_product,
    "missing_hs_mappings": _build_missing_hs_mappings_product,
    "one_to_many_hs_mappings": _build_one_to_many_hs_mappings_product,
    "value_split_hs_mappings": _build_value_split_hs_mappings_product,}


def run_hs_sync_diagnostics(
    *,
    config_path: Path | None = None,
    run_id: str | None = None) -> dict[str, Path]:
    """
    Run HS-sync concordance diagnostics.

    This workflow does not yet apply weighted harmonization.
    It only checks mapping readiness.
    """

    outputs: dict[str, Path] = {}
    runtime: dict[str, pd.DataFrame] = {}
    tables: dict[str, pd.DataFrame] = {}

    with timed_workflow(logger, "HS-sync diagnostics"):

        with _workflow_step(1):
            cfg = load_hs_sync_config(config_path)

            input_cfg = get_input_config(cfg)
            output_cfg = get_output_config(cfg)

            data_dir = _resolve_data_stage_dir(input_cfg)
            out_dir = _resolve_data_stage_dir(output_cfg)

            pattern = get_input_pattern(cfg)

            out_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Input directory: %s", rel_path(data_dir))
            logger.info("Input pattern: %s", pattern)
            logger.info("Output directory: %s", rel_path(out_dir))
            logger.info("Target HS: %s", get_target_hs(cfg))
            logger.info("Minimum year: %s", get_min_year(cfg))

        with _workflow_step(2):
            trade_columns = [
                get_year_col(cfg),
                get_hs_col(cfg),
                get_cmd_col(cfg),
                get_value_col(cfg),
            ]

            source_summary = build_trade_source_pair_summary(
                                            data_dir=data_dir,
                                            pattern=pattern,
                                            engine=get_engine_name(cfg),
                                            year_col=get_year_col(cfg),
                                            hs_col=get_hs_col(cfg),
                                            cmd_col=get_cmd_col(cfg),
                                            value_col=get_value_col(cfg),
                                            min_year=get_min_year(cfg),
                                            threads=get_engine_threads(cfg),
                                            show_progress=get_progress_bool(cfg),
                                        )

            concordance_path = (
                get_data_dir(get_concordance_stage(cfg))
                / get_concordance_file(cfg)
            )

            concordance_df = load_hs_concordance(
                concordance_path,
                cfg=cfg,
            )

            logger.info("Source summary shape: %s", source_summary.shape)
            logger.info("Concordance shape: %s", concordance_df.shape)

        with _workflow_step(3):
            for product_name, builder in PRODUCT_BUILDERS.items():
                if not is_product_enabled(product_name, cfg, default=False):
                    logger.info("Skipping %s; disabled in config.", product_name)
                    continue

                dependency = get_product_dependency(product_name, cfg)

                if dependency is not None and dependency not in runtime:
                    logger.warning(
                        "Skipping %s; dependency %s was not built.",
                        product_name,
                        dependency,
                    )
                    continue

                table = builder(
                    runtime=runtime,
                    source_summary=source_summary,
                    concordance_df=concordance_df,
                    cfg=cfg,
                )

                runtime[product_name] = table

                if get_product_write_bool(product_name, cfg, default=True):
                    tables[product_name] = table

                logger.info("%s shape: %s", product_name, table.shape)

        with _workflow_step(4):
            overwrite = get_write_overwrite_bool(cfg)
            write_csv_enabled = get_write_csv_bool(cfg)
            write_parquet_enabled = get_write_parquet_bool(cfg)

            if not tables:
                logger.warning("No HS-sync tables were built; nothing to write.")

            for product_name, table in tables.items():
                filename = get_product_filename(product_name, cfg)

                if write_csv_enabled:
                    outputs[f"{product_name}_csv"] = write_csv(
                        table,
                        out_dir / f"{filename}.csv",
                        overwrite=overwrite,
                        allow_empty=True,
                    )

                if write_parquet_enabled:
                    outputs[f"{product_name}_parquet"] = write_parquet(
                        table,
                        out_dir / f"{filename}.parquet",
                        overwrite=overwrite,
                        allow_empty=True,
                    )

    outputs["output_dir"] = out_dir
    return outputs



def run_target_h5_mapping_coverage_audit(
    *,
    config_path: Path | None = None,
    run_id: str | None = None) -> dict[str, Path]:
    """
    Run target-H5 mapping coverage diagnostics.

    This workflow answers:
        How many trade records can be mapped to target H5 without
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

    with timed_workflow(logger, "Target-H5 mapping coverage audit"):

        with _target_h5_workflow_step(1):
            cfg = load_hs_sync_config(config_path)

            # These getters come from your new coverage.* config.
            input_dir = _resolve_data_stage_dir(
                {
                    "stage": get_coverage_input_stage(cfg),
                    "subdir": get_coverage_input_subdir(cfg),
                }
            )

            mapping_dir = _resolve_data_stage_dir(
                {
                    "stage": get_coverage_mappings_stage(cfg),
                    "subdir": get_coverage_mappings_subdir(cfg),
                }
            )

            out_dir = _resolve_data_stage_dir(
                {
                    "stage": get_coverage_output_stage(cfg),
                    "subdir": get_coverage_output_subdir(cfg),
                }
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            input_pattern = get_coverage_input_pattern(cfg)
            target_hs = get_coverage_target_hs(cfg)
            min_year = get_coverage_min_year(cfg)
            max_year = get_coverage_max_year(cfg)

            logger.info("Coverage input directory: %s", rel_path(input_dir))
            logger.info("Coverage input pattern: %s", input_pattern)
            logger.info("Mapping directory: %s", rel_path(mapping_dir))
            logger.info("Output directory: %s", rel_path(out_dir))
            logger.info("Target HS: %s", target_hs)
            logger.info("Year range: %s–%s", min_year, max_year)

        with _target_h5_workflow_step(2):
            source_summary = build_trade_source_pair_summary(
                data_dir=input_dir,
                pattern=input_pattern,
                engine=get_engine_name(cfg),
                year_col=get_year_col(cfg),
                hs_col=get_hs_col(cfg),
                cmd_col=get_cmd_col(cfg),
                value_col=get_value_col(cfg),
                min_year=min_year,
                threads=get_engine_threads(cfg),
                show_progress=get_progress_bool(cfg),
            )

            # For now, filter max_year after summary if your helper only supports min_year.
            if "year" in source_summary.columns:
                source_summary = source_summary[source_summary["year"] <= max_year]

            mapping_files = get_coverage_mapping_files(cfg)

            mappings: dict[str, pd.DataFrame] = {}
            for mapping_name, filename in mapping_files.items():
                path = mapping_dir / filename
                mappings[mapping_name] = pd.read_csv(path)
                logger.info(
                    "Loaded mapping %s: %s shape=%s",
                    mapping_name,
                    rel_path(path),
                    mappings[mapping_name].shape,
                )

            logger.info("Source summary shape: %s", source_summary.shape)

        with _target_h5_workflow_step(3):
            # This is the function you add in coverage.py next.
            coverage = build_target_h5_mapping_coverage(
                source_summary=source_summary,
                mappings=mappings,
                target_hs=target_hs,
                mapping_columns=get_coverage_mapping_columns(cfg),
                easy_without_weights=get_coverage_easy_without_weights_categories(cfg),
                needs_weights=get_coverage_needs_weights_categories(cfg),
                missing_categories=get_coverage_missing_categories(cfg),
            )

            runtime["target_h5_mapping_coverage"] = coverage
            tables["target_h5_mapping_coverage"] = coverage

            summary = summarize_target_h5_mapping_coverage(coverage)
            runtime["target_h5_mapping_summary"] = summary
            tables["target_h5_mapping_summary"] = summary

            interpretation_report = build_target_h5_mapping_interpretation_report(summary)

            interpretation_display_report = (build_target_h5_mapping_interpretation_display_report(interpretation_report))
            runtime["target_h5_mapping_interpretation_display_report"] = (interpretation_display_report)
            tables["target_h5_mapping_interpretation_display_report"] = (interpretation_display_report)

            runtime["target_h5_mapping_interpretation_report"] = interpretation_report
            tables["target_h5_mapping_interpretation_report"] = interpretation_report

            logger.info(
                "Target-H5 interpretation report:\n%s",
                interpretation_report[
                    [
                        "result_class",
                        "share_rows_percent",
                        "share_value_percent",
                        "interpretation",
                    ]
                ].to_string(index=False),
            )

            missing = get_target_h5_missing_mappings(
                coverage,
                top_n=get_product_options(
                    "target_h5_missing_mappings",
                    cfg,
                ).get("top_n", 50),
            )
            runtime["target_h5_missing_mappings"] = missing
            tables["target_h5_missing_mappings"] = missing

            needs_weights = coverage[
                coverage["conversion_need"] == "needs_weights"
            ].copy()
            runtime["target_h5_needs_weights"] = needs_weights
            tables["target_h5_needs_weights"] = needs_weights

            easy = coverage[
                coverage["conversion_need"] == "easy_without_weights"
            ].copy()
            runtime["target_h5_easy_without_weights"] = easy
            tables["target_h5_easy_without_weights"] = easy

            for name, table in tables.items():
                logger.info("%s shape: %s", name, table.shape)

        with _target_h5_workflow_step(4):
            from vacoda.utils.logger import make_timestamped_log_path
            
            overwrite = get_coverage_write_overwrite_bool(cfg)
            write_csv_enabled = get_coverage_write_csv_bool(cfg)
            write_parquet_enabled = get_coverage_write_parquet_bool(cfg)

            for product_name, table in tables.items():
                filename = get_product_filename(product_name, cfg)
                product_out_dir = out_dir

                if product_name == "target_h5_mapping_interpretation_display_report":
                    output_kind = (
                        get_target_h5_mapping_interpretation_display_report_output_kind(cfg)
                    )
                    product_out_dir = get_output_dir(output_kind)
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

            if "target_h5_mapping_interpretation_display_report" in runtime:
                interpretation_display_report = runtime[
                    "target_h5_mapping_interpretation_display_report"
                ]
                display_output_kind = (
                    get_target_h5_mapping_interpretation_display_report_output_kind(cfg)
                )
                display_report_dir = get_output_dir(display_output_kind)
                display_report_dir.mkdir(parents=True, exist_ok=True)
                display_filename = get_product_filename(
                    "target_h5_mapping_interpretation_display_report",
                    cfg,
                )
                markdown_path = display_report_dir / f"{display_filename}.md"
                markdown_text = build_target_h5_mapping_interpretation_display_report(
                    runtime["target_h5_mapping_interpretation_report"],
                    output="markdown",
                )

                markdown_path.write_text(
                    "# Target-H5 mapping coverage interpretation\n\n"
                    + markdown_text
                    + "\n",
                    encoding="utf-8",
                )

                outputs["target_h5_mapping_interpretation_display_report_md"] = markdown_path
                logger.info("Markdown interpretation report written: %s", rel_path(markdown_path))

            report_dir = get_output_dir(get_coverage_report_output_kind(cfg))
            report_dir.mkdir(parents=True, exist_ok=True)

            report_filename = get_coverage_report_filename(cfg)
            report_base_path = report_dir / report_filename

            if get_coverage_report_timestamp_bool(cfg):
                report_path = make_timestamped_log_path(report_base_path, timestamp=run_id,)
            else:
                report_path = report_base_path

            if "target_h5_mapping_interpretation_report" in runtime:
                interpretation_report = runtime["target_h5_mapping_interpretation_report"]

                outputs["target_h5_mapping_interpretation_report_csv"] = write_csv(
                            interpretation_report,
                            report_path,
                            overwrite=True,
                            allow_empty=True)

                logger.info("Interpretation report written: %s", rel_path(report_path))

            if get_coverage_report_write_latest_bool(cfg) and get_coverage_report_timestamp_bool(cfg):
                latest_path = report_base_path.with_name(f"{report_base_path.stem}_latest{report_base_path.suffix}")

                outputs["target_h5_mapping_interpretation_report_latest_csv"] = write_csv(
                        interpretation_report,
                        latest_path,
                        overwrite=True,
                        allow_empty=True)

                logger.info("Latest interpretation report written: %s", rel_path(latest_path))


    outputs["output_dir"] = out_dir
    return outputs