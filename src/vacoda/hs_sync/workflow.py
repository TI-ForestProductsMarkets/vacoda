"""
vacoda.hs_sync.workflow

End-to-end HS synchronization diagnostics workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from vacoda.config.paths import get_data_dir
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
)

from vacoda.hs_sync.coverage import (
    build_hs_concordance_coverage_from_source_summary,
    get_missing_hs_mappings,
    get_one_to_many_hs_mappings,
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
    Callable[..., pd.DataFrame],
] = {
    "hs_concordance_coverage": _build_hs_concordance_coverage_product,
    "hs_concordance_summary": _build_hs_concordance_summary_product,
    "missing_hs_mappings": _build_missing_hs_mappings_product,
    "one_to_many_hs_mappings": _build_one_to_many_hs_mappings_product,
    "value_split_hs_mappings": _build_value_split_hs_mappings_product,
}


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