"""
vacoda.hs_coverage.workflow

End-to-end HS-coverage diagnostic workflows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vacoda.config.hs_coverage import (
    get_coverage_input_config,
    get_coverage_output_config,
    get_coverage_report_config,
    get_coverage_write_config,
    get_coverage_input_data_pattern,
    get_coverage_engine_name,
    get_coverage_threads,
    get_coverage_report_output_kind,
    get_coverage_report_filename,
    get_coverage_product_columns,
    get_coverage_product_thresholds,
    get_coverage_product_top_n,
    get_coverage_product_dependency,
    is_coverage_product_enabled,
    load_hs_coverage_config,
    get_coverage_progress_bool,
    get_coverage_product_filename
    )
from vacoda.config.paths import get_data_dir, get_output_dir
from vacoda.hs_coverage.coverage import (
    build_hs_system_overall_summary,
    build_reporter_year_hs_coverage,
    build_year_hs_counts,
    build_year_hs_mix_summary,
    make_year_hs_count_pivot,
    make_year_hs_percentage_pivot,
    build_hs_coverage_summary_tables,
    build_year_hs_cmd4_coverage,
    build_cmd4_hs_diversity_summary,
    build_cmd4_hs_timeline,
    build_cmd4_hs_transition_events,
    build_top_cmd4_by_year_hs,
)
from vacoda.hs_coverage.reports import build_year_hs_coverage_report_text
from vacoda.io.writer import write_csv, write_parquet
from vacoda.utils.logger import get_logger, rel_path, make_timestamped_log_path
from vacoda.utils.timing import timed_step, timed_workflow


logger = get_logger("HS_COVERAGE.WORKFLOW")


WORKFLOW_STEPS = {
    1: "Load HS-coverage config",
    2: "Build year × HS-system counts",
    3: "Build derived HS-system summaries",
    4: "Build year × HS × cmd4 coverage",
    5: "Build derived cmd4 HS coverage summaries",
    6: "Build reporter × year × HS coverage",
    7: "Build country-level HS coverage summary tables",
    8: "Write machine-readable tables",
    9: "Build and save coverage report",
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
    Convenience wrapper for a timed HS-coverage workflow step.
    """
    return timed_step(
        logger,
        step_num,
        WORKFLOW_STEPS[step_num],
        total_steps=TOTAL_STEPS,
    )


def _get_required_runtime_table(
    *,
    product_name: str,
    runtime: dict[str, pd.DataFrame],
    cfg: dict,
) -> pd.DataFrame | None:
    """
    Return the dependency table required by product_name.

    Dependencies are declared in YAML via:

        depends_on: some_product

    Runtime tables are stored by logical product name, not filename.

    Returns
    -------
    pd.DataFrame | None
        Dependency table if available. Returns None if the dependency was not
        built, allowing the workflow to skip the dependent product gracefully.
    """
    dependency_name = get_coverage_product_dependency(product_name, cfg)

    if dependency_name is None:
        raise ValueError(
            f"Product '{product_name}' requires a dependency, "
            "but no 'depends_on' value is configured."
        )

    if dependency_name not in runtime:
        logger.warning(
            "Skipping %s; dependency %s was not built.",
            product_name,
            dependency_name,
        )
        return None

    return runtime[dependency_name]


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

    # Write-ready products.
    # Key = logical product name from YAML, not filename.
    tables: dict[str, pd.DataFrame] = {}

    # Internal runtime products.
    # These may preserve indexes/pivot structure needed by downstream products.
    runtime: dict[str, pd.DataFrame] = {}

    with timed_workflow(logger, "HS Coverage audit"):

        # ------------------------------------------------------------------
        # STEP 1 — config and paths
        # ------------------------------------------------------------------
        n_step = 1
        with _workflow_step(n_step):
            cfg = load_hs_coverage_config(config_path)

            input_cfg = get_coverage_input_config(cfg)
            output_cfg = get_coverage_output_config(cfg)
            report_cfg = get_coverage_report_config(cfg)
            write_cfg = get_coverage_write_config(cfg)

            data_dir = _resolve_data_stage_dir(input_cfg)
            out_dir = _resolve_data_stage_dir(output_cfg)

            report_kind = get_coverage_report_output_kind(cfg)
            report_filename = get_coverage_report_filename(cfg)
            report_dir = get_output_dir(report_kind)

            pattern = get_coverage_input_data_pattern(cfg)
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
        n_step += 1
        with _workflow_step(n_step):
            product_name = "year_hs_counts"

            if is_coverage_product_enabled(product_name, cfg, default=True):
                year_columns = get_coverage_product_columns(
                    product_name,
                    cfg,
                    required=("year", "hs"),
                )

                year_col = year_columns["year"]
                hs_col = year_columns["hs"]

                if engine in {"duckdb", "polars"}:
                    logger.info("Using %s engine for data processing.", engine)
                    logger.info(
                        "%s query may take a while; progress is shown by step "
                        "timing, not per file.",
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

                runtime[product_name] = year_hs_counts
                tables[product_name] = year_hs_counts

                logger.info("Counts shape: %s", year_hs_counts.shape)

            else:
                logger.info("Skipping %s; disabled in config.", product_name)

        # ------------------------------------------------------------------
        # STEP 3 — derived year-HS summaries
        # ------------------------------------------------------------------
        n_step += 1
        with _workflow_step(n_step):
            year_columns = get_coverage_product_columns(
                "year_hs_counts",
                cfg,
                required=("year", "hs"),
            )

            year_col = year_columns["year"]
            hs_col = year_columns["hs"]

            # --------------------------------------------------------------
            # 3A — HS-system overall summary
            # --------------------------------------------------------------
            product_name = "hs_system_overall_summary"

            if is_coverage_product_enabled(product_name, cfg, default=True):
                source = _get_required_runtime_table(
                    product_name=product_name,
                    runtime=runtime,
                    cfg=cfg,
                )

                if source is not None:
                    overall_summary = build_hs_system_overall_summary(
                        source,
                        year_col=year_col,
                        hs_col=hs_col,
                    )

                    runtime[product_name] = overall_summary
                    tables[product_name] = overall_summary

                    logger.info(
                        "Overall summary shape: %s",
                        overall_summary.shape,
                    )

            else:
                logger.info("Skipping %s; disabled in config.", product_name)

            # --------------------------------------------------------------
            # 3B — year × HS count pivot
            # --------------------------------------------------------------
            product_name = "year_hs_count_pivot"

            if is_coverage_product_enabled(product_name, cfg, default=True):
                source = _get_required_runtime_table(
                    product_name=product_name,
                    runtime=runtime,
                    cfg=cfg,
                )

                if source is not None:
                    count_pivot = make_year_hs_count_pivot(
                        source,
                        year_col=year_col,
                        hs_col=hs_col,
                    )

                    # Runtime keeps the pivot index.
                    runtime[product_name] = count_pivot

                    # Tables are write-ready, so reset index.
                    tables[product_name] = count_pivot.reset_index()

                    logger.info("Count pivot shape: %s", count_pivot.shape)

            else:
                logger.info("Skipping %s; disabled in config.", product_name)

            # --------------------------------------------------------------
            # 3C — year × HS percentage pivot
            # --------------------------------------------------------------
            product_name = "year_hs_percentage_pivot"

            if is_coverage_product_enabled(product_name, cfg, default=True):
                source = _get_required_runtime_table(
                    product_name=product_name,
                    runtime=runtime,
                    cfg=cfg,
                )

                if source is not None:
                    percentage_pivot = make_year_hs_percentage_pivot(source)

                    # Runtime keeps the pivot index.
                    runtime[product_name] = percentage_pivot

                    # Tables are write-ready, so reset index.
                    tables[product_name] = percentage_pivot.reset_index()

                    logger.info(
                        "Percentage pivot shape: %s",
                        percentage_pivot.shape,
                    )

            else:
                logger.info("Skipping %s; disabled in config.", product_name)

            # --------------------------------------------------------------
            # 3D — year × HS mix summary
            # --------------------------------------------------------------
            product_name = "year_hs_mix_summary"

            if is_coverage_product_enabled(product_name, cfg, default=True):
                source = _get_required_runtime_table(
                    product_name=product_name,
                    runtime=runtime,
                    cfg=cfg,
                )

                if source is not None:
                    thresholds = get_coverage_product_thresholds(
                        product_name,
                        cfg,
                    )

                    mix_summary = build_year_hs_mix_summary(
                        source,
                        clean_threshold=float(thresholds.get("clean", 95.0)),
                        mostly_clean_threshold=float(
                            thresholds.get("mostly_clean", 90.0)
                        ),
                    )

                    runtime[product_name] = mix_summary
                    tables[product_name] = mix_summary

                    logger.info("Mix summary shape: %s", mix_summary.shape)

            else:
                logger.info("Skipping %s; disabled in config.", product_name)


        # ------------------------------------------------------------------
        # STEP 4 — Build year × HS × cmd4 coverage
        # ------------------------------------------------------------------
        n_step += 1
        with _workflow_step(n_step):
            base_product_name = "year_hs_cmd4_coverage"

            if is_coverage_product_enabled(base_product_name, cfg, default=False):
                columns = get_coverage_product_columns(
                    base_product_name,
                    cfg,
                    required=("year", "hs", "cmd"),)

                cmd4_coverage = build_year_hs_cmd4_coverage(
                    data_dir=data_dir,
                    pattern=pattern,
                    engine=engine,
                    year_col=columns["year"],
                    hs_col=columns["hs"],
                    cmd_col=columns["cmd"],
                    threads=threads,
                )

                runtime[base_product_name] = cmd4_coverage
                tables[base_product_name] = cmd4_coverage

                logger.info(
                    "Year × HS × cmd4 coverage shape: %s",
                    cmd4_coverage.shape,
                )

            else:
                logger.info("Skipping %s; disabled in config.", base_product_name)



            # ------------------------------------------------------------------
            # STEP 5 — Build derived cmd4 HS coverage summaries
            # ------------------------------------------------------------------
            n_step += 1
            with _workflow_step(n_step):
                base_product_name = "year_hs_cmd4_coverage"

            if base_product_name not in runtime:
                logger.warning(
                    "Skipping derived cmd4 HS summaries; %s was not built.",
                    base_product_name)

            else:
                # Reuse column mapping from the base product.
                columns = get_coverage_product_columns(
                        base_product_name,
                        cfg,
                        required=("year", "hs", "cmd"))
                year_col = columns["year"]
                hs_col = columns["hs"]

                # --------------------------------------------------------------
                # 5A — top cmd4 by year × HS
                # --------------------------------------------------------------
                product_name = "top_cmd4_by_year_hs"

                if is_coverage_product_enabled(product_name, cfg, default=True):
                    source = _get_required_runtime_table(
                                    product_name=product_name,
                                    runtime=runtime,
                                    cfg=cfg)

                    if source is not None:
                        top_n = get_coverage_product_top_n(
                            product_name,
                            cfg,
                            default=10)
                        
                        top_cmd4 = build_top_cmd4_by_year_hs(
                            source,
                            year_col=year_col,
                            hs_col=hs_col,
                            n=top_n)
                    
                        runtime[product_name] = top_cmd4
                        tables[product_name] = top_cmd4

                        logger.info("%s shape: %s", product_name, top_cmd4.shape)

                else:
                    logger.info("Skipping %s; disabled in config.", product_name)

                # --------------------------------------------------------------
                # 5B — cmd4 × year HS timeline
                # --------------------------------------------------------------
                product_name = "cmd4_hs_timeline"

                if is_coverage_product_enabled(product_name, cfg, default=True):
                    source = _get_required_runtime_table(
                                product_name=product_name,
                                runtime=runtime,
                                cfg=cfg)
            

                    if source is not None:
                        cmd4_timeline = build_cmd4_hs_timeline(
                                    source,
                                    year_col=year_col,
                                    hs_col=hs_col)
                

                    runtime[product_name] = cmd4_timeline
                    tables[product_name] = cmd4_timeline

                    logger.info("%s shape: %s", product_name, cmd4_timeline.shape)

                else:
                    logger.info("Skipping %s; disabled in config.", product_name)

                # --------------------------------------------------------------
                # 5C — cmd4 HS diversity summary
                # --------------------------------------------------------------
                product_name = "cmd4_hs_diversity_summary"

                if is_coverage_product_enabled(product_name, cfg, default=True):
                    source = _get_required_runtime_table(
                                product_name=product_name,
                                runtime=runtime,
                                cfg=cfg)
            

                    if source is not None:
                        diversity = build_cmd4_hs_diversity_summary(
                                        source,
                                        year_col=year_col,
                                        hs_col=hs_col)
                

                    runtime[product_name] = diversity
                    tables[product_name] = diversity

                    logger.info("%s shape: %s", product_name, diversity.shape)

                else:
                    logger.info("Skipping %s; disabled in config.", product_name)

                # --------------------------------------------------------------
                # 5D — cmd4 HS transition events
                # --------------------------------------------------------------
                product_name = "cmd4_hs_transition_events"

                if is_coverage_product_enabled(product_name, cfg, default=True):
                    source = _get_required_runtime_table(
                                product_name=product_name,
                                runtime=runtime,
                                cfg=cfg
                            )

                    if source is not None:
                        transitions = build_cmd4_hs_transition_events(
                                    source,
                                    year_col=year_col,
                                    hs_col=hs_col)
                

                        runtime[product_name] = transitions
                        tables[product_name] = transitions

                        logger.info("%s shape: %s", product_name, transitions.shape)

                else:
                    logger.info("Skipping %s; disabled in config.", product_name)
        

        # ------------------------------------------------------------------
        # STEP 6 — reporter × year × HS coverage
        # ------------------------------------------------------------------
        n_step += 1
        with _workflow_step(n_step):
            product_name = "reporter_year_hs_coverage"

            if is_coverage_product_enabled(product_name, cfg, default=False):
                reporter_columns = get_coverage_product_columns(
                    product_name,
                    cfg,
                    required=(
                        "reporter_code",
                        "reporter_iso",
                        "reporter_desc",
                        "year",
                        "hs",
                        "hs_search",
                        "original",
                        "cmd",
                        "flow",
                        "value",
                        "netweight",
                    ),
                )

                reporter_year_hs_coverage = build_reporter_year_hs_coverage(
                    data_dir=data_dir,
                    pattern=pattern,
                    engine=engine,
                    threads=threads,
                    columns=reporter_columns,
                )

                runtime[product_name] = reporter_year_hs_coverage
                tables[product_name] = reporter_year_hs_coverage

                logger.info(
                    "Reporter coverage shape: %s",
                    reporter_year_hs_coverage.shape,
                )

            else:
                logger.info("Skipping %s; disabled in config.", product_name)


        # ------------------------------------------------------------------
        # STEP 7 — Build country-level HS coverage summary tables
        # ------------------------------------------------------------------
        n_step += 1
        with _workflow_step(n_step):
            source_product = "reporter_year_hs_coverage"

            if source_product not in runtime:
                logger.warning(
                            "Skipping country-level HS coverage summaries; "
                            "%s was not built.",
                            source_product)
            else:
                source = runtime[source_product]

                summary_tables = build_hs_coverage_summary_tables(source)

                for product_name, table in summary_tables.items():
                    if is_coverage_product_enabled(product_name, cfg, default=True):
                        runtime[product_name] = table
                        tables[product_name] = table

                        logger.info("%s shape: %s",
                                    product_name,
                                    table.shape)
                    else:
                        logger.info(
                                    "Skipping %s; disabled in config.",
                                    product_name)


        # ------------------------------------------------------------------
        # STEP 8 — write machine-readable tables
        # ------------------------------------------------------------------
        n_step += 1
        with _workflow_step(n_step):
            overwrite = bool(write_cfg.get("overwrite", True))
            write_csv_enabled = bool(write_cfg.get("csv", True))
            write_parquet_enabled = bool(write_cfg.get("parquet", True))
            allow_empty = bool(write_cfg.get("allow_empty", True))

            if not tables:
                logger.warning("No tables were built; nothing to write.")

            for product_name, table in tables.items():
                filename = get_coverage_product_filename(product_name, cfg)

                if write_csv_enabled:
                    outputs[f"{product_name}_csv"] = write_csv(
                        table,
                        out_dir / f"{filename}.csv",
                        overwrite=overwrite,
                        allow_empty=allow_empty,
                    )

                if write_parquet_enabled:
                    outputs[f"{product_name}_parquet"] = write_parquet(
                        table,
                        out_dir / f"{filename}.parquet",
                        overwrite=overwrite,
                        allow_empty=allow_empty,
                    )


        # ------------------------------------------------------------------
        # STEP 9 — build and save human-readable report
        # ------------------------------------------------------------------
        n_step += 1
        with _workflow_step(n_step):
            if bool(report_cfg.get("enabled", True)):
                report_text = build_year_hs_coverage_report_text(
                    audit_info=audit_info,
                    overall_summary=runtime.get("hs_system_overall_summary"),
                    count_pivot=runtime.get("year_hs_count_pivot"),
                    percentage_pivot=runtime.get("year_hs_percentage_pivot"),
                    summary=runtime.get("year_hs_mix_summary"),
                    reporter_coverage=runtime.get("reporter_year_hs_coverage"),
                )

                report_base_path = report_dir / report_filename

                report_timestamp = bool(report_cfg.get("timestamp", True))
                write_latest = bool(report_cfg.get("write_latest", True))

                if report_timestamp:
                    report_path = make_timestamped_log_path(
                        report_base_path,
                        timestamp=run_id,
                    )
                else:
                    report_path = report_base_path

                report_path.write_text(report_text, encoding="utf-8")
                outputs["report"] = report_path

                if write_latest and report_timestamp:
                    latest_path = report_base_path.with_name(
                        f"{report_base_path.stem}_latest"
                        f"{report_base_path.suffix}"
                    )
                    latest_path.write_text(report_text, encoding="utf-8")
                    outputs["report_latest"] = latest_path

                    logger.info(
                        "Saved latest report copy: %s",
                        rel_path(latest_path),
                    )

                logger.info("Saved report: %s", rel_path(report_path))

            else:
                logger.info("Skipping text report; disabled in config.")

    outputs["output_dir"] = out_dir

    return outputs