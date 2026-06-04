"""
vacoda.hs_sync.coverage_audit

End-to-end HS synchronization coverage diagnostics workflow.
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
from vacoda.io.reports import TextReport, write_text_report
from vacoda.io.writer import write_csv, write_parquet
from vacoda.utils.logger import get_logger, rel_path
from vacoda.utils.timing import timed_step, timed_workflow


logger = get_logger("HS_SYNC.COVERAGE_AUDIT")


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


def _format_value(value: float) -> str:
    """Format large values compactly for human-readable reports."""
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
            f"Input data is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )


def build_needs_weights_inspection_report(
    source_code_coverage: pd.DataFrame,
    *,
    top_n: int = 10) -> str:
    """Build a text report inspecting source code pairs that need weights."""
    report = TextReport()

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
    _require_columns(source_code_coverage, required)

    df = source_code_coverage.copy()
    needs = df[df["conversion_need"] == "needs_weights"].copy()

    target_hs_values = sorted(df["target_hs"].dropna().astype(str).unique())
    overall_first_year = int(df["first_year"].min())
    overall_last_year = int(df["last_year"].max())

    report.section("Needs-weights inspection")
    report.line(f"Target HS system(s): {', '.join(target_hs_values)}")
    report.line(f"Coverage period in full file: {overall_first_year}–{overall_last_year}")

    if needs.empty:
        report.line()
        report.line("No source HS-code pairs require weights.")
        return report.render()

    source_hs_values = sorted(needs["source_hs"].dropna().astype(str).unique())
    needs_first_year = int(needs["first_year"].min())
    needs_last_year = int(needs["last_year"].max())

    total_value = float(needs["value_total"].sum())
    total_rows = int(needs["n_rows"].sum())

    report.line(f"Period covered by needs-weights cases: {needs_first_year}–{needs_last_year}")
    report.line(f"Source HS systems requiring weights: {', '.join(source_hs_values)}")
    report.line()
    report.line(
        "Interpretation: each listed source HS-code pair maps to more than one "
        "target code in the configured target HS system. These cases cannot be "
        "converted deterministically without weights or another split rule."
    )
    report.line()
    report.line(f"Needs-weights source HS-code pairs: {len(needs):,}")
    report.line(f"Rows affected: {total_rows:,}")
    report.line(f"Value affected: {_format_value(total_value)}")

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

    report.line()
    report.line("By source_hs:")
    report.dataframe(by_source_display)

    needs_display = needs.sort_values("value_total", ascending=False).copy()
    needs_display["value"] = needs_display["value_total"].map(_format_value)
    needs_display["value_share_within_needs"] = (
        needs_display["value_total"] / total_value * 100
    ).map(lambda x: f"{x:.2f}%")

    report.line()
    report.line(f"Top {len(needs):,} needs-weights cases by value:")
    report.dataframe(
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
        ]
    )

    needs_sorted = needs.sort_values("value_total", ascending=False).copy()
    needs_sorted["cum_value"] = needs_sorted["value_total"].cumsum()
    needs_sorted["cum_value_share"] = needs_sorted["cum_value"] / total_value * 100
    needs_sorted["value"] = needs_sorted["value_total"].map(_format_value)
    needs_sorted["cum_value_share"] = needs_sorted["cum_value_share"].map(
        lambda x: f"{x:.2f}%"
    )

    report.line()
    report.line("Concentration:")
    report.dataframe(
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
        ].head(top_n)
    )

    top1 = needs_sorted.iloc[0]
    top_n_value_share = (
        float(needs_sorted.head(top_n)["value_total"].sum()) / total_value * 100
    )

    report.section("Interpretation")
    report.line(
        f"{len(needs):,} source HS-code pairs require weights to map into "
        f"{', '.join(target_hs_values)}."
    )
    report.line(
        f"The largest case is {top1['source_hs']} {top1['source_cmd']} "
        f"→ {top1['target_hs']} targets {top1['target_codes']}, "
        f"accounting for {float(top1['value_total']) / total_value * 100:.2f}% "
        "of the value requiring weights."
    )
    report.line(
        f"The top {min(top_n, len(needs))} cases account for "
        f"{top_n_value_share:.2f}% of the value requiring weights."
    )

    return report.render()


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
    runtime: dict[str, object] = {}
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
            
            runtime["mapping_coverage"] = coverage
            tables["mapping_coverage"] = coverage

            summary = summarize_target_hs_mapping_coverage(coverage)
            runtime["mapping_summary"] = summary
            tables["mapping_summary"] = summary

            interpretation_report = build_target_hs_mapping_interpretation_report(source_code_coverage)

            interpretation_display_report = (build_target_hs_mapping_interpretation_display_report(interpretation_report))
            runtime["mapping_interpretation_report"] = (interpretation_report)
            tables["mapping_interpretation_report"] = (interpretation_report)

            runtime["mapping_interpretation_display_report"] = interpretation_display_report
            tables["mapping_interpretation_display_report"] = interpretation_display_report

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

            if cfg.is_product_enabled("needs_weights_inspection", default=False):
                needs_weights_report = build_needs_weights_inspection_report(
                    source_code_coverage,
                    top_n=cfg.product_options("needs_weights_inspection").get("top_n", 10),
                )
                runtime["needs_weights_inspection"] = needs_weights_report

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

            if (
                "needs_weights_inspection" in runtime
                and cfg.product_write_enabled("needs_weights_inspection", default=True)
            ):
                report_filename = cfg.product_filename("needs_weights_inspection")
                if Path(report_filename).suffix == "":
                    report_filename = f"{report_filename}.md"

                report_out_dir = cfg.product_output_dir(
                    "needs_weights_inspection",
                    default_dir=out_dir,
                )
                report_outputs = write_text_report(
                    runtime["needs_weights_inspection"],
                    report_dir=report_out_dir,
                    filename=report_filename,
                    timestamp=cfg.coverage.report.timestamp,
                    write_latest=cfg.coverage.report.write_latest,
                    run_id=run_id,
                )

                outputs["needs_weights_inspection_report"] = report_outputs["report"]
                if "latest" in report_outputs:
                    outputs["needs_weights_inspection_latest"] = report_outputs["latest"]

    outputs["output_dir"] = out_dir
    return outputs