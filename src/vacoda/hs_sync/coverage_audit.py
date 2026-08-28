"""
vacoda.hs_sync.coverage_audit

End-to-end HS synchronization coverage diagnostics workflow.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from vacoda.config.hs_sync import HsSyncConfig

from vacoda.hs_sync.coverage import (
    # Source HS-system diagnostics
    build_year_hs_counts,
    build_hs_system_overall_summary,
    make_year_hs_count_pivot,
    make_year_hs_percentage_pivot,
    build_year_hs_mix_summary,
    build_reporter_year_hs_coverage,
    build_hs_coverage_summary_tables,

    # Target-HS mapping diagnostics
    build_target_hs_source_code_coverage,
    build_target_hs_mapping_coverage,
    summarize_target_hs_mapping_coverage,
    get_target_hs_missing_mappings,
    build_target_hs_mapping_interpretation_report,
    build_target_hs_mapping_interpretation_display_report,
)

from vacoda.hs_sync.harvard.diagnostics import build_trade_source_pair_summary
from vacoda.config.paths import rel_path
from vacoda.io.reports import TextReport, write_text_report
from vacoda.io.writer import write_csv, write_parquet
from vacoda.utils.logger import get_logger


#logger = get_logger("HS_SYNC.COVERAGE_AUDIT")


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

def build_target_hs_mapping_coverage_artifact(
    *,
    target_hs: str,
    min_year: int,
    max_year: int,
    interpretation_display_report: pd.DataFrame,
    source_code_coverage: pd.DataFrame,
    missing_mappings: pd.DataFrame,
    needs_weights_top_n: int = 10,
    unsupported_top_n: int = 30,) -> str:
    """Build the final human-facing target-HS coverage report."""
    report = TextReport()

    report.line(f"# Target-{target_hs} mapping coverage interpretation")
    report.line()
    report.line(
        f"For all {min_year}–{max_year} records, how much can be expressed "
        f"in the {target_hs}/HS2017 target system without weights?"
    )
    report.line()

    report.line("## Executive summary")
    report.line()
    report.markdown_dataframe(interpretation_display_report)

    # HS-system composition
    report.line()
    report.line("## HS-system composition of input data")
    hs_composition = (
        source_code_coverage.groupby("source_hs")
        .agg(
            n_rows=("n_rows", "sum"),
            n_source_codes=("source_cmd", "nunique"),
            first_year=("first_year", "min"),
            last_year=("last_year", "max"),
            value_total=("value_total", "sum"),
        )
        .reset_index()
        .sort_values("source_hs")
    )
    report.dataframe(hs_composition)

    # Unsupported diagnostics
    unsupported = source_code_coverage[
        source_code_coverage["conversion_need"] == "unsupported_source_hs"
    ].copy()

    if not unsupported.empty:
        report.line()
        report.line("## Unsupported source-HS diagnostics")

        by_source = (
            unsupported.groupby("source_hs")
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

        report.line("By source_hs:")
        report.dataframe(by_source)

        report.line()
        report.line(f"Top {unsupported_top_n} unsupported source codes by value:")
        report.dataframe(
            unsupported.sort_values("value_total", ascending=False)
            [
                [
                    "source_hs",
                    "source_cmd",
                    "first_year",
                    "last_year",
                    "n_rows",
                    "value_total",
                    "actual_mapping_category",
                    "conversion_need",
                ]
            ]
            .head(unsupported_top_n)
        )

    # Needs-weights diagnostics from old script 5
    report.line()
    report.line("## Needs-weights diagnostics")
    needs_text = build_needs_weights_inspection_report(
        source_code_coverage,
        top_n=needs_weights_top_n,
    )
    report.line(needs_text)

    # Consistency checks
    report.line()
    report.line("## Consistency checks")

    if {
        "share_source_codes_percent",
        "share_rows_percent",
        "share_value_percent",
    }.issubset(interpretation_display_report.columns):
        source_code_share_sum = interpretation_display_report[
            "share_source_codes_percent"
        ].sum()
        row_share_sum = interpretation_display_report["share_rows_percent"].sum()
        value_share_sum = interpretation_display_report["share_value_percent"].sum()

        report.line(f"Source-code pair share sum: {source_code_share_sum}")
        report.line(f"Row share sum: {row_share_sum}")
        report.line(f"Value share sum: {value_share_sum}")

    return report.render()


def load_target_hs_coverage_inputs(
    *,
    cfg: HsSyncConfig,
    logger: None | object = None,
) -> dict[str, object]:
    input_dir = cfg.coverage_input_dir()
    mapping_dir = cfg.coverage_mapping_dir()
    out_dir = cfg.coverage_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_format = cfg.coverage.input.format
    target_hs = cfg.coverage.target_hs
    min_year = cfg.coverage.min_year
    max_year = cfg.coverage.max_year

    logger.info("Coverage input directory: %s", rel_path(input_dir))
    logger.info("Coverage input format: %s", input_format)
    logger.info("Mapping directory: %s", rel_path(mapping_dir))
    logger.info("Output directory: %s", rel_path(out_dir))
    logger.info("Coverage target HS: %s", target_hs)
    logger.info("Coverage year range: %s–%s", min_year, max_year)

    source_summary = build_trade_source_pair_summary(
        data_dir=input_dir,
        pattern=cfg.coverage.input.pattern,
        input_format=input_format,
        engine=cfg.engine.name,
        year_col=cfg.columns.year,
        hs_col=cfg.columns.hs,
        cmd_col=cfg.columns.cmd,
        value_col=cfg.columns.value,
        min_year=min_year,
        threads=cfg.engine.threads,
        show_progress=cfg.progress.enabled,
    )

    if "year" in source_summary.columns:
        source_summary = source_summary[source_summary["year"] <= max_year]

    mappings: dict[str, pd.DataFrame] = {}
    for mapping_name, path in cfg.coverage_mapping_files().items():
        mappings[mapping_name] = pd.read_csv(path)
        logger.info(
            "Loaded mapping %s: %s shape=%s",
            mapping_name,
            rel_path(path),
            mappings[mapping_name].shape,
        )

    return {
        "input_dir": input_dir,
        "mapping_dir": mapping_dir,
        "output_dir": out_dir,
        "source_summary": source_summary,
        "mappings": mappings,
    }

def build_target_hs_coverage_products(
    *,
    cfg: HsSyncConfig,
    input_dir: Path,
    source_summary: pd.DataFrame,
    mappings: dict[str, pd.DataFrame],
    logger: None | object = None,
) -> dict[str, pd.DataFrame]:

    tables: dict[str, pd.DataFrame] = {}

    # ==============================================================
    # A. Source HS-system coverage
    # ==============================================================

    if cfg.is_product_enabled("year_hs_counts"):
        year_hs_counts = build_year_hs_counts(
            input_dir,
            pattern=cfg.coverage.input.pattern,
            input_format=cfg.coverage.input.format,
            engine=cfg.engine.name,
            year_col=cfg.columns.year,
            hs_col=cfg.columns.hs,
            threads=cfg.engine.threads,
            show_progress=cfg.progress.enabled,
        )

        tables["year_hs_counts"] = year_hs_counts

    if cfg.is_product_enabled(
        "hs_system_overall_summary"
    ):
        source = tables["year_hs_counts"]

        tables[
            "hs_system_overall_summary"
        ] = build_hs_system_overall_summary(
            source,
            year_col=cfg.columns.year,
            hs_col=cfg.columns.hs,
        )

    if cfg.is_product_enabled(
        "year_hs_count_pivot"
    ):
        source = tables["year_hs_counts"]

        tables[
            "year_hs_count_pivot"
        ] = make_year_hs_count_pivot(
            source,
            year_col=cfg.columns.year,
            hs_col=cfg.columns.hs,
        )

    if cfg.is_product_enabled(
        "year_hs_percentage_pivot"
    ):
        tables[
            "year_hs_percentage_pivot"
        ] = make_year_hs_percentage_pivot(
            tables["year_hs_count_pivot"]
        )

    if cfg.is_product_enabled(
        "year_hs_mix_summary"
    ):
        options = cfg.product_options(
            "year_hs_mix_summary"
        )

        tables[
            "year_hs_mix_summary"
        ] = build_year_hs_mix_summary(
            tables["year_hs_count_pivot"],
            clean_threshold=float(
                options.get(
                    "clean_threshold",
                    95.0,
                )
            ),
            mostly_clean_threshold=float(
                options.get(
                    "mostly_clean_threshold",
                    90.0,
                )
            ),
        )

    if cfg.is_product_enabled(
        "reporter_year_hs_coverage"
    ):
        reporter_coverage = (
            build_reporter_year_hs_coverage(
                input_dir,
                pattern=cfg.coverage.input.pattern,
                input_format=cfg.coverage.input.format,
                engine=cfg.engine.name,
                threads=cfg.engine.threads,
                columns={
                    "reporter_code": (
                        cfg.columns.reporter
                    ),
                    "reporter_iso": (
                        cfg.columns.reporter_iso
                    ),
                    "reporter_desc": (
                        cfg.columns.reporter_desc
                    ),
                    "year": (
                        cfg.columns.year
                    ),
                    "hs": (
                        cfg.columns.hs
                    ),
                    "hs_search": (
                        cfg.columns.hs_search
                    ),
                    "original": (
                        cfg.columns.is_original_hs
                    ),
                    "cmd": (
                        cfg.columns.cmd
                    ),
                    "flow": (
                        cfg.columns.flow
                    ),
                    "value": (
                        cfg.columns.value
                    ),
                    "netweight": (
                        cfg.columns.net_weight
                    ),
                },
            )
        )

        tables[
            "reporter_year_hs_coverage"
        ] = reporter_coverage

        summary_tables = (
            build_hs_coverage_summary_tables(
                reporter_coverage
            )
        )

        for name, table in summary_tables.items():
            if cfg.is_product_enabled(name):
                tables[name] = table

    # ==============================================================
    # B. Target-HS mapping coverage
    # ==============================================================

    target_hs = cfg.coverage.target_hs

    coverage = build_target_hs_mapping_coverage(
        source_summary=source_summary,
        mappings=mappings,
        target_hs=target_hs,
        mapping_columns=(
            cfg.coverage.mappings.columns.as_dict()
        ),
        easy_without_weights=(
            cfg.coverage.categories.easy_without_weights
        ),
        needs_weights=(
            cfg.coverage.categories.needs_weights
        ),
        missing_categories=(
            cfg.coverage.categories.missing
        ),
    )

    source_code_coverage = (
        build_target_hs_source_code_coverage(
            coverage
        )
    )

    summary = (
        summarize_target_hs_mapping_coverage(
            coverage
        )
    )

    interpretation_report = (
        build_target_hs_mapping_interpretation_report(
            source_code_coverage
        )
    )

    interpretation_display_report = (
        build_target_hs_mapping_interpretation_display_report(
            interpretation_report
        )
    )

    missing = get_target_hs_missing_mappings(
        coverage,
        top_n=cfg.product_options(
            "missing_mappings"
        ).get("top_n", 50),
    )

    needs_weights = coverage[
        coverage["conversion_need"]
        == "needs_weights"
    ].copy()

    easy = coverage[
        coverage["conversion_need"]
        == "easy_without_weights"
    ].copy()

    tables.update(
        {
            "source_code_coverage": (
                source_code_coverage
            ),
            "mapping_coverage": coverage,
            "mapping_summary": summary,
            "missing_mappings": missing,
            "needs_weights": needs_weights,
            "easy_without_weights": easy,
            "_mapping_interpretation_report": (
                interpretation_report
            ),
            "_mapping_interpretation_display_report": (
                interpretation_display_report
            ),
        }
    )

    if logger is not None:
        logger.info(
            "%s interpretation report:\n%s",
            target_hs,
            interpretation_report[
                [
                    "result_class",
                    "share_rows_percent",
                    "share_value_percent",
                    "interpretation",
                ]
            ].to_string(index=False),
        )

        for name, table in tables.items():
            logger.info(
                "%s shape: %s",
                name,
                table.shape,
            )

    return tables

def write_target_hs_coverage_outputs(
    *,
    cfg: HsSyncConfig,
    tables: dict[str, pd.DataFrame],
    run_id: str | None,
    logger: None | object = None,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    out_dir = cfg.coverage_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    overwrite = cfg.coverage.write.overwrite
    write_csv_enabled = cfg.coverage.write.csv
    write_parquet_enabled = cfg.coverage.write.parquet

    writable_tables = {
        name: table
        for name, table in tables.items()
        if not name.startswith("_")
    }

    for product_name, table in writable_tables.items():
        if not cfg.product_write_enabled(product_name, default=True):
            continue

        filename = cfg.product_filename(product_name)
        product_out_dir = cfg.product_output_dir(
            product_name,
            default_dir=out_dir,
        )
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

    artifact_dir = cfg.coverage.artifact.resolve_output_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    report_filename = cfg.coverage.artifact.filename
    if Path(report_filename).suffix == "":
        report_filename = f"{report_filename}.md"

    coverage_report = build_target_hs_mapping_coverage_artifact(
        target_hs=cfg.coverage.target_hs,
        min_year=cfg.coverage.min_year,
        max_year=cfg.coverage.max_year,
        interpretation_display_report=tables[
            "_mapping_interpretation_display_report"
        ],
        source_code_coverage=tables["source_code_coverage"],
        missing_mappings=tables["missing_mappings"],
        unsupported_top_n=cfg.coverage.artifact.options.get(
            "unsupported_top_n",
            30,
        ),
        needs_weights_top_n=cfg.coverage.artifact.options.get(
            "needs_weights_top_n",
            10,
        ),
    )

    report_outputs = write_text_report(
        coverage_report,
        report_dir=artifact_dir,
        filename=report_filename,
        timestamp=cfg.coverage.artifact.timestamp,
        write_latest=cfg.coverage.artifact.write_latest,
        run_id=run_id,
    )

    outputs["coverage_report"] = report_outputs["report"]
    if "latest" in report_outputs:
        outputs["coverage_report_latest"] = report_outputs["latest"]

    outputs["output_dir"] = out_dir
    outputs["artifact_dir"] = artifact_dir

    logger.info("Target-HS mapping coverage outputs written.")
    logger.info("  Output directory: %s", out_dir)
    logger.info("  Artifact directory: %s", artifact_dir)

    return outputs



def run_target_hs_mapping_coverage_audit(
    *,
    cfg: HsSyncConfig,
    run_id: str | None = None,
    logger: None | object = None,
) -> dict[str, object]:
    """
    Compatibility wrapper for target-HS mapping coverage diagnostics.

    This keeps older flows, especially the Harvard coverage flow, working
    after the main HS-sync workflow was refactored into explicit steps.

    New code should usually call:
        load_target_hs_coverage_inputs(...)
        build_target_hs_coverage_products(...)
        write_target_hs_coverage_outputs(...)

    directly from workflow.py.
    """

    log = logger or get_logger("HS_SYNC.COVERAGE_AUDIT")

    outputs: dict[str, object] = {
        "status": "started",
        "run_id": run_id,
    }

    coverage_inputs = load_target_hs_coverage_inputs(
        cfg=cfg,
        logger=log,
    )

    coverage_tables = build_target_hs_coverage_products(
        cfg=cfg,
        input_dir=coverage_inputs["input_dir"],
        source_summary=coverage_inputs["source_summary"],
        mappings=coverage_inputs["mappings"],
        logger=log,
    )

    coverage_outputs = write_target_hs_coverage_outputs(
        cfg=cfg,
        tables=coverage_tables,
        run_id=run_id,
        logger=log,
    )

    outputs.update(coverage_outputs)
    outputs["status"] = "completed"

    outputs["coverage_tables"] = {
        name: table.shape
        for name, table in coverage_tables.items()
    }

    return outputs