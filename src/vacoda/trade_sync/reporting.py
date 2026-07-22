"""
vacoda.trade_sync.reporting

Create the final Trade Sync decision report and diagnostic figures.

Outputs
-------
- Timestamped Markdown decision report
- Optional latest Markdown copy
- Reconciliation-status figure
- Annual trade-value totals figure
- Country-reliability figure
- Agreement/discrepancy figure
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from vacoda.config.trade_sync import TradeSyncConfig
from vacoda.config.paths import get_figure_dir
from vacoda.utils.display import get_symbols


# ======================================================================
# Helpers
# ======================================================================


def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL literal."""
    return "'" + str(value).replace("'", "''") + "'"


def valid_output(path: Path) -> bool:
    """Return True when an existing output appears usable."""
    return path.is_file() and path.stat().st_size > 0


def _format_integer(value: float | int | None) -> str:
    if value is None:
        return "NA"

    return f"{int(round(float(value))):,}"


def _format_percent(
    value: float | None,
    *,
    digits: int = 2,
) -> str:
    if value is None:
        return "NA"

    return f"{100.0 * float(value):.{digits}f}%"


def _format_fraction_as_percent(
    value: float | None,
    *,
    digits: int = 2,
) -> str:
    return _format_percent(value, digits=digits)


def _format_usd(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"

    value = float(value)

    if abs(value) >= 1e12:
        return f"${value / 1e12:,.3f} trillion"

    if abs(value) >= 1e9:
        return f"${value / 1e9:,.3f} billion"

    if abs(value) >= 1e6:
        return f"${value / 1e6:,.3f} million"

    return f"${value:,.2f}"


def _markdown_table(
    dataframe: pd.DataFrame,
    *,
    index: bool = False,
) -> str:
    """Convert a DataFrame to Markdown."""
    if dataframe.empty:
        return "_No rows available._"

    return dataframe.to_markdown(
        index=index,
        floatfmt=".4f",
    )


def _metric_lookup(
    diagnostics: pd.DataFrame,
    *,
    scope: str,
    metric: str,
    year: int | None = None,
    status: str | None = None,
    direction: str | None = None,
) -> float | None:
    """Retrieve one long-format diagnostic metric."""

    mask = (
        diagnostics["diagnostic_scope"].eq(scope)
        & diagnostics["metric"].eq(metric)
    )

    if year is None:
        mask &= diagnostics["year"].isna()
    else:
        mask &= diagnostics["year"].eq(year)

    if status is None:
        mask &= diagnostics["reconciliation_status"].isna()
    else:
        mask &= diagnostics["reconciliation_status"].eq(
            status
        )

    if direction is None:
        mask &= diagnostics["reporting_direction"].isna()
    else:
        mask &= diagnostics["reporting_direction"].eq(
            direction
        )

    values = diagnostics.loc[mask, "metric_value"]

    if values.empty:
        return None

    value = values.iloc[0]

    if pd.isna(value):
        return None

    return float(value)


# ======================================================================
# Figures
# ======================================================================


def _plot_reconciliation_status_by_year(
    *,
    con: duckdb.DuckDBPyConnection,
    reconciled_sql: str,
    output_path: Path,
) -> None:
    """Plot annual row counts by reconciliation status."""

    frame = con.execute(
        f"""
        SELECT
            year,
            reconciliation_status,
            COUNT(*) AS n_rows

        FROM read_parquet({reconciled_sql})

        GROUP BY
            year,
            reconciliation_status

        ORDER BY
            year,
            reconciliation_status
        """
    ).fetchdf()

    pivot = frame.pivot(
        index="year",
        columns="reconciliation_status",
        values="n_rows",
    ).fillna(0)

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    pivot.plot(
        kind="bar",
        stacked=True,
        ax=axis,
    )

    axis.set_title(
        "Trade Sync reconciliation status by year"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("Number of bilateral product rows")
    axis.legend(
        title="Reconciliation status",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.grid(axis="x", linestyle="--", alpha=0.5)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_trade_value_totals_by_year(
    *,
    con: duckdb.DuckDBPyConnection,
    reconciled_sql: str,
    output_path: Path,
) -> None:
    """Compare annual exporter, importer and reconciled totals."""

    frame = con.execute(
        f"""
        SELECT
            year,

            SUM(export_value_fob)
                FILTER (
                    WHERE export_value_fob IS NOT NULL
                      AND isfinite(export_value_fob)
                      AND export_value_fob > 0
                ) AS exporter_fob_total,

            SUM(import_value_fob_corrected)
                FILTER (
                    WHERE import_value_fob_corrected IS NOT NULL
                      AND isfinite(import_value_fob_corrected)
                      AND import_value_fob_corrected > 0
                ) AS importer_fob_equivalent_total,

            SUM(trade_value_reconciled)
                FILTER (
                    WHERE trade_value_reconciled IS NOT NULL
                ) AS reconciled_total

        FROM read_parquet({reconciled_sql})

        GROUP BY year
        ORDER BY year
        """
    ).fetchdf()

    frame = frame.set_index("year") / 1e9

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    frame.plot(
        marker="o",
        ax=axis,
    )

    axis.set_title(
        "Annual trade-value totals before and after Trade Sync"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("Trade value, USD billions")
    axis.legend(
        labels=[
            "Exporter FOB reports",
            "Corrected importer FOB-equivalent reports",
            "Reconciled trade",
        ]
    )
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.grid(axis="x", linestyle="--", alpha=0.5)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_country_reliability_by_year(
    *,
    con: duckdb.DuckDBPyConnection,
    reliability_sql: str,
    output_path: Path,
) -> None:
    """Plot median country reliability by year and direction."""

    frame = con.execute(
        f"""
        SELECT
            year,
            reporting_direction,
            MEDIAN(reliability_index)
                AS median_reliability_index,
            QUANTILE_CONT(reliability_index, 0.25)
                AS p25_reliability_index,
            QUANTILE_CONT(reliability_index, 0.75)
                AS p75_reliability_index

        FROM read_parquet({reliability_sql})

        GROUP BY
            year,
            reporting_direction

        ORDER BY
            year,
            reporting_direction
        """
    ).fetchdf()

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    for direction, group in frame.groupby(
        "reporting_direction"
    ):
        group = group.sort_values("year")

        axis.plot(
            group["year"],
            group["median_reliability_index"],
            marker="o",
            label=f"{direction.title()} median",
        )

        axis.fill_between(
            group["year"],
            group["p25_reliability_index"],
            group["p75_reliability_index"],
            alpha=0.15,
        )

    axis.set_title(
        "Country reliability index by year"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel(
        "Reliability index: 1 − mean bilateral deviation"
    )
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.grid(axis="x", linestyle="--", alpha=0.5)
    axis.set_ylim(0, 1)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_agreement_and_discrepancy_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot agreement-tolerance share and median discrepancy."""

    year_metrics = diagnostics.loc[
        diagnostics["diagnostic_scope"].eq("year")
        & diagnostics["metric"].isin(
            [
                "within_agreement_tolerance_share",
                "median_relative_difference",
            ]
        )
    ].copy()

    pivot = year_metrics.pivot(
        index="year",
        columns="metric",
        values="metric_value",
    ).sort_index()

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    if "within_agreement_tolerance_share" in pivot:
        axis.plot(
            pivot.index,
            100
            * pivot["within_agreement_tolerance_share"],
            marker="o",
            label="Within 5% agreement",
        )

    if "median_relative_difference" in pivot:
        axis.plot(
            pivot.index,
            100
            * pivot["median_relative_difference"],
            marker="o",
            label="Median bilateral difference",
        )

    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.grid(axis="x", linestyle="--", alpha=0.5)
    axis.set_title(
        "Agreement and bilateral discrepancy by year"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("Share or relative difference, percent")
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


# ======================================================================
# Main reporting function
# ======================================================================


def build_trade_sync_report(
    *,
    cfg: TradeSyncConfig,
    reconciled_trade_path: Path,
    country_reliability_path: Path,
    reconciliation_audit_path: Path,
    diagnostics_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """Create the final Trade Sync Markdown decision report."""

    reconciled_trade_path = Path(reconciled_trade_path)
    country_reliability_path = Path(country_reliability_path)
    reconciliation_audit_path = Path(reconciliation_audit_path)
    diagnostics_path = Path(diagnostics_path)

    for label, path in (
        ("reconciled trade", reconciled_trade_path),
        ("country reliability", country_reliability_path),
        ("reconciliation audit", reconciliation_audit_path),
        ("Trade Sync diagnostics", diagnostics_path),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{label.capitalize()} not found: {path}"
            )

    if not cfg.artifacts.enable:
        return {
            "status": "disabled",
            "report_path": None,
            "latest_path": None,
            "ready_for_processed_export": False,
        }

    report_path = cfg.artifacts_path(run_id)
    latest_path = cfg.artifacts_latest_path()

    figure_dir = (
        get_figure_dir()
        / "trade_sync"
        / f"target_{cfg.scope.target_hs.lower()}"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_report_path: Path | None = None

    latest_report_path = latest_path

    if not force and valid_output(report_path):
        logger.info(
            "REUSE — Trade Sync report already exists: %s",
            report_path,
        )

        return {
            "status": "reused",
            "output_path": report_path,
            "report_path": report_path,
            "latest_report_path": latest_report_path,
            "ready_for_processed_export": True,
        }

    reconciled_sql = _sql_literal(
        reconciled_trade_path
    )
    reliability_sql = _sql_literal(
        country_reliability_path
    )
    audit_sql = _sql_literal(
        reconciliation_audit_path
    )
    diagnostics_sql = _sql_literal(
        diagnostics_path
    )

    info = get_symbols("info")
    ok = get_symbols("check")

    logger.info(
        "%s Building Trade Sync decision report.",
        info,
    )
    logger.info(
        "  Reconciled trade: %s",
        reconciled_trade_path,
    )
    logger.info(
        "  Diagnostics: %s",
        diagnostics_path,
    )
    logger.info(
        "  Report: %s",
        report_path,
    )
    logger.info(
        "  Figure directory: %s",
        figure_dir,
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        diagnostics = con.execute(
            f"""
            SELECT *
            FROM read_parquet({diagnostics_sql})
            """
        ).fetchdf()

        audit = con.execute(
            f"""
            SELECT *
            FROM read_parquet({audit_sql})
            """
        ).fetchdf()

        if len(audit) != 1:
            raise RuntimeError(
                "Reconciliation audit must contain exactly one row."
            )

        # --------------------------------------------------------------
        # Report-level values
        # --------------------------------------------------------------

        n_rows = _metric_lookup(
            diagnostics,
            scope="overall",
            metric="n_rows",
        )
        n_reconciled = _metric_lookup(
            diagnostics,
            scope="overall",
            metric="n_reconciled_value_rows",
        )
        n_weighted = _metric_lookup(
            diagnostics,
            scope="overall",
            metric="n_reliability_weighted_rows",
        )
        n_single_report = _metric_lookup(
            diagnostics,
            scope="overall",
            metric="n_single_report_retained_rows",
        )
        n_not_eligible = _metric_lookup(
            diagnostics,
            scope="overall",
            metric="n_not_eligible_rows",
        )
        total_reconciled = _metric_lookup(
            diagnostics,
            scope="overall",
            metric="total_reconciled_value",
        )
        total_export = _metric_lookup(
            diagnostics,
            scope="overall",
            metric="total_reported_export_fob",
        )
        total_import = _metric_lookup(
            diagnostics,
            scope="overall",
            metric="total_corrected_import_fob",
        )
        median_difference = _metric_lookup(
            diagnostics,
            scope="overall",
            metric="median_relative_difference",
        )
        p90_difference = _metric_lookup(
            diagnostics,
            scope="overall",
            metric="p90_relative_difference",
        )
        median_change_simple = _metric_lookup(
            diagnostics,
            scope="overall",
            metric=(
                "median_absolute_change_from_simple_average"
            ),
        )
        p90_change_simple = _metric_lookup(
            diagnostics,
            scope="overall",
            metric=(
                "p90_absolute_change_from_simple_average"
            ),
        )
        agreement_share = _metric_lookup(
            diagnostics,
            scope="overall",
            metric="within_agreement_tolerance_share",
        )

        usable_coverage = (
            n_reconciled / n_rows
            if n_rows
            else None
        )

        weighted_share = (
            n_weighted / n_rows
            if n_rows
            else None
        )

        single_report_share = (
            n_single_report / n_rows
            if n_rows
            else None
        )

        not_eligible_share = (
            n_not_eligible / n_rows
            if n_rows
            else None
        )

        # --------------------------------------------------------------
        # Summary tables
        # --------------------------------------------------------------

        status_summary = con.execute(
            f"""
            SELECT
                reconciliation_status AS status,
                COUNT(*) AS rows,
                100.0 * COUNT(*)
                    / SUM(COUNT(*)) OVER ()
                    AS share_percent,
                SUM(trade_value_reconciled)
                    FILTER (
                        WHERE trade_value_reconciled IS NOT NULL
                    ) AS reconciled_value

            FROM read_parquet({reconciled_sql})

            GROUP BY reconciliation_status

            ORDER BY rows DESC
            """
        ).fetchdf()

        status_summary["reconciled_value"] = (
            status_summary["reconciled_value"]
            .map(_format_usd)
        )

        annual_summary = con.execute(
            f"""
            SELECT
                year,
                COUNT(*) AS rows,
                COUNT(*) FILTER (
                    WHERE trade_value_reconciled IS NOT NULL
                ) AS rows_with_reconciled_value,
                SUM(trade_value_reconciled)
                    FILTER (
                        WHERE trade_value_reconciled IS NOT NULL
                    ) AS reconciled_value,
                MEDIAN(reliability_deviation)
                    FILTER (
                        WHERE reliability_evidence_case = 'both_positive'
                        AND reliability_deviation IS NOT NULL
                    ) AS median_positive_positive_deviation

            FROM read_parquet({reconciled_sql})

            GROUP BY year
            ORDER BY year
            """
        ).fetchdf()

        annual_summary["reconciled_value"] = (
            annual_summary["reconciled_value"]
            .map(_format_usd)
        )

        reliability_summary = con.execute(
            f"""
            SELECT
                reporting_direction,
                COUNT(*) AS country_year_scores,
                MEDIAN(n_evidence_observations)
                    AS median_evidence_rows,
                QUANTILE_CONT(
                    n_evidence_observations,
                    0.10
                ) AS p10_evidence_rows,
                MEDIAN(mean_reliability_deviation)
                    AS median_mean_deviation,
                MEDIAN(reliability_index)
                    AS median_reliability_index,
                MEDIAN(implicit_zero_share)
                    AS median_implicit_zero_share,
                COUNT(*) FILTER (
                    WHERE n_evidence_observations < 30
                ) AS scores_below_30_rows

            FROM read_parquet({reliability_sql})

            GROUP BY reporting_direction
            ORDER BY reporting_direction
            """
        ).fetchdf()

        # --------------------------------------------------------------
        # Figures
        # --------------------------------------------------------------

        status_figure = (
            figure_dir
            / "reconciliation_status_by_year.png"
        )
        totals_figure = (
            figure_dir
            / "trade_value_totals_by_year.png"
        )
        reliability_figure = (
            figure_dir
            / "country_reliability_by_year.png"
        )
        agreement_figure = (
            figure_dir
            / "agreement_and_discrepancy_by_year.png"
        )

        _plot_reconciliation_status_by_year(
            con=con,
            reconciled_sql=reconciled_sql,
            output_path=status_figure,
        )

        _plot_trade_value_totals_by_year(
            con=con,
            reconciled_sql=reconciled_sql,
            output_path=totals_figure,
        )

        _plot_country_reliability_by_year(
            con=con,
            reliability_sql=reliability_sql,
            output_path=reliability_figure,
        )

        _plot_agreement_and_discrepancy_by_year(
            diagnostics=diagnostics,
            output_path=agreement_figure,
        )

        # --------------------------------------------------------------
        # Decision logic
        # --------------------------------------------------------------

        audit_passed = bool(
            audit.iloc[0].get(
                "n_invalid_reconciled_values",
                0,
            )
            == 0
            and audit.iloc[0].get(
                "n_missing_expected_reconciled_values",
                0,
            )
            == 0
            and audit.iloc[0].get(
                "n_duplicate_keys",
                0,
            )
            == 0
        )

        ready_for_processed_export = bool(
            audit_passed
            and n_rows is not None
            and n_reconciled is not None
            and n_rows == n_reconciled + n_not_eligible
        )

        decision = (
            "APPROVED"
            if ready_for_processed_export
            else "REVIEW REQUIRED"
        )

        # --------------------------------------------------------------
        # Markdown
        # --------------------------------------------------------------

        lines = [
            "# Trade Sync Decision Report",
            "",
            f"**Decision:** `{decision}`  ",
            f"**Run ID:** `{run_id}`  ",
            f"**Target HS:** `{cfg.scope.target_hs}`  ",
            f"**Generated:** `{datetime.now().isoformat(timespec='seconds')}`",
            "",
            "## Executive summary",
            "",
            (
                f"Trade Sync processed **{_format_integer(n_rows)}** "
                "bilateral product-year rows. "
                f"A reconciled FOB-basis value is available for "
                f"**{_format_integer(n_reconciled)}** rows "
                f"({_format_percent(usable_coverage)})."
            ),
            "",
            (
                f"Reliability-weighted reconciliation was applied to "
                f"**{_format_integer(n_weighted)}** rows "
                f"({_format_percent(weighted_share)}), while "
                f"**{_format_integer(n_single_report)}** unilateral "
                f"reports were retained "
                f"({_format_percent(single_report_share)})."
            ),
            "",
            (
                f"**{_format_integer(n_not_eligible)}** rows "
                f"({_format_percent(not_eligible_share)}) had no usable "
                "FOB-basis trade value."
            ),
            "",
            (
                f"The final reconciled trade total is "
                f"**{_format_usd(total_reconciled)}**."
            ),
            "",
            "## Decision checks",
            "",
            "| Check | Result |",
            "|---|---:|",
            f"| Reconciliation audit passed | `{audit_passed}` |",
            f"| Full population preserved | `{_format_integer(n_rows)}` rows |",
            (
                "| Rows with reconciled value | "
                f"{_format_integer(n_reconciled)} |"
            ),
            (
                "| Rows not eligible | "
                f"{_format_integer(n_not_eligible)} |"
            ),
            (
                "| Ready for processed export | "
                f"`{ready_for_processed_export}` |"
            ),
            "",
            "## Input and reconciled totals",
            "",
            "| Measure | Value |",
            "|---|---:|",
            (
                "| Reported exporter FOB total | "
                f"{_format_usd(total_export)} |"
            ),
            (
                "| Corrected importer FOB-equivalent total | "
                f"{_format_usd(total_import)} |"
            ),
            (
                "| Final reconciled total | "
                f"{_format_usd(total_reconciled)} |"
            ),
            "",
            "## Bilateral discrepancy",
            "",
            (
                f"The median bilateral difference relative to the larger "
                f"report is **{_format_fraction_as_percent(median_difference)}**; "
                f"the 90th percentile is "
                f"**{_format_fraction_as_percent(p90_difference)}**."
            ),
            "",
            (
                "Relative to a simple bilateral average, the median "
                f"absolute reliability-weighting adjustment is "
                f"**{_format_fraction_as_percent(median_change_simple)}**, "
                f"and the 90th percentile is "
                f"**{_format_fraction_as_percent(p90_change_simple)}**."
            ),
            "",
            (
                "The share of comparable reports within the configured "
                f"5% agreement tolerance is "
                f"**{_format_fraction_as_percent(agreement_share)}**."
            ),
            "",
            "## Reconciliation composition",
            "",
            _markdown_table(status_summary),
            "",
            f"![Reconciliation status by year]({status_figure.as_posix()})",
            "",
            "## Annual results",
            "",
            _markdown_table(annual_summary),
            "",
            f"![Annual trade totals]({totals_figure.as_posix()})",
            "",
            "## Country reliability",
            "",
            (
                "Country reliability is estimated separately by year and "
                "reporting direction. The primary model quantity is mean "
                "bilateral deviation, where lower values indicate more "
                "consistent reporting. The derived reliability index is "
                "`1 − mean deviation`, where higher values indicate more "
                "consistent reporting."
            ),
            "",
            _markdown_table(reliability_summary),
            "",
            f"![Country reliability by year]({reliability_figure.as_posix()})",
            "",
            "## Agreement and discrepancy",
            "",
            f"![Agreement and discrepancy by year]({agreement_figure.as_posix()})",
            "",
            "## Methodological notes",
            "",
            (
                "- Positive-positive observations are reconciled using "
                "country-year-direction reliability weights."
            ),
            (
                "- Positive-versus-implicit-zero observations participate "
                "in country reliability estimation and reconciliation."
            ),
            (
                "- Genuinely missing partner reports are not interpreted "
                "as zeros."
            ),
            (
                "- Unilateral usable observations are retained because "
                "`retain_unmatched_flows` is enabled."
            ),
            (
                "- The configured agreement tolerance is currently used "
                "for diagnostics and does not bypass reliability-weighted "
                "reconciliation."
            ),
            "",
            "## Output artifacts",
            "",
            f"- Reconciled trade: `{reconciled_trade_path}`",
            f"- Country reliability: `{country_reliability_path}`",
            f"- Reconciliation audit: `{reconciliation_audit_path}`",
            f"- Diagnostics: `{diagnostics_path}`",
            f"- Report: `{report_path}`",
            "",
        ]

        report_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        if not valid_output(report_path):
            raise RuntimeError(
                "Trade Sync decision report was not written correctly."
            )

        if cfg.artifacts.write_latest:
            latest_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copyfile(
                report_path,
                latest_path,
            )

            latest_report_path = latest_path

        if not valid_output(report_path):
            raise RuntimeError(
                "Trade Sync decision report was not written correctly."
            )

        logger.info(
            "%s Trade Sync decision report created.",
            ok,
        )
        logger.info(
            "  Decision: %s",
            decision,
        )
        logger.info(
            "  Ready for processed export: %s",
            ready_for_processed_export,
        )
        logger.info(
            "  Report: %s",
            report_path,
        )

        if cfg.artifacts.write_latest:
            logger.info(
                "  Latest report: %s",
                latest_report_path,
            )

        logger.info(
            "  Figures written: 4"
        )

        return {
            "status": "completed",
            "output_path": report_path,
            "report_path": report_path,
            "latest_report_path": latest_report_path,
            "figure_paths": [
                status_figure,
                totals_figure,
                reliability_figure,
                agreement_figure,
            ],
            "decision": decision,
            "audit_passed": audit_passed,
            "ready_for_processed_export": (
                ready_for_processed_export
            ),
            "n_rows": int(round(n_rows or 0)),
            "n_reconciled_value_rows": int(
                round(n_reconciled or 0)
            ),
            "n_not_eligible_rows": int(
                round(n_not_eligible or 0)
            ),
            "total_reconciled_value": float(
                total_reconciled or 0.0
            ),
        }

    finally:
        con.close()