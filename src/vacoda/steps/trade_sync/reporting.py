"""
vacoda.steps.trade_sync.reporting

Create the final Trade Sync decision report and diagnostic figures.

Outputs
-------
- Timestamped Markdown decision report
- Optional latest Markdown copy
- Reconciliation-status figure
- Annual trade-value totals figure
- Country-reliability figure
- Positive-positive agreement/discrepancy figure
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any
import pycountry

import duckdb
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from vacoda.config.paths import get_figure_dir
from vacoda.utils.markdown import (
    insert_markdown_toc,
    markdown_image,
)
from vacoda.config.steps.trade_sync import TradeSyncConfig
from vacoda.utils.display import get_symbols


# ======================================================================
# Helpers
# ======================================================================

_COUNTRY_DISPLAY_NAME_OVERRIDES = {
    "BOL": "Bolivia",
    "HKG": "Hong Kong",
    "IRN": "Iran",
    "KOR": "South Korea",
    "LAO": "Laos",
    "MDA": "Moldova",
    "PRY": "Paraguay",
    "RUS": "Russia",
    "TZA": "Tanzania",
    "VEN": "Venezuela",
    "VNM": "Vietnam",
}

def _country_name_from_iso(country_iso: str) -> str:
    """Return a concise display name for an ISO alpha-3 code."""

    code = str(country_iso).upper()

    if code in _COUNTRY_DISPLAY_NAME_OVERRIDES:
        return _COUNTRY_DISPLAY_NAME_OVERRIDES[code]

    country = pycountry.countries.get(alpha_3=code)

    if country is None:
        return code

    return country.name

def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL literal."""
    return "'" + str(value).replace("'", "''") + "'"


def valid_output(path: Path) -> bool:
    """Return True when an existing output appears usable."""
    return path.is_file() and path.stat().st_size > 0


def _format_integer(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{int(round(float(value))):,}"


def _format_percent(
    value: float | None,
    *,
    digits: int = 2,
) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{100.0 * float(value):.{digits}f}%"


def _format_usd(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"

    numeric = float(value)

    if abs(numeric) >= 1e12:
        return f"${numeric / 1e12:,.3f} trillion"
    if abs(numeric) >= 1e9:
        return f"${numeric / 1e9:,.3f} billion"
    if abs(numeric) >= 1e6:
        return f"${numeric / 1e6:,.3f} million"

    return f"${numeric:,.2f}"


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

    mask &= (
        diagnostics["year"].isna()
        if year is None
        else diagnostics["year"].eq(year)
    )
    mask &= (
        diagnostics["reconciliation_status"].isna()
        if status is None
        else diagnostics["reconciliation_status"].eq(status)
    )
    mask &= (
        diagnostics["reporting_direction"].isna()
        if direction is None
        else diagnostics["reporting_direction"].eq(direction)
    )

    values = diagnostics.loc[mask, "metric_value"]

    if values.empty or pd.isna(values.iloc[0]):
        return None

    return float(values.iloc[0])


# ======================================================================
# Figures
# ======================================================================

def _plot_positive_positive_deviation_distribution(
    *,
    con: duckdb.DuckDBPyConnection,
    reconciled_sql: str,
    output_path: Path,
) -> None:
    """
    Plot the distribution of relative differences between genuine
    positive exporter and importer mirror reports.
    """

    frame = con.execute(
        f"""
        SELECT
            ABS(
                reconciliation_export_value
                - reconciliation_import_value
            )
            / GREATEST(
                reconciliation_export_value,
                reconciliation_import_value
            ) AS relative_difference
        FROM read_parquet({reconciled_sql})
        WHERE reliability_evidence_case = 'both_positive'
          AND reconciliation_export_value > 0
          AND reconciliation_import_value > 0
        """
    ).fetchdf()

    if frame.empty:
        raise RuntimeError(
            "No positive-positive observations are available for the "
            "deviation-distribution figure."
        )

    relative_difference = frame["relative_difference"].dropna()

    if relative_difference.empty:
        raise RuntimeError(
            "Positive-positive relative differences are all missing."
        )

    bin_edges = [index / 10 for index in range(11)]
    bin_labels = [
        "[0%, 10%]",
        "(10%, 20%]",
        "(20%, 30%]",
        "(30%, 40%]",
        "(40%, 50%]",
        "(50%, 60%]",
        "(60%, 70%]",
        "(70%, 80%]",
        "(80%, 90%]",
        "(90%, 100%]",
    ]

    counts = pd.cut(
        relative_difference.clip(lower=0, upper=1),
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=True,
    ).value_counts(sort=False)

    if int(counts.sum()) != len(relative_difference):
        raise RuntimeError(
            "Deviation-distribution bins do not account for all "
            "positive-positive observations."
        )

    shares = 100.0 * counts / counts.sum()

    figure, axis = plt.subplots(figsize=(11, 6))

    bars = axis.bar(
        shares.index.astype(str),
        shares.values,
        width=1.0,
        edgecolor="white",
        linewidth=0.5,
    )

    axis.bar_label(
        bars,
        labels=[f"{value:.1f}%" for value in shares],
        padding=3,
        fontsize=9,
    )

    axis.set_title(
        "Distribution of positive mirror-report differences\n"
        f"All years, n = {len(relative_difference):,}"
    )
    axis.set_xlabel(
        "Difference relative to the larger mirror report"
    )
    axis.set_ylabel("Share of rows (%)")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.set_axisbelow(True)

    axis.set_ylim(0, shares.max() * 1.12)
    axis.tick_params(
        axis="x",
        labelrotation=35,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_country_directional_reliability(
    *,
    con: duckdb.DuckDBPyConnection,
    reliability_sql: str,
    output_path: Path,
    year: int,
    minimum_evidence: int = 30,
    maximum_countries: int = 25,
) -> None:
    """
    Compare exporter and importer reliability by country for one year.

    Countries are selected by the largest absolute difference between
    exporter and importer reliability, subject to a minimum evidence
    threshold in both directions.
    """

    frame = con.execute(
        f"""
        WITH directional_scores AS (
            SELECT
                country_iso,
                reporting_direction,
                reliability_index,
                n_evidence_observations
            FROM read_parquet({reliability_sql})
            WHERE year = {int(year)}
              AND reliability_index IS NOT NULL
              AND isfinite(reliability_index)
              AND n_evidence_observations >= {int(minimum_evidence)}
              AND reporting_direction IN ('export', 'import')
        ),
        country_scores AS (
            SELECT
                country_iso,

                MAX(reliability_index) FILTER (
                    WHERE reporting_direction = 'export'
                ) AS export_reliability,

                MAX(reliability_index) FILTER (
                    WHERE reporting_direction = 'import'
                ) AS import_reliability,

                MAX(n_evidence_observations) FILTER (
                    WHERE reporting_direction = 'export'
                ) AS export_evidence,

                MAX(n_evidence_observations) FILTER (
                    WHERE reporting_direction = 'import'
                ) AS import_evidence

            FROM directional_scores
            GROUP BY country_iso
        )
        SELECT
            country_iso,
            export_reliability,
            import_reliability,
            export_evidence,
            import_evidence,
            ABS(
                export_reliability - import_reliability
            ) AS reliability_gap
        FROM country_scores
        WHERE export_reliability IS NOT NULL
          AND import_reliability IS NOT NULL
        ORDER BY reliability_gap DESC, country_iso
        LIMIT {int(maximum_countries)}
        """
    ).fetchdf()

    frame["country_name"] = frame["country_iso"].map(
        _country_name_from_iso
    )
    if frame.empty:
        raise RuntimeError(
            "No countries have sufficient exporter and importer "
            f"reliability evidence for {year}."
        )

    frame = frame.sort_values(
        ["reliability_gap", "country_iso"],
        ascending=[True, True],
    ).reset_index(drop=True)

    y_positions = range(len(frame))

    figure_height = max(
        6.0,
        0.34 * len(frame) + 1.8,
    )

    figure, axis = plt.subplots(
        figsize=(11, figure_height)
    )

    for row_number, row in frame.iterrows():
        axis.plot(
            [
                row["export_reliability"],
                row["import_reliability"],
            ],
            [row_number, row_number],
            color="0.70",
            linewidth=1.4,
            zorder=1,
        )

    axis.scatter(
        frame["export_reliability"],
        list(y_positions),
        marker="o",
        s=42,
        label="Exporter reliability",
        zorder=3,
    )

    axis.scatter(
        frame["import_reliability"],
        list(y_positions),
        marker="s",
        s=42,
        label="Importer reliability",
        zorder=3,
    )

    axis.set_yticks(list(y_positions))
    axis.set_yticklabels(frame["country_name"])

    axis.set_title(
        "Largest exporter–importer reliability gaps by country\n"
        f"{year}, top {len(frame)}, minimum "
        f"{minimum_evidence:,} evidence rows per direction"
    )
    axis.set_xlabel("Reliability index")
    axis.set_ylabel("Reporting country")
    axis.set_xlim(0, 1)
    axis.grid(axis="x", linestyle="--", alpha=0.5)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

def _plot_selected_country_reliability_over_time(
    *,
    con: duckdb.DuckDBPyConnection,
    reliability_sql: str,
    output_path: Path,
    reporting_direction: str,
    maximum_year: int,
    minimum_evidence: int = 30,
    minimum_years: int = 6,
    maximum_countries: int = 8,
) -> None:
    """
    Plot reliability trajectories for countries with the greatest
    temporal variation in one reporting direction.
    """

    if reporting_direction not in {"export", "import"}:
        raise ValueError(
            "reporting_direction must be 'export' or 'import'."
        )

    frame = con.execute(
        f"""
        WITH eligible_scores AS (
            SELECT
                year,
                country_iso,
                reliability_index
            FROM read_parquet({reliability_sql})
            WHERE reporting_direction = '{reporting_direction}'
              AND year <= {int(maximum_year)}
              AND reliability_index IS NOT NULL
              AND isfinite(reliability_index)
              AND n_evidence_observations >= {int(minimum_evidence)}
        ),
        country_variation AS (
            SELECT
                country_iso,
                COUNT(*) AS n_years,
                STDDEV_SAMP(reliability_index)
                    AS reliability_stddev,
                MAX(reliability_index)
                    - MIN(reliability_index)
                    AS reliability_range
            FROM eligible_scores
            GROUP BY country_iso
            HAVING COUNT(*) >= {int(minimum_years)}
        ),
        selected_countries AS (
            SELECT country_iso
            FROM country_variation
            ORDER BY
                reliability_stddev DESC,
                reliability_range DESC,
                country_iso
            LIMIT {int(maximum_countries)}
        )
        SELECT
            scores.year,
            scores.country_iso,
            scores.reliability_index
        FROM eligible_scores AS scores
        INNER JOIN selected_countries AS selected
            USING (country_iso)
        ORDER BY
            scores.country_iso,
            scores.year
        """
    ).fetchdf()

    if frame.empty:
        raise RuntimeError(
            "No countries have sufficient reliability history for "
            f"the {reporting_direction} trajectory figure."
        )

    frame["country_name"] = frame["country_iso"].map(
        _country_name_from_iso
    )

    figure, axis = plt.subplots(figsize=(11, 7))

    for country_name, group in frame.groupby(
        "country_name",
        sort=True,
    ):
        group = group.sort_values("year")

        axis.plot(
            group["year"],
            group["reliability_index"],
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=country_name,
        )

    minimum_year = int(frame["year"].min())
    maximum_year = int(frame["year"].max())

    axis.set_title(
        f"Selected-country {reporting_direction} reliability over time\n"
        "Countries with the greatest temporal variation"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("Reliability index")
    axis.set_ylim(0, 1)
    axis.set_xticks(
        range(minimum_year, maximum_year + 1)
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )
    axis.set_axisbelow(True)

    axis.legend(
        title="Reporting country",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

def _plot_country_reliability_level_vs_gap(
    *,
    con: duckdb.DuckDBPyConnection,
    reliability_sql: str,
    output_path: Path,
    year: int,
    minimum_evidence: int = 30,
    marker_size = 8,
    line_alpha = 0.9,
    line_width = 0.4,
) -> None:
    """
    Plot average directional reliability against the absolute
    exporter-importer reliability gap for each reporting country.
    """

    frame = con.execute(
        f"""
        WITH directional_scores AS (
            SELECT
                country_iso,
                reporting_direction,
                reliability_index,
                n_evidence_observations
            FROM read_parquet({reliability_sql})
            WHERE year = {int(year)}
              AND reporting_direction IN ('export', 'import')
              AND reliability_index IS NOT NULL
              AND isfinite(reliability_index)
              AND n_evidence_observations >= {int(minimum_evidence)}
        ),
        country_scores AS (
            SELECT
                country_iso,

                MAX(reliability_index) FILTER (
                    WHERE reporting_direction = 'export'
                ) AS export_reliability,

                MAX(reliability_index) FILTER (
                    WHERE reporting_direction = 'import'
                ) AS import_reliability,

                MAX(n_evidence_observations) FILTER (
                    WHERE reporting_direction = 'export'
                ) AS export_evidence,

                MAX(n_evidence_observations) FILTER (
                    WHERE reporting_direction = 'import'
                ) AS import_evidence

            FROM directional_scores
            GROUP BY country_iso
        )
        SELECT
            country_iso,
            export_reliability,
            import_reliability,
            export_evidence,
            import_evidence,
            (
                export_reliability + import_reliability
            ) / 2.0 AS mean_reliability,
            ABS(
                export_reliability - import_reliability
            ) AS reliability_gap
        FROM country_scores
        WHERE export_reliability IS NOT NULL
          AND import_reliability IS NOT NULL
        ORDER BY country_iso
        """
    ).fetchdf()

    if frame.empty:
        raise RuntimeError(
            "No countries have sufficient exporter and importer "
            f"reliability evidence for {year}."
        )

    figure, axis = plt.subplots(figsize=(10, 7))

    density = axis.hexbin(
        frame["mean_reliability"],
        frame["reliability_gap"],
        gridsize=18,
        mincnt=1,
        linewidths=0.4,
        cmap="YlGnBu",
        alpha=0.95
    )

    maximum_count = int(density.get_array().max())

    colorbar = figure.colorbar(
        density,
        ax=axis,
        pad=0.02,
        ticks=range(1, maximum_count + 1),
    )
    colorbar.set_label("Number of reporting countries")
    colorbar.ax.yaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    largest_gap_codes = set(
        frame.nlargest(5, "reliability_gap")["country_iso"]
    )
    lowest_reliability_codes = set(
        frame.nsmallest(5, "mean_reliability")["country_iso"]
    )
    highest_reliability_codes = set(
        frame.nlargest(5, "mean_reliability")["country_iso"]
    )

    label_codes = (
        largest_gap_codes
        | lowest_reliability_codes
        | highest_reliability_codes
    )

    label_frame = frame[
        frame["country_iso"].isin(label_codes)
    ].copy() 

    label_frame["country_name"] = label_frame[
        "country_iso"
    ].map(_country_name_from_iso)

    axis.scatter(
        label_frame["mean_reliability"],
        label_frame["reliability_gap"],
        marker="o",
        facecolors="none",
        edgecolors="black",
        s=marker_size,
        linewidths=line_width,
        zorder=3,
    )

    for row in label_frame.itertuples(index=False):
        axis.annotate(
            row.country_name,
            (
                row.mean_reliability,
                row.reliability_gap,
            ),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            zorder=4,
        )

    median_mean_reliability = float(
        frame["mean_reliability"].median()
    )
    median_gap = float(
        frame["reliability_gap"].median()
    )

    axis.axvline(
        median_mean_reliability,
        linestyle=":",
        linewidth=1.4,
        alpha=line_alpha,
    )

    axis.axhline(
        median_gap,
        linestyle=":",
        linewidth=1.4,
        alpha=line_alpha,
    )

    axis.set_title(
        "Country reliability level and directional imbalance\n"
        f"{year}, minimum {minimum_evidence:,} evidence rows "
        "per direction"
    )
    axis.set_xlabel(
        "Mean exporter-importer reliability index"
    )
    axis.set_ylabel(
        "Absolute exporter-importer reliability gap"
    )

    axis.set_xlim(0, 1)
    axis.set_ylim(
        0,
        max(
            0.05,
            float(frame["reliability_gap"].max()) * 1.12,
        ),
    )

    axis.grid(
        linestyle="--",
        alpha=0.35,
    )
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

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
        GROUP BY year, reconciliation_status
        ORDER BY year, reconciliation_status
        """
    ).fetchdf()

    pivot = frame.pivot(
        index="year",
        columns="reconciliation_status",
        values="n_rows",
    ).fillna(0)

    figure, axis = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", stacked=True, ax=axis)

    axis.set_title("Trade Sync reconciliation status by year")
    axis.set_xlabel("Year")
    axis.set_ylabel("Number of bilateral product rows")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.legend(
        title="Reconciliation status",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
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
            SUM(export_value_fob) FILTER (
                WHERE export_value_fob IS NOT NULL
                  AND isfinite(export_value_fob)
                  AND export_value_fob > 0
            ) AS exporter_fob_total,
            SUM(import_value_fob_corrected) FILTER (
                WHERE import_value_fob_corrected IS NOT NULL
                  AND isfinite(import_value_fob_corrected)
                  AND import_value_fob_corrected > 0
            ) AS importer_fob_equivalent_total,
            SUM(trade_value_reconciled) FILTER (
                WHERE trade_value_reconciled IS NOT NULL
            ) AS reconciled_total
        FROM read_parquet({reconciled_sql})
        GROUP BY year
        ORDER BY year
        """
    ).fetchdf()

    frame = frame.set_index("year") / 1e9

    figure, axis = plt.subplots(figsize=(10, 6))
    frame.plot(marker="o", ax=axis)

    axis.set_title(
        "Annual reported and Trade Sync totals (different coverage)"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("Trade value, USD billions")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.legend(
        labels=[
            "Exporter FOB reports",
            "Importer FOB-equivalent reports",
            "Reconciled trade",
        ]
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
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
            MEDIAN(reliability_index) AS median_reliability_index,
            QUANTILE_CONT(reliability_index, 0.25)
                AS p25_reliability_index,
            QUANTILE_CONT(reliability_index, 0.75)
                AS p75_reliability_index
        FROM read_parquet({reliability_sql})
        GROUP BY year, reporting_direction
        ORDER BY year, reporting_direction
        """
    ).fetchdf()

    figure, axis = plt.subplots(figsize=(10, 6))

    for direction, group in frame.groupby("reporting_direction"):
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

    axis.set_title("Country reliability index by year")
    axis.set_xlabel("Year")
    axis.set_ylabel("Reliability index: 1 − mean evidence deviation")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_positive_positive_agreement_by_year(
    *,
    con: duckdb.DuckDBPyConnection,
    reconciled_sql: str,
    agreement_tolerance: float,
    output_path: Path,
) -> None:
    """
    Plot agreement and discrepancy using genuine positive-positive
    comparisons only.
    """

    frame = con.execute(
        f"""
        WITH positive_pairs AS (
            SELECT
                year,
                ABS(
                    reconciliation_export_value
                    - reconciliation_import_value
                )
                / GREATEST(
                    reconciliation_export_value,
                    reconciliation_import_value
                ) AS relative_difference
            FROM read_parquet({reconciled_sql})
            WHERE reliability_evidence_case = 'both_positive'
              AND reconciliation_export_value > 0
              AND reconciliation_import_value > 0
        )
        SELECT
            year,
            MEDIAN(relative_difference)
                AS median_relative_difference,
            AVG(
                CASE
                    WHEN relative_difference <= {float(agreement_tolerance)}
                        THEN 1.0
                    ELSE 0.0
                END
            ) AS agreement_share
        FROM positive_pairs
        GROUP BY year
        ORDER BY year
        """
    ).fetchdf()

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        frame["year"],
        100 * frame["agreement_share"],
        marker="o",
        label=(
            f"Within {100 * agreement_tolerance:.0f}% agreement"
        ),
    )
    axis.plot(
        frame["year"],
        100 * frame["median_relative_difference"],
        marker="o",
        label="Median positive-positive difference",
    )

    axis.set_title(
        "Positive-report agreement and discrepancy by year"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("Share or relative difference, percent")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
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
            "latest_report_path": None,
            "ready_for_processed_export": False,
        }

    report_path = cfg.artifacts_path(run_id)
    latest_path = cfg.artifacts_latest_path()

    figure_dir = (
        get_figure_dir()
        / "trade_sync"
        / f"target_{cfg.scope.target_hs.lower()}"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not force and valid_output(report_path):
        logger.info(
            "REUSE — Trade Sync report already exists: %s",
            report_path,
        )

        return {
            "status": "reused",
            "output_path": report_path,
            "report_path": report_path,
            "latest_report_path": (
                latest_path if latest_path.exists() else None
            ),
            "ready_for_processed_export": True,
        }

    reconciled_sql = _sql_literal(reconciled_trade_path)
    reliability_sql = _sql_literal(country_reliability_path)
    audit_sql = _sql_literal(reconciliation_audit_path)
    diagnostics_sql = _sql_literal(diagnostics_path)

    agreement_tolerance = float(
        cfg.diagnostics.agreement_tolerance.relative_difference
    )

    info = get_symbols("info")
    ok = get_symbols("check")

    logger.info(
        "%s Building Trade Sync decision report.",
        info,
    )
    logger.info("  Reconciled trade: %s", reconciled_trade_path)
    logger.info("  Diagnostics: %s", diagnostics_path)
    logger.info("  Report: %s", report_path)
    logger.info("  Figure directory: %s", figure_dir)

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(cfg.engine.threads)}")

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

        required_audit_columns = {
            "n_invalid_reconciled_values",
            "n_missing_expected_reconciled_values",
            "n_duplicate_keys",
        }
        missing_audit_columns = (
            required_audit_columns - set(audit.columns)
        )

        if missing_audit_columns:
            raise RuntimeError(
                "Reconciliation audit is missing required columns: "
                + ", ".join(sorted(missing_audit_columns))
            )

        # --------------------------------------------------------------
        # Core diagnostic totals
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

        # Compute agreement measures directly from genuine two-sided
        # positive reports. This avoids mixing implicit-zero evidence
        # into descriptive report-agreement statistics.
        discrepancy_row = con.execute(
            f"""
            WITH positive_pairs AS (
                SELECT
                    ABS(
                        reconciliation_export_value
                        - reconciliation_import_value
                    )
                    / GREATEST(
                        reconciliation_export_value,
                        reconciliation_import_value
                    ) AS relative_difference,

                    ABS(
                        trade_value_reconciled
                        - (
                            reconciliation_export_value
                            + reconciliation_import_value
                        ) / 2.0
                    )
                    / (
                        (
                            reconciliation_export_value
                            + reconciliation_import_value
                        ) / 2.0
                    ) AS absolute_change_from_simple_average

                FROM read_parquet({reconciled_sql})

                WHERE reliability_evidence_case = 'both_positive'
                  AND reconciliation_export_value > 0
                  AND reconciliation_import_value > 0
            )
            SELECT
                COUNT(*) AS n_positive_positive_rows,
                MEDIAN(relative_difference)
                    AS median_relative_difference,
                QUANTILE_CONT(relative_difference, 0.90)
                    AS p90_relative_difference,
                MEDIAN(absolute_change_from_simple_average)
                    AS median_absolute_change_from_simple_average,
                QUANTILE_CONT(
                    absolute_change_from_simple_average,
                    0.90
                ) AS p90_absolute_change_from_simple_average,
                AVG(
                    CASE
                        WHEN relative_difference
                            <= {agreement_tolerance}
                            THEN 1.0
                        ELSE 0.0
                    END
                ) AS agreement_share
            FROM positive_pairs
            """
        ).fetchone()

        if discrepancy_row is None:
            raise RuntimeError(
                "Could not calculate positive-positive discrepancy "
                "statistics."
            )

        (
            n_positive_positive,
            median_difference,
            p90_difference,
            median_change_simple,
            p90_change_simple,
            agreement_share,
        ) = discrepancy_row

        evidence_row = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE reliability_evidence_case IS NOT NULL
                ) AS n_evidence_rows,
                COUNT(*) FILTER (
                    WHERE reliability_evidence_case IN (
                        'export_positive_import_implicit_zero',
                        'import_positive_export_implicit_zero'
                    )
                ) AS n_implicit_zero_rows
            FROM read_parquet({reconciled_sql})
            """
        ).fetchone()

        if evidence_row is None:
            raise RuntimeError(
                "Could not calculate reliability-evidence composition."
            )

        n_evidence_rows = int(evidence_row[0] or 0)
        n_implicit_zero_rows = int(evidence_row[1] or 0)
        implicit_zero_share = (
            n_implicit_zero_rows / n_evidence_rows
            if n_evidence_rows
            else None
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
                SUM(trade_value_reconciled) FILTER (
                    WHERE trade_value_reconciled IS NOT NULL
                ) AS reconciled_value
            FROM read_parquet({reconciled_sql})
            GROUP BY reconciliation_status
            ORDER BY rows DESC
            """
        ).fetchdf()

        status_summary["reconciled_value"] = (
            status_summary["reconciled_value"].map(_format_usd)
        )

        annual_summary = con.execute(
            f"""
            SELECT
                year,
                COUNT(*) AS rows,
                COUNT(*) FILTER (
                    WHERE trade_value_reconciled IS NOT NULL
                ) AS rows_with_reconciled_value,
                SUM(trade_value_reconciled) FILTER (
                    WHERE trade_value_reconciled IS NOT NULL
                ) AS reconciled_value,
                MEDIAN(reliability_deviation) FILTER (
                    WHERE reliability_evidence_case = 'both_positive'
                      AND reliability_deviation IS NOT NULL
                ) AS median_positive_positive_deviation,
                AVG(
                    CASE
                        WHEN reliability_evidence_case IN (
                            'export_positive_import_implicit_zero',
                            'import_positive_export_implicit_zero'
                        )
                            THEN 1.0
                        WHEN reliability_evidence_case IS NOT NULL
                            THEN 0.0
                        ELSE NULL
                    END
                ) AS implicit_zero_evidence_share
            FROM read_parquet({reconciled_sql})
            GROUP BY year
            ORDER BY year
            """
        ).fetchdf()

        annual_summary["reconciled_value"] = (
            annual_summary["reconciled_value"].map(_format_usd)
        )

        annual_summary[
            "median_positive_positive_deviation"
        ] = annual_summary[
            "median_positive_positive_deviation"
        ].map(
            lambda value: (
                _format_percent(value)
                if not pd.isna(value)
                else "NA"
            )
        )

        annual_summary[
            "implicit_zero_evidence_share"
        ] = annual_summary[
            "implicit_zero_evidence_share"
        ].map(
            lambda value: (
                _format_percent(value)
                if not pd.isna(value)
                else "NA"
            )
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

        # Flag a substantially incomplete final year.
        annual_rows_numeric = con.execute(
            f"""
            SELECT year, COUNT(*) AS rows
            FROM read_parquet({reconciled_sql})
            GROUP BY year
            ORDER BY year
            """
        ).fetchdf()

        maximum_year = int(annual_rows_numeric["year"].max())
        latest_rows = int(
            annual_rows_numeric.loc[
                annual_rows_numeric["year"].eq(maximum_year),
                "rows",
            ].iloc[0]
        )
        previous_rows_median = float(
            annual_rows_numeric.loc[
                annual_rows_numeric["year"].lt(maximum_year),
                "rows",
            ].median()
        )

        latest_year_provisional = bool(
            previous_rows_median > 0
            and latest_rows < 0.75 * previous_rows_median
        )

        available_reliability_years = con.execute(
            f"""
            SELECT DISTINCT year
            FROM read_parquet({reliability_sql})
            WHERE reliability_index IS NOT NULL
            ORDER BY year
            """
        ).fetchdf()["year"].astype(int).tolist()

        preferred_year = (
            maximum_year - 1
            if latest_year_provisional
            else maximum_year
        )

        eligible_years = [
            year
            for year in available_reliability_years
            if year <= preferred_year
        ]

        if not eligible_years:
            raise RuntimeError(
                "No reliability year is available for the country comparison."
            )

        country_reliability_plot_year = max(eligible_years)


        treatment_definitions = pd.DataFrame(
            [
                {
                    "Input situation": (
                        "Exporter value positive; importer FOB-equivalent "
                        "value positive"
                    ),
                    "Status": "bilateral_reconciled",
                    "Trade Sync treatment": (
                        "Combine both reports using mean-deviation-derived "
                        "weights with the paper's 0.10 import-report preference."
                    ),
                    "Output": "Reliability-weighted reconciled value",
                },
                {
                    "Input situation": (
                        "Exporter side classified as implicit zero; "
                        "importer FOB-equivalent value positive"
                    ),
                    "Status": (
                        "bilateral_reconciled_export_implicit_zero"
                    ),
                    "Trade Sync treatment": (
                        "Reconcile the positive importer report against "
                        "zero evidence using the final adjusted weights."
                    ),
                    "Output": "Reliability-weighted reconciled value",
                },
                {
                    "Input situation": (
                        "Exporter value positive; importer side "
                        "classified as implicit zero"
                    ),
                    "Status": (
                        "bilateral_reconciled_import_implicit_zero"
                    ),
                    "Trade Sync treatment": (
                        "Reconcile the positive exporter report against "
                        "zero evidence using the final adjusted weights."
                    ),
                    "Output": "Reliability-weighted reconciled value",
                },
                {
                    "Input situation": (
                        "Usable exporter value; no usable or expected "
                        "importer mirror value"
                    ),
                    "Status": "export_only",
                    "Trade Sync treatment": (
                        "Retain the exporter-reported FOB value without "
                        "bilateral weighting."
                    ),
                    "Output": "Exporter value retained",
                },
                {
                    "Input situation": (
                        "Usable importer FOB-equivalent value; no usable or "
                        "expected exporter mirror value"
                    ),
                    "Status": "import_only",
                    "Trade Sync treatment": (
                        "Retain the importer FOB-equivalent value "
                        "without bilateral weighting."
                    ),
                    "Output": "Importer FOB-equivalent value retained",
                },
                {
                    "Input situation": (
                        "Neither side provides a usable FOB-basis value"
                    ),
                    "Status": "not_eligible",
                    "Trade Sync treatment": (
                        "Preserve the row but leave the reconciled trade "
                        "value missing."
                    ),
                    "Output": "No reconciled value",
                },
            ]
        )

        population_accounting = pd.DataFrame(
            [
                {
                    "Population component": (
                        "Reliability-weighted reconciliation"
                    ),
                    "Rows": int(round(n_weighted or 0)),
                    "Share of CIF/FOB input": (
                        _format_percent(weighted_share)
                    ),
                },
                {
                    "Population component": (
                        "Single usable report retained"
                    ),
                    "Rows": int(round(n_single_report or 0)),
                    "Share of CIF/FOB input": (
                        _format_percent(single_report_share)
                    ),
                },
                {
                    "Population component": (
                        "No usable FOB-basis value"
                    ),
                    "Rows": int(round(n_not_eligible or 0)),
                    "Share of CIF/FOB input": (
                        _format_percent(not_eligible_share)
                    ),
                },
                {
                    "Population component": "Total CIF/FOB input",
                    "Rows": int(round(n_rows or 0)),
                    "Share of CIF/FOB input": "100.00%",
                },
            ]
        )


        # --------------------------------------------------------------
        # Figures
        # --------------------------------------------------------------
        figure_paths = {
            "status": (
                figure_dir / "reconciliation_status_by_year.png"
            ),
            "totals": (
                figure_dir / "trade_value_totals_by_year.png"
            ),
            "reliability": (
                figure_dir / "country_reliability_by_year.png"
            ),
            "agreement": (
                figure_dir
                / "positive_positive_agreement_by_year.png"
            ),
            "deviation_distribution": (
                figure_dir
                / "positive_positive_deviation_distribution.png"
            ),
            "country_directional_reliability": (
                figure_dir
                / "country_export_import_reliability_gap.png"
            ),
            "country_reliability_level_gap": (
                figure_dir
                / "country_reliability_level_vs_gap.png"
            ),
            "export_reliability_trajectory": (
                figure_dir
                / "selected_country_export_reliability_over_time.png"
            ),
        }
        
        _plot_reconciliation_status_by_year(
            con=con,
            reconciled_sql=reconciled_sql,
            output_path=figure_paths["status"],
        )
        _plot_trade_value_totals_by_year(
            con=con,
            reconciled_sql=reconciled_sql,
            output_path=figure_paths["totals"],
        )
        _plot_country_reliability_by_year(
            con=con,
            reliability_sql=reliability_sql,
            output_path=figure_paths["reliability"],
        )
        _plot_country_directional_reliability(
            con=con,
            reliability_sql=reliability_sql,
            output_path=figure_paths["country_directional_reliability"],
            year=country_reliability_plot_year,
            minimum_evidence=30,
            maximum_countries=30,
        )
        _plot_positive_positive_agreement_by_year(
            con=con,
            reconciled_sql=reconciled_sql,
            agreement_tolerance=agreement_tolerance,
            output_path=figure_paths["agreement"],
        )
        _plot_positive_positive_deviation_distribution(
            con=con,
            reconciled_sql=reconciled_sql,
            output_path=figure_paths["deviation_distribution"],
        )

        _plot_country_reliability_level_vs_gap(
            con=con,
            reliability_sql=reliability_sql,
            output_path=figure_paths["country_reliability_level_gap"],
            year=country_reliability_plot_year,
            minimum_evidence=30,
        )

        _plot_selected_country_reliability_over_time(
            con=con,
            reliability_sql=reliability_sql,
            output_path=figure_paths["export_reliability_trajectory"],
            reporting_direction="export",
            maximum_year=country_reliability_plot_year,
            minimum_evidence=30,
            minimum_years=6,
            maximum_countries=8,
        )

        # --------------------------------------------------------------
        # Decision logic
        # --------------------------------------------------------------

        audit_row = audit.iloc[0]

        n_invalid_reconciled = int(
            audit_row["n_invalid_reconciled_values"]
        )
        n_missing_expected = int(
            audit_row["n_missing_expected_reconciled_values"]
        )
        n_duplicate_keys = int(
            audit_row["n_duplicate_keys"]
        )

        audit_passed = bool(
            int(audit_row["n_invalid_reconciled_values"]) == 0
            and int(
                audit_row[
                    "n_missing_expected_reconciled_values"
                ]
            ) == 0
            and int(audit_row["n_duplicate_keys"]) == 0
        )

        population_partition_passed = bool(
            n_rows is not None
            and n_reconciled is not None
            and n_not_eligible is not None
            and int(round(n_rows))
            == int(round(n_reconciled))
            + int(round(n_not_eligible))
        )

        ready_for_processed_export = bool(
            audit_passed
            and population_partition_passed
        )

        decision = (
            "APPROVED"
            if ready_for_processed_export
            else "REVIEW REQUIRED"
        )

        provisional_note = ""

        if latest_year_provisional:
            provisional_note = (
                f" Results for {maximum_year} contain substantially "
                "fewer records than preceding years and should be "
                "treated as provisional."
            )

        # --------------------------------------------------------------
        # Markdown
        # --------------------------------------------------------------

        lines = [
        "# Trade Sync Decision Report",
        "",
        f"**Decision:** {decision}  ",
        f"**Run ID:** {run_id}  ",
        f"**Target HS:** {cfg.scope.target_hs}  ",
        (
            f"**Input period:** {cfg.scope.min_year}–"
            f"{cfg.scope.max_year}  "
        ),
        (
            f"**Generated:** {datetime.now().isoformat(timespec='seconds')}"
        ),
        "",
        "__TOC__",
        "",
        "",
        "## 1. Purpose of Trade Sync",
        "",
        (
            "International bilateral trade can be reported twice: once "
            "by the exporter and once by the importer. Even after the "
            "importer-reported CIF value has been converted to an "
            "FOB-equivalent value, the two mirror reports may disagree "
            "substantially."
        ),
        "",
        (
            "The purpose of Trade Sync is to create one harmonized "
            "FOB-basis trade value for every "
            "`year × exporter × importer × product` observation. "
            "Where bilateral reliability evidence is available, including "
            "positive-versus-implicit-zero cases, Trade Sync reconciles the "
            "two reporting sides using reliability-based weights. Where only "
            "one usable report exists and the missing mirror is not classified "
            "as an implicit zero, that report is retained. Rows without either "
            "usable value remain in the dataset but receive no reconciled "
            "trade value."
        ),
        "",
        "## 2. Input data and observation unit",
        "",
        (
            f"Trade Sync received `{_format_integer(n_rows)}` "
            "bilateral exporter–importer–product–year rows from the "
            "upstream CIF/FOB-corrected dataset."
        ),
        "",
        "| Property | Value |",
        "|---|---|",
        "| Upstream module | CIF/FOB correction |",
        (
            "| Observation unit | "
            "Year × exporter × importer × target product |"
        ),
        f"| Target HS classification | {cfg.scope.target_hs} |",
        (
            f"| Covered years | "
            f"{cfg.scope.min_year}–{cfg.scope.max_year} |"
        ),
        (
            "| Export-side input | "
            "export_value_fob |"
        ),
        (
            "| Import-side input | "
            "import_value_fob_corrected |"
        ),
        (
            "| Final Trade Sync value | "
            "trade_value_reconciled |"
        ),
        "",
        "## 3. Methodology",
        "",
        "### 3.1 Mirror values",
        "",
        (
            "Trade Sync works at this key: "
            "`year × exporter × importer × product`"
        ),
        "",
        "",
        (
            "For a given bilateral product-year row, Trade Sync may "
            "observe two FOB-basis mirror values:"
        ),
        "",
        (
            "- the FOB value reported by the exporter; and"
        ),
        (
            "- the importer-reported CIF value converted to an "
            "FOB-equivalent basis using the model-estimated CIF/FOB factor for eligible rows."
        ),
        "",
        (
            "The two values represent the same intended trade flow but "
            "may differ because of reporting errors, timing, partner or "
            "product classification differences, missing declarations, "
            "or other inconsistencies."
        ),
        "",
        "### 3.2 Bilateral deviation",
        "",
        (
            "For rows where both mirror reports are positive, their "
            "relative disagreement is measured against the larger report:"
        ),
        "",
        "$$",
        (
            "d = "
            "\\frac{|V^{export} - V^{import,FOB}|}"
            "{\\max(V^{export}, V^{import,FOB})}"
            "\\qquad 0 \\leq d \\leq 1"

        ),
        "$$",
        "",
        (
            "A deviation of `0` means exact agreement. A deviation of "
            "`0.20` means that the reports differ by 20% of the larger "
            "reported value. A value close to `1` indicates very large "
            "disagreement."
        ),
        "",
        "### 3.3 Country-year-direction reliability",
        "",
        (
        "Across all relevant rows involving one country in one year and one reporting direction, "
        "how consistently did that country agree with its mirror partners?"
        ),
        "",
        (
            "Reporting reliability is estimated separately for each "
            "`country × year × reporting direction`: exporter reporting "
            "and importer reporting."
        ),
        "",
        (
            "The workflow aggregates the bilateral deviation evidence "
            "associated with that country-year-direction. Its mean "
            "evidence deviation is converted into a reliability index:"
        ),
        "",
        "$$",
        (
            "R_{c,t,r} = 1 - \\overline{d}_{c,t,r}"
             "\\qquad 0 \\leq R_{c,t,r} \\leq 1"
        ),
        "$$",
        "",
        (
            "A higher reliability index indicates that the country's "
            "reports are, on average, more consistent with corresponding "
            "mirror evidence. The index is not a prediction-accuracy "
            "score."
        ),
        "",
        (
            "The following figure places each eligible country according to "
            "its average exporter-importer reliability and the absolute gap "
            "between the two directional scores. Countries toward the right "
            "have higher average reliability, while countries near the bottom "
            "have more balanced exporter and importer reliability."
        ),
        "",
        markdown_image(
            "Country reliability level and directional gap",
            figure_paths["country_reliability_level_gap"],
            document_path=report_path,
            height=600,
            width=700, 
        ),
        "",
        "### 3.4 Reliability-weighted reconciliation",
        "",
        (
            "When both reporting sides participate in reconciliation, "
            "Trade Sync follows the weighting approach of Bremer, Perger, "
            "and Schumacher. The weights are derived from the "
            "country-year-direction mean deviations rather than directly from "
            "the reliability indices."
        ),
        "",
        (
            "Because a lower mean deviation indicates more reliable "
            "reporting, each report receives a base weight proportional "
            "to the partner report's mean deviation:"
        ),
        "",
        "$$",
        (
            "w^{export}_{base} = "
            "\\frac{\\overline{d}^{import}}"
            "{\\overline{d}^{export} + \\overline{d}^{import}}"
        ),
        "$$",
        "",
        "$$",
        (
            "w^{import}_{base} = "
            "\\frac{\\overline{d}^{export}}"
            "{\\overline{d}^{export} + \\overline{d}^{import}}"
        ),
        "$$",
        "",
        (
            "This inverse-deviation construction gives the larger base weight "
            "to the reporting side with the lower country-year-direction "
            "mean deviation."
            "When both mean deviations are zero, equal base weights of "
            "0.5 and 0.5 are used."
        ),
        "",
        (
            "Following Bremer, Perger, and Schumacher, Trade Sync then applies "
            "an import-report preference by transferring 0.10 weight from the "
            "exporter side to the importer side:"
        ),
        "",
        "$$",
        (
            "w^{export} = w^{export}_{base} - 0.10"
        ),
        "$$",
        "",
        "$$",
        (
            "w^{import} = w^{import}_{base} + 0.10"
        ),
        "$$",
        "",
        (
            "The adjusted weights therefore continue to sum to one. "
            "The workflow audits that both adjusted weights remain within "
            "the interval [0, 1]."
        ),
        "",
        (
            "The harmonized value is then calculated as:"
        ),
        "",
        "$$",
        (
            "V^{reconciled} = "
            "w^{export}V^{export} + "
            "w^{import}V^{import,FOB}"
        ),
        "$$",
        "",
        (
            "Consequently, reconciliation reflects both the relative "
            "consistency of exporter and importer reporting and the "
            "paper's explicit additional preference for importer reports. "
            "For a genuine two-positive row, the reconciled value remains "
            "bounded by the exporter-reported FOB value and the importer "
            "FOB-equivalent value."
        ),
        "",
        "### 3.5 Implicit zeros versus genuine missingness",
        "",
        (
            "An implicit zero is not necessarily a literal reported zero. "
            "It is a workflow classification used when one side reports a "
            "positive bilateral flow, the corresponding mirror flow is absent, "
            "and the missing-side country reported at least one positive trade "
            "flow in the same reporting direction and year."
        ),
        "",
        (
            "Such positive-versus-implicit-zero cases contribute maximum "
            "disagreement evidence and participate in reliability "
            "estimation and weighted reconciliation."
        ),
        "",
        (
            "This is different from a genuinely unavailable partner "
            "report. Genuinely missing partner information is not "
            "automatically interpreted as zero. When only one usable "
            "report remains, the available value is retained without "
            "bilateral weighting."
        ),
        "",
        "### 3.6 Treatment by input situation",
        "",
        _markdown_table(treatment_definitions),
        "",
        "### 3.7 Processing sequence",
        "",
        "```text",
        "CIF/FOB-corrected bilateral input",
        "        ↓",
        "Identify exporter FOB and importer FOB-equivalent mirror values",
        "        ↓",
        "Classify positive, implicit-zero and unavailable cases",
        "        ↓",
        "Construct bilateral reliability evidence",
        "        ↓",
        "Estimate country-year-direction reliability",
        "        ↓",
        "Normalize mean-deviation-based weights",
        "        ↓",
        "Apply the 0.10 import-weight preference",
        "        ↓",
        "Reconcile two-sided and implicit-zero cases using the final adjusted weights",
        "        ↓",
        "Retain usable unilateral reports",
        "        ↓",
        "Preserve rows without a usable value as not eligible",
        "        ↓",
        "Run reconciliation audit",
        "        ↓",
        "Approve output for Weight Sync",
        "```",
        "",
        "## 4. Processing outcome",
        "",
        (
            f"Of the `{_format_integer(n_rows)}` rows received from "
            f"CIF/FOB, `{_format_integer(n_reconciled)}` "
            f"({_format_percent(usable_coverage)}) produced a usable "
            "Trade Sync value."
        ),
        "",
        (
            f"Reliability-weighted reconciliation was applied to "
            f"`{_format_integer(n_weighted)}` rows "
            f"({_format_percent(weighted_share)} of the CIF/FOB input). "
            f"A further `{_format_integer(n_single_report)}` rows "
            f"({_format_percent(single_report_share)}) contained only "
            "one usable report and were retained without bilateral "
            "weighting."
        ),
        "",
        (
            f"Together, these two groups account for all "
            f"`{_format_integer(n_reconciled)}` rows with a usable "
            "Trade Sync value."
        ),
        "",
        (
            f"The remaining `{_format_integer(n_not_eligible)}` rows "
            f"({_format_percent(not_eligible_share)}) had neither a "
            "usable exporter FOB value nor a usable importer FOB-equivalent value. These rows were preserved, but no "
            "reconciled trade value could be produced."
        ),
        "",
        _markdown_table(population_accounting),
        "",
        (
            f"All `{_format_integer(n_rows)}` CIF/FOB input rows were "
            "preserved in the Trade Sync output."
        ),
        "",
        (
            f"The total reconciled trade value across rows with a usable "
            f"output is `{_format_usd(total_reconciled)}`."
        ),
        "",
        (
            provisional_note.strip()
            if provisional_note
            else ""
        ),
        "",
        "## 5. Decision basis",
        "",
        (
            f"The Trade Sync output is `{decision.lower()}` for Weight "
            "Sync because:"
        ),
        "",
        (
            f"- all `{_format_integer(n_rows)}` rows received from "
            "CIF/FOB were preserved;"
        ),
        (
            f"- all `{_format_integer(n_reconciled)}` rows expected to "
            "receive a reconciled or retained value received one;"
        ),
        (
            f"- the `{_format_integer(n_not_eligible)}` rows without a "
            "value were explicitly classified as `not_eligible`, rather "
            "than being silently dropped;"
        ),
        (
            f"- the audit found "
            f"`{_format_integer(n_invalid_reconciled)}` invalid "
            "reconciled values;"
        ),
        (
            f"- the audit found "
            f"`{_format_integer(n_missing_expected)}` missing expected "
            "reconciled values; and"
        ),
        (
            f"- the audit found "
            f"`{_format_integer(n_duplicate_keys)} duplicate bilateral "
            "product-year keys`."
        ),
        "",
        "### Decision checks",
        "",
        "| Check | Result |",
        "|---|---:|",
        (
            "| CIF/FOB input rows received | "
            f"{_format_integer(n_rows)} |"
        ),
        (
            "| Rows preserved in Trade Sync output | "
            f"{_format_integer(n_rows)} |"
        ),
        (
            "| Reliability-weighted rows | "
            f"{_format_integer(n_weighted)} "
            f"({_format_percent(weighted_share)}) |"
        ),
        (
            "| Single-report rows retained | "
            f"{_format_integer(n_single_report)} "
            f"({_format_percent(single_report_share)}) |"
        ),
        (
            "| Rows with a usable Trade Sync value | "
            f"{_format_integer(n_reconciled)} "
            f"({_format_percent(usable_coverage)}) |"
        ),
        (
            "| Rows without a usable FOB-basis value | "
            f"{_format_integer(n_not_eligible)} "
            f"({_format_percent(not_eligible_share)}) |"
        ),
        (
            "| Invalid reconciled values | "
            f"{_format_integer(n_invalid_reconciled)} |"
        ),
        (
            "| Missing expected reconciled values | "
            f"{_format_integer(n_missing_expected)} |"
        ),
        (
            "| Duplicate bilateral product-year keys | "
            f"{_format_integer(n_duplicate_keys)} |"
        ),
        f"| Reconciliation audit passed | `{audit_passed}` |",
        (
            "| Ready for Weight Sync | "
            f"`{ready_for_processed_export}` |"
        ),
        "",
        "## 6. Input and reconciled value totals",
        "",
        "| Measure | Value |",
        "|---|---:|",
        (
            "| Reported exporter FOB total | "
            f"{_format_usd(total_export)} |"
        ),
        (
            "| Importer FOB-equivalent total | "
            f"{_format_usd(total_import)} |"
        ),
        (
            "| Final reconciled total | "
            f"{_format_usd(total_reconciled)} |"
        ),
        "",
        (
            "The reconciled aggregate is not required to lie between the "
            "aggregate exporter and importer totals because those totals "
            "cover different combinations of available bilateral reports. "
            "The row-level boundedness statement applies specifically to "
            "genuine two-sided weighted reconciliation."
        ),
        "",
        "## 7. Mirror-report disagreement",
        "",
        (
            f"There are `{_format_integer(n_positive_positive)}` rows "
            "where both the exporter and importer FOB-equivalent reports are "
            "positive."
        ),
        "",
        (
            "Among these genuine two-sided comparisons, the median "
            "difference relative to the larger report is "
            f"`{_format_percent(median_difference)}`, while the 90th "
            f"percentile is `{_format_percent(p90_difference)}`."
        ),
        "",
        (
            f"Only `{_format_percent(agreement_share)}` of these pairs "
            f"fall within the configured "
            f"{100 * agreement_tolerance:.0f}% agreement tolerance."
        ),
        "",
        (
            "These results show that exporter and importer mirror reports "
            "remain highly dispersed. This supports harmonization using "
            "reporting-reliability evidence rather than automatically "
            "selecting one side or applying an unweighted average."
        ),
        "",
        (
            "Relative to a simple bilateral average, the final adjusted "
            "reconciliation weights change the reconciled value by a median "
            "absolute amount of "
            f"`{_format_percent(median_change_simple)}`. The 90th "
            f"percentile adjustment is "
            f"`{_format_percent(p90_change_simple)}`."
        ),
        "",
        (
            "The following distribution shows whether disagreement is "
            "concentrated near low deviations or spread across more severe "
            "difference bands. It includes only rows where both mirror reports "
            "are positive."
        ),
        "",
        markdown_image(
            "Distribution of positive-positive mirror-report differences",
            figure_paths["deviation_distribution"],
            document_path=report_path,
        ),
        "",
        markdown_image(
            "Positive-report agreement and discrepancy by year",
            figure_paths["agreement"],
            document_path=report_path,
        ),
        "",
        "## 8. Reliability evidence",
        "",
        (
            f"Reliability estimation used "
            f"`{_format_integer(n_evidence_rows)}` evidence rows."
        ),
        "",
        (
            f"Of these, `{_format_integer(n_positive_positive)}` "
            "contained two positive mirror reports, while "
            f"`{_format_integer(n_implicit_zero_rows)}` "
            f"({_format_percent(implicit_zero_share)}) paired one "
            "positive report with an implicit zero."
        ),
        "",
        (
            "The country reliability index therefore reflects two forms of "
            "mirror-report inconsistency:"
        ),
        "",
        "- disagreement between two positive mirror reports; and",
        (
            "- the frequency with which a country has positive reports "
            "that are not mirrored on the partner side."
        ),
        "",
        (
            "For this reason, a low reliability index does not mean that "
            "a statistical prediction model performed poorly. It means "
            "that the country's reporting evidence was inconsistent with "
            "its mirror evidence."
        ),
        "",
        _markdown_table(
            reliability_summary.rename(
                columns={
                    "reporting_direction": "Reporting direction",
                    "country_year_scores": "Country-year scores",
                    "median_evidence_rows": "Median evidence rows",
                    "p10_evidence_rows": "P10 evidence rows",
                    "median_mean_deviation": "Median mean deviation",
                    "median_reliability_index": "Median reliability index",
                    "median_implicit_zero_share": "Median implicit-zero share",
                    "scores_below_30_rows": "Scores below 30 evidence rows",
                }
            )
        ),
        "",
        markdown_image(
            "Country reliability by year",
            figure_paths["reliability"],
            document_path=report_path,
        ),
        "",
        (
            "The following figure compares exporter and importer reliability "
            "for countries with the largest directional gaps. Only countries "
            "with at least 30 reliability-evidence observations in both "
            "directions are included."
        ),
        "",
        markdown_image(
            "Country exporter versus importer reliability",
            figure_paths["country_directional_reliability"],
            document_path=report_path,
            height=700,
            width=750,

        ),
        "",
        (
            "The following figure shows exporter-reliability trajectories for "
            "countries with the greatest temporal variation. Countries must "
            "have at least 30 evidence observations in at least six years to "
            "be eligible. Selection is based on variation, not on the absolute "
            "level of reliability."
        ),
        "",
        markdown_image(
            "Selected-country exporter reliability over time",
            figure_paths["export_reliability_trajectory"],
            document_path=report_path,
        ),
        "",
        "## 9. Reconciliation composition",
        "",
        (
            "The following table reports how the complete CIF/FOB input "
            "population was classified and treated:"
        ),
        "",
        _markdown_table(
            status_summary.rename(
                columns={
                    "status": "Status",
                    "rows": "Rows",
                    "share_percent": "Share of input (%)",
                    "reconciled_value": "Reconciled value",
                }
            )
        ),
        "",
        markdown_image(
            "Reconciliation status by year",
            figure_paths["status"],
            document_path=report_path,
        ),
        "",
        "## 10. Annual results",
        "",
        _markdown_table(
            annual_summary.rename(
                columns={
                    "year": "Year",
                    "rows": "Rows",
                    "rows_with_reconciled_value": "Rows with Trade Sync value",
                    "reconciled_value": "Reconciled value",
                    "median_positive_positive_deviation": (
                        "Median positive-positive deviation"
                    ),
                    "implicit_zero_evidence_share": (
                        "Implicit-zero evidence share"
                    ),
                }
            )
        ),
        "",
        markdown_image(
            "Annual trade value totals",
            figure_paths["totals"],
            document_path=report_path,
        ),
        "",
        (
            "The reconciled total may exceed both exporter and importer aggregate totals because Trade Sync combines "
            "the union of available reports: bilateral rows are reconciled, while exporter-only and importer-only rows "
            "are both retained. The boundedness property applies row-by-row to bilateral reconciliation, not to these aggregate totals."
        ),
        "",
        "## 11. Interpretation and limitations",
        "",
        (
            "- Trade Sync harmonizes reporting evidence; it does not "
            "establish the unobservable true trade value with certainty."
        ),
        (
            "- Reliability scores incorporate positive-positive disagreement and, where present, implicit-zero evidence."
        ),
        (
            "- Retained unilateral values are usable but are not "
            "corroborated by a corresponding usable mirror report."
        ),
        (
            "- The results depend on the upstream CIF/FOB correction because importer FOB-equivalent "
            "values enter Trade Sync directly."
        ),
        (
            "- The configured agreement tolerance is diagnostic only. "
            "It does not override the final adjusted reconciliation weights."
        ),
        (
            f"- Results for {maximum_year} should be treated as "
            "provisional when their reporting coverage is materially "
            "lower than preceding years."
            if latest_year_provisional
            else (
                "- No material final-year row-count shortfall was "
                "detected by the current provisional-year diagnostic."
            )
        ),
        (
            "- An absent mirror report is treated as an implicit zero when "
            "the country reported at least one positive trade flow in the "
            "same direction and year. This activity test is defined at the "
            "country–year–direction level, not at the partner or product "
            "level, so some missing bilateral product flows may be penalized "
            "as implicit zeros even when reporting coverage is uneven across "
            "partners or HS codes."
        ),
        (
            "- Mirror reports are aligned within the same calendar year. "
            "Exports recorded when goods leave the exporting country may be "
            "reported as imports only after entering the partner country in "
            "the following year. Such timing differences, especially around "
            "year-end, may increase measured disagreement or produce apparent "
            "implicit-zero cases even when both reports refer to the same "
            "underlying trade flow."
        ),
        "",
        "## 12. Output artifacts",
        "",
        f"- Reconciled trade: `{reconciled_trade_path}`",
        f"- Country reliability: `{country_reliability_path}`",
        f"- Reconciliation audit: `{reconciliation_audit_path}`",
        f"- Diagnostics: `{diagnostics_path}`",
        f"- Report: `{report_path}`",
        "",
    ]

    
        insert_markdown_toc(
            lines,
            min_level=2,
            max_level=3,
        )

        report_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        if not valid_output(report_path):
            raise RuntimeError(
                "Trade Sync decision report was not written correctly."
            )

        latest_report_path: Path | None = None

        if cfg.artifacts.write_latest:
            latest_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.copyfile(report_path, latest_path)
            latest_report_path = latest_path

        logger.info(
            "%s Trade Sync decision report created.",
            ok,
        )
        logger.info("  Decision: %s", decision)
        logger.info(
            "  Ready for processed export: %s",
            ready_for_processed_export,
        )
        logger.info("  Report: %s", report_path)

        if latest_report_path is not None:
            logger.info(
                "  Latest report: %s",
                latest_report_path,
            )

        logger.info(
                "  Figures written: %d",
                len(figure_paths),
            )

        return {
            "status": "completed",
            "output_path": report_path,
            "report_path": report_path,
            "latest_report_path": latest_report_path,
            "figure_paths": list(figure_paths.values()),
            "figures": figure_paths,
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
            "n_positive_positive_rows": int(
                n_positive_positive or 0
            ),
            "n_implicit_zero_evidence_rows": (
                n_implicit_zero_rows
            ),
            "total_reconciled_value": float(
                total_reconciled or 0.0
            ),
        }

    finally:
        con.close()
