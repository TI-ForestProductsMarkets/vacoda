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
from vacoda.utils.sql import sql_literal

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

def _plot_quantity_reconciliation_effect_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the effect of final quantity reconciliation by year."""

    median_frame = diagnostics[
        diagnostics["diagnostic_scope"].eq("quantity_year")
        & diagnostics["metric"].eq(
            "median_quantity_reconciled_vs_mean_fraction"
        )
    ][["year", "metric_value"]].rename(
        columns={
            "metric_value": "median_difference",
        }
    )

    p90_frame = diagnostics[
        diagnostics["diagnostic_scope"].eq("quantity_year")
        & diagnostics["metric"].eq(
            "p90_quantity_reconciled_vs_mean_fraction"
        )
    ][["year", "metric_value"]].rename(
        columns={
            "metric_value": "p90_difference",
        }
    )

    frame = median_frame.merge(
        p90_frame,
        on="year",
        how="inner",
    ).sort_values("year")

    if frame.empty:
        raise RuntimeError(
            "No year-level quantity reconciliation-effect "
            "diagnostics are available."
        )

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        frame["year"],
        100 * frame["median_difference"],
        marker="o",
        label="Median",
    )

    axis.plot(
        frame["year"],
        100 * frame["p90_difference"],
        marker="o",
        label="P90",
    )

    axis.set_title(
        "Final reconciled quantity versus simple bilateral mean"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("Absolute difference (%)")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

def _plot_quantity_plausibility_by_disagreement_band(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot unit-value evidence by quantity-disagreement severity."""

    relevant = diagnostics[
        diagnostics["diagnostic_scope"].eq(
            "quantity_disagreement_band"
        )
        & diagnostics["metric"].isin(
            {
                "plausibility_evidence_share",
                "clear_plausibility_preference_share",
            }
        )
    ].copy()

    if relevant.empty:
        raise RuntimeError(
            "No quantity plausibility diagnostics are available."
        )

    frame = (
        relevant
        .pivot_table(
            index="reconciliation_status",
            columns="metric",
            values="metric_value",
            aggfunc="first",
        )
        .reset_index()
    )

    band_order = [
        "00_10",
        "10_50",
        "50_90",
        "90_100",
    ]

    band_labels = {
        "00_10": "0–10%",
        "10_50": "10–50%",
        "50_90": "50–90%",
        "90_100": "90–100%",
    }

    frame["band_order"] = frame[
        "reconciliation_status"
    ].map(
        {
            band: index
            for index, band in enumerate(band_order)
        }
    )

    frame = frame.sort_values("band_order")

    x = range(len(frame))

    width = 0.38

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.bar(
        [position - width / 2 for position in x],
        100 * frame["plausibility_evidence_share"],
        width=width,
        label="Comparable unit-value evidence",
    )

    axis.bar(
        [position + width / 2 for position in x],
        100 * frame["clear_plausibility_preference_share"],
        width=width,
        label="Clear plausibility preference",
    )

    axis.set_xticks(list(x))
    axis.set_xticklabels(
        [
            band_labels.get(code, code)
            for code in frame["reconciliation_status"]
        ]
    )

    axis.set_title(
        "Unit-value plausibility by quantity-disagreement severity"
    )
    axis.set_xlabel(
        "Exporter–importer quantity difference"
    )
    axis.set_ylabel("Share of rows (%)")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

def _plot_quantity_deviation_distribution(
    *,
    con: duckdb.DuckDBPyConnection,
    reconciled_sql: str,
    output_path: Path,
) -> None:
    """Plot disagreement between usable exporter and importer kg quantities."""

    frame = con.execute(
        f"""
        SELECT
            ABS(
                export_synchronized_quantity_kg
                - import_synchronized_quantity_kg
            )
            / GREATEST(
                export_synchronized_quantity_kg,
                import_synchronized_quantity_kg
            ) AS relative_difference

        FROM read_parquet({reconciled_sql})

        WHERE export_synchronized_quantity_kg IS NOT NULL
          AND isfinite(export_synchronized_quantity_kg)
          AND export_synchronized_quantity_kg > 0
          AND import_synchronized_quantity_kg IS NOT NULL
          AND isfinite(import_synchronized_quantity_kg)
          AND import_synchronized_quantity_kg > 0
        """
    ).fetchdf()

    if frame.empty:
        raise RuntimeError(
            "No bilateral positive quantity observations are available."
        )

    relative_difference = frame["relative_difference"].dropna()

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
        "Distribution of bilateral quantity differences\n"
        f"All years, n = {len(relative_difference):,}"
    )
    axis.set_xlabel(
        "Quantity difference relative to the larger report"
    )
    axis.set_ylabel("Share of rows (%)")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


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
    quantity_reconciliation_audit_path: Path,
    diagnostics_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """Create the final Trade Sync Markdown decision report."""

    reconciled_trade_path = Path(reconciled_trade_path)
    country_reliability_path = Path(country_reliability_path)
    reconciliation_audit_path = Path(reconciliation_audit_path)
    quantity_reconciliation_audit_path = Path(quantity_reconciliation_audit_path)
    diagnostics_path = Path(diagnostics_path)

    for label, path in (
        ("reconciled trade", reconciled_trade_path),
        ("country reliability", country_reliability_path),
        ("reconciliation audit", reconciliation_audit_path),
        ("quantity reconciliation audit", quantity_reconciliation_audit_path),
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

    reconciled_sql = sql_literal(reconciled_trade_path)
    reliability_sql = sql_literal(country_reliability_path)
    audit_sql = sql_literal(reconciliation_audit_path)
    diagnostics_sql = sql_literal(diagnostics_path)
    quantity_audit_sql = sql_literal(quantity_reconciliation_audit_path)

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

        quantity_audit = con.execute(
            f"""
            SELECT *
            FROM read_parquet({quantity_audit_sql})
            """
        ).fetchdf()

        if len(quantity_audit) != 1:
            raise RuntimeError(
                "Quantity reconciliation audit must contain exactly one row."
            )

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

        n_both_quantity = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="n_both_quantity_rows",
        )

        n_export_quantity_only = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="n_export_quantity_only_rows",
        )

        n_import_quantity_only = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="n_import_quantity_only_rows",
        )

        n_no_quantity = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="n_no_quantity_rows",
        )

        median_quantity_difference = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="median_quantity_relative_difference",
        )

        p90_quantity_difference = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="p90_quantity_relative_difference",
        )

        p95_quantity_difference = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="p95_quantity_relative_difference",
        )

        p99_quantity_difference = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="p99_quantity_relative_difference",
        )

        quantity_within_10pct = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="quantity_within_10pct_share",
        )

        # --------------------------------------------------------------
        # Quantity reconciliation composition
        # --------------------------------------------------------------

        n_quantity_reliability_weighted = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="n_quantity_reliability_weighted",
        )

        n_quantity_simple_mean_fallback = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="n_quantity_simple_mean_fallback",
        )

        n_quantity_export_retained = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="n_quantity_export_retained",
        )

        n_quantity_import_retained = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="n_quantity_import_retained",
        )

        n_quantity_missing = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="n_quantity_missing",
        )

        n_plausibility_decisive = _metric_lookup(
            diagnostics,
            scope="quantity_plausibility_first",
            metric="n_plausibility_decisive_rows",
        )

        plausibility_decisive_share = _metric_lookup(
            diagnostics,
            scope="quantity_plausibility_first",
            metric="plausibility_decisive_share",
        )

        n_plausibility_export_selected = _metric_lookup(
            diagnostics,
            scope="quantity_plausibility_first",
            metric="n_plausibility_export_selected",
        )

        n_plausibility_import_selected = _metric_lookup(
            diagnostics,
            scope="quantity_plausibility_first",
            metric="n_plausibility_import_selected",
        )

        median_plausibility_first_change = _metric_lookup(
            diagnostics,
            scope="quantity_plausibility_first",
            metric="median_plausibility_first_change",
        )

        p90_plausibility_first_change = _metric_lookup(
            diagnostics,
            scope="quantity_plausibility_first",
            metric="p90_plausibility_first_change",
        )

        plausibility_first_improves_share = _metric_lookup(
            diagnostics,
            scope="quantity_plausibility_first",
            metric="plausibility_first_improves_share",
        )

        median_robust_distance_improvement = _metric_lookup(
            diagnostics,
            scope="quantity_plausibility_first",
            metric="median_robust_distance_improvement",
        )

        # --------------------------------------------------------------
        # Effect of quantity reconciliation
        # --------------------------------------------------------------

        median_quantity_reconciled_vs_mean = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="median_quantity_reconciled_vs_mean_fraction",
        )

        p90_quantity_reconciled_vs_mean = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="p90_quantity_reconciled_vs_mean_fraction",
        )

        median_quantity_reliability_vs_value_weighted = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric=(
                "median_quantity_reliability_vs_value_weighted_fraction"
            ),
        )

        p90_quantity_reliability_vs_value_weighted = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric=(
                "p90_quantity_reliability_vs_value_weighted_fraction"
            ),
        )


        # --------------------------------------------------------------
        # Quantity-specific reliability
        # --------------------------------------------------------------

        median_export_quantity_reliability = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="median_export_quantity_reliability",
        )

        median_import_quantity_reliability = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="median_import_quantity_reliability",
        )

        median_quantity_reliability_gap = _metric_lookup(
            diagnostics,
            scope="quantity_overall",
            metric="median_quantity_reliability_gap",
        )


        # --------------------------------------------------------------
        # Unit-value plausibility
        # --------------------------------------------------------------

        n_plausibility_comparable = _metric_lookup(
            diagnostics,
            scope="quantity_plausibility",
            metric="n_plausibility_comparable_rows",
        )

        export_more_plausible_share = _metric_lookup(
            diagnostics,
            scope="quantity_plausibility",
            metric="export_more_plausible_share",
        )

        import_more_plausible_share = _metric_lookup(
            diagnostics,
            scope="quantity_plausibility",
            metric="import_more_plausible_share",
        )

        n_extreme_quantity_disagreement = _metric_lookup(
            diagnostics,
            scope="quantity_plausibility",
            metric="n_extreme_quantity_disagreement",
        )

        n_extreme_with_plausibility = _metric_lookup(
            diagnostics,
            scope="quantity_plausibility",
            metric="n_extreme_with_plausibility_evidence",
        )

        extreme_clear_plausibility_share = _metric_lookup(
            diagnostics,
            scope="quantity_plausibility",
            metric="extreme_clear_plausibility_preference_share",
        )

        n_reference_comparable_rows = _metric_lookup(
            diagnostics,
            scope="quantity_reference",
            metric="n_reference_comparable_rows",
        )

        reference_between_reporters_share = _metric_lookup(
            diagnostics,
            scope="quantity_reference",
            metric="reference_between_reporters_share",
        )

        current_closest_to_reference_share = _metric_lookup(
            diagnostics,
            scope="quantity_reference",
            metric="current_closest_to_reference_share",
        )

        export_closest_to_reference_share = _metric_lookup(
            diagnostics,
            scope="quantity_reference",
            metric="export_closest_to_reference_share",
        )

        import_closest_to_reference_share = _metric_lookup(
            diagnostics,
            scope="quantity_reference",
            metric="import_closest_to_reference_share",
        )

        median_reference_vs_current_fraction = _metric_lookup(
            diagnostics,
            scope="quantity_reference",
            metric="median_reference_vs_current_fraction",
        )

        p90_reference_vs_current_fraction = _metric_lookup(
            diagnostics,
            scope="quantity_reference",
            metric="p90_reference_vs_current_fraction",
        )

        median_current_reference_log_distance = _metric_lookup(
            diagnostics,
            scope="quantity_reference",
            metric="median_current_reference_log_distance",
        )

        median_export_reference_log_distance = _metric_lookup(
            diagnostics,
            scope="quantity_reference",
            metric="median_export_reference_log_distance",
        )

        median_import_reference_log_distance = _metric_lookup(
            diagnostics,
            scope="quantity_reference",
            metric="median_import_reference_log_distance",
        )

        median_reference_observations = _metric_lookup(
            diagnostics,
            scope="quantity_reference",
            metric="median_reference_observations",
        )

        # --------------------------------------------------------------
        # Final Trade-Synced quantity
        # --------------------------------------------------------------

        n_trade_synced_quantity_rows = _metric_lookup(
            diagnostics,
            scope="quantity_trade_sync",
            metric="n_trade_synced_quantity_rows",
        )

        n_unit_value_guided_export = _metric_lookup(
            diagnostics,
            scope="quantity_trade_sync",
            metric="n_unit_value_guided_export",
        )

        n_unit_value_guided_import = _metric_lookup(
            diagnostics,
            scope="quantity_trade_sync",
            metric="n_unit_value_guided_import",
        )

        n_unit_value_guided_mean = _metric_lookup(
            diagnostics,
            scope="quantity_trade_sync",
            metric="n_unit_value_guided_mean",
        )

        n_bilateral_mean_no_reference = _metric_lookup(
            diagnostics,
            scope="quantity_trade_sync",
            metric="n_bilateral_mean_no_reference",
        )

        n_trade_sync_export_retained = _metric_lookup(
            diagnostics,
            scope="quantity_trade_sync",
            metric="n_export_quantity_retained",
        )

        n_trade_sync_import_retained = _metric_lookup(
            diagnostics,
            scope="quantity_trade_sync",
            metric="n_import_quantity_retained",
        )

        n_trade_sync_no_usable_quantity = _metric_lookup(
            diagnostics,
            scope="quantity_trade_sync",
            metric="n_no_usable_quantity",
        )

        unit_value_guided_export_share = _metric_lookup(
            diagnostics,
            scope="quantity_trade_sync",
            metric="unit_value_guided_export_share",
        )

        unit_value_guided_import_share = _metric_lookup(
            diagnostics,
            scope="quantity_trade_sync",
            metric="unit_value_guided_import_share",
        )

        unit_value_guided_mean_share = _metric_lookup(
            diagnostics,
            scope="quantity_trade_sync",
            metric="unit_value_guided_mean_share",
        )

        bilateral_mean_no_reference_share = _metric_lookup(
            diagnostics,
            scope="quantity_trade_sync",
            metric="bilateral_mean_no_reference_share",
        )
        quantity_disagreement_bands = diagnostics[
            diagnostics["diagnostic_scope"].eq(
                "quantity_disagreement_band"
            )
        ].copy()

        if not quantity_disagreement_bands.empty:
            quantity_disagreement_bands = (
                quantity_disagreement_bands
                .pivot_table(
                    index="reconciliation_status",
                    columns="metric",
                    values="metric_value",
                    aggfunc="first",
                )
                .reset_index()
                .rename(
                    columns={
                        "reconciliation_status": "band",
                    }
                )
            )

            band_labels = {
                "00_10": "0–10%",
                "10_50": "10–50%",
                "50_90": "50–90%",
                "90_100": "90–100%",
            }

            quantity_disagreement_bands["band_label"] = (
                quantity_disagreement_bands["band"]
                .map(band_labels)
                .fillna(quantity_disagreement_bands["band"])
            )

            quantity_disagreement_bands["band_order"] = (
                quantity_disagreement_bands["band"]
                .map(
                    {
                        "00_10": 0,
                        "10_50": 1,
                        "50_90": 2,
                        "90_100": 3,
                    }
                )
            )

            quantity_disagreement_bands = (
                quantity_disagreement_bands
                .sort_values("band_order")
                .reset_index(drop=True)
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

        quantity_trade_sync_summary = pd.DataFrame(
            [
                {
                    "Outcome": "Trade-synced quantity available",
                    "Rows": _format_integer(
                        n_trade_synced_quantity_rows
                    ),
                    "Share of bilateral rows": "—",
                },
                {
                    "Outcome": "Unit-value guided — exporter selected",
                    "Rows": _format_integer(
                        n_unit_value_guided_export
                    ),
                    "Share of bilateral rows": _format_percent(
                        unit_value_guided_export_share
                    ),
                },
                {
                    "Outcome": "Unit-value guided — importer selected",
                    "Rows": _format_integer(
                        n_unit_value_guided_import
                    ),
                    "Share of bilateral rows": _format_percent(
                        unit_value_guided_import_share
                    ),
                },
                {
                    "Outcome": "Unit-value guided — bilateral mean selected",
                    "Rows": _format_integer(
                        n_unit_value_guided_mean
                    ),
                    "Share of bilateral rows": _format_percent(
                        unit_value_guided_mean_share
                    ),
                },
                {
                    "Outcome": "Bilateral mean — no unit-value reference",
                    "Rows": _format_integer(
                        n_bilateral_mean_no_reference
                    ),
                    "Share of bilateral rows": _format_percent(
                        bilateral_mean_no_reference_share
                    ),
                },
                {
                    "Outcome": "Exporter-only quantity retained",
                    "Rows": _format_integer(
                        n_trade_sync_export_retained
                    ),
                    "Share of bilateral rows": "—",
                },
                {
                    "Outcome": "Importer-only quantity retained",
                    "Rows": _format_integer(
                        n_trade_sync_import_retained
                    ),
                    "Share of bilateral rows": "—",
                },
                {
                    "Outcome": "No usable quantity",
                    "Rows": _format_integer(
                        n_trade_sync_no_usable_quantity
                    ),
                    "Share of bilateral rows": "—",
                },
            ]
        )

        quantity_trade_sync_band_display = (
            quantity_disagreement_bands[
                [
                    "band_label",
                    "n_rows",
                    "trade_synced_export_share",
                    "trade_synced_import_share",
                    "trade_synced_mean_share",
                    "trade_synced_no_reference_share",
                ]
            ].copy()
            if not quantity_disagreement_bands.empty
            else pd.DataFrame()
        )

        if not quantity_trade_sync_band_display.empty:
            for column in (
                "trade_synced_export_share",
                "trade_synced_import_share",
                "trade_synced_mean_share",
                "trade_synced_no_reference_share",
            ):
                quantity_trade_sync_band_display[column] *= 100
        quantity_reference_summary = pd.DataFrame(
            [
                {
                    "Measure": "Comparable bilateral rows",
                    "Value": _format_integer(
                        n_reference_comparable_rows
                    ),
                },
                {
                    "Measure": "Reference lies between reports",
                    "Value": _format_percent(
                        reference_between_reporters_share
                    ),
                },
                {
                    "Measure": "Current reconciliation closest",
                    "Value": _format_percent(
                        current_closest_to_reference_share
                    ),
                },
                {
                    "Measure": "Exporter quantity closest",
                    "Value": _format_percent(
                        export_closest_to_reference_share
                    ),
                },
                {
                    "Measure": "Importer quantity closest",
                    "Value": _format_percent(
                        import_closest_to_reference_share
                    ),
                },
                {
                    "Measure": "Median reference-vs-current change",
                    "Value": _format_percent(
                        median_reference_vs_current_fraction
                    ),
                },
                {
                    "Measure": "P90 reference-vs-current change",
                    "Value": _format_percent(
                        p90_reference_vs_current_fraction
                    ),
                },
                {
                    "Measure": "Median reference observations",
                    "Value": _format_integer(
                        median_reference_observations
                    ),
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
            "quantity_deviation_distribution": (
                figure_dir
                / "quantity_deviation_distribution.png"
            ),

            "quantity_reconciliation_effect": (
                figure_dir
                / "quantity_reconciliation_effect_by_year.png"
            ),

            "quantity_plausibility_by_disagreement": (
                figure_dir
                / "quantity_plausibility_by_disagreement.png"
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
        _plot_quantity_deviation_distribution(
            con=con,
            reconciled_sql=reconciled_sql,
            output_path=figure_paths[
                "quantity_deviation_distribution"
            ],
        )

        _plot_quantity_reconciliation_effect_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "quantity_reconciliation_effect"
            ],
        )

        _plot_quantity_plausibility_by_disagreement_band(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "quantity_plausibility_by_disagreement"
            ],
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


        quantity_method_summary = pd.DataFrame(
            [
                {
                    "Method": "Quantity reliability weighted",
                    "Rows": int(
                        round(
                            n_quantity_reliability_weighted
                            or 0
                        )
                    ),
                },
                {
                    "Method": "Bilateral simple-mean fallback",
                    "Rows": int(
                        round(
                            n_quantity_simple_mean_fallback
                            or 0
                        )
                    ),
                },
                {
                    "Method": "Exporter quantity retained",
                    "Rows": int(
                        round(
                            n_quantity_export_retained
                            or 0
                        )
                    ),
                },
                {
                    "Method": "Importer quantity retained",
                    "Rows": int(
                        round(
                            n_quantity_import_retained
                            or 0
                        )
                    ),
                },
                {
                    "Method": "No usable quantity",
                    "Rows": int(
                        round(
                            n_quantity_missing
                            or 0
                        )
                    ),
                },
            ]
        )


        quantity_band_display = (
            quantity_disagreement_bands[
                [
                    "band_label",
                    "n_rows",
                    "median_quantity_reliability_gap",
                    "median_reconciled_vs_mean_fraction",
                    "plausibility_evidence_share",
                    "clear_plausibility_preference_share",
                    "plausibility_first_trigger_share",
                    "plausibility_first_improves_share",
                    "median_plausibility_first_change",
                    "reference_between_reporters_share",
                    "current_closest_to_reference_share",
                    "export_closest_to_reference_share",
                    "import_closest_to_reference_share",
                    "median_reference_vs_current_fraction",
                ]
            ].copy()
            if not quantity_disagreement_bands.empty
            else pd.DataFrame()
        )

        if not quantity_band_display.empty:
            quantity_band_display[
                "median_quantity_reliability_gap"
            ] *= 100

            quantity_band_display[
                "median_reconciled_vs_mean_fraction"
            ] *= 100

            quantity_band_display[
                "plausibility_evidence_share"
            ] *= 100

            quantity_band_display[
                "clear_plausibility_preference_share"
            ] *= 100

            quantity_band_display[
                "plausibility_first_trigger_share"
            ] *= 100

            quantity_band_display[
                "plausibility_first_improves_share"
            ] *= 100

            quantity_band_display[
                "median_plausibility_first_change"
            ] *= 100

            quantity_band_display[
                "reference_between_reporters_share"
            ] *= 100

            quantity_band_display[
                "current_closest_to_reference_share"
            ] *= 100

            quantity_band_display[
                "export_closest_to_reference_share"
            ] *= 100

            quantity_band_display[
                "import_closest_to_reference_share"
            ] *= 100

            quantity_band_display[
                "median_reference_vs_current_fraction"
            ] *= 100
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
        "",
        "## 9. Bilateral quantity reconciliation",
        "",
        (
            "The Quantity module provides standardized kilogram quantities "
            "for the exporter and importer sides. Trade Sync converts this "
            "bilateral evidence into one `trade_synced_quantity_kg` for each "
            "`year × exporter × importer × product` observation."
        ),
        "",
        (
            "When both usable quantities are available, the reconciled trade "
            "value is used as the common value anchor. Exporter quantity, "
            "importer quantity, and their simple bilateral mean each imply a "
            "unit value. Trade Sync selects the quantity candidate whose "
            "implied unit value is closest to the robust product-year "
            "unit-value reference. If only one usable quantity exists, it is "
            "retained unchanged; if neither exists, the synchronized quantity "
            "remains missing."
        ),
        "",
        "### 9.1 Final Trade-Synced quantity",
        "",
        (
            "The following table summarizes the final quantity outcome applied "
            "by Trade Sync. Percentages for the unit-value-guided methods use "
            "rows with usable quantities on both sides as the denominator."
        ),
        "",
        _markdown_table(quantity_trade_sync_summary),
        "",
        (
            "For bilateral observations, the next table shows how candidate "
            "selection changes as exporter–importer quantity disagreement "
            "becomes more severe."
        ),
        "",
        _markdown_table(
            quantity_trade_sync_band_display.rename(
                columns={
                    "band_label": "Quantity difference",
                    "n_rows": "Rows",
                    "trade_synced_export_share": (
                        "Exporter selected (%)"
                    ),
                    "trade_synced_import_share": (
                        "Importer selected (%)"
                    ),
                    "trade_synced_mean_share": (
                        "Mean selected (%)"
                    ),
                    "trade_synced_no_reference_share": (
                        "Mean — no reference (%)"
                    ),
                }
            )
        ),
        "",
        "",
        "### 9.2 Quantity availability",
        "",
        "| Quantity availability | Rows |",
        "|---|---:|",
        (
            "| Both exporter and importer quantity usable | "
            f"{_format_integer(n_both_quantity)} |"
        ),
        (
            "| Exporter quantity only | "
            f"{_format_integer(n_export_quantity_only)} |"
        ),
        (
            "| Importer quantity only | "
            f"{_format_integer(n_import_quantity_only)} |"
        ),
        (
            "| Neither quantity usable | "
            f"{_format_integer(n_no_quantity)} |"
        ),
        "",
        "### 9.3 Mirror-report disagreement",
        "",
        (
            "Among rows with usable quantities on both sides, the median "
            "difference relative to the larger quantity is "
            f"`{_format_percent(median_quantity_difference)}`. "
            "The P90, P95, and P99 differences are "
            f"`{_format_percent(p90_quantity_difference)}`, "
            f"`{_format_percent(p95_quantity_difference)}`, and "
            f"`{_format_percent(p99_quantity_difference)}`, respectively."
        ),
        "",
        (
            f"`{_format_percent(quantity_within_10pct)}` of bilateral "
            "quantity pairs agree within 10%."
        ),
        "",
        markdown_image(
            "Distribution of bilateral quantity differences",
            figure_paths["quantity_deviation_distribution"],
            document_path=report_path,
        ),
        "",
        "### 9.4 Quantity reporting reliability",
        "",
        (
            "The median quantity-specific reliability index is "
            f"`{_format_percent(median_export_quantity_reliability)}` "
            "for exporter-side reporting and "
            f"`{_format_percent(median_import_quantity_reliability)}` "
            "for importer-side reporting."
        ),
        "",
        (
            "The median absolute exporter–importer reliability gap within "
            "bilateral quantity observations is "
            f"`{_format_percent(median_quantity_reliability_gap)}`."
        ),
        "",
        (
            "These quantity reliability measures are mirror-consistency "
            "signals. They indicate how consistently a reporting side agrees "
            "with its mirror evidence; they are not direct observations of "
            "the unknown true quantity."
        ),
        "",
        "### 9.5 Reconciliation method composition",
        "",
        _markdown_table(quantity_method_summary),
        "",
        (
            "When both usable quantities are available and quantity-specific "
            "reliability weights can be formed, Trade Sync applies the "
            "reliability-weighted quantity. Bilateral rows without usable "
            "quantity reliability fall back to the simple mean. Unilateral "
            "usable quantities are retained unchanged."
        ),
        "",
        "### 9.6 Effect of reconciliation",
        "",
        (
            "Relative to a simple bilateral mean, the final reconciled "
            "quantity differs by a median absolute amount of "
            f"`{_format_percent(median_quantity_reconciled_vs_mean)}` "
            "and by "
            f"`{_format_percent(p90_quantity_reconciled_vs_mean)}` "
            "at the 90th percentile."
        ),
        "",
        (
            "Compared with applying the existing trade-value reliability "
            "weights to quantity, the quantity-specific reliability candidate "
            "differs by a median absolute amount of "
            f"`{_format_percent(median_quantity_reliability_vs_value_weighted)}` "
            "and a P90 amount of "
            f"`{_format_percent(p90_quantity_reliability_vs_value_weighted)}`."
        ),
        "",
        markdown_image(
            "Final quantity reconciliation versus simple bilateral mean",
            figure_paths["quantity_reconciliation_effect"],
            document_path=report_path,
        ),
        "",
        "### 9.7 Unit-value plausibility",
        "",
        (
            "A product-year implied-unit-value reference can be evaluated for "
            f"`{_format_integer(n_plausibility_comparable)}` bilateral "
            "quantity observations."
        ),
        "",
        (
            "Among rows with comparable plausibility evidence, the exporter "
            "quantity is closer to the product-year unit-value reference in "
            f"`{_format_percent(export_more_plausible_share)}` of cases, "
            "while the importer quantity is closer in "
            f"`{_format_percent(import_more_plausible_share)}`."
        ),
        "",
        (
            "This plausibility evidence is intentionally diagnostic at this "
            "stage. It has not been folded into the production quantity "
            "weights."
        ),
        "",
        "### 9.8 Extreme-disagreement evidence",
        "",
        (
            f"`{_format_integer(n_extreme_quantity_disagreement)}` bilateral "
            "quantity observations differ by more than 90% relative to the "
            "larger report."
        ),
        "",
        (
            "Of these extreme-disagreement rows, "
            f"`{_format_integer(n_extreme_with_plausibility)}` have usable "
            "unit-value plausibility evidence."
        ),
        "",
        (
            "A clear difference of at least one robust-distance unit between "
            "the two implied-unit-value candidates occurs in "
            f"`{_format_percent(extreme_clear_plausibility_share)}` of the "
            "extreme rows with comparable plausibility evidence."
        ),
        "",
        _markdown_table(
            quantity_band_display.rename(
                columns={
                    "band_label": "Quantity difference",
                    "n_rows": "Rows",
                    "median_quantity_reliability_gap": (
                        "Median reliability gap (%)"
                    ),
                    "median_reconciled_vs_mean_fraction": (
                        "Median reconciliation effect (%)"
                    ),
                    "plausibility_evidence_share": (
                        "Plausibility evidence (%)"
                    ),
                    "clear_plausibility_preference_share": (
                        "Clear plausibility preference (%)"
                    ),
                }
            )
        ),
        "",
        markdown_image(
            "Unit-value plausibility by quantity-disagreement severity",
            figure_paths[
                "quantity_plausibility_by_disagreement"
            ],
            document_path=report_path,
        ),
        "",
        "### 9.9 Plausibility-first counterfactual",
        "",
        (
            "To evaluate whether row-specific unit-value evidence could improve "
            "the current reliability-only reconciliation, a counterfactual "
            "quantity was constructed. When exporter and importer implied "
            "unit-value robust distances differ by at least one unit, the "
            "quantity associated with the smaller distance is selected. "
            "Otherwise the current reliability-weighted quantity is retained."
        ),
        "",
        (
            f"This rule is decisive for "
            f"`{_format_integer(n_plausibility_decisive)}` bilateral rows "
            f"({_format_percent(plausibility_decisive_share)} of bilateral "
            "quantity observations)."
        ),
        "",
        (
            f"The exporter quantity is selected in "
            f"`{_format_integer(n_plausibility_export_selected)}` cases and "
            f"the importer quantity in "
            f"`{_format_integer(n_plausibility_import_selected)}` cases."
        ),
        "",
        (
            "Where the plausibility-first rule is decisive, it changes the "
            "current reconciled quantity by a median absolute amount of "
            f"`{_format_percent(median_plausibility_first_change)}` and a "
            f"P90 amount of "
            f"`{_format_percent(p90_plausibility_first_change)}`."
        ),
        "",
        (
            "Most importantly, the plausibility-first candidate produces a "
            "smaller product-year unit-value robust distance than the current "
            "reliability-weighted quantity in "
            f"`{_format_percent(plausibility_first_improves_share)}` of "
            "comparable decisive cases."
        ),
        "",
        (
            "The median reduction in robust unit-value distance is "
            f"`{median_robust_distance_improvement:.3f}`."
            if median_robust_distance_improvement is not None
            else (
                "The median robust-distance improvement could not be evaluated."
            )
        ),
        "",
        (
            "This remains a counterfactual diagnostic. The production "
            "`quantity_reconciled_kg` is unchanged."
        ),
        "",

        "### 9.10 Product-year unit-value reference quantity",
        "",
        (
            "For each bilateral product-year observation, Trade Sync "
            "derives a diagnostic reference quantity from the reconciled "
            "trade value and the robust product-year median unit value:"
        ),
        "",
        (
            "`reference quantity = reconciled trade value / "
            "product-year median unit value`"
        ),
        "",
        (
            "This reference is not treated as the true quantity. It is "
            "used as a product-year benchmark against which the exporter "
            "quantity, importer quantity, and current reliability-weighted "
            "quantity can be compared."
        ),
        "",
        _markdown_table(quantity_reference_summary),
        "",
        (
            "The following table shows how these relationships change "
            "as exporter-importer quantity disagreement becomes more severe."
        ),
        "",
        _markdown_table(quantity_band_display),
        "",
        "### 9.11 Methodological assessment",
        "",
        (
            "The current quantity reconciliation therefore uses two distinct "
            "levels of evidence. Quantity-specific reporter reliability "
            "determines the production bilateral weights, while reconciled-value "
            "unit-value plausibility independently tests whether one quantity "
            "candidate appears substantially more consistent with the "
            "product-year value structure."
        ),
        "",
        (
            "The unit-value evidence should be interpreted as a diagnostic "
            "cross-check rather than proof that one reporter is correct. "
            "The disagreement-band and extreme-case results determine whether "
            "this evidence is sufficiently discriminating and stable to justify "
            "incorporating it into a later quantity-reconciliation formulation."
        ),
        "",
    
        "## 10. Reconciliation composition",
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
        "## 11. Annual results",
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
        "## 12. Interpretation and limitations",
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
        "## 13. Output artifacts",
        "",
        f"- Reconciled trade and quantity: `{reconciled_trade_path}`",
        f"- Country reliability: `{country_reliability_path}`",
        f"- Reconciliation audit: `{reconciliation_audit_path}`",
        (
            f"- Quantity reconciliation audit: "
            f"`{quantity_reconciliation_audit_path}`"
        ),
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
