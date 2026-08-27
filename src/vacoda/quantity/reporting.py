"""
vacoda.quantity.reporting

Create the final Quantity decision report and diagnostic figures.

This report is intentionally diagnostic-first: it documents the current
synchronization result while also exposing candidate evidence that is not yet
admitted by synchronize.py (notably alternate quantity in kg-equivalent units).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from vacoda.config.paths import get_figure_dir
from vacoda.config.quantity import QuantityConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.markdown import (
    insert_markdown_toc,
    markdown_image,
)
from vacoda.utils.sql import sql_literal
from vacoda.utils.display import get_table_figure_layout, get_table_pair
from vacoda.visualization.relationships import plot_figure_pair

# ======================================================================
# Helpers
# ======================================================================

UNIT_COLORS = {
    "kg": "tab:orange",
    "N/A": "tab:blue",
    "m": "tab:green",
    "m²": "tab:red",
    "m³": "tab:purple",
    "u": "tab:brown",
    "1000u": "tab:pink",
    "12u": "tab:gray",
    "2u": "tab:olive",
    "BBL": "tab:cyan",
    "GT": "goldenrod",
    "U (jeu/pack)": "teal",
    "ct/l": "slategray",
    "l": "steelblue",
    "kg 90% sdt": "darkkhaki",
}

QUANTITY_FIELD_ORDER = [
    "quantity",
    "alt_quantity",
    "net_weight_kg",
    "gross_weight_kg",
]

def valid_output(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _format_integer(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{int(round(float(value))):,}"


def _format_percent(
    value: float | None,
    *,
    digits: int = 2,
    already_percent: bool = False,
) -> str:
    if value is None or pd.isna(value):
        return "NA"
    numeric = float(value)
    if not already_percent:
        numeric *= 100.0
    return f"{numeric:.{digits}f}%"


def _format_number(
    value: float | None,
    *,
    digits: int = 4,
) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):,.{digits}f}"

def _population_share(
    rows: float | int | None,
    population: float | int,
) -> float | None:
    """Return rows as percentage of the full Quantity input population."""
    if rows is None or pd.isna(rows) or not population:
        return None

    return 100.0 * float(rows) / float(population)

def _markdown_table(dataframe: pd.DataFrame, *, index: bool = False) -> str:
    if dataframe.empty:
        return "_No rows available._"
    return dataframe.to_markdown(index=index, floatfmt=".4f")


def _diagnostic_group(
    diagnostics: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    return diagnostics.loc[
        diagnostics["diagnostic_group"].eq(name)
    ].copy()


def _metric_value(frame: pd.DataFrame, *metric_names: str) -> Any:
    if frame.empty:
        return None
    metric_column = frame["metric"].astype(str)
    for metric_name in metric_names:
        match = frame.loc[metric_column.eq(metric_name)]
        if not match.empty:
            return match.iloc[0]["value"]
    return None


def _as_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _simple_rows_table(
    frame: pd.DataFrame,
    *,
    metric_name: str,
    percentage_name: str = "Share (%)",
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame[["metric", "value", "percentage"]].copy()
    out.columns = [metric_name, "Rows", percentage_name]
    out["Rows"] = out["Rows"].round().astype("Int64")
    return out.sort_values("Rows", ascending=False)


def _source_state_table(
    frame: pd.DataFrame,
    *,
    percentage_name: str = "Share (%)",
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame[["metric", "category_1", "value", "percentage"]].copy()
    out.columns = ["Candidate source", "Evidence state", "Rows", percentage_name]
    out["Rows"] = out["Rows"].round().astype("Int64")
    return out.sort_values(["Candidate source", "Evidence state"])


def _top_hs_table(frame: pd.DataFrame, *, n: int = 20) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame[["metric", "value", "percentage"]].copy()
    out.columns = ["HS code", "Rows", "Share (%)"]
    out["Rows"] = out["Rows"].round().astype("Int64")
    return out.sort_values("Rows", ascending=False).head(n)


def _agreement_state_table(
    frame: pd.DataFrame,
    exact_frame: pd.DataFrame,
    tolerance_frame: pd.DataFrame,
    agreement_tolerance: float,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    out = frame[
        [
            "metric",
            "category_1",
            "value",
            "percentage",
            "category_2",
            "category_3",
        ]
    ].copy()

    out.columns = [
        "Comparison",
        "Evidence state",
        "Rows compared",
        "Population share (%)",
        "Median ratio",
        "Median relative gap",
    ]

    # Exact matches
    if not exact_frame.empty:
        exact = exact_frame[
            [
                "metric",
                "category_1",
                "value",
                "percentage",
            ]
        ].copy()

        exact.columns = [
            "Comparison",
            "Evidence state",
            "Exact-match rows",
            "Exact-match population share (%)",
        ]

        out = out.merge(
            exact,
            on=["Comparison", "Evidence state"],
            how="left",
        )

    # Within configured tolerance
    if not tolerance_frame.empty:
        tolerance = tolerance_frame[
            [
                "metric",
                "category_1",
                "value",
                "percentage",
            ]
        ].copy()

        tolerance.columns = [
            "Comparison",
            "Evidence state",
            f"Within {100 * agreement_tolerance:.0f}% rows",
            f"Within {100 * agreement_tolerance:.0f}% population share (%)",
        ]

        out = out.merge(
            tolerance,
            on=["Comparison", "Evidence state"],
            how="left",
        )

    # Split reported_reported etc.
    out = _split_evidence_state(
        out,
        column="Evidence state",
        left_name="First-source evidence",
        right_name="Second-source evidence",
    )

    # Row counts
    row_columns = [
        "Rows compared",
        "Exact-match rows",
        f"Within {100 * agreement_tolerance:.0f}% rows",
    ]

    for column in row_columns:
        if column in out.columns:
            out[column] = (
                pd.to_numeric(out[column], errors="coerce")
                .round()
                .astype("Int64")
            )

    # Numeric statistics
    out = _as_numeric(
        out,
        [
            "Population share (%)",
            "Exact-match population share (%)",
            f"Within {100 * agreement_tolerance:.0f}% population share (%)",
            "Median ratio",
            "Median relative gap",
        ],
    )

    # Reported before Estimated
    evidence_order = {
        "Reported": 0,
        "Estimated": 1,
    }

    out = (
        out
        .assign(
            _first_order=out[
                "First-source evidence"
            ].map(evidence_order),
            _second_order=out[
                "Second-source evidence"
            ].map(evidence_order),
        )
        .sort_values(
            [
                "Comparison",
                "_first_order",
                "_second_order",
            ]
        )
        .drop(
            columns=[
                "_first_order",
                "_second_order",
            ]
        )
    )

    return out[
        [
            "Comparison",
            "First-source evidence",
            "Second-source evidence",
            "Rows compared",
            "Population share (%)",
            "Exact-match rows",
            "Exact-match population share (%)",
            f"Within {100 * agreement_tolerance:.0f}% rows",
            f"Within {100 * agreement_tolerance:.0f}% population share (%)",
            "Median ratio",
            "Median relative gap",
        ]
    ]


def _split_evidence_state(
    frame: pd.DataFrame,
    *,
    column: str,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    """Split states such as 'reported_estimated' into explicit columns."""
    frame = frame.copy()

    split = frame[column].astype(str).str.split("_", n=1, expand=True)

    frame.insert(
        frame.columns.get_loc(column),
        left_name,
        split[0].str.replace("_", " ").str.capitalize(),
    )
    frame.insert(
        frame.columns.get_loc(column) + 1,
        right_name,
        split[1].str.replace("_", " ").str.capitalize(),
    )

    return frame.drop(columns=[column])

# ======================================================================
# Figures
# ======================================================================


def _save_figure(figure, output_path: Path) -> None:
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_unit_distribution(
    *,
    diagnostics: pd.DataFrame,
    diagnostic_group: str,
    title: str,
    height: float = 4.5,
    width: float = 10.0,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, diagnostic_group)

    if frame.empty:
        return

    frame = (
        frame.groupby("metric", as_index=False)
        .agg(rows=("value", "sum"))
        .sort_values("rows", ascending=True)
    )

    total = frame["rows"].sum()

    if total <= 0:
        return

    frame["share"] = 100.0 * frame["rows"] / total

    figure_height = max(height, 0.38 * len(frame))

    figure, axis = plt.subplots(
        figsize=(width, figure_height)
    )

    bars = axis.barh(
        frame["metric"].astype(str),
        frame["share"],
    )

    axis.bar_label(
        bars,
        labels=[
            f"{value:.1f}%"
            if value >= 0.1
            else f"{value:.3f}%"
            for value in frame["share"]
        ],
        padding=4,
        fontsize=8,
    )

    axis.set_title(title)
    axis.set_xlabel("Share of observations (%)")
    axis.set_ylabel("Quantity unit")

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )
    axis.set_axisbelow(True)

    # Leave room for labels at the end of bars.
    axis.set_xlim(
        0,
        max(frame["share"].max() * 1.15, 5),
    )

    _save_figure(figure, output_path)

def _plot_unit_share_by_year(
    *,
    diagnostics: pd.DataFrame,
    diagnostic_group: str,
    title: str,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, diagnostic_group)
    frame = frame.dropna(subset=["year", "metric", "percentage"])

    if frame.empty:
        return

    pivot = frame.pivot_table(
        index="year",
        columns="metric",
        values="percentage",
        aggfunc="sum",
        fill_value=0.0,
    )

    colors = [
        UNIT_COLORS.get(str(column), "lightgray")
        for column in pivot.columns
    ]

    figure, axis = plt.subplots(figsize=(11, 6))

    pivot.plot(
        kind="bar",
        stacked=True,
        ax=axis,
        color=colors,
    )

    axis.set_title(title)
    axis.set_xlabel("Year")
    axis.set_ylabel("Share of observations (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.4)

    axis.legend(
        title="Unit",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    _save_figure(figure, output_path)


def _plot_agreement_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, "quantity_net_weight_by_year")
    frame = frame.dropna(subset=["year", "percentage"]).sort_values("year")
    if frame.empty:
        return
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(frame["year"], frame["percentage"], marker="o")
    axis.set_title(
    "Quantity/net-weight agreement contribution by year"
    )
    axis.set_ylabel(
        "Full-population share within tolerance (%)"
    )
    axis.set_xlabel("Year")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    _save_figure(figure, output_path)


def _plot_relative_gap_distribution(
    *,
    con: duckdb.DuckDBPyConnection,
    synchronized_sql: str,
    output_path: Path,
) -> None:
    frame = con.execute(
        f"""
        SELECT quantity_net_weight_relative_gap AS relative_gap
        FROM read_parquet({synchronized_sql})
        WHERE quantity_net_weight_relative_gap IS NOT NULL
          AND isfinite(quantity_net_weight_relative_gap)
        """
    ).fetchdf()
    if frame.empty:
        return
    values = frame["relative_gap"].dropna()
    if values.empty:
        return
    clipped = values.clip(lower=0, upper=0.50)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(100.0 * clipped, bins=50)
    axis.set_title("Distribution of quantity–net-weight relative gaps")
    axis.set_xlabel("Symmetric relative difference (%)")
    axis.set_ylabel("Number of observations")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    _save_figure(figure, output_path)


def _plot_alt_agreement_by_evidence_state(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(
        diagnostics,
        "alt_quantity_kg_agreement",
    )

    frame = frame.dropna(
        subset=["metric", "category_1", "percentage"]
    )

    if frame.empty:
        return

    frame["evidence_state"] = (
        frame["category_1"]
        .astype(str)
        .str.replace("_", " / ")
        .str.title()
    )

    pivot = frame.pivot_table(
        index="evidence_state",
        columns="metric",
        values="percentage",
        aggfunc="first",
    ).sort_index()

    figure, axis = plt.subplots(figsize=(11, 6))

    pivot.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(
        "Alternate-quantity kg agreement by evidence state"
    )

    axis.set_xlabel("Evidence state")
    axis.set_ylabel(
        "Rows within configured tolerance (%)"
    )
    axis.set_ylim(0, 100)
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )
    axis.legend(title="Comparison")

    _save_figure(
        figure,
        output_path,
    )


def _plot_net_weight_coverage_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, "net_weight_by_year")
    frame = frame.dropna(subset=["year", "metric", "percentage"])
    if frame.empty:
        return
    pivot = frame.pivot_table(
        index="year",
        columns="metric",
        values="percentage",
        aggfunc="first",
        fill_value=0.0,
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    for column in pivot.columns:
        axis.plot(pivot.index, pivot[column], marker="o", label=str(column))
    axis.set_title("Positive net-weight coverage by year")
    axis.set_xlabel("Year")
    axis.set_ylabel("Positive coverage within evidence-state rows (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.legend(title="Evidence state")
    _save_figure(figure, output_path)

def _plot_synchronized_quantity_distribution(
    *,
    con: duckdb.DuckDBPyConnection,
    synchronized_sql: str,
    output_path: Path,
    bin_width: float = 0.25,
) -> None:
    """
    Plot the distribution of positive synchronized_quantity_kg
    on a log10 scale.

    Histogram bins are calculated in DuckDB so the full synchronized
    population does not need to be loaded into pandas.
    """

    frame = con.execute(
        f"""
        WITH positive AS (
            SELECT
                LOG10(synchronized_quantity_kg) AS log_weight
            FROM read_parquet({synchronized_sql})
            WHERE synchronized_quantity_kg > 0
              AND isfinite(synchronized_quantity_kg)
        )

        SELECT
            FLOOR(log_weight / {bin_width}) * {bin_width}
                AS log_bin,
            COUNT(*) AS n_rows

        FROM positive
        GROUP BY log_bin
        ORDER BY log_bin
        """
    ).fetchdf()

    if frame.empty:
        return

    median_log_weight = con.execute(
        f"""
        SELECT MEDIAN(LOG10(synchronized_quantity_kg))
        FROM read_parquet({synchronized_sql})
        WHERE synchronized_quantity_kg > 0
          AND isfinite(synchronized_quantity_kg)
        """
    ).fetchone()[0]

    figure, axis = plt.subplots(figsize=(11, 6))

    axis.bar(
        frame["log_bin"],
        frame["n_rows"],
        width=bin_width * 0.95,
        align="edge",
    )

    if median_log_weight is not None:
        median_weight = 10 ** float(median_log_weight)

        axis.axvline(
            median_log_weight,
            linestyle="--",
            linewidth=1.5,
            label=f"Median = {median_weight:,.0f} kg",
        )

        axis.legend()

    axis.set_title(
        "Distribution of positive synchronized quantity"
    )
    axis.set_xlabel(
        "Synchronized quantity (kg, log10 scale)"
    )
    axis.set_ylabel("Number of observations")

    minimum_tick = int(frame["log_bin"].min())
    maximum_tick = int(frame["log_bin"].max()) + 1

    ticks = list(
        range(minimum_tick, maximum_tick + 1)
    )

    def tick_label(power: int) -> str:
        value = 10 ** power

        if power < 0:
            return f"{value:g}"
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:g}B"
        if value >= 1_000_000:
            return f"{value / 1_000_000:g}M"
        if value >= 1_000:
            return f"{value / 1_000:g}k"

        return f"{value:g}"

    axis.set_xticks(ticks)
    axis.set_xticklabels(
        [tick_label(tick) for tick in ticks]
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    _save_figure(
        figure,
        output_path,
    )

def _plot_net_weight_distribution(
    *,
    con: duckdb.DuckDBPyConnection,
    prepared_sql: str,
    output_path: Path,
    bin_width: float = 0.25,
) -> None:
    """
    Plot the distribution of positive net_weight_kg on a log10 scale.

    Histogram bins are calculated in DuckDB so the full Quantity
    population does not need to be loaded into pandas.
    """

    frame = con.execute(
        f"""
        WITH positive AS (
            SELECT
                LOG10(net_weight_kg) AS log_weight
            FROM read_parquet({prepared_sql})
            WHERE net_weight_kg > 0
              AND isfinite(net_weight_kg)
        )

        SELECT
            FLOOR(log_weight / {bin_width}) * {bin_width}
                AS log_bin,
            COUNT(*) AS n_rows

        FROM positive
        GROUP BY log_bin
        ORDER BY log_bin
        """
    ).fetchdf()

    if frame.empty:
        return

    median_log_weight = con.execute(
        f"""
        SELECT MEDIAN(LOG10(net_weight_kg))
        FROM read_parquet({prepared_sql})
        WHERE net_weight_kg > 0
          AND isfinite(net_weight_kg)
        """
    ).fetchone()[0]

    figure, axis = plt.subplots(figsize=(11, 6))

    axis.bar(
        frame["log_bin"],
        frame["n_rows"],
        width=bin_width * 0.95,
        align="edge",
    )

    if median_log_weight is not None:
        median_weight = 10 ** float(median_log_weight)

        axis.axvline(
            median_log_weight,
            linestyle="--",
            linewidth=1.5,
            label=f"Median = {median_weight:,.0f} kg",
        )

        axis.legend()

    axis.set_title("Distribution of positive net weight")
    axis.set_xlabel("Net weight (kg, log10 scale)")
    axis.set_ylabel("Number of observations")

    minimum_tick = int(frame["log_bin"].min())
    maximum_tick = int(frame["log_bin"].max()) + 1

    ticks = list(range(minimum_tick, maximum_tick + 1))

    def tick_label(power: int) -> str:
        value = 10 ** power

        if power < 0:
            return f"{value:g}"
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:g}B"
        if value >= 1_000_000:
            return f"{value / 1_000_000:g}M"
        if value >= 1_000:
            return f"{value / 1_000:g}k"

        return f"{value:g}"

    axis.set_xticks(ticks)
    axis.set_xticklabels(
        [tick_label(tick) for tick in ticks]
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    _save_figure(
        figure,
        output_path,
    )

def _plot_net_weight_log_distribution_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, "net_weight_distribution_by_year")
    if frame.empty:
        return
    frame = _as_numeric(frame, ["category_1", "category_2", "category_3"])
    frame = frame.dropna(subset=["year", "category_1", "category_2", "category_3"])
    if frame.empty:
        return
    frame = frame.sort_values("year")
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(frame["year"], frame["category_1"], marker="o", label="p10")
    axis.plot(frame["year"], frame["category_2"], marker="o", label="median")
    axis.plot(frame["year"], frame["category_3"], marker="o", label="p90")
    axis.set_title("Net-weight distribution by year")
    axis.set_xlabel("Year")
    axis.set_ylabel("log10(net_weight_kg)")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.legend()
    _save_figure(figure, output_path)


def _plot_gross_net_ratio_distribution(
    *,
    con: duckdb.DuckDBPyConnection,
    prepared_sql: str,
    output_path: Path,
) -> None:
    frame = con.execute(
        f"""
        SELECT gross_weight_kg / net_weight_kg AS ratio
        FROM read_parquet({prepared_sql})
        WHERE gross_weight_kg > 0
          AND net_weight_kg > 0
          AND isfinite(gross_weight_kg / net_weight_kg)
        """
    ).fetchdf()
    if frame.empty:
        return
    values = frame["ratio"].dropna()
    if values.empty:
        return
    # Plotting is clipped only for readability; report statistics remain full.
    clipped = values.clip(lower=0, upper=5.0)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(clipped, bins=60)
    axis.axvline(1.0, linestyle="--")
    axis.set_title("Gross-weight / net-weight ratio distribution")
    axis.set_xlabel("gross_weight_kg / net_weight_kg (clipped at 5)")
    axis.set_ylabel("Number of observations")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    _save_figure(figure, output_path)


def _plot_hierarchy_coverage(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, "methodology_funnel")
    if frame.empty:
        return
    frame["step_order"] = pd.to_numeric(frame["category_1"], errors="coerce")
    frame = frame.sort_values("step_order")
    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(frame["metric"], frame["percentage"])
    axis.bar_label(
        bars,
        labels=[f"{value:.2f}%" for value in frame["percentage"]],
        padding=3,
        fontsize=8,
    )
    axis.set_title("Cumulative kg coverage by synchronization methodology step")
    axis.set_xlabel("Methodology step")
    axis.set_ylabel("Cumulative population coverage (%)")
    axis.set_ylim(0, 100)
    axis.tick_params(axis="x", labelrotation=25)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    _save_figure(figure, output_path)


def _plot_synchronization_status_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, "synchronization_status_by_year")
    frame = frame.dropna(subset=["year", "metric", "percentage"])
    if frame.empty:
        return
    pivot = frame.pivot_table(
        index="year",
        columns="metric",
        values="percentage",
        aggfunc="sum",
        fill_value=0.0,
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", stacked=True, ax=axis)
    axis.set_title("Current Quantity synchronization outcomes by year")
    axis.set_xlabel("Year")
    axis.set_ylabel("Share of rows (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.legend(title="Synchronization status", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_figure(figure, output_path)


def _plot_m3_by_hs(
    *,
    diagnostics: pd.DataFrame,
    diagnostic_group: str,
    title: str,
    output_path: Path,
    maximum_codes: int = 30,
    height: float = 6.0,
    hs_level: str = "HS6",
) -> None:
    frame = _diagnostic_group(
        diagnostics,
        diagnostic_group,
    )

    if frame.empty:
        return

    frame = (
        frame.groupby("metric", as_index=False)
        .agg(
            rows=("value", "sum"),
            share=("percentage", "sum"),
        )
        .nlargest(maximum_codes, "rows")
        .sort_values("share", ascending=True)
    )

    if frame.empty:
        return

    labels = frame["metric"].astype(str).tolist()
    values = frame["share"].astype(float).to_numpy()

    y_positions = range(len(frame))

    figure_height = max(
        height,
        0.32 * len(frame) + 2.0,
    )

    figure, axis = plt.subplots(
        figsize=(10, figure_height)
    )

    bars = axis.barh(
        y_positions,
        values,
    )

    axis.set_yticks(list(y_positions))
    axis.set_yticklabels(labels)

    axis.bar_label(
        bars,
        labels=[
            f"{value:.2f}%"
            for value in values
        ],
        padding=3,
        fontsize=8,
    )

    axis.set_title(title)
    axis.set_xlabel("Share of m³ observations (%)")
    axis.set_ylabel(f"{hs_level} code")
    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )
    axis.set_axisbelow(True)

    axis.set_xlim(
        0,
        max(values) * 1.15,
    )

    _save_figure(
        figure,
        output_path,
    )


# ======================================================================
# Main reporting function
# ======================================================================


def build_quantity_report(
    *,
    cfg: QuantityConfig,
    prepared_input_path: Path,
    reconciliation_evidence_path: Path,
    synchronized_quantity_path: Path,
    diagnostics_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """Create the final Quantity Markdown decision report."""

    prepared_input_path = Path(prepared_input_path)
    reconciliation_evidence_path = Path(reconciliation_evidence_path)
    synchronized_quantity_path = Path(synchronized_quantity_path)
    diagnostics_path = Path(diagnostics_path)

    for label, path in (
        ("prepared Quantity input", prepared_input_path),
        ("Quantity reconciliation evidence", reconciliation_evidence_path),
        ("synchronized Quantity output", synchronized_quantity_path),
        ("Quantity diagnostics", diagnostics_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    if not cfg.artifacts.enable:
        return {
            "status": "disabled",
            "report_path": None,
            "latest_report_path": None,
            "ready_for_cif_fob": False,
        }

    report_path = cfg.artifacts_path(run_id)
    latest_path = cfg.artifacts_latest_path()
    figure_dir = (
        get_figure_dir()
        / "quantity"
        / f"target_{cfg.scope.target_hs.lower()}"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # if not force and valid_output(report_path):
    #     logger.info("REUSE — Quantity report already exists: %s", report_path)
    #     return {
    #         "status": "reused",
    #         "output_path": report_path,
    #         "report_path": report_path,
    #         "latest_report_path": latest_path if latest_path.exists() else None,
    #         "ready_for_cif_fob": True,
    #     }

    prepared_sql = sql_literal(prepared_input_path)
    synchronized_sql = sql_literal(synchronized_quantity_path)
    diagnostics_sql = sql_literal(diagnostics_path)

    agreement_tolerance = float(
        cfg.method.kg_reconciliation.agreement.relative_tolerance
    )

    info = get_symbols("info")
    ok = get_symbols("check")

    logger.info("%s Building Quantity decision report.", info)
    logger.info("  Prepared input: %s", prepared_input_path)
    logger.info("  Reconciliation evidence: %s", reconciliation_evidence_path)
    logger.info("  Synchronized quantity: %s", synchronized_quantity_path)
    logger.info("  Diagnostics: %s", diagnostics_path)
    logger.info("  Report: %s", report_path)

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(cfg.engine.threads)}")
        diagnostics = con.execute(
            f"SELECT * FROM read_parquet({diagnostics_sql})"
        ).fetchdf()

        # --------------------------------------------------------------
        # Population
        # --------------------------------------------------------------
        population_row = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,

                COUNT(*) FILTER (
                    WHERE synchronized_quantity_kg > 0
                ) AS n_synchronized,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status =
                        'reported_three_exact'
                ) AS n_reported_three_exact,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status LIKE
                        'reported_pair_exact_%'
                ) AS n_reported_pair_exact,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status LIKE
                        'reported_%_precedence'
                ) AS n_reported_precedence,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status LIKE
                        'estimated_%_fallback'
                ) AS n_estimated_fallback,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status =
                        'no_kg_evidence'
                ) AS n_no_kg,

                COUNT(*) FILTER (
                    WHERE NOT COALESCE(quantity_kg_candidate > 0, FALSE)
                      AND NOT COALESCE(alt_quantity_kg_candidate > 0, FALSE)
                      AND NOT COALESCE(net_weight_kg_candidate > 0, FALSE)
                      AND COALESCE(has_gross_weight_kg, FALSE)
                ) AS n_potential_gross_weight_fallback

            FROM read_parquet({synchronized_sql})
            """
        ).fetchone()

        (
            n_rows,
            n_synchronized,
            n_reported_three_exact,
            n_reported_pair_exact,
            n_reported_precedence,
            n_estimated_fallback,
            n_no_kg,
            n_potential_gross_weight_fallback,
        ) = population_row

        synchronized_share = n_synchronized / n_rows if n_rows else None
        no_kg_share = n_no_kg / n_rows if n_rows else None

        # --------------------------------------------------------------
        # Inventory
        # --------------------------------------------------------------
        inventory = _diagnostic_group(diagnostics, "field_inventory")
        inventory_rows = []

        for field_name in QUANTITY_FIELD_ORDER if not inventory.empty else []:
            field_frame = inventory.loc[inventory["metric"].eq(field_name)]

            def get_stat(statistic: str, column: str):
                match = field_frame.loc[field_frame["category_1"].eq(statistic)]
                return None if match.empty else match.iloc[0][column]

            inventory_rows.append(
                {
                    "Field": field_name,
                    "Total rows": _format_integer(get_stat("total_rows", "value")),
                    "Positive rows": _format_integer(get_stat("positive_rows", "value")),
                    "Positive share": _format_percent(
                        get_stat("positive_rows", "percentage"),
                        already_percent=True,
                    ),
                    "Zero rows": _format_integer(get_stat("zero_rows", "value")),
                    "Zero share": _format_percent(
                        get_stat("zero_rows", "percentage"),
                        already_percent=True,
                    ),
                    "Negative rows": _format_integer(get_stat("negative_rows", "value")),
                    "Negative share": _format_percent(
                        get_stat("negative_rows", "percentage"),
                        already_percent=True,
                    ),
                    "Missing rows": _format_integer(get_stat("missing_rows", "value")),
                    "Missing share": _format_percent(
                        get_stat("missing_rows", "percentage"),
                        already_percent=True,
                    ),
                }
            )
        inventory_display = pd.DataFrame(inventory_rows)

        # --------------------------------------------------------------
        # Unit distributions
        # --------------------------------------------------------------
        def unit_table(group: str) -> pd.DataFrame:
            frame = _diagnostic_group(diagnostics, group)
            if frame.empty:
                return pd.DataFrame()
            # Diagnostics may retain more than one provenance row under the
            # same canonical unit; collapse for the visible unit table.
            frame = (
                frame.groupby(["metric", "category_1"], dropna=False, as_index=False)
                .agg(Rows=("value", "sum"))
            )
            total = frame["Rows"].sum()
            frame["Share (%)"] = 100.0 * frame["Rows"] / total if total else 0.0
            frame.columns = ["Unit", "Unit code", "Rows", "Share (%)"]
            frame["Rows"] = frame["Rows"].round().astype("Int64")
            return frame.sort_values("Rows", ascending=False)

        quantity_units = unit_table("quantity_unit_distribution")
        alt_quantity_units = unit_table("alt_quantity_unit_distribution")

        quantity_m3 = _top_hs_table(
            _diagnostic_group(diagnostics, "quantity_m3_by_hs"), n=30
        )
        alt_quantity_m3 = _top_hs_table(
            _diagnostic_group(diagnostics, "alt_quantity_m3_by_hs"), n=30
        )

        quantity_m3_hs4 = _top_hs_table(
            _diagnostic_group(
                diagnostics,
                "quantity_m3_by_hs4",
            ),
            n=100,
        )

        alt_quantity_m3_hs4 = _top_hs_table(
            _diagnostic_group(
                diagnostics,
                "alt_quantity_m3_by_hs4",
            ),
            n=100,
        )

        def add_cumulative_share(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty:
                return frame

            frame = frame.copy()
            frame = frame.sort_values("Rows", ascending=False)

            frame["Cumulative share (%)"] = (
                frame["Share (%)"].cumsum()
            )

            return frame


        quantity_m3_hs4 = add_cumulative_share(
            quantity_m3_hs4
        )

        alt_quantity_m3_hs4 = add_cumulative_share(alt_quantity_m3_hs4)

        def unit_share_by_code(
            frame: pd.DataFrame,
            unit_code: int,
        ) -> float | None:
            """Return full-population share for one quantity unit."""
            if frame.empty:
                return None

            match = frame.loc[
                frame["Unit code"].astype(str).eq(str(unit_code))
            ]

            if match.empty:
                return None

            return float(match.iloc[0]["Share (%)"])


        # Full-population share reported in m³.
        quantity_m3_population_share = unit_share_by_code(
            quantity_units,
            12,
        )
        alt_quantity_m3_population_share = unit_share_by_code(
            alt_quantity_units,
            12,
        )

        # Share of all m³ observations covered by the displayed top HS codes.
        quantity_m3_top_share = (
            float(quantity_m3["Share (%)"].sum())
            if not quantity_m3.empty
            else None
        )

        alt_quantity_m3_top_share = (
            float(alt_quantity_m3["Share (%)"].sum())
            if not alt_quantity_m3.empty
            else None
        )

        # Share of the FULL Quantity population represented by those top HS codes.
        quantity_m3_top_population_share = (
            quantity_m3_population_share
            * quantity_m3_top_share
            / 100.0
            if (
                quantity_m3_population_share is not None
                and quantity_m3_top_share is not None
            )
            else None
        )

        alt_quantity_m3_top_population_share = (
            alt_quantity_m3_population_share
            * alt_quantity_m3_top_share
            / 100.0
            if (
                alt_quantity_m3_population_share is not None
                and alt_quantity_m3_top_share is not None
            )
            else None
        )

        # --------------------------------------------------------------
        # Quantity / net-weight evidence-state comparison
        # --------------------------------------------------------------

        evidence_state = _diagnostic_group(
            diagnostics,
            "quantity_net_weight_by_evidence_state",
        )

        exact_match_state = _diagnostic_group(
            diagnostics,
            "quantity_net_weight_exact_match_by_evidence_state",
        )

        within_tolerance_state = _diagnostic_group(
            diagnostics,
            "quantity_net_weight_within_tolerance_by_evidence_state",
        )

        if evidence_state.empty:
            evidence_state_display = pd.DataFrame()

        else:
            # ----------------------------------------------------------
            # Base comparison population
            # ----------------------------------------------------------
            evidence_state_display = evidence_state[
                [
                    "metric",
                    "value",
                    "percentage",
                    "category_1",
                    "category_2",
                ]
            ].copy()

            evidence_state_display.columns = [
                "Evidence state",
                "Rows compared",
                "Population share (%)",
                "Median q/net-weight ratio",
                "Median relative gap",
            ]

            # ----------------------------------------------------------
            # Exact matches
            # ----------------------------------------------------------
            if not exact_match_state.empty:
                exact_display = exact_match_state[
                    [
                        "metric",
                        "value",
                        "percentage",
                    ]
                ].copy()

                exact_display.columns = [
                    "Evidence state",
                    "Exact-match rows",
                    "Exact-match population share (%)",
                ]

                evidence_state_display = evidence_state_display.merge(
                    exact_display,
                    on="Evidence state",
                    how="left",
                )

            # ----------------------------------------------------------
            # Agreement within configured tolerance
            # ----------------------------------------------------------
            if not within_tolerance_state.empty:
                tolerance_display = within_tolerance_state[
                    [
                        "metric",
                        "value",
                        "percentage",
                    ]
                ].copy()

                tolerance_display.columns = [
                    "Evidence state",
                    f"Within-{100 * agreement_tolerance:.0f}% rows",
                    f"Within-{100 * agreement_tolerance:.0f}% population share (%)",
                ]

                evidence_state_display = evidence_state_display.merge(
                    tolerance_display,
                    on="Evidence state",
                    how="left",
                )

            # ----------------------------------------------------------
            # Numeric formatting
            # ----------------------------------------------------------
            row_columns = [
                "Rows compared",
                "Exact-match rows",
                f"Within-{100 * agreement_tolerance:.0f}% rows",
            ]

            for column in row_columns:
                if column in evidence_state_display.columns:
                    evidence_state_display[column] = (
                        pd.to_numeric(
                            evidence_state_display[column],
                            errors="coerce",
                        )
                        .round()
                        .astype("Int64")
                    )

            evidence_state_display = _as_numeric(
                evidence_state_display,
                [
                    "Population share (%)",
                    "Exact-match population share (%)",
                    f"Within-{100 * agreement_tolerance:.0f}% population share (%)",
                    "Median q/net-weight ratio",
                    "Median relative gap",
                ],
            )

            # ----------------------------------------------------------
            # Split reported_reported etc. into explicit columns
            # ----------------------------------------------------------
            evidence_state_display = _split_evidence_state(
                evidence_state_display,
                column="Evidence state",
                left_name="Quantity evidence",
                right_name="Net-weight evidence",
            )

            # ----------------------------------------------------------
            # Reported first, then estimated
            # ----------------------------------------------------------
            evidence_order = {
                "Reported": 0,
                "Estimated": 1,
            }

            evidence_state_display = (
                evidence_state_display
                .assign(
                    _quantity_order=evidence_state_display[
                        "Quantity evidence"
                    ].map(evidence_order),
                    _net_weight_order=evidence_state_display[
                        "Net-weight evidence"
                    ].map(evidence_order),
                )
                .sort_values(
                    [
                        "_quantity_order",
                        "_net_weight_order",
                    ]
                )
                .drop(
                    columns=[
                        "_quantity_order",
                        "_net_weight_order",
                    ]
                )
            )

            # ----------------------------------------------------------
            # Final visible column order
            # ----------------------------------------------------------
            evidence_state_display = evidence_state_display[
                [
                    "Quantity evidence",
                    "Net-weight evidence",
                    "Rows compared",
                    "Population share (%)",
                    "Exact-match rows",
                    "Exact-match population share (%)",
                    f"Within-{100 * agreement_tolerance:.0f}% rows",
                    f"Within-{100 * agreement_tolerance:.0f}% population share (%)",
                    "Median q/net-weight ratio",
                    "Median relative gap",
                ]
            ]

        agreement_stats = _diagnostic_group(
            diagnostics,
            "quantity_net_weight_statistics",
        )
        agreement_lookup = {
            row.metric: row.value for row in agreement_stats.itertuples()
        }

        agreement_population = _diagnostic_group(
            diagnostics,
            "quantity_net_weight_population_agreement",
        )

        if agreement_population.empty:
            agreement_population_display = pd.DataFrame()

        else:
            agreement_population_display = agreement_population[
                [
                    "metric",
                    "value",
                    "percentage",
                ]
            ].copy()

            agreement_population_display.columns = [
                "Statistic",
                "Rows",
                "Population share (%)",
            ]

            agreement_population_display["Statistic"] = (
                agreement_population_display["Statistic"]
                .replace(
                    {
                        "comparable":
                            "Quantity/net-weight comparison available",
                        "within_1pct":
                            "Agreement within 1%",
                        "within_5pct":
                            "Agreement within 5%",
                        "within_10pct":
                            "Agreement within 10%",
                    }
                )
            )

            agreement_population_display["Rows"] = (
                agreement_population_display["Rows"]
                .round()
                .astype("Int64")
            )

            agreement_population_display = _as_numeric(
                agreement_population_display,
                ["Population share (%)"],
            )

            statistic_order = {
                "Quantity/net-weight comparison available": 0,
                "Agreement within 1%": 1,
                "Agreement within 5%": 2,
                "Agreement within 10%": 3,
            }

            agreement_population_display = (
                agreement_population_display
                .assign(
                    _order=agreement_population_display[
                        "Statistic"
                    ].map(statistic_order)
                )
                .sort_values("_order")
                .drop(columns="_order")
            )


        # --------------------------------------------------------------
        # Alt kg candidate diagnostics
        # --------------------------------------------------------------
        alt_candidate_display = _source_state_table(
            _diagnostic_group(
                diagnostics,
                "alt_quantity_kg_candidate_coverage",
            ),
            percentage_name="Population share (%)",
        )
        alt_incremental_display = _source_state_table(
            _diagnostic_group(diagnostics, "alt_quantity_kg_incremental_coverage"),
            percentage_name="Incremental population coverage (%)",
        )

        alt_agreement_display = _agreement_state_table(
            _diagnostic_group(
                diagnostics,
                "alt_quantity_kg_agreement",
            ),
            _diagnostic_group(
                diagnostics,
                "alt_quantity_kg_exact_match",
            ),
            _diagnostic_group(
                diagnostics,
                "alt_quantity_kg_within_tolerance",
            ),
            agreement_tolerance,
        )

        alt_stats = _diagnostic_group(
            diagnostics,
            "alt_quantity_kg_statistics",
        )
        if alt_stats.empty:
            alt_stats_display = pd.DataFrame()
        else:
            alt_stats_display = alt_stats[
                ["metric", "value", "category_1", "category_2", "category_3"]
            ].copy()
            alt_stats_display.columns = [
                "Comparison",
                "Comparable rows",
                "Median ratio",
                "Within 1%",
                "Within 5%",
            ]
            alt_stats_display["Comparable rows"] = (
                alt_stats_display["Comparable rows"].round().astype("Int64")
            )
            alt_stats_display = _as_numeric(
                alt_stats_display,
                ["Median ratio", "Within 1%", "Within 5%"],
            )


        three_source_consensus = _diagnostic_group(
            diagnostics,
            "three_source_consensus",
        )

        if three_source_consensus.empty:
            three_source_consensus_display = pd.DataFrame()
        else:
            three_source_consensus_display = (
                three_source_consensus[
                    ["metric", "value"]
                ].copy()
            )

            three_source_consensus_display.columns = [
                "Three-source outcome",
                "Rows",
            ]

            three_source_consensus_display["Rows"] = (
                three_source_consensus_display["Rows"]
                .round()
                .astype("Int64")
            )

            three_source_consensus_display[
                "Population share (%)"
            ] = three_source_consensus_display["Rows"].apply(
                lambda value: _population_share(value, n_rows)
            )

            three_source_consensus_display["Rows"] = (
                three_source_consensus_display["Rows"]
                .round()
                .astype("Int64")
            )

            outcome_labels = {
                "all_three_agree": "All three agree",
                "quantity_net_agree_alt_outlier": (
                    "Quantity + net weight agree; alt quantity differs"
                ),
                "quantity_alt_agree_net_outlier": (
                    "Quantity + alt quantity agree; net weight differs"
                ),
                "net_alt_agree_quantity_outlier": (
                    "Net weight + alt quantity agree; quantity differs"
                ),
                "no_pair_agrees": "No clear pairwise consensus",
            }

            three_source_consensus_display[
                "Three-source outcome"
            ] = three_source_consensus_display[
                "Three-source outcome"
            ].replace(outcome_labels)

            three_source_consensus_display = (
                three_source_consensus_display.sort_values(
                    "Rows",
                    ascending=False,
                )
            )


        three_source_provenance = _diagnostic_group(
            diagnostics,
            "three_source_consensus_by_evidence_state",
        )

        if three_source_provenance.empty:
            three_source_all_agree_provenance_display = pd.DataFrame()
            three_source_outlier_provenance_display = pd.DataFrame()

        else:
            three_source_provenance = three_source_provenance[
                [
                    "metric",
                    "category_1",
                    "category_2",
                    "category_3",
                    "value",
                ]
            ].copy()

            three_source_provenance.columns = [
                "Outcome",
                "Quantity evidence",
                "Net-weight evidence",
                "Alt-quantity evidence",
                "Rows",
            ]

            three_source_provenance["Rows"] = (
                three_source_provenance["Rows"]
                .round()
                .astype("Int64")
            )

            three_source_provenance[
                "Population share (%)"
            ] = three_source_provenance["Rows"].apply(
                lambda value: _population_share(value, n_rows)
            )

            three_source_provenance["Rows"] = (
                three_source_provenance["Rows"]
                .round()
                .astype("Int64")
            )

            outcome_labels = {
                "all_three_agree": "All three agree",
                "quantity_net_agree_alt_outlier": (
                    "Quantity + net weight agree; alt quantity differs"
                ),
                "quantity_alt_agree_net_outlier": (
                    "Quantity + alt quantity agree; net weight differs"
                ),
                "net_alt_agree_quantity_outlier": (
                    "Net weight + alt quantity agree; quantity differs"
                ),
                "no_pair_agrees": "No clear pairwise consensus",
            }

            three_source_provenance["Outcome"] = (
                three_source_provenance["Outcome"]
                .replace(outcome_labels)
            )

            three_source_all_agree_provenance_display = (
                three_source_provenance.loc[
                    three_source_provenance["Outcome"].eq(
                        "All three agree"
                    )
                ].copy()
            )

            three_source_all_agree_provenance_display = (
                three_source_all_agree_provenance_display[
                    [
                        "Quantity evidence",
                        "Net-weight evidence",
                        "Alt-quantity evidence",
                        "Rows",
                        "Population share (%)",
                    ]
                ].sort_values(
                    "Rows",
                    ascending=False,
                )
            )

            three_source_outlier_provenance_display = (
                three_source_provenance.loc[
                    ~three_source_provenance["Outcome"].eq(
                        "All three agree"
                    )
                ].copy()
            )

            # Keep the dominant provenance pattern for each disagreement outcome.
            three_source_outlier_provenance_display = (
                three_source_outlier_provenance_display
                .sort_values(
                    ["Outcome", "Rows"],
                    ascending=[True, False],
                )
                .groupby(
                    "Outcome",
                    as_index=False,
                    sort=False,
                )
                .first()
            )

            # Make the disagreement itself easier to read.
            disagreement_labels = {
                "Quantity + alt quantity agree; net weight differs":
                    "Net weight differs",
                "Quantity + net weight agree; alt quantity differs":
                    "Alt quantity differs",
                "Net weight + alt quantity agree; quantity differs":
                    "Quantity differs",
                "No clear pairwise consensus":
                    "No pair agrees",
            }

            three_source_outlier_provenance_display[
                "Disagreement pattern"
            ] = three_source_outlier_provenance_display[
                "Outcome"
            ].replace(disagreement_labels)

            # Collapse the three provenance columns into one readable description.
            three_source_outlier_provenance_display[
                "Main provenance pattern"
            ] = (
                "Q "
                + three_source_outlier_provenance_display[
                    "Quantity evidence"
                ].astype(str)
                + " / Net "
                + three_source_outlier_provenance_display[
                    "Net-weight evidence"
                ].astype(str)
                + " / Alt "
                + three_source_outlier_provenance_display[
                    "Alt-quantity evidence"
                ].astype(str)
            )

            three_source_outlier_provenance_display = (
                three_source_outlier_provenance_display[
                    [
                        "Disagreement pattern",
                        "Main provenance pattern",
                        "Rows",
                        "Population share (%)",
                    ]
                ]
                .sort_values(
                    "Rows",
                    ascending=False,
                )
            )
        # --------------------------------------------------------------
        # Supplementary dimensions alongside net weight
        # --------------------------------------------------------------
        quantity_supplementary = _diagnostic_group(
            diagnostics,
            "net_weight_quantity_supplementary_overlap",
        )

        if quantity_supplementary.empty:
            quantity_supplementary_display = pd.DataFrame()
        else:
            quantity_supplementary_display = quantity_supplementary[
                [
                    "metric",
                    "value",
                    "percentage",
                    "category_2",
                ]
            ].copy()

            quantity_supplementary_display.columns = [
                "Quantity unit",
                "Rows",
                "Population share (%)",
                "Share within overlap (%)",
            ]

            quantity_supplementary_display["Rows"] = (
                quantity_supplementary_display["Rows"]
                .round()
                .astype("Int64")
            )

            quantity_supplementary_display = _as_numeric(
                quantity_supplementary_display,
                [
                    "Population share (%)",
                    "Share within overlap (%)",
                ],
            )

            quantity_supplementary_display = (
                quantity_supplementary_display
                .sort_values("Rows", ascending=False)
            )

        alt_supplementary = _diagnostic_group(
            diagnostics,
            "net_weight_alt_quantity_supplementary_overlap",
        )

        if alt_supplementary.empty:
            alt_supplementary_display = pd.DataFrame()
        else:
            alt_supplementary_display = alt_supplementary[
                [
                    "metric",
                    "value",
                    "percentage",
                    "category_2",
                ]
            ].copy()

            alt_supplementary_display.columns = [
                "Alt-quantity unit",
                "Rows",
                "Population share (%)",
                "Share within overlap (%)",
            ]

            alt_supplementary_display["Rows"] = (
                alt_supplementary_display["Rows"]
                .round()
                .astype("Int64")
            )

            alt_supplementary_display = _as_numeric(
                alt_supplementary_display,
                [
                    "Population share (%)",
                    "Share within overlap (%)",
                ],
            )

            alt_supplementary_display = (
                alt_supplementary_display
                .sort_values("Rows", ascending=False)
            )

        supplementary_examples = _diagnostic_group(
            diagnostics,
            "net_weight_quantity_supplementary_examples",
        )

        if supplementary_examples.empty:
            supplementary_examples_display = pd.DataFrame()
        else:
            supplementary_examples_display = supplementary_examples[
                [
                    "category_1",
                    "year",
                    "category_2",
                    "value",
                    "metric",
                ]
            ].copy()

            supplementary_examples_display.columns = [
                "HS code",
                "Year",
                "Net weight (kg)",
                "Quantity",
                "Quantity unit",
            ]

            supplementary_examples_display = _as_numeric(
                supplementary_examples_display,
                [
                    "Net weight (kg)",
                    "Quantity",
                ],
            )


            def m3_relationship_table(
                diagnostic_group: str,
            ) -> pd.DataFrame:
                frame = _diagnostic_group(
                    diagnostics,
                    diagnostic_group,
                )

                if frame.empty:
                    return pd.DataFrame()

                out = frame[
                    [
                        "metric",
                        "value",
                        "percentage",
                        "category_1",
                        "category_2",
                        "category_3",
                    ]
                ].copy()

                out.columns = [
                    "HS6",
                    "Comparable rows",
                    "Share of comparable m³ rows (%)",
                    "Median kg/m³",
                    "P10 kg/m³",
                    "P90 kg/m³",
                ]

                out["Comparable rows"] = (
                    pd.to_numeric(
                        out["Comparable rows"],
                        errors="coerce",
                    )
                    .round()
                    .astype("Int64")
                )

                out = _as_numeric(
                    out,
                    [
                        "Share of comparable m³ rows (%)",
                        "Median kg/m³",
                        "P10 kg/m³",
                        "P90 kg/m³",
                    ],
                )

                return (
                    out
                    .sort_values(
                        "Comparable rows",
                        ascending=False,
                    )
                    .head(30)
                )


        quantity_m3_net_relationship = m3_relationship_table(
            "net_weight_quantity_m3_by_hs",
        )

        alt_quantity_m3_net_relationship = m3_relationship_table(
            "net_weight_alt_quantity_m3_by_hs",
        )

        supplementary_hs = _diagnostic_group(
            diagnostics,
            "net_weight_supplementary_by_hs",
        )
        supplementary_hs_tables: dict[str, pd.DataFrame] = {}
        if not supplementary_hs.empty:
            for unit_code, group in supplementary_hs.groupby("category_1"):
                table = group[
                    ["metric", "category_2", "value", "percentage"]
                ].copy()
                table.columns = ["HS code", "Unit", "Rows", "Share within unit (%)"]
                table["Rows"] = table["Rows"].round().astype("Int64")
                supplementary_hs_tables[str(unit_code)] = table.sort_values(
                    "Rows", ascending=False
                ).head(15)

        # --------------------------------------------------------------
        # Supplementary candidate overlap
        #
        # These diagnostics inspect whether one final
        # supplementary_quantity / supplementary_unit pair is sufficient
        # to preserve the non-kg physical information carried by
        # quantity and alt_quantity.
        # --------------------------------------------------------------

        supplementary_candidate_overlap = _diagnostic_group(
            diagnostics,
            "supplementary_candidate_overlap",
        )

        if supplementary_candidate_overlap.empty:
            supplementary_candidate_overlap_display = pd.DataFrame()
        else:
            supplementary_candidate_overlap_display = (
                supplementary_candidate_overlap[
                    ["metric", "value", "percentage"]
                ].copy()
            )

            supplementary_candidate_overlap_display.columns = [
                "Candidate pattern",
                "Rows",
                "Population share (%)",
            ]

            supplementary_pattern_labels = {
                "quantity_only":
                    "Non-kg quantity only",
                "alt_quantity_only":
                    "Non-kg alt_quantity only",
                "both_same_unit":
                    "Both non-kg candidates, same unit",
                "both_different_unit":
                    "Both non-kg candidates, different units",
                "neither":
                    "No usable non-kg supplementary candidate",
            }

            supplementary_candidate_overlap_display[
                "Candidate pattern"
            ] = supplementary_candidate_overlap_display[
                "Candidate pattern"
            ].replace(supplementary_pattern_labels)

            supplementary_candidate_overlap_display["Rows"] = (
                supplementary_candidate_overlap_display["Rows"]
                .round()
                .astype("Int64")
            )

            supplementary_candidate_overlap_display = _as_numeric(
                supplementary_candidate_overlap_display,
                ["Population share (%)"],
            )

            supplementary_candidate_overlap_display = (
                supplementary_candidate_overlap_display
                .sort_values(
                    "Rows",
                    ascending=False,
                )
            )

        supplementary_unit_pairs = _diagnostic_group(
            diagnostics,
            "supplementary_unit_pair",
        )

        if supplementary_unit_pairs.empty:
            supplementary_unit_pairs_display = pd.DataFrame()
        else:
            supplementary_unit_pairs_display = (
                supplementary_unit_pairs[
                    [
                        "metric",
                        "category_1",
                        "category_2",
                        "category_3",
                        "value",
                        "percentage",
                    ]
                ].copy()
            )

            supplementary_unit_pairs_display.columns = [
                "Quantity unit + alt-quantity unit",
                "Quantity unit code",
                "Alt-quantity unit code",
                "Relationship",
                "Rows",
                "Population share (%)",
            ]

            supplementary_unit_pairs_display[
                "Relationship"
            ] = supplementary_unit_pairs_display[
                "Relationship"
            ].replace(
                {
                    "same_unit": "Same unit",
                    "different_unit": "Different units",
                }
            )

            supplementary_unit_pairs_display["Rows"] = (
                supplementary_unit_pairs_display["Rows"]
                .round()
                .astype("Int64")
            )

            supplementary_unit_pairs_display = _as_numeric(
                supplementary_unit_pairs_display,
                ["Population share (%)"],
            )

            supplementary_unit_pairs_display = (
                supplementary_unit_pairs_display
                .sort_values(
                    "Rows",
                    ascending=False,
                )
                .head(30)
            )

        # --------------------------------------------------------------
        # Final supplementary output
        # --------------------------------------------------------------

        final_physical_coverage = _diagnostic_group(
            diagnostics,
            "final_physical_information_coverage",
        )

        canonical_kg_rows = _metric_value(
            final_physical_coverage,
            "canonical_kg",
        )
        supplementary_without_kg_rows = _metric_value(
            final_physical_coverage,
            "supplementary_without_kg",
        )
        no_physical_quantity_rows = _metric_value(
            final_physical_coverage,
            "no_physical_quantity",
        )

        canonical_kg_share = (
            100.0 * canonical_kg_rows / n_rows
            if n_rows and canonical_kg_rows is not None
            else None
        )

        supplementary_without_kg_share = (
            100.0 * supplementary_without_kg_rows / n_rows
            if n_rows and supplementary_without_kg_rows is not None
            else None
        )

        no_physical_quantity_share = (
            100.0 * no_physical_quantity_rows / n_rows
            if n_rows and no_physical_quantity_rows is not None
            else None
        )

        non_canonical_rows = (
            float(supplementary_without_kg_rows or 0)
            + float(no_physical_quantity_rows or 0)
        )

        supplementary_rescue_rate = (
            100.0 * float(supplementary_without_kg_rows or 0) / non_canonical_rows
            if non_canonical_rows > 0
            else None
        )

        if final_physical_coverage.empty:
            final_physical_coverage_display = pd.DataFrame()

        else:
            final_physical_coverage_display = (
                final_physical_coverage[
                    [
                        "metric",
                        "value",
                        "percentage",
                    ]
                ].copy()
            )

            final_physical_coverage_display.columns = [
                "Final physical-information state",
                "Rows",
                "Population share (%)",
            ]

            final_physical_coverage_display[
                "Final physical-information state"
            ] = final_physical_coverage_display[
                "Final physical-information state"
            ].replace(
                {
                    "canonical_kg":
                        "Canonical kg available",

                    "supplementary_without_kg":
                        "No kg; supplementary quantity retained",

                    "no_physical_quantity":
                        "Neither kg nor supplementary quantity",
                }
            )

            final_physical_coverage_display["Rows"] = (
                final_physical_coverage_display["Rows"]
                .round()
                .astype("Int64")
            )

            final_physical_coverage_display = _as_numeric(
                final_physical_coverage_display,
                ["Population share (%)"],
            )

            coverage_order = {
                "Canonical kg available": 0,
                "No kg; supplementary quantity retained": 1,
                "Neither kg nor supplementary quantity": 2,
            }

            final_physical_coverage_display = (
                final_physical_coverage_display
                .assign(
                    _order=final_physical_coverage_display[
                        "Final physical-information state"
                    ].map(coverage_order)
                )
                .sort_values("_order")
                .drop(columns="_order")
            )

        supplementary_output_summary = _diagnostic_group(
            diagnostics,
            "supplementary_output_summary",
        )

        if supplementary_output_summary.empty:
            supplementary_output_summary_display = pd.DataFrame()
        else:
            supplementary_output_summary_display = (
                supplementary_output_summary[
                    ["metric", "value", "percentage"]
                ].copy()
            )
            supplementary_output_summary_display.columns = [
                "Final supplementary outcome",
                "Rows",
                "Population share (%)",
            ]
            supplementary_output_summary_display[
                "Final supplementary outcome"
            ] = supplementary_output_summary_display[
                "Final supplementary outcome"
            ].replace(
                {
                    "supplementary_positive_rows":
                        "Rows with supplementary quantity",
                    "supplementary_missing_rows":
                        "Rows without supplementary quantity",
                    "selected_from_quantity":
                        "Selected from quantity",
                    "selected_from_alt_quantity":
                        "Selected from alt_quantity",
                    "reported_supplementary":
                        "Reported supplementary quantity",
                    "estimated_supplementary":
                        "Estimated supplementary quantity",
                    "quantity_unit_null_mismatch":
                        "Quantity/unit null mismatch",
                }
            )
            supplementary_output_summary_display["Rows"] = (
                supplementary_output_summary_display["Rows"]
                .round()
                .astype("Int64")
            )
            supplementary_output_summary_display = _as_numeric(
                supplementary_output_summary_display,
                ["Population share (%)"],
            )

        supplementary_output_units = _diagnostic_group(
            diagnostics,
            "supplementary_output_unit_distribution",
        )

        if supplementary_output_units.empty:
            supplementary_output_units_display = pd.DataFrame()
        else:
            supplementary_output_units_display = (
                supplementary_output_units[
                    [
                        "metric",
                        "category_1",
                        "category_2",
                        "value",
                        "percentage",
                    ]
                ].copy()
            )
            supplementary_output_units_display.columns = [
                "Supplementary unit",
                "Selected source",
                "Evidence state",
                "Rows",
                "Population share (%)",
            ]
            supplementary_output_units_display["Rows"] = (
                supplementary_output_units_display["Rows"]
                .round()
                .astype("Int64")
            )
            supplementary_output_units_display = _as_numeric(
                supplementary_output_units_display,
                ["Population share (%)"],
            )
            supplementary_output_units_display = (
                supplementary_output_units_display
                .sort_values("Rows", ascending=False)
            )

        # --------------------------------------------------------------
        # Net-weight diagnostics
        # --------------------------------------------------------------
        net_weight_year = _diagnostic_group(diagnostics, "net_weight_by_year")
        if net_weight_year.empty:
            net_weight_year_display = pd.DataFrame()
        else:
            net_weight_year_display = net_weight_year[
                ["year", "metric", "value", "percentage"]
            ].copy()
            net_weight_year_display.columns = [
                "Year",
                "Evidence state",
                "Positive rows",
                "Positive coverage within evidence-state rows (%)",
            ]
            net_weight_year_display["Positive rows"] = (
                net_weight_year_display["Positive rows"].round().astype("Int64")
            )
            net_weight_year_display = net_weight_year_display.sort_values(
                ["Year", "Evidence state"]
            )

        net_weight_distribution = _diagnostic_group(
            diagnostics,
            "net_weight_distribution_by_year",
        )
        if net_weight_distribution.empty:
            net_weight_distribution_display = pd.DataFrame()
        else:
            net_weight_distribution_display = net_weight_distribution[
                ["year", "value", "category_1", "category_2", "category_3"]
            ].copy()
            net_weight_distribution_display.columns = [
                "Year",
                "Positive rows",
                "p10 log10(kg)",
                "Median log10(kg)",
                "p90 log10(kg)",
            ]
            net_weight_distribution_display["Positive rows"] = (
                net_weight_distribution_display["Positive rows"].round().astype("Int64")
            )
            net_weight_distribution_display = _as_numeric(
                net_weight_distribution_display,
                ["p10 log10(kg)", "Median log10(kg)", "p90 log10(kg)"],
            ).sort_values("Year")

        # --------------------------------------------------------------
        # Gross-weight diagnostics
        # --------------------------------------------------------------
        gross_weight = _diagnostic_group(diagnostics, "gross_weight_intervention")
        gross_lookup = {
            row.metric: {"value": row.value, "percentage": row.percentage}
            for row in gross_weight.itertuples()
        }
        gross_available = gross_lookup.get("gross_weight_available", {}).get("value")
        gross_after_current = _metric_value(
            gross_weight,
            "gross_weight_available_after_quantity_kg_and_net_weight_missing",
            "gross_weight_available_after_current_sources_missing",
        )
        gross_after_alt = _metric_value(
            gross_weight,
            "gross_weight_available_after_quantity_kg_net_weight_and_alt_kg_missing",
            "gross_weight_available_after_alt_kg_missing",
        )
        gross_used = gross_lookup.get("gross_weight_proxy_used", {}).get("value")

        gross_relationship = _diagnostic_group(
            diagnostics,
            "gross_net_weight_relationship",
        )
        gross_relationship_lookup = {
            row.metric: row.value for row in gross_relationship.itertuples()
        }
        gross_comparable = gross_relationship_lookup.get("comparable_rows")
        gross_below_net = gross_relationship_lookup.get("gross_below_net_rows")
        gross_below_net_share = (
            gross_below_net / gross_comparable
            if gross_comparable and gross_below_net is not None
            else None
        )

        gross_evidence = _diagnostic_group(
            diagnostics,
            "gross_net_weight_by_evidence_state",
        )

        if gross_evidence.empty:
            gross_evidence_display = pd.DataFrame()
        else:
            gross_evidence_display = gross_evidence[
                [
                    "metric",
                    "value",
                    "percentage",
                    "category_1",
                    "category_2",
                    "category_3",
                ]
            ].copy()

            gross_evidence_display.columns = [
                "Evidence state",
                "Comparable rows",
                "Gross < net (%)",
                "Median gross/net ratio",
                "p10 ratio",
                "p90 ratio",
            ]

            gross_evidence_display = _split_evidence_state(
                gross_evidence_display,
                column="Evidence state",
                left_name="Gross-weight evidence",
                right_name="Net-weight evidence",
            )

            gross_evidence_display["Comparable rows"] = (
                gross_evidence_display["Comparable rows"]
                .round()
                .astype("Int64")
            )

            gross_evidence_display = _as_numeric(
                gross_evidence_display,
                [
                    "Median gross/net ratio",
                    "p10 ratio",
                    "p90 ratio",
                ],
            )

        gross_without_net_year = _diagnostic_group(
            diagnostics,
            "gross_weight_without_net_by_year",
        )
        if gross_without_net_year.empty:
            gross_without_net_year_display = pd.DataFrame()
        else:
            gross_without_net_year_display = gross_without_net_year[
                ["year", "value", "percentage"]
            ].copy()
            gross_without_net_year_display.columns = [
                "Year",
                "Gross-positive / net-unavailable rows",
                "Share of such rows (%)",
            ]
            gross_without_net_year_display[
                "Gross-positive / net-unavailable rows"
            ] = gross_without_net_year_display[
                "Gross-positive / net-unavailable rows"
            ].round().astype("Int64")
            gross_without_net_year_display = gross_without_net_year_display.sort_values("Year")

        gross_without_net_hs_display = _top_hs_table(
            _diagnostic_group(diagnostics, "gross_weight_without_net_by_hs"),
            n=20,
        )

        # --------------------------------------------------------------
        # Adopted synchronization methodology diagnostics
        # --------------------------------------------------------------

        methodology_funnel = _diagnostic_group(
            diagnostics,
            "methodology_funnel",
        )

        if methodology_funnel.empty:
            methodology_funnel_display = pd.DataFrame()
        else:
            methodology_funnel = methodology_funnel.copy()
            methodology_funnel["Step"] = pd.to_numeric(
                methodology_funnel["category_1"],
                errors="coerce",
            )
            methodology_funnel["Cumulative rows"] = pd.to_numeric(
                methodology_funnel["category_2"],
                errors="coerce",
            )
            methodology_funnel["Incremental population share (%)"] = pd.to_numeric(
                methodology_funnel["category_3"],
                errors="coerce",
            )

            methodology_funnel_display = methodology_funnel[
                [
                    "Step",
                    "metric",
                    "value",
                    "Incremental population share (%)",
                    "Cumulative rows",
                    "percentage",
                ]
            ].copy()

            methodology_funnel_display.columns = [
                "Step",
                "Methodology step",
                "Rows resolved at step",
                "Population share (%)",
                "Cumulative rows resolved",
                "Cumulative coverage (%)",
            ]

            methodology_labels = {
                "reported_three_exact":
                    "Reported three-source exact agreement",
                "reported_pair_exact":
                    "Reported exact pair agreement",
                "reported_precedence":
                    "Reported source precedence (Q > A > N)",
                "estimated_fallback":
                    "Estimated fallback (Q > A > N)",
            }

            methodology_funnel_display[
                "Methodology step"
            ] = methodology_funnel_display[
                "Methodology step"
            ].replace(methodology_labels)

            methodology_funnel_display[
                "Rows resolved at step"
            ] = methodology_funnel_display[
                "Rows resolved at step"
            ].round().astype("Int64")

            methodology_funnel_display[
                "Cumulative rows resolved"
            ] = methodology_funnel_display[
                "Cumulative rows resolved"
            ].round().astype("Int64")

            methodology_funnel_display = (
                methodology_funnel_display.sort_values("Step")
            )

        methodology_rules = _diagnostic_group(
            diagnostics,
            "methodology_rule_detail",
        )

        if methodology_rules.empty:
            methodology_rules_display = pd.DataFrame()
        else:
            methodology_rules_display = methodology_rules[
                [
                    "metric",
                    "category_1",
                    "category_2",
                    "category_3",
                    "value",
                    "percentage",
                ]
            ].copy()

            methodology_rules_display.columns = [
                "Applied rule",
                "Final source",
                "Evidence tier",
                "Resolution type",
                "Rows",
                "Population share (%)",
            ]

            methodology_rules_display["Rows"] = (
                methodology_rules_display["Rows"]
                .round()
                .astype("Int64")
            )

            methodology_rules_display = (
                methodology_rules_display.sort_values(
                    "Rows",
                    ascending=False,
                )
            )

        methodology_summary = _diagnostic_group(
            diagnostics,
            "methodology_summary",
        )

        if methodology_summary.empty:
            methodology_summary_display = pd.DataFrame()
        else:
            methodology_summary_display = methodology_summary[
                ["metric", "value", "percentage"]
            ].copy()

            methodology_summary_display.columns = [
                "Evidence-quality summary",
                "Rows",
                "Population share (%)",
            ]

            methodology_summary_labels = {
                "resolved_by_reported_evidence":
                    "Resolved using reported evidence",
                "resolved_by_exact_reported_agreement":
                    "Resolved by exact reported agreement",
                "resolved_by_reported_precedence":
                    "Resolved by reported source precedence",
                "resolved_by_estimated_fallback":
                    "Resolved using estimated fallback",
                "no_usable_kg_evidence":
                    "No usable Q/A/N kg evidence",
            }

            methodology_summary_display[
                "Evidence-quality summary"
            ] = methodology_summary_display[
                "Evidence-quality summary"
            ].replace(methodology_summary_labels)

            methodology_summary_display["Rows"] = (
                methodology_summary_display["Rows"]
                .round()
                .astype("Int64")
            )

        sync_status = _diagnostic_group(
            diagnostics,
            "synchronization_status",
        )

        if sync_status.empty:
            sync_status_display = pd.DataFrame()
        else:
            sync_status_display = sync_status[
                ["metric", "category_1", "value", "percentage"]
            ].copy()

            sync_status_display.columns = [
                "Status",
                "Final source",
                "Rows",
                "Population share (%)",
            ]

            sync_status_display["Rows"] = (
                sync_status_display["Rows"]
                .round()
                .astype("Int64")
            )

            sync_status_display = sync_status_display.sort_values(
                "Rows",
                ascending=False,
            )

        remaining_gap_display = _simple_rows_table(
            _diagnostic_group(
                diagnostics,
                "remaining_kg_gap",
            ),
            metric_name="Outcome",
            percentage_name="Population share (%)",
        )

        missing_hs_display = _top_hs_table(
            _diagnostic_group(
                diagnostics,
                "missing_kg_by_hs",
            ),
            n=20,
        )

        # --------------------------------------------------------------
        # Figures
        # --------------------------------------------------------------
        figure_paths = {
            "quantity_units": figure_dir / "quantity_unit_distribution.png",
            "alt_quantity_units": figure_dir / "alt_quantity_unit_distribution.png",
            "quantity_units_year": figure_dir / "quantity_unit_share_by_year.png",
            "alt_quantity_units_year": figure_dir / "alt_quantity_unit_share_by_year.png",
            "agreement_year": figure_dir / "quantity_net_weight_agreement_by_year.png",
            "gap_distribution": figure_dir / "quantity_net_weight_relative_gap_distribution.png",
            "alt_agreement_state": figure_dir / "alt_quantity_kg_agreement_by_evidence_state.png",
            "net_weight_year": figure_dir / "net_weight_coverage_by_year.png",
            "net_weight_distribution": figure_dir / "net_weight_log_distribution_by_year.png",
            "synchronized_quantity_distribution": figure_dir  / "synchronized_quantity_kg_distribution.png",
            "net_weight_input_distribution":  figure_dir / "net_weight_input_distribution.png",
            "gross_net_ratio": figure_dir / "gross_net_weight_ratio_distribution.png",
            "hierarchy": figure_dir / "kg_coverage_by_reconciliation_step.png",
            "sync_status_year": figure_dir / "quantity_sync_status_by_year.png",
            "quantity_m3_hs": figure_dir / "quantity_m3_by_hs_code.png",
            "alt_quantity_m3_hs": figure_dir / "alt_quantity_m3_by_hs_code.png",
            "quantity_m3_hs4": figure_dir / "quantity_m3_by_hs4_code.png",
            "alt_quantity_m3_hs4": figure_dir / "alt_quantity_m3_by_hs4_code.png",
        }

        _plot_unit_distribution(
            diagnostics=diagnostics,
            diagnostic_group="quantity_unit_distribution",
            title="Quantity unit distribution",
            output_path=figure_paths["quantity_units"],
        )
        _plot_unit_distribution(
            diagnostics=diagnostics,
            diagnostic_group="alt_quantity_unit_distribution",
            title="Canonical alternate-quantity unit distribution",
            output_path=figure_paths["alt_quantity_units"],
        )
        _plot_unit_share_by_year(
            diagnostics=diagnostics,
            diagnostic_group="quantity_unit_by_year",
            title="Quantity unit composition by year",
            output_path=figure_paths["quantity_units_year"],
        )
        _plot_unit_share_by_year(
            diagnostics=diagnostics,
            diagnostic_group="alt_quantity_unit_by_year",
            title="Alternate quantity unit composition by year",
            output_path=figure_paths["alt_quantity_units_year"],
        )
        _plot_agreement_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths["agreement_year"],
        )
        _plot_relative_gap_distribution(
            con=con,
            synchronized_sql=synchronized_sql,
            output_path=figure_paths["gap_distribution"],
        )
        _plot_alt_agreement_by_evidence_state(
            diagnostics=diagnostics,
            output_path=figure_paths["alt_agreement_state"],
        )
        _plot_net_weight_distribution(
            con=con,
            prepared_sql=prepared_sql,
            output_path=figure_paths[
                "net_weight_input_distribution"
            ],
        )

        _plot_synchronized_quantity_distribution(
            con=con,
            synchronized_sql=synchronized_sql,
            output_path=figure_paths[
                "synchronized_quantity_distribution"
            ],
        )
        _plot_net_weight_coverage_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths["net_weight_year"],
        )
        _plot_net_weight_log_distribution_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths["net_weight_distribution"],
        )
        _plot_gross_net_ratio_distribution(
            con=con,
            prepared_sql=prepared_sql,
            output_path=figure_paths["gross_net_ratio"],
        )
        _plot_hierarchy_coverage(
            diagnostics=diagnostics,
            output_path=figure_paths["hierarchy"],
        )
        _plot_synchronization_status_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths["sync_status_year"],
        )
        _plot_m3_by_hs(
            diagnostics=diagnostics,
            diagnostic_group="quantity_m3_by_hs",
            title="HS codes with quantity reported in cubic metres",
            hs_level="HS6",
            output_path=figure_paths["quantity_m3_hs"],
        )
        _plot_m3_by_hs(
            diagnostics=diagnostics,
            diagnostic_group="alt_quantity_m3_by_hs",
            title="HS codes with alternate quantity reported in cubic metres",
            hs_level="HS6",
            output_path=figure_paths["alt_quantity_m3_hs"],
        )
        _plot_m3_by_hs(
            diagnostics=diagnostics,
            diagnostic_group="quantity_m3_by_hs4",
            title="HS4 distribution of quantity reported in cubic metres",
            output_path=figure_paths["quantity_m3_hs4"],
            maximum_codes=100,
            hs_level="HS4",
        )

        _plot_m3_by_hs(
            diagnostics=diagnostics,
            diagnostic_group="alt_quantity_m3_by_hs4",
            title="HS4 distribution of alternate quantity reported in cubic metres",
            output_path=figure_paths["alt_quantity_m3_hs4"],
            maximum_codes=100,
            hs_level="HS4",
        )

        # --------------------------------------------------------------
        # Decision logic: still evaluates CURRENT synchronize.py output.
        # Candidate alt-quantity diagnostics do not silently change it.
        # --------------------------------------------------------------
        population_partition_passed = bool(
            n_rows == n_synchronized + n_no_kg
        )

        methodology_partition_passed = bool(
            n_rows
            == (
                n_reported_three_exact
                + n_reported_pair_exact
                + n_reported_precedence
                + n_estimated_fallback
                + n_no_kg
            )
        )

        synchronized_output_valid = bool(
            synchronized_quantity_path.is_file()
            and synchronized_quantity_path.stat().st_size > 0
        )

        ready_for_cif_fob = bool(
            synchronized_output_valid
            and population_partition_passed
            and methodology_partition_passed
        )

        decision = (
            "APPROVED"
            if ready_for_cif_fob
            else "REVIEW REQUIRED"
        )

        gross_weight_statement = (
            "Gross weight is diagnostic-only and is not used in the "
            "synchronized kg quantity."
        )

        if n_potential_gross_weight_fallback > 0:
            gross_weight_statement += (
                f" Gross weight nevertheless exists for "
                f"`{n_potential_gross_weight_fallback:,}` rows where "
                "quantity, alternate-quantity, and net-weight kg candidates "
                "are all unavailable."
            )

        # --------------------------------------------------------------
        # Markdown
        # --------------------------------------------------------------
        lines: list[str] = [
            "# Quantity Decision Report",
            "",
            f"**Decision:** {decision}  ",
            f"**Run ID:** {run_id}  ",
            f"**Target HS:** {cfg.scope.target_hs}  ",
            f"**Input period:** {cfg.scope.min_year}–{cfg.scope.max_year}  ",
            f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
            "",
            "__TOC__",
            "",
            "## 1. Purpose of Quantity synchronization",
            "",
            (
                "The Quantity module receives the HS-synchronized Comtrade "
                "observations and evaluates the available physical-quantity "
                "information before downstream value processing."
            ),
            "",
            (
                "The objective is to construct a transparent kg-based physical "
                "quantity while preserving supplementary physical dimensions "
                "and source provenance. Gross weight is evaluated separately "
                "because it is not equivalent to net physical weight."
            ),
            "",
            "## 2. Input data and observation population",
            "",
            f"Quantity received `{n_rows:,}` observations from the HS Sync output.",
            "",
            (
                "**Reporting convention:** unless explicitly labelled as a "
                "conditional statistic, every percentage in this report is "
                f"calculated relative to the full input population of "
                f"`{n_rows:,}` observations, which represents 100%."
            ),
            "",
            (
                "Conditional statistics, such as agreement rates among comparable "
                "observations, are labelled separately and state their comparison "
                "population explicitly."
            ),
            "",
            "| Property | Value |",
            "|---|---|",
            "| Upstream module | HS Sync |",
            f"| Target HS classification | {cfg.scope.target_hs} |",
            f"| Covered years | {cfg.scope.min_year}–{cfg.scope.max_year} |",
            "| Canonical synchronized field | `synchronized_quantity_kg` |",
            f"| Agreement tolerance | {100 * agreement_tolerance:.2f}% |",
            (
                "| Configured source-selection method | "
                f"`{cfg.method.kg_reconciliation.source_selection.method}` |"
            ),
            "",
            "### 2.1 Quantity-field inventory",
            "",
            _markdown_table(inventory_display),
            "### 2.2 Net-weight distribution",
            "",
            (
                "The distribution below shows the scale of positive `net_weight_kg` "
                "observations entering the Quantity module. Because trade weights span "
                "several orders of magnitude, the horizontal scale is expressed as "
                "`log10(net_weight_kg)`."
            ),
            "",
            markdown_image(
                "Distribution of positive net weight",
                figure_paths["net_weight_input_distribution"],
                document_path=report_path,
                width=850,
            ),
            "",
            "",
            "## 3. Physical-unit structure",
            "",
            "### 3.1 `quantity` unit distribution",
            "",
            (
                "The `quantity` field is expected to be reported in kilograms "
                "or in a unit that is exactly convertible to kilograms. "
                "Other units are retained for diagnostic purposes."
            ),
            "",
        ]
        if figure_paths["quantity_units"].exists():
            lines.extend([
                get_table_figure_layout(
                    quantity_units,
                    markdown_image(
                        "Quantity unit distribution",
                        figure_paths["quantity_units"],
                        document_path=report_path,
                        width=650,
                    ),
                ),
                "",
            ])
        else:
            lines.extend([
                _markdown_table(quantity_units),
                "",
            ])

        lines.extend([
            "### 3.2 Canonical `alt_quantity` unit distribution",
            "",
            (
                "Alternate quantities expressed as `kg`, `1000 KG`, or `g` "
                "are normalized to canonical kilograms using exact scale "
                "conversions. Special mass bases such as `kg 90% sdt` "
                "remain separate."
            ),
            "",
        ])

        if figure_paths["alt_quantity_units"].exists():
            lines.extend([
                get_table_figure_layout(
                    alt_quantity_units,
                    markdown_image(
                        "Canonical alternate quantity unit distribution",
                        figure_paths["alt_quantity_units"],
                        document_path=report_path,
                        width=650,
                    ),
                ),
                "",
            ])
        else:
            lines.extend([
                _markdown_table(alt_quantity_units),
                "",
            ])

        lines.extend([
            "### 3.3 Unit composition over time",
            "",
            "Unit composition by year is retained to reveal changes in reporting practice.",
            "",
        ])

        if (
            figure_paths["quantity_units_year"].exists()
            and figure_paths["alt_quantity_units_year"].exists()
        ):
            lines.extend([
                plot_figure_pair(
                    markdown_image(
                        "Quantity units by year",
                        figure_paths["quantity_units_year"],
                        document_path=report_path,
                        width=620,
                    ),
                    markdown_image(
                        "Alternate quantity units by year",
                        figure_paths["alt_quantity_units_year"],
                        document_path=report_path,
                        width=620,
                    ),
                ),
                "",
            ])

            lines.extend([
                "### 3.4 Cubic-metre observations by HS code",
                "",
                (
                    "Cubic-metre reporting is first summarized at the broader "
                    "four-digit HS heading level and then examined at the detailed "
                    "six-digit product level."
                ),
                "",
                "#### 3.4.1 Broad HS4 concentration",
                "",
                "**`quantity` reported in m³**",
                "",
            ])

            if figure_paths["quantity_m3_hs4"].exists():
                lines.extend([
                    get_table_figure_layout(
                        quantity_m3_hs4,
                        markdown_image(
                            "Quantity m³ observations by HS4 code",
                            figure_paths["quantity_m3_hs4"],
                            document_path=report_path,
                            width=650,
                        ),
                    ),
                    "",
                ])
            else:
                lines.extend([
                    _markdown_table(quantity_m3_hs4),
                    "",
                ])

            lines.extend([
                "**`alt_quantity` reported in m³**",
                "",
            ])

            if figure_paths["alt_quantity_m3_hs4"].exists():
                lines.extend([
                    get_table_figure_layout(
                        alt_quantity_m3_hs4,
                        markdown_image(
                            "Alternate quantity m³ observations by HS4 code",
                            figure_paths["alt_quantity_m3_hs4"],
                            document_path=report_path,
                            width=650,
                            height=700
                        ),
                    ),
                    "",
                ])
            else:
                lines.extend([
                    _markdown_table(alt_quantity_m3_hs4),
                    "",
                ])

            lines.extend([
                "#### 3.4.2 Detailed HS6 concentration",
                "",
                "**`quantity` reported in m³**",
                "",
                (
                f"The {len(quantity_m3)} most frequent HS codes shown below "
                f"account for {_format_percent(quantity_m3_top_share, already_percent=True)} "
                "of all positive `quantity` observations reported in m³. "
                "Together, these displayed HS codes represent approximately "
                f"{_format_percent(quantity_m3_top_population_share, already_percent=True)} "
                "of the full Quantity population, out of the total "
                f"{_format_percent(quantity_m3_population_share, already_percent=True)} "
                "reported in m³."
                ),
                "",
            ])

        if figure_paths["quantity_m3_hs"].exists():
            lines.extend([
                get_table_figure_layout(
                    quantity_m3,
                    markdown_image(
                        "Quantity m³ observations by HS code",
                        figure_paths["quantity_m3_hs"],
                        document_path=report_path,
                        width=650,
                        height=700
                    ),
                ),
                "",
            ])
        else:
            lines.extend([
                _markdown_table(quantity_m3),
                "",
            ])

        lines.extend([
            "**`alt_quantity` reported in m³**",
            "",
            (
                f"The {len(alt_quantity_m3)} most frequent HS codes shown below "
                f"account for {_format_percent(alt_quantity_m3_top_share, already_percent=True)} "
                "of all positive `alt_quantity` observations reported in m³. "
                "Together, these displayed HS codes represent approximately "
                f"{_format_percent(alt_quantity_m3_top_population_share, already_percent=True)} "
                "of the full Quantity population, out of the total "
                f"{_format_percent(alt_quantity_m3_population_share, already_percent=True)} "
                "reported in m³."
            ),
            "",
        ])

        if figure_paths["alt_quantity_m3_hs"].exists():
            lines.extend([
                get_table_figure_layout(
                    alt_quantity_m3,
                    markdown_image(
                        "Alternate quantity m³ observations by HS code",
                        figure_paths["alt_quantity_m3_hs"],
                        document_path=report_path,
                        width=650,
                        height=700
                    ),
                ),
                "",
            ])
        else:
            lines.extend([
                _markdown_table(alt_quantity_m3),
                "",
            ])

        # Canonical kg evidence + alt candidate diagnostics
        lines.extend([
            "## 4. Canonical kg evidence",
            "",
            "Three kg-equivalent evidence families are evaluated:",
            "",
            "1. `quantity` reported directly in `kg`.",
            "2. `net_weight_kg`.",
            (
                "3. `alt_quantity` that is either directly expressed in `kg` "
                "or exactly convertible to kg (for example `1000 KG`)."
            ),
            "",
            (
                "Special mass bases such as `kg 90% sdt` are not treated "
                "as ordinary net kg."
            ),
            "",
            "### 4.1 Quantity vs. net-weight evidence",
            "",
            (
                "All percentages in this table use the full Quantity input "
                f"population of `{n_rows:,}` observations as the denominator."
            ),
            "",
            "",
            _markdown_table(evidence_state_display),
            "",
            "### 4.2 Alternate quantity as kg-equivalent candidate evidence",
            "",
            (
                "Alternate quantity is evaluated as canonical kg evidence when it is "
                "directly expressed in kg or exactly convertible to kg. It participates "
                "in the adopted synchronization hierarchy with source precedence "
                "`quantity > alt_quantity > net_weight`. Population shares use the full "
                "Quantity input population."
            ),
            "",
            _markdown_table(alt_candidate_display),
            "",
            "**Incremental coverage if quantity kg and net weight are unavailable**",
            "",
            _markdown_table(alt_incremental_display),
            "",
            "### 4.3 Alternate-quantity agreement with existing kg evidence",
            "",
            (
                "All percentages in this table use the full Quantity input "
                f"population of `{n_rows:,}` observations as the denominator."
            ),
            "",
            _markdown_table(alt_agreement_display),
            "",
        ])
        lines.extend([
            "### 4.4 Three-source consensus diagnostics",
            "",
            (
                "For rows where `quantity kg`, `net weight`, and `canonical "
                "alternate-quantity kg` are all available, the three pairwise "
                "comparisons are evaluated using the configured agreement "
                f"tolerance of {100 * agreement_tolerance:.2f}%. "
                "Row counts are shown together with their share of the full "
                "Quantity input population."
            ),
            "",
            _markdown_table(three_source_consensus_display),
            "",
        ])
        lines.extend([
            "### 4.5 Evidence provenance within three-source agreement",
            "",
            (
                "Evidence provenance is shown for rows where all three "
                "kg-equivalent sources agree. Population shares use the full "
                "Quantity input population as the denominator."
            ),
            "",
            _markdown_table(
                three_source_all_agree_provenance_display
            ),
            "",
        ])
        lines.extend([
            "### 4.6 Provenance of three-source disagreements",
            "",
            (
                "This table shows the dominant evidence-provenance pattern "
                "for each type of three-source disagreement. It helps assess "
                "whether disagreements are mainly associated with estimated "
                "evidence or also persist among fully reported observations. "
                "Population shares use the full Quantity input population."
            ),
            "",
            _markdown_table(
                three_source_outlier_provenance_display
            ),
            "",
        ])
        if not alt_stats_display.empty:
            lines.extend([
                "**Whole-population comparison summary**",
                "",
                _markdown_table(alt_stats_display),
                "",
            ])
        if figure_paths["alt_agreement_state"].exists():
            lines.extend([
                markdown_image(
                    "Alternate-quantity kg agreement by evidence state",
                    figure_paths["alt_agreement_state"],
                    document_path=report_path,
                    width=800,
                ),
                "",
            ])

        # Existing q/net comparison
        lines.extend([
            "## 5. Quantity versus net-weight consistency",
            "",
            (
                "For observations where quantity in kg and net weight are both "
                "usable, the symmetric relative gap is"
            ),
            "",
            "$$",
            "d = \\frac{|Q-W|}{(|Q|+|W|)/2}",
            "$$",
            "",
            (
                "Population shares use the full Quantity input population of "
                f"`{n_rows:,}` observations as the denominator."
            ),
            "",
            _markdown_table(agreement_population_display),
            "",
            "**Distribution among comparable observations**",
            "",
            "| Statistic | Value |",
            "|---|---:|",
            (
                f"| Median quantity / net-weight ratio | "
                f"{_format_number(agreement_lookup.get('median_ratio'))} |"
            ),
            (
                f"| 10th percentile ratio | "
                f"{_format_number(agreement_lookup.get('p10_ratio'))} |"
            ),
            (
                f"| 90th percentile ratio | "
                f"{_format_number(agreement_lookup.get('p90_ratio'))} |"
            ),
            (
                f"| Median relative gap | "
                f"{_format_percent(agreement_lookup.get('median_relative_gap'))} |"
            ),
            (
                f"| 90th percentile relative gap | "
                f"{_format_percent(agreement_lookup.get('p90_relative_gap'))} |"
            ),
            (
                f"| 95th percentile relative gap | "
                f"{_format_percent(agreement_lookup.get('p95_relative_gap'))} |"
            ),
            "",
        ])
        if (
            figure_paths["agreement_year"].exists()
            and figure_paths["gap_distribution"].exists()
        ):
            lines.extend([
                plot_figure_pair(
                    markdown_image(
                        "Quantity and net-weight agreement by year",
                        figure_paths["agreement_year"],
                        document_path=report_path,
                        width=600,
                    ),
                    markdown_image(
                        "Quantity and net-weight relative-gap distribution",
                        figure_paths["gap_distribution"],
                        document_path=report_path,
                        width=600,
                    ),
                ),
                "",
            ])

        # Supplementary dimensions
        lines.extend([
            "## 6. Net weight alongside supplementary physical quantities",
            "",
            (
                "A positive net weight can coexist with a non-kg quantity because "
                "the fields may describe different physical dimensions of the same "
                "trade observation—for example mass together with cubic metres, "
                "square metres, or number of units. These overlaps are therefore "
                "diagnosed rather than treated as dimensional conflicts."
            ),
            "",
            (
                "These overlap diagnostics include both reported and estimated "
                "values. Population shares use the full Quantity input population; "
                "`Share within overlap (%)` is retained as an explicitly conditional "
                "composition statistic."
            ),
            "",
            "### 6.1 Non-kg `quantity` with positive net weight",
            "",
            _markdown_table(quantity_supplementary_display),
            "",
            "### 6.2 Non-kg `alt_quantity` with positive net weight",
            "",
            _markdown_table(alt_supplementary_display),
            "",
            "### 6.3 Example observations and net-weight relationships",
            "",
            (
                "The examples below illustrate how a positive net weight can coexist "
                "with a supplementary physical quantity expressed in another unit."
            ),
            "",
            _markdown_table(supplementary_examples_display),
            "",
            (
                "For cubic-metre observations, the coexistence of volume and net weight "
                "also permits an exploratory product-level comparison. The ratio "
                "`net_weight_kg / quantity_m3` is therefore summarized by HS6 below. "
                "A relatively narrow P10–P90 range around the median would indicate a "
                "more stable relationship between reported volume and net weight; a wide "
                "range would indicate substantial within-product variation."
            ),
            "",
            "#### 6.3.1 `quantity` m³ versus net weight by HS6",
            "",
            (
                "The table shows the 30 HS6 codes with the largest number of observations "
                "for which both positive net weight and positive `quantity` reported in m³ "
                "are available. `Share of comparable m³ rows (%)` uses all such comparable "
                "m³ observations as its denominator."
            ),
            "",
            _markdown_table(quantity_m3_net_relationship),
            "",
            "#### 6.3.2 `alt_quantity` m³ versus net weight by HS6",
            "",
            (
                "The same diagnostic is applied to positive `alt_quantity` observations "
                "reported in m³. The relationship is exploratory and is not currently used "
                "as a synchronization rule."
            ),
            "",
            _markdown_table(alt_quantity_m3_net_relationship),
            "",
            "### 6.4 HS structure of major supplementary units",
            "",
        ])
        unit_names = {"2": "m²", "5": "u", "12": "m³"}
        for unit_code in ("2", "5", "12"):
            table = supplementary_hs_tables.get(unit_code, pd.DataFrame())
            lines.extend([
                f"**Unit {unit_names.get(unit_code, unit_code)} (code {unit_code})**",
                "",
                _markdown_table(table),
                "",
            ])

        lines.extend([
            "### 6.5 Candidate structure for the final supplementary fields",
            "",
            (
                "The intended main physical output contains three fields: "
                "`synchronized_quantity_kg`, `supplementary_quantity`, and "
                "`supplementary_unit`. Before collapsing the original non-kg "
                "`quantity` and `alt_quantity` information into a single "
                "supplementary pair, the overlap between the two candidate "
                "sources is inspected explicitly."
            ),
            "",
            (
                "Canonical kg candidates are excluded from this diagnostic. "
                "Rows with an unavailable/unknown unit are also not treated as "
                "usable supplementary candidates because their physical "
                "dimension cannot be identified."
            ),
            "",
            _markdown_table(supplementary_candidate_overlap_display),
            "",
            (
                "The key distinction is between rows where both non-kg "
                "candidates describe the same unit and rows where they describe "
                "different units. A material share of different-unit overlap "
                "would mean that forcing both fields into one supplementary "
                "quantity/unit pair could discard a genuine second physical "
                "dimension."
            ),
            "",
            "### 6.6 Unit pairs when both non-kg candidates are present",
            "",
            (
                "The table below shows the most frequent `quantity` / "
                "`alt_quantity` unit combinations among rows where both fields "
                "contain usable non-kg candidates. Population shares use the "
                "full Quantity input population."
            ),
            "",
            _markdown_table(supplementary_unit_pairs_display),
            "",
        ])

        lines.extend([
            "### 6.7 Final supplementary output",
            "",
            (
                "The adopted supplementary rule is now applied directly in "
                "`synchronize.py`: a usable non-kg `quantity` is selected first; "
                "otherwise a usable non-kg `alt_quantity` is selected. Canonical "
                "kg-equivalent values and unknown-unit values are excluded."
            ),
            "",
            (
                "The resulting canonical supplementary fields are "
                "`supplementary_quantity` and `supplementary_unit`; "
                "`supplementary_quantity_source` and "
                "`is_supplementary_quantity_estimated` preserve provenance."
            ),
            "",
            "**Final supplementary coverage and provenance**",
            "",
            _markdown_table(supplementary_output_summary_display),
            "",
            "**Final supplementary-unit distribution**",
            "",
            _markdown_table(supplementary_output_units_display),
            "",
            (
                "The quantity/unit null-mismatch count is a structural invariant "
                "and should remain zero."
            ),
            "",
        ])

        # Net weight
        lines.extend([
            "## 7. Net-weight diagnostics",
            "",
            "### 7.1 Positive net-weight coverage by year and estimation state",
            "",
        ])

        if figure_paths["net_weight_year"].exists():
            lines.extend([
                get_table_figure_layout(
                    net_weight_year_display,
                    markdown_image(
                        "Net-weight coverage by year",
                        figure_paths["net_weight_year"],
                        document_path=report_path,
                        width=650,
                    ),
                ),
                "",
            ])
        else:
            lines.extend([
                _markdown_table(net_weight_year_display),
                "",
            ])
        lines.extend([
            "### 7.2 Net-weight magnitude distribution",
            "",
            (
                "Because trade weights span many orders of magnitude, distribution "
                "statistics are shown on the log10(kg) scale rather than with a "
                "raw linear histogram."
            ),
            "",
        ])

        if figure_paths["net_weight_distribution"].exists():
            lines.extend([
                get_table_figure_layout(
                    net_weight_distribution_display,
                    markdown_image(
                        "Net-weight log distribution by year",
                        figure_paths["net_weight_distribution"],
                        document_path=report_path,
                        width=650,
                    ),
                ),
                "",
            ])
        else:
            lines.extend([
                _markdown_table(net_weight_distribution_display),
                "",
            ])

        # Gross weight
        lines.extend([
            "## 8. Gross-weight diagnostics",
            "",
            gross_weight_statement,
            "",
            "### 8.1 Availability and possible intervention",
            "",
            "| Gross-weight diagnostic | Rows |",
            "|---|---:|",
            f"| Gross weight available | {_format_integer(gross_available)} |",
            (
                "| Gross weight available after quantity kg + net weight missing | "
                f"{_format_integer(gross_after_current)} |"
            ),
            (
                "| Gross weight available after quantity kg + net weight + alt kg missing | "
                f"{_format_integer(gross_after_alt)} |"
            ),
            f"| Gross-weight proxy actually used | {_format_integer(gross_used)} |",
            "",
            f"Configuration setting `use_as_quantity_proxy = {cfg.method.gross_weight.use_as_quantity_proxy}`.",
            "",
            "### 8.2 Gross-weight versus net-weight relationship",
            "",
            "| Statistic | Value |",
            "|---|---:|",
            f"| Comparable positive rows | {_format_integer(gross_comparable)} |",
            f"| Rows with gross weight < net weight | {_format_integer(gross_below_net)} |",
            f"| Share with gross weight < net weight | {_format_percent(gross_below_net_share)} |",
            f"| Median gross/net ratio | {_format_number(gross_relationship_lookup.get('median_ratio'))} |",
            f"| p10 ratio | {_format_number(gross_relationship_lookup.get('p10_ratio'))} |",
            f"| p90 ratio | {_format_number(gross_relationship_lookup.get('p90_ratio'))} |",
            "",
            "### 8.3 Gross/net relationship by estimation state",
            "",
            _markdown_table(gross_evidence_display),
            "",
        ])
        if figure_paths["gross_net_ratio"].exists():
            lines.extend([
                markdown_image(
                    "Gross-weight / net-weight ratio distribution",
                    figure_paths["gross_net_ratio"],
                    document_path=report_path,
                    width=800,
                ),
                "",
            ])
        lines.extend([
            "### 8.4 Positive gross weight where net weight is unavailable",
            "",
            _markdown_table(gross_without_net_year_display),
            "",
            "**Top HS codes**",
            "",
            _markdown_table(gross_without_net_hs_display),
            "",
            (
                "Gross weight remains diagnostic-only unless explicitly enabled. "
                "Rows where gross weight is smaller than net weight are especially "
                "important evidence against silently treating gross weight as a "
                "general net-mass substitute."
            ),
            "",
        ])

        # Adopted synchronization methodology
        lines.extend([
            "## 9. Synchronization methodology",
            "",
            (
                "The synchronization hierarchy is now explicit. Reported kg "
                "evidence is used before estimated evidence. Within reported "
                "evidence, exact agreement is resolved first; remaining reported "
                "rows use the adopted source precedence "
                "`quantity > alt_quantity > net_weight`. Estimated evidence is "
                "used only when no reported kg candidate is available."
            ),
            "",
            "**Applied hierarchy**",
            "",
            "```text",
            "reported kg evidence available",
            "    ↓",
            "exact 3-way agreement",
            "    ↓",
            "exact pair agreement",
            "    ↓",
            "quantity > alt_quantity > net_weight",
            "    ↓",
            "estimated quantity > alt_quantity > net_weight",
            "    ↓",
            "no_kg_evidence",
            "```",
            "",
            (
                "The configured 5% tolerance remains a diagnostic statistic and "
                "does not participate in source selection."
            ),
            "",
            "### 9.1 Methodology funnel",
            "",
            _markdown_table(methodology_funnel_display),
            "",
        ])

        if figure_paths["hierarchy"].exists():
            lines.extend([
                get_table_figure_layout(
                    methodology_funnel_display,
                    markdown_image(
                        "Cumulative kg coverage by methodology step",
                        figure_paths["hierarchy"],
                        document_path=report_path,
                        width=650,
                    ),
                ),
                "",
            ])

        lines.extend([
            "### 9.2 Detailed applied synchronization rules",
            "",
            (
                "This table is a mutually exclusive row-level partition: every "
                "observation receives exactly one synchronization rule."
            ),
            "",
            _markdown_table(methodology_rules_display),
            "",
            "### 9.3 Evidence quality of the final synchronized values",
            "",
            _markdown_table(methodology_summary_display),
            "",
            (
                "Exact reported agreement consists of three-source exact matches "
                "plus exact reported-pair matches. Reported precedence represents "
                "rows where reported evidence existed but exact agreement did not "
                "resolve the row."
            ),
            "",
            "### 9.4 Synchronization status by year",
            "",
        ])

        if figure_paths["sync_status_year"].exists():
            lines.extend([
                get_table_figure_layout(
                    sync_status_display,
                    markdown_image(
                        "Quantity synchronization rules by year",
                        figure_paths["sync_status_year"],
                        document_path=report_path,
                        width=650,
                    ),
                ),
                "",
            ])
        else:
            lines.extend([
                _markdown_table(sync_status_display),
                "",
            ])

        lines.extend([
            "### 9.5 Final synchronized coverage",
            "",
            (
                f"`{n_synchronized:,}` of `{n_rows:,}` rows "
                f"({_format_percent(synchronized_share)}) receive a positive "
                "synchronized kg quantity under the adopted methodology."
            ),
            "",
            "## 10. Remaining kg gaps",
            "",
            "### 10.1 Final physical-information coverage",
            "",
            (
                "The Quantity module produces three final physical-information fields: "
                "`synchronized_quantity_kg`, `supplementary_quantity`, and "
                "`supplementary_unit`."
            ),
            "",
            (
                "The table below summarizes the mutually exclusive final coverage "
                "states across the complete trade population."
            ),
            "",
            _markdown_table(final_physical_coverage_display),
            "",
            (
                "Canonical kg is available for "
                f"{_format_percent(canonical_kg_share, already_percent=True)} "
                "of observations. A further "
                f"{_format_percent(
                    supplementary_without_kg_share,
                    already_percent=True,
                )} "
                "cannot be represented in kg but retain a usable supplementary "
                "physical quantity. Only "
                f"{_format_percent(
                    no_physical_quantity_share,
                    already_percent=True,
                )} "
                "have neither canonical kg nor supplementary physical information."
            ),
            "",
            (
                "Among observations without canonical kg, the supplementary fields "
                "preserve usable physical information for "
                f"**{_format_percent(
                    supplementary_rescue_rate,
                    already_percent=True,
                )}**."
            ),
            "",
            "",
            _markdown_table(remaining_gap_display),
            "",
            (
                f"`{n_no_kg:,}` rows "
                f"({_format_percent(no_kg_share)}) have no usable Q/A/N kg "
                "candidate after canonicalization. Reported disagreements are "
                "not classified as unresolved conflicts because the adopted "
                "source precedence resolves them."
            ),
            "",
            "### 10.2 Distribution of synchronized kg",
            "",
            (
                "Among rows receiving a positive `synchronized_quantity_kg`, "
                "the distribution below shows the scale of the final canonical "
                "physical quantity. Because synchronized trade quantities span "
                "several orders of magnitude, the horizontal scale is expressed "
                "as `log10(synchronized_quantity_kg)`."
            ),
            "",
            markdown_image(
                "Distribution of positive synchronized quantity",
                figure_paths["synchronized_quantity_distribution"],
                document_path=report_path,
                width=850,
            ),
            "",
            "### 10.3 HS structure of rows with no synchronized kg",
            "",
            _markdown_table(missing_hs_display),
            "",
            "## 11. Methodological interpretation",
            "",
            (
                "The Quantity synchronization now separates three concepts. "
                "First, exact unit normalization creates comparable canonical kg "
                "candidates. Second, provenance establishes the primary evidence "
                "hierarchy: reported evidence precedes estimated evidence. Third, "
                "row-level synchronization uses exact agreement before applying "
                "source precedence."
            ),
            "",
            (
                "The adopted reported-source order is "
                "`quantity > alt_quantity > net_weight`. If three reported "
                "candidates are exactly equal, the common value is used. If any "
                "reported pair is exactly equal, the common pair value is used. "
                "If reported evidence remains without exact agreement, the "
                "highest-priority reported source is selected. Estimated evidence "
                "uses the same source order only when no reported kg candidate is "
                "available. Gross weight remains diagnostic-only."
            ),
            "",
            "## 12. Validation and decision",
            "",
            "| Validation | Result |",
            "|---|---|",
            f"| Synchronized output exists and is non-empty | {'PASS' if synchronized_output_valid else 'FAIL'} |",
            f"| Population accounting is complete | {'PASS' if population_partition_passed else 'FAIL'} |",
            f"| Methodology-rule accounting is complete | {'PASS' if methodology_partition_passed else 'FAIL'} |",
            f"| Final current Quantity decision | **{decision}** |",
            "",
            "## 13. Downstream recommendation",
            "",
        ])

        if ready_for_cif_fob:
            lines.extend([
                (
                    f"{ok} The Quantity output is structurally consistent with "
                    "the adopted synchronization methodology and is ready for "
                    "downstream use when pipeline integration is intentionally "
                    "enabled."
                ),
                "",
            ])
        else:
            lines.extend([
                "Quantity synchronization requires review before downstream use.",
                "",
            ])

        lines.extend([
            "## 14. Output provenance",
            "",
            "| Artifact | Path |",
            "|---|---|",
            f"| Prepared input | `{prepared_input_path}` |",
            f"| Reconciliation evidence | `{reconciliation_evidence_path}` |",
            f"| Synchronized quantity | `{synchronized_quantity_path}` |",
            f"| Diagnostics | `{diagnostics_path}` |",
            "",
        ])

        ## Add a table of contents to the report
        insert_markdown_toc(
            lines,
            min_level=2,
            max_level=3,
        )

        report_text = "\n".join(lines)
        report_path.write_text(report_text, encoding="utf-8")

        if cfg.artifacts.write_latest:
            shutil.copy2(report_path, latest_path)

        logger.info("%s Quantity decision report written: %s", ok, report_path)
        if cfg.artifacts.write_latest:
            logger.info("Latest Quantity report: %s", latest_path)

    finally:
        con.close()

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": report_path,
        "report_path": report_path,
        "latest_report_path": latest_path if cfg.artifacts.write_latest else None,
        "decision": decision,
        "ready_for_cif_fob": ready_for_cif_fob,
        "n_rows": int(n_rows),
        "n_synchronized": int(n_synchronized),
        "n_reported_three_exact": int(
            n_reported_three_exact
        ),
        "n_reported_pair_exact": int(
            n_reported_pair_exact
        ),
        "n_reported_precedence": int(
            n_reported_precedence
        ),
        "n_estimated_fallback": int(
            n_estimated_fallback
        ),
        "n_no_kg_evidence": int(n_no_kg),
    }
