"""
vacoda.steps.validation.reporting

Create the VACODA Validation decision and model-development report.

The report summarizes the evidence produced by the Validation workflow,
including:

- structural-integrity results;
- candidate training population;
- training-threshold sensitivity;
- joint threshold-grid evidence;
- temporal and product coverage;
- distributional evidence relevant to model development.

Scientific calculations belong in the producing Validation steps.
This module consumes their audit and diagnostics artifacts and turns them
into human-readable tables, figures, and methodological interpretation.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any
import math
import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from vacoda.config.paths import get_figure_dir
from vacoda.config.steps.validation import ValidationConfig
from vacoda.utils.display import get_symbols, get_table_figure_layout
from vacoda.utils.markdown import (
    insert_markdown_toc,
    markdown_image,
    markdown_styled_heading
)
from vacoda.visualization.relationships import plot_figure_pair
from vacoda.utils.sql import sql_literal


# ======================================================================
# Helpers
# ======================================================================


def valid_output(
    path: Path,
) -> bool:
    """Return True when an existing output appears usable."""

    return (
        path.is_file()
        and path.stat().st_size > 0
    )


def _format_integer(
    value: float | int | None,
) -> str:
    """Format an integer-like value for the report."""

    if value is None or pd.isna(value):
        return "NA"

    return f"{int(round(float(value))):,}"


def _format_percent(
    value: float | None,
    *,
    digits: int = 2,
) -> str:
    """Format a fractional share as a percentage."""

    if value is None or pd.isna(value):
        return "NA"

    return (
        f"{100.0 * float(value):.{digits}f}%"
    )


def _format_number(
    value: float | int | None,
    *,
    digits: int = 4,
) -> str:
    """Format a general numeric value."""

    if value is None or pd.isna(value):
        return "NA"

    return f"{float(value):,.{digits}f}"


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


def _scope_frame(
    diagnostics: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    """Return one Validation diagnostic scope."""

    return diagnostics[
        diagnostics["scope"].eq(scope)
    ].copy()


# ======================================================================
# Figures
# ======================================================================
def _plot_quantity_uv_slope_distribution(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the distribution of HS × year quantity–UV slopes."""

    frame = _scope_frame(
        diagnostics,
        "quantity_uv_relationship_groups",
    )

    frame = frame[
        frame["variable"] == "quantity_uv_slope"
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "No quantity–UV slope diagnostics are available."
        )

    values = pd.to_numeric(
        frame["value"],
        errors="coerce",
    ).dropna()

    if values.empty:
        raise RuntimeError(
            "No valid quantity–UV slope values are available."
        )

    median = float(values.median())
    q25 = float(values.quantile(0.25))
    q75 = float(values.quantile(0.75))

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    axis.hist(
        values,
        bins=40,
    )

    axis.axvline(
        0.0,
        linestyle="--",
        linewidth=1.5,
        label="No relationship (β = 0)",
    )

    axis.axvline(
        median,
        linestyle="-",
        linewidth=1.8,
        label=f"Median β = {median:.3f}",
        color='darkorange'
    )

    axis.axvline(
        q25,
        linestyle=":",
        linewidth=1.2,
        label=f"P25 = {q25:.3f}",
        color='orange'
    )

    axis.axvline(
        q75,
        linestyle=":",
        linewidth=1.2,
        label=f"P75 = {q75:.3f}",
        color='orange'
    )

    axis.set_title(
        "Distribution of HS × year quantity–UV slopes"
    )
    axis.set_xlabel(
        "Log-log slope β"
    )
    axis.set_ylabel(
        "Number of HS × year groups"
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )
    axis.set_axisbelow(True)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

def _plot_quantity_uv_r_squared_distribution(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the distribution of HS × year quantity–UV R² values."""

    frame = _scope_frame(
        diagnostics,
        "quantity_uv_relationship_groups",
    )

    frame = frame[
        frame["variable"] == "quantity_uv_r_squared"
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "No quantity–UV R² diagnostics are available."
        )

    values = pd.to_numeric(
        frame["value"],
        errors="coerce",
    ).dropna()

    if values.empty:
        raise RuntimeError(
            "No valid quantity–UV R² values are available."
        )

    median = float(values.median())

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    axis.hist(
        values,
        bins=40,
    )

    axis.axvline(
        median,
        linestyle="-",
        linewidth=1.8,
        label=f"Median R² = {median:.3f}",
        color='darkorange'
    )

    axis.set_title(
        "Distribution of HS × year quantity–UV R²"
    )
    axis.set_xlabel(
        "R²"
    )
    axis.set_ylabel(
        "Number of HS × year groups"
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )
    axis.set_axisbelow(True)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def _plot_training_uv_distance_distribution(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
    use_factor: bool = False,
) -> None:
    """Plot selected UV-distance distribution on log or factor scale."""

    frame = _scope_frame(
        diagnostics,
        "training_uv_distance_distribution",
    ).copy()

    if frame.empty:
        raise RuntimeError(
            "Training UV-distance diagnostics are unavailable."
        )

    frame = frame.sort_values(
        "quantile"
    )

    values = pd.to_numeric(
        frame["value"],
        errors="coerce",
    )

    if use_factor:
        plot_values = values.map(math.exp)
        y_label = "Factor from HS × year UV reference"
    else:
        plot_values = values
        y_label = "Absolute log-distance from HS × year reference"

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        100.0 * frame["quantile"],
        plot_values,
        marker="o",
    )

    axis.set_title(
        "Selected unit-value distance distribution"
    )

    axis.set_xlabel(
        "Quantile (%)"
    )

    axis.set_ylabel(
        y_label
    )

    if use_factor:
        axis.axhline(
            1.0,
            linestyle="--",
            linewidth=1.2,
            label="Exact reference (1×)",
        )
        axis.legend()

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

def _plot_training_reference_support_distribution(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot HS × year reference-support quantiles."""

    frame = _scope_frame(
        diagnostics,
        "training_reference_support_distribution",
    ).copy()

    if frame.empty:
        raise RuntimeError(
            "Training reference-support diagnostics are unavailable."
        )

    frame = frame.sort_values(
        "quantile"
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        100.0 * frame["quantile"],
        frame["value"],
        marker="o",
    )

    axis.set_title(
        "HS × year unit-value reference support"
    )

    axis.set_xlabel(
        "Quantile (%)"
    )

    axis.set_ylabel(
        "Reference observations"
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

def _plot_training_uv_dispersion_distribution(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot HS × year log-UV MAD quantiles."""

    frame = _scope_frame(
        diagnostics,
        "training_uv_dispersion_distribution",
    ).copy()

    if frame.empty:
        raise RuntimeError(
            "Training UV-dispersion diagnostics are unavailable."
        )

    frame = frame.sort_values(
        "quantile"
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        100.0 * frame["quantile"],
        frame["value"],
        marker="o",
    )

    axis.set_title(
        "HS × year unit-value dispersion"
    )

    axis.set_xlabel(
        "Quantile (%)"
    )

    axis.set_ylabel(
        "Log-unit-value MAD"
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

def _plot_training_normalized_uv_distance_distribution(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot selected UV distance relative to HS × year log-UV MAD."""

    frame = _scope_frame(
        diagnostics,
        "training_normalized_uv_distance_distribution",
    ).copy()

    if frame.empty:
        raise RuntimeError(
            "Training normalized UV-distance diagnostics "
            "are unavailable."
        )

    frame = frame.sort_values(
        "quantile"
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        100.0 * frame["quantile"],
        frame["value"],
        marker="o",
    )

    axis.set_title(
        "Selected unit-value distance relative to HS × year dispersion"
    )

    axis.set_xlabel(
        "Quantile (%)"
    )

    axis.set_ylabel(
        "UV log distance / log-UV MAD"
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

def _plot_training_population_funnel(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot row retention through the candidate-training funnel."""

    frame = _scope_frame(
        diagnostics,
        "population_funnel",
    )

    if frame.empty:
        raise RuntimeError(
            "No population-funnel diagnostics are available."
        )

    order = [
        "all_rows",
        "both_quantities_available",
        "both_quantities_non_estimated",
        "both_observed_with_disagreement",
        "both_observed_with_uv_distance",
        "complete_training_evidence",
    ]

    labels = {
        "all_rows": "All rows",
        "both_quantities_available": (
            "Bilateral quantity observations"
        ),
        "both_quantities_non_estimated": (
            "Direct bilateral observations"
        ),
        "both_observed_with_disagreement": (
            "Mirror disagreement available"
        ),
        "both_observed_with_uv_distance": (
            "UV distance available"
        ),
        "complete_training_evidence": (
            "All candidate-selection diagnostics"
        ),
    }

    frame["order"] = frame[
        "criterion"
    ].map(
        {
            criterion: index
            for index, criterion in enumerate(order)
        }
    )

    frame = (
        frame[
            frame["criterion"].isin(order)
        ]
        .sort_values("order")
        .reset_index(drop=True)
    )

    if frame.empty:
        raise RuntimeError(
            "Population-funnel diagnostics do not contain "
            "the expected criteria."
        )

    display_labels = [
        labels.get(
            criterion,
            criterion,
        )
        for criterion in frame["criterion"]
    ]

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    bars = axis.barh(
        display_labels,
        frame["n_rows"],
    )

    axis.bar_label(
        bars,
        labels=[
            f"{int(value):,}"
            for value in frame["n_rows"]
        ],
        padding=4,
        fontsize=8,
    )

    # First funnel stage at the top
    axis.invert_yaxis()

    axis.set_title(
        "Candidate quantity-training population"
    )
    axis.set_xlabel(
        "Number of observations"
    )
    axis.set_ylabel(
        "Evidence requirement"
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.5,
    )
    axis.set_axisbelow(True)

    # Give the value labels some room to the right
    axis.margins(x=0.12)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

def _plot_training_evidence_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot directly observed bilateral quantity coverage by year."""

    frame = _scope_frame(
        diagnostics,
        "year_coverage",
    )

    if frame.empty:
        raise RuntimeError(
            "No year-coverage model diagnostics are available."
        )

    frame = frame.sort_values(
        "year"
    ).copy()

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        frame["year"],
        100.0 * frame["share_of_all"],
        marker="o",
    )

    axis.set_title(
        "Directly observed bilateral quantity evidence by year"
    )
    axis.set_xlabel(
        "Year"
    )
    axis.set_ylabel(
        "Share of annual Validation rows (%)"
    )
    axis.set_xticks(
        frame["year"].astype(int)
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def _plot_training_evidence_distributions(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot selected quantiles for the main pseudo-truth evidence measures.

    Values are displayed on their native scales. This figure is intended
    to reveal the shape of candidate-training evidence before final
    thresholds are selected.
    """

    frame = _scope_frame(
        diagnostics,
        "distribution",
    )

    if frame.empty:
        raise RuntimeError(
            "No distribution diagnostics are available."
        )

    variables = {
        "quantity_relative_difference": (
            "Mirror quantity disagreement"
        ),
        "bilateral_max_unit_value_log_distance": (
            "Worst-side reporter UV distance"
        ),
    }

    frame = frame[
        frame["variable"].isin(
            variables
        )
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "Expected model-development distribution variables "
            "are unavailable."
        )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    for variable, group in frame.groupby(
        "variable",
        sort=False,
    ):
        group = group.sort_values(
            "quantile"
        )

        axis.plot(
            100.0 * group["quantile"],
            group["value"],
            marker="o",
            label=variables.get(
                variable,
                variable,
            ),
        )

    axis.set_title(
        "Distributional evidence in directly observed bilateral rows"
    )
    axis.set_xlabel(
        "Quantile (%)"
    )
    axis.set_ylabel(
        "Diagnostic value"
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def _plot_export_import_quantity_ratio_distribution(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the distribution of exporter/importer log quantity ratios."""

    frame = _scope_frame(
        diagnostics,
        "quantity_ratio_distribution",
    )

    if frame.empty:
        raise RuntimeError(
            "No quantity-ratio distribution diagnostics are available."
        )

    frame = frame.sort_values(
        "value"
    ).copy()

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.bar(
        frame["value"],
        frame["n_rows"],
        width=0.08,
        alpha=0.85,
    )

    axis.axvline(
        0.0,
        linestyle="--",
        linewidth=1.5,
        label="Exporter = importer",
        color="red",
    )

    axis.set_title(
        "Exporter/importer quantity ratio distribution"
    )
    axis.set_xlabel(
        "log(exporter quantity / importer quantity)"
    )
    axis.set_ylabel(
        "Number of observations"
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )




def _candidate_label(value: str) -> str:
    """Return a readable label for a fixed quantity candidate."""

    labels = {
        "simple_mean": "Simple mean",
        "value_weighted": "Value-weighted",
        "quantity_reliability_weighted": "Quantity-reliability-weighted",
        "tie": "Tie",
    }

    return labels.get(str(value), str(value))


def _disagreement_band_order(
    value: str,
) -> tuple[int, str]:
    """Return the intended order of mirror-quantity disagreement bands."""

    label = str(value).strip()

    order = {
        "5% or less": 0,
        "5–10%": 1,
        "10–20%": 2,
        "20–30%": 3,
        "30–50%": 4,
        "50–90%": 5,
        "> 90%": 6,
    }

    return (
        order.get(label, 999),
        label,
    )

def _plot_quantity_candidate_uv_summary(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot overall UV-distance evidence for the fixed quantity candidates."""

    frame = _scope_frame(
        diagnostics,
        "quantity_candidate_summary",
    )

    frame = frame[
        frame["variable"].isin(
            [
                "median_uv_log_distance",
                "p90_uv_log_distance",
                "p95_uv_log_distance",
            ]
        )
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "No fixed quantity-candidate summary diagnostics are available."
        )

    pivot = frame.pivot_table(
        index="candidate",
        columns="variable",
        values="value",
        aggfunc="first",
    )

    candidates = [
        value
        for value in (
            "simple_mean",
            "value_weighted",
            "quantity_reliability_weighted",
        )
        if value in pivot.index
    ]

    metrics = [
        ("median_uv_log_distance", "Median"),
        ("p90_uv_log_distance", "P90"),
        ("p95_uv_log_distance", "P95"),
    ]

    figure, axis = plt.subplots(figsize=(10, 6))

    x = list(range(len(candidates)))
    width = 0.24

    for offset, (metric, label) in enumerate(metrics):
        values = [
            float(pivot.loc[candidate, metric])
            if metric in pivot.columns and pd.notna(pivot.loc[candidate, metric])
            else float("nan")
            for candidate in candidates
        ]
        positions = [
            value + (offset - 1) * width
            for value in x
        ]
        axis.bar(
            positions,
            values,
            width=width,
            label=label,
        )

    axis.set_xticks(x)
    axis.set_xticklabels(
        [_candidate_label(value) for value in candidates]
    )
    axis.set_ylabel("UV log distance")
    axis.set_title("Fixed quantity candidates: unit-value plausibility")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.set_axisbelow(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_quantity_candidate_by_disagreement(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
    scale: str = "log",
) -> None:
    """
    Plot candidate median UV distance across mirror-disagreement bands.

    scale:
        "log"    -> absolute log-distance from the product-year UV reference
        "factor" -> multiplicative factor from the product-year UV reference
    """

    if scale not in {"log", "factor"}:
        raise ValueError(
            "scale must be 'log' or 'factor'."
        )

    frame = _scope_frame(
        diagnostics,
        "quantity_candidate_by_disagreement",
    )

    frame = frame[
        frame["variable"].eq("median_uv_log_distance")
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "No quantity-candidate disagreement diagnostics are available."
        )

    bands = sorted(
        frame["disagreement_band"]
        .dropna()
        .astype(str)
        .unique(),
        key=_disagreement_band_order,
    )

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    for candidate in (
        "simple_mean",
        "value_weighted",
        "quantity_reliability_weighted",
    ):
        subset = frame[
            frame["candidate"].eq(candidate)
        ].copy()

        if subset.empty:
            continue

        lookup = {
            str(row.disagreement_band): row.value
            for row in subset.itertuples(index=False)
        }

        values = [
            float(lookup[band])
            if band in lookup
            and pd.notna(lookup[band])
            else float("nan")
            for band in bands
        ]

        if scale == "factor":
            values = [
                math.exp(value)
                if pd.notna(value)
                else float("nan")
                for value in values
            ]

        axis.plot(
            bands,
            values,
            marker="o",
            label=_candidate_label(candidate),
        )

    axis.set_xlabel(
        "Mirror quantity disagreement"
    )

    if scale == "log":
        axis.set_title(
            "Candidate UV plausibility by mirror-quantity disagreement"
        )
        axis.set_ylabel(
            "Median UV log distance"
        )

    else:
        axis.set_title(
            "Candidate UV factor-distance by mirror-quantity disagreement"
        )
        axis.set_ylabel(
            "Median factor from product-year UV reference"
        )

        axis.axhline(
            1.0,
            linestyle="--",
            linewidth=1.2,
            label="Exact reference (1×)",
        )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )
    axis.set_axisbelow(True)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

def _plot_quantity_candidate_by_disagreement_deprecated(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot candidate median UV distance across mirror-disagreement bands."""

    frame = _scope_frame(
        diagnostics,
        "quantity_candidate_by_disagreement",
    )

    frame = frame[
        frame["variable"].eq("median_uv_log_distance")
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "No quantity-candidate disagreement diagnostics are available."
        )

    bands = sorted(
        frame["disagreement_band"].dropna().astype(str).unique(),
        key=_disagreement_band_order,
    )

    figure, axis = plt.subplots(figsize=(11, 6))

    for candidate in (
        "simple_mean",
        "value_weighted",
        "quantity_reliability_weighted",
    ):
        subset = frame[
            frame["candidate"].eq(candidate)
        ].copy()

        if subset.empty:
            continue

        lookup = {
            str(row.disagreement_band): row.value
            for row in subset.itertuples(index=False)
        }

        values = [
            float(lookup[band])
            if band in lookup and pd.notna(lookup[band])
            else float("nan")
            for band in bands
        ]

        axis.plot(
            bands,
            values,
            marker="o",
            label=_candidate_label(candidate),
        )

    axis.set_title(
        "Candidate UV plausibility by mirror-quantity disagreement"
    )
    axis.set_xlabel("Mirror quantity disagreement")
    axis.set_ylabel("Median UV log distance")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.set_axisbelow(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_quantity_candidate_closest_reference(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot closest-to-reference shares for the fixed quantity candidates."""

    frame = _scope_frame(
        diagnostics,
        "quantity_candidate_closest_reference",
    )

    if frame.empty:
        raise RuntimeError(
            "No closest-reference quantity-candidate diagnostics are available."
        )

    order = [
        "simple_mean",
        "value_weighted",
        "quantity_reliability_weighted",
        "tie",
    ]

    frame = frame[
        frame["candidate"].isin(order)
    ].copy()

    frame["order"] = frame["candidate"].map(
        {value: index for index, value in enumerate(order)}
    )
    frame = frame.sort_values("order")

    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(
        [_candidate_label(value) for value in frame["candidate"]],
        100.0 * pd.to_numeric(frame["value"], errors="coerce"),
    )
    axis.bar_label(
        bars,
        labels=[
            f"{100.0 * float(value):.1f}%"
            if pd.notna(value)
            else "NA"
            for value in frame["value"]
        ],
        padding=3,
        fontsize=8,
    )
    axis.set_ylabel("Share of comparable rows (%)")
    axis.set_title("Candidate closest to the product-year UV reference")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_quantity_candidate_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot candidate median UV distance by year."""

    frame = _scope_frame(
        diagnostics,
        "quantity_candidate_by_year",
    )

    frame = frame[
        frame["variable"].eq("median_uv_log_distance")
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "No quantity-candidate temporal diagnostics are available."
        )

    figure, axis = plt.subplots(figsize=(11, 6))

    for candidate in (
        "simple_mean",
        "value_weighted",
        "quantity_reliability_weighted",
    ):
        subset = frame[
            frame["candidate"].eq(candidate)
        ].copy()

        if subset.empty:
            continue

        subset = subset.sort_values("year")

        axis.plot(
            pd.to_numeric(subset["year"], errors="coerce"),
            pd.to_numeric(subset["value"], errors="coerce"),
            marker="o",
            label=_candidate_label(candidate),
        )

    axis.set_title("Candidate unit-value plausibility by year")
    axis.set_xlabel("Year")
    axis.set_ylabel("Median UV log distance")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.set_axisbelow(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_quantity_candidate_product_distribution(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot product-level distributions of candidate median UV distance."""

    frame = _scope_frame(
        diagnostics,
        "quantity_candidate_by_product",
    )

    frame = frame[
        frame["variable"].eq("median_uv_log_distance")
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "No quantity-candidate product diagnostics are available."
        )

    candidates = [
        value
        for value in (
            "simple_mean",
            "value_weighted",
            "quantity_reliability_weighted",
        )
        if frame["candidate"].eq(value).any()
    ]

    data = [
        pd.to_numeric(
            frame.loc[frame["candidate"].eq(candidate), "value"],
            errors="coerce",
        ).dropna().to_numpy()
        for candidate in candidates
    ]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.boxplot(
        data,
        labels=[_candidate_label(value) for value in candidates],
        showfliers=False,
    )
    axis.set_title(
        "Product-level distribution of candidate median UV distance"
    )
    axis.set_ylabel("Median UV log distance within HS product")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_quantity_candidate_spread(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the spread among fixed quantity candidates by disagreement band."""

    frame = _scope_frame(
        diagnostics,
        "quantity_candidate_spread",
    )

    if frame.empty:
        raise RuntimeError(
            "No quantity-candidate spread diagnostics are available."
        )

    bands = sorted(
        frame["disagreement_band"].dropna().astype(str).unique(),
        key=_disagreement_band_order,
    )

    figure, axis = plt.subplots(figsize=(11, 6))

    for variable, label in (
        ("median_relative_candidate_spread", "Median"),
        ("p95_relative_candidate_spread", "P95"),
    ):
        subset = frame[
            frame["variable"].eq(variable)
        ].copy()
        lookup = {
            str(row.disagreement_band): row.value
            for row in subset.itertuples(index=False)
        }
        values = [
            100.0 * float(lookup[band])
            if band in lookup and pd.notna(lookup[band])
            else float("nan")
            for band in bands
        ]
        axis.plot(
            bands,
            values,
            marker="o",
            label=label,
        )

    axis.set_title("Spread among the three fixed quantity candidates")
    axis.set_xlabel("Mirror quantity disagreement")
    axis.set_ylabel("Relative candidate spread (%)")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.set_axisbelow(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_quantity_candidate_conditioned_summary(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot quantity-conditioned UV residuals for fixed candidates."""

    frame = _scope_frame(
        diagnostics,
        "quantity_candidate_conditioned_summary",
    )
    frame = frame[
        frame["variable"].eq("median_conditioned_uv_abs_residual")
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "No quantity-conditioned candidate summary diagnostics are available."
        )

    labels = {
        "simple_mean": "Simple mean",
        "value_weighted": "Value-weighted",
        "quantity_reliability_weighted": "Quantity-reliability-weighted",
    }
    frame["label"] = frame["candidate"].map(labels).fillna(frame["candidate"])

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(frame["label"], frame["value"])
    axis.set_title("Quantity-conditioned UV plausibility by candidate")
    axis.set_ylabel("Median absolute log-UV residual")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_quantity_candidate_conditioned_by_disagreement(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot conditioned candidate residuals by mirror-disagreement band."""

    frame = _scope_frame(
        diagnostics,
        "quantity_candidate_conditioned_by_disagreement",
    )
    frame = frame[
        frame["variable"].eq("median_conditioned_uv_abs_residual")
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "No conditioned candidate-by-disagreement diagnostics are available."
        )

    band_order = [
        "5% or less",
        "5–10%",
        "10–20%",
        "20–30%",
        "30–50%",
        "50–90%",
        "> 90%",
    ]
    candidate_labels = {
        "simple_mean": "Simple mean",
        "value_weighted": "Value-weighted",
        "quantity_reliability_weighted": "Quantity-reliability-weighted",
    }

    figure, axis = plt.subplots(figsize=(11, 6))
    for candidate, group in frame.groupby("candidate", sort=False):
        group = group.copy()
        group["order"] = group["disagreement_band"].map(
            {value: index for index, value in enumerate(band_order)}
        )
        group = group.sort_values("order")
        axis.plot(
            group["disagreement_band"],
            group["value"],
            marker="o",
            label=candidate_labels.get(candidate, candidate),
        )

    axis.set_title("Quantity-conditioned UV plausibility by mirror disagreement")
    axis.set_xlabel("Mirror quantity disagreement")
    axis.set_ylabel("Median absolute log-UV residual")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_quantity_candidate_closest_conditioned(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot closest candidate under quantity-conditioned UV residuals."""

    frame = _scope_frame(
        diagnostics,
        "quantity_candidate_closest_conditioned",
    )
    if frame.empty:
        raise RuntimeError(
            "No closest quantity-conditioned candidate diagnostics are available."
        )

    labels = {
        "simple_mean": "Simple mean",
        "value_weighted": "Value-weighted",
        "quantity_reliability_weighted": "Quantity-reliability-weighted",
        "tie": "Tie",
    }
    frame["label"] = frame["candidate"].map(labels).fillna(frame["candidate"])

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(frame["label"], 100.0 * frame["value"])
    axis.set_title("Candidate closest to quantity-conditioned expected UV")
    axis.set_ylabel("Share of comparable rows (%)")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


# ======================================================================
# Main reporting function
# ======================================================================


def build_validation_report(
    *,
    cfg: ValidationConfig,
    prepared_input_path: Path,
    structural_audit_path: Path,
    diagnostics_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Create the current VACODA Validation Markdown decision report.

    At the present model-development stage, the report documents:

    - the canonical Validation input;
    - structural-integrity results;
    - candidate training population evidence;
    - sensitivity to possible training-sample thresholds;
    - coverage across years and products;
    - the current training-population decision status.

    The report will be extended as later Validation stages become active.
    """

    prepared_input_path = Path(prepared_input_path)
    structural_audit_path = Path(structural_audit_path)
    diagnostics_path = Path(diagnostics_path)

    for label, path in (
        (
            "prepared Validation input",
            prepared_input_path,
        ),
        (
            "Validation structural audit",
            structural_audit_path,
        ),
        (
            "Validation model diagnostics",
            diagnostics_path,
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    # ------------------------------------------------------------------
    # Report configuration
    # ------------------------------------------------------------------

    if not cfg.artifacts.enable:
        return {
            "status": "disabled",
            "output_path": None,
            "report_path": None,
            "latest_report_path": None,
            "ready_for_training_decision": False,
        }

    report_path = cfg.artifacts_path(
        run_id
    )

    latest_path = (
        cfg.artifacts_latest_path()
    )

    figure_dir = (
        get_figure_dir()
        / "validation"
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

    # if (
    #     not force
    #     and valid_output(report_path)
    # ):
    #     logger.info(
    #         "REUSE — Validation report already exists: %s",
    #         report_path,
    #     )

    #     return {
    #         "status": "reused",
    #         "output_path": report_path,
    #         "report_path": report_path,
    #         "latest_report_path": (
    #             latest_path
    #             if latest_path.exists()
    #             else None
    #         ),
    #         "ready_for_training_decision": True,
    #     }

    structural_sql = sql_literal(
        structural_audit_path
    )

    diagnostics_sql = sql_literal(
        diagnostics_path
    )

    info = get_symbols(
        "info"
    )

    ok = get_symbols(
        "check"
    )

    logger.info(
        "%s Building Validation decision report.",
        info,
    )

    logger.info(
        "  Prepared input: %s",
        prepared_input_path,
    )

    logger.info(
        "  Structural audit: %s",
        structural_audit_path,
    )

    logger.info(
        "  Model diagnostics: %s",
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

        # ==============================================================
        # Read produced artifacts
        # ==============================================================

        structural_audit = con.execute(
            f"""
            SELECT *
            FROM read_parquet({structural_sql})
            """
        ).fetchdf()

        diagnostics = con.execute(
            f"""
            SELECT *
            FROM read_parquet({diagnostics_sql})
            """
        ).fetchdf()

        if len(structural_audit) != 1:
            raise RuntimeError(
                "Validation structural audit must contain "
                "exactly one row."
            )

        if diagnostics.empty:
            raise RuntimeError(
                "Validation model diagnostics contain no rows."
            )

        required_diagnostic_columns = {
            "scope",
            "criterion",
            "candidate",
            "n_rows",
            "share_of_all",
            "share_of_parent",
        }

        missing_diagnostic_columns = (
            required_diagnostic_columns
            - set(diagnostics.columns)
        )

        if missing_diagnostic_columns:
            raise RuntimeError(
                "Validation model diagnostics are missing "
                "required columns: "
                + ", ".join(
                    sorted(
                        missing_diagnostic_columns
                    )
                )
            )

        # ==============================================================
        # Structural audit
        # ==============================================================

        audit_row = structural_audit.iloc[
            0
        ]

        audit_passed = bool(
            audit_row["audit_passed"]
        )

        n_rows = int(
            audit_row["n_rows"]
        )

        n_duplicate_rows = int(
            audit_row[
                "n_duplicate_rows"
            ]
        )

        n_null_key_rows = int(
            audit_row[
                "n_null_key_rows"
            ]
        )

        n_invalid_year_rows = int(
            audit_row[
                "n_invalid_year_rows"
            ]
        )

        n_invalid_hs_rows = int(
            audit_row[
                "n_invalid_hs_rows"
            ]
        )

        n_invalid_nominal_value_rows = int(
            audit_row[
                "n_invalid_nominal_value_rows"
            ]
        )

        n_invalid_constant_value_rows = int(
            audit_row[
                "n_invalid_constant_value_rows"
            ]
        )

        n_invalid_simple_mean_rows = int(
            audit_row[
                "n_invalid_simple_mean_rows"
            ]
        )

        n_invalid_value_weighted_rows = int(
            audit_row[
                "n_invalid_value_weighted_rows"
            ]
        )

        n_invalid_quantity_reliability_weighted_rows = int(
            audit_row[
                "n_invalid_quantity_reliability_weighted_rows"
            ]
        )

        n_missing_quantity_rows = int(
            audit_row[
                "n_missing_quantity_rows"
            ]
        )

        n_available_quantity_rows = int(
            audit_row[
                "n_available_quantity_rows"
            ]
        )

        structural_summary = pd.DataFrame(
            [
                {
                    "Check": "Rows",
                    "Result": _format_integer(
                        n_rows
                    ),
                },
                {
                    "Check": (
                        "Duplicate rows"
                    ),
                    "Result": _format_integer(
                        n_duplicate_rows
                    ),
                },
                {
                    "Check": "Null rows",
                    "Result": _format_integer(
                        n_null_key_rows
                    ),
                },
                {
                    "Check": "Invalid year rows",
                    "Result": _format_integer(
                        n_invalid_year_rows
                    ),
                },
                {
                    "Check": "Invalid HS rows",
                    "Result": _format_integer(
                        n_invalid_hs_rows
                    ),
                },
                {
                    "Check": "Invalid nominal values",
                    "Result": _format_integer(
                        n_invalid_nominal_value_rows
                    ),
                },
                {
                    "Check": (
                        "Invalid constant-USD values"
                    ),
                    "Result": _format_integer(
                        n_invalid_constant_value_rows
                    ),
                },
                {
                    "Check": "Invalid simple-mean quantities",
                    "Result": _format_integer(
                        n_invalid_simple_mean_rows
                    ),
                },
                {
                    "Check": "Invalid value-weighted quantities",
                    "Result": _format_integer(
                        n_invalid_value_weighted_rows
                    ),
                },
                {
                    "Check": (
                        "Invalid quantity-reliability-weighted quantities"
                    ),
                    "Result": _format_integer(
                        n_invalid_quantity_reliability_weighted_rows
                    ),
                },
                {
                    "Check": "Structural audit passed",
                    "Result": str(
                        audit_passed
                    ),
                },
            ]
        )


        # ============================================================
        # Training Sample Diagnostics
        # ============================================================

        training_candidate_final_rule = _scope_frame(
            diagnostics,
            "training_candidate_final_rule",
        )

        training_sample_summary = _scope_frame(
            diagnostics,
            "training_sample_summary",
        )

        training_uv_distance = _scope_frame(
            diagnostics,
            "training_uv_distance_distribution",
        )

        training_reference_support = _scope_frame(
            diagnostics,
            "training_reference_support_distribution",
        )

        training_uv_dispersion = _scope_frame(
            diagnostics,
            "training_uv_dispersion_distribution",
        )

        training_normalized_uv_distance = _scope_frame(
            diagnostics,
            "training_normalized_uv_distance_distribution",
        )
        training_normalized_uv_cutoffs = _scope_frame(
            diagnostics,
            "training_normalized_uv_cutoff_sensitivity",
        )
        training_by_year = _scope_frame(
            diagnostics,
            "training_sample_by_year",
        )

        training_by_product = _scope_frame(
            diagnostics,
            "training_sample_by_product",
        )

        training_summary_labels = {
            "all_rows": "All Validation rows",
            "selected_unit_value_available": (
                "Selected unit value available"
            ),
            "reference_population": (
                "Direct bilateral reference population"
            ),
            "hs_year_reference_available": (
                "HS × year UV reference available"
            ),
            "uv_distance_available": (
                "Selected UV distance available"
            ),
            "hard_eligible": (
                "Mechanically eligible for training evaluation"
            ),
        }

        training_sample_summary_display = (
            training_sample_summary.copy()
        )

        training_sample_summary_display[
            "Population"
        ] = training_sample_summary_display[
            "criterion"
        ].map(
            training_summary_labels
        )

        training_sample_summary_display[
            "Rows"
        ] = training_sample_summary_display[
            "n_rows"
        ].map(
            _format_integer
        )

        training_sample_summary_display[
            "Share of Validation rows"
        ] = training_sample_summary_display[
            "share_of_all"
        ].map(
            _format_percent
        )

        training_sample_summary_display = (
            training_sample_summary_display[
                [
                    "Population",
                    "Rows",
                    "Share of Validation rows",
                ]
            ]
        )


        def _quantile_label(
            value: float,
        ) -> str:

            return (
                f"P{100 * float(value):g}"
            )


        training_distribution_rows = []

        for label, frame in (
            (
                "Selected UV log distance",
                training_uv_distance,
            ),
            (
                "HS × year reference support",
                training_reference_support,
            ),
            (
                "HS × year UV log MAD",
                training_uv_dispersion,
            ),
            (
                "Selected UV distance / HS × year log MAD",
                training_normalized_uv_distance,
            ),
        ):

            for row in frame.itertuples(
                index=False
            ):
                training_distribution_rows.append(
                    {
                        "Evidence": label,
                        "Quantile": _quantile_label(
                            row.quantile
                        ),
                        "Value": _format_number(
                            row.value,
                            digits=4,
                        ),
                    }
                )

        training_distribution_display = (
            pd.DataFrame(
                training_distribution_rows
            )
        )

        training_normalized_uv_distance_display = (
            training_normalized_uv_distance[
                [
                    "quantile",
                    "value",
                ]
            ].copy()
        )

        training_normalized_uv_distance_display["Quantile"] = (
            training_normalized_uv_distance_display["quantile"]
            .map(_quantile_label)
        )

        training_normalized_uv_distance_display["Distance (MAD)"] = (
            training_normalized_uv_distance_display["value"]
            .map(
                lambda value: _format_number(
                    value,
                    digits=4,
                )
            )
        )

        training_normalized_uv_distance_display = (
            training_normalized_uv_distance_display[
                [
                    "Quantile",
                    "Distance (MAD)",
                ]
            ]
        )
        training_uv_dispersion_display = training_uv_dispersion[
            [
                "quantile",
                "value",
            ]
        ].copy()

        training_uv_dispersion_display["Quantile"] = (
            training_uv_dispersion_display["quantile"]
            .map(_quantile_label)
        )

        training_uv_dispersion_display["Log-UV MAD"] = (
            training_uv_dispersion_display["value"]
            .map(
                lambda value: _format_number(
                    value,
                    digits=4,
                )
            )
        )

        training_uv_dispersion_display[
            "Multiplicative dispersion"
        ] = (
            training_uv_dispersion_display["value"]
            .map(
                lambda value: (
                    f"{math.exp(float(value)):.2f}×"
                )
            )
        )
        training_uv_dispersion_display = (
            training_uv_dispersion_display[
                [
                    "Quantile",
                    "Log-UV MAD",
                    "Multiplicative dispersion",
                ]
            ]
        )
        training_reference_support_display = training_reference_support[
            [
                "quantile",
                "value",
            ]
        ].copy()

        training_reference_support_display["Quantile"] = (
            training_reference_support_display["quantile"]
            .map(_quantile_label)
        )

        training_reference_support_display["Reference observations"] = (
            training_reference_support_display["value"]
            .map(_format_integer)
        )

        training_reference_support_display = (
            training_reference_support_display[
                [
                    "Quantile",
                    "Reference observations",
                ]
            ]
        )
        training_uv_distance_display = training_uv_distance[
            [
                "quantile",
                "value",
            ]
        ].copy()

        training_uv_distance_display["Quantile"] = (
            training_uv_distance_display["quantile"]
            .map(_quantile_label)
        )

        training_uv_distance_display["Log distance"] = (
            training_uv_distance_display["value"]
            .map(
                lambda value: _format_number(
                    value,
                    digits=4,
                )
            )
        )

        training_uv_distance_display["Factor from reference"] = (
            training_uv_distance_display["value"]
            .map(
                lambda value: f"{math.exp(float(value)):.2f}×"
            )
        )

        training_uv_distance_display = (
            training_uv_distance_display[
                [
                    "Quantile",
                    "Log distance",
                    "Factor from reference",
                ]
            ]
        )

        candidate_training_rule_display = pd.DataFrame(
            [
                {
                    "Criterion": (
                        "Direct bilateral + support ≥ 30 + normalized UV distance ≤ 2.5 MAD"
                    ),
                    "Rows": _format_integer(
                        training_candidate_final_rule.iloc[0]["n_rows"]
                    ),
                    "Share of evaluable reference rows": _format_percent(
                        training_candidate_final_rule.iloc[0]["share_of_parent"]
                    ),
                    "Share of all Validation rows": _format_percent(
                        training_candidate_final_rule.iloc[0]["share_of_all"]
                    ),
                }
            ]
        )
        # ============================================================
        # Normalized UV cutoff sensitivity
        # ============================================================

        reference_cutoffs = (
            training_normalized_uv_cutoffs[
                training_normalized_uv_cutoffs[
                    "criterion"
                ].eq(
                    "reference_population"
                )
            ]
            .copy()
            .sort_values(
                "uv_distance_threshold"
            )
        )

        hard_cutoffs = (
            training_normalized_uv_cutoffs[
                training_normalized_uv_cutoffs[
                    "criterion"
                ].eq(
                    "hard_eligible_population"
                )
            ]
            .copy()
            .sort_values(
                "uv_distance_threshold"
            )
        )


        normalized_cutoff_display = (
            reference_cutoffs[
                [
                    "uv_distance_threshold",
                    "n_rows",
                    "share_of_parent",
                ]
            ]
            .rename(
                columns={
                    "n_rows": (
                        "reference_rows"
                    ),
                    "share_of_parent": (
                        "reference_share"
                    ),
                }
            )
            .merge(
                hard_cutoffs[
                    [
                        "uv_distance_threshold",
                        "n_rows",
                        "share_of_parent",
                    ]
                ].rename(
                    columns={
                        "n_rows": (
                            "hard_rows"
                        ),
                        "share_of_parent": (
                            "hard_share"
                        ),
                    }
                ),
                on="uv_distance_threshold",
                how="outer",
                validate="one_to_one",
            )
            .sort_values(
                "uv_distance_threshold"
            )
        )


        normalized_cutoff_display = pd.DataFrame(
            {
                "Maximum normalized UV distance": (
                    normalized_cutoff_display[
                        "uv_distance_threshold"
                    ].map(
                        lambda value: (
                            f"≤ {float(value):g} MAD"
                        )
                    )
                ),

                "Direct bilateral reference rows": (
                    normalized_cutoff_display[
                        "reference_rows"
                    ].map(
                        _format_integer
                    )
                ),

                "Share of evaluable reference rows": (
                    normalized_cutoff_display[
                        "reference_share"
                    ].map(
                        _format_percent
                    )
                ),

                "All hard-eligible rows": (
                    normalized_cutoff_display[
                        "hard_rows"
                    ].map(
                        _format_integer
                    )
                ),

                "Share of evaluable hard-eligible rows": (
                    normalized_cutoff_display[
                        "hard_share"
                    ].map(
                        _format_percent
                    )
                ),
            }
        )
        # ==============================================================
        # Quantity availability by reporting side
        # ==============================================================

        quantity_availability = _scope_frame(
            diagnostics,
            "quantity_availability",
        )

        if quantity_availability.empty:
            raise RuntimeError(
                "No quantity-availability diagnostics are available."
            )

        quantity_availability_order = [
            "both",
            "export_only",
            "import_only",
            "neither",
        ]

        quantity_availability_labels = {
            "both": (
                "Both exporter and importer quantities available"
            ),
            "export_only": (
                "Exporter quantity only"
            ),
            "import_only": (
                "Importer quantity only"
            ),
            "neither": (
                "Neither exporter nor importer quantity available"
            ),
        }

        quantity_availability["order"] = (
            quantity_availability[
                "criterion"
            ].map(
                {
                    criterion: index
                    for index, criterion
                    in enumerate(
                        quantity_availability_order
                    )
                }
            )
        )

        quantity_availability = (
            quantity_availability[
                quantity_availability[
                    "criterion"
                ].isin(
                    quantity_availability_order
                )
            ]
            .sort_values("order")
            .reset_index(drop=True)
        )

        if len(quantity_availability) != 4:
            raise RuntimeError(
                "Quantity-availability diagnostics must contain "
                "exactly four mutually exclusive categories."
            )

        n_quantity_availability_rows = int(
            quantity_availability[
                "n_rows"
            ].sum()
        )

        if n_quantity_availability_rows != n_rows:
            raise RuntimeError(
                "Quantity-availability categories do not sum to "
                "the full Validation population."
            )

        quantity_availability_display = pd.DataFrame(
            {
                "Quantity availability": (
                    quantity_availability[
                        "criterion"
                    ].map(
                        quantity_availability_labels
                    )
                ),
                "Rows": (
                    quantity_availability[
                        "n_rows"
                    ].map(
                        _format_integer
                    )
                ),
                "Share of all Validation rows": (
                    quantity_availability[
                        "share_of_all"
                    ].map(
                        _format_percent
                    )
                ),
            }
        )


        quantity_availability_counts = {
            str(row.criterion): int(row.n_rows)
            for row in quantity_availability.itertuples(
                index=False
            )
        }

        n_quantity_both = (
            quantity_availability_counts["both"]
        )

        n_quantity_export_only = (
            quantity_availability_counts[
                "export_only"
            ]
        )

        n_quantity_import_only = (
            quantity_availability_counts[
                "import_only"
            ]
        )

        n_quantity_neither = (
            quantity_availability_counts["neither"]
        )

        n_quantity_any = (
            n_quantity_both
            + n_quantity_export_only
            + n_quantity_import_only
        )

        quantity_any_share = (
            n_quantity_any / n_rows
            if n_rows > 0
            else 0.0
        )

        quantity_bilateral_share = (
            n_quantity_both / n_rows
            if n_rows > 0
            else 0.0
        )


        # ==============================================================
        # Training-population funnel
        # ==============================================================

        funnel = _scope_frame(
            diagnostics,
            "population_funnel",
        )

        funnel_order = [
            "all_rows",
            "both_quantities_available",
            "both_quantities_non_estimated",
            "both_observed_with_disagreement",
            "both_observed_with_uv_distance",
            "complete_training_evidence",
        ]

        funnel_labels = {
            "all_rows": "All Validation rows",
            "both_quantities_available": (
                "Bilateral quantity observation"
            ),
            "both_quantities_non_estimated": (
                "Direct bilateral quantity observations"
            ),
            "both_observed_with_disagreement": (
                "Direct bilateral observations with exporter–importer quantity difference available"
            ),
            "both_observed_with_uv_distance": (
                "Direct bilateral observations with unit-value distance available"
            ),
            "complete_training_evidence": (
                "Direct bilateral observations with all candidate-selection diagnostics available"
            ),
        }

        funnel["order"] = funnel[
            "criterion"
        ].map(
            {
                criterion: index
                for index, criterion
                in enumerate(funnel_order)
            }
        )

        funnel = (
            funnel[
                funnel["criterion"].isin(
                    funnel_order
                )
            ]
            .sort_values("order")
            .reset_index(drop=True)
        )

        funnel_display = pd.DataFrame(
            {
                "Population": funnel[
                    "criterion"
                ].map(
                    funnel_labels
                ),
                "Rows": funnel[
                    "n_rows"
                ].map(
                    _format_integer
                ),
                "Share of all": funnel[
                    "share_of_all"
                ].map(
                    _format_percent
                ),
                "Share of parent (previous group)": funnel[
                    "share_of_parent"
                ].map(
                    _format_percent
                ),
            }
        )

        # ==============================================================
        # Threshold sensitivity
        # ==============================================================

        sensitivity = _scope_frame(
            diagnostics,
            "threshold_sensitivity",
        )

        sensitivity_rows: list[
            dict[str, Any]
        ] = []

        for row in sensitivity.itertuples(
            index=False
        ):

            criterion = str(
                row.criterion
            )

            if (
                criterion
                == "quantity_relative_difference"
            ):
                threshold_label = (
                    f"≤ "
                    f"{100.0 * float(row.disagreement_threshold):.0f}%"
                )

                evidence_label = (
                    "Mirror quantity disagreement"
                )

            elif (
                criterion
                == "unit_value_reference_observations"
            ):
                threshold_label = (
                    f"≥ "
                    f"{int(row.uv_support_threshold):,}"
                )

                evidence_label = (
                    "Product-year UV support"
                )

            elif (
                criterion
                == "bilateral_max_unit_value_log_distance"
            ):
                threshold_label = (
                    f"≤ "
                    f"{float(row.uv_distance_threshold):g}"
                )

                evidence_label = (
                    "Worst-side reporter UV distance"
                )

            else:
                threshold_label = "—"
                evidence_label = criterion

            sensitivity_rows.append(
                {
                    "Evidence": evidence_label,
                    "Threshold": threshold_label,
                    "Rows": _format_integer(
                        row.n_rows
                    ),
                    "Share of direct bilateral observations": (
                        _format_percent(
                            row.share_of_parent
                        )
                    ),
                }
            )

        sensitivity_display = pd.DataFrame(
            sensitivity_rows
        )

        # ==============================================================
        # Joint threshold grid
        # ==============================================================

        threshold_grid = _scope_frame(
            diagnostics,
            "threshold_grid",
        )

        threshold_grid = threshold_grid[
            [
                "disagreement_threshold",
                "uv_support_threshold",
                "uv_distance_threshold",
                "n_rows",
                "share_of_parent",
            ]
        ].copy()

        threshold_grid = threshold_grid.sort_values(
            [
                "disagreement_threshold",
                "uv_support_threshold",
                "uv_distance_threshold",
            ]
        )

        threshold_grid_display = pd.DataFrame(
            {
                "Maximum Mirror quantity disagreement": (
                    threshold_grid[
                        "disagreement_threshold"
                    ].map(
                        lambda value: (
                            f"≤ {100.0 * float(value):.0f}%"
                        )
                    )
                ),
                "Minimum Product-year UV support": (
                    threshold_grid[
                        "uv_support_threshold"
                    ].map(
                        lambda value: (
                            f"{int(value):,}"
                        )
                    )
                ),
                "Maximum Worst-side reporter UV distance": (
                    threshold_grid[
                        "uv_distance_threshold"
                    ].map(
                        lambda value: (
                            f"{float(value):g}"
                        )
                    )
                ),
                "Candidate rows": (
                    threshold_grid[
                        "n_rows"
                    ].map(
                        _format_integer
                    )
                ),
                "Share of direct bilateral observations": (
                    threshold_grid[
                        "share_of_parent"
                    ].map(
                        _format_percent
                    )
                ),
            }
        )

        evidence_matrix = _scope_frame(
            diagnostics,
            "evidence_matrix",
        )

        evidence_matrix = evidence_matrix[
            [
                "disagreement_band",
                "uv_distance_band",
                "n_rows",
                "share_of_parent",
            ]
        ].copy()

        evidence_matrix = evidence_matrix.sort_values(
            [
                "disagreement_band",
                "uv_distance_band",
            ]
        )

        evidence_matrix_display = pd.DataFrame(
            {
                "Mirror agreement": (
                    evidence_matrix[
                        "disagreement_band"
                    ]
                ),
                "Bilateral UV plausibility": (
                    evidence_matrix[
                        "uv_distance_band"
                    ]
                ),
                "Rows": (
                    evidence_matrix[
                        "n_rows"
                    ].map(
                        _format_integer
                    )
                ),
                "Share of direct bilateral observations": (
                    evidence_matrix[
                        "share_of_parent"
                    ].map(
                        _format_percent
                    )
                ),
            }
        )

        # ==============================================================
        # Year coverage
        # ==============================================================

        year_coverage = _scope_frame(
            diagnostics,
            "year_coverage",
        ).sort_values(
            "year"
        )

        for column in (
            "median_quantity_difference_vs_import",
            "median_absolute_quantity_difference_vs_import",
            "exporter_higher_share",
        ):
            if column not in year_coverage.columns:
                year_coverage[column] = None

        year_display = pd.DataFrame(
            {
                "Year": (
                    year_coverage[
                        "year"
                    ].astype(
                        "Int64"
                    )
                ),
                "All rows": (
                    year_coverage[
                        "total_rows"
                    ].map(
                        _format_integer
                    )
                ),
                "Direct bilateral quantity rows": (
                    year_coverage[
                        "n_rows"
                    ].map(
                        _format_integer
                    )
                ),
                "Share": (
                    year_coverage[
                        "share_of_all"
                    ].map(
                        _format_percent
                    )
                ),
                "Median Mirror quantity disagreement": (
                    year_coverage[
                        "median_quantity_relative_difference"
                    ].map(
                        _format_percent
                    )
                ),
                "Median Worst-side reporter UV distance": (
                    year_coverage[
                        "median_worst_side_uv_log_distance"
                    ].map(
                        lambda value: (
                            _format_number(
                                value,
                                digits=3,
                            )
                        )
                    )
                ),
                "Median Product-year UV support": (
                    year_coverage[
                        "median_uv_reference_observations"
                    ].map(
                        _format_integer
                    )
                ),

                "Median exporter difference vs importer": (
                    year_coverage[
                        "median_quantity_difference_vs_import"
                    ].map(
                        _format_percent
                    )
                ),

                "Median absolute difference vs importer": (
                    year_coverage[
                        "median_absolute_quantity_difference_vs_import"
                    ].map(
                        _format_percent
                    )
                ),

                "Exporter quantity higher": (
                    year_coverage[
                        "exporter_higher_share"
                    ].map(
                        _format_percent
                    )
                ),
            }
        )

        # ==============================================================
        # Product coverage summary
        # ==============================================================

        product_coverage = _scope_frame(
            diagnostics,
            "product_coverage",
        )

        n_products = int(
            len(
                product_coverage
            )
        )

        n_products_with_direct_evidence = int(
            (
                product_coverage["n_rows"]
                > 0
            ).sum()
        )

        median_product_direct_share = (
            float(
                product_coverage[
                    "share_of_all"
                ].median()
            )
            if not product_coverage.empty
            else None
        )

        product_rows = product_coverage[
            [
                "target_cmd",
                "total_rows",
                "n_rows",
                "share_of_all",
                "median_quantity_relative_difference",
                "median_worst_side_uv_log_distance",
            ]
        ].copy()

        product_rows = product_rows.sort_values(
            [
                "n_rows",
                "target_cmd",
            ],
            ascending=[
                False,
                True,
            ],
        )

        product_display = product_rows.head(
            20
        ).copy()

        product_display["total_rows"] = (
            product_display[
                "total_rows"
            ].map(
                _format_integer
            )
        )

        product_display["n_rows"] = (
            product_display[
                "n_rows"
            ].map(
                _format_integer
            )
        )

        product_display["share_of_all"] = (
            product_display[
                "share_of_all"
            ].map(
                _format_percent
            )
        )

        product_display[
            "median_quantity_relative_difference"
        ] = (
            product_display[
                "median_quantity_relative_difference"
            ].map(
                _format_percent
            )
        )

        product_display[
            "median_worst_side_uv_log_distance"
        ] = (
            product_display[
                "median_worst_side_uv_log_distance"
            ].map(
                lambda value: (
                    _format_number(
                        value,
                        digits=3,
                    )
                )
            )
        )

        product_display = product_display.rename(
            columns={
                "target_cmd": "HS code",
                "total_rows": "All rows",
                "n_rows": (
                    "Direct bilateral quantity rows"
                ),
                "share_of_all": "Direct share",
                "median_quantity_relative_difference": (
                    "Median Mirror quantity disagreement"
                ),
                "median_worst_side_uv_log_distance": (
                    "Median Worst-side reporter UV distance"
                ),
            }
        )

        # ==============================================================
        # Distribution diagnostics
        # ==============================================================

        distribution = _scope_frame(
            diagnostics,
            "distribution",
        )

        distribution_labels = {
            "quantity_relative_difference": (
                "Mirror quantity disagreement"
            ),
            "quantity_difference_vs_import": (
                "Exporter quantity relative to importer"
            ),
            "quantity_absolute_difference_vs_import": (
                "Absolute exporter quantity difference relative to importer"
            ),
            "bilateral_max_unit_value_log_distance": (
                "Worst-side reporter UV distance"
            ),
            "unit_value_reference_observations": (
                "Product-year UV support"
            ),
        }

        distribution_display = (
            distribution[
                [
                    "variable",
                    "quantile",
                    "value",
                ]
            ]
            .copy()
        )

        distribution_display[
            "variable"
        ] = (
            distribution_display[
                "variable"
            ].map(
                distribution_labels
            )
        )

        distribution_display[
            "quantile"
        ] = (
            distribution_display[
                "quantile"
            ].map(
                lambda value: (
                    f"P{int(round(100 * float(value)))}"
                )
            )
        )

        percentage_distribution_variables = {
            "Mirror quantity disagreement",
            "Exporter quantity relative to importer",
            "Absolute exporter quantity difference relative to importer",
        }

        distribution_display[
            "value"
        ] = (
            distribution_display.apply(
                lambda row: (
                    _format_percent(
                        row["value"]
                    )
                    if row["variable"]
                    in percentage_distribution_variables
                    else _format_number(
                        row["value"],
                        digits=3,
                    )
                ),
                axis=1,
            )
        )

        distribution_display = (
            distribution_display
            .pivot(
                index="variable",
                columns="quantile",
                values="value",
            )
            .reset_index()
            .rename(
                columns={
                    "variable": "Evidence"
                }
            )
        )

        percentile_order = [
            "P1",
            "P5",
            "P10",
            "P25",
            "P50",
            "P75",
            "P90",
            "P95",
            "P99",
        ]

        distribution_display = distribution_display[
            [
                "Evidence",
                *percentile_order,
            ]
        ]

        mirror_disagreement_p75 = "NA"
        mirror_p75_series = distribution_display.loc[
            distribution_display["Evidence"].eq(
                "Mirror quantity disagreement"
            ),
            "P75",
        ]
        if not mirror_p75_series.empty:
            mirror_disagreement_p75 = str(
                mirror_p75_series.iloc[0]
            )

        # ==============================================================
        # Current population totals
        # ==============================================================

        funnel_lookup = {
            row.criterion: int(
                row.n_rows
            )
            for row in funnel.itertuples(
                index=False
            )
        }


        n_both_quantities = funnel_lookup.get(
            "both_quantities_available",
            0,
        )

        n_both_observed = funnel_lookup.get(
            "both_quantities_non_estimated",
            0,
        )

        n_complete_training_evidence = (
            funnel_lookup.get(
                "complete_training_evidence",
                0,
            )
        )

        # ==============================================================
        # Fixed quantity-candidate comparison
        # ==============================================================

        candidate_summary = _scope_frame(
            diagnostics,
            "quantity_candidate_summary",
        )

        candidate_by_disagreement = _scope_frame(
            diagnostics,
            "quantity_candidate_by_disagreement",
        )

        candidate_closest = _scope_frame(
            diagnostics,
            "quantity_candidate_closest_reference",
        )

        candidate_by_year = _scope_frame(
            diagnostics,
            "quantity_candidate_by_year",
        )

        candidate_by_product = _scope_frame(
            diagnostics,
            "quantity_candidate_by_product",
        )

        candidate_spread = _scope_frame(
            diagnostics,
            "quantity_candidate_spread",
        )

        candidate_conditioned_summary = _scope_frame(
            diagnostics,
            "quantity_candidate_conditioned_summary",
        )

        candidate_conditioned_by_disagreement = _scope_frame(
            diagnostics,
            "quantity_candidate_conditioned_by_disagreement",
        )

        candidate_closest_conditioned = _scope_frame(
            diagnostics,
            "quantity_candidate_closest_conditioned",
        )

        candidate_scopes = {
            "quantity_candidate_summary": candidate_summary,
            "quantity_candidate_by_disagreement": candidate_by_disagreement,
            "quantity_candidate_closest_reference": candidate_closest,
            "quantity_candidate_by_year": candidate_by_year,
            "quantity_candidate_by_product": candidate_by_product,
            "quantity_candidate_spread": candidate_spread,
            "quantity_candidate_conditioned_summary": candidate_conditioned_summary,
            "quantity_candidate_conditioned_by_disagreement": candidate_conditioned_by_disagreement,
            "quantity_candidate_closest_conditioned": candidate_closest_conditioned,
        }

        missing_candidate_scopes = [
            scope
            for scope, frame in candidate_scopes.items()
            if frame.empty
        ]

        if missing_candidate_scopes:
            raise RuntimeError(
                "Validation quantity-candidate diagnostics are incomplete: "
                + ", ".join(missing_candidate_scopes)
            )

        # ==============================================================
        # Quantity-conditioned candidate comparison
        # ==============================================================

        candidate_order = [
            "simple_mean",
            "value_weighted",
            "quantity_reliability_weighted",
        ]

        candidate_labels = {
            "simple_mean": "Simple mean",
            "value_weighted": "Value-weighted",
            "quantity_reliability_weighted": "Quantity-reliability-weighted",
            "tie": "Tie",
        }

        conditioned_summary_display = (
            candidate_conditioned_summary
            .pivot_table(
                index="candidate",
                columns="variable",
                values="value",
                aggfunc="first",
            )
            .reset_index()
            .rename(
                columns={
                    "candidate": "Candidate",
                    "median_conditioned_uv_abs_residual": "Median residual",
                    "p90_conditioned_uv_abs_residual": "P90 residual",
                    "p95_conditioned_uv_abs_residual": "P95 residual",
                }
            )
        )
        conditioned_summary_display["Candidate"] = (
            conditioned_summary_display["Candidate"]
            .map(candidate_labels)
            .fillna(conditioned_summary_display["Candidate"])
        )

        conditioned_disagreement_display = (
            candidate_conditioned_by_disagreement
            .pivot_table(
                index=["disagreement_band", "candidate"],
                columns="variable",
                values="value",
                aggfunc="first",
            )
            .reset_index()
            .rename(
                columns={
                    "disagreement_band": "Mirror disagreement",
                    "candidate": "Candidate",
                    "median_conditioned_uv_abs_residual": "Median residual",
                    "p90_conditioned_uv_abs_residual": "P90 residual",
                    "p95_conditioned_uv_abs_residual": "P95 residual",
                }
            )
        )

        conditioned_disagreement_display["_band_order"] = (
            conditioned_disagreement_display["Mirror disagreement"]
            .map(lambda value: _disagreement_band_order(value)[0])
        )

        conditioned_disagreement_display["_candidate_order"] = (
            conditioned_disagreement_display["Candidate"].map(
                {
                    value: index
                    for index, value in enumerate(candidate_order)
                }
            )
        )

        conditioned_disagreement_display = (
            conditioned_disagreement_display
            .sort_values(
                [
                    "_band_order",
                    "_candidate_order",
                ]
            )
            .drop(
                columns=[
                    "_band_order",
                    "_candidate_order",
                ]
            )
        )


        conditioned_disagreement_display["Candidate"] = (
            conditioned_disagreement_display["Candidate"]
            .map(candidate_labels)
            .fillna(conditioned_disagreement_display["Candidate"])
        )

        conditioned_closest_display = candidate_closest_conditioned[[
            "candidate", "n_rows", "value"
        ]].copy()
        conditioned_closest_display["candidate"] = (
            conditioned_closest_display["candidate"]
            .map(candidate_labels)
            .fillna(conditioned_closest_display["candidate"])
        )
        conditioned_closest_display["value"] = conditioned_closest_display[
            "value"
        ].apply(_format_percent)
        conditioned_closest_display = conditioned_closest_display.rename(
            columns={
                "candidate": "Closest candidate",
                "n_rows": "Rows",
                "value": "Share",
            }
        )
        candidate_summary_wide = (
            candidate_summary
            .pivot_table(
                index="candidate",
                columns="variable",
                values="value",
                aggfunc="first",
            )
            .reindex(candidate_order)
        )

        candidate_counts = (
            candidate_summary
            .groupby("candidate", as_index=True)
            .agg(
                n_rows=("n_rows", "max"),
                n_candidate_uv_rows=("n_candidate_uv_rows", "max"),
            )
            .reindex(candidate_order)
        )

        candidate_summary_display = (
            candidate_counts
            .join(candidate_summary_wide)
            .reset_index()
            .rename(
                columns={
                    "candidate": "Candidate",
                    "n_rows": "Available rows",
                    "n_candidate_uv_rows": "Rows with UV distance",
                    "median_uv_log_distance": "Median UV distance",
                    "p90_uv_log_distance": "P90 UV distance",
                    "p95_uv_log_distance": "P95 UV distance",
                    "p99_uv_log_distance": "P99 UV distance",
                    "median_export_log_gap": "Median log-gap to exporter",
                    "median_import_log_gap": "Median log-gap to importer",
                }
            )
        )

        candidate_summary_display["Candidate"] = (
            candidate_summary_display["Candidate"].map(_candidate_label)
        )

        for column in (
            "Available rows",
            "Rows with UV distance",
        ):
            candidate_summary_display[column] = (
                candidate_summary_display[column].map(_format_integer)
            )

        for column in (
            "Median UV distance",
            "P90 UV distance",
            "P95 UV distance",
            "P99 UV distance",
            "Median log-gap to exporter",
            "Median log-gap to importer",
        ):
            candidate_summary_display[column] = (
                candidate_summary_display[column]
                .map(lambda value: _format_number(value, digits=4))
            )

        disagreement_display = candidate_by_disagreement[
            candidate_by_disagreement["variable"].isin(
                [
                    "median_uv_log_distance",
                    "p90_uv_log_distance",
                    "p95_uv_log_distance",
                ]
            )
        ].copy()

        disagreement_display = (
            disagreement_display
            .pivot_table(
                index=["disagreement_band", "candidate"],
                columns="variable",
                values="value",
                aggfunc="first",
            )
            .reset_index()
            .rename(
                columns={
                    "disagreement_band": "Mirror disagreement",
                    "candidate": "Candidate",
                    "median_uv_log_distance": "Median UV distance",
                    "p90_uv_log_distance": "P90 UV distance",
                    "p95_uv_log_distance": "P95 UV distance",
                }
            )
        )

        disagreement_display["_band_order"] = (
            disagreement_display["Mirror disagreement"]
            .map(lambda value: _disagreement_band_order(value)[0])
        )
        disagreement_display["_candidate_order"] = (
            disagreement_display["Candidate"].map(
                {value: index for index, value in enumerate(candidate_order)}
            )
        )
        disagreement_display = (
            disagreement_display
            .sort_values(["_band_order", "_candidate_order"])
            .drop(columns=["_band_order", "_candidate_order"])
        )
        disagreement_display["Candidate"] = (
            disagreement_display["Candidate"].map(_candidate_label)
        )

        for column in (
            "Median UV distance",
            "P90 UV distance",
            "P95 UV distance",
        ):
            disagreement_display[column] = (
                disagreement_display[column]
                .map(lambda value: _format_number(value, digits=4))
            )

        closest_order = [
            "simple_mean",
            "value_weighted",
            "quantity_reliability_weighted",
            "tie",
        ]

        closest_display = candidate_closest[
            ["candidate", "n_rows", "value"]
        ].copy()
        closest_display["_order"] = closest_display["candidate"].map(
            {value: index for index, value in enumerate(closest_order)}
        )
        closest_display = (
            closest_display
            .sort_values("_order")
            .drop(columns="_order")
            .rename(
                columns={
                    "candidate": "Closest candidate",
                    "n_rows": "Rows",
                    "value": "Share",
                }
            )
        )
        closest_display["Closest candidate"] = (
            closest_display["Closest candidate"].map(_candidate_label)
        )
        closest_display["Rows"] = closest_display["Rows"].map(
            _format_integer
        )
        closest_display["Share"] = closest_display["Share"].map(
            _format_percent
        )

        spread_display = (
            candidate_spread
            .pivot_table(
                index="disagreement_band",
                columns="variable",
                values="value",
                aggfunc="first",
            )
            .reset_index()
            .rename(
                columns={
                    "disagreement_band": "Mirror disagreement",
                    "median_relative_candidate_spread": "Median candidate spread",
                    "p95_relative_candidate_spread": "P95 candidate spread",
                }
            )
        )
        spread_display["_order"] = spread_display[
            "Mirror disagreement"
        ].map(lambda value: _disagreement_band_order(value)[0])
        spread_display = (
            spread_display.sort_values("_order").drop(columns="_order")
        )
        for column in (
            "Median candidate spread",
            "P95 candidate spread",
        ):
            spread_display[column] = spread_display[column].map(
                _format_percent
            )

        candidate_year_display = candidate_by_year[
            candidate_by_year["variable"].eq("median_uv_log_distance")
        ].copy()
        candidate_year_display = (
            candidate_year_display
            .pivot_table(
                index="year",
                columns="candidate",
                values="value",
                aggfunc="first",
            )
            .reset_index()
            .rename(
                columns={
                    "year": "Year",
                    "simple_mean": "Simple mean",
                    "value_weighted": "Value-weighted",
                    "quantity_reliability_weighted": (
                        "Quantity-reliability-weighted"
                    ),
                }
            )
        )
        candidate_year_display["Year"] = (
            pd.to_numeric(
                candidate_year_display["Year"],
                errors="coerce",
            )
            .astype("Int64")
        )
        for column in (
            "Simple mean",
            "Value-weighted",
            "Quantity-reliability-weighted",
        ):
            if column in candidate_year_display.columns:
                candidate_year_display[column] = (
                    candidate_year_display[column]
                    .map(lambda value: _format_number(value, digits=4))
                )

        candidate_product_display = candidate_by_product[
            candidate_by_product["variable"].eq("median_uv_log_distance")
        ][
            ["target_cmd", "candidate", "n_rows", "value"]
        ].copy()
        candidate_product_display = candidate_product_display.rename(
            columns={
                "target_cmd": "HS product",
                "candidate": "Candidate",
                "n_rows": "Rows",
                "value": "Median UV distance",
            }
        )
        candidate_product_display["Candidate"] = (
            candidate_product_display["Candidate"].map(_candidate_label)
        )
        candidate_product_display["Rows"] = (
            candidate_product_display["Rows"].map(_format_integer)
        )
        candidate_product_display["Median UV distance"] = (
            candidate_product_display["Median UV distance"]
            .map(lambda value: _format_number(value, digits=4))
        )

        # ==============================================================
        # Figures
        # ==============================================================

        figure_paths = {

            "training_uv_distance_distribution": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_training_uv_distance_distribution.png"
                )
            ),
            "training_reference_support_distribution": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_training_reference_support_distribution.png"
                )
            ),

            "training_uv_dispersion_distribution": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_training_uv_dispersion_distribution.png"
                )
            ),

            "training_normalized_uv_distance_distribution": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_training_normalized_uv_distance_distribution.png"
                )
            ),
            "training_population_funnel": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_training_population_funnel.png"
                )
            ),
            "training_evidence_by_year": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_training_evidence_by_year.png"
                )
            ),
            "training_evidence_distributions": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_training_evidence_distributions.png"
                )
            ),
            "export_import_quantity_ratio_distribution": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_export_import_quantity_ratio_distribution.png"
                )
            ),

            "quantity_uv_slope_distribution": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_quantity_uv_slope_distribution.png"
                )
            ),

            "quantity_uv_r_squared_distribution": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_quantity_uv_r_squared_distribution.png"
                )
            ),
            "quantity_candidate_uv_summary": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_quantity_candidate_uv_summary.png"
                )
            ),
            "quantity_candidate_by_disagreement": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_quantity_candidate_by_disagreement.png"
                )
            ),
            "quantity_candidate_by_disagreement_factor": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_quantity_candidate_by_disagreement_factor.png"
                )
            ),
            "quantity_candidate_closest_reference": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_quantity_candidate_closest_reference.png"
                )
            ),
            "quantity_candidate_by_year": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_quantity_candidate_by_year.png"
                )
            ),
            "quantity_candidate_product_distribution": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_quantity_candidate_product_distribution.png"
                )
            ),
            "quantity_candidate_spread": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_quantity_candidate_spread.png"
                )
            ),
            "quantity_candidate_conditioned_summary": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_quantity_candidate_conditioned_summary.png"
                )
            ),
            "quantity_candidate_conditioned_by_disagreement": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_quantity_candidate_conditioned_by_disagreement.png"
                )
            ),
            "quantity_candidate_closest_conditioned": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_quantity_candidate_closest_conditioned.png"
                )
            ),
        }

        _plot_training_population_funnel(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_population_funnel"
            ],
        )

        _plot_training_evidence_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_evidence_by_year"
            ],
        )

        _plot_training_evidence_distributions(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_evidence_distributions"
            ],
        )
        _plot_training_uv_distance_distribution(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_uv_distance_distribution"
            ],
        )
        _plot_training_reference_support_distribution(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_reference_support_distribution"
            ],
        )

        _plot_training_uv_dispersion_distribution(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_uv_dispersion_distribution"
            ],
        )

        _plot_training_normalized_uv_distance_distribution(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_normalized_uv_distance_distribution"
            ],
        )
        _plot_export_import_quantity_ratio_distribution(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "export_import_quantity_ratio_distribution"
            ],
        )
        
        _plot_quantity_uv_slope_distribution(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "quantity_uv_slope_distribution"
            ],
        )

        _plot_quantity_uv_r_squared_distribution(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "quantity_uv_r_squared_distribution"
            ],
        )

        _plot_quantity_candidate_uv_summary(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "quantity_candidate_uv_summary"
            ],
        )

        _plot_quantity_candidate_by_disagreement(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "quantity_candidate_by_disagreement"
            ],
            scale="log",
        )

        _plot_quantity_candidate_by_disagreement(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "quantity_candidate_by_disagreement_factor"
            ],
            scale="factor",
        )

        _plot_quantity_candidate_closest_reference(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "quantity_candidate_closest_reference"
            ],
        )

        _plot_quantity_candidate_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "quantity_candidate_by_year"
            ],
        )

        _plot_quantity_candidate_product_distribution(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "quantity_candidate_product_distribution"
            ],
        )

        _plot_quantity_candidate_spread(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "quantity_candidate_spread"
            ],
        )

        _plot_quantity_candidate_conditioned_summary(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "quantity_candidate_conditioned_summary"
            ],
        )

        _plot_quantity_candidate_conditioned_by_disagreement(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "quantity_candidate_conditioned_by_disagreement"
            ],
        )

        _plot_quantity_candidate_closest_conditioned(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "quantity_candidate_closest_conditioned"
            ],
        )
    

        # ==============================================================
        # Current methodological status
        # ==============================================================

        training_selection_enabled = bool(
            cfg.diagnostics.selection.enabled
        )

        if training_selection_enabled:
            training_status = (
                "A final candidate-training selection is configured."
            )
        else:
            training_status = (
                "No final training population has "
                "been selected yet. The current diagnostics are intended "
                "to support that decision before model fitting."
            )


        # ==============================================================
        # Mirror agreement × bilateral UV plausibility
        # ==============================================================

        evidence_matrix = _scope_frame(
            diagnostics,
            "evidence_matrix",
        )

        if evidence_matrix.empty:
            raise RuntimeError(
                "Validation evidence-matrix diagnostics are unavailable."
            )

        matrix_display = (
            evidence_matrix
            .pivot_table(
                index="disagreement_band",
                columns="uv_distance_band",
                values="n_rows",
                aggfunc="sum",
                fill_value=0,
            )
        )

        matrix_display = matrix_display.map(
            _format_integer
        )

        matrix_display.index.name = (
            "Mirror quantity disagreement"
        )


        # ==============================================================
        # Side-specific unit-value plausibility
        # ==============================================================

        side_plausibility = _scope_frame(
            diagnostics,
            "side_plausibility",
        )

        if side_plausibility.empty:
            raise RuntimeError(
                "Validation side-plausibility diagnostics are unavailable."
            )

        side_plausibility_display = (
            side_plausibility
            .pivot_table(
                index="disagreement_band",
                columns="variable",
                values="value",
                aggfunc="first",
            )
            .reset_index()
        )

        side_plausibility_display = (
            side_plausibility_display.rename(
                columns={
                    "disagreement_band": (
                        "Mirror quantity disagreement"
                    ),

                    "median_export_uv_distance": (
                        "Median Exporter UV distance"
                    ),

                    "median_import_uv_distance": (
                        "Median Importer UV distance"
                    ),

                    "median_bilateral_max_uv_distance": (
                        "Median Worst-side reporter UV distance"
                    ),

                    "both_within_025_share": (
                        "Both within 0.25"
                    ),

                    "both_within_050_share": (
                        "Both within 0.50"
                    ),

                    "both_within_100_share": (
                        "Both within 1.00"
                    ),
                }
            )
        )

        for column in (
            "Both within 0.25",
            "Both within 0.50",
            "Both within 1.00",
        ):
            side_plausibility_display[
                column
            ] = side_plausibility_display[
                column
            ].map(
                _format_percent
            )

        # ==============================================================
        # Importer-referenced quantity comparison
        # ==============================================================

        importer_reference = _scope_frame(
            diagnostics,
            "importer_reference",
        )

        if importer_reference.empty:
            raise RuntimeError(
                "Importer-reference quantity diagnostics are unavailable."
            )

        importer_reference_lookup = {
            str(row.variable): (
                float(row.value)
                if pd.notna(row.value)
                else None
            )
            for row in importer_reference.itertuples(
                index=False
            )
        }

        n_importer_reference_rows = int(
            importer_reference["n_rows"].max()
        )

        mean_difference_vs_import = (
            importer_reference_lookup.get(
                "mean_quantity_difference_vs_import"
            )
        )

        median_difference_vs_import = (
            importer_reference_lookup.get(
                "median_quantity_difference_vs_import"
            )
        )

        median_absolute_difference_vs_import = (
            importer_reference_lookup.get(
                "median_absolute_quantity_difference_vs_import"
            )
        )

        exporter_higher_share = (
            importer_reference_lookup.get(
                "exporter_quantity_higher_share"
            )
        )

        importer_higher_share = (
            importer_reference_lookup.get(
                "importer_quantity_higher_share"
            )
        )

        exact_agreement_share = (
            importer_reference_lookup.get(
                "exact_quantity_agreement_share"
            )
        )

        importer_reference_display = pd.DataFrame(
            [
                {
                    "Measure": (
                        "Comparable directly observed bilateral rows"
                    ),
                    "Result": _format_integer(
                        n_importer_reference_rows
                    ),
                },
                {
                    "Measure": (
                        "Mean exporter difference relative to importer"
                    ),
                    "Result": _format_percent(
                        mean_difference_vs_import
                    ),
                },
                {
                    "Measure": (
                        "Median exporter difference relative to importer"
                    ),
                    "Result": _format_percent(
                        median_difference_vs_import
                    ),
                },
                {
                    "Measure": (
                        "Median absolute difference relative to importer"
                    ),
                    "Result": _format_percent(
                        median_absolute_difference_vs_import
                    ),
                },
                {
                    "Measure": (
                        "Exporter quantity larger than importer"
                    ),
                    "Result": _format_percent(
                        exporter_higher_share
                    ),
                },
                {
                    "Measure": (
                        "Importer quantity larger than exporter"
                    ),
                    "Result": _format_percent(
                        importer_higher_share
                    ),
                },
                {
                    "Measure": (
                        "Exact exporter-importer agreement"
                    ),
                    "Result": _format_percent(
                        exact_agreement_share
                    ),
                },
            ]
        )

        # ==============================================================
        # Quantity–UV relationship
        # ==============================================================

        quantity_uv_relationship = _scope_frame(
            diagnostics,
            "quantity_uv_relationship",
        )

        quantity_uv_summary = _scope_frame(
            diagnostics,
            "quantity_uv_summary",
        )

        if quantity_uv_relationship.empty:
            raise RuntimeError(
                "Quantity–UV relationship diagnostics are unavailable."
            )

        if quantity_uv_summary.empty:
            raise RuntimeError(
                "Quantity–UV summary diagnostics are unavailable."
            )
        
        quantity_uv_summary_lookup = {
            str(row.variable): (
                float(row.value)
                if pd.notna(row.value)
                else None
            )
            for row in quantity_uv_summary.itertuples(
                index=False
            )
        }


        n_quantity_uv_groups = int(
            quantity_uv_summary_lookup[
                "n_product_year_groups"
            ]
        )

        median_quantity_uv_slope = (
            quantity_uv_summary_lookup[
                "median_log_uv_quantity_slope"
            ]
        )

        p25_quantity_uv_slope = (
            quantity_uv_summary_lookup[
                "p25_log_uv_quantity_slope"
            ]
        )

        p75_quantity_uv_slope = (
            quantity_uv_summary_lookup[
                "p75_log_uv_quantity_slope"
            ]
        )

        negative_quantity_uv_share = (
            quantity_uv_summary_lookup[
                "negative_slope_share"
            ]
        )

        median_quantity_uv_r2 = (
            quantity_uv_summary_lookup[
                "median_log_uv_quantity_r2"
            ]
        )

        quantity_uv_summary_display = pd.DataFrame(
            [
                {
                    "Measure": (
                        "HS × year groups with at least 30 observations"
                    ),
                    "Result": _format_integer(
                        n_quantity_uv_groups
                    ),
                },
                {
                    "Measure": (
                        "Median quantity–UV slope"
                    ),
                    "Result": _format_number(
                        median_quantity_uv_slope,
                        digits=3,
                    ),
                },
                {
                    "Measure": (
                        "25th percentile slope"
                    ),
                    "Result": _format_number(
                        p25_quantity_uv_slope,
                        digits=3,
                    ),
                },
                {
                    "Measure": (
                        "75th percentile slope"
                    ),
                    "Result": _format_number(
                        p75_quantity_uv_slope,
                        digits=3,
                    ),
                },
                {
                    "Measure": (
                        "Groups with negative slope"
                    ),
                    "Result": _format_percent(
                        negative_quantity_uv_share
                    ),
                },
                {
                    "Measure": (
                        "Median R²"
                    ),
                    "Result": _format_number(
                        median_quantity_uv_r2,
                        digits=3,
                    ),
                },
            ]
        )
        # ==============================================================
        # Report
        # ==============================================================

        lines = [
            (
                "# VACODA Validation Decision Report — "
                f"{cfg.scope.target_hs.upper()}"
            ),
            "",
            f"**Run ID:** `{run_id}`  ",
            (
                "**Pipeline scope:** "
                f"`{cfg.scope.min_year}–{cfg.scope.max_year}`  "
            ),
            (
                "**Structural audit:** "
                f"`{'PASS' if audit_passed else 'FAIL'}`  "
            ),
            (
                "**Model fitted:** `False`  "
            ),
            (
                "**Training population selected:** "
                f"`{training_selection_enabled}`"
            ),
            "",
            (
                "This report documents the current Validation state and "
                "the evidence used to develop the quantity-validation "
                "model. At this stage no machine-learning model has been "
                "fitted and no quantity observations have been corrected "
                "or imputed."
            ),
            "",
            "__TOC__",
            "",           
            "## 1. Validation objective",
            "",
            (
                "The final VACODA Validation workflow assesses whether "
                "the available bilateral quantity evidence is credible, "
                "identifies observations that require further attention, "
                "compares fixed reconciliation candidates, and ultimately determines whether missing or implausible "
                "quantities can be reconstructed with sufficient evidence."
            ),
            "",
            "",
            "## 2. Structural integrity audit",
            "",
            (
                f"- The prepared Validation dataset contains "
                f"`{_format_integer(n_rows)}` observations across "
                f"`{cfg.scope.min_year}–{cfg.scope.max_year}`."
            ),
            "",
            (
                "The structural audit verifies the retained reporter quantities "
                "and all three fixed bilateral quantity candidates before "
                "scientific candidate evaluation."
            ),
            "",
            _markdown_table(
                structural_summary
            ),
            "",
            (
                "The structural audit is deliberately limited to technical "
                "integrity: canonical grain, required fields, year and HS "
                "scope and numeric validity of the retained quantity evidence. "
                "Economic plausibility is not decided here."
            ),
            "",
            "## 3. Quantity diagnostics",
            "",
            (
                "Quantity diagnostics examine the available quantity "
                "information before model development. They provide evidence "
                "on agreement between reporting partners, consistency with "
                "product-year unit values, and the amount of usable evidence "
                "available for candidate training selection."
            ),
            "",
            (
                "The following terminology is used throughout this section:"
            ),
            "",
            (
                "- **Mirror quantity disagreement** —  for each HS product `p` and year `t`, the symmetric relative difference "
                "between exporter- and importer-reported quantities. A value of `0%` means exact agreement. Larger values indicate greater disagreement."
            ),
            "",
            "$$",
            r"d^{Q}_{p,t}"
            r" = "
            r"\frac{\left|Q^{E}_{p,t} - Q^{I}_{p,t}\right|}"
            r"{\max\left(Q^{E}_{p,t}, Q^{I}_{p,t}\right)}",
            "$$",
            (
                "- **Unit value (UV)** — for each HS product `p` and year `t`, trade value per kilogram:"
            ),
            "",
            "$$",
            r"UV_{i,p,t}"
            r" = "
            r"\frac{V_{i,p,t}}{Q_{i,p,t}}",
            "$$",
            "",
            (
                "- **Product-year UV reference** — for each HS product `p` and year "
                "`t`, the reference is the exponential of the median log unit value "
                "across all valid exporter- and importer-implied unit values:"
            ),
            "",
            "$$",
            r"UV^{\mathrm{ref}}_{p,t}"
            r" = "
            r"\exp\left("
            r"\operatorname{median}_{i}"
            r"\left[\log\left(UV_{i,p,t}\right)\right]"
            r"\right)",
            "$$",
            "",
            (
                "- **Product-year UV support** — for each HS product `p` and year `t`,the number of observations "
                "supporting the product-year UV reference, where "
                r"$\mathcal{I}_{p,t}$ "
                "is the set of valid observations used to construct the product-year UV reference."
            ),
            "",
            "$$",
            r"N^{\mathrm{ref}}_{p,t}"
            r" = "
            r"\left|\mathcal{I}_{p,t}\right|",
            "$$",
            "",
            (
                "- **UV distance** — for each HS product `p` and year `t`, the absolute log-distance between an observation's "
                "implied unit value and the product-year UV reference. A value close to `0` means that the implied unit value is close "
                "to the reference. Larger values indicate greater separation."
            ),
            "",
            "$$",
            r"D^{UV}_{i,p,t}"
            r" = "
            r"\left|"
            r"\log\left(UV_{i,p,t}\right)"
            r" - "
            r"\log\left(UV^{\mathrm{ref}}_{p,t}\right)"
            r"\right|",
            "$$",
            "",
            (
                "- **Exporter UV distance** and **Importer UV distance** — for each HS product `p` and year `t`, the "
                "corresponding UV distances calculated separately using the exporter- "
                "and importer-reported quantities:"
            ),
            "",
            "$$",
            r"D^{UV,E}_{p,t}"
            r" = "
            r"\left|"
            r"\log\left(UV^{E}_{p,t}\right)"
            r" - "
            r"\log\left(UV^{\mathrm{ref}}_{p,t}\right)"
            r"\right|",
            "$$",
            "",
            "$$",
            r"D^{UV,I}_{p,t}"
            r" = "
            r"\left|"
            r"\log\left(UV^{I}_{p,t}\right)"
            r" - "
            r"\log\left(UV^{\mathrm{ref}}_{p,t}\right)"
            r"\right|",
            "$$",
            "",
            (
                "- **Worst-side reporter UV distance** — for each HS product `p` and year `t`, the larger of the exporter "
                "and importer UV distances. A small value therefore requires both "
                "reporting sides to be close to the product-year UV reference:"
            ),
            "",
            "$$",
            r"D^{UV,\mathrm{worst}}_{p,t}"
            r" = "
            r"\max\left("
            r"D^{UV,E}_{p,t}, "
            r"D^{UV,I}_{p,t}"
            r"\right)",
            "$$",
            "",
            "## 3. Quantity diagnostics",
            "",
            (
                "Quantity diagnostics examine the available quantity "
                "information before model development. They provide evidence "
                "on agreement between reporting partners, consistency with "
                "product-year unit values, and the amount of usable evidence "
                "available for candidate training selection."
            ),
            "",
            (
                "The following terminology is used throughout this section:"
            ),
            "",
            (
                "- **Mirror quantity disagreement** —  `|exporter quantity - importer quantity| / max(exporter quantity, importer quantity)`<br>"
                "&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; symmetric relative difference between exporter- and importer-reported quantities:<br>"
            ),
            (
                "&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; A value of `0%` means exact agreement. Larger values indicate "
                "greater disagreement. Neither reporter is treated as the "
                "reference."
            ),
            "",
            (
                "- **Unit value (UV)** — `trade value / quantity`, trade value per kilogram:<br>"
            ),
            "",
            (
                "- **Product-year UV reference** — for each HS product `p` and year "
                "`t`, the reference is the exponential of the median log unit value "
                "across all valid exporter- and importer-implied unit values:"
            ),
            "",
            "$$",
            r"UV^{\mathrm{ref}}_{p,t}"
            r" = "
            r"\exp\left("
            r"\operatorname{median}_{i}"
            r"\left[\log\left(UV_{i,p,t}\right)\right]"
            r"\right)",
            "$$",
            "",
            "",
            (
                "- **Product-year UV support** — the number of observations "
                "supporting the product-year UV reference."
            ),
            "",
            (
                "- **UV distance** — `|log(implied UV) − log(product-year UV reference)|`<br>"
                "&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; the distance between an observation's implied unit value and the product-year UV reference:<br>"
                "&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; A value close to `0` means that the implied unit value is "
                "close to the reference. Larger values indicate greater separation."
            ),
            "",
            (
                "- **Worst-side reporter UV distance** — the larger of the "
                "exporter and importer UV distances. A low value therefore "
                "requires both reporter quantities to imply plausible unit values."
            ),
            "",
            (
                "- **Exporter UV distance** and **Importer UV distance** — "
                "the corresponding UV distances calculated separately using "
                "the exporter- and importer-reported quantities."
            ),
            "",
            (
                "- **Worst-side reporter UV distance** — the larger of the exporter "
                "and importer UV distances. A small value therefore requires "
                "both reporting sides to be close to the product-year UV "
                "reference."
            ),
            "",
            "### 3.1. Quantity availability by reporting side",
            "",
            (
                "Validation first establishes which reporter-side quantity "
                "information is actually available before applying any distinction "
                "between directly reported and estimated quantities. The categories "
                "below are mutually exclusive and depend only on whether a usable "
                "positive finite quantity is available from the exporter, importer, "
                "both, or neither."
            ),
            "",
            _markdown_table(
                quantity_availability_display
            ),
            "",
            (
                f"A usable quantity is available from at least one reporting side "
                f"for `{_format_integer(n_quantity_any)}` observations "
                f"(`{_format_percent(quantity_any_share)}` of all Validation rows)."
            ),
            "",
            (
                f"Both exporter and importer quantities are simultaneously available "
                f"for `{_format_integer(n_quantity_both)}` observations "
                f"(`{_format_percent(quantity_bilateral_share)}` of all Validation "
                f"rows). These rows form the bilateral quantity population on which "
                f"reporter agreement can subsequently be evaluated."
            ),
            "",
            (
                "The difference between overall quantity coverage and bilateral "
                "quantity coverage therefore reflects rows for which quantity "
                "information is supplied by only one reporting side."
            ),
            "",
            "",
            "### 3.2. Quantity-evidence population",
            "",
            (
                "This diagnostic identifies the subset of Validation "
                "rows for which quantity information from both reporting "
                "sides is available and the diagnostics needed for later "
                "candidate selection can be calculated."
            ),
            "",
            (
                "The population is narrowed using the following terms and "
                "data requirements:"
            ),
            "",
            (
                "- **Bilateral quantity observations** — both an exporter "
                "quantity and an importer quantity are available for the "
                "same bilateral trade observation. Either quantity may still "
                "be marked as estimated."
            ),
            "",
            (
                "- **Direct bilateral quantity observations** — both the "
                "exporter and importer quantities are directly reported "
                "rather than marked as estimated."
            ),
            "",
            (
                "- **Candidate-selection diagnostics** — the Mirror quantity "
                "disagreement and UV diagnostics required for later candidate "
                "training selection."
            ),
            "",
            "",
            (
                "The table below shows how the Validation population narrows "
                "as these quantities and diagnostics become available."
            ),
            "",
            _markdown_table(
                funnel_display
            ),
            "",
            markdown_image(
                "Candidate quantity-training population",
                figure_paths[
                    "training_population_funnel"
                ],
                document_path=report_path,
            ),
            "",
            (
                f"Of all Validation rows, "
                f"`{_format_percent(n_both_quantities / n_rows)}` contain "
                "both exporter and importer quantities. Among these bilateral "
                f"quantity observations, "
                f"`{_format_percent(n_both_observed / n_both_quantities)}` "
                "have both quantities directly reported rather than estimated."
            ),
            "",
            (
                f"This leaves `{_format_integer(n_both_observed)}` direct "
                "bilateral quantity observations, representing "
                f"`{_format_percent(n_both_observed / n_rows)}` of all "
                "Validation rows."
            ),
            "",
            (
                "The final three rows of the table do not yet apply quality "
                "thresholds. They only indicate whether the required "
                "candidate-selection diagnostics can be calculated. In the "
                f"current dataset, all `{_format_integer(n_both_observed)}` "
                "direct bilateral quantity observations have Mirror quantity "
                "disagreement and UV distance available."
            ),
            "",
            "### 3.3. Diagnostics for directly observed bilateral quantities",
            "",
            (
                "- Unless stated otherwise, all diagnostics in this section are "
                f"evaluated on the `{_format_integer(n_both_observed)}` observations for which both exporter "
                "and importer quantities are available and directly reported, "
                "rather than estimated."
            ),
            "",
            markdown_styled_heading("3.3.1. Exporter quantity relative to importer"),
            "",
            (
                "For bilateral rows where both exporter and importer "
                "quantities are directly observed, Validation also measures "
                "the exporter-reported quantity relative to the importer-"
                "reported quantity."
            ),
            "",
            (
                "- The importer quantity is used as the reference denominator:"
            ),
            "",
            (
                "&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;`(exporter quantity − importer quantity) / importer quantity`"
            ),
            "",
            (
                "- This measure retains direction. A positive value means "
                "the exporter reports a larger quantity than the importer; "
                "a negative value means the exporter reports a smaller "
                "quantity. For example, if the importer reports `100 kg` "
                "and the exporter reports `120 kg`, the result is `+20%`. "
                "If the exporter reports `80 kg`, the result is `−20%`."
            ),
            "",
            (
                "- This importer-referenced measure complements the symmetric "
                "exporter–importer quantity-difference measure used elsewhere "
                "in the diagnostics. The symmetric measure describes the "
                "magnitude of disagreement without choosing either reporter "
                "as the reference. The importer-referenced measure additionally "
                "shows the direction of that disagreement."
            ),
            "",
            _markdown_table(
                importer_reference_display
            ),
            "",
            markdown_image(
                "Exporter/importer quantity ratio distribution",
                figure_paths[
                    "export_import_quantity_ratio_distribution"
                ],
                document_path=report_path,
            ),
            "",
            (
                f"- The mean exporter difference relative to the importer is "
                f"`{_format_percent(mean_difference_vs_import)}`. This very "
                "large value indicates an extremely right-skewed distribution "
                "rather than a typical exporter quantity being this much "
                "larger than the importer quantity."
            ),
            "",
            (
                "- Because the importer quantity is the denominator, rows with "
                "a very small importer-reported quantity and a much larger "
                "exporter-reported quantity can generate extremely large "
                "positive ratios. A relatively small number of such observations "
                "can therefore have a very large influence on the arithmetic mean."
            ),
            "",
            (
                f"- The median signed difference is "
                f"`{_format_percent(median_difference_vs_import)}`. "
                "This indicates that the typical bilateral observation shows "
                "very little overall directional difference between exporter "
                "and importer quantities."
            ),
            "",
            (
                f"- However, the median absolute difference is "
                f"`{_format_percent(median_absolute_difference_vs_import)}`. "
                "This shows that exporter and importer quantities frequently "
                "differ substantially in magnitude even though the direction "
                "of those differences is broadly balanced."
            ),
            "",
            (
                f"- Exporter quantities are larger than importer quantities in "
                f"`{_format_percent(exporter_higher_share)}` of comparable "
                "rows, while importer quantities are larger in "
                f"`{_format_percent(importer_higher_share)}`. Exact agreement "
                f"occurs in `{_format_percent(exact_agreement_share)}` of rows."
            ),
            "",
            (
                "- The importer is used here as the comparison base, not as an "
                "assumption that every importer-reported quantity is correct. "
                "The purpose of this diagnostic is to distinguish three "
                "different properties of the bilateral quantity discrepancy: "
                "its direction, its typical magnitude, and the presence of "
                "extreme asymmetric observations."
            ),
            "",
            markdown_styled_heading("3.3.2. Threshold sensitivity"),
            "",
            (
                "- This diagnostic evaluates three evidence measures within the directly observed bilateral population, "
                "where both exporter and importer quantities are available and neither quantity is estimated. For each measure, "
                "the table shows how many observations satisfy increasingly permissive or restrictive thresholds. "
            ),
            "",
            (
                "- The thresholds "
                "are cumulative rather than mutually exclusive; for example, the ≤10% Mirror-disagreement group includes all "
                "observations in the ≤5% group."
            ),
            "",
            (
                "- Mirror quantity disagreement measures exporter–importer quantity inconsistency, Product-year UV support measures "
                "the number of observations underlying the product-year unit-value reference, and Worst-side reporter UV distance "
                "measures how far the less plausible reporter-implied unit value lies from that reference."

            ),
            "",
            _markdown_table(
                sensitivity_display
            ),
            "",
            (
                "These results show how restrictive each individual "
                "criterion would be within the directly observed bilateral "
                "quantity population."
            ),
            "",
            markdown_styled_heading("3.3.3. Joint threshold combinations"),
            "",
            (
                "The following grid applies the three candidate-selection "
                "diagnostics jointly. Each row represents one possible "
                "candidate training population:"
            ),
            "",
            (
                "`direct bilateral quantities` + "
                "`maximum Mirror quantity disagreement` + "
                "`minimum Product-year UV support` + "
                "`maximum Worst-side reporter UV distance`."
            ),
            "",
            _markdown_table(
                threshold_grid_display
            ),
            "",
            (
                "The purpose of this grid is to expose the trade-off "
                "between purity and sample size. No combination becomes "
                "the production training definition merely because it has "
                "a large or small population."
            ),
            "",
            markdown_styled_heading("3.3.4. Mirror quantity disagreement and bilateral UV plausibility"),
            "",
            (
                "- Mirror quantity disagreement and bilateral UV distance "
                "provide complementary evidence about quantity credibility. "
                "The following matrix groups direct bilateral observations "
                "simultaneously by Mirror quantity disagreement and "
                "Worst-side reporter UV distance."
            ),
            "",
            (
                "- Worst-side reporter UV distance is the larger of the exporter and "
                "importer UV distances. A row therefore enters a low "
                "Worst-side reporter UV-distance category only when both reporting "
                "sides imply unit values close to the Product-year UV "
                "reference."
            ),
            "",
            _markdown_table(
                matrix_display,
                index=True,
            ),
            "",
            (
                "The low-disagreement, low-unit-value-distance cells are "
                "the strongest current candidates for training population. "
                "Rows with strong mirror agreement but weak unit-value "
                "plausibility remain questionable because both reporters "
                "can agree on a quantity that is economically unusual."
            ),
            "",
            markdown_styled_heading("3.3.5. Side-specific unit-value plausibility"),
            "",
            (
                "The following table evaluates Exporter UV distance and "
                "Importer UV distance separately within each Mirror quantity "
                "disagreement band. Worst-side reporter UV distance shows the less "
                "plausible of the two reporting sides."
            ),
            "",
            _markdown_table(
                side_plausibility_display
            ),
            "",
            markdown_styled_heading("3.3.6. Distributional evidence"),
            "",
            (
                "The following diagnostics describe the distribution of "
                "`Mirror quantity disagreement`, `Product-year UV support`, "
                "and `Worst-side reporter UV distance` among directly observed "
                "bilateral quantity rows."
            ),
            "",
            _markdown_table(
                distribution_display
            ),
            "",
            markdown_image(
                "Quantity-model training evidence distributions",
                figure_paths[
                    "training_evidence_distributions"
                ],
                document_path=report_path,
            ),
            "",
            (
                "Key take aways from these distributions:\n"
                "- For Mirror quantity disagreement, the blue line rises quickly and is already around 0.59 at the median. "
                "That means disagreement is not just an issue in a small number of extreme rows — it is widespread. By the 75th "
                f"percentile it is about {mirror_disagreement_p75}, so three quarters "
                f"of rows have disagreement up to {mirror_disagreement_p75}, and " 
                f"the upper tail is essentially near the maximum possible disagreement."
            ),
            "",
            (
            "- For Worst-side reporter UV distance, the orange line is much lower through the middle of the distribution, then rises sharply" 
                "in the upper tail. So most rows are relatively closer to their Product-year UV reference, while a smaller fraction " 
                "becomes very far away."
            ),
            "",
            (
                "- Mirror disagreement is broadly high across the population, whereas very large UV-distance problems are concentrated " 
                "more in the upper tail."
            ),
            "",
            (
                "Reporter-side unit-value distances are calculated using nominal reconciled "
                "trade value because the product-year UV reference is defined on the same "
                "nominal within-year basis."
            ),
            "",
            "",
            markdown_styled_heading("3.3.7. Quantity–UV relationship"),
            "",
            (
                "The previous UV diagnostics compare implied unit values "
                "with a Product-year UV reference. This assumes that a single "
                "reference is informative across different trade quantities. "
                "Before relying strongly on that assumption, Validation also "
                "examines whether unit value systematically varies with "
                "quantity."
            ),
            "",
            (
                "For this study, only direct bilateral quantity observations "
                "are used. To avoid choosing either exporter or importer as "
                "the true quantity, a symmetric bilateral quantity is defined "
                "as:"
            ),
            "",
            (
                "`bilateral quantity = "
                "sqrt(exporter quantity × importer quantity)`"
            ),
            "",
            (
                "The corresponding bilateral unit value is:"
            ),
            "",
            (
                "`bilateral UV = reconciled nominal trade value "
                "/ bilateral quantity`"
            ),
            "",
            (
                "For each HS product and year, Validation estimates the "
                "log-log relationship:"
            ),
            "",
            (
                "`log(bilateral UV) = α + β × "
                "log(bilateral quantity)`"
            ),
            "",
            (
                "The coefficient `β` measures how unit value changes with "
                "trade quantity. A negative value means that larger quantities "
                "tend to have lower unit values; a value near zero indicates "
                "little systematic relationship; and a positive value means "
                "that unit value tends to increase with quantity."
            ),
            "",
            (
                "For example, a slope of `−0.20` means that a 1% increase "
                "in quantity is associated with approximately a 0.20% "
                "decrease in unit value."
            ),
            "",
            (
                "The summary below uses HS × year groups containing at least "
                "30 direct bilateral observations. The minimum group size is "
                "used only to summarize reasonably supported relationships; "
                "the underlying diagnostics retain all available groups."
            ),
            "",
            _markdown_table(quantity_uv_summary_display),
            "",
            markdown_image(
                "Distribution of HS × year quantity–UV slopes",
                figure_paths["quantity_uv_slope_distribution"],
                document_path=report_path,
                width=850,
            ),
            "",
            (
                "The slope distribution shows how strongly bilateral unit value "
                "changes with bilateral quantity across supported HS × year groups. "
                "A negative slope means that larger quantities tend to be associated "
                "with lower unit values."
            ),
            "",
            markdown_image(
                "Distribution of HS × year quantity–UV R² values",
                figure_paths["quantity_uv_r_squared_distribution"],
                document_path=report_path,
                width=850,
            ),
            "",
            (
                "The R² distribution shows how much of the within-group variation "
                "in bilateral unit value is explained by bilateral quantity."
            ),
            "",
            (
                "This diagnostic is exploratory and does not currently alter "
                "the candidate training population. Its purpose is to test "
                "whether UV plausibility should eventually be conditioned on "
                "trade quantity rather than evaluated only against a single "
                "Product-year UV reference."
            ),
            "",
            "## 4. Quantity Candidate Evaluation",
            "",
            (
                "Validation retains exporter and importer quantities as the "
                "underlying reporter evidence and compares three fixed bilateral "
                "quantity candidates:"
            ),
            "",
            "- **Simple mean** — gives exporter and importer quantities equal weight.",
            "- **Value-weighted** — applies the existing Trade Sync value-reliability weights to the two quantities.",
            "- **Quantity-reliability-weighted** — combines the two quantities using reliability estimated directly from quantity mirror consistency.",
            "",
            "",
            (
                "All candidate unit-value diagnostics use the same reconciled "
                "nominal trade value and the same product-year UV reference. "
                "The comparison is therefore like-for-like across candidates."
            ),
            "",
            "### 4.1. Overall candidate comparison",
            "",
            _markdown_table(candidate_summary_display),
            "",
            markdown_image(
                "Fixed quantity candidates: unit-value plausibility",
                figure_paths["quantity_candidate_uv_summary"],
                document_path=report_path,
                width=850,
            ),
            "",
            (
                "The median and upper-tail UV distances show how each fixed "
                "candidate behaves against the common product-year UV reference. "
                "The exporter/importer log-gap columns show how far each candidate "
                "typically moves from the two underlying reporter quantities."
            ),
            "",
            "### 4.2. Candidate behavior as mirror disagreement increases",
            "",
            _markdown_table(disagreement_display),
            "",
            plot_figure_pair(
                markdown_image(
                    "Candidate UV plausibility by mirror disagreement — log scale",
                    figure_paths[
                        "quantity_candidate_by_disagreement"
                    ],
                    document_path=report_path,
                    width=650,
                    height=450,
                ),
                markdown_image(
                    "Candidate UV plausibility by mirror disagreement — factor scale",
                    figure_paths[
                        "quantity_candidate_by_disagreement_factor"
                    ],
                    document_path=report_path,
                    width=650,
                    height=450,
                ),
            ),
            "",
            (
                "This comparison is especially important because candidate "
                "construction matters most when exporter and importer quantities "
                "diverge. Similar performance in low-disagreement rows does not "
                "establish that the candidates behave similarly when mirror "
                "disagreement is large."
            ),
            "",
            "### 4.3. Closest candidate to the product-year UV reference",
            "",
            _markdown_table(closest_display),
            "",
            markdown_image(
                "Candidate closest to the product-year UV reference",
                figure_paths["quantity_candidate_closest_reference"],
                document_path=report_path,
                width=850,
            ),
            "",
            "### 4.4. Candidate spread",
            "",
            (
                "- Candidate spread measures how far apart the three fixed quantity "
                "candidates are on the same bilateral row. The spread is therefore the distance between the largest and "
                "smallest candidate quantity, expressed relative to the max candidate quantity."
            ),
            "",
            "$$",
            r"S_Q"
            r" = "
            r"\frac{"
            r"\max\left("
            r"Q_{\mathrm{mean}},"
            r"Q_{\mathrm{value\text{-}weighted}},"
            r"Q_{\mathrm{quantity\text{-}weighted}}"
            r"\right)"
            r" - "
            r"\min\left("
            r"Q_{\mathrm{mean}},"
            r"Q_{\mathrm{value\text{-}weighted}},"
            r"Q_{\mathrm{quantity\text{-}weighted}}"
            r"\right)"
            r"}{"
            r"\max\left("
            r"Q_{\mathrm{mean}},"
            r"Q_{\mathrm{value\text{-}weighted}},"
            r"Q_{\mathrm{quantity\text{-}weighted}}"
            r"\right)"
            r"}",
            "$$",
            "",
            (
                "   - A small spread means the final methodological choice has little "
                "numerical effect for that disagreement regime."
            ),
            (
                "   - A large spread means the choice is consequential."
            ),
            "",
            _markdown_table(spread_display),
            "",
            markdown_image(
                "Spread among the three fixed quantity candidates",
                figure_paths["quantity_candidate_spread"],
                document_path=report_path,
                width=850,
            ),
            "",
            "### 4.5. Temporal stability",
            "",
            _markdown_table(candidate_year_display),
            "",
            markdown_image(
                "Candidate unit-value plausibility by year",
                figure_paths["quantity_candidate_by_year"],
                document_path=report_path,
                width=850,
            ),
            "",
            (
                "The year comparison checks whether a candidate's apparent "
                "plausibility is persistent through time rather than driven by "
                "a small number of years."
            ),
            "",
            "### 4.6. Product stability",
            "",
            markdown_image(
                "Product-level distribution of candidate median UV distance",
                figure_paths["quantity_candidate_product_distribution"],
                document_path=report_path,
                width=850,
            ),
            "",
            (
                "The product-level distribution shows whether candidate behavior "
                "is broadly shared across HS products or concentrated in a subset "
                "of products. The full per-product diagnostics remain in the "
                "diagnostics artifact rather than being expanded into a very large "
                "report table."
            ),
            "",
            "### 4.7. Quantity-conditioned UV plausibility",
            "",
            (
                "The unconditional Product-year UV reference does not account "
                "for the systematic relationship between trade quantity and "
                "unit value documented in Section 3.3.7. Validation therefore "
                "also evaluates each fixed candidate against the UV expected "
                "at that candidate's own quantity level."
            ),
            "",
            (
                "Within each HS × year group containing at least 30 direct "
                "bilateral observations, Validation estimates "
                "`log(UV) = α + β × log(quantity)`. For each candidate, the "
                "absolute residual from this relationship is then calculated. "
                "Smaller residuals indicate greater quantity-conditioned UV "
                "plausibility. This is a diagnostic comparison, not an automatic "
                "candidate-selection rule."
            ),
            "",
            _markdown_table(conditioned_summary_display),
            "",
            markdown_image(
                "Quantity-conditioned UV plausibility by candidate",
                figure_paths["quantity_candidate_conditioned_summary"],
                document_path=report_path,
                width=850,
            ),
            "",
            "#### 4.7.1. Quantity-conditioned plausibility by mirror disagreement",
            "",
            _markdown_table(conditioned_disagreement_display),
            "",
            markdown_image(
                "Quantity-conditioned candidate plausibility by mirror disagreement",
                figure_paths["quantity_candidate_conditioned_by_disagreement"],
                document_path=report_path,
                width=850,
            ),
            "",
            "#### 4.7.2. Candidate closest to quantity-conditioned expected UV",
            "",
            _markdown_table(conditioned_closest_display),
            "",
            markdown_image(
                "Candidate closest to quantity-conditioned expected UV",
                figure_paths["quantity_candidate_closest_conditioned"],
                document_path=report_path,
                width=850,
            ),
            "",
            (
                "This quantity-conditioned comparison is particularly useful "
                "for the high-disagreement bands, where the three candidate "
                "quantities differ most and where an unconditional Product-year "
                "median UV can be least informative."
            ),
            "",
            "## 5. Diagnostic Coverage",
            "",
            (
                "The following diagnostics describe the temporal and "
                "product coverage of the available quantity evidence. "
                "They are intended to support the selection of a "
                "defensible training population that is large enough, "
                "sufficiently clean, and adequately represented across "
                "years and products."
            ),
            "",
            "### 5.1. Temporal coverage",
            "",
            (
                "A candidate training population must provide "
                "useful evidence across the full time horizon rather than "
                "being concentrated in only a few years."
            ),
            "",
            _markdown_table(
                year_display
            ),
            "",
            markdown_image(
                "Direct bilateral quantity evidence by year",
                figure_paths[
                    "training_evidence_by_year"
                ],
                document_path=report_path,
            ),
            "",
            "### 5.2. Product coverage",
            "",
            (
                f"- The current diagnostics contain `{_format_integer(n_products)}` "
                "HS products. "
                f"`{_format_integer(n_products_with_direct_evidence)}` "
                "have at least one directly observed bilateral quantity "
                "observation."
            ),
            "",
            (
                "- The median within-product share represented by directly "
                "observed bilateral quantity evidence is "
                f"`{_format_percent(median_product_direct_share)}`."
            ),
            "",
            (
                "- The following table shows the 20 products with the largest "
                "numbers of directly observed bilateral quantity rows. "
                "This is a coverage diagnostic, not a ranking of product "
                "quality."
            ),
            "",
            _markdown_table(
                product_display
            ),
            "",
            "## 6. Training-sample diagnostics",
            "",
            (
                "The training-sample step does not yet define a final "
                "model-training population. It constructs the evidence "
                "needed to make that decision."
            ),
            "",
            (
                "The HS × year reference is derived from directly observed "
                "bilateral quantity rows where neither reporter quantity is "
                "estimated. Each usable selected quantity is then evaluated "
                "through its implied unit value relative to that reference."
            ),
            "",
            "### 6.1. Training-sample population",
            "",
            _markdown_table(
                training_sample_summary_display
            ),
            "",
            "### 6.2. Selected-unit-value distance",
            "",
            (
                "The following distribution shows how far the unit value implied by "
                "the selected quantity lies from the robust HS × year reference. "
                "Validation computes this distance on the absolute log scale, while "
                "the factor scale expresses the same distance more intuitively. "
                "A factor of `1×` means exact agreement with the reference."
            ),
            "",
            get_table_figure_layout(
                training_uv_distance_display,
                markdown_image(
                    "Selected unit-value distance distribution",
                    figure_paths[
                        "training_uv_distance_distribution"
                    ],
                    document_path=report_path,
                    width=700,
                    height=None,
                ),
                table_width=38,
            ),
            "",
            "### 6.3. HS × year reference support",
            "",
            (
                "Reference support is shown across distinct HS × year groups. "
                "The number of supporting observations indicates how much evidence "
                "underlies each product-year UV reference and helps assess whether "
                "a minimum support requirement is needed."
            ),
            "",
            get_table_figure_layout(
                training_reference_support_display,
                markdown_image(
                    "HS × year unit-value reference support",
                    figure_paths[
                        "training_reference_support_distribution"
                    ],
                    document_path=report_path,
                    width=700,
                    height=None,
                ),
                table_width=34,
            ),
            "",
            "### 6.4. HS × year unit-value dispersion",
            "",
            (
                "The log-MAD describes within-group dispersion around the "
                "HS × year unit-value reference. This helps assess whether one "
                "global absolute UV-distance cutoff is defensible or whether "
                "dispersion-aware criteria should later be considered."
            ),
            "",
            get_table_figure_layout(
                training_uv_dispersion_display,
                markdown_image(
                    "HS × year unit-value dispersion",
                    figure_paths[
                        "training_uv_dispersion_distribution"
                    ],
                    document_path=report_path,
                    width=700,
                    height=None,
                ),
                table_width=34,
            ),
            "",
            "### 6.5. Dispersion-normalized selected-unit-value distance",
            "",
            (
                "Absolute UV distance does not have the same interpretation "
                "for every HS × year group because unit-value dispersion differs "
                "substantially across product-year groups. Validation therefore "
                "also evaluates the selected-unit-value log distance relative to "
                "the robust within-group log-UV MAD."
            ),
            "",
            (
                "The diagnostic is defined as "
                "`selected UV log distance / HS × year log-UV MAD`. "
                "A value of 1 means that the selected unit value lies one "
                "within-group MAD away from the HS × year median reference. "
                "No cutoff is applied at this stage."
            ),
            "",
            get_table_figure_layout(
                training_normalized_uv_distance_display,
                markdown_image(
                    "Selected unit-value distance relative to HS × year dispersion",
                    figure_paths[
                        "training_normalized_uv_distance_distribution"
                    ],
                    document_path=report_path,
                    width=700,
                    height=None,
                ),
                table_width=34,
            ),
            "",
            markdown_styled_heading(text="6.5.1. Candidate normalized-distance cutoffs"),
            "",
            (
                "To support the eventual training-population decision, "
                "Validation evaluates several candidate cutoffs for the "
                "dispersion-normalized unit-value distance. These are "
                "diagnostic comparison points only; none is yet applied as "
                "the production training rule."
            ),
            "",
            (
                "The direct bilateral reference population contains rows with "
                "independent, non-estimated quantity evidence from both reporting "
                "sides. The hard-eligible population additionally includes rows "
                "whose selected quantity and value can be evaluated against the "
                "HS × year reference."
            ),
            "",
            _markdown_table(
                normalized_cutoff_display
            ),
            "",
            markdown_styled_heading(text="6.5.2. Candidate joint training rule"),
            "",
            (
                "The currently preferred candidate rule combines the direct "
                "bilateral reference population with a minimum HS × year "
                "reference support of 30 and a maximum normalized UV distance "
                "of 2.5 MAD."
            ),
            "",
            _markdown_table(
                candidate_training_rule_display
            ),
            "",
            "## 7. Current methodological assessment",
            "",
            training_status,
            "",
            (
                "- The bilateral quantity methodology has now been selected "
                "and independently audited. The remaining methodological "
                "decision before model fitting is the definition of the "
                "training population."
            ),
            "",
            (
                "- No minimum HS × year support threshold or maximum "
                "unit-value-distance threshold is imposed yet. The raw and "
                "dispersion-normalized distributions above are intended to "
                "support that decision from the observed data."
            ),
            "",
            (
                "- Once the training population is selected, model performance "
                "should be tested using controlled missingness and corruption "
                "of high-confidence observations rather than by assuming that "
                "all available quantities are true."
            ),
            "",
            "## 8. Output artifacts",
            "",
            (
                f"- Prepared Validation input: "
                f"`{prepared_input_path}`"
            ),
            (
                f"- Structural audit: "
                f"`{structural_audit_path}`"
            ),
            (
                f"- Diagnostics: "
                f"`{diagnostics_path}`"
            ),
            f"- Report: `{report_path}`",
            "",
        ]

        insert_markdown_toc(
            lines,
            min_level=2,
            max_level=4,
            flat_sublevels=True,
        )

        report_path.write_text(
            "\n".join(
                lines
            ),
            encoding="utf-8",
        )

        if not valid_output(
            report_path
        ):
            raise RuntimeError(
                "Validation decision report was not written correctly."
            )

        latest_report_path: Path | None = None

        if cfg.artifacts.write_latest:
            latest_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copyfile(
                report_path,
                latest_path,
            )

            latest_report_path = (
                latest_path
            )

        logger.info(
            "%s Validation decision report created.",
            ok,
        )

        logger.info(
            "  Structural audit passed: %s",
            audit_passed,
        )

        logger.info(
            "  Training population selected: %s",
            training_selection_enabled,
        )

        logger.info(
            "  Report: %s",
            report_path,
        )

        if latest_report_path is not None:
            logger.info(
                "  Latest report: %s",
                latest_report_path,
            )

        logger.info(
            "  Figures written: %d",
            len(
                figure_paths
            ),
        )

        return {
            "status": "completed",
            "run_id": run_id,

            "output_path": report_path,
            "report_path": report_path,

            "latest_report_path": (
                latest_report_path
            ),

            "figure_paths": (
                figure_paths
            ),

            "audit_passed": (
                audit_passed
            ),

            "training_population_selected": (
                training_selection_enabled
            ),

            "ready_for_training_decision": True,

            "duration_seconds": None,
        }

    finally:
        con.close()