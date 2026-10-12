"""
vacoda.steps.validation.reporting

Create the VACODA Validation model-development report.

The report follows the canonical bilateral Validation methodology:

1. structural integrity of the Validation input;
2. selected canonical bilateral quantity methodology;
3. source-reporter diagnostics, shown descriptively only;
4. HS × year unit-value reference evidence;
5. configured model-training population;
6. expanding-window model evaluation;
7. placeholder for later production inference diagnostics.

Scientific calculations belong in the producing Validation steps.  This module
consumes structural-audit and diagnostics artifacts and turns them into concise
human-readable tables, figures, and methodological interpretation.
"""

from __future__ import annotations

from pathlib import Path
import math
import shutil
from typing import Any

import duckdb
import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from vacoda.config.paths import get_figure_dir
from vacoda.config.steps.validation import ValidationConfig
from vacoda.utils.display import get_symbols, get_table_figure_layout
from vacoda.utils.markdown import insert_markdown_toc, markdown_image
from vacoda.utils.sql import sql_literal
from vacoda.visualization.relationships import plot_figure_pair


# ======================================================================
# Helpers
# ======================================================================


def valid_output(path: Path) -> bool:
    """Return True when an existing output appears usable."""

    return path.is_file() and path.stat().st_size > 0



def _format_integer(value: float | int | None) -> str:
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

    return f"{100.0 * float(value):.{digits}f}%"


def _plot_weighted_ensemble_curves(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> Path | None:

    curve = diagnostics.loc[
        diagnostics["scope"]
        == "model_weighted_ensemble_curve"
    ].copy()

    if curve.empty:
        return None

    curve = curve.loc[
        curve["variable"] == "mae_log"
    ].copy()

    if curve.empty:
        return None

    curve["weight"] = pd.to_numeric(
        curve["weight"],
        errors="coerce",
    )

    curve["value"] = pd.to_numeric(
        curve["value"],
        errors="coerce",
    )

    curve = curve.dropna(
        subset=["weight", "value"]
    )

    if curve.empty:
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    for (
        model_name,
        candidate,
    ), group in curve.groupby(
        ["model_name", "candidate"],
        sort=True,
    ):
        group = group.sort_values(
            "weight"
        )

        label = (
            f"{model_name} + {candidate}"
        )

        ax.plot(
            group["weight"].to_numpy(dtype=float),
            group["value"].to_numpy(dtype=float),
            marker="o",
            label=label,
        )

    ax.set_xlabel(
        "Weight on Model A"
    )
    ax.set_ylabel(
        "Out-of-fold MAE (log scale)"
    )
    ax.set_title(
        "Weighted ensemble performance curves"
    )
    ax.grid(
        True,
        alpha=0.3,
    )
    ax.legend(
        loc="best",
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path

def _plot_reference_window_comparison(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = _require_diagnostic_scope(
        diagnostics,
        "reference_window_summary",
    )

    frame = frame[
        frame["variable"].eq("groups_with_support")
    ].copy()

    order = ["0_0", "1_0", "2_0"]

    frame["order"] = frame["criterion"].map(
        {value: index for index, value in enumerate(order)}
    )
    frame = frame.sort_values("order")

    figure, axis = plt.subplots(figsize=(9, 5.5))

    x = range(len(frame))

    bars = axis.bar(
        x,
        100.0 * pd.to_numeric(
            frame["share_of_parent"],
            errors="coerce",
        ),
    )

    axis.set_xticks(list(x))
    axis.set_xticklabels(
        frame["criterion"].astype(str)
    )

    axis.bar_label(
        bars,
        labels=[
            _format_percent(value)
            for value in frame["share_of_parent"]
        ],
        padding=3,
        fontsize=8,
    )

    axis.set_title(
        "HS × year reference support by temporal window"
    )
    axis.set_xlabel("Reference window")
    axis.set_ylabel(
        "HS × year groups meeting support threshold (%)"
    )
    axis.set_ylim(0, 100)
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

def _format_number(
    value: float | int | None,
    *,
    digits: int = 4,
) -> str:
    """Format a general numeric value."""

    if value is None or pd.isna(value):
        return "NA"

    return f"{float(value):,.{digits}f}"

def _model_plot_label(value: str) -> str:
    """Return compact model labels for Validation figures."""

    labels = {
        "emb_mlp_bs4096_lr0p001": "Emb MLP 4096",
        "emb_mlp_bs8192_lr0p001": "Emb MLP 8192",
        "emb_mlp_bs16384_lr0p001": "Emb MLP 16384",
        "extra_trees": "Extra Trees",
        "random_forest": "Random Forest",
        "histogram_gradient_boosting": "HGB",
        "mlp": "MLP",
        "ridge": "Ridge",
    }

    return labels.get(
        str(value),
        str(value),
    )

def _model_plot_marker(
    model_name: str,
) -> str:
    """Return a distinct marker per model."""

    markers = {
        "ridge": "X",
        "mlp": "P",
        "extra_trees": "s",
        "random_forest": "D",
        "histogram_gradient_boosting": "^",
        "emb_mlp_bs4096_lr0p001": "o",
        "emb_mlp_bs8192_lr0p001": "v",
        "emb_mlp_bs16384_lr0p001": "*",
    }

    return markers.get(
        model_name,
        "o",
    )

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


def _plot_model_residual_correlation(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot pairwise out-of-fold residual correlations."""

    frame = _require_diagnostic_scope(
        diagnostics,
        "model_residual_correlation",
    ).copy()

    frame = frame[
        frame["variable"].eq(
            "residual_correlation"
        )
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "Model residual-correlation diagnostics are unavailable."
        )

    frame["value"] = pd.to_numeric(
        frame["value"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "model_name",
            "candidate",
            "value",
        ]
    )

    model_names = sorted(
        set(
            frame["model_name"].astype(str)
        )
        | set(
            frame["candidate"].astype(str)
        )
    )

    matrix = pd.DataFrame(
        1.0,
        index=model_names,
        columns=model_names,
        dtype=float,
    )

    for row in frame.itertuples(
        index=False
    ):
        model_a = str(
            row.model_name
        )
        model_b = str(
            row.candidate
        )
        correlation = float(
            row.value
        )

        matrix.loc[
            model_a,
            model_b,
        ] = correlation

        matrix.loc[
            model_b,
            model_a,
        ] = correlation

    labels = [
        _model_plot_label(
            model_name
        )
        for model_name in model_names
    ]

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    image = axis.imshow(
        matrix.to_numpy(),
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    axis.set_xticks(
        range(
            len(model_names)
        )
    )
    axis.set_yticks(
        range(
            len(model_names)
        )
    )

    axis.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        labels
    )

    for row_index in range(
        len(model_names)
    ):
        for column_index in range(
            len(model_names)
        ):
            axis.text(
                column_index,
                row_index,
                (
                    f"{matrix.iloc[row_index, column_index]:.2f}"
                ),
                ha="center",
                va="center",
                fontsize=8,
            )

    axis.set_title(
        "Correlation of out-of-fold model residuals"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Pearson residual correlation"
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

def _plot_model_ensemble_pair_diagnostic(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot ensemble-pair quality against residual diversity.

    Lower MAE and lower residual correlation are preferable.

    All candidate pairs are shown for context, while Pareto-efficient
    pairs are emphasized. Cross-family Pareto pairs are labelled;
    embedding-versus-embedding Pareto pairs remain visible but unlabelled.
    """

    frame = _require_diagnostic_scope(
        diagnostics,
        "model_ensemble_pair_diagnostic",
    ).copy()

    pivot = (
        frame.pivot_table(
            index=[
                "model_name",
                "candidate",
            ],
            columns="variable",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    required = {
        "residual_correlation",
        "mean_pair_mae_log",
    }

    missing = (
        required
        - set(
            pivot.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Ensemble-pair diagnostics are missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    pivot["residual_correlation"] = pd.to_numeric(
        pivot[
            "residual_correlation"
        ],
        errors="coerce",
    )

    pivot["mean_pair_mae_log"] = pd.to_numeric(
        pivot[
            "mean_pair_mae_log"
        ],
        errors="coerce",
    )

    pivot = pivot.dropna(
        subset=[
            "residual_correlation",
            "mean_pair_mae_log",
        ]
    ).copy()

    # --------------------------------------------------------------
    # Identify Pareto-efficient pairs
    # --------------------------------------------------------------
    #
    # Both axes are minimized:
    #
    #   x: residual correlation
    #   y: mean standalone MAE
    #
    # After sorting from lowest residual correlation upward,
    # a pair enters the Pareto frontier only when it improves
    # upon the best mean MAE encountered so far.
    # --------------------------------------------------------------

    ordered = pivot.sort_values(
        [
            "residual_correlation",
            "mean_pair_mae_log",
        ],
        ascending=[
            True,
            True,
        ],
    ).copy()

    pareto_indices = []
    best_mae_so_far = math.inf

    for row in ordered.itertuples():

        mae = float(
            row.mean_pair_mae_log
        )

        if mae < best_mae_so_far:

            pareto_indices.append(
                row.Index
            )

            best_mae_so_far = mae

    pivot["pareto"] = (
        pivot.index.isin(
            pareto_indices
        )
    )

    background = pivot[
        ~pivot["pareto"]
    ].copy()

    frontier = (
        pivot[
            pivot["pareto"]
        ]
        .sort_values(
            "residual_correlation"
        )
        .copy()
    )

    # --------------------------------------------------------------
    # Figure
    # --------------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(10, 6.5)
    )

    # Dominated pairs remain visible as context.
    axis.scatter(
        background[
            "residual_correlation"
        ],
        background[
            "mean_pair_mae_log"
        ],
        alpha=0.35,
        s=45,
        label="Other candidate pairs",
    )

    # Pareto-efficient pairs are emphasized.
    axis.scatter(
        frontier[
            "residual_correlation"
        ],
        frontier[
            "mean_pair_mae_log"
        ],
        s=75,
        label="Pareto-efficient pairs",
    )

    # --------------------------------------------------------------
    # Connect Pareto-efficient pairs
    # --------------------------------------------------------------

    axis.plot(
        frontier[
            "residual_correlation"
        ],
        frontier[
            "mean_pair_mae_log"
        ],
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )

    # --------------------------------------------------------------
    # Label only cross-family Pareto pairs
    # --------------------------------------------------------------

    embedding_models = {
        "emb_mlp_bs4096_lr0p001",
        "emb_mlp_bs8192_lr0p001",
        "emb_mlp_bs16384_lr0p001",
    }

    frontier_to_label = frontier[
        ~(
            frontier["model_name"].isin(
                embedding_models
            )
            &
            frontier["candidate"].isin(
                embedding_models
            )
        )
    ].copy()

    offsets = [
        (6, 6),
        (6, -12),
        (6, 6),
        (6, -12),
        (6, 6),
        (6, -12),
    ]

    for index, row in enumerate(
        frontier_to_label.itertuples(
            index=False
        )
    ):

        label = (
            f"{_model_plot_label(row.model_name)} + "
            f"{_model_plot_label(row.candidate)}"
        )

        axis.annotate(
            label,
            (
                float(
                    row.residual_correlation
                ),
                float(
                    row.mean_pair_mae_log
                ),
            ),
            xytext=(
                offsets[
                    index % len(offsets)
                ]
            ),
            textcoords="offset points",
            fontsize=8,
        )

    axis.set_title(
        "Candidate ensemble pairs: quality versus residual diversity"
    )

    axis.set_xlabel(
        "Residual correlation — lower means more diverse"
    )

    axis.set_ylabel(
        "Mean standalone MAE of pair — lower means stronger"
    )

    axis.grid(
        axis="both",
        linestyle="--",
        alpha=0.5,
    )

    axis.set_axisbelow(
        True
    )

    axis.legend(
        loc="upper right"
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

def _plot_model_ensemble_pair_diagnostic_deprecated(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot ensemble-pair quality against residual diversity.

    Lower MAE and lower residual correlation are preferable,
    so potentially useful ensemble pairs lie toward the lower-left.
    """

    frame = _require_diagnostic_scope(
        diagnostics,
        "model_ensemble_pair_diagnostic",
    ).copy()

    pivot = (
        frame.pivot_table(
            index=[
                "model_name",
                "candidate",
            ],
            columns="variable",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    required = {
        "residual_correlation",
        "mean_pair_mae_log",
    }

    missing = (
        required
        - set(
            pivot.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Ensemble-pair diagnostics are missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    pivot["residual_correlation"] = pd.to_numeric(
        pivot[
            "residual_correlation"
        ],
        errors="coerce",
    )

    pivot["mean_pair_mae_log"] = pd.to_numeric(
        pivot[
            "mean_pair_mae_log"
        ],
        errors="coerce",
    )

    pivot = pivot.dropna(
        subset=[
            "residual_correlation",
            "mean_pair_mae_log",
        ]
    )

    figure, axis = plt.subplots(
        figsize=(10, 6.5)
    )

    axis.scatter(
        pivot[
            "residual_correlation"
        ],
        pivot[
            "mean_pair_mae_log"
        ],
    )

    for row in pivot.itertuples(
        index=False
    ):

        label = (
            f"{_model_plot_label(row.model_name)} + "
            f"{_model_plot_label(row.candidate)}"
        )

        axis.annotate(
            label,
            (
                float(
                    row.residual_correlation
                ),
                float(
                    row.mean_pair_mae_log
                ),
            ),
            xytext=(
                5,
                4,
            ),
            textcoords="offset points",
            fontsize=7,
        )

    axis.set_title(
        "Candidate ensemble pairs: quality versus residual diversity"
    )

    axis.set_xlabel(
        "Residual correlation — lower means more diverse"
    )

    axis.set_ylabel(
        "Mean standalone MAE of pair — lower means stronger"
    )

    axis.grid(
        axis="both",
        linestyle="--",
        alpha=0.5,
    )

    axis.set_axisbelow(
        True
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

def _plot_training_uv_mad_distribution(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot HS × year log-UV MAD quantiles."""

    frame = _scope_frame(
        diagnostics,
        "training_uv_mad_distribution",
    ).copy()

    if frame.empty:
        raise RuntimeError(
            "Training UV-MAD diagnostics are unavailable."
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

def _candidate_label(value: str) -> str:
    """Return a readable label for the selected quantity methodology."""

    labels = {
        "simple_mean": "Simple mean",
        "value_weighted": "Value-weighted",
        "quantity_reliability_weighted": "Quantity-reliability-weighted",
    }

    return labels.get(str(value), str(value))



def _require_diagnostic_scope(
    diagnostics: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    """Return a required scope or raise a clear error."""

    frame = _scope_frame(
        diagnostics,
        scope,
    )

    if frame.empty:
        raise RuntimeError(
            f"Validation diagnostics do not contain required scope: {scope}"
        )

    return frame


# ======================================================================
# Figures
# ======================================================================
def _plot_training_coverage_by_product(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = _require_diagnostic_scope(
        diagnostics,
        "training_coverage_by_product",
    ).copy()

    frame["training_share"] = pd.to_numeric(
        frame["share_of_all"],
        errors="coerce",
    )

    frame = frame.sort_values(
        ["training_share", "target_cmd"]
    ).reset_index(drop=True)

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        range(len(frame)),
        100.0 * frame["training_share"],
    )

    axis.set_title(
        "Training-sample coverage across HS products"
    )
    axis.set_xlabel(
        "HS products ranked by training coverage"
    )
    axis.set_ylabel(
        "Training-selected share of product rows (%)"
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

def _plot_source_quantity_availability(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot source-reporter quantity availability."""

    frame = _require_diagnostic_scope(
        diagnostics,
        "source_quantity_availability",
    )

    order = (
        "both",
        "export_only",
        "import_only",
        "neither",
    )

    labels = {
        "both": "Both sides",
        "export_only": "Exporter only",
        "import_only": "Importer only",
        "neither": "Neither side",
    }

    frame = frame[
        frame["criterion"].isin(order)
    ].copy()

    frame["order"] = frame["criterion"].map(
        {value: index for index, value in enumerate(order)}
    )

    frame = frame.sort_values("order")

    figure, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    bars = axis.bar(
        [labels[value] for value in frame["criterion"]],
        100.0 * pd.to_numeric(frame["share_of_all"], errors="coerce"),
    )

    axis.bar_label(
        bars,
        labels=[
            _format_percent(value)
            for value in frame["share_of_all"]
        ],
        padding=3,
        fontsize=8,
    )

    axis.set_title(
        "Source-reporter quantity availability"
    )
    axis.set_ylabel(
        "Share of canonical bilateral rows (%)"
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)



def _plot_source_disagreement_bands(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot descriptive exporter/importer quantity disagreement bands."""

    frame = _require_diagnostic_scope(
        diagnostics,
        "source_quantity_disagreement_band",
    )

    order = (
        "0-5%",
        "5-10%",
        "10-25%",
        "25-50%",
        "50-100%",
    )

    frame = frame[
        frame["disagreement_band"].isin(order)
    ].copy()

    frame["order"] = frame["disagreement_band"].map(
        {value: index for index, value in enumerate(order)}
    )

    frame = frame.sort_values("order")

    figure, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    bars = axis.bar(
        frame["disagreement_band"],
        100.0 * pd.to_numeric(frame["share_of_parent"], errors="coerce"),
    )

    axis.bar_label(
        bars,
        labels=[
            _format_percent(value)
            for value in frame["share_of_parent"]
        ],
        padding=3,
        fontsize=8,
    )

    axis.set_title(
        "Exporter/importer quantity disagreement — source diagnostic"
    )
    axis.set_xlabel(
        "Relative disagreement band"
    )
    axis.set_ylabel(
        "Share of rows with both source quantities (%)"
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)



def _plot_training_population_funnel(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot row retention through the configured training rule."""

    frame = _require_diagnostic_scope(
        diagnostics,
        "training_population_funnel",
    )

    order = (
        "all_rows",
        "training_eligible",
        "support",
        "positive_uv_mad",
        "uv_distance_available",
        "training_selected",
    )

    labels = {
        "all_rows": "All canonical rows",
        "training_eligible": "Usable quantity + value",
        "support": "HS×year support threshold",
        "positive_uv_mad": "Positive UV MAD",
        "uv_distance_available": "Normalized UV evaluable",
        "training_selected": "Final training sample",
    }

    frame = frame[
        frame["criterion"].isin(order)
    ].copy()

    frame["order"] = frame["criterion"].map(
        {value: index for index, value in enumerate(order)}
    )

    frame = frame.sort_values("order")

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    bars = axis.barh(
        [labels[value] for value in frame["criterion"]],
        pd.to_numeric(frame["n_rows"], errors="coerce"),
    )

    axis.bar_label(
        bars,
        labels=[
            _format_integer(value)
            for value in frame["n_rows"]
        ],
        padding=4,
        fontsize=8,
    )

    axis.invert_yaxis()
    axis.set_title(
        "Configured model-training population"
    )
    axis.set_xlabel(
        "Number of canonical bilateral observations"
    )
    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.5,
    )
    axis.set_axisbelow(True)
    axis.margins(x=0.12)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)



def _plot_normalized_uv_cutoff_sensitivity(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
    configured_cutoff: float,
) -> None:
    """Plot training retention across normalized-UV cutoffs."""

    frame = _require_diagnostic_scope(
        diagnostics,
        "training_normalized_uv_cutoff_sensitivity",
    ).sort_values(
        "uv_distance_threshold"
    )

    x = pd.to_numeric(
        frame["uv_distance_threshold"],
        errors="coerce",
    )

    y = 100.0 * pd.to_numeric(
        frame["share_of_parent"],
        errors="coerce",
    )

    figure, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    axis.plot(
        x,
        y,
        marker="o",
    )

    axis.axvline(
        float(configured_cutoff),
        linestyle="--",
        linewidth=1.4,
        label=f"Configured cutoff = {configured_cutoff:g} MAD",
    )

    axis.set_title(
        "Normalized unit-value cutoff sensitivity"
    )
    axis.set_xlabel(
        "Maximum normalized UV distance (MAD units)"
    )
    axis.set_ylabel(
        "Retained share of UV-evaluable rows (%)"
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



def _plot_training_coverage_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot final training-sample coverage by year."""

    frame = _require_diagnostic_scope(
        diagnostics,
        "training_coverage_by_year",
    ).sort_values("year")

    figure, axis = plt.subplots(
        figsize=(10, 5.5)
    )

    axis.plot(
        pd.to_numeric(frame["year"], errors="coerce"),
        100.0 * pd.to_numeric(frame["share_of_all"], errors="coerce"),
        marker="o",
    )

    axis.set_title(
        "Final training-sample coverage by year"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel(
        "Training-selected share of annual Validation rows (%)"
    )
    axis.set_xticks(
        sorted(
            frame["year"].dropna().astype(int).unique()
        )
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)



def _plot_model_performance_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot expanding-window MAE by candidate model and validation year."""

    frame = _require_diagnostic_scope(
        diagnostics,
        "model_performance_by_year",
    )

    frame = frame[
        frame["variable"].eq("mae_log")
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "Model performance-by-year diagnostics contain no mae_log rows."
        )

    figure, axis = plt.subplots(
        figsize=(10, 5.5)
    )

    for model_name, group in frame.groupby(
        "model_name",
        sort=False,
    ):
        group = group.sort_values("year")
        axis.plot(
            pd.to_numeric(group["year"], errors="coerce"),
            pd.to_numeric(group["value"], errors="coerce"),
            marker="o",
            label=str(model_name),
        )

    axis.set_title(
        "Expanding-window model performance by validation year"
    )
    axis.set_xlabel(
        "Validation year"
    )
    axis.set_ylabel(
        "MAE on log quantity"
    )
    axis.set_xticks(
        sorted(
            frame["year"].dropna().astype(int).unique()
        )
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



def _build_validation_figure_paths(
    *,
    figure_dir: Path,
    target_hs: str,
) -> dict[str, Path]:
    """Return figure paths used by the streamlined Validation report."""

    target_hs = str(target_hs).lower()
    prefix = f"target_{target_hs}_validation_"

    return {
        "training_coverage_by_product": (
            figure_dir / f"{prefix}training_coverage_by_product.png"
        ),
        "reference_window_comparison": (
            figure_dir / f"{prefix}reference_window_comparison.png"
        ),
        "training_uv_dispersion_distribution": (
            figure_dir / f"{prefix}training_uv_dispersion_distribution.png"
        ),
        "training_uv_distance_distribution": (
            figure_dir / f"{prefix}training_uv_distance_distribution.png"
        ),

        "training_reference_support_distribution": (
            figure_dir / f"{prefix}training_reference_support_distribution.png"
        ),

        "training_uv_mad_distribution": (
            figure_dir / f"{prefix}training_uv_mad_distribution.png"
        ),

        "training_normalized_uv_distance_distribution": (
            figure_dir / f"{prefix}training_normalized_uv_distance_distribution.png"
        ),
        "source_quantity_availability": (
            figure_dir / f"{prefix}source_quantity_availability.png"
        ),
        "source_quantity_disagreement": (
            figure_dir / f"{prefix}source_quantity_disagreement.png"
        ),
        "training_population_funnel": (
            figure_dir / f"{prefix}training_population_funnel.png"
        ),
        "normalized_uv_cutoff_sensitivity": (
            figure_dir / f"{prefix}normalized_uv_cutoff_sensitivity.png"
        ),
        "training_coverage_by_year": (
            figure_dir / f"{prefix}training_coverage_by_year.png"
        ),
        "model_performance_by_year": (
            figure_dir
            / f"{prefix}model_performance_by_year.png"
        ),
        "model_residual_correlation": (
            figure_dir
            / f"{prefix}model_residual_correlation.png"
        ),

        "model_ensemble_pair_diagnostic": (
            figure_dir
            / f"{prefix}model_ensemble_pair_diagnostic.png"
        ),
        "weighted_ensemble_curves": (
            figure_dir
            / f"{prefix}weighted_ensemble_curves.png"
        ),
    }


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
    Create the VACODA Validation Markdown model-development report.

    The report documents the canonical bilateral quantity methodology,
    descriptive source-reporter diagnostics, the configured HS × year UV
    training-sample rule, training coverage, and expanding-window model
    evaluation.  Production inference remains a later Validation stage.
    """

    prepared_input_path = Path(prepared_input_path)
    structural_audit_path = Path(structural_audit_path)
    diagnostics_path = Path(diagnostics_path)

    for label, path in (
        ("prepared Validation input", prepared_input_path),
        ("Validation structural audit", structural_audit_path),
        ("Validation diagnostics", diagnostics_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    if not cfg.artifacts.enable:
        return {
            "status": "disabled",
            "output_path": None,
            "report_path": None,
            "latest_report_path": None,
            "ready_for_training_decision": False,
        }

    report_path = cfg.artifacts_path(run_id)
    latest_path = cfg.artifacts_latest_path()

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

    structural_sql = sql_literal(
        structural_audit_path
    )
    diagnostics_sql = sql_literal(
        diagnostics_path
    )

    info = get_symbols("info")
    ok = get_symbols("check")

    logger.info(
        "%s Building Validation model-development report.",
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
        "  Diagnostics: %s",
        diagnostics_path,
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

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
                "Validation structural audit must contain exactly one row."
            )

        if diagnostics.empty:
            raise RuntimeError(
                "Validation diagnostics contain no rows."
            )

        required_diagnostic_columns = {
            "scope",
            "criterion",
            "n_rows",
            "share_of_all",
            "share_of_parent",
            "value",
        }

        missing = (
            required_diagnostic_columns
            - set(diagnostics.columns)
        )

        if missing:
            raise RuntimeError(
                "Validation diagnostics are missing required columns: "
                + ", ".join(sorted(missing))
            )

        # ============================================================== 
        # Structural integrity
        # ============================================================== 
        audit_row = structural_audit.iloc[0]

        if "audit_passed" not in structural_audit.columns:
            raise RuntimeError(
                "Validation structural audit is missing audit_passed."
            )

        audit_passed = bool(
            audit_row["audit_passed"]
        )

        structural_fields = (
            ("n_rows", "Rows"),
            ("n_duplicate_rows", "Duplicate rows"),
            ("n_null_key_rows", "Null canonical-key rows"),
            ("n_invalid_year_rows", "Invalid year rows"),
            ("n_invalid_hs_rows", "Invalid HS rows"),
            ("n_invalid_nominal_value_rows", "Invalid nominal-value rows"),
            (
                "n_invalid_constant_value_rows",
                "Invalid constant-USD value rows",
            ),
            (
                "n_invalid_value_weighted_rows",
                "Invalid value-weighted quantity rows",
            ),
            ("n_missing_quantity_rows", "Rows without quantity"),
            ("n_available_quantity_rows", "Rows with quantity"),
        )

        structural_rows = []

        for column, label in structural_fields:
            if column in structural_audit.columns:
                structural_rows.append(
                    {
                        "Check": label,
                        "Result": _format_integer(
                            audit_row[column]
                        ),
                    }
                )

        structural_rows.append(
            {
                "Check": "Structural audit passed",
                "Result": str(audit_passed),
            }
        )

        structural_summary = pd.DataFrame(
            structural_rows
        )

        # ============================================================== 
        # Canonical quantity methodology
        # ============================================================== 
        methodology = _require_diagnostic_scope(
            diagnostics,
            "quantity_methodology",
        )

        selected_candidates = (
            methodology["candidate"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(selected_candidates) != 1:
            raise RuntimeError(
                "Validation report requires exactly one canonical "
                "quantity methodology."
            )

        selected_candidate = str(
            selected_candidates[0]
        )
        selected_candidate_label = _candidate_label(
            selected_candidate
        )

        # ============================================================== 
        # Source diagnostics
        # ============================================================== 
        source_availability = _require_diagnostic_scope(
            diagnostics,
            "source_quantity_availability",
        )

        source_labels = {
            "both": "Both exporter and importer quantities available",
            "export_only": "Exporter quantity only",
            "import_only": "Importer quantity only",
            "neither": "Neither source quantity available",
            "both_non_estimated": "Both source quantities non-estimated",
        }

        source_availability_display = pd.DataFrame(
            [
                {
                    "Source evidence": source_labels.get(
                        str(row.criterion),
                        str(row.criterion),
                    ),
                    "Rows": _format_integer(row.n_rows),
                    "Share of all rows": _format_percent(
                        row.share_of_all
                    ),
                }
                for row in source_availability.itertuples(
                    index=False
                )
            ]
        )

        source_disagreement = _require_diagnostic_scope(
            diagnostics,
            "source_quantity_disagreement_distribution",
        ).sort_values("quantile")

        source_disagreement_display = pd.DataFrame(
            {
                "Quantile": [
                    f"P{100.0 * float(value):g}"
                    for value in source_disagreement["quantile"]
                ],
                "Relative exporter/importer disagreement": [
                    _format_percent(value)
                    for value in source_disagreement["value"]
                ],
            }
        )

        # ============================================================== 
        # Training rule and coverage
        # ============================================================== 
        training_funnel = _require_diagnostic_scope(
            diagnostics,
            "training_population_funnel",
        )

        funnel_labels = {
            "all_rows": "All canonical rows",
            "training_eligible": "Technically usable quantity/value",
            "support": "HS×year UV support threshold satisfied",
            "positive_uv_mad": "Positive finite HS×year UV MAD",
            "uv_distance_available": "Normalized UV distance evaluable",
            "training_selected": "Final training sample",
        }

        training_funnel_display = pd.DataFrame(
            [
                {
                    "Stage": funnel_labels.get(
                        str(row.criterion),
                        str(row.criterion),
                    ),
                    "Rows": _format_integer(row.n_rows),
                    "Share of all rows": _format_percent(
                        row.share_of_all
                    ),
                    "Retention from previous stage": _format_percent(
                        row.share_of_parent
                    ),
                }
                for row in training_funnel.itertuples(
                    index=False
                )
            ]
        )

        training_definition = _require_diagnostic_scope(
            diagnostics,
            "training_definition",
        )

        if len(training_definition) != 1:
            raise RuntimeError(
                "Validation diagnostics must contain exactly one "
                "training_definition row."
            )

        training_definition_row = training_definition.iloc[0]
        min_reference_observations = int(
            round(
                float(
                    training_definition_row[
                        "uv_support_threshold"
                    ]
                )
            )
        )
        max_normalized_uv_distance_mad = float(
            training_definition_row[
                "uv_distance_threshold"
            ]
        )
        n_training_selected = int(
            training_definition_row["n_rows"]
        )
        training_selected_share = float(
            training_definition_row["share_of_all"]
        )

        training_rule_display = pd.DataFrame(
            [
                {
                    "Criterion": "Minimum HS×year UV reference support",
                    "Configured value": (
                        f"≥ {min_reference_observations:,} observations"
                    ),
                },
                {
                    "Criterion": "HS×year log-UV dispersion",
                    "Configured value": "> 0 MAD",
                },
                {
                    "Criterion": "Maximum normalized UV distance",
                    "Configured value": (
                        f"≤ {max_normalized_uv_distance_mad:g} MAD"
                    ),
                },
            ]
        )

        training_by_year = _require_diagnostic_scope(
            diagnostics,
            "training_coverage_by_year",
        ).sort_values("year")

        training_year_display = pd.DataFrame(
            {
                "Year": training_by_year["year"].astype(int),
                "Training rows": [
                    _format_integer(value)
                    for value in training_by_year["n_rows"]
                ],
                "Share of annual rows": [
                    _format_percent(value)
                    for value in training_by_year["share_of_all"]
                ],
                "Share of technically eligible rows": [
                    _format_percent(value)
                    for value in training_by_year["share_of_parent"]
                ],
                "Products represented": [
                    _format_integer(value)
                    for value in training_by_year["value"]
                ],
            }
        )

        training_by_product = _require_diagnostic_scope(
            diagnostics,
            "training_coverage_by_product",
        )

        n_products = int(
            training_by_product["target_cmd"]
            .dropna()
            .astype(str)
            .nunique()
        )
        n_unseen_products = int(
            training_by_product["criterion"]
            .eq("unseen")
            .sum()
        )

        product_display = (
            training_by_product
            .sort_values(
                ["n_rows", "target_cmd"],
                ascending=[True, True],
            )
            .head(20)
            .copy()
        )

        product_display = pd.DataFrame(
            {
                "HS product": product_display["target_cmd"].astype(str),
                "Training rows": [
                    _format_integer(value)
                    for value in product_display["n_rows"]
                ],
                "Technically eligible rows": [
                    _format_integer(value)
                    for value in product_display["eligible_rows"]
                ],
                "Training share": [
                    _format_percent(value)
                    for value in product_display["share_of_all"]
                ],
                "Training-year coverage": [
                    _format_percent(value)
                    for value in product_display[
                        "training_year_coverage"
                    ]
                ],
            }
        )

        rejection = _scope_frame(
            diagnostics,
            "training_rejection_by_product",
        )

        if rejection.empty:
            rejection_display = pd.DataFrame()
        else:
            rejection_display = (
                rejection.pivot_table(
                    index="target_cmd",
                    columns="criterion",
                    values="n_rows",
                    aggfunc="first",
                )
                .reset_index()
            )

            rejection_columns = [
                column
                for column in (
                    "target_cmd",
                    "training_eligible",
                    "reference_support",
                    "positive_uv_mad",
                    "uv_distance_available",
                    "normalized_uv_distance",
                )
                if column in rejection_display.columns
            ]

            rejection_display = rejection_display[
                rejection_columns
            ].rename(
                columns={
                    "target_cmd": "HS product",
                    "training_eligible": "Eligible",
                    "reference_support": "After support",
                    "positive_uv_mad": "After positive MAD",
                    "uv_distance_available": "UV evaluable",
                    "normalized_uv_distance": "After ≤ cutoff",
                }
            )


        ## Model weighted ensembling
        weighted_best = _scope_frame(
            diagnostics,
            "model_weighted_ensemble_best",
        ).copy()

        # ============================================================== 
        # Reference-window and UV evidence
        # ============================================================== 
        reference_window = _require_diagnostic_scope(
            diagnostics,
            "reference_window_summary",
        )

        support_rows = reference_window[
            reference_window["variable"].eq(
                "groups_with_support"
            )
        ].copy()
        mad_rows = reference_window[
            reference_window["variable"].eq(
                "median_log_mad"
            )
        ].copy()

        reference_display_rows = []

        for criterion in (
            "0_0",
            "1_0",
            "2_0",
        ):
            support = support_rows[
                support_rows["criterion"].eq(criterion)
            ]
            mad = mad_rows[
                mad_rows["criterion"].eq(criterion)
            ]

            if support.empty:
                continue

            support_row = support.iloc[0]
            mad_row = (
                mad.iloc[0]
                if not mad.empty
                else None
            )

            reference_display_rows.append(
                {
                    "Window": criterion,
                    "Groups meeting support": _format_integer(
                        support_row["n_rows"]
                    ),
                    "Share of HS×year groups": _format_percent(
                        support_row["share_of_parent"]
                    ),
                    "Median reference support": _format_number(
                        support_row["value"],
                        digits=1,
                    ),
                    "Median log-UV MAD": _format_number(
                        mad_row["value"]
                        if mad_row is not None
                        else None,
                        digits=4,
                    ),
                }
            )

        reference_window_display = pd.DataFrame(
            reference_display_rows
        )

        support_distribution = _require_diagnostic_scope(
            diagnostics,
            "training_reference_support_distribution",
        ).sort_values("quantile")

        support_distribution_display = pd.DataFrame(
            {
                "Quantile": [
                    f"P{100.0 * float(value):g}"
                    for value in support_distribution["quantile"]
                ],
                "HS×year reference observations": [
                    _format_number(value, digits=1)
                    for value in support_distribution["value"]
                ],
            }
        )

        mad_distribution = _require_diagnostic_scope(
            diagnostics,
            "training_uv_mad_distribution",
        ).sort_values("quantile")

        normalized_distribution = _require_diagnostic_scope(
            diagnostics,
            "training_normalized_uv_distance_distribution",
        ).sort_values("quantile")

        uv_distribution_display = pd.DataFrame(
            {
                "Quantile": [
                    f"P{100.0 * float(value):g}"
                    for value in normalized_distribution["quantile"]
                ],
                "Normalized UV distance (MAD)": [
                    _format_number(value, digits=3)
                    for value in normalized_distribution["value"]
                ],
                "HS×year log-UV MAD": [
                    _format_number(value, digits=4)
                    for value in mad_distribution["value"]
                ],
            }
        )

        cutoff_sensitivity = _require_diagnostic_scope(
            diagnostics,
            "training_normalized_uv_cutoff_sensitivity",
        ).sort_values("uv_distance_threshold")

        cutoff_sensitivity_display = pd.DataFrame(
            {
                "Maximum normalized UV distance": [
                    f"≤ {float(value):g} MAD"
                    for value in cutoff_sensitivity[
                        "uv_distance_threshold"
                    ]
                ],
                "Rows retained": [
                    _format_integer(value)
                    for value in cutoff_sensitivity["n_rows"]
                ],
                "Share of UV-evaluable rows": [
                    _format_percent(value)
                    for value in cutoff_sensitivity[
                        "share_of_parent"
                    ]
                ],
                "Share of all Validation rows": [
                    _format_percent(value)
                    for value in cutoff_sensitivity[
                        "share_of_all"
                    ]
                ],
            }
        )

        mad_distribution_display = pd.DataFrame(
            {
                "Quantile": [
                    f"P{100.0 * float(value):g}"
                    for value in mad_distribution["quantile"]
                ],
                "HS×year log-UV MAD": [
                    _format_number(value, digits=4)
                    for value in mad_distribution["value"]
                ],
            }
        )
        # ============================================================== 
        # Model evaluation
        # ============================================================== 
        model_overall = _require_diagnostic_scope(
            diagnostics,
            "model_performance_overall",
        )

        model_pivot = model_overall.pivot_table(
            index="model_name",
            columns="variable",
            values="value",
            aggfunc="first",
        ).reset_index()

        for required_metric in (
            "mae_log",
            "rmse_log",
            "median_absolute_log_error",
        ):
            if required_metric not in model_pivot.columns:
                model_pivot[required_metric] = math.nan

        model_performance_display = pd.DataFrame(
            {
                "Model": model_pivot["model_name"].astype(str),
                "MAE log": model_pivot["mae_log"],
                "Median absolute log error": (
                    model_pivot["median_absolute_log_error"]
                ),
                "RMSE log": model_pivot["rmse_log"],
            }
        ).sort_values("MAE log")

        # --------------------------------------------------------------
        # Residual-correlation diagnostic
        # --------------------------------------------------------------

        model_residual_correlation = _require_diagnostic_scope(
            diagnostics,
            "model_residual_correlation",
        )

        paired_oof_rows = pd.to_numeric(
            model_residual_correlation["n_rows"],
            errors="coerce",
        ).dropna().unique()

        model_residual_correlation_display = pd.DataFrame(
            {
                "Model A": (
                    model_residual_correlation[
                        "model_name"
                    ].map(_model_plot_label).astype(str)
                ),
                "Model B": (
                    model_residual_correlation[
                        "candidate"
                    ].map(_model_plot_label).astype(str)
                ),
                "Residual correlation": pd.to_numeric(
                    model_residual_correlation[
                        "value"
                    ],
                    errors="coerce",
                ),
            }
        ).sort_values(
            [
                "Residual correlation",
                "Model A",
                "Model B",
            ]
        )


        # --------------------------------------------------------------
        # Ensemble-pair diagnostic
        # --------------------------------------------------------------

        ensemble_pair = _require_diagnostic_scope(
            diagnostics,
            "model_ensemble_pair_diagnostic",
        )

        ensemble_pair_pivot = ensemble_pair.pivot_table(
            index=[
                "model_name",
                "candidate",
            ],
            columns="variable",
            values="value",
            aggfunc="first",
        ).reset_index()

        for required_metric in (
            "residual_correlation",
            "model_mae_log",
            "candidate_mae_log",
            "mae_difference",
            "mean_pair_mae_log",):
            if required_metric not in ensemble_pair_pivot.columns:
                ensemble_pair_pivot[
                    required_metric
                ] = math.nan


        ensemble_pair_display = pd.DataFrame(
            {
                "Model A": (
                    ensemble_pair_pivot[
                        "model_name"
                    ].map(_model_plot_label).astype(str)
                ),
                "Model B": (
                    ensemble_pair_pivot[
                        "candidate"
                    ].map(_model_plot_label).astype(str)
                ),
                "MAE A": (
                    ensemble_pair_pivot[
                        "model_mae_log"
                    ]
                ),
                "MAE B": (
                    ensemble_pair_pivot[
                        "candidate_mae_log"
                    ]
                ),
                "Residual correlation": (
                    ensemble_pair_pivot[
                        "residual_correlation"
                    ]
                ),
                "MAE difference": (
                    ensemble_pair_pivot[
                        "mae_difference"
                    ]
                ),
                "Mean pair MAE": (
                    ensemble_pair_pivot[
                        "mean_pair_mae_log"
                    ]
                ),
            }
        ).sort_values(
            [
                "Residual correlation",
                "Mean pair MAE",
            ]
        )

        uv_distance_distribution = _require_diagnostic_scope(
            diagnostics,
            "training_uv_distance_distribution",
        ).sort_values("quantile")

        uv_distance_display = pd.DataFrame(
            {
                "Quantile": [
                    f"P{100.0 * float(value):g}"
                    for value in uv_distance_distribution["quantile"]
                ],
                "Absolute UV log distance": [
                    _format_number(value, digits=4)
                    for value in uv_distance_distribution["value"]
                ],
                "Equivalent factor": [
                    _format_number(
                        math.exp(float(value)),
                        digits=2,
                    )
                    for value in uv_distance_distribution["value"]
                ],
            }
        )

        normalized_uv_distance_display = pd.DataFrame(
            {
                "Quantile": [
                    f"P{100.0 * float(value):g}"
                    for value in normalized_distribution["quantile"]
                ],
                "Normalized UV distance (MAD)": [
                    _format_number(value, digits=3)
                    for value in normalized_distribution["value"]
                ],
            }
        )


        ## Model weighted ensembling
        # --------------------------------------------------------------
        # Weighted-ensemble best results
        # --------------------------------------------------------------

        weighted_ensemble_best = _require_diagnostic_scope(
            diagnostics,
            "model_weighted_ensemble_best",
        )

        weighted_ensemble_best_pivot = (
            weighted_ensemble_best.pivot_table(
                index=[
                    "model_name",
                    "candidate",
                ],
                columns="variable",
                values="value",
                aggfunc="first",
            )
            .reset_index()
        )

        for required_metric in (
            "best_weight_model_a",
            "best_ensemble_mae_log",
            "model_a_mae_log",
            "model_b_mae_log",
            "better_single_mae_log",
            "improvement_vs_best_single",
        ):
            if (
                required_metric
                not in weighted_ensemble_best_pivot.columns
            ):
                weighted_ensemble_best_pivot[
                    required_metric
                ] = math.nan

        weighted_ensemble_best_display = pd.DataFrame(
            {
                "Model A": (
                    weighted_ensemble_best_pivot[
                        "model_name"
                    ].astype(str)
                ),

                "Model B": (
                    weighted_ensemble_best_pivot[
                        "candidate"
                    ].astype(str)
                ),

                "Best weight on A": (
                    weighted_ensemble_best_pivot[
                        "best_weight_model_a"
                    ]
                ),

                "Weight on B": (
                    1.0
                    - pd.to_numeric(
                        weighted_ensemble_best_pivot[
                            "best_weight_model_a"
                        ],
                        errors="coerce",
                    )
                ),

                "Best ensemble MAE": (
                    weighted_ensemble_best_pivot[
                        "best_ensemble_mae_log"
                    ]
                ),

                "Best single-model MAE": (
                    weighted_ensemble_best_pivot[
                        "better_single_mae_log"
                    ]
                ),

                "MAE improvement": (
                    weighted_ensemble_best_pivot[
                        "improvement_vs_best_single"
                    ]
                ),
            }
        ).sort_values(
            "Best ensemble MAE"
        )

        # ============================================================== 
        # Figures
        # ============================================================== 
        figure_paths = _build_validation_figure_paths(
            figure_dir=figure_dir,
            target_hs=cfg.scope.target_hs,
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

        _plot_training_uv_mad_distribution(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_uv_mad_distribution"
            ],
        )

        _plot_training_normalized_uv_distance_distribution(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_normalized_uv_distance_distribution"
            ],
        )

        _plot_reference_window_comparison(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "reference_window_comparison"
            ],
        )
        _plot_training_coverage_by_product(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_coverage_by_product"
            ],
        )
        _plot_source_quantity_availability(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "source_quantity_availability"
            ],
        )
        _plot_source_disagreement_bands(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "source_quantity_disagreement"
            ],
        )
        _plot_training_population_funnel(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_population_funnel"
            ],
        )
        _plot_normalized_uv_cutoff_sensitivity(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "normalized_uv_cutoff_sensitivity"
            ],
            configured_cutoff=max_normalized_uv_distance_mad,
        )
        _plot_training_coverage_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_coverage_by_year"
            ],
        )
        _plot_model_performance_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "model_performance_by_year"
            ],
        )

        _plot_model_residual_correlation(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "model_residual_correlation"
            ],
        )

        _plot_model_ensemble_pair_diagnostic(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "model_ensemble_pair_diagnostic"
            ],
        )

        _plot_weighted_ensemble_curves(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "weighted_ensemble_curves"
            ],
        )

        # ============================================================== 
        # Report narrative
        # ============================================================== 
        training_cfg = cfg.model.training_sample
        validation_cfg = cfg.model.validation

        enabled_model_names = list(
            cfg.model.models.enabled().keys()
        )
        reference_before = int(
            training_cfg.reference_year_window.before
        )

        reference_after = int(
            training_cfg.reference_year_window.after
        )

        min_reference_n = int(
            training_cfg.min_hs_year_reference_observations
        )

        max_uv_distance = float(
            training_cfg.max_normalized_uv_distance_mad
        )

        validation_strategy = str(
            validation_cfg.strategy
        )
        minimum_training_years = int(
            validation_cfg.minimum_training_years
        )
        configured_years = list(
            range(
                int(cfg.scope.min_year),
                int(cfg.scope.max_year) + 1,
            )
        )

        if validation_strategy == "expanding_window":
            configured_validation_years = (
                configured_years[
                    minimum_training_years:
                ]
            )
            validation_description = (
                "Candidate models are evaluated using expanding-window "
                "validation. Each validation year is predicted using only "
                "earlier training years, with at least "
                f"{minimum_training_years} preceding training years required "
                "before the first validation fold."
            )
            performance_image_alt = (
                "Expanding-window model performance"
            )
        else:
            configured_validation_years = []
            validation_description = (
                "Candidate models are evaluated using the configured "
                f"`{validation_strategy}` validation strategy."
            )
            performance_image_alt = (
                "Model performance by validation year"
            )

        lines = [
            (
                "# VACODA Validation Model-Development Report — "
                f"{str(cfg.scope.target_hs).upper()}"
            ),
            "",
            "",
            f"- **Run ID:** `{run_id}`",
            f"- **Target HS:** `{cfg.scope.target_hs.upper()}`",
            (
                f"- **Configured year range:** "
                f"{cfg.scope.min_year}–{cfg.scope.max_year}"
            ),
                "- **Training Sample Criteria:**",
                (
                    f"  - Reference-year window: `before={reference_before}, "
                    f"after={reference_after}`"
                ),
                (
                    f"  - Minimum HS × year reference observations: "
                    f"`{min_reference_n}`"
                ),
                (
                    f"  - Maximum normalized UV distance: "
                    f"`{max_uv_distance:g} MAD`"
                ),
            "- **Validation Criteria:**",
            (
                    f"  - Validation Strategy: `{validation_strategy}`"
            ),
            (
                    f"  - Minimum training years: `{minimum_training_years}`"
            ),
            (
                "  - Validation years: "
                f"`{', '.join(str(year) for year in configured_validation_years)}`"
            ),
            "- **Enabled candidate models:**",
            *[
                f"  - `{name}`"
                for name in enabled_model_names
            ],
            "- **Artifacts:**",
                f"  - Prepared input: `{prepared_input_path}`",
                f"  - Structural audit: `{structural_audit_path}`",
                f"  - Diagnostics: `{diagnostics_path}`",
                f"  - Figure directory: `{figure_dir}`",
                f"  - Report: `{report_path}`",
            "",
            "__TOC__",
            "",
            "",
            "",
            "## 1. Structural integrity",
            "",
            _markdown_table(structural_summary),
            "",
            "## 2. Canonical bilateral quantity methodology",
            "",
            (
                "The quantity-selection stage has fixed one canonical bilateral "
                "quantity methodology for downstream Validation. The current "
                f"methodology is `{selected_candidate_label}`."
            ),
            "",
            (
                "All subsequent unit-value, training-sample, and model-evaluation "
                "diagnostics refer to this canonical selected quantity rather than "
                "treating exporter or importer quantities as separate truth labels."
            ),
            "",
            "## 3. Source-reporter diagnostics",
            "",
            (
                "The following diagnostics describe the quantity evidence supplied "
                "by the two reporting sides before reconciliation. They are retained "
                "to characterize source-data quality and provenance only. **No "
                "reporter-availability or mirror-disagreement threshold is used in "
                "the model-training sample.**"
            ),
            "",
            "### 3.1. Quantity availability by reporting side",
            "",
            get_table_figure_layout(
                source_availability_display,
                markdown_image(
                    "Source-reporter quantity availability",
                    figure_paths["source_quantity_availability"],
                    document_path=report_path,
                    width=800,
                ),
            ),
            "",
            "### 3.2. Exporter/importer disagreement",
            "",
            (
                "Where both source quantities are available, the distribution below "
                "shows how strongly they disagree. This is a diagnostic of the raw "
                "reporting evidence, not a training-selection criterion."
            ),
            "",
            get_table_figure_layout(
                source_disagreement_display,
                markdown_image(
                    "Exporter/importer quantity disagreement bands",
                    figure_paths["source_quantity_disagreement"],
                    document_path=report_path,
                    width=800,
                ),
            ),
            "",
            markdown_image(
                "Exporter/importer quantity disagreement bands",
                figure_paths["source_quantity_disagreement"],
                document_path=report_path,
                width=800,
            ),
            "",
            "## 4. HS × year unit-value evidence",
            "",
            (
                "For technically usable canonical bilateral observations, Validation "
                "forms a robust HS × focal-year unit-value reference in log space. "
                "The configured reference window pools the focal year with preceding "
                "years only; the reference centre is the median log unit value and "
                "dispersion is the median absolute deviation (MAD)."
            ),
            "",

            "### 4.1. Reference-window comparison",
            "",
            (
                "The table compares same-year, one-year-back, and two-year-back "
                "causal reference windows."
            ),
            "",
            get_table_figure_layout(
                reference_window_display,
                markdown_image(
                    "HS × year reference support by temporal window",
                    figure_paths["reference_window_comparison"],
                    document_path=report_path,
                    #width=800,
                ),
            ),
            "",

            "### 4.2. HS × year reference support",
            "",
            (
                "Reference support is the number of canonical eligible observations "
                "contributing to each HS × year unit-value reference."
            ),
            "",
            get_table_figure_layout(
                support_distribution_display,
                markdown_image(
                    "HS × year unit-value reference support",
                    figure_paths["training_reference_support_distribution"],
                    document_path=report_path,
                    #width=800,
                ),
            ),
            "",

            "### 4.3. HS × year unit-value dispersion",
            "",
            (
                "The log-UV MAD describes the natural within-group dispersion around "
                "the robust HS × year reference."
            ),
            "",
            get_table_figure_layout(
                mad_distribution_display,
                markdown_image(
                    "HS × year unit-value dispersion",
                    figure_paths["training_uv_dispersion_distribution"],
                    document_path=report_path,
                    #width=800,
                ),
            ),
            "",
            "### 4.4. Canonical unit-value distance",
            "",
            (
                "For each technically eligible canonical observation, Validation "
                "measures the absolute log-distance between its implied unit value "
                "and the corresponding HS × year reference."
            ),
            "",
            get_table_figure_layout(
                uv_distance_display,
                markdown_image(
                    "Canonical unit-value distance distribution",
                    figure_paths["training_uv_distance_distribution"],
                    document_path=report_path,
                    #width=800,
                ),
            ),
            "",

            "### 4.5. Dispersion-normalized unit-value distance",
            "",
            (
                "Because natural unit-value dispersion differs across HS × year "
                "groups, the absolute log-distance is also divided by the group MAD. "
                "A value of 1 means that the observation lies one within-group MAD "
                "from the reference."
            ),
            "",
            get_table_figure_layout(
                normalized_uv_distance_display,
                markdown_image(
                    "Unit-value distance relative to HS × year dispersion",
                    figure_paths["training_normalized_uv_distance_distribution"],
                    document_path=report_path,
                    #width=800,
                ),
            ),
            "",
            "## 5. Training sample selection and coverage",
            "",
            (
                "The model-training sample is defined only from canonical bilateral "
                "unit-value evidence. Reporter-side quantity agreement is not part "
                "of this rule."
            ),
            "",
            "### 5.1. Configured training rule",
            "",
            _markdown_table(training_rule_display),
            "",
            (
                f"The resulting training sample contains **{_format_integer(n_training_selected)}** "
                f"rows, or **{_format_percent(training_selected_share)}** of all "
                "Validation rows."
            ),
            "",
            "### 5.2. Population funnel",
            "",
            get_table_figure_layout(training_funnel_display,
                markdown_image(
                    "Training-sample population funnel",
                    figure_paths["training_population_funnel"],
                    document_path=report_path,
                    width=850,
                ),
            ),
            "",
            "### 5.3. Normalized-UV cutoff sensitivity",
            "",
            (
                "The cutoff sensitivity is retained to show how the final tail "
                "restriction changes sample size. It does not reopen reporter-side "
                "truth assumptions."
            ),
            "",
            get_table_figure_layout(
                cutoff_sensitivity_display,
                markdown_image(
                    "Normalized unit-value cutoff sensitivity",
                    figure_paths["normalized_uv_cutoff_sensitivity"],
                    document_path=report_path,
                    width=800,
                ),
            ),
            "",
            "### 5.4. Training coverage by year",
            "",
            get_table_figure_layout(
                training_year_display,
                markdown_image(
                    "Final training-sample coverage by year",
                    figure_paths["training_coverage_by_year"],
                    document_path=report_path,
                    width=850,
                ),
            ),
            "",
            "### 5.5. Product coverage",
            "",
            (
                f"The Validation population contains **{_format_integer(n_products)}** "
                f"HS products; **{_format_integer(n_unseen_products)}** have no "
                "training-selected observations under the configured UV-based rule."
            ),
            "",
            (
                "The table below shows the 20 products with the smallest final "
                "training populations."
            ),
            "",
            get_table_figure_layout(
                product_display,
                markdown_image(
                    "Training-sample coverage across HS products",
                    figure_paths["training_coverage_by_product"],
                    document_path=report_path,
                    width=850,
                ),
            ),
            "",
            "### 5.6. Rejection path for products unseen in training",
            "",
            (
                "For products with no final training observations, the cumulative "
                "table shows whether rows disappear because of insufficient "
                "reference support, non-positive/undefined MAD, unavailable UV "
                "distance, or the configured normalized-UV tail cutoff."
            ),
            "",
            _markdown_table(rejection_display),
            "",
            "## 6. Model evaluation",
            "",
            validation_description,
            "",
            "### 6.1. Overall out-of-fold performance",
            "",
            _markdown_table(model_performance_display),
            "",
            "### 6.2. Performance by validation year",
            "",
            markdown_image(
                performance_image_alt,
                figure_paths["model_performance_by_year"],
                document_path=report_path,
                width=850,
            ),
            "",

            "### 6.3. Residual correlation between candidate models",
            "",
            (
                "Ensemble usefulness depends not only on standalone predictive "
                "accuracy but also on whether candidate models make different "
                "out-of-fold errors. The table below reports the Pearson "
                "correlation between signed log-scale residuals for identical "
                "held-out canonical observations."
            ),
            "",
            (
                "Lower residual correlation indicates greater predictive "
                "diversity. High correlation indicates that two models tend to "
                "over- or under-predict the same observations in similar ways."
            ),
            "",
            (
                "Paired OOF rows: "
                f"{_format_integer(int(paired_oof_rows[0]))}"
                if len(paired_oof_rows) == 1
                else "Paired OOF rows: varies across model pairs"
            ),
            "",
            get_table_figure_layout(
                model_residual_correlation_display,
                markdown_image(
                    "Correlation of out-of-fold model residuals",
                    figure_paths[
                        "model_residual_correlation"
                    ],
                    document_path=report_path,
                    width=800,
                ),
            ),
            "",

            "### 6.4. Ensemble-pair diagnostic",
            "",
            (
                "Residual diversity must be interpreted together with standalone "
                "model quality. A weak model is not automatically a useful "
                "ensemble member merely because its residuals are weakly "
                "correlated with those of another model."
            ),
            "",
            (
                "The following table therefore combines paired out-of-fold MAE "
                "with residual correlation. Promising ensemble pairs are those "
                "that combine comparatively low MAE with comparatively low "
                "residual correlation."
            ),
            "",
            get_table_figure_layout(
                ensemble_pair_display,
                markdown_image(
                    "Candidate ensemble pairs: quality versus residual diversity",
                    figure_paths[
                        "model_ensemble_pair_diagnostic"
                    ],
                    document_path=report_path,
                    width=800,
                ),
            ),
            "",

            "### 6.5. Weighted ensemble performance",
            "",
            (
                "The candidate pairs above are evaluated as weighted "
                "two-model ensembles in log-prediction space. For each "
                "pair, Model A receives weight `w` and Model B receives "
                "weight `1 - w`. Performance is evaluated directly on "
                "the same out-of-fold held-out observations used for "
                "single-model evaluation."
            ),
            "",
            (
                "- The summary below reports the best tested weighting "
                "for each model pair. A positive MAE improvement means "
                "that the weighted ensemble outperforms the stronger "
                "of its two constituent models."
            ),
            "",
            _markdown_table(
                weighted_ensemble_best_display
            ),
            "",
            (
                "- The figure shows how out-of-fold MAE changes as the "
                "weight assigned to Model A varies from 0 to 1. "
                "Endpoints therefore correspond to the standalone "
                "models, while interior points represent blended "
                "predictions. A curve dipping below both endpoints "
                "provides direct evidence that combining the models "
                "improves predictive accuracy."
            ),
            "",
            markdown_image(
                "Weighted ensemble performance curves",
                figure_paths[
                    "weighted_ensemble_curves"
                ],
                document_path=report_path,
                width=850,
            ),
            "",
            "## 7. Current methodological status",
            "",
            (
                f"- Canonical bilateral quantity methodology: **{selected_candidate_label}**."
            ),
            (
                "- Training sample: fixed by HS×year support, positive MAD, and "
                f"normalized UV distance ≤ {max_normalized_uv_distance_mad:g} MAD."
            ),
            (
                "- Exporter/importer disagreement is retained only as descriptive "
                "source-data evidence and does not determine training eligibility."
            ),
            (
                f"- Model validation strategy: `{validation_strategy}`."
            ),
            (
                "- Production inference remains a subsequent Validation stage."
            ),
            "",
            "## 8. Inference",
            "",
            (
                "Production inference is not yet reported by the current Validation "
                "workflow. This section is reserved for the final selected model, "
                "the canonical bilateral rows requiring reconstruction, prediction "
                "coverage, post-inference unit-value plausibility, and the completed "
                "quantity output."
            ),
            "",
        ]

        insert_markdown_toc(
            lines,
            min_level=2,
            max_level=4,
            flat_sublevels=True,
        )

        report_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        if not valid_output(report_path):
            raise RuntimeError(
                "Validation report was not written correctly."
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
            latest_report_path = latest_path

        logger.info(
            "%s Validation model-development report created.",
            ok,
        )
        logger.info(
            "  Structural audit passed: %s",
            audit_passed,
        )
        logger.info(
            "  Selected training rows: %s",
            f"{n_training_selected:,}",
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
            len(figure_paths),
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "output_path": report_path,
            "report_path": report_path,
            "latest_report_path": latest_report_path,
            "figure_paths": figure_paths,
            "audit_passed": audit_passed,
            "training_population_selected": True,
            "ready_for_training_decision": True,
            "duration_seconds": None,
        }

    finally:
        con.close()
