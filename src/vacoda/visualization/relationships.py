"""
vacoda.visualization.relationships

Two-variable (2-D) relationship plots.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from vacoda.cif_fob.display import (
    short_algorithm_name,
    short_predictor_set_name,
    split_estimation_run_name,
    )
from typing import Iterable, Sequence
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

def _metric_is_percentage(metric: str) -> bool:
    lowered = metric.lower()
    return (
        "ape" in lowered
        or "share" in lowered
        or "percent" in lowered
    )


def _display_metric_values(
    frame: pd.DataFrame,
    *,
    metric: str,
) -> pd.Series:
    values = pd.to_numeric(
        frame[metric],
        errors="coerce",
    )

    if _metric_is_percentage(metric):
        return values * 100.0

    return values


def _metric_axis_label(
    metric: str,
    *,
    label: str | None = None,
) -> str:
    if label is None:
        label = metric

    if _metric_is_percentage(metric):
        return f"{label} (%)"

    return label


def plot_observed_vs_predicted_with_tolerance_bands(
    frame: pd.DataFrame,
    *,
    observed_column: str,
    predicted_column: str,
    tolerance_bands: Sequence[float] = (0.05, 0.10, 0.20),
    title: str = (
        "Primary model: observed vs predicted CIF/FOB factor "
        "with tolerance bands"
    ),
    gridsize: int = 45,
    min_count: int = 3,
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.995,
    display_lower: float | None = None,
    display_upper: float | None = None,
    show_density: bool = True,
    show_band_coverage: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot observed versus predicted values with percentage-tolerance bands.

    A prediction lies inside a p-percent band when:

        observed * (1 - p) <= predicted <= observed * (1 + p)

    Quantile limits affect only the displayed range. Coverage percentages
    are calculated using all valid observations.
    """
    required_columns = {
        observed_column,
        predicted_column,
    }
    missing_columns = required_columns.difference(frame.columns)

    if missing_columns:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if not tolerance_bands:
        raise ValueError(
            "At least one tolerance band is required."
        )

    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError(
            "Quantiles must satisfy "
            "0 <= lower_quantile < upper_quantile <= 1."
        )

    bands = sorted(
        {float(value) for value in tolerance_bands},
        reverse=True,
    )

    if any(value <= 0 or value >= 1 for value in bands):
        raise ValueError(
            "Tolerance bands must be greater than 0 and less than 1."
        )

    # Convert columns before using them.
    observed = pd.to_numeric(
        frame[observed_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    predicted = pd.to_numeric(
        frame[predicted_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(observed)
        & np.isfinite(predicted)
        & (observed > 0)
        & (predicted > 0)
    )

    observed = observed[valid]
    predicted = predicted[valid]

    if observed.size == 0:
        raise ValueError(
            "No finite positive observed/predicted pairs are available."
        )

    # Use the combined distribution so both axes have identical limits.
    combined = np.concatenate(
        [observed, predicted]
    )

    quantile_lower = float(
        np.quantile(combined, lower_quantile)
    )

    quantile_upper = float(
        np.quantile(combined, upper_quantile)
    )

    lower_limit = (
        float(display_lower)
        if display_lower is not None
        else quantile_lower
    )

    upper_limit = (
        float(display_upper)
        if display_upper is not None
        else quantile_upper
    )

    if lower_limit >= upper_limit:
        raise ValueError(
            "Invalid plot limits: lower limit must be below upper limit."
        )

    displayed = (
        (observed >= lower_limit)
        & (observed <= upper_limit)
        & (predicted >= lower_limit)
        & (predicted <= upper_limit)
    )

    observed_display = observed[displayed]
    predicted_display = predicted[displayed]

    if observed_display.size == 0:
        raise ValueError(
            "No observations remain within the requested display range."
        )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(8.5, 7.0)
        )
    else:
        fig = ax.figure

    x_reference = np.linspace(
        lower_limit,
        upper_limit,
        500,
    )

    band_styles = {
            0.20: {
                "facecolor": "#c9dfef",
                "edgecolor": "#4f9ed1",
                "alpha": 0.40,
                "linestyle": ":",
            },
            0.10: {
                "facecolor": "#f6d6bd",
                "edgecolor": "#e9892d",
                "alpha": 0.40,
                "linestyle": "--",
            },
            0.05: {
                "facecolor": "#c6dfc1",
                "edgecolor": "#4e9a4a",
                "alpha": 0.40,
                "linestyle": "-",
            },
            0.02: {
                "facecolor": "#eadcf2",
                "edgecolor": "#8f63a8",
                "alpha": 0.40,
                "linestyle": "-.",
            },
        }
    # Draw widest band first.
    for tolerance in bands:
        lower_band = (
            x_reference * (1.0 - tolerance)
        )
        upper_band = (
            x_reference * (1.0 + tolerance)
        )

        coverage = float(
            np.mean(
                np.abs(predicted - observed)
                / observed
                <= tolerance
            )
        )

        style = band_styles.get(
            tolerance,
            {
                "facecolor": "#bdd7e7",
                "edgecolor": "#4f8bad",
                "alpha": 0.30,
                "linestyle": "--",
            },
        )

        if show_band_coverage:
            label = (
                f"Within ±{tolerance:.0%}: "
                f"{coverage:.1%} of observations"
            )
        else:
            label = f"Within ±{tolerance:.0%}"

        ax.fill_between(
            x_reference,
            lower_band,
            upper_band,
            facecolor=style["facecolor"],
            alpha=style["alpha"],
            linewidth=0,
            label=label,
            zorder=1,
        )

        ax.plot(
            x_reference,
            lower_band,
            color=style["edgecolor"],
            linewidth=1.0,
            linestyle=style["linestyle"],
            alpha=0.95,
            zorder=2,
        )

        ax.plot(
            x_reference,
            upper_band,
            color=style["edgecolor"],
            linewidth=1.0,
            linestyle=style["linestyle"],
            alpha=0.95,
            zorder=2,
        )

    if show_density:
        ax.hexbin(
            observed_display,
            predicted_display,
            gridsize=gridsize,
            mincnt=min_count,
            extent=(
                lower_limit,
                upper_limit,
                lower_limit,
                upper_limit,
            ),
            cmap="Greys",
            norm=LogNorm(vmin=min_count),
            linewidths=0,
            alpha=0.70,
            zorder=3,
        )

    ax.plot(
        [lower_limit, upper_limit],
        [lower_limit, upper_limit],
        linestyle="--",
        linewidth=1.3,
        color="0.15",
        label="Perfect prediction",
        zorder=4,
    )

    ax.set_xlim(
        lower_limit,
        upper_limit,
    )
    ax.set_ylim(
        lower_limit,
        upper_limit,
    )
    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        "Observed CIF/FOB factor"
    )
    ax.set_ylabel(
        "Predicted CIF/FOB factor"
    )
    ax.set_title(title)

    ax.grid(alpha=0.18)

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        edgecolor="0.85",
        fontsize=9,
    )

    fig.text(
        0.5,
        0.015,
        (
            f"{observed_display.size:,} of "
            f"{observed.size:,} observations displayed; "
            f"range: {lower_quantile:.1%}–"
            f"{upper_quantile:.1%} quantiles."
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0, 0.04, 1, 1)
    )

    return fig, ax

def plot_observed_vs_predicted_density(
    frame: pd.DataFrame,
    *,
    observed_column: str,
    predicted_column: str,
    title: str = "Primary model: observed vs predicted CIF/FOB factor",
    gridsize: int = 45,
    min_count: int = 3,
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.995,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot observed versus predicted CIF/FOB factors using a restrained
    hexbin-density visualization.

    Quantile limits affect only the displayed range, not model evaluation.
    """
    required = {observed_column, predicted_column}
    missing = required.difference(frame.columns)

    if missing:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    observed = pd.to_numeric(
        frame[observed_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    predicted = pd.to_numeric(
        frame[predicted_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(observed)
        & np.isfinite(predicted)
        & (observed > 0)
        & (predicted > 0)
    )

    observed = observed[valid]
    predicted = predicted[valid]

    if observed.size == 0:
        raise ValueError(
            "No finite positive observed/predicted pairs are available."
        )

    combined = np.concatenate([observed, predicted])

    lower_limit = float(np.quantile(combined, lower_quantile))
    upper_limit = float(np.quantile(combined, upper_quantile))

    displayed = (
        (observed >= lower_limit)
        & (observed <= upper_limit)
        & (predicted >= lower_limit)
        & (predicted <= upper_limit)
    )

    observed_display = observed[displayed]
    predicted_display = predicted[displayed]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.2, 7.0))
    else:
        fig = ax.figure

    density = ax.hexbin(
            observed_display,
            predicted_display,
            gridsize=gridsize,
            mincnt=min_count,
            extent=(
                lower_limit,
                upper_limit,
                lower_limit,
                upper_limit,
            ),
            cmap="Blues",
            norm=LogNorm(vmin=min_count),
            linewidths=0,
            alpha=0.9,
        )

    ax.plot(
        [lower_limit, upper_limit],
        [lower_limit, upper_limit],
        linestyle="--",
        linewidth=1.2,
        color="0.35",
        label="Perfect prediction",
    )

    ax.set_xlim(lower_limit, upper_limit)
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Observed CIF/FOB factor")
    ax.set_ylabel("Predicted CIF/FOB factor")
    ax.set_title(title)

    ax.grid(alpha=0.18)
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=10,
    )
    colorbar = fig.colorbar(
        density,
        ax=ax,
        pad=0.018,
        shrink=0.82,
        aspect=28,
    )
    colorbar.set_label("Count per hexagon")
    ax.text(
            0.67,
            0.83,
            "High observed factors are\ncompressed toward the centre",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="0.35",
        )
    fig.text(
        0.5,
        0.015,
        (
            f"{observed_display.size:,} of {observed.size:,} observations shown; "
            f"display range: {lower_quantile:.1%}–{upper_quantile:.1%} quantiles."
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.035, 1, 1))

    return fig, ax



def plot_central_vs_tail_tradeoff(
    summary: pd.DataFrame,
    *,
    x_metric: str = "mean_median_ape",
    y_metric: str = "mean_upper_tail_median_ape",
    primary_run_name: str | None = None,
    x_label: str = "Median APE",
    y_label: str = "Upper-tail median APE",
    title: str = "CIF/FOB central accuracy vs upper-tail accuracy",
    highlight_runs: dict[str, str] | None = None,
):
    """
    Plot model trade-off between central accuracy and upper-tail accuracy.

    Each point is one algorithm × predictor-set run.

    Lower-left is better when both metrics are error metrics.
    """
    required = {"model_name", x_metric, y_metric}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(
            "Model summary is missing required columns: "
            + ", ".join(sorted(missing))
        )

    plot_data = summary.copy()
    plot_data = plot_data.dropna(
        subset=[x_metric, y_metric]
    ).copy()

    if plot_data.empty:
        raise ValueError(
            "No finite values available for central-vs-tail plot."
        )

    parsed = plot_data["model_name"].apply(
        split_estimation_run_name
    )

    plot_data["algorithm"] = [
        item[0] for item in parsed
    ]
    plot_data["predictor_set"] = [
        item[1] for item in parsed
    ]
    plot_data["algorithm_label"] = plot_data["algorithm"].map(
        short_algorithm_name
    )

    plot_data["x_display"] = _display_metric_values(
        plot_data,
        metric=x_metric,
    )
    plot_data["y_display"] = _display_metric_values(
        plot_data,
        metric=y_metric,
    )

    if highlight_runs is None:
        highlight_runs = {}

    if primary_run_name is not None:
        highlight_runs = {
            **highlight_runs,
            str(primary_run_name): "Primary: HGB m5",
        }

    fig, ax = plt.subplots(
        figsize=(10, 6.5)
    )

    for algorithm, group in plot_data.groupby(
        "algorithm_label",
        sort=True,
    ):
        ax.scatter(
            group["x_display"],
            group["y_display"],
            label=algorithm,
            alpha=0.75,
            s=55,
        )

    # Highlight selected decision-relevant runs only.
    highlight_data = plot_data[
        plot_data["model_name"].astype(str).isin(
            set(highlight_runs.keys())
        )
    ].copy()

    if not highlight_data.empty:
        ax.scatter(
            highlight_data["x_display"],
            highlight_data["y_display"],
            s=160,
            marker="*",
            label="Highlighted runs",
            zorder=5,
        )

        for _, row in highlight_data.iterrows():
            run_name = str(row["model_name"])
            label = highlight_runs.get(
                run_name,
                run_name,
            )

            ax.annotate(
                label,
                xy=(
                    row["x_display"],
                    row["y_display"],
                ),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
                weight="bold",
                alpha=0.9,
                arrowprops={
                    "arrowstyle": "-",
                    "lw": 0.8,
                    "alpha": 0.6,
                },
            )

    ax.set_xlabel(
        _metric_axis_label(
            x_metric,
            label=f"{x_label} — lower is better",
        )
    )
    ax.set_ylabel(
        _metric_axis_label(
            y_metric,
            label=f"{y_label} — lower is better",
        )
    )

    ax.set_title(title)
    ax.grid(alpha=0.25)

    ax.legend(
        fontsize=8,
        frameon=False,
        loc="best",
    )

    # Add small explanatory note inside the plot.
    ax.text(
        0.01,
        0.01,
        "Each point is one algorithm × predictor-set run. Lower-left is better.",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.75,
        va="bottom",
    )

    fig.tight_layout()

    return fig, ax, plot_data

def plot_corrected_trade_comparison(
    *,
    frame: pd.DataFrame,
    category_column: str,
    coverage_column: str,
    reduction_column: str,
    title: str,
    x_label: str,
    coverage_label: str = "Correction coverage (%)",
    reduction_label: str = "Value reduction (%)",
    figsize: tuple[float, float] = (10, 5.5),
    rotate_xticks: int = 0,
) -> tuple[Figure, Axes]:
    """
    Plot correction coverage and value reduction together.

    Bars show correction coverage on the left axis.
    A line shows value reduction on the right axis.
    """
    required_columns = {
        category_column,
        coverage_column,
        reduction_column,
    }
    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise RuntimeError(
            "Corrected-trade comparison plot is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    plot_data = frame.copy()

    plot_data[coverage_column] = pd.to_numeric(
        plot_data[coverage_column],
        errors="coerce",
    )
    plot_data[reduction_column] = pd.to_numeric(
        plot_data[reduction_column],
        errors="coerce",
    )

    plot_data = plot_data.loc[
        plot_data[category_column].notna()
        & plot_data[coverage_column].notna()
        & plot_data[reduction_column].notna()
    ].copy()

    if plot_data.empty:
        raise RuntimeError(
            "No valid rows available for corrected-trade comparison plot."
        )

    labels = plot_data[category_column].astype(str).tolist()
    x = np.arange(len(plot_data))

    fig, ax1 = plt.subplots(figsize=figsize)

    bars = ax1.bar(
        x,
        plot_data[coverage_column].to_numpy(dtype=float),
        alpha=0.8,
        label=coverage_label,
    )

    ax1.set_xlabel(x_label,)
    ax1.set_ylabel(coverage_label, color="tab:blue")
    ax1.set_xticks(x)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticklabels(
        labels,
        rotation=rotate_xticks,
        ha="right" if rotate_xticks else "center",
    )
    ax1.grid(axis="y", alpha=0.25)
    ax1.spines["left"].set_color("tab:blue")

    ## Create a second y-axis for the reduction line plot
    ax2 = ax1.twinx()
    line = ax2.plot(
        x,
        plot_data[reduction_column].to_numpy(dtype=float),
        marker="o",
        linewidth=2.0,
        color="crimson",
        markerfacecolor="crimson",
        markeredgecolor="darkred",
        label=reduction_label,
    )

    ax2.set_ylabel(reduction_label, color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.spines["right"].set_color("crimson")

    ax1.set_title(title)

    handles = [bars, line[0]]
    labels = [coverage_label, reduction_label]
    ax1.legend(handles, labels, loc="upper right", frameon=True)

    fig.tight_layout()

    return fig, ax1

def plot_temporal_validation_metric(
    folds: pd.DataFrame,
    *,
    metric: str = "median_ape",
    selected_runs: Iterable[str] | None = None,
    primary_run_name: str | None = None,
    year_column: str = "validation_year",
    metric_label: str = "Median APE",
    title: str = "Temporal validation performance",
):
    """
    Plot validation metric over time for selected model runs.

    Good default selected runs:
      - primary HGB run
      - HGB squared run with same predictor set
      - OLS run with same predictor set
    """
    required = {"model_name", year_column, metric}
    missing = required - set(folds.columns)
    if missing:
        raise ValueError(
            "Temporal folds are missing required columns: "
            + ", ".join(sorted(missing))
        )

    plot_data = folds.copy()
    plot_data = plot_data.dropna(
        subset=["model_name", year_column, metric]
    )

    if selected_runs is not None:
        selected_runs = [str(value) for value in selected_runs]
        plot_data = plot_data[
            plot_data["model_name"].astype(str).isin(selected_runs)
        ].copy()

    if plot_data.empty:
        raise ValueError(
            "No temporal-validation rows available after filtering."
        )

    parsed = plot_data["model_name"].apply(
        split_estimation_run_name
    )
    plot_data["algorithm"] = [
        item[0] for item in parsed
    ]
    plot_data["predictor_set"] = [
        item[1] for item in parsed
    ]
    plot_data["algorithm_label"] = plot_data["algorithm"].map(
        short_algorithm_name
    )
    plot_data["predictor_set_label"] = plot_data["predictor_set"].map(
        short_predictor_set_name
    )

    plot_data["display_label"] = (
        plot_data["algorithm_label"]
        + " / "
        + plot_data["predictor_set_label"].astype(str)
    )

    if primary_run_name is not None:
        is_primary = (
            plot_data["model_name"].astype(str)
            == str(primary_run_name)
        )
        plot_data.loc[
            is_primary,
            "display_label",
        ] = (
            plot_data.loc[
                is_primary,
                "display_label",
            ]
            + "  [primary]"
        )

    plot_data["metric_display"] = _display_metric_values(
        plot_data,
        metric=metric,
    )

    plot_data[year_column] = pd.to_numeric(
        plot_data[year_column],
        errors="coerce",
    ).astype(int)

    grouped = (
        plot_data
        .groupby(
            [year_column, "model_name", "display_label"],
            as_index=False,
        )["metric_display"]
        .mean()
        .sort_values([year_column, "display_label"])
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for label, group in grouped.groupby(
        "display_label",
        sort=True,
    ):
        ax.plot(
            group[year_column],
            group["metric_display"],
            marker="o",
            label=label,
        )

    ax.set_xlabel("Validation year")
    ax.set_ylabel(
        _metric_axis_label(
            metric,
            label=metric_label,
        )
    )
    ax.set_xticks(sorted(grouped[year_column].unique()))
    ax.set_xticklabels([str(int(y)) for y in sorted(grouped[year_column].unique())])
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout()

    return fig, ax, grouped


def plot_rmse_vs_mae_tradeoff(
    summary: pd.DataFrame,
    *,
    primary_run_name: str | None = None,
    title: str = "MAE vs RMSE trade-off",
):
    """
    Plot mean log MAE against mean log RMSE.

    This helps explain why squared-loss models can look better on RMSE
    while absolute-loss models can be better on central accuracy.
    """
    return plot_central_vs_tail_tradeoff(
        summary,
        x_metric="mean_mae_log_factor",
        y_metric="mean_rmse_log_factor",
        primary_run_name=primary_run_name,
        x_label="Mean log MAE",
        y_label="Mean log RMSE",
        title=title,
        annotate_top_n=10,
    )

def plot_numeric_relationship(df, 
                              x_col="netweight_kg", 
                              y_col="value_usd", 
                              alpha=0.1, 
                              figsize=(8, 6),
                              log_x=False,
                              log_y=False
):
    """
    Plot the relationship between two numerical columns, optionally on a log-log scale.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    x_col : str, default='netweight_kg'
        Column for x-axis.
    y_col : str, default='value_usd'
        Column for y-axis.
    alpha : float, default=0.1
        Scatter transparency.
    figsize : tuple, default=(8, 6)
        Figure size.
    log_x : bool, default=False
        Whether to use a logarithmic scale for the x-axis.
    log_y : bool, default=False
        Whether to use a logarithmic scale for the y-axis.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        The figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[x_col], df[y_col], alpha=alpha)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(f"{x_col.replace('_', ' ').title()}" + (" (log scale)" if log_x else ""))
    ax.set_ylabel(f"{y_col.replace('_', ' ').title()}" + (" (log scale)" if log_y else ""))
    ax.set_title(
        f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}" +
        (" (log-log)" if log_x and log_y else " (log-x)" if log_x else " (log-y)" if log_y else "")
    )

    fig.tight_layout()
    return fig, ax

def plot_correlation_map(df, 
                         title='Correlation Map',
                         fig_size=(12, 10)):
    """
    Plot a correlation map for numeric columns in the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    title : str, default='Correlation Map'
        Title of the plot.

    Returns
    -------
    None
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=fig_size)
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', mask=mask)
    plt.title(title)
    plt.show()


def plot_overlapped_distribution(
    df_left,
    df_right,
    column,
    left_label="API",
    right_label="DB",
    bins=100,
    figsize=(10, 6),
    min_positive=True,
    dropna=True,
    log_scale=True,
    kde=True,
):
    plot_df = pd.concat(
        [
            df_left[[column]].assign(source=left_label),
            df_right[[column]].assign(source=right_label),
        ],
        ignore_index=True,
    )

    mask = pd.Series(True, index=plot_df.index)

    if dropna:
        mask &= plot_df[column].notna()

    if min_positive:
        mask &= plot_df[column] > 0

    plot_df = plot_df.loc[mask].copy()

    plt.figure(figsize=figsize)
    ax = sns.histplot(
        data=plot_df,
        x=column,
        hue="source",
        bins=bins,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        kde=kde,
        log_scale=(log_scale, False),
    )

    scale_label = "log scale" if log_scale else "normal scale"
    plt.title(f"Overlapped distribution of {column} ({scale_label} x-axis)")
    plt.xlabel(f"{column} ({scale_label})")
    plt.ylabel("Density")

    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Source")

    plt.tight_layout()
    plt.show()