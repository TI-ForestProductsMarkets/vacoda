"""
vacoda.visualization.relationships

Two-variable (2-D) relationship plots.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from vacoda.visualization.labels import (
    short_algorithm_name,
    short_predictor_set_name,
    split_estimation_run_name,
    )
from typing import Iterable
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

def plot_observed_vs_predicted_density(
    frame: pd.DataFrame,
    *,
    observed_column: str,
    predicted_column: str,
    title: str = "Observed vs predicted CIF/FOB factor",
    gridsize: int = 60,
    min_count: int = 1,
    lower_quantile: float | None = None,
    upper_quantile: float | None = 0.995,
    show_colorbar: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot observed versus predicted values using hexagonal density bins.

    The diagonal reference line represents perfect prediction:
    predicted = observed.

    Parameters
    ----------
    frame
        DataFrame containing observed and predicted values.
    observed_column
        Name of the observed-factor column.
    predicted_column
        Name of the predicted-factor column.
    title
        Figure title.
    gridsize
        Approximate number of hexagonal bins across the x-axis.
    min_count
        Minimum observations required for a hexagon to be displayed.
    lower_quantile
        Optional lower quantile used only to determine plot limits.
        Set to None to use the minimum observed value.
    upper_quantile
        Optional upper quantile used only to determine plot limits.
        This prevents a few extreme values from compressing the main plot.
        Set to None to use the maximum value.
    show_colorbar
        Whether to show the observation-count color bar.
    ax
        Existing matplotlib axes. A new figure is created when omitted.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes.
    """
    required_columns = {observed_column, predicted_column}
    missing_columns = required_columns.difference(frame.columns)

    if missing_columns:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
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
            "No finite, positive observed/predicted pairs are available."
        )

    combined = np.concatenate([observed, predicted])

    if lower_quantile is None:
        lower_limit = float(np.min(combined))
    else:
        if not 0 <= lower_quantile < 1:
            raise ValueError("lower_quantile must be between 0 and 1.")
        lower_limit = float(np.quantile(combined, lower_quantile))

    if upper_quantile is None:
        upper_limit = float(np.max(combined))
    else:
        if not 0 < upper_quantile <= 1:
            raise ValueError("upper_quantile must be between 0 and 1.")
        upper_limit = float(np.quantile(combined, upper_quantile))

    if lower_limit >= upper_limit:
        raise ValueError(
            "Plot limits are invalid: lower limit must be below upper limit."
        )

    # Keep only observations inside the displayed range.
    # Quantile clipping affects the visualization only, not model metrics.
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
            "No observations remain within the requested plot limits."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 7.0))
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
        norm=LogNorm(),
    )

    # Perfect-prediction reference line.
    ax.plot(
        [lower_limit, upper_limit],
        [lower_limit, upper_limit],
        linestyle="--",
        linewidth=1.5,
        label="Perfect prediction",
    )

    ax.set_xlim(lower_limit, upper_limit)
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Observed CIF/FOB factor")
    ax.set_ylabel("Predicted CIF/FOB factor")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    if show_colorbar:
        colorbar = fig.colorbar(density, ax=ax)
        colorbar.set_label("Number of observations")

    n_total = observed.size
    n_displayed = observed_display.size

    ax.text(
        0.98,
        0.02,
        (
            f"Displayed: {n_displayed:,} of {n_total:,}\n"
            f"Plot limit: {upper_quantile:.1%} quantile"
            if upper_quantile is not None
            else f"Displayed: {n_displayed:,} of {n_total:,}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout()

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