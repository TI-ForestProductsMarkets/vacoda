"""
vacoda.visualization.distributions

One-variable (1-D) distribution plots.
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from vacoda.cif_fob.display import (
    short_algorithm_name,
    short_predictor_set_name,
    split_estimation_run_name,
    )
from collections.abc import Sequence

from vacoda.utils.diagnostics import compute_data_coverage


def _metric_is_percentage(metric: str) -> bool:
    """Return whether a metric should be displayed as percent."""
    lowered = metric.lower()
    return (
        "ape" in lowered
        or "share" in lowered
        or "percent" in lowered
    )


def _prepare_metric_for_plot(
    frame: pd.DataFrame,
    *,
    metric: str,
) -> pd.Series:
    """Return metric values in display scale."""
    values = pd.to_numeric(
        frame[metric],
        errors="coerce",
    )

    if _metric_is_percentage(metric):
        return values * 100.0

    return values

def plot_model_error_by_group(
    frame: pd.DataFrame,
    *,
    group_column: str,
    model_column: str,
    error_column: str = "ape",
    aggregation: str = "median",
    group_order: list[str] | None = None,
    title: str | None = None,
    x_label: str = "",
    y_label: str | None = None,
):
    """
    Plot grouped model errors by category.

    Parameters
    ----------
    frame
        Long-format frame containing one row per observation and model.
    group_column
        Column defining the comparison groups, such as ``value_band``.
    model_column
        Column containing model labels.
    error_column
        Row-level error column, such as ``ape``.
    aggregation
        Error summary to plot. Supported values are:

        - ``"median"``
        - ``"mean"``
        - ``"p90"``
        - ``"p95"``

    group_order
        Optional explicit order for the group categories.
    title
        Figure title.
    x_label
        X-axis label.
    y_label
        Y-axis label. When omitted, it is derived from ``aggregation``.

    Returns
    -------
    tuple
        ``(fig, ax, grouped)``
    """

    required_columns = {
        group_column,
        model_column,
        error_column,
    }

    missing_columns = sorted(
        required_columns - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Model-group comparison frame is missing columns: "
            + ", ".join(missing_columns)
        )

    aggregation = aggregation.strip().lower()

    supported_aggregations = {
        "median",
        "mean",
        "p90",
        "p95",
    }

    if aggregation not in supported_aggregations:
        raise ValueError(
            f"Unsupported aggregation {aggregation!r}. "
            f"Expected one of {sorted(supported_aggregations)}."
        )

    plot_data = frame[
        [
            group_column,
            model_column,
            error_column,
        ]
    ].copy()

    plot_data[error_column] = pd.to_numeric(
        plot_data[error_column],
        errors="coerce",
    )

    plot_data = plot_data.dropna(
        subset=[
            group_column,
            model_column,
            error_column,
        ]
    )

    plot_data = plot_data.loc[
        np.isfinite(plot_data[error_column])
    ].copy()

    if plot_data.empty:
        raise ValueError(
            "No valid observations are available for the "
            "model-error comparison plot."
        )

    grouped_source = (
        plot_data
        .groupby(
            [
                group_column,
                model_column,
            ],
            observed=False,
            sort=False,
        )[error_column]
    )

    if aggregation == "median":
        aggregated = grouped_source.median()
        aggregation_label = "Median"
    elif aggregation == "mean":
        aggregated = grouped_source.mean()
        aggregation_label = "Mean"
    elif aggregation == "p90":
        aggregated = grouped_source.quantile(0.90)
        aggregation_label = "P90"
    else:
        aggregated = grouped_source.quantile(0.95)
        aggregation_label = "P95"

    metric_column = f"{aggregation}_error_percent"

    grouped = (
        aggregated
        .mul(100.0)
        .reset_index(name=metric_column)
    )

    if group_order is not None:
        grouped[group_column] = pd.Categorical(
            grouped[group_column],
            categories=group_order,
            ordered=True,
        )

        grouped = grouped.sort_values(
            [
                group_column,
                model_column,
            ],
            kind="stable",
        )

    pivot = grouped.pivot(
        index=group_column,
        columns=model_column,
        values=metric_column,
    )

    if group_order is not None:
        pivot = pivot.reindex(group_order)

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        title
        or f"{aggregation_label} model error by group"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(
        y_label
        or f"{aggregation_label} APE (%)"
    )
    ax.tick_params(
        axis="x",
        rotation=0,
    )
    ax.legend(
        title="Model",
        frameon=False,
    )

    fig = ax.get_figure()
    fig.tight_layout()

    return fig, ax, grouped


def plot_grouped_error_bars(
    frame: pd.DataFrame,
    *,
    group_column: str,
    error_column: str = "ape",
    count_column: str | None = None,
    min_observations: int = 100,
    top_n: int | None = 20,
    title: str = "Model error by group",
    x_label: str = "Median APE (%)",
    y_label: str = "Group",
    higher_is_worse: bool = True,
    sort_by_error: bool = True,
    group_order: Sequence[str] | None = None,
):
    """
    Plot median model error by group.

    Intended for heterogeneity diagnostics:
      - product group
      - factor band
      - value band
      - distance band
      - route type

    Expects one row per holdout prediction.

    Parameters
    ----------
    sort_by_error:
        If True, sort groups by median error.

    group_order:
        Optional explicit display order for naturally ordered groups such as
        value bands or distance bands. When supplied, it takes precedence over
        sort_by_error.
    """
    required = {
        group_column,
        error_column,
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            "Input frame is missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = frame.copy()

    data[group_column] = (
        data[group_column]
        .astype("string")
        .str.strip()
    )

    data[error_column] = pd.to_numeric(
        data[error_column],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            group_column,
            error_column,
        ]
    )

    if data.empty:
        raise ValueError(
            "No data available for grouped error plot."
        )

    grouped = (
        data
        .groupby(
            group_column,
            as_index=False,
            observed=True,
        )
        .agg(
            n_observations=(error_column, "size"),
            median_error=(error_column, "median"),
            mean_error=(error_column, "mean"),
            p90_error=(
                error_column,
                lambda values: values.quantile(0.90),
            ),
        )
    )

    grouped = grouped.loc[
        grouped["n_observations"] >= int(min_observations)
    ].copy()

    if grouped.empty:
        raise ValueError(
            "No groups satisfy the minimum observation requirement."
        )

    # APE is normally stored as a fraction, so convert it to percent.
    if "ape" in error_column.lower():
        grouped["median_error_display"] = (
            grouped["median_error"] * 100.0
        )
        grouped["mean_error_display"] = (
            grouped["mean_error"] * 100.0
        )
        grouped["p90_error_display"] = (
            grouped["p90_error"] * 100.0
        )
    else:
        grouped["median_error_display"] = grouped["median_error"]
        grouped["mean_error_display"] = grouped["mean_error"]
        grouped["p90_error_display"] = grouped["p90_error"]

    if group_order is not None:
        normalized_group_order = tuple(
            str(group).strip()
            for group in group_order
        )

        if len(set(normalized_group_order)) != len(normalized_group_order):
            raise ValueError(
                "group_order contains duplicate labels."
            )

        actual_groups = set(
            grouped[group_column]
            .dropna()
            .astype(str)
        )

        expected_groups = set(normalized_group_order)

        unexpected_groups = sorted(
            actual_groups - expected_groups
        )

        if unexpected_groups:
            raise ValueError(
                f"Unexpected labels for {group_column!r}: "
                + ", ".join(
                    repr(group)
                    for group in unexpected_groups
                )
                + ". The data labels and group_order must match exactly."
            )

        order_lookup = {
            group: position
            for position, group in enumerate(normalized_group_order)
        }

        grouped["_group_sort_order"] = (
            grouped[group_column]
            .astype(str)
            .map(order_lookup)
        )

        grouped = (
            grouped
            .sort_values(
                "_group_sort_order",
                ascending=True,
                kind="stable",
            )
            .drop(columns="_group_sort_order")
        )

    elif sort_by_error:
        grouped = grouped.sort_values(
            "median_error_display",
            ascending=not higher_is_worse,
            kind="stable",
        )

    if top_n is not None:
        top_n = int(top_n)

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero when provided."
            )

        grouped = grouped.head(top_n).copy()

    grouped[group_column] = (
        grouped[group_column]
        .astype(str)
    )

    fig_height = max(
        4.5,
        0.34 * len(grouped) + 1.5,
    )

    fig, ax = plt.subplots(
        figsize=(10.5, fig_height)
    )

    y_positions = np.arange(len(grouped))

    ax.barh(
        y_positions,
        grouped["median_error_display"].to_numpy(dtype=float),
    )

    ax.set_yticks(y_positions)

    ax.set_yticklabels(
        grouped[group_column].tolist()
    )

    ax.invert_yaxis()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    for y_position, value, n_obs in zip(
        y_positions,
        grouped["median_error_display"],
        grouped["n_observations"],
    ):
        ax.text(
            value,
            y_position,
            f"  {value:.2f}%  n={int(n_obs):,}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()

    return fig, ax, grouped

def plot_model_metric_ranking(
    summary: pd.DataFrame,
    *,
    metric: str = "mean_mae_log_factor",
    primary_run_name: str | None = None,
    top_n: int | None = 20,
    higher_is_better: bool = False,
    title: str | None = None,
    metric_label: str | None = None,
):
    """
    Plot ranked model runs for one validation metric.

    Best use:
      - mean_mae_log_factor
      - mean_median_ape
      - mean_p90_ape
      - mean_rmse_log_factor

    Parameters
    ----------
    summary:
        Temporal validation summary with one row per model run.
    metric:
        Column to rank by.
    primary_run_name:
        Full primary run name, e.g.
        'hist_gradient_boosting__model_5_full_history'.
    top_n:
        Maximum number of rows to display. If primary is outside top_n,
        it is still included.
    higher_is_better:
        Use True for R² or Spearman. Use False for error metrics.
    title:
        Optional chart title.
    metric_label:
        Optional x-axis label.

    Returns
    -------
    fig, ax, plot_data
    """
    required = {"model_name", metric}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(
            "Model summary is missing required columns: "
            + ", ".join(sorted(missing))
        )

    plot_data = summary.copy()
    plot_data = plot_data[
        plot_data[metric].notna()
    ].copy()

    if plot_data.empty:
        raise ValueError(
            f"No finite values available for metric {metric!r}."
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

    plot_data["model_number"] = (
        plot_data["predictor_set_label"]
        .astype(str)
        .str.extract(r"M(\d+)", expand=False)
        .astype(int)
    )

    plot_data["display_label"] = (
        plot_data["algorithm_label"]
        + " / "
        + plot_data["predictor_set_label"].astype(str)
    )

    if primary_run_name is not None:
        plot_data["is_primary"] = (
            plot_data["model_name"].astype(str) == str(primary_run_name)
        )
        plot_data.loc[
            plot_data["is_primary"],
            "display_label",
        ] = (
            plot_data.loc[
                plot_data["is_primary"],
                "display_label",
            ]
            + "  [primary]"
        )
    else:
        plot_data["is_primary"] = False

    plot_data["metric_display"] = _prepare_metric_for_plot(
        plot_data,
        metric=metric,
    )

    plot_data = plot_data.sort_values(
        metric,
        ascending=not higher_is_better,
    )

    if top_n is not None and len(plot_data) > top_n:
        top = plot_data.head(top_n).copy()

        if primary_run_name is not None:
            primary = plot_data[
                plot_data["model_name"].astype(str)
                == str(primary_run_name)
            ]
            if not primary.empty:
                top = pd.concat(
                    [top, primary],
                    ignore_index=True,
                ).drop_duplicates(
                    subset=["model_name"],
                    keep="first",
                )

        plot_data = top.sort_values(
            metric,
            ascending=not higher_is_better,
        )

    plot_data = plot_data.sort_values(
        ["model_number", "algorithm_label"],
        ascending=[True, True],
    )

    fig_height = max(
        4.5,
        0.34 * len(plot_data) + 1.5,
    )
    fig, ax = plt.subplots(
        figsize=(11, fig_height)
    )

    y_positions = np.arange(len(plot_data))

    ax.barh(
        y_positions,
        plot_data["metric_display"].to_numpy(dtype=float),
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        plot_data["display_label"].tolist()
    )
    ax.invert_yaxis()

    if metric_label is None:
        metric_label = metric

    if _metric_is_percentage(metric):
        metric_label = f"{metric_label} (%)"

    ax.set_xlabel(metric_label)
    ax.set_ylabel("Model run")

    if title is None:
        title = f"Model ranking by {metric}"

    ax.set_title(title)

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    return fig, ax, plot_data


def plot_observed_vs_predicted_p95(
    summary: pd.DataFrame,
    *,
    primary_run_name: str | None = None,
    top_n: int = 12,
    ranking_metric: str = "mean_mae_log_factor",
    observed_column: str = "mean_observed_p95",
    predicted_column: str = "mean_predicted_p95",
    title: str = "Observed vs predicted upper-tail factor by model",
):
    """
    Compare observed and predicted P95 factors for top model runs.

    This is useful for showing whether a model compresses the upper tail.
    """
    required = {
        "model_name",
        ranking_metric,
        observed_column,
        predicted_column,
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(
            "Model summary is missing required columns: "
            + ", ".join(sorted(missing))
        )

    plot_data = summary.copy()
    plot_data = plot_data.dropna(
        subset=[
            ranking_metric,
            observed_column,
            predicted_column,
        ]
    )

    plot_data = plot_data.sort_values(
        ranking_metric,
        ascending=True,
    )

    if len(plot_data) > top_n:
        top = plot_data.head(top_n).copy()

        if primary_run_name is not None:
            primary = plot_data[
                plot_data["model_name"].astype(str)
                == str(primary_run_name)
            ]
            if not primary.empty:
                top = pd.concat(
                    [top, primary],
                    ignore_index=True,
                ).drop_duplicates(
                    subset=["model_name"],
                    keep="first",
                )

        plot_data = top

    parsed = plot_data["model_name"].apply(
        split_estimation_run_name
    )
    plot_data["algorithm"] = [
        item[0] for item in parsed
    ]
    plot_data["predictor_set"] = [
        item[1] for item in parsed
    ]
    plot_data["predictor_set_label"] = plot_data["predictor_set"].map(
        short_predictor_set_name
    )
    plot_data["display_label"] = (
        plot_data["algorithm"].map(short_algorithm_name)
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

    y_positions = np.arange(len(plot_data))
    bar_height = 0.38

    fig_height = max(
        4.5,
        0.36 * len(plot_data) + 1.5,
    )
    fig, ax = plt.subplots(
        figsize=(11, fig_height)
    )

    ax.barh(
        y_positions - bar_height / 2,
        plot_data[observed_column].to_numpy(dtype=float),
        height=bar_height,
        label="Observed P95",
    )
    ax.barh(
        y_positions + bar_height / 2,
        plot_data[predicted_column].to_numpy(dtype=float),
        height=bar_height,
        label="Predicted P95",
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        plot_data["display_label"].tolist()
    )
    ax.invert_yaxis()

    ax.set_xlabel("CIF/FOB factor")
    ax.set_ylabel("Model run")
    ax.set_title(title)
    ax.legend()
    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    return fig, ax, plot_data

def plot_nebula_boxplot(
    df,
    factor_col="observed_cif_fob_factor",
    upper_display=3.0,
    lower_display=0.0,
    max_points=250_000,
    random_seed=42,
    x_label = "Observed CIF/FOB factor",
    title="Distribution of usable direct CIF/FOB factors",
):
    """
    Plot CIF/FOB factor distribution as a combined nebula point-cloud and boxplot.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing CIF/FOB factor values.
    factor_col : str
        Column name containing CIF/FOB factor.
    upper_display : float
        Upper x-axis display limit and data display filter.
        Values above this are excluded from the visual display.
    lower_display : float
        Lower x-axis display limit.
    max_points : int
        Maximum number of points to draw in the nebula plot.
        Useful because the dataset can be very large.
    random_seed : int
        Random seed for reproducible jitter and sampling.
    title : str
        Figure title.

    Returns
    -------
    fig, axes, stats : matplotlib Figure, Axes array, dict
        The figure, axes, and summary statistics used in the plot.
    """

    # Keep valid positive values only
    x_all = df[factor_col].dropna()
    x_all = x_all[x_all > 0]

    # Display subset
    x_display = x_all[(x_all >= lower_display) & (x_all <= upper_display)].to_numpy()

    if len(x_display) == 0:
        raise ValueError("No valid CIF/FOB factor values available within display range.")

    # Quartiles calculated on displayed data
    q1 = np.quantile(x_display, 0.25)
    median = np.quantile(x_display, 0.50)
    q3 = np.quantile(x_display, 0.75)
    iqr = q3 - q1

    upper_15_iqr = q3 + 1.5 * iqr
    upper_25_iqr = q3 + 2.5 * iqr

    # Sampling for readable point cloud
    rng = np.random.default_rng(random_seed)

    if len(x_display) > max_points:
        sample_idx = rng.choice(len(x_display), size=max_points, replace=False)
        x_points = x_display[sample_idx]
    else:
        x_points = x_display

    # Vertical jitter for nebula effect
    y_points = rng.normal(loc=0, scale=0.055, size=len(x_points))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    ax1, ax2 = axes

    # -------------------------
    # Top panel: nebula
    # -------------------------
    ax1.axvspan(q1, q3, alpha=0.15, label="IQR, middle 50%")
    ax1.scatter(x_points, y_points, s=2, alpha=0.04)

    ax1.axvline(1.0, linestyle="--", linewidth=1, label="Factor = 1", color="grey")
    ax1.axvline(median, linestyle="-", linewidth=1.8, label=f"Median = {median:.3f}")
    ax1.axvline(
        upper_15_iqr,
        linestyle=":",
        linewidth=1.3,
        label=f"Q3 + 1.5×IQR = {upper_15_iqr:.3f}",
    )
    ax1.axvline(
        upper_25_iqr,
        linestyle="-.",
        linewidth=1.3,
        color="orange",
        label=f"Q3 + 2.5×IQR = {upper_25_iqr:.3f}",
    )

    ax1.set_yticks([])
    ax1.set_ylabel("Point cloud")
    ax1.set_title(title)
    ax1.legend(loc="upper right", fontsize=8, frameon=True)

    # -------------------------
    # Bottom panel: boxplot
    # -------------------------
    ax2.boxplot(
        x_display,
        vert=False,
        showfliers=True,
        widths=0.55,
        medianprops={
            "color": "tab:blue",
            "linewidth": 1.8,},
    )

    ax2.axvspan(q1, q3, alpha=0.15)
    ax2.axvline(1.0, linestyle="--", linewidth=1, color="grey")
    ax2.axvline(median, linestyle="-", linewidth=1.8)

    ax2.set_yticks([])
    ax2.set_ylabel("Boxplot")
    ax2.set_xlabel(x_label)
    ax2.set_xlim(lower_display, upper_display)

    plt.tight_layout()

    stats = {
        "n_total_positive": len(x_all),
        "n_displayed": len(x_display),
        "display_share_percent": 100 * len(x_display) / len(x_all),
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "q3_plus_1_5_iqr": upper_15_iqr,
        "q3_plus_2_5_iqr": upper_25_iqr,
        "lower_display": lower_display,
        "upper_display": upper_display,
    }

    return fig, axes, stats



def plot_cif_fob_corrected_agreement_bands(
    frame: pd.DataFrame,
    *,
    title: str = (
        "Residual exporter/importer disagreement after CIF/FOB correction"
    ),
):
    required = {"agreement_band", "rows", "percent"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            "Agreement-band frame is missing columns: "
            + ", ".join(sorted(missing))
        )

    plot_frame = frame.copy()

    # Normalize raw labels
    normalized = (
        plot_frame["agreement_band"]
        .astype(str)
        .str.strip()
        .str.replace("–", "-", regex=False)
        .str.lower()
    )

    label_map = {
        "within 5%": "Within 5%",
        "5-10%": "5–10%",
        "10-25%": "10–25%",
        "25-50%": "25–50%",
        "50-100%": "50–100%",
        "more than factor 2": "More than factor 2",
    }

    plot_frame["agreement_band"] = normalized.map(label_map)

    # Safety check
    if plot_frame["agreement_band"].isna().any():
        bad = frame.loc[
            plot_frame["agreement_band"].isna(),
            "agreement_band",
        ].unique()
        raise RuntimeError(
            "Unexpected agreement-band labels: "
            + ", ".join(map(str, bad))
        )

    order = [
        "Within 5%",
        "5–10%",
        "10–25%",
        "25–50%",
        "50–100%",
        "More than factor 2",
    ]

    plot_frame["agreement_band"] = pd.Categorical(
        plot_frame["agreement_band"],
        categories=order,
        ordered=True,
    )

    plot_frame = plot_frame.sort_values("agreement_band")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    y = np.arange(len(plot_frame))
    bars = ax.barh(
        y,
        plot_frame["percent"].astype(float),
        alpha=0.85,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_frame["agreement_band"].astype(str))
    ax.set_xlabel("Share of usable paired rows (%)")
    ax.set_ylabel("Value-disagreement band")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, (_, row) in zip(bars, plot_frame.iterrows(), strict=False):
        width = float(row["percent"])
        rows = int(row["rows"])
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}% ({rows:,})",
            va="center",
            fontsize=9,
        )

    xmax = max(60, plot_frame["percent"].max() * 1.18)
    ax.set_xlim(0, xmax)

    fig.tight_layout()
    return fig, ax


def plot_data_coverage_pie(df: pd.DataFrame, cols: list, indicator="absence"):
    stats = compute_data_coverage(df, cols, indicator=indicator)

    word = "absent" if indicator == "absence" else "present"
    labels = [f"{c} {word}" for c in cols] + [f"All {word}", f"None {word}"]
    keys = [f"{c}_{indicator}_%" for c in cols] + [f"all_{indicator}_%", f"none_{indicator}_%"]
    pcts = [stats[k] for k in keys]

    fig, ax = plt.subplots()
    wedges, _, _ = ax.pie(
        pcts,
        labels=None,
        autopct="%1.1f%%",
        pctdistance=0.75,
        startangle=140,
    )
    ax.legend(wedges, labels, loc="best")
    ax.set_title(f"Data {word.capitalize()}: {', '.join(cols)}")
    plt.show()


def plot_data_coverage_bar(df: pd.DataFrame, 
                           cols: list,
                           criteria="absence"):
    """
    Plot a horizontal bar chart showing the data coverage for the specified columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    cols : list
        List of columns to plot.
    criteria : str, default="absence"
        Criteria for data coverage. Can be "absence" or "presence".
    """
    
    # Check if all columns are float or numeric
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        import warnings
        warnings.warn(f"plot_data_coverage_bar: The following columns are not numeric and will be skipped: {non_numeric}. No plot will be shown.")
        return

    stats = compute_data_coverage(df, cols, indicator=criteria)

    word = "absent" if criteria == "absence" else "present"
    #opposite = "present" if criteria == "absence" else "absent"
    #labels = [f"All {opposite}"] + [f"{c} {word}" for c in cols] + [f"All {word}"]
    labels = [f"None {word}"] + [f"{c} {word}" for c in cols] + [f"At least one {word}"]
    keys = [f"none_{criteria}_%"] + [f"{c}_{criteria}_%" for c in cols] + [f"at_least_one_{criteria}_%"]
    #keys = [f"none_{criteria}_%"] + [f"{c}_{criteria}_%" for c in cols] + [f"all_{criteria}_%"]
    pcts = [stats[k] for k in keys]
    sizes = [round(p / 100 * stats["total_rows"]) for p in pcts]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.7)))
    bars = ax.barh(labels, pcts, color=colors[: len(labels)])

    for bar, pct, count in zip(bars, pcts, sizes):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.2f}%  (n={count:,})",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("% of rows")
    ax.set_xlim(0, max(pcts) * 1.4)
    ax.set_title(f"Data {word.capitalize()}: {', '.join(cols)}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()


def plot_bar_distribution(df, 
                            feature, 
                            show_percent=True,
                            figsize=(10, 5), 
                            rotation=45,
                            title=None):
    """
    Plot a bar chart showing the distribution of values in an ID-like column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    feature : str
        Column to plot.
    show_percent : bool, default=True
        Whether to show percentages instead of counts.
    figsize : tuple, default=(10, 5)
        Figure size.
    title : str, optional
        Custom title.

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    if show_percent:
        df[feature].value_counts(normalize=True).mul(100).plot(kind="bar", ax=ax)
    else:
        df[feature].value_counts().plot(kind="bar", ax=ax)

    ax.set_title(title or f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count" if not show_percent else "Percentage")
    ax.tick_params(axis="x", rotation=rotation)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.grid(axis="x", linestyle="", alpha=0.0)

    fig.tight_layout()
    return fig, ax

def plot_bar_distributions(df, 
                           features_list,
                           show_percent=True,
                            figsize=(10, 5), 
                            rotation=45,
                            title=None):
    """
    Plot multiple bar charts showing the distribution of values in ID-like columns.
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    features_list : list of str
        Columns to plot.
    show_percent : bool, default=True
        Whether to show percentages instead of counts.
    figsize : tuple, default=(10, 5)
        Figure size.
    title : str, optional
        Custom title prefix for each plot (the column name will be appended).
    """
    for feature in features_list:
        plot_bar_distribution(
            df=df,
            feature=feature,
            show_percent=show_percent,
            figsize=figsize,
            rotation=rotation,
            title=f"{title + ': ' if title else ''}Distribution of {feature}",
        )


def plot_distribution(df, 
                      column, 
                      title=None, 
                      figsize=(10, 5), 
                      categorical_threshold=50, 
                      rotation=45):
    """
    Plot a distribution for a single column.

    Uses:
    - bar plot for columns with relatively few unique values
    - boxplot for columns with many unique numeric values

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    column : str
        Column to plot.
    title : str, optional
        Custom title.
    figsize : tuple, default=(10, 5)
        Figure size.
    categorical_threshold : int, default=50
        Threshold for deciding bar plot vs boxplot.

    Returns
    -------
    fig, ax
    """
    unique_count = df[column].nunique(dropna=True)

    fig, ax = plt.subplots(figsize=figsize)

    if unique_count <= categorical_threshold:
        df[column].value_counts().plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        ax.set_xlabel(column)
        ax.tick_params(axis="x", rotation=rotation)
    else:
        df.boxplot(column=column, ax=ax)
        ax.set_ylabel(column)
        ax.set_xlabel("")

    ax.set_title(title or f"Distribution of {column}")
    fig.tight_layout()
    return fig, ax


def plot_multiple_bar_distributions(df, 
                                    columns_list, 
                                    rotation=45,
                                    figsize=(10, 5)):
    """
    Create one figure per bar column and return them as a dictionary.

    Returns
    -------
    dict[str, tuple[Figure, Axes]]
        Mapping column -> (fig, ax)
    """
    plots = {}
    for col in columns_list:
        plots[col] = plot_bar_distribution(df=df, column=col, figsize=figsize, rotation=rotation)
    return plots


def plot_distribution_and_density(
    df,
    column,
    bins=100,
    figsize=(12, 6),
    kde_fill=True,
    dropna=True,
    min_positive=True,
    log_scale=True,
    title_prefix=None,
):
    """
    Plot a histogram and a KDE for a numeric column, with optional log scale.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    column : str
        Column to plot.
    bins : int, default=100
        Number of bins for histogram.
    figsize : tuple, default=(12, 6)
        Figure size for plots.
    kde_fill : bool, default=True
        Whether to fill the KDE plot.
    dropna : bool, default=True
        Whether to drop NA values.
    min_positive : bool, default=True
        Whether to restrict to strictly positive values (recommended for log scale).
    log_scale : bool, default=True
        If True, use log scale for histogram and log10 for KDE. If False, use normal scale.
    title_prefix : str, optional
        Prefix for plot titles.

    Returns
    -------
    None
    """
    vals = df[column]
    if dropna:
        vals = vals.dropna()
    if min_positive:
        vals = vals[vals > 0]

    if log_scale:
        # Histogram with log-scale x-axis
        plt.figure(figsize=figsize)
        sns.histplot(vals, bins=bins, kde=True, log_scale=(True, False))
        plt.title(f"{title_prefix + ': ' if title_prefix else ''}Distribution of {column} (log scale)")
        plt.xlabel(f"{column} (log scale)")
        plt.ylabel("Frequency")
        plt.show()

        # KDE of log10 values
        plt.figure(figsize=figsize)
        sns.kdeplot(np.log10(vals), fill=kde_fill)
        plt.title(f"{title_prefix + ': ' if title_prefix else ''}Density of log10({column})")
        plt.xlabel(f"log10({column})")
        plt.ylabel("Density")
        plt.show()
    else:
        # Histogram with normal scale
        plt.figure(figsize=figsize)
        sns.histplot(vals, bins=bins, kde=True)
        plt.title(f"{title_prefix + ': ' if title_prefix else ''}Distribution of {column} (normal scale)")
        plt.xlabel(f"{column}")
        plt.ylabel("Frequency")
        plt.show()

        # KDE of normal values
        plt.figure(figsize=figsize)
        sns.kdeplot(vals, fill=kde_fill)
        plt.title(f"{title_prefix + ': ' if title_prefix else ''}Density of {column}")
        plt.xlabel(f"{column}")
        plt.ylabel("Density")
        plt.show()



