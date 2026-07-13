"""
vacoda.visualization.audits

Visualization helpers for evidence-coverage and representativeness audits.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISTANCE_BAND_ORDER: tuple[str, ...] = (
    "<1k km",
    "1k–3k km",
    "3k–7k km",
    "7k–12k km",
    ">12k km",
)

def plot_representation_ratio_bars(
    audit: pd.DataFrame,
    *,
    dimension: str,
    ratio_column: str = "clean_representation_ratio",
    group_column: str = "group",
    prediction_rows_column: str = "prediction_rows",
    training_rows_column: str = "clean_training_rows",
    min_prediction_rows: int = 100,
    top_n: int | None = None,
    title: str = "Training-evidence representation",
    x_label: str = "Representation ratio",
    y_label: str = "Group",
    sort_by_ratio: bool = True,
    group_order: list[str] | None = None,
):
    """
    Plot representation ratio for one audit dimension.

    Ratio interpretation:
      - 1.0 means training evidence is proportional to prediction population.
      - <1.0 means underrepresented in training evidence.
      - >1.0 means overrepresented in training evidence.
    """
    required = {
        "dimension",
        group_column,
        ratio_column,
        prediction_rows_column,
        training_rows_column,
    }

    missing = required - set(audit.columns)

    if missing:
        raise ValueError(
            "Audit table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    plot_data = audit.loc[
        audit["dimension"].astype(str).eq(str(dimension))
    ].copy()

    if plot_data.empty:
        raise ValueError(
            f"No audit rows found for dimension: {dimension!r}"
        )

    plot_data[ratio_column] = pd.to_numeric(
        plot_data[ratio_column],
        errors="coerce",
    )

    plot_data[prediction_rows_column] = pd.to_numeric(
        plot_data[prediction_rows_column],
        errors="coerce",
    ).fillna(0)

    plot_data[training_rows_column] = pd.to_numeric(
        plot_data[training_rows_column],
        errors="coerce",
    ).fillna(0)

    plot_data = plot_data.dropna(
        subset=[group_column, ratio_column]
    )

    plot_data = plot_data[
        plot_data[prediction_rows_column] >= int(min_prediction_rows)
    ].copy()

    if plot_data.empty:
        raise ValueError(
            "No groups satisfy the minimum prediction-row requirement."
        )
    
    if group_order is not None:
        group_order = [str(group) for group in group_order]

        plot_data[group_column] = (
            plot_data[group_column]
            .astype("string")
        )

        order_lookup = {
            group: position
            for position, group in enumerate(group_order)
        }

        plot_data["_group_sort_order"] = (
            plot_data[group_column]
            .map(order_lookup)
            .fillna(len(group_order))
            .astype(int)
        )

        plot_data = (
            plot_data
            .sort_values(
                ["_group_sort_order", group_column],
                ascending=[True, True],
            )
            .drop(columns="_group_sort_order")
        )

    elif sort_by_ratio:
        plot_data = plot_data.sort_values(
            ratio_column,
            ascending=True,
        )

    if top_n is not None:
        plot_data = plot_data.head(int(top_n)).copy()

    plot_data[group_column] = plot_data[group_column].astype(str)

    fig_height = max(
        4.5,
        0.36 * len(plot_data) + 1.5,
    )

    fig, ax = plt.subplots(
        figsize=(10.5, fig_height)
    )

    y_positions = np.arange(len(plot_data))

    ax.barh(
        y_positions,
        plot_data[ratio_column].to_numpy(dtype=float),
    )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        plot_data[group_column].tolist()
    )
    ax.invert_yaxis()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    for y_position, ratio, pred_rows, train_rows in zip(
        y_positions,
        plot_data[ratio_column],
        plot_data[prediction_rows_column],
        plot_data[training_rows_column],
    ):
        ax.text(
            ratio,
            y_position,
            f"  {ratio:.2f}  pred={int(pred_rows):,} train={int(train_rows):,}",
            va="center",
            fontsize=8,
        )

    ax.text(
        0.01,
        0.01,
        "Ratio < 1 = underrepresented; ratio > 1 = overrepresented",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.75,
        va="bottom",
    )

    fig.tight_layout()

    return fig, ax, plot_data