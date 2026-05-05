"""
vacoda.visualization.relationships

Two-variable (2-D) relationship plots.
"""

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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