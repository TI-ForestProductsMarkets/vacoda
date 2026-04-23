"""
vacoda.visualization.relationships

Two-variable (2-D) relationship plots.
"""

import matplotlib.pyplot as plt


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
