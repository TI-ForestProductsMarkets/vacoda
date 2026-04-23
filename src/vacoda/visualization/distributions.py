"""
vacoda.visualization.distributions

One-variable (1-D) distribution plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from vacoda.utils.diagnostics import compute_data_coverage


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
    opposite = "present" if criteria == "absence" else "absent"
    labels = [f"All {opposite}"] + [f"{c} {word}" for c in cols] + [f"All {word}"]
    keys = [f"none_{criteria}_%"] + [f"{c}_{criteria}_%" for c in cols] + [f"all_{criteria}_%"]
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
    ax.set_xlim(0, max(pcts) * 1.3)
    ax.set_title(f"Data {word.capitalize()}: {', '.join(cols)}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()


def plot_bar_distribution(df, 
                                column, 
                                figsize=(10, 5), 
                                rotation=45,
                                title=None):
    """
    Plot a bar chart showing the distribution of values in an ID-like column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    column : str
        Column to plot.
    figsize : tuple, default=(10, 5)
        Figure size.
    title : str, optional
        Custom title.

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    df[column].value_counts().plot(kind="bar", ax=ax)

    ax.set_title(title or f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=rotation)

    fig.tight_layout()
    return fig, ax


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