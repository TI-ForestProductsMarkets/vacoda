"""
vacoda.visualization.trade_patterns

Trade-specific comparison plots across reporter, partner, and year.
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.lines import Line2D


def plot_price_by_top_reporters(
    df,
    column="price_per_kg",
    n_reporters=10,
    reporter_col="reporter_country_name",
    ranking_col="value_usd",
    figsize=(12, 6),
    title=None,
):
    """
    Plot the distribution of a price/value column across top reporters.

    Top reporters are ranked by total `ranking_col`.
    """
    top_reporters = (
        df.groupby(reporter_col)[ranking_col].sum().nlargest(n_reporters).index
    )

    subset = df[df[reporter_col].isin(top_reporters)].copy()

    fig, ax = plt.subplots(figsize=figsize)
    subset.boxplot(column=column, by=reporter_col, rot=90, ax=ax)

    ax.set_yscale("log")
    ax.set_title(title or f"{column} by top {n_reporters} reporters (log scale)")
    ax.set_xlabel("")
    ax.set_ylabel(column)
    fig.suptitle("")
    fig.tight_layout()

    return fig, ax


def plot_top_pairs_boxplot(
    df,
    value_col="price_per_kg",
    reporter_col="reporter_country_name",
    partner_col="partner_country_name",
    year_col="year",
    earliest_year=None,
    latest_year=None,
    commodity_code_col="commodity_code",
    top_n=10,
    color="#D9B08C",
    figsize=(16, 7),
    title=None,
):
    """
    Boxplot of value_col for the top-N reporter-partner pairs ranked by std.
    """
    work_df = df.copy()

    if earliest_year is None:
        earliest_year = work_df[year_col].min()
    if latest_year is None:
        latest_year = work_df[year_col].max()

    work_df = work_df[work_df[year_col].between(earliest_year, latest_year)]

    top_pairs = (
        work_df.groupby([reporter_col, partner_col])[value_col]
        .agg(["median", "std"])
        .sort_values("std", ascending=False)
        .head(top_n)
        .reset_index()
    )

    pair_set = set(zip(top_pairs[reporter_col], top_pairs[partner_col]))

    plot_df = work_df[
        work_df[[reporter_col, partner_col]].apply(tuple, axis=1).isin(pair_set)
    ].copy()

    plot_df["pair"] = plot_df[reporter_col] + " -> " + plot_df[partner_col]

    order = top_pairs.assign(
        pair=top_pairs[reporter_col] + " -> " + top_pairs[partner_col]
    )["pair"].tolist()

    hs_codes = work_df[commodity_code_col].dropna().unique()
    auto_title = (
        f"Top {top_n} reporter-partner pairs by {value_col} variability "
        f"for HS code(s): {', '.join(map(str, hs_codes))} "
        f"({earliest_year}–{latest_year})"
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=plot_df,
        x="pair",
        y=value_col,
        order=order,
        ax=ax,
        showmeans=True,
        color=color,
        meanprops={"marker": "o", "markerfacecolor": "white", "markersize": 7},
        medianprops={"color": "red", "linewidth": 2},
    )

    sns.stripplot(
        data=plot_df,
        x="pair",
        y=value_col,
        order=order,
        ax=ax,
        color="black",
        alpha=0.35,
        size=3,
        jitter=0.25,
    )

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="Mean",
        ),
        Line2D([], [], color="red", linewidth=2, label="Median"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", title="Markers")

    ax.set_title(title or auto_title)
    ax.set_xlabel("")
    ax.set_ylabel(value_col)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    return fig, ax


def plot_pair_price_over_years_interactive(
    df,
    reporter,
    partner,
    value_col="price_per_kg",
    reporter_col="reporter_country_name",
    partner_col="partner_country_name",
    year_col="year",
    commodity_code_col="commodity_code",
    earliest_year=None,
    latest_year=None,
    color="steelblue",
    draw_trend=False,
    show_points=True,
    show_outliers=True,
    fig_height=550,
    title=None,
):
    """
    Interactive Plotly yearly boxplot for one or more reporter-partner pairs.

    Parameters
    ----------
    reporter : str or list of str
        Reporter country name(s) to include.
    partner : str or list of str
        Partner country name(s) to include.
    draw_trend : bool
        Overlay a median trend line across years.
    show_points : bool
        Show individual non-outlier data points.
    show_outliers : bool
        Show outliers (in red) when True, hide them when False.
    """
    reporters = [reporter] if isinstance(reporter, str) else reporter
    partners = [partner] if isinstance(partner, str) else partner

    subset = df[
        df[reporter_col].isin(reporters)
        & df[partner_col].isin(partners)
        & df[value_col].notna()
    ].copy()

    if earliest_year is not None:
        subset = subset[subset[year_col] >= earliest_year]
    if latest_year is not None:
        subset = subset[subset[year_col] <= latest_year]

    if subset.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data for {reporters} → {partners}",
            showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
        )
        fig.update_layout(height=fig_height)
        return fig

    subset[year_col] = subset[year_col].astype(int)

    # IQR-based outlier flag per year
    q1 = subset.groupby(year_col)[value_col].transform(lambda s: s.quantile(0.25))
    q3 = subset.groupby(year_col)[value_col].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    subset["_is_outlier"] = (subset[value_col] < (q1 - 1.5 * iqr)) | (
        subset[value_col] > (q3 + 1.5 * iqr)
    )

    year_order = sorted(subset[year_col].unique().tolist())
    fig = go.Figure()

    for year in year_order:
        yr_data = subset[subset[year_col] == year]
        normal = yr_data[~yr_data["_is_outlier"]]
        outliers = yr_data[yr_data["_is_outlier"]]
        box_data = yr_data if show_outliers else normal

        fig.add_trace(go.Box(
            x=[str(year)] * len(box_data),
            y=box_data[value_col],
            name=str(year),
            boxmean=True,
            boxpoints=False,
            marker_color=color,
            line_color=color,
            showlegend=False,
        ))

        if show_points and not normal.empty:
            fig.add_trace(go.Scatter(
                x=[str(year)] * len(normal),
                y=normal[value_col],
                mode="markers",
                marker=dict(color="black", size=4, opacity=0.3),
                showlegend=False,
                hoverinfo="y",
            ))

        if show_outliers and not outliers.empty:
            fig.add_trace(go.Scatter(
                x=[str(year)] * len(outliers),
                y=outliers[value_col],
                mode="markers",
                marker=dict(color="red", size=5, opacity=0.85),
                name="Outlier" if year == year_order[0] else None,
                showlegend=year == year_order[0],
                hoverinfo="y",
            ))

    if draw_trend:
        trend = subset.groupby(year_col)[value_col].median().reindex(year_order)
        fig.add_trace(go.Scatter(
            x=[str(y) for y in year_order],
            y=trend.values,
            mode="lines+markers",
            line=dict(color="darkorange", width=2),
            marker=dict(size=7),
            name="Yearly median",
        ))

    hs_codes = subset[commodity_code_col].dropna().unique()
    pair_label = ", ".join(
        f"{r} → {p}" for r in reporters for p in partners
    )
    auto_title = (
        f"{value_col} by year — {pair_label} "
        f"| HS {', '.join(map(str, hs_codes))} "
        f"({min(year_order)}–{max(year_order)})"
    )

    y_title = "Price($) / kg" if value_col == "price_per_kg" else value_col
    fig.update_layout(
        title=title or auto_title,
        yaxis_title=y_title,
        xaxis_title="Year",
        xaxis=dict(categoryorder="array", categoryarray=[str(y) for y in year_order]),
        height=fig_height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def plot_pair_price_over_years(
    df,
    reporter,
    partner,
    value_col="price_per_kg",
    reporter_col="reporter_country_name",
    partner_col="partner_country_name",
    year_col="year",
    commodity_code_col="commodity_code",
    earliest_year=None,
    latest_year=None,
    color="#3274A8",
    draw_trend=False,
    figsize=(14, 7),
    title=None,
):
    """
    Plot yearly boxplots for a single reporter-partner pair, with optional median trend.
    """
    
    subset = df[
        (df[reporter_col] == reporter)
        & (df[partner_col] == partner)
        & (df[value_col].notna())
    ].copy()

    if earliest_year is not None:
        subset = subset[subset[year_col] >= earliest_year]
    if latest_year is not None:
        subset = subset[subset[year_col] <= latest_year]

    if subset.empty:
        raise ValueError(
            f"No rows found for pair {reporter} -> {partner} in requested year range."
        )

    subset[year_col] = subset[year_col].astype(int)

    q1 = subset.groupby(year_col)[value_col].transform(lambda s: s.quantile(0.25))
    q3 = subset.groupby(year_col)[value_col].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    subset["is_outlier"] = (subset[value_col] < (q1 - 1.5 * iqr)) | (
        subset[value_col] > (q3 + 1.5 * iqr)
    )

    year_order = sorted(subset[year_col].unique().tolist())

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=subset,
        x=year_col,
        y=value_col,
        order=year_order,
        ax=ax,
        color=color,
        showfliers=False,
    )

    sns.stripplot(
        data=subset,
        x=year_col,
        y=value_col,
        order=year_order,
        ax=ax,
        color="black",
        alpha=0.25,
        size=3,
        jitter=0.2,
    )

    outliers = subset[subset["is_outlier"]]
    if not outliers.empty:
        sns.stripplot(
            data=outliers,
            x=year_col,
            y=value_col,
            order=year_order,
            ax=ax,
            color="red",
            alpha=0.9,
            size=4,
            jitter=0.2,
        )

    legend_handles = [
        Line2D(
            [], [], marker="o", linestyle="None", color="red", label="Outlier points"
        )
    ]

    if draw_trend:
        trend = subset.groupby(year_col)[value_col].median().reindex(year_order)
        ax.plot(
            range(len(year_order)),
            trend.values,
            color="darkorange",
            marker="o",
            linewidth=2,
        )
        legend_handles.insert(
            0,
            Line2D(
                [],
                [],
                color="darkorange",
                marker="o",
                linewidth=2,
                label="Yearly median trend",
            ),
        )

    start_year = min(year_order)
    end_year = max(year_order)
    hs_codes = subset[commodity_code_col].dropna().unique()

    auto_title = (
        f"{value_col} distribution by year for {reporter} -> {partner} "
        f"for HS code(s) {', '.join(map(str, hs_codes))} "
        f"({start_year}-{end_year})"
    )

    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_title(title or auto_title)
    ax.set_xlabel("Year")
    ax.set_ylabel(value_col)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    return fig, ax


def plot_distribution_by_group(
    df,
    group_col,
    value_col="price_per_kg",
    show_points=True,
    figsize=(14, 7),
    title=None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = df[[group_col, value_col]].dropna()

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=plot_df,
        x=group_col,
        y=value_col,
        ax=ax,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markersize": 6},
        medianprops={"color": "red", "linewidth": 2},
    )

    if show_points:
        sns.stripplot(
            data=plot_df,
            x=group_col,
            y=value_col,
            ax=ax,
            color="black",
            alpha=0.3,
            size=3,
            jitter=0.25,
        )

    ax.set_title(title or f"{value_col} distribution by {group_col}")
    ax.set_xlabel("")
    ax.set_ylabel(value_col)
    ax.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    return fig, ax


def plot_interactive_distribution_by_group(
    df,
    group_col,
    value_col="price_per_kg",
    marker_color="steelblue",
    marker_size=6,
    marker_opacity=0.4,
    line_color="steelblue",
    show_points=True,
    fig_height=550,
    title=None,
):
    """
    Interactive plotly boxplot of value_col distribution grouped by group_col.
    """
    import plotly.graph_objects as go

    plot_df = df[[group_col, value_col]].dropna()
    groups = sorted(plot_df[group_col].unique())

    fig = go.Figure()

    for group in groups:
        group_data = plot_df[plot_df[group_col] == group][value_col]
        fig.add_trace(go.Box(
            y=group_data,
            name=str(group),
            boxmean=True,
            boxpoints="all" if show_points else False,
            jitter=0.3,
            pointpos=0,
            marker=dict(size=marker_size, opacity=marker_opacity, color=marker_color),
            marker_color=marker_color,
            line_color=line_color,
        ))

    if value_col == "price_per_kg":
        y_title = "Price($) / kg"
    else:
        y_title = value_col

    fig.update_layout(
        title=title or f"{value_col} distribution by {group_col}",
        yaxis_title=y_title,
        xaxis_title="",
        showlegend=False,
        xaxis_tickangle=-45,
        height=fig_height,
    )

    return fig
