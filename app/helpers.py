"""
Helper functions for dashboard data processing and visualization.
"""
from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go
import pandas as pd
from vacoda.io.loaders import load_processed_trade_data
from vacoda.utils.logger import get_logger

logger = get_logger("helpers")


def _coerce_estimated_mask(series: pd.Series) -> pd.Series:
    """Return boolean mask where True means row is estimated.

    Supports common encodings such as bool, 0/1 numeric, and string tokens.
    """
    mask = pd.Series(False, index=series.index, dtype=bool)

    if series.empty:
        return mask

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_valid = numeric.notna()
    if numeric_valid.any():
        mask.loc[numeric_valid] = numeric.loc[numeric_valid] > 0

    if (~numeric_valid).any():
        text = series.astype("string").str.strip().str.lower()
        estimated_tokens = {"y", "yes", "true", "t", "estimated", "est", "e"}
        mask.loc[~numeric_valid] = text.loc[~numeric_valid].isin(estimated_tokens)

    return mask


def make_weight_ticks(max_val: float) -> list[int]:
    """
    Build a log-spaced list of round numbers from 0 up to max_val.
    Steps follow a 1-2-5 sequence per decade, e.g.:
      0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, ...
    The actual data maximum is always appended as the final tick.
    """
    if max_val <= 0:
        return [0, 1]
    ticks: list[int] = [0]
    max_exp = math.floor(math.log10(max_val))
    for exp in range(0, max_exp + 1):
        for mantissa in (1, 2, 5):
            val = mantissa * 10 ** exp
            if val <= max_val:
                ticks.append(int(val))
    ticks.append(math.ceil(max_val))
    return sorted(set(ticks))

def load_dashboard_data():
    """
    Load data optimized for dashboard usage.
    Drops rows where netweight_kg is missing or non-positive,
    as the dashboard is not designed to handle unresolved weight cases.
    """
    df = load_processed_trade_data()

    ## Preproess data
    df = preprocess_data(df)
    logger.debug(f"Preprocessed data shape: {df.shape}")
    return df


def preprocess_data(df):
    """
    Apply any additional preprocessing steps needed for the dashboard.
    Currently a placeholder for future transformations.
    """
    ## 1. Drop rows with missing or non-positive netweight_kg, as the dashboard relies on this for filtering and visualization.
    df = df[df["netweight_kg"].notna() & (df["netweight_kg"] > 0)].reset_index(drop=True)
    logger.debug(f"After dropping rows with missing/non-positive netweight_kg: {len(df):,} rows")

    ## 2. Select only necessary columns for the dashboard to optimize performance. This can be adjusted based on actual dashboard needs.
    columns_to_keep = [
        "reporter_country_name",
        "partner_country_name",
        "reporter_continent",
        "partner_continent",
        "value_usd",
        "year",
        "commodity_code",
        "commodity_description",
        "netweight_kg",
        "price_per_kg",
        "flow_description",
        "estimation_code",
        #"partner_reliability",
    ]
    available_columns = [c for c in columns_to_keep if c in df.columns]
    df = df[available_columns]
    logger.debug(f"After selecting relevant columns: {len(df):,} rows, {len(columns_to_keep)} columns")

    return df


def get_filter_options(df) -> dict[str, Any]:
    """
    Build sidebar options from the dataframe.
    """
    numeric_columns = [
        col for col in df.columns
        if df[col].dtype.kind in {"i", "u", "f"}
    ]

    return {
        "commodity_codes": sorted(df["commodity_code"].dropna().unique().tolist()),
        "min_year": int(df["year"].min()),
        "max_year": int(df["year"].max()),
        "numeric_columns": numeric_columns,
    }


def apply_trade_filters(
    df,
    commodity_code=None,
    earliest_year=None,
    latest_year=None,
    flow_option=None,
    estimation_option="All",
    min_weight=None,
    max_weight=None,
):
    """
    Apply dashboard filters to the trade dataframe.
    """
    out = df.copy()

    if commodity_code is not None:
        out = out[out["commodity_code"] == commodity_code]
    logger.debug(f"After commodity_code filter: {len(out):,} rows")
    if earliest_year is not None:
        out = out[out["year"] >= earliest_year]
    logger.debug(f"After earliest_year filter: {len(out):,} rows")
    if latest_year is not None:
        out = out[out["year"] <= latest_year]
    logger.debug(f"After latest_year filter: {len(out):,} rows")

    # Flow filter — flow_option is a list of selected flow_description values
    if flow_option:
        out = out[out["flow_description"].isin(flow_option)]
    logger.debug(f"After flow filter: {len(out):,} rows")
    # Estimation filter
    estimation_mask = None
    if "is_estimated" in out.columns:
        estimation_mask = _coerce_estimated_mask(out["is_estimated"])
    elif "estimation_code" in out.columns:
        estimation_mask = _coerce_estimated_mask(out["estimation_code"])

    if estimation_option == "Only non-estimated" and estimation_mask is not None:
        out = out.loc[~estimation_mask].copy()
    elif estimation_option == "Only estimated" and estimation_mask is not None:
        out = out.loc[estimation_mask].copy()
    logger.debug(f"After estimation filter: {len(out):,} rows")
    if min_weight is not None and min_weight > 0 and "netweight_kg" in out.columns:
        out = out[out["netweight_kg"] >= min_weight]
    logger.debug(f"After min_weight filter: {len(out):,} rows")

    if max_weight is not None and "netweight_kg" in out.columns and out["netweight_kg"].max() > max_weight:
        out = out[out["netweight_kg"] <= max_weight]
    logger.debug(f"After max_weight filter: {len(out):,} rows")

    return out


def prepare_top_pairs_data(
    df,
    value_col="price_per_kg",
    reporter_col="reporter_country_name",
    partner_col="partner_country_name",
    top_n=10,
):
    """
    Prepare data for top reporter-partner variability boxplot.
    """
    required_cols = [reporter_col, partner_col, value_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = df.dropna(subset=[reporter_col, partner_col, value_col]).copy()

    if work_df.empty:
        return work_df, []

    top_pairs = (
        work_df.groupby([reporter_col, partner_col], observed=True)[value_col]
        .agg(["median", "std"])
        .sort_values("std", ascending=False)
        .head(top_n)
        .reset_index()
    )

    # Vectorized join is much faster than row-wise tuple/apply for larger datasets.
    pair_keys = top_pairs[[reporter_col, partner_col]].drop_duplicates()
    plot_df = work_df.merge(pair_keys, on=[reporter_col, partner_col], how="inner").copy()

    plot_df["pair"] = plot_df[reporter_col] + " -> " + plot_df[partner_col]

    order = (
        top_pairs.assign(pair=top_pairs[reporter_col] + " -> " + top_pairs[partner_col])["pair"]
        .tolist()
    )

    return plot_df, order


def make_top_pairs_boxplot(
    df,
    value_col="price_per_kg",
    reporter_col="reporter_country_name",
    partner_col="partner_country_name",
    commodity_code_col="commodity_code",
    weight_col="netweight_kg",
    year_col="year",
    marker_size=6,
    marker_opacity=0.4,
    top_n=10,
    show_points=True,
):
    """
    Create an interactive plotly boxplot for top reporter-partner pairs by variability.
    """
    plot_df, order = prepare_top_pairs_data(
        df=df,
        value_col=value_col,
        reporter_col=reporter_col,
        partner_col=partner_col,
        top_n=top_n,
    )

    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available for selected filters", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    earliest_year = int(df[year_col].min())
    latest_year = int(df[year_col].max())
    hs_codes = sorted(df[commodity_code_col].dropna().astype(str).unique().tolist())

    title = (
        f"Top {top_n} reporter-partner pairs by {value_col} variability "
        f"for hs_code: {', '.join(hs_codes)} "
        f"({earliest_year}-{latest_year})"
    )

    fig = go.Figure()

    # Show the companion column in the hover (whichever is not on the y-axis)
    if value_col == weight_col:
        companion_col = "price_per_kg"
    else:
        companion_col = weight_col
    companion_label = companion_col.replace("_", " ")

    for pair in order:
        pair_rows = plot_df[plot_df["pair"] == pair]
        pair_data = pair_rows[value_col]
        companion_data = pair_rows[companion_col].values if companion_col in pair_rows.columns else None
        fig.add_trace(go.Box(
            y=pair_data,
            name=pair,
            boxmean=True,
            marker_color="steelblue",
            line_color="steelblue",
            boxpoints="all" if show_points else False,
            jitter=0.3,
            pointpos=0,
            marker=dict(size=marker_size, opacity=marker_opacity),
            customdata=companion_data,
            hovertemplate=(
                f"{value_col}: %{{y:,.4f}}<br>"
                f"{companion_label}: %{{customdata:,.2f}}"
                "<extra>%{fullData.name}</extra>"
            ) if companion_data is not None else None,
        ))

    if value_col == "price_per_kg":
        y_title = "Price($) / kg"
    else:
        y_title = value_col

    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        xaxis_title="",
        showlegend=False,
        xaxis_tickangle=-45,
        height=550,
    )

    return fig
