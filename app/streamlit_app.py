import logging
import inspect
from pathlib import Path
from time import perf_counter
from vacoda.utils.logger import setup_logging

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from helpers import (
    apply_trade_filters,
    get_filter_options,
    make_top_pairs_boxplot,
    make_weight_ticks,
    preprocess_data,
)
from vacoda.visualization.trade_patterns import (
    plot_interactive_distribution_by_group,
    plot_pair_price_over_years_interactive,
)

setup_logging(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

st.set_page_config(page_title="UN-ComTrade Explorer", layout="wide")

st.title("UN-ComTrade Explorer")
# st.caption("Interactive exploration of trade patterns and price variability")


def _extract_commodity_description(df: pd.DataFrame) -> str | None:
    """Extract a single commodity description from known column names."""
    candidates = ["commodity_description", "commodity_desc", "hs_description", "product_description"]
    for col in candidates:
        if col in df.columns:
            series = (
                df[col]
                .dropna()
                .astype("string")
                .str.strip()
                .loc[lambda s: s != ""]
            )
            if not series.empty:
                return str(series.iloc[0])
    return None


def _mask_iqr_outliers(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.Series:
    """Return True for non-outlier rows using per-group IQR fences."""
    work = df[group_cols + [value_col]].copy()
    valid = work[group_cols + [value_col]].notna().all(axis=1)
    mask = pd.Series(False, index=df.index)
    if not valid.any():
        return mask

    q1 = work.loc[valid].groupby(group_cols, observed=True)[value_col].transform(lambda s: s.quantile(0.25))
    q3 = work.loc[valid].groupby(group_cols, observed=True)[value_col].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    inlier_valid = (work.loc[valid, value_col] >= lower) & (work.loc[valid, value_col] <= upper)
    mask.loc[valid] = inlier_valid.values
    return mask


# --------------------------------------------------
# Load data
# --------------------------------------------------
DEFAULT_PARQUET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "Comtrade_441012_processed.parquet"
)


@st.cache_data
def load_data():
    if not DEFAULT_PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Default dashboard parquet not found: {DEFAULT_PARQUET_PATH}. "
            "Please run the pipeline or upload a parquet in the sidebar."
        )
    df_default = pd.read_parquet(DEFAULT_PARQUET_PATH)
    return df_default

# Optional upload: if provided, use uploaded parquet; otherwise use fixed default parquet.
uploaded_file = st.sidebar.file_uploader(
    "Upload parquet file",
    type=["parquet"],
    help="Optional: upload a processed parquet file to explore instead of the default dataset.",
)

if uploaded_file is not None:
    upload_signature = (uploaded_file.name, uploaded_file.size)
    try:
        if st.session_state.get("uploaded_signature") != upload_signature:
            with st.spinner("Loading uploaded parquet..."):
                t0_load = perf_counter()
                df_uploaded_raw = pd.read_parquet(uploaded_file)
                st.session_state["commodity_description"] = _extract_commodity_description(df_uploaded_raw)
                df_uploaded = preprocess_data(df_uploaded_raw)
                st.session_state["uploaded_df"] = df_uploaded
                st.session_state["uploaded_signature"] = upload_signature
                st.session_state["load_seconds"] = perf_counter() - t0_load
                st.session_state["load_source"] = "uploaded parquet (new upload)"
        df = st.session_state["uploaded_df"]
        if "load_seconds" not in st.session_state:
            st.session_state["load_seconds"] = 0.0
            st.session_state["load_source"] = "uploaded parquet (cached)"
        st.sidebar.success(f"Using uploaded file: {uploaded_file.name}")
    except Exception as exc:
        st.sidebar.error(f"Failed to read uploaded parquet: {exc}")
        st.stop()
else:
    st.session_state.pop("uploaded_df", None)
    st.session_state.pop("uploaded_signature", None)
    t0_load = perf_counter()
    df_raw = load_data()
    st.session_state["commodity_description"] = _extract_commodity_description(df_raw)
    df = preprocess_data(df_raw)
    st.session_state["load_seconds"] = perf_counter() - t0_load
    st.session_state["load_source"] = f"default fixed dataset ({DEFAULT_PARQUET_PATH.name})"

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Filters")
st.sidebar.caption("Adjust filters, then click Apply to update charts.")

filter_options = get_filter_options(df)

with st.sidebar.form("filter_form", clear_on_submit=False):
    commodity_code = st.selectbox(
        "Commodity code",
        options=filter_options["commodity_codes"],
    )

    value_col = st.selectbox(
        "Value column",
        options=filter_options["numeric_columns"],
        index=filter_options["numeric_columns"].index("price_per_kg")
        if "price_per_kg" in filter_options["numeric_columns"]
        else 0,
    )

    year_range = st.slider(
        "Year range",
        min_value=int(filter_options["min_year"]),
        max_value=int(filter_options["max_year"]),
        value=(int(filter_options["min_year"]), int(filter_options["max_year"])),
    )

    _weight_ticks = make_weight_ticks(df["netweight_kg"].max())
    weight_range = st.select_slider(
        "Net weight (kg)",
        options=_weight_ticks,
        value=(_weight_ticks[0], _weight_ticks[-1]),
        format_func=lambda v: f"{v:,}",
    )

    flow_option = st.multiselect(
        "Flow",
        options=sorted(df["flow_description"].dropna().unique().tolist()),
        default=[],
    )

    estimation_option = st.selectbox(
        "Estimated rows",
        options=["All", "Only non-estimated", "Only estimated"],
        index=0,
    )

    top_n = st.slider(
        "Top N pairs",
        min_value=3,
        max_value=20,
        value=10,
    )

    reporter_country = st.multiselect(
        "Reporter country",
        options=sorted(df["reporter_country_name"].dropna().unique().tolist()),
        default=["Bahamas"],
    )

    partner_country = st.multiselect(
        "Partner country",
        options=sorted(df["partner_country_name"].dropna().unique().tolist()),
        default=["USA"],
    )

    group_col = st.selectbox(
        "Group by", ["reporter_continent", "partner_continent", "flow_description"]
    )

    show_points = st.checkbox("Overlay points", value=True)
    hide_outliers = st.checkbox("Hide outliers (IQR)", value=False)
    show_year_trend = st.checkbox("Show trend line (yearly median)", value=True)
    st.form_submit_button("Apply filters")

# --------------------------------------------------
# Filter data
# --------------------------------------------------
t0_filter = perf_counter()
filtered_df = apply_trade_filters(
    df=df,
    commodity_code=commodity_code,
    earliest_year=year_range[0],
    latest_year=year_range[1],
    flow_option=flow_option,
    estimation_option=estimation_option,
    min_weight=weight_range[0],
    max_weight=weight_range[1],
)
filter_seconds = perf_counter() - t0_filter

# Summary dataframe: align with outlier-hide mode so top metrics reflect what user sees.
summary_df = filtered_df
if hide_outliers and not filtered_df.empty:
    summary_inlier_mask = _mask_iqr_outliers(filtered_df, value_col=value_col, group_cols=["year"])
    summary_df = filtered_df.loc[summary_inlier_mask].copy()

# Guardrail: plotting every point can freeze UI for large selections.
max_rows_for_points = 60_000
plot_show_points = show_points and (len(filtered_df) <= max_rows_for_points)
if show_points and not plot_show_points:
    st.info(
        f"Point overlay automatically disabled for {len(filtered_df):,} rows "
        f"(limit: {max_rows_for_points:,}) to keep the app responsive."
    )

if hide_outliers:
    st.info("Outliers are hidden using 1.5×IQR fences per chart grouping.")

st.subheader("Filtered data summary")

# Baseline: commodity code is always selected, so compare against that slice only
_baseline_df = df[df["commodity_code"] == commodity_code]

_commodity_desc = st.session_state.get("commodity_description")
if _commodity_desc:
    st.caption(f"Commodity: {_commodity_desc}")
else:
    st.caption(f"Selected commodity code: {commodity_code}")

_total_rows = len(_baseline_df)
_filtered_rows = len(summary_df)
_row_pct = _filtered_rows / _total_rows * 100 if _total_rows > 0 else 0.0

_total_reporters = _baseline_df["reporter_country_name"].nunique()
_filtered_reporters = summary_df["reporter_country_name"].nunique() if not summary_df.empty else 0

_total_partners = _baseline_df["partner_country_name"].nunique()
_filtered_partners = summary_df["partner_country_name"].nunique() if not summary_df.empty else 0

_reporter_pct = (_filtered_reporters / _total_reporters * 100) if _total_reporters > 0 else 0.0
_partner_pct = (_filtered_partners / _total_partners * 100) if _total_partners > 0 else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{_filtered_rows:,}", f"{_row_pct:.1f}% of total", delta_color="off")
col2.metric("Reporter countries", f"{_filtered_reporters:,}", f"{_reporter_pct:.1f}% of total", delta_color="off")
col3.metric("Partner countries", f"{_filtered_partners:,}", f"{_partner_pct:.1f}% of total", delta_color="off")
if hide_outliers:
    st.caption("Summary reflects outlier-hidden view (IQR by year).")

# st.sidebar.markdown("---")
# st.sidebar.subheader("Live filtered summary")
# st.sidebar.metric("Rows", f"{_filtered_rows:,}", f"{_row_pct:.1f}% of total", delta_color="off")
# st.sidebar.metric(
#     "Reporter countries",
#     f"{_filtered_reporters:,}",
#     f"{_reporter_pct:.1f}% of total",
#     delta_color="off",
# )
# st.sidebar.metric(
#     "Partner countries",
#     f"{_filtered_partners:,}",
#     f"{_partner_pct:.1f}% of total",
#     delta_color="off",
# )

# --------------------------------------------------
# Plot- Value distribution by year
# --------------------------------------------------
st.subheader(f"{value_col} distribution by year")

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    t0_plot = perf_counter()
    hover_metric_cols = ["netweight_kg", "price_per_kg", "value_usd"]
    hover_label_cols = ["reporter_country_name", "partner_country_name"]
    plot_cols = []
    for c in ["year", value_col] + [c for c in hover_metric_cols + hover_label_cols if c in filtered_df.columns]:
        if c not in plot_cols:
            plot_cols.append(c)
    year_plot_df = filtered_df[plot_cols].dropna(subset=["year", value_col]).copy()
    if value_col == "price_per_kg":
        year_plot_df = year_plot_df[year_plot_df["price_per_kg"] > 0].copy()
    if hide_outliers and not year_plot_df.empty:
        inlier_mask = _mask_iqr_outliers(year_plot_df, value_col=value_col, group_cols=["year"])
        year_plot_df = year_plot_df.loc[inlier_mask].copy()

    #if value_col == "price_per_kg":
    #    st.caption("For readability, rows with price_per_kg <= 0 are excluded from this panel.")

    if year_plot_df.empty:
        st.warning("No positive values available for the selected settings.")
        year_dist_plot_seconds = perf_counter() - t0_plot
        density_fig = go.Figure()
        density_fig.add_annotation(
            text="No data after filtering",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        density_fig.update_layout(height=550)
        fig = go.Figure()
        fig.add_annotation(
            text="No data after filtering",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(height=550)
        col_density, col_year = st.columns([1, 3], gap="medium")
        with col_density:
            st.plotly_chart(density_fig, width="stretch")
        with col_year:
            st.plotly_chart(fig, width="stretch")
    else:
        year_plot_df["year"] = year_plot_df["year"].astype(int).astype(str)
        year_order = sorted(year_plot_df["year"].unique().tolist(), key=int)

        hover_data = {"year": False}
        for c in hover_metric_cols:
            if c in year_plot_df.columns and c != value_col:
                hover_data[c] = ":,.4f" if c == "price_per_kg" else ":,.2f"
        for c in hover_label_cols:
            if c in year_plot_df.columns:
                hover_data[c] = True

        fig = px.box(
            year_plot_df,
            x="year",
            y=value_col,
            points="all" if plot_show_points else False,
            hover_data=hover_data,
            category_orders={"year": year_order},
            title=f"{value_col} distribution by year",
        )

        if show_year_trend and not year_plot_df.empty:
            year_median = (
                year_plot_df.groupby("year", observed=True)[value_col]
                .median()
                .reindex(year_order)
            )
            fig.add_trace(
                go.Scatter(
                    x=year_order,
                    y=year_median.values,
                    mode="lines+markers",
                    name="Yearly median",
                    line=dict(color="darkorange", width=2),
                    marker=dict(size=7),
                )
            )

        # Jitter/pointpos are valid only for Box traces (not Scatter trend traces).
        fig.update_traces(
            marker=dict(size=4, opacity=0.30),
            jitter=0.30,
            pointpos=0,
            selector=dict(type="box"),
        )
        fig.update_traces(
            marker=dict(size=7),
            selector=dict(type="scatter"),
        )
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title=("Price($) / kg" if value_col == "price_per_kg" else value_col),
            showlegend=show_year_trend,
            xaxis_tickangle=-45,
            height=550,
        )

        density_series = pd.to_numeric(year_plot_df[value_col], errors="coerce").dropna()
        histogram_kwargs = {}
        if value_col == "price_per_kg":
            density_series = density_series[density_series > 0]
            if not density_series.empty:
                min_x = float(density_series.min())
                max_x = float(density_series.max())
                if max_x > min_x:
                    bin_size = (max_x - min_x) / 60
                else:
                    # Keep a non-zero bin size for nearly-constant positive values.
                    bin_size = max(min_x / 10, 1e-6)
                histogram_kwargs["xbins"] = dict(start=min_x, end=max_x + bin_size, size=bin_size)

        density_fig = go.Figure()
        density_fig.add_trace(
            go.Histogram(
                x=density_series,
                histnorm="probability density",
                nbinsx=60,
                marker=dict(color="steelblue"),
                opacity=0.75,
                hovertemplate=f"{value_col}: %{{x:,.6f}}<br>Density: %{{y:.4f}}<extra></extra>",
                **histogram_kwargs,
            )
        )
        density_fig.update_layout(
            title=f"{value_col} density",
            xaxis_title=("Price($) / kg" if value_col == "price_per_kg" else value_col),
            yaxis_title="Density",
            showlegend=False,
            height=550,
            bargap=0.03,
        )

        year_dist_plot_seconds = perf_counter() - t0_plot
        col_density, col_year = st.columns([1, 3], gap="medium")
        with col_density:
            st.plotly_chart(density_fig, width="stretch")
        with col_year:
            st.plotly_chart(fig, width="stretch")
if filtered_df.empty:
    year_dist_plot_seconds = 0.0

st.markdown("---")


# --------------------------------------------------
# Plot- Top pairs variability
# --------------------------------------------------
st.subheader("Top reporter-partner pairs by variability")

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    t0_plot = perf_counter()
    top_pairs_df = filtered_df
    if hide_outliers and not filtered_df.empty:
        inlier_mask = _mask_iqr_outliers(
            filtered_df,
            value_col=value_col,
            group_cols=["reporter_country_name", "partner_country_name"],
        )
        top_pairs_df = filtered_df.loc[inlier_mask].copy()
    fig = make_top_pairs_boxplot(
        df=top_pairs_df,
        value_col=value_col,
        weight_col="netweight_kg",
        top_n=top_n,
        show_points=plot_show_points,
    )
    top_pairs_plot_seconds = perf_counter() - t0_plot
    st.plotly_chart(fig, width="stretch")
if filtered_df.empty:
    top_pairs_plot_seconds = 0.0

st.markdown("---")


# --------------------------------------------------
# Plot- Pair price over years
# --------------------------------------------------
st.subheader(f"{value_col} by year — selected pairs")

if not reporter_country or not partner_country:
    st.info("Select at least one reporter and one partner country to show yearly distribution.")
elif filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    t0_plot = perf_counter()
    pair_kwargs = dict(
        df=filtered_df,
        reporter=reporter_country,
        partner=partner_country,
        value_col=value_col,
        earliest_year=year_range[0],
        latest_year=year_range[1],
        show_points=plot_show_points,
        draw_trend=True,
    )
    # Backward compatibility: only pass show_outliers if the loaded function supports it.
    if "show_outliers" in inspect.signature(plot_pair_price_over_years_interactive).parameters:
        pair_kwargs["show_outliers"] = not hide_outliers
    fig = plot_pair_price_over_years_interactive(**pair_kwargs)
    pair_year_plot_seconds = perf_counter() - t0_plot
    st.plotly_chart(fig, width="stretch")
if (not reporter_country or not partner_country) or filtered_df.empty:
    pair_year_plot_seconds = 0.0

st.markdown("---")


# --------------------------------------------------
# Plot- Distribution by group
# --------------------------------------------------
st.subheader(f"{value_col} distribution by {group_col}")

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    t0_plot = perf_counter()
    group_df = filtered_df
    if hide_outliers and not filtered_df.empty:
        inlier_mask = _mask_iqr_outliers(filtered_df, value_col=value_col, group_cols=[group_col])
        group_df = filtered_df.loc[inlier_mask].copy()
    fig = plot_interactive_distribution_by_group(
        group_df,
        group_col=group_col,
        value_col=value_col,
        show_points=plot_show_points,
        marker_color="steelblue",
        marker_opacity=0.9,
        marker_size=6,
    )
    group_plot_seconds = perf_counter() - t0_plot
    st.plotly_chart(fig, width="stretch")
if filtered_df.empty:
    group_plot_seconds = 0.0

st.sidebar.markdown("---")
with st.sidebar.expander("Performance", expanded=False):
    st.caption("Measured on this rerun")
    st.write(f"Data source: {st.session_state.get('load_source', 'n/a')}")
    st.metric("Load data", f"{st.session_state.get('load_seconds', 0.0):.2f}s")
    st.metric("Apply filters", f"{filter_seconds:.2f}s")
    st.metric("Plot: year distribution", f"{year_dist_plot_seconds:.2f}s")
    st.metric("Plot: top pairs", f"{top_pairs_plot_seconds:.2f}s")
    st.metric("Plot: pair by year", f"{pair_year_plot_seconds:.2f}s")
    st.metric("Plot: distribution by group", f"{group_plot_seconds:.2f}s")

logger.info(
    "PERF | source=%s | load=%.2fs | filter=%.2fs | year_dist=%.2fs | top_pairs=%.2fs | pair_year=%.2fs | group=%.2fs | rows=%s",
    st.session_state.get("load_source", "n/a"),
    float(st.session_state.get("load_seconds", 0.0)),
    float(filter_seconds),
    float(year_dist_plot_seconds),
    float(top_pairs_plot_seconds),
    float(pair_year_plot_seconds),
    float(group_plot_seconds),
    f"{_filtered_rows:,}",
)
