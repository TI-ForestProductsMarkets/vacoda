from __future__ import annotations

import pandas as pd


def add_reporter_year_shares(
    df: pd.DataFrame,
    *,
    reporter_col: str = "ReporterCode",
    year_col: str = "RefYear",
    value_col: str = "primary_value_total",
    weight_col: str = "netwgt_total",
    row_share_col: str = "share_rows_in_reporter_year",
) -> pd.DataFrame:
    """
    Add value and net-weight shares within each reporter-year.

    This enriches the reporter-year-classification table with:

    - share_value_in_reporter_year
    - share_netwgt_in_reporter_year

    Parameters
    ----------
    df:
        Reporter-year-classification coverage table.
    reporter_col:
        Reporter identifier column.
    year_col:
        Year column.
    value_col:
        Trade value column.
    weight_col:
        Net weight column.
    row_share_col:
        Existing row-share column. Kept unchanged if already present.

    Returns
    -------
    pd.DataFrame
        Copy of input dataframe with added share columns.
    """

    out = df.copy()

    group_cols = [reporter_col, year_col]

    if value_col in out.columns:
        total_value = out.groupby(group_cols)[value_col].transform("sum")
        out["share_value_in_reporter_year"] = out[value_col].where(
            total_value.ne(0), pd.NA
        ) / total_value.where(total_value.ne(0), pd.NA)

    if weight_col in out.columns:
        total_weight = out.groupby(group_cols)[weight_col].transform("sum")
        out["share_netwgt_in_reporter_year"] = out[weight_col].where(
            total_weight.ne(0), pd.NA
        ) / total_weight.where(total_weight.ne(0), pd.NA)

    if row_share_col not in out.columns and "n_rows" in out.columns:
        total_rows = out.groupby(group_cols)["n_rows"].transform("sum")
        out[row_share_col] = out["n_rows"] / total_rows

    return out

def build_reporter_hs_timeline(
    df: pd.DataFrame,
    *,
    reporter_col: str = "ReporterCode",
    reporter_iso_col: str = "ReporterISO",
    reporter_name_col: str = "ReporterDesc",
    year_col: str = "RefYear",
    classification_col: str = "ClassificationCode",
    dominance_metric: str = "share_rows_in_reporter_year",
) -> pd.DataFrame:
    """
    Build continuous HS-classification timelines per reporter.

    The function first selects the dominant classification for each
    reporter-year, then compresses consecutive years with the same
    classification into periods.

    Parameters
    ----------
    df:
        Reporter-year-classification coverage table.
    reporter_col:
        Reporter identifier column.
    reporter_iso_col:
        Reporter ISO column.
    reporter_name_col:
        Reporter name column.
    year_col:
        Year column.
    classification_col:
        HS classification system column, e.g. H0, H1, H2.
    dominance_metric:
        Column used to choose the dominant classification in a reporter-year.
        Usually one of:

        - share_rows_in_reporter_year
        - share_value_in_reporter_year
        - share_netwgt_in_reporter_year

    Returns
    -------
    pd.DataFrame
        Timeline table with one row per continuous classification period.
    """

    required = {
        reporter_col,
        reporter_iso_col,
        reporter_name_col,
        year_col,
        classification_col,
        dominance_metric,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dominant = (
        df.sort_values(
            [reporter_col, year_col, dominance_metric],
            ascending=[True, True, False],
        )
        .drop_duplicates([reporter_col, year_col])
        .copy()
    )

    dominant = dominant.sort_values([reporter_col, year_col]).reset_index(drop=True)

    # Start a new period when reporter changes, classification changes,
    # or the years are not consecutive.
    dominant["_prev_reporter"] = dominant[reporter_col].shift()
    dominant["_prev_classification"] = dominant[classification_col].shift()
    dominant["_prev_year"] = dominant[year_col].shift()

    dominant["_new_period"] = (
        dominant[reporter_col].ne(dominant["_prev_reporter"])
        | dominant[classification_col].ne(dominant["_prev_classification"])
        | dominant[year_col].ne(dominant["_prev_year"] + 1)
    )

    dominant["_period_id"] = dominant["_new_period"].cumsum()

    timeline = (
        dominant.groupby(
            [
                reporter_col,
                reporter_iso_col,
                reporter_name_col,
                classification_col,
                "_period_id",
            ],
            dropna=False,
        )
        .agg(
            start_year=(year_col, "min"),
            end_year=(year_col, "max"),
            n_years=(year_col, "nunique"),
        )
        .reset_index()
        .drop(columns="_period_id")
        .sort_values([reporter_col, "start_year"])
        .reset_index(drop=True)
    )

    return timeline


def build_reporter_year_dominant_hs(
    df: pd.DataFrame,
    *,
    reporter_col: str = "ReporterCode",
    reporter_iso_col: str = "ReporterISO",
    reporter_name_col: str = "ReporterDesc",
    year_col: str = "RefYear",
    classification_col: str = "ClassificationCode",
    original_col: str = "IsOriginalClassification",
    rows_col: str = "n_rows",
    cmd_codes_col: str = "n_cmd_codes",
    flows_col: str = "n_flows",
    value_col: str = "primary_value_total",
    weight_col: str = "netwgt_total",
    row_share_col: str = "share_rows_in_reporter_year",
    dominance_metric: str = "share_rows_in_reporter_year",
) -> pd.DataFrame:
    """
    Build one-row-per-reporter-year table with the dominant HS classification.

    This table is useful as a decision layer before HS-sync.

    Parameters
    ----------
    df:
        Reporter-year-classification coverage table.
    dominance_metric:
        Column used to decide the dominant classification.
        Common choices:

        - share_rows_in_reporter_year
        - share_value_in_reporter_year
        - share_netwgt_in_reporter_year

    Returns
    -------
    pd.DataFrame
        Reporter-year decision table.
    """

    enriched = add_reporter_year_shares(
        df,
        reporter_col=reporter_col,
        year_col=year_col,
        value_col=value_col,
        weight_col=weight_col,
        row_share_col=row_share_col,
    )

    required = {
        reporter_col,
        reporter_iso_col,
        reporter_name_col,
        year_col,
        classification_col,
        rows_col,
        cmd_codes_col,
        flows_col,
        row_share_col,
        dominance_metric,
    }
    missing = required - set(enriched.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    group_cols = [reporter_col, year_col]

    totals = (
        enriched.groupby(group_cols, dropna=False)
        .agg(
            ReporterISO=(reporter_iso_col, "first"),
            ReporterDesc=(reporter_name_col, "first"),
            n_classifications=(classification_col, "nunique"),
            n_rows_total=(rows_col, "sum"),
            n_cmd_codes_total=(cmd_codes_col, "sum"),
            n_flows_total=(flows_col, "sum"),
            primary_value_total=(value_col, "sum")
            if value_col in enriched.columns
            else (rows_col, "size"),
            netwgt_total=(weight_col, "sum")
            if weight_col in enriched.columns
            else (rows_col, "size"),
        )
        .reset_index()
    )

    # Select dominant classification per reporter-year.
    dominant = (
        enriched.sort_values(
            [reporter_col, year_col, dominance_metric],
            ascending=[True, True, False],
        )
        .drop_duplicates(group_cols)
        .copy()
    )

    keep_cols = [
        reporter_col,
        year_col,
        classification_col,
        row_share_col,
        dominance_metric,
    ]

    optional_cols = [
        "share_value_in_reporter_year",
        "share_netwgt_in_reporter_year",
        original_col,
    ]

    keep_cols += [c for c in optional_cols if c in dominant.columns]
    keep_cols = list(dict.fromkeys(keep_cols))

    dominant = dominant[keep_cols].rename(
        columns={
            classification_col: "dominant_classification_code",
            row_share_col: "dominant_row_share",
            "share_value_in_reporter_year": "dominant_value_share",
            "share_netwgt_in_reporter_year": "dominant_netwgt_share",
            original_col: "dominant_is_original_classification",
        }
    )

    out = totals.merge(dominant, on=group_cols, how="left")

    out["mixed_hs"] = out["n_classifications"] > 1

    # Simple first-pass risk label.
    out["hs_sync_risk_level"] = "low"

    out.loc[out["mixed_hs"], "hs_sync_risk_level"] = "medium"

    if "dominant_row_share" in out.columns:
        out.loc[out["dominant_row_share"].lt(0.95), "hs_sync_risk_level"] = "high"

    if "dominant_is_original_classification" in out.columns:
        out.loc[
            out["dominant_is_original_classification"].eq(False),
            "hs_sync_risk_level",
        ] = "medium"

    out = out.sort_values([reporter_col, year_col]).reset_index(drop=True)

    return out


def build_hs_transition_events(
    df: pd.DataFrame,
    *,
    reporter_col: str = "ReporterCode",
    reporter_iso_col: str = "ReporterISO",
    reporter_name_col: str = "ReporterDesc",
    year_col: str = "RefYear",
    classification_col: str = "ClassificationCode",
    dominance_metric: str = "share_rows_in_reporter_year",
) -> pd.DataFrame:
    """
    Build table of HS classification transition events per reporter.

    The function first identifies the dominant classification for each
    reporter-year, then detects year-to-year changes.

    Parameters
    ----------
    df:
        Reporter-year-classification coverage table.
    dominance_metric:
        Column used to select dominant classification.

    Returns
    -------
    pd.DataFrame
        Transition-event table.
    """

    required = {
        reporter_col,
        reporter_iso_col,
        reporter_name_col,
        year_col,
        classification_col,
        dominance_metric,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dominant = (
        df.sort_values(
            [reporter_col, year_col, dominance_metric],
            ascending=[True, True, False],
        )
        .drop_duplicates([reporter_col, year_col])
        .sort_values([reporter_col, year_col])
        .copy()
    )

    dominant["previous_year"] = dominant.groupby(reporter_col)[year_col].shift()
    dominant["previous_classification_code"] = dominant.groupby(reporter_col)[
        classification_col
    ].shift()

    dominant["year_gap"] = dominant[year_col] - dominant["previous_year"]

    transitions = dominant[
        dominant["previous_classification_code"].notna()
        & dominant[classification_col].ne(dominant["previous_classification_code"])
    ].copy()

    transitions = transitions.rename(
        columns={
            year_col: "transition_year",
            classification_col: "to_classification_code",
            "previous_classification_code": "from_classification_code",
        }
    )

    transitions = transitions[
        [
            reporter_col,
            reporter_iso_col,
            reporter_name_col,
            "previous_year",
            "transition_year",
            "year_gap",
            "from_classification_code",
            "to_classification_code",
        ]
    ]

    transitions = transitions.sort_values(
        [reporter_col, "transition_year"]
    ).reset_index(drop=True)

    return transitions

def build_mixed_classification_reporter_years(
    df: pd.DataFrame,
    *,
    reporter_col: str = "ReporterCode",
    reporter_iso_col: str = "ReporterISO",
    reporter_name_col: str = "ReporterDesc",
    year_col: str = "RefYear",
    classification_col: str = "ClassificationCode",
    rows_col: str = "n_rows",
    value_col: str = "primary_value_total",
    weight_col: str = "netwgt_total",
    row_share_col: str = "share_rows_in_reporter_year",
    dominance_metric: str = "share_rows_in_reporter_year",
) -> pd.DataFrame:
    """
    Build table of reporter-years that contain multiple HS classifications.

    This is useful for inspecting mixed-classification cases before HS-sync.

    Parameters
    ----------
    df:
        Reporter-year-classification coverage table.
    dominance_metric:
        Column used to identify the top classification in mixed years.

    Returns
    -------
    pd.DataFrame
        Mixed reporter-year diagnostic table.
    """

    enriched = add_reporter_year_shares(
        df,
        reporter_col=reporter_col,
        year_col=year_col,
        value_col=value_col,
        weight_col=weight_col,
        row_share_col=row_share_col,
    )

    required = {
        reporter_col,
        reporter_iso_col,
        reporter_name_col,
        year_col,
        classification_col,
        rows_col,
        row_share_col,
        dominance_metric,
    }
    missing = required - set(enriched.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    group_cols = [reporter_col, year_col]

    counts = (
        enriched.groupby(group_cols, dropna=False)
        .agg(
            ReporterISO=(reporter_iso_col, "first"),
            ReporterDesc=(reporter_name_col, "first"),
            n_classifications=(classification_col, "nunique"),
            n_rows_total=(rows_col, "sum"),
            primary_value_total=(value_col, "sum")
            if value_col in enriched.columns
            else (rows_col, "size"),
            netwgt_total=(weight_col, "sum")
            if weight_col in enriched.columns
            else (rows_col, "size"),
        )
        .reset_index()
    )

    mixed_keys = counts[counts["n_classifications"] > 1][group_cols]

    if mixed_keys.empty:
        return pd.DataFrame(
            columns=[
                reporter_col,
                "ReporterISO",
                "ReporterDesc",
                year_col,
                "n_classifications",
                "top_classification_code",
                "top_row_share",
                "top_value_share",
                "top_netwgt_share",
                "n_rows_total",
                "primary_value_total",
                "netwgt_total",
            ]
        )

    mixed_detail = enriched.merge(mixed_keys, on=group_cols, how="inner")

    top = (
        mixed_detail.sort_values(
            [reporter_col, year_col, dominance_metric],
            ascending=[True, True, False],
        )
        .drop_duplicates(group_cols)
        .copy()
    )

    keep_cols = [
        reporter_col,
        year_col,
        classification_col,
        row_share_col,
    ]

    optional_cols = [
        "share_value_in_reporter_year",
        "share_netwgt_in_reporter_year",
    ]

    keep_cols += [c for c in optional_cols if c in top.columns]
    keep_cols = list(dict.fromkeys(keep_cols))

    top = top[keep_cols].rename(
        columns={
            classification_col: "top_classification_code",
            row_share_col: "top_row_share",
            "share_value_in_reporter_year": "top_value_share",
            "share_netwgt_in_reporter_year": "top_netwgt_share",
        }
    )

    out = counts.merge(top, on=group_cols, how="inner")

    out = out[
        [
            reporter_col,
            "ReporterISO",
            "ReporterDesc",
            year_col,
            "n_classifications",
            "top_classification_code",
            "top_row_share",
            *([c for c in ["top_value_share", "top_netwgt_share"] if c in out.columns]),
            "n_rows_total",
            "primary_value_total",
            "netwgt_total",
        ]
    ]

    out = out.sort_values(
        [reporter_col, year_col]
    ).reset_index(drop=True)

    return out