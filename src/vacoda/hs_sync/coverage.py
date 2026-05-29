"""
vacoda.hs_sync.coverage
Functions for calculating and summarizing the coverage of the HS concordance.
"""
from __future__ import annotations

import pandas as pd

def build_hs_concordance_coverage_from_source_summary(
    source_summary: pd.DataFrame,
    concordance_df: pd.DataFrame,
    target_hs: str = "H5",
    weight_col: str = "weight",
    weight_sum_tolerance: float = 0.001,
) -> pd.DataFrame:
    return build_hs_concordance_coverage(
        source_summary=source_summary,
        concordance_df=concordance_df,
        target_hs=target_hs,
        weight_col=weight_col,
        weight_sum_tolerance=weight_sum_tolerance,
    )


def get_one_to_many_hs_mappings(
    coverage: pd.DataFrame,
    *,
    only_value_splits: bool = False,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Return source HS/code pairs that map to more than one target code.

    Parameters
    ----------
    coverage:
        Output of build_hs_concordance_coverage().

    only_value_splits:
        If True, keep only mappings where the source value would be split
        across multiple target codes.

    top_n:
        Optional number of rows to return, sorted by trade value.

    Returns
    -------
    pd.DataFrame
        Filtered one-to-many mapping diagnostics.
    """

    required_cols = {
        "mapping_type",
        "produces_value_split",
        "trade_value_total",
        "n_target_codes",
    }

    missing = required_cols - set(coverage.columns)
    if missing:
        raise ValueError(
            f"coverage is missing required columns: {sorted(missing)}"
        )

    out = coverage.loc[
        coverage["mapping_type"].eq("one_to_many")
    ].copy()

    if only_value_splits:
        out = out.loc[out["produces_value_split"]].copy()

    out = out.sort_values(
        ["trade_value_total", "n_target_codes"],
        ascending=[False, False],
    ).reset_index(drop=True)

    if top_n is not None:
        out = out.head(top_n)

    return out

def build_hs_concordance_coverage_from_source_summary_deprecated(
    source_summary: pd.DataFrame,
    concordance_df: pd.DataFrame,
    target_hs: str = "H5",
    weight_col: str = "weight",
    weight_sum_tolerance: float = 0.001,
) -> pd.DataFrame:
    """
    Build HS concordance coverage diagnostics from a compact source-pair summary.

    source_summary must contain:
        source_hs
        source_cmd
        n_trade_rows
        first_year
        last_year
        trade_value_total
    """

    required_source_cols = {
        "source_hs",
        "source_cmd",
        "n_trade_rows",
        "first_year",
        "last_year",
        "trade_value_total",
    }

    missing_source = required_source_cols - set(source_summary.columns)
    if missing_source:
        raise ValueError(
            f"source_summary missing required columns: {sorted(missing_source)}"
        )

    required_concordance_cols = {
        "source_hs",
        "source_cmd",
        "target_hs",
        "target_cmd",
        weight_col,
    }

    missing_concordance = required_concordance_cols - set(concordance_df.columns)
    if missing_concordance:
        raise ValueError(
            "concordance_df missing required columns: "
            f"{sorted(missing_concordance)}"
        )

    src = source_summary.copy()
    src["source_hs"] = src["source_hs"].astype("string")
    src["source_cmd"] = src["source_cmd"].astype("string")

    conc = concordance_df.copy()
    conc["source_hs"] = conc["source_hs"].astype("string")
    conc["source_cmd"] = conc["source_cmd"].astype("string")
    conc["target_hs"] = conc["target_hs"].astype("string")
    conc["target_cmd"] = conc["target_cmd"].astype("string")
    conc[weight_col] = pd.to_numeric(conc[weight_col], errors="coerce")

    conc_target = conc.loc[conc["target_hs"] == target_hs].copy()

    mapping_summary = (
        conc_target.groupby(["source_hs", "source_cmd"], dropna=False, as_index=False)
        .agg(
            n_mapping_rows=("target_cmd", "size"),
            n_target_codes=("target_cmd", "nunique"),
            target_codes=("target_cmd", lambda x: sorted(set(x.dropna()))),
            weight_sum=(weight_col, "sum"),
            min_weight=(weight_col, "min"),
            max_weight=(weight_col, "max"),
        )
    )

    out = src.merge(
        mapping_summary,
        on=["source_hs", "source_cmd"],
        how="left",
    )

    out["has_mapping"] = out["n_mapping_rows"].notna()

    out["n_mapping_rows"] = out["n_mapping_rows"].fillna(0).astype("int64")
    out["n_target_codes"] = out["n_target_codes"].fillna(0).astype("int64")

    out["is_already_target_hs"] = out["source_hs"].eq(target_hs)

    out["mapping_type"] = "missing"

    out.loc[
        out["is_already_target_hs"],
        "mapping_type",
    ] = "already_target"

    out.loc[
        out["has_mapping"] & (out["n_target_codes"] == 1),
        "mapping_type",
    ] = "one_to_one"

    out.loc[
        out["has_mapping"] & (out["n_target_codes"] > 1),
        "mapping_type",
    ] = "one_to_many"

    out["produces_value_split"] = (
        out["has_mapping"]
        & (out["n_target_codes"] > 1)
        & out["min_weight"].notna()
        & out["max_weight"].notna()
        & (out["min_weight"] < 1.0)
    )

    lower = 1.0 - weight_sum_tolerance
    upper = 1.0 + weight_sum_tolerance

    out["weight_sum_status"] = "not_applicable"

    out.loc[
        out["has_mapping"],
        "weight_sum_status",
    ] = "check"

    out.loc[
        out["has_mapping"] & out["weight_sum"].between(lower, upper),
        "weight_sum_status",
    ] = "ok_approx_1"

    out.loc[
        out["has_mapping"] & ~out["weight_sum"].between(lower, upper),
        "weight_sum_status",
    ] = "not_1"

    out = out.sort_values(
        ["has_mapping", "source_hs", "trade_value_total", "source_cmd"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    return out

def get_missing_hs_mappings(
    coverage: pd.DataFrame,
    top_n: int | None = 50,
) -> pd.DataFrame:
    """
    Return source HS/code pairs that appear in the trade data
    but have no concordance into the target HS system.
    """

    missing = coverage.loc[~coverage["has_mapping"]].copy()

    missing = missing.sort_values(
        ["trade_value_total", "n_trade_rows"],
        ascending=[False, False],
    )

    if top_n is not None:
        missing = missing.head(top_n)

    return missing

def build_hs_concordance_coverage(
    source_summary: pd.DataFrame,
    concordance_df: pd.DataFrame,
    target_hs: str = "H5",
    weight_col: str = "weight",
    weight_sum_tolerance: float = 0.001) -> pd.DataFrame:
    """
    Build HS concordance coverage diagnostics from a compact source-pair summary.

    Expected source_summary columns:
        source_hs
        source_cmd
        n_trade_rows
        first_year
        last_year
        trade_value_total

    Expected concordance_df columns:
        source_hs
        source_cmd
        target_hs
        target_cmd
        weight
    """

    required_source_cols = {
        "source_hs",
        "source_cmd",
        "n_trade_rows",
        "first_year",
        "last_year",
        "trade_value_total",
    }

    missing_source = required_source_cols - set(source_summary.columns)
    if missing_source:
        raise ValueError(
            f"source_summary missing required columns: {sorted(missing_source)}"
        )

    required_concordance_cols = {
        "source_hs",
        "source_cmd",
        "target_hs",
        "target_cmd",
        weight_col,
    }

    missing_concordance = required_concordance_cols - set(concordance_df.columns)
    if missing_concordance:
        raise ValueError(
            "concordance_df missing required columns: "
            f"{sorted(missing_concordance)}"
        )

    src = source_summary.copy()
    src["source_hs"] = src["source_hs"].astype("string")
    src["source_cmd"] = src["source_cmd"].astype("string")

    conc = concordance_df.copy()
    conc["source_hs"] = conc["source_hs"].astype("string")
    conc["source_cmd"] = conc["source_cmd"].astype("string")
    conc["target_hs"] = conc["target_hs"].astype("string")
    conc["target_cmd"] = conc["target_cmd"].astype("string")
    conc[weight_col] = pd.to_numeric(conc[weight_col], errors="coerce")

    conc_target = conc.loc[conc["target_hs"] == target_hs].copy()

    mapping_summary = (
        conc_target.groupby(["source_hs", "source_cmd"], dropna=False, as_index=False)
        .agg(
            n_mapping_rows=("target_cmd", "size"),
            n_target_codes=("target_cmd", "nunique"),
            target_codes=("target_cmd", lambda x: sorted(set(x.dropna()))),
            weight_sum=(weight_col, "sum"),
            min_weight=(weight_col, "min"),
            max_weight=(weight_col, "max"),
        )
    )

    out = src.merge(
        mapping_summary,
        on=["source_hs", "source_cmd"],
        how="left",
    )

    out["has_mapping"] = out["n_mapping_rows"].notna()

    out["n_mapping_rows"] = out["n_mapping_rows"].fillna(0).astype("int64")
    out["n_target_codes"] = out["n_target_codes"].fillna(0).astype("int64")

    out["is_already_target_hs"] = out["source_hs"].eq(target_hs)

    out["mapping_type"] = "missing"

    out.loc[
        out["is_already_target_hs"],
        "mapping_type",
    ] = "already_target"

    out.loc[
        out["has_mapping"] & (out["n_target_codes"] == 1),
        "mapping_type",
    ] = "one_to_one"

    out.loc[
        out["has_mapping"] & (out["n_target_codes"] > 1),
        "mapping_type",
    ] = "one_to_many"

    out["produces_value_split"] = (
        out["has_mapping"]
        & (out["n_target_codes"] > 1)
        & out["min_weight"].notna()
        & out["max_weight"].notna()
        & (out["min_weight"] < 1.0)
    )

    lower = 1.0 - weight_sum_tolerance
    upper = 1.0 + weight_sum_tolerance

    out["weight_sum_status"] = "not_applicable"

    out.loc[
        out["has_mapping"],
        "weight_sum_status",
    ] = "check"

    out.loc[
        out["has_mapping"] & out["weight_sum"].between(lower, upper),
        "weight_sum_status",
    ] = "ok_approx_1"

    out.loc[
        out["has_mapping"] & ~out["weight_sum"].between(lower, upper),
        "weight_sum_status",
    ] = "not_1"

    out = out.sort_values(
        ["has_mapping", "source_hs", "trade_value_total", "source_cmd"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    return out