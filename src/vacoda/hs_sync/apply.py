"""
vacoda.hs_sync.apply
Functions to apply HS synchronization to trade data, harmonizing product codes to a common HS vintage.
"""

from __future__ import annotations

import pandas as pd

def apply_hs_sync():
    pass

def expand_rows_with_weights():
    pass


def apply_weighted_hs_concordance(
    trade_df: pd.DataFrame,
    concordance_df: pd.DataFrame,
    target_hs: str = "H5",
    min_year: int = 2002,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    cmd_col: str = "CmdCode",
    value_col: str = "PrimaryValue",
    weight_col: str = "weight",
    keep_unmapped: bool = True) -> pd.DataFrame:
    """
    Harmonize Comtrade product codes to a target HS system using a weighted
    concordance table.

    This function does not require the user to provide source_cmd or target_cmd
    manually. Those are read from concordance_df.

    Expected trade_df columns:
        RefYear
        ClassificationCode
        CmdCode
        PrimaryValue
        plus any other dimensions, e.g. ReporterCode, PartnerCode, FlowCode

    Expected concordance_df columns:
        source_hs
        source_cmd
        target_hs
        target_cmd
        weight
    """

    required_trade_cols = {year_col, hs_col, cmd_col, value_col}
    missing_trade = required_trade_cols - set(trade_df.columns)
    if missing_trade:
        raise ValueError(f"trade_df missing columns: {sorted(missing_trade)}")

    required_conc_cols = {
        "source_hs",
        "source_cmd",
        "target_hs",
        "target_cmd",
        weight_col,
    }
    missing_conc = required_conc_cols - set(concordance_df.columns)
    if missing_conc:
        raise ValueError(f"concordance_df missing columns: {sorted(missing_conc)}")

    df = trade_df.loc[trade_df[year_col] >= min_year].copy()

    # Keep original information for diagnostics
    df["original_hs"] = df[hs_col]
    df["original_cmd"] = df[cmd_col]

    # Rows already in the target HS system do not need conversion
    already_target = df[df[hs_col] == target_hs].copy()
    already_target["harmonized_hs"] = target_hs
    already_target["harmonized_cmd"] = already_target[cmd_col]
    already_target[weight_col] = 1.0
    already_target[f"{value_col}_harmonized"] = already_target[value_col]
    already_target["mapping_status"] = "already_target"

    # Rows from other HS systems need mapping
    needs_mapping = df[df[hs_col] != target_hs].copy()

    conc = concordance_df.loc[
        concordance_df["target_hs"] == target_hs
    ].copy()

    mapped = needs_mapping.merge(
        conc,
        left_on=[hs_col, cmd_col],
        right_on=["source_hs", "source_cmd"],
        how="left",
        validate="many_to_many",
    )

    has_mapping = mapped["target_cmd"].notna()

    mapped.loc[has_mapping, "harmonized_hs"] = mapped.loc[has_mapping, "target_hs"]
    mapped.loc[has_mapping, "harmonized_cmd"] = mapped.loc[has_mapping, "target_cmd"]
    mapped.loc[has_mapping, f"{value_col}_harmonized"] = (
        mapped.loc[has_mapping, value_col] * mapped.loc[has_mapping, weight_col]
    )
    mapped.loc[has_mapping, "mapping_status"] = "mapped"

    if keep_unmapped:
        mapped.loc[~has_mapping, "harmonized_hs"] = pd.NA
        mapped.loc[~has_mapping, "harmonized_cmd"] = pd.NA
        mapped.loc[~has_mapping, f"{value_col}_harmonized"] = pd.NA
        mapped.loc[~has_mapping, "mapping_status"] = "unmapped"
    else:
        mapped = mapped.loc[has_mapping].copy()

    out = pd.concat(
        [already_target, mapped],
        ignore_index=True,
        sort=False,
    )

    return out

def harmonize_hs_to_target(
    db: pd.DataFrame,
    mapping: pd.DataFrame,
    target_hs: str = "H5",
    min_year: int = 2002,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    cmd_col: str = "CmdCode",
    value_col: str = "PrimaryValue",
    weight_col: str = "weight",
    aggregate: bool = True) -> pd.DataFrame:
    """
    Harmonize original HS vintages in a Comtrade-style table to one target HS vintage.

    This follows a Harvard-style weighted concordance idea:
    one source HS code may map to multiple target HS codes, and trade value is split
    by conversion weights.

    Parameters
    ----------
    db:
        Trade data with at least year, HS system, product code, and value columns.

    mapping:
        Concordance table with columns:
            source_hs
            source_cmd
            target_hs
            target_cmd
            weight

        Example:
            source_hs source_cmd target_hs target_cmd weight
            H3        441012     H5        441011     0.70
            H3        441012     H5        441012     0.30

    target_hs:
        Target HS vintage, e.g. "H5".

    min_year:
        Keep data from this year onward.

    aggregate:
        If True, aggregate duplicated rows after mapping.

    Returns
    -------
    pd.DataFrame
        Harmonized dataframe with:
            original HS columns preserved
            harmonized HS columns added
            weighted value column added
    """

    required_db_cols = {year_col, hs_col, cmd_col, value_col}
    missing_db_cols = required_db_cols - set(db.columns)
    if missing_db_cols:
        raise ValueError(f"db is missing required columns: {sorted(missing_db_cols)}")

    required_mapping_cols = {
        "source_hs",
        "source_cmd",
        "target_hs",
        "target_cmd",
        weight_col,
    }
    missing_mapping_cols = required_mapping_cols - set(mapping.columns)
    if missing_mapping_cols:
        raise ValueError(
            f"mapping is missing required columns: {sorted(missing_mapping_cols)}"
        )

    # 1. Filter years
    df = db.loc[db[year_col] >= min_year].copy()

    # 2. Preserve original classification information
    df["original_hs"] = df[hs_col]
    df["original_cmd"] = df[cmd_col]

    # 3. Rows already in target HS do not need concordance
    already_target = df[df[hs_col] == target_hs].copy()
    already_target["target_hs"] = target_hs
    already_target["target_cmd"] = already_target[cmd_col]
    already_target[weight_col] = 1.0
    already_target[f"{value_col}_mapped"] = already_target[value_col]

    # 4. Rows from other HS vintages need weighted mapping
    needs_mapping = df[df[hs_col] != target_hs].copy()

    mapped = needs_mapping.merge(
        mapping,
        left_on=[hs_col, cmd_col],
        right_on=["source_hs", "source_cmd"],
        how="left",
        validate="many_to_many",
    )

    # 5. Keep only mappings into the desired target HS
    mapped = mapped[mapped["target_hs"] == target_hs].copy()

    # 6. Apply conversion weights
    mapped[f"{value_col}_mapped"] = mapped[value_col] * mapped[weight_col]

    # 7. Combine target rows and mapped rows
    out = pd.concat([already_target, mapped], ignore_index=True, sort=False)

    # 8. Replace working HS/code columns by harmonized target version
    out[hs_col] = out["target_hs"]
    out[cmd_col] = out["target_cmd"]

    if not aggregate:
        return out

    # 9. Aggregate after many-to-many mapping
    protected_cols = {
        value_col,
        f"{value_col}_mapped",
        weight_col,
        "source_hs",
        "source_cmd",
        "target_hs",
        "target_cmd",
        "original_hs",
        "original_cmd",
    }

    group_cols = [
        c
        for c in out.columns
        if c not in protected_cols
        and not c.endswith("_mapped")
    ]

    agg = (
        out.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            **{
                f"{value_col}_mapped": (f"{value_col}_mapped", "sum"),
                "n_source_rows": (value_col, "size"),
                "n_original_hs_systems": ("original_hs", "nunique"),
            }
        )
    )

    return agg