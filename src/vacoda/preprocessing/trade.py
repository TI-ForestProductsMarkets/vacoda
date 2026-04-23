"""
Module: vacoda.preprocessing.trade
Description: This module contains functions for preprocessing the main trade data table from the UN-Comtrade dataset.
"""

from __future__ import annotations

import pandas as pd

from vacoda.mappings.trade_map import TRADE_MAP_DB, TRADE_MAP_API

# List of all required internal field names for the pipeline
REQUIRED_FIELDS = [
    "reporter_code", "partner_code", "commodity_code", "year", "flow_code",
    "value_usd_raw", "netweight_kg_raw", "estimation_code", 
]

# Mapping registry for easy lookup by source type
TRADE_MAP_REGISTRY = {
    "db": TRADE_MAP_DB,
    "api": TRADE_MAP_API,
}

def get_trade_map(source_type: str):
    """
    Return the mapping dictionary for the given source type.
    Example: get_trade_map('db') or get_trade_map('api')
    """
    try:
        return TRADE_MAP_REGISTRY[source_type]
    except KeyError:
        raise ValueError(f"Unknown trade data source type: {source_type}")


def standardize_trade_columns(df, source_type: str):
    """
    Rename columns in df to standardized internal names using the mapping for the given source_type.
    """
    trade_map = get_trade_map(source_type)
    # Only rename columns that exist in the DataFrame
    rename_dict = {k: v for k, v in trade_map.items() if k in df.columns}
    return df.rename(columns=rename_dict) 
 
def prepare_trade_table(df: pd.DataFrame, unique_threshold=10) -> pd.DataFrame:
    """
    Prepares the main trade table by renaming columns, converting data types, and ensuring consistency in key columns.
    Parameters:
    -----------
    - df: pd.DataFrame - The input DataFrame containing the raw trade data.
    
    Returns:
    --------
    - pd.DataFrame: A cleaned and standardized DataFrame ready for analysis.
    """
    out = df.copy()
    out = standardize_trade_columns(out, source_type="db")

    # Coerce all integer join keys (robust, vectorized, no for loop)
    int_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Vectorized diagnostics for non-numeric values before casting
    non_numeric_mask = out[int_cols].apply(lambda col: ~pd.to_numeric(col, errors="coerce").notna())
    problematic = out[int_cols][non_numeric_mask]
    for col in problematic:
        vals = out.loc[non_numeric_mask[col], col].unique()
        if len(vals) > 0:
            print(f"Non-numeric values in column '{col}':", vals)

    # Vectorized conversion: apply to_numeric, then astype Int64 only for columns that can be safely cast
    out[int_cols] = out[int_cols].apply(pd.to_numeric, errors="coerce")
    
    # Only cast columns that are not all-NaN or already Int64
    needs_cast = [col for col in int_cols if out[col].notna().any() or out[col].dtype != "Int64"]
    if needs_cast:
        out[needs_cast] = out[needs_cast].astype("Int64", errors="ignore")

    float_cols = df.select_dtypes(include=["float"]).columns.tolist()
    out[float_cols] = out[float_cols].apply(pd.to_numeric, errors="coerce")

    # Vectorized assignment of 'category' or 'string' dtype for object columns
    obj_cols = out.select_dtypes(include=["object", "string"]).columns
    n_uniques = out[obj_cols].nunique(dropna=False)
    low_card_cols = n_uniques[(n_uniques / len(out) < 0.1) | (n_uniques < unique_threshold)].index
    high_card_cols = n_uniques.index.difference(low_card_cols)

    # Strip whitespace for all object columns first (vectorized)
    out[obj_cols] = out[obj_cols].apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Assign dtypes in batch
    out[low_card_cols] = out[low_card_cols].astype("category")
    out[high_card_cols] = out[high_card_cols].astype("string")
    
    return out
