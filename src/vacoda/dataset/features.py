"""
Module: vacoda.preprocessing.derived_features
Description: Core feature engineering (estimation flags, HS validity, price features).
             Geographic and partner-source logic live in geography.py and partner_source.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_estimation_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds boolean flags to indicate whether the value, quantity, or weight in a trade record is estimated based on the 'estimation_code' column.
    """
    out = df.copy()
    out["is_estimated"]       = out["estimation_code"] != 0
    out["quantity_estimated"] = out["estimation_code"].isin([2, 6])
    out["weight_estimated"]   = out["estimation_code"].isin([4, 6])
    return out


def add_hs_validity_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a boolean flag 'hs_mismatch' to indicate whether the HS code is valid for the given year based on 
    the 'hs_valid_from' and 'hs_valid_until' columns.
    """
    out = df.copy()

    out["hs_valid_from"]  = pd.to_datetime(out["hs_valid_from"],  errors="coerce")
    out["hs_valid_until"] = pd.to_datetime(out["hs_valid_until"], errors="coerce")
    out["year_date"]      = pd.to_datetime(out["year"].astype("Int64").astype(str), format="%Y", errors="coerce")

    out["hs_mismatch"] = ~( (out["year_date"] >= out["hs_valid_from"]) & (out["year_date"] <= out["hs_valid_until"]))
    return out


def add_price_features(df: pd.DataFrame,
                       weight_col: str,
                       value_col: str) -> pd.DataFrame:
    out = df.copy()
    valid = (
        out[weight_col].notna()
        & out[weight_col].gt(0)
        & out[value_col].notna()
        & out[value_col].gt(0)
    )

    out["price_per_kg"] = np.nan
    out.loc[valid, "price_per_kg"] = (
        out.loc[valid, value_col] / out.loc[valid, weight_col]
    )

    out["calc_kg_per_usd"] = np.nan
    out.loc[valid, "calc_kg_per_usd"] = (
        out.loc[valid, weight_col] / out.loc[valid, value_col]
    )

    return out


