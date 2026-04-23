"""
Functions to prepare metadata tables from raw sheets. 
These are used to populate the `metadata` module of the processed dataset, 
which is then used for lookups and reference in the main trade table and for diagnostics. 
Each function takes the raw sheet as input and outputs a cleaned and standardized DataFrame 
with relevant columns renamed and duplicates dropped.
"""
from __future__ import annotations

import pandas as pd
from vacoda.utils.logger import get_logger

logger = get_logger(__name__)


# Utility to standardize metadata dtypes
def standardize_metadata_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all metadata columns have the correct dtype for downstream processing.
    Adjust this mapping as your metadata schema evolves.
    """
    dtype_map = {
        "country_code": "Int64",
        "country_iso2": "string",
        "country_iso3": "string",
        "flow_code": "string",
        "unit_code": "Int64",
        "estimation_code": "Int64",
        "commodity_code": "Int64",
        # Add more columns as needed
    }
    for col, dtype in dtype_map.items():
        if col in df.columns:
            # Use pandas nullable types for Int64 and string
            if dtype == "Int64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "string":
                df[col] = df[col].astype("string")
    return df




def prepare_reporter_metadata(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Country_ISONUM",
        "Country_Abbreviation",
        "Country_ISO3166A2",
        "Country_ISO3166A3",
        "Country_StartValidYear",
        "Country_EndValidYear",
    ]
    out = df[cols].copy()
    out = out.rename(
        columns={
            "Country_ISONUM"        : "country_code",
            "Country_Abbreviation"  : "country_name",
            "Country_ISO3166A2"     : "country_iso2",
            "Country_ISO3166A3"     : "country_iso3",
            "Country_StartValidYear": "country_valid_from",
            "Country_EndValidYear"  : "country_valid_until",
        }
    )
    # Ensure code columns are string for robust merging
    out["country_code"] = out["country_code"].astype("string").str.strip()
    out["country_iso2"] = out["country_iso2"].astype("string").str.strip()
    out["country_iso3"] = out["country_iso3"].astype("string").str.strip()
    return out.drop_duplicates()


def prepare_hs_version_metadata(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["HS_HSCode", "HS_valid_from", "HS_valid_until", "HS_HSCode2"]
    out = df[cols].copy()
    out = out.rename(
        columns={
            "HS_HSCode"     : "hs_code_short",
            "HS_valid_from" : "hs_valid_from",
            "HS_valid_until": "hs_valid_until",
            "HS_HSCode2"    : "hs_version_name",
        }
    )
    return out.drop_duplicates()


def prepare_commodity_metadata(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["HS_HSCode", "HS_CommodityCode", "HS_ProductDescription"]
    out = df[cols].copy()
    out = out.rename(
        columns={
            "HS_HSCode": "hs_code_short",
            "HS_CommodityCode": "commodity_code",
            "HS_ProductDescription": "commodity_description",
        }
    )
    return out.drop_duplicates()


def prepare_flow_metadata(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Flow_IDFlow", "Flow_Description", "Flow_Category"]
    out = df[cols].copy()
    out = out.rename(
        columns={
            "Flow_IDFlow": "flow_code",
            "Flow_Description": "flow_description",
            "Flow_Category": "flow_category",
        }
    )
    # Keep only numeric flow codes that can match trade data, then normalize to string keys.
    flow_num = pd.to_numeric(df["Flow_IDFlow"], errors="coerce")
    non_numeric = flow_num.isna()
    dropped_codes = df.loc[non_numeric, "Flow_IDFlow"].tolist()
    if dropped_codes:
        logger.debug(
            f"prepare_flow_metadata: dropped {len(dropped_codes)} non-numeric flow code(s) "
            f"that cannot match trade data: {dropped_codes}"
        )
    out = out.loc[~non_numeric].copy()
    out["flow_code"] = flow_num.loc[~non_numeric].astype("Int64").astype("string")
    out["flow_category"] = out["flow_category"].astype("category")
    out["flow_description"] = out["flow_description"].astype("category")
    return out.drop_duplicates()


def prepare_unit_metadata(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Unit_Token", "Unit_ShortName", "Unit_Description"]
    out = df[cols].copy()
    out = out.rename(
        columns={
            "Unit_Token"       : "unit_code",
            "Unit_ShortName"   : "unit_short_name",
            "Unit_Description" : "unit_description",
        }
    )
    out["unit_code"] = out["unit_code"].astype("string").str.strip()
    return out.drop_duplicates()


def prepare_estimation_metadata(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Est_Code", "Est_Description"]
    out = df[cols].copy()
    out = out.rename(
        columns={
            "Est_Code": "estimation_code",
            "Est_Description": "estimation_description",
        }
    )
    out["estimation_code"] = out["estimation_code"].astype("string").str.strip()
    return out.drop_duplicates()


