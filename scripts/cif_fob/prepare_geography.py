"""
script.prepare_geography.py
This script prepares the country-level geography reference from CEPII's geo dataset.
It selects relevant columns, standardizes country codes, converts numeric fields, cleans continent labels, and derives log-transformed variables. 
The prepared reference is saved as a Parquet file for use in the CIF-FOB model.
"""
from pathlib import Path

import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[2]

input_path = (
    project_root
    / "references"
    / "geography"
    / "source"
    / "geo_cepii.xls"
)

output_path = (
    project_root
    / "references"
    / "geography"
    / "prepared"
    / "country_geography.parquet"
)


# ============================================================
# Load source
# ============================================================

geo_raw = pd.read_excel(
    input_path,
    sheet_name="geo_cepii",
)


# ============================================================
# Validate source schema
# ============================================================

required_cols = {
    "iso3",
    "landlocked",
    "area",
    "dis_int",
    "continent",
}

missing_cols = required_cols.difference(geo_raw.columns)

if missing_cols:
    raise ValueError(
        f"Missing required CEPII columns: {sorted(missing_cols)}"
    )


# ============================================================
# Select and rename columns
# ============================================================

country_geo = (
    geo_raw[
        [
            "iso3",
            "area",
            "dis_int",
            "landlocked",
            "continent",
        ]
    ]
    .rename(
        columns={
            "iso3": "country_iso3",
            "area": "area_km2",
            "dis_int": "internal_distance_km",
            "landlocked": "landlocked",
            "continent": "continent",
        }
    )
    .copy()
)


# ============================================================
# Standardize country codes
# ============================================================

country_geo["country_iso3"] = (
    country_geo["country_iso3"]
    .astype("string")
    .str.strip()
    .str.upper()
)


# ============================================================
# Convert numeric fields
# ============================================================

country_geo["area_km2"] = pd.to_numeric(
    country_geo["area_km2"],
    errors="coerce",
)

country_geo["internal_distance_km"] = pd.to_numeric(
    country_geo["internal_distance_km"],
    errors="coerce",
)

country_geo["landlocked"] = pd.to_numeric(
    country_geo["landlocked"],
    errors="coerce",
).astype("Int8")


# ============================================================
# Clean continent labels
# ============================================================

country_geo["continent"] = (
    country_geo["continent"]
    .astype("string")
    .str.strip()
    .replace(
        {
            "No": "Unclassified",
            "": pd.NA,
            ".": pd.NA,
        }
    )
)


# ============================================================
# Derive transformed variables
# ============================================================

country_geo["log_area_km2"] = np.where(
    country_geo["area_km2"].gt(0),
    np.log(country_geo["area_km2"]),
    np.nan,
)

country_geo["log_internal_distance_km"] = np.where(
    country_geo["internal_distance_km"].gt(0),
    np.log(country_geo["internal_distance_km"]),
    np.nan,
)


# ============================================================
# Validate duplicate country rows
#
# geo_cepii may contain multiple city rows for the same country.
# The retained country-level fields must be identical before
# collapsing to one row per ISO3 code.
# ============================================================

duplicate_iso3 = country_geo.loc[
    country_geo.duplicated(
        "country_iso3",
        keep=False,
    ),
    "country_iso3",
].dropna().unique()

if len(duplicate_iso3) > 0:
    duplicate_check = (
        country_geo.loc[
            country_geo["country_iso3"].isin(duplicate_iso3)
        ]
        .groupby("country_iso3")
        .agg(
            rows=("country_iso3", "size"),
            area_values=("area_km2", "nunique"),
            internal_distance_values=(
                "internal_distance_km",
                "nunique",
            ),
            landlocked_values=("landlocked", "nunique"),
            continent_values=("continent", "nunique"),
        )
    )

    inconsistent_duplicates = duplicate_check.loc[
        (duplicate_check["area_values"] > 1)
        | (duplicate_check["internal_distance_values"] > 1)
        | (duplicate_check["landlocked_values"] > 1)
        | (duplicate_check["continent_values"] > 1)
    ]

    if not inconsistent_duplicates.empty:
        raise ValueError(
            "Inconsistent country-level values found for duplicate "
            f"ISO3 codes:\n{inconsistent_duplicates}"
        )


# ============================================================
# Collapse to one row per country
# ============================================================

country_geo = (
    country_geo
    .drop_duplicates(subset=["country_iso3"])
    .sort_values("country_iso3")
    .reset_index(drop=True)
)


# ============================================================
# Final validation
# ============================================================

duplicate_count = country_geo.duplicated(
    "country_iso3"
).sum()

if duplicate_count:
    raise ValueError(
        f"Found {duplicate_count} duplicate ISO3 codes "
        "after deduplication."
    )

missing_iso3 = country_geo["country_iso3"].isna().sum()

if missing_iso3:
    raise ValueError(
        f"Found {missing_iso3} rows with missing country_iso3."
    )

invalid_landlocked = (
    country_geo["landlocked"]
    .dropna()
    .loc[
        ~country_geo["landlocked"]
        .dropna()
        .isin([0, 1])
    ]
)

if not invalid_landlocked.empty:
    raise ValueError(
        "Unexpected values found in landlocked: "
        f"{sorted(invalid_landlocked.unique().tolist())}"
    )


# ============================================================
# Diagnostics
# ============================================================

print("\nShape:")
print(country_geo.shape)

print("\nData types:")
print(country_geo.dtypes)

print("\nMissing values:")
print(country_geo.isna().sum())

print(
    "\nDuplicate ISO3 codes:",
    duplicate_count,
)

print("\nLandlocked:")
print(
    country_geo["landlocked"]
    .value_counts(dropna=False)
)

print("\nContinents:")
print(
    country_geo["continent"]
    .value_counts(dropna=False)
)

print("\nArea summary:")
print(
    country_geo["area_km2"].describe(
        percentiles=[
            0.01,
            0.50,
            0.95,
            0.99,
        ]
    )
)

print("\nInternal-distance summary:")
print(
    country_geo["internal_distance_km"].describe(
        percentiles=[
            0.01,
            0.50,
            0.95,
            0.99,
        ]
    )
)

print("\nPreview:")
print(country_geo.head())


# ============================================================
# Save prepared reference
# ============================================================

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

country_geo.to_parquet(
    output_path,
    index=False,
)

print(f"\nSaved to: {output_path}")