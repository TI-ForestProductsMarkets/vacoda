"""
script.prepare_pair_geography.py
This script prepares the country-pair geography reference from CEPII's distance dataset.
It selects relevant columns, standardizes country codes, converts distance measures to numeric,
and creates a model distance variable. The prepared reference is saved as a Parquet file for use in the CIF-FOB model.
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
    / "dist_cepii.xls"
)

output_path = (
    project_root
    / "references"
    / "geography"
    / "prepared"
    / "country_pair_geography.parquet"
)


# ============================================================
# Load source
# ============================================================

raw = pd.read_excel(
    input_path,
    sheet_name="dist_cepii",
)


# ============================================================
# Validate source schema
# ============================================================

required_cols = {
    "iso_o",
    "iso_d",
    "dist",
    "distcap",
    "distw",
    "distwces",
    "contig",
}

missing_cols = required_cols.difference(raw.columns)

if missing_cols:
    raise ValueError(
        f"Missing required CEPII columns: {sorted(missing_cols)}"
    )


# ============================================================
# Select and rename columns
# ============================================================

pair_geo = (
    raw[
        [
            "iso_o",
            "iso_d",
            "dist",
            "distcap",
            "distw",
            "distwces",
            "contig",
        ]
    ]
    .rename(
        columns={
            "iso_o": "origin_iso3",
            "iso_d": "destination_iso3",
            "dist": "distance_simple_km",
            "distcap": "distance_capital_km",
            "distw": "distance_weighted_km",
            "distwces": "distance_weighted_ces_km",
            "contig": "contiguous",
        }
    )
    .copy()
)


# ============================================================
# Standardize country codes
# ============================================================

for col in [
    "origin_iso3",
    "destination_iso3",
]:
    pair_geo[col] = (
        pair_geo[col]
        .astype("string")
        .str.strip()
        .str.upper()
    )


# ============================================================
# Convert distance measures
# ============================================================

distance_cols = [
    "distance_simple_km",
    "distance_capital_km",
    "distance_weighted_km",
    "distance_weighted_ces_km",
]

for col in distance_cols:
    pair_geo[col] = pd.to_numeric(
        pair_geo[col],
        errors="coerce",
    )


pair_geo["contiguous"] = pd.to_numeric(
    pair_geo["contiguous"],
    errors="raise",
).astype("Int8")


# ============================================================
# Create model distance
#
# Preferred:
#   1. Population-weighted distance
#   2. Capital-to-capital distance as fallback
# ============================================================

pair_geo["distance_model_km"] = (
    pair_geo["distance_weighted_km"]
    .combine_first(
        pair_geo["distance_capital_km"]
    )
)

pair_geo["distance_model_source"] = np.select(
    [
        pair_geo["distance_weighted_km"].notna(),
        (
            pair_geo["distance_weighted_km"].isna()
            & pair_geo["distance_capital_km"].notna()
        ),
    ],
    [
        "weighted",
        "capital",
    ],
    default="missing",
)

pair_geo["distance_model_source"] = (
    pair_geo["distance_model_source"]
    .astype("string")
)

pair_geo["log_distance_model_km"] = np.where(
    pair_geo["distance_model_km"].gt(0),
    np.log(pair_geo["distance_model_km"]),
    np.nan,
)


# ============================================================
# Sort and validate uniqueness
# ============================================================

pair_geo = (
    pair_geo
    .sort_values(
        [
            "origin_iso3",
            "destination_iso3",
        ]
    )
    .reset_index(drop=True)
)

duplicate_pairs = pair_geo.duplicated(
    [
        "origin_iso3",
        "destination_iso3",
    ]
).sum()

if duplicate_pairs:
    raise ValueError(
        f"Found {duplicate_pairs} duplicate country pairs."
    )


# ============================================================
# Validation checks
# ============================================================

print("\nShape:")
print(pair_geo.shape)

print("\nData types:")
print(pair_geo.dtypes)

print("\nMissing values:")
print(pair_geo.isna().sum())

print("\nModel-distance sources:")
print(
    pair_geo["distance_model_source"]
    .value_counts(dropna=False)
)

print(
    "\nMissing model distance:",
    pair_geo["distance_model_km"].isna().sum(),
)

print(
    "Missing log model distance:",
    pair_geo["log_distance_model_km"].isna().sum(),
)

print("\nModel-distance summary:")
print(
    pair_geo["distance_model_km"].describe(
        percentiles=[
            0.01,
            0.50,
            0.95,
            0.99,
        ]
    )
)

print("\nContiguity:")
print(
    pair_geo["contiguous"]
    .value_counts(dropna=False)
)

print(
    "\nDuplicate pairs:",
    duplicate_pairs,
)

print("\nPreview:")
print(pair_geo.head())


# ============================================================
# Save prepared reference
# ============================================================

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

pair_geo.to_parquet(
    output_path,
    index=False,
)

print(f"\nSaved to: {output_path}")