"""
script.prepare_oil_price.py
This script prepares the annual average crude oil price reference from the World Bank's CMO-Historical-Data-Annual dataset.
It loads the nominal annual prices, standardizes column names, converts values to numeric, removes non-data rows, validates values, derives log prices, 
and saves the cleaned reference as a Parquet file for use in the CIF-FOB model.
"""

import numpy as np
import pandas as pd
from vacoda.config.paths import get_references_macroeconomic_dir

input_path = get_references_macroeconomic_dir() / "source" / "CMO-Historical-Data-Annual.xlsx"
output_path = get_references_macroeconomic_dir() / "prepared" / "annual_oil_prices.parquet"


# ============================================================
# Load source
# ============================================================

raw = pd.read_excel(
    input_path,
    sheet_name="Annual Prices (Nominal)",
    skiprows=6,
    usecols="A:B",
)


# ============================================================
# Validate source shape
# ============================================================

if raw.shape[1] != 2:
    raise ValueError(
        "Expected exactly two columns from the World Bank oil-price "
        f"sheet, but found {raw.shape[1]}."
    )


# ============================================================
# Standardize columns
# ============================================================

oil = raw.copy()

oil.columns = [
    "year",
    "crude_oil_average_usd_per_bbl",
]


# ============================================================
# Convert source values
# ============================================================

oil["year"] = pd.to_numeric(
    oil["year"],
    errors="coerce",
)

oil["crude_oil_average_usd_per_bbl"] = pd.to_numeric(
    oil["crude_oil_average_usd_per_bbl"],
    errors="coerce",
)


# ============================================================
# Remove non-data rows
# ============================================================

oil = (
    oil
    .dropna(
        subset=[
            "year",
            "crude_oil_average_usd_per_bbl",
        ]
    )
    .copy()
)


# ============================================================
# Validate source values
# ============================================================

non_integer_years = oil.loc[
    oil["year"] % 1 != 0,
    "year",
]

if not non_integer_years.empty:
    raise ValueError(
        "Found non-integer year values: "
        f"{non_integer_years.tolist()}"
    )

non_positive_prices = oil.loc[
    oil["crude_oil_average_usd_per_bbl"] <= 0,
    [
        "year",
        "crude_oil_average_usd_per_bbl",
    ],
]

if not non_positive_prices.empty:
    raise ValueError(
        "Oil prices must be strictly positive before taking logs:\n"
        f"{non_positive_prices}"
    )


# ============================================================
# Set final types and derive log price
# ============================================================

oil["year"] = oil["year"].astype("int16")

oil["log_crude_oil_average"] = np.log(
    oil["crude_oil_average_usd_per_bbl"]
)

oil = (
    oil
    .sort_values("year")
    .reset_index(drop=True)
)


# ============================================================
# Final validation
# ============================================================

duplicate_years = oil.duplicated("year").sum()

if duplicate_years:
    raise ValueError(
        f"Found {duplicate_years} duplicate year rows."
    )

if oil.empty:
    raise ValueError(
        "No valid oil-price observations remained after cleaning."
    )

missing_values = oil.isna().sum()

if missing_values.any():
    raise ValueError(
        "Missing values remain in prepared oil-price data:\n"
        f"{missing_values[missing_values > 0]}"
    )


# ============================================================
# Diagnostics
# ============================================================

print("\nShape:")
print(oil.shape)

print("\nData types:")
print(oil.dtypes)

print("\nMissing values:")
print(oil.isna().sum())

print(
    "\nYear range:",
    f"{oil['year'].min()}–{oil['year'].max()}",
)

print(
    "\nDuplicate years:",
    duplicate_years,
)

print("\nOil-price summary:")
print(
    oil["crude_oil_average_usd_per_bbl"]
    .describe(
        percentiles=[
            0.01,
            0.50,
            0.95,
            0.99,
        ]
    )
)

print("\nPreview:")
print(oil.head())

print("\nLatest years:")
print(oil.tail())


# ============================================================
# Save prepared reference
# ============================================================

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

oil.to_parquet(
    output_path,
    index=False,
)

print(f"\nSaved to: {output_path}")