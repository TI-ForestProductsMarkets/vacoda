"""
Prepare macroeconomic references used by the CIF/FOB model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def prepare_annual_oil_prices(
    input_path: Path,
    output_path: Path,
    *,
    sheet_name: str = "Annual Prices (Nominal)",
    verbose: bool = True,
) -> pd.DataFrame:
    """Prepare annual average crude-oil prices."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"World Bank oil-price source not found: {input_path}"
        )

    raw = pd.read_excel(
        input_path,
        sheet_name=sheet_name,
        skiprows=6,
        usecols="A:B",
    )

    if raw.shape[1] != 2:
        raise ValueError(
            "Expected exactly two columns from the oil-price sheet, "
            f"but found {raw.shape[1]}."
        )

    oil = raw.copy()
    oil.columns = [
        "year",
        "crude_oil_average_usd_per_bbl",
    ]

    oil["year"] = pd.to_numeric(
        oil["year"],
        errors="coerce",
    )
    oil["crude_oil_average_usd_per_bbl"] = pd.to_numeric(
        oil["crude_oil_average_usd_per_bbl"],
        errors="coerce",
    )

    oil = (
        oil.dropna(
            subset=[
                "year",
                "crude_oil_average_usd_per_bbl",
            ]
        )
        .copy()
    )

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
            "Oil prices must be strictly positive:\n"
            + non_positive_prices.to_string(index=False)
        )

    oil["year"] = oil["year"].astype("int16")

    oil["log_crude_oil_average"] = np.log(
        oil["crude_oil_average_usd_per_bbl"]
    )

    oil = (
        oil.sort_values("year")
        .reset_index(drop=True)
    )

    duplicate_years = oil["year"].duplicated(
        keep=False
    )

    if duplicate_years.any():
        raise ValueError(
            "Duplicate oil-price years found:\n"
            + oil.loc[
                duplicate_years,
                ["year", "crude_oil_average_usd_per_bbl"],
            ].to_string(index=False)
        )

    if oil.empty:
        raise ValueError(
            "No valid oil-price observations remained."
        )

    if oil.isna().any().any():
        missing = oil.isna().sum()
        raise ValueError(
            "Missing values remain in prepared oil-price data:\n"
            + missing[missing > 0].to_string()
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    oil.to_parquet(
        output_path,
        index=False,
    )

    if verbose:
        print("\nPrepared annual oil-price reference")
        print(f"Rows:        {len(oil):,}")
        print(
            "Year range:  "
            f"{oil['year'].min()}–{oil['year'].max()}"
        )
        print(f"Saved to:   {output_path}")

    return oil