"""
Prepare normalized geography references from CEPII legacy datasets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def prepare_legacy_country_geography(
    input_path: Path,
    output_path: Path,
    *,
    sheet_name: str = "geo_cepii",
    verbose: bool = True,
) -> pd.DataFrame:
    """Prepare one country-level geography row per ISO3 code."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"CEPII legacy country-geography source not found: {input_path}"
        )

    geo_raw = pd.read_excel(input_path, sheet_name=sheet_name)

    required_cols = {"iso3", "landlocked", "area", "dis_int", "continent"}
    missing_cols = required_cols.difference(geo_raw.columns)
    if missing_cols:
        raise ValueError(f"Missing required CEPII columns: {sorted(missing_cols)}")

    country_geo = (
        geo_raw[["iso3", "area", "dis_int", "landlocked", "continent"]]
        .rename(
            columns={
                "iso3": "country_iso3",
                "area": "area_km2",
                "dis_int": "internal_distance_km",
            }
        )
        .copy()
    )

    country_geo["country_iso3"] = (
        country_geo["country_iso3"].astype("string").str.strip().str.upper()
    )
    country_geo["area_km2"] = pd.to_numeric(country_geo["area_km2"], errors="coerce")
    country_geo["internal_distance_km"] = pd.to_numeric(
        country_geo["internal_distance_km"], errors="coerce"
    )
    country_geo["landlocked"] = pd.to_numeric(
        country_geo["landlocked"], errors="coerce"
    ).astype("Int8")
    country_geo["continent"] = (
        country_geo["continent"]
        .astype("string")
        .str.strip()
        .replace({"No": "Unclassified", "": pd.NA, ".": pd.NA})
    )

    country_geo["log_area_km2"] = np.where(
        country_geo["area_km2"].gt(0), np.log(country_geo["area_km2"]), np.nan
    )
    country_geo["log_internal_distance_km"] = np.where(
        country_geo["internal_distance_km"].gt(0),
        np.log(country_geo["internal_distance_km"]),
        np.nan,
    )

    duplicate_iso3 = (
        country_geo.loc[
            country_geo.duplicated("country_iso3", keep=False), "country_iso3"
        ]
        .dropna()
        .unique()
    )

    if len(duplicate_iso3) > 0:
        duplicate_check = (
            country_geo.loc[country_geo["country_iso3"].isin(duplicate_iso3)]
            .groupby("country_iso3", dropna=False)
            .agg(
                rows=("country_iso3", "size"),
                area_values=("area_km2", "nunique"),
                internal_distance_values=("internal_distance_km", "nunique"),
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
                "Inconsistent country-level values found for duplicate ISO3 codes:\n"
                f"{inconsistent_duplicates}"
            )

    country_geo = (
        country_geo.drop_duplicates(subset=["country_iso3"])
        .sort_values("country_iso3")
        .reset_index(drop=True)
    )

    if country_geo["country_iso3"].isna().any():
        raise ValueError("Found rows with missing country_iso3.")
    if country_geo["country_iso3"].duplicated().any():
        raise ValueError("Found duplicate country_iso3 values.")

    invalid_area = country_geo["area_km2"].isna() | country_geo["area_km2"].le(0)
    if invalid_area.any():
        raise ValueError(
            "Found missing or nonpositive area_km2 values:\n"
            + country_geo.loc[invalid_area, ["country_iso3", "area_km2"]].to_string(index=False)
        )

    invalid_internal_distance = (
        country_geo["internal_distance_km"].isna()
        | country_geo["internal_distance_km"].le(0)
    )
    if invalid_internal_distance.any():
        raise ValueError(
            "Found missing or nonpositive internal_distance_km values:\n"
            + country_geo.loc[
                invalid_internal_distance,
                ["country_iso3", "internal_distance_km"],
            ].to_string(index=False)
        )

    invalid_landlocked = (
        country_geo["landlocked"].isna()
        | ~country_geo["landlocked"].isin([0, 1])
    )
    if invalid_landlocked.any():
        raise ValueError(
            "Unexpected values found in landlocked:\n"
            + country_geo.loc[
                invalid_landlocked, ["country_iso3", "landlocked"]
            ].to_string(index=False)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    country_geo.to_parquet(output_path, index=False)

    if verbose:
        print("\nPrepared CEPII legacy country geography")
        print(f"Rows:       {len(country_geo):,}")
        print(f"Countries:  {country_geo['country_iso3'].nunique():,}")
        print(f"Saved to:   {output_path}")

    return country_geo


def prepare_legacy_pair_geography(
    input_path: Path,
    output_path: Path,
    *,
    sheet_name: str = "dist_cepii",
    verbose: bool = True,
) -> pd.DataFrame:
    """Prepare the CEPII legacy bilateral geography reference."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"CEPII legacy pair-geography source not found: {input_path}"
        )

    raw = pd.read_excel(input_path, sheet_name=sheet_name)

    required_cols = {"iso_o", "iso_d", "dist", "distcap", "distw", "distwces", "contig"}
    missing_cols = required_cols.difference(raw.columns)
    if missing_cols:
        raise ValueError(f"Missing required CEPII columns: {sorted(missing_cols)}")

    pair_geo = (
        raw[["iso_o", "iso_d", "dist", "distcap", "distw", "distwces", "contig"]]
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

    for column in ["origin_iso3", "destination_iso3"]:
        pair_geo[column] = pair_geo[column].astype("string").str.strip().str.upper()

    for column in [
        "distance_simple_km",
        "distance_capital_km",
        "distance_weighted_km",
        "distance_weighted_ces_km",
    ]:
        pair_geo[column] = pd.to_numeric(pair_geo[column], errors="coerce")

    pair_geo["contiguous"] = pd.to_numeric(
        pair_geo["contiguous"], errors="coerce"
    ).astype("Int8")

    pair_geo["distance_model_km"] = pair_geo["distance_weighted_km"].combine_first(
        pair_geo["distance_capital_km"]
    )
    pair_geo["distance_model_source"] = pd.Series(
        np.select(
            [
                pair_geo["distance_weighted_km"].notna(),
                pair_geo["distance_weighted_km"].isna()
                & pair_geo["distance_capital_km"].notna(),
            ],
        ["weighted", "capital"],
        default="missing",
    ),
    index = pair_geo.index,
    dtype="string"
    )

    pair_geo["log_distance_model_km"] = np.where(
        pair_geo["distance_model_km"].gt(0),
        np.log(pair_geo["distance_model_km"]),
        np.nan,
    )

    pair_geo = pair_geo.sort_values(
        ["origin_iso3", "destination_iso3"]
    ).reset_index(drop=True)

    missing_pair_keys = pair_geo[["origin_iso3", "destination_iso3"]].isna().any(axis=1)
    if missing_pair_keys.any():
        raise ValueError(
            "Found rows with missing country-pair keys:\n"
            + pair_geo.loc[
                missing_pair_keys, ["origin_iso3", "destination_iso3"]
            ].to_string(index=False)
        )

    duplicate_pairs = pair_geo.duplicated(
        ["origin_iso3", "destination_iso3"], keep=False
    )
    if duplicate_pairs.any():
        raise ValueError(
            "Found duplicate country pairs:\n"
            + pair_geo.loc[
                duplicate_pairs, ["origin_iso3", "destination_iso3"]
            ].to_string(index=False)
        )

    invalid_model_distance = (
        pair_geo["distance_model_km"].isna()
        | pair_geo["distance_model_km"].le(0)
    )
    if invalid_model_distance.any():
        raise ValueError(
            "Found missing or nonpositive model distances:\n"
            + pair_geo.loc[
                invalid_model_distance,
                [
                    "origin_iso3",
                    "destination_iso3",
                    "distance_weighted_km",
                    "distance_capital_km",
                    "distance_model_km",
                ],
            ].to_string(index=False)
        )

    invalid_contiguous = (
        pair_geo["contiguous"].isna()
        | ~pair_geo["contiguous"].isin([0, 1])
    )
    if invalid_contiguous.any():
        raise ValueError(
            "Unexpected values found in contiguous:\n"
            + pair_geo.loc[
                invalid_contiguous,
                ["origin_iso3", "destination_iso3", "contiguous"],
            ].to_string(index=False)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair_geo.to_parquet(output_path, index=False)

    if verbose:
        print("\nPrepared CEPII legacy pair geography")
        print(f"Rows:            {len(pair_geo):,}")
        print(f"Distinct pairs:  {len(pair_geo):,}")
        print("\nModel-distance sources:")
        print(pair_geo["distance_model_source"].value_counts(dropna=False))
        print(f"\nSaved to: {output_path}")

    return pair_geo
