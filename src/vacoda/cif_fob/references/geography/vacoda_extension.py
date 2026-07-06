"""
Prepare VACODA-owned country-geography extensions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INTERNAL_DISTANCE_SCALE = 0.67
INTERNAL_DISTANCE_TOLERANCE_KM = 1e-5

REQUIRED_COLUMNS = {
    "country_iso3",
    "country_name",
    "area_km2",
    "internal_distance_km",
    "internal_distance_method",
    "landlocked",
    "continent",
    "identity_source_name",
    "identity_source_url",
    "area_source_name",
    "area_source_url",
    "reason_added",
    "provenance_class",
    "review_status",
    "notes",
}

ALLOWED_CONTINENTS = {
    "Africa",
    "America",
    "Antarctica",
    "Asia",
    "Europe",
    "Oceania",
    "Unclassified",
}
ALLOWED_REVIEW_STATUSES = {"proposed", "approved", "rejected"}
EXPECTED_INTERNAL_DISTANCE_METHOD = "0.67 * sqrt(area_km2 / pi)"


def _standardize_string(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, ".": pd.NA})
    )


def derive_internal_distance(area_km2: pd.Series) -> pd.Series:
    """Derive CEPII-compatible internal distance from total area."""
    return INTERNAL_DISTANCE_SCALE * np.sqrt(area_km2 / np.pi)


def prepare_vacoda_geography_crosswalk(
    input_path: Path,
    output_path: Path,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"VACODA geography crosswalk source not found: {input_path}"
        )

    crosswalk = pd.read_csv(
        input_path,
        dtype="string",
        keep_default_na=True,
    )

    required_columns = [
        "trade_iso3",
        "cepii_iso3",
        "mapping_type",
        "usable_country",
        "usable_pair",
        "is_proxy",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in crosswalk.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required VACODA crosswalk columns: "
            + ", ".join(missing_columns)
        )

    crosswalk = crosswalk[required_columns].copy()

    for column in ["trade_iso3", "cepii_iso3"]:
        crosswalk[column] = (
            crosswalk[column]
            .astype("string")
            .str.strip()
            .str.upper()
            .replace({"": pd.NA})
        )

    crosswalk["mapping_type"] = (
        crosswalk["mapping_type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    boolean_columns = [
        "usable_country",
        "usable_pair",
        "is_proxy",
    ]

    boolean_map = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    for column in boolean_columns:
        normalized = (
            crosswalk[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        invalid_values = normalized.notna() & ~normalized.isin(
            boolean_map
        )

        if invalid_values.any():
            raise ValueError(
                f"Invalid Boolean values in {column}:\n"
                + crosswalk.loc[
                    invalid_values,
                    ["trade_iso3", column],
                ].to_string(index=False)
            )

        crosswalk[column] = (
            normalized.map(boolean_map).astype("boolean")
        )

    if crosswalk["trade_iso3"].isna().any():
        raise ValueError(
            "VACODA crosswalk contains missing trade_iso3 values."
        )

    duplicate_codes = crosswalk["trade_iso3"].duplicated(
        keep=False
    )

    if duplicate_codes.any():
        raise ValueError(
            "Duplicate trade_iso3 values found:\n"
            + crosswalk.loc[
                duplicate_codes
            ].to_string(index=False)
        )

    missing_mapping_type = crosswalk[
        "mapping_type"
    ].isna()

    if missing_mapping_type.any():
        raise ValueError(
            "VACODA crosswalk contains missing mapping_type values."
        )

    missing_flags = crosswalk[
        boolean_columns
    ].isna().any(axis=1)

    if missing_flags.any():
        raise ValueError(
            "VACODA crosswalk contains missing Boolean flags:\n"
            + crosswalk.loc[
                missing_flags
            ].to_string(index=False)
        )

    usable_without_target = (
        (
            crosswalk["usable_country"]
            | crosswalk["usable_pair"]
        )
        & crosswalk["cepii_iso3"].isna()
    )

    if usable_without_target.any():
        raise ValueError(
            "Usable mappings must have a CEPII target:\n"
            + crosswalk.loc[
                usable_without_target
            ].to_string(index=False)
        )

    proxy_without_target = (
        crosswalk["is_proxy"]
        & crosswalk["cepii_iso3"].isna()
    )

    if proxy_without_target.any():
        raise ValueError(
            "Proxy mappings must have a CEPII target:\n"
            + crosswalk.loc[
                proxy_without_target
            ].to_string(index=False)
        )

    unusable_with_active_flag = (
        crosswalk["cepii_iso3"].isna()
        & (
            crosswalk["usable_country"]
            | crosswalk["usable_pair"]
            | crosswalk["is_proxy"]
        )
    )

    if unusable_with_active_flag.any():
        raise ValueError(
            "Rows without a target must be fully inactive:\n"
            + crosswalk.loc[
                unusable_with_active_flag
            ].to_string(index=False)
        )

    crosswalk = (
        crosswalk
        .sort_values("trade_iso3")
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    crosswalk.to_parquet(
        output_path,
        index=False,
    )

    if verbose:
        print("\nPrepared VACODA geography crosswalk")
        print(f"Rows:      {len(crosswalk):,}")
        print(f"Saved to: {output_path}")

    return crosswalk

def prepare_vacoda_country_geography(
    input_path: Path,
    output_path: Path,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Prepare the VACODA country-geography extension Parquet.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            "VACODA country-geography extension CSV not found: "
            f"{input_path}"
        )

    source = pd.read_csv(
        input_path,
        dtype="string",
        keep_default_na=True,
    )

    missing_columns = sorted(REQUIRED_COLUMNS.difference(source.columns))
    if missing_columns:
        raise ValueError(
            "Missing required VACODA extension columns: "
            + ", ".join(missing_columns)
        )

    country_geo = source.copy()
    country_geo["country_iso3"] = (
        _standardize_string(country_geo["country_iso3"]).str.upper()
    )

    string_columns = [
        "country_name",
        "internal_distance_method",
        "continent",
        "identity_source_name",
        "identity_source_url",
        "area_source_name",
        "area_source_url",
        "reason_added",
        "provenance_class",
        "review_status",
        "notes",
    ]
    for column in string_columns:
        country_geo[column] = _standardize_string(
            country_geo[column]
        )
    country_geo["review_status"] = (
        country_geo["review_status"].str.lower()
    )

    for column in [
        "area_km2",
        "area_reference_year",
        "internal_distance_km",
        "landlocked",
    ]:
        if column in country_geo.columns:
            country_geo[column] = pd.to_numeric(
                country_geo[column], errors="coerce"
            )

    country_geo["landlocked"] = (
        country_geo["landlocked"].astype("Int8")
    )
    if "area_reference_year" in country_geo.columns:
        country_geo["area_reference_year"] = (
            country_geo["area_reference_year"].astype("Int64")
        )

    missing_iso3 = country_geo["country_iso3"].isna()
    if missing_iso3.any():
        raise ValueError(
            "Rows with missing country_iso3:\n"
            + country_geo.loc[missing_iso3].to_string(index=False)
        )

    invalid_iso3 = ~country_geo[
        "country_iso3"
    ].str.fullmatch(r"[A-Z]{3}")
    if invalid_iso3.any():
        raise ValueError(
            "Invalid ISO3 codes found:\n"
            + country_geo.loc[
                invalid_iso3,
                ["country_iso3", "country_name"],
            ].to_string(index=False)
        )

    duplicate_iso3 = country_geo["country_iso3"].duplicated(
        keep=False
    )
    if duplicate_iso3.any():
        raise ValueError(
            "Duplicate ISO3 codes found in VACODA extension:\n"
            + country_geo.loc[
                duplicate_iso3,
                ["country_iso3", "country_name"],
            ]
            .sort_values("country_iso3")
            .to_string(index=False)
        )

    invalid_area = (
        country_geo["area_km2"].isna()
        | country_geo["area_km2"].le(0)
    )
    if invalid_area.any():
        raise ValueError(
            "Rows with missing or nonpositive area_km2:\n"
            + country_geo.loc[
                invalid_area,
                ["country_iso3", "country_name", "area_km2"],
            ].to_string(index=False)
        )

    invalid_landlocked = (
        country_geo["landlocked"].isna()
        | ~country_geo["landlocked"].isin([0, 1])
    )
    if invalid_landlocked.any():
        raise ValueError(
            "Invalid landlocked values found:\n"
            + country_geo.loc[
                invalid_landlocked,
                ["country_iso3", "country_name", "landlocked"],
            ].to_string(index=False)
        )

    invalid_continent = (
        country_geo["continent"].isna()
        | ~country_geo["continent"].isin(ALLOWED_CONTINENTS)
    )
    if invalid_continent.any():
        raise ValueError(
            "Unexpected continent labels found:\n"
            + country_geo.loc[
                invalid_continent,
                ["country_iso3", "country_name", "continent"],
            ].to_string(index=False)
        )

    invalid_review_status = (
        country_geo["review_status"].isna()
        | ~country_geo["review_status"].isin(
            ALLOWED_REVIEW_STATUSES
        )
    )
    if invalid_review_status.any():
        raise ValueError(
            "Unexpected review_status values found:\n"
            + country_geo.loc[
                invalid_review_status,
                ["country_iso3", "country_name", "review_status"],
            ].to_string(index=False)
        )

    rejected_rows = country_geo.loc[
        country_geo["review_status"] == "rejected",
        ["country_iso3", "country_name"],
    ]
    if not rejected_rows.empty:
        raise ValueError(
            "Rejected rows must not be included in the source CSV:\n"
            + rejected_rows.to_string(index=False)
        )

    missing_provenance = (
        country_geo[
            [
                "identity_source_name",
                "identity_source_url",
                "area_source_name",
                "area_source_url",
                "reason_added",
                "provenance_class",
            ]
        ]
        .isna()
        .any(axis=1)
    )
    if missing_provenance.any():
        raise ValueError(
            "Rows with incomplete provenance fields:\n"
            + country_geo.loc[
                missing_provenance,
                [
                    "country_iso3",
                    "country_name",
                    "identity_source_name",
                    "identity_source_url",
                    "area_source_name",
                    "area_source_url",
                    "reason_added",
                    "provenance_class",
                ],
            ].to_string(index=False)
        )

    invalid_method = (
        country_geo["internal_distance_method"]
        != EXPECTED_INTERNAL_DISTANCE_METHOD
    )
    if invalid_method.any():
        raise ValueError(
            "Unexpected internal_distance_method values found. "
            f"Expected: {EXPECTED_INTERNAL_DISTANCE_METHOD!r}\n"
            + country_geo.loc[
                invalid_method,
                ["country_iso3", "internal_distance_method"],
            ].to_string(index=False)
        )

    calculated_internal_distance = derive_internal_distance(
        country_geo["area_km2"]
    )
    missing_source_distance = country_geo[
        "internal_distance_km"
    ].isna()
    if missing_source_distance.any():
        raise ValueError(
            "Rows with missing source internal_distance_km:\n"
            + country_geo.loc[
                missing_source_distance,
                [
                    "country_iso3",
                    "country_name",
                    "internal_distance_km",
                ],
            ].to_string(index=False)
        )

    distance_difference = (
        country_geo["internal_distance_km"]
        - calculated_internal_distance
    ).abs()
    inconsistent_distance = distance_difference.gt(
        INTERNAL_DISTANCE_TOLERANCE_KM
    )
    if inconsistent_distance.any():
        diagnostic = country_geo.loc[
            inconsistent_distance,
            [
                "country_iso3",
                "country_name",
                "area_km2",
                "internal_distance_km",
            ],
        ].copy()
        diagnostic["difference_km"] = distance_difference.loc[
            inconsistent_distance
        ]
        raise ValueError(
            "Source internal_distance_km does not agree with the "
            "CEPII-compatible formula:\n"
            + diagnostic.to_string(index=False)
        )

    country_geo["internal_distance_km"] = (
        calculated_internal_distance
    )
    country_geo["log_area_km2"] = np.log(
        country_geo["area_km2"]
    )
    country_geo["log_internal_distance_km"] = np.log(
        country_geo["internal_distance_km"]
    )
    country_geo["reference_family"] = (
        "vacoda_country_geography_extension"
    )
    country_geo["prepared_from"] = str(input_path)
    country_geo["internal_distance_scale"] = (
        INTERNAL_DISTANCE_SCALE
    )

    core_columns = [
        "country_iso3",
        "country_name",
        "area_km2",
        "internal_distance_km",
        "landlocked",
        "continent",
        "log_area_km2",
        "log_internal_distance_km",
    ]
    provenance_columns = [
        column
        for column in country_geo.columns
        if column not in core_columns
    ]
    country_geo = (
        country_geo[core_columns + provenance_columns]
        .sort_values("country_iso3")
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    country_geo.to_parquet(output_path, index=False)

    read_back = pd.read_parquet(output_path)
    if len(read_back) != len(country_geo):
        raise RuntimeError(
            "Prepared Parquet row count differs after read-back."
        )
    if set(read_back["country_iso3"]) != set(
        country_geo["country_iso3"]
    ):
        raise RuntimeError(
            "Prepared Parquet ISO3 coverage differs after read-back."
        )

    if verbose:
        print(
            f"\nPrepared VACODA country geography: "
            f"{country_geo.shape}"
        )
        print("\nReview status:")
        print(
            country_geo["review_status"]
            .value_counts(dropna=False)
            .sort_index()
        )
        print(f"\nSaved to: {output_path}")

    return country_geo
