"""
Prepare all geography references required by the CIF/FOB model.

This workflow:

1. prepares source-specific CEPII and VACODA references;
2. validates the prepared artifacts;
3. assembles the final model-facing production references.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from vacoda.cif_fob.references.geography.cepii_gravity import (
    prepare_gravity_pair_geography,
)
from vacoda.cif_fob.references.geography.cepii_legacy import (
    prepare_legacy_country_geography,
    prepare_legacy_pair_geography,
)
from vacoda.cif_fob.references.geography.vacoda_extension import (
    prepare_vacoda_country_geography,
    prepare_vacoda_geography_crosswalk,
)

def build_production_geography_references(
    *,
    legacy_country_path: Path,
    gravity_pair_path: Path,
    vacoda_country_path: Path,
    vacoda_crosswalk_path: Path,
    production_dir: Path,
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Assemble final model-facing geography references.

    Country geography combines CEPII legacy countries with VACODA
    country extensions.

    Pair geography uses the CEPII Gravity reference directly.

    The VACODA geography crosswalk is written separately for country
    and pair resolution. The files retain both usability flags so the
    downstream loader can apply the relevant flag.
    """

    input_paths = {
        "legacy country geography": Path(legacy_country_path),
        "Gravity pair geography": Path(gravity_pair_path),
        "VACODA country geography": Path(vacoda_country_path),
        "VACODA geography crosswalk": Path(vacoda_crosswalk_path),
    }

    missing_inputs = {
        label: path
        for label, path in input_paths.items()
        if not path.is_file()
    }

    if missing_inputs:
        details = "\n".join(
            f"  {label}: {path}"
            for label, path in missing_inputs.items()
        )
        raise FileNotFoundError(
            "Prepared geography inputs are missing:\n"
            + details
        )

    production_dir = Path(production_dir)
    production_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    production_paths = {
        "country_geography": (
            production_dir / "country_geography.parquet"
        ),
        "country_pair_geography": (
            production_dir / "country_pair_geography.parquet"
        ),
        "country_crosswalk": (
            production_dir / "country_crosswalk.parquet"
        ),
        "pair_crosswalk": (
            production_dir / "pair_crosswalk.parquet"
        ),
    }

    # ============================================================
    # Country geography
    # ============================================================

    legacy_country = pd.read_parquet(
        legacy_country_path
    )

    vacoda_country = pd.read_parquet(
        vacoda_country_path
    )

    required_country_columns = [
        "country_iso3",
        "area_km2",
        "internal_distance_km",
        "landlocked",
        "continent",
        "log_area_km2",
        "log_internal_distance_km",
    ]

    for label, frame in [
        ("CEPII legacy country geography", legacy_country),
        ("VACODA country geography", vacoda_country),
    ]:
        missing_columns = [
            column
            for column in required_country_columns
            if column not in frame.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{label} is missing required columns: "
                + ", ".join(missing_columns)
            )

    legacy_country = legacy_country[
        required_country_columns
    ].copy()

    vacoda_country = vacoda_country[
        required_country_columns
    ].copy()

    shared_country_codes = sorted(
        set(legacy_country["country_iso3"].dropna())
        & set(vacoda_country["country_iso3"].dropna())
    )

    if shared_country_codes:
        raise ValueError(
            "VACODA country extensions overlap with CEPII legacy "
            "country geography:\n  "
            + "\n  ".join(shared_country_codes)
        )

    country_geography = pd.concat(
        [
            legacy_country,
            vacoda_country,
        ],
        ignore_index=True,
    )

    duplicate_countries = country_geography[
        "country_iso3"
    ].duplicated(
        keep=False
    )

    if duplicate_countries.any():
        raise ValueError(
            "Duplicate countries found in production geography:\n"
            + country_geography.loc[
                duplicate_countries
            ].to_string(index=False)
        )

    country_geography = (
        country_geography
        .sort_values("country_iso3")
        .reset_index(drop=True)
    )

    country_geography.to_parquet(
        production_paths["country_geography"],
        index=False,
    )

    # ============================================================
    # Pair geography
    # ============================================================

    pair_geography = pd.read_parquet(
        gravity_pair_path
    )

    required_pair_columns = [
        "origin_iso3",
        "destination_iso3",
    ]

    missing_pair_columns = [
        column
        for column in required_pair_columns
        if column not in pair_geography.columns
    ]

    if missing_pair_columns:
        raise ValueError(
            "Gravity pair geography is missing required columns: "
            + ", ".join(missing_pair_columns)
        )

    duplicate_pairs = pair_geography.duplicated(
        [
            "origin_iso3",
            "destination_iso3",
        ],
        keep=False,
    )

    if duplicate_pairs.any():
        raise ValueError(
            "Duplicate country pairs found in Gravity geography:\n"
            + pair_geography.loc[
                duplicate_pairs,
                [
                    "origin_iso3",
                    "destination_iso3",
                ],
            ].to_string(index=False)
        )

    pair_geography = (
        pair_geography
        .sort_values(
            [
                "origin_iso3",
                "destination_iso3",
            ]
        )
        .reset_index(drop=True)
    )

    pair_geography.to_parquet(
        production_paths["country_pair_geography"],
        index=False,
    )

    # ============================================================
    # Country and pair crosswalks
    # ============================================================

    crosswalk = pd.read_parquet(
        vacoda_crosswalk_path
    )

    required_crosswalk_columns = [
        "trade_iso3",
        "cepii_iso3",
        "mapping_type",
        "usable_country",
        "usable_pair",
        "is_proxy",
    ]

    missing_crosswalk_columns = [
        column
        for column in required_crosswalk_columns
        if column not in crosswalk.columns
    ]

    if missing_crosswalk_columns:
        raise ValueError(
            "VACODA geography crosswalk is missing required columns: "
            + ", ".join(missing_crosswalk_columns)
        )

    crosswalk = (
        crosswalk[required_crosswalk_columns]
        .sort_values("trade_iso3")
        .reset_index(drop=True)
    )

    duplicate_crosswalk_codes = crosswalk[
        "trade_iso3"
    ].duplicated(
        keep=False
    )

    if duplicate_crosswalk_codes.any():
        raise ValueError(
            "Duplicate trade codes found in geography crosswalk:\n"
            + crosswalk.loc[
                duplicate_crosswalk_codes
            ].to_string(index=False)
        )

    crosswalk.to_parquet(
        production_paths["country_crosswalk"],
        index=False,
    )

    crosswalk.to_parquet(
        production_paths["pair_crosswalk"],
        index=False,
    )

    if verbose:
        print("\nProduction geography references assembled:")
        print(
            "  Countries: "
            f"{len(country_geography):,}"
        )
        print(
            "  Country pairs: "
            f"{len(pair_geography):,}"
        )
        print(
            "  Crosswalk rows: "
            f"{len(crosswalk):,}"
        )

        for label, path in production_paths.items():
            print(f"  {label}: {path}")

    return production_paths

def prepare_geography_references(
    project_root: Path,
    *,
    gravity_reference_year: int = 2021,
    threads: int = 16,
    verbose: bool = True,
) -> dict[str, Path]:
    """Prepare all source-specific geography reference Parquets."""
    project_root = Path(project_root).resolve()

    if threads <= 0:
        raise ValueError("threads must be greater than zero.")

    geography_root = project_root / "references" / "geography"

    legacy_source_dir = geography_root / "cepii_legacy" / "source"
    legacy_prepared_dir = geography_root / "cepii_legacy" / "prepared"

    gravity_source_dir = (
        geography_root / "cepii_gravity_2021" / "source"
    )
    gravity_prepared_dir = (
        geography_root / "cepii_gravity_2021" / "prepared"
    )

    vacoda_source_dir = (
        geography_root / "vacoda_extensions" / "source"
    )
    vacoda_prepared_dir = (
        geography_root / "vacoda_extensions" / "prepared"
    )

    production_dir = geography_root / "production"

    source_paths = {
        "CEPII legacy country source": (
            legacy_source_dir / "geo_cepii.xls"
        ),
        "CEPII legacy pair source": (
            legacy_source_dir / "dist_cepii.xls"
        ),
        "CEPII Gravity source": (
            gravity_source_dir / "Gravity_V202211.csv"
        ),
        "VACODA country extension source": (
            vacoda_source_dir / "country_geography_extensions.csv"
        ),
        "VACODA geography crosswalk source": (
            vacoda_source_dir / "comtrade_geography_crosswalk.csv"
        ),
    }

    missing_sources = {
        label: path
        for label, path in source_paths.items()
        if not path.is_file()
    }
    if missing_sources:
        details = "\n".join(
            f"  {label}: {path}"
            for label, path in missing_sources.items()
        )
        raise FileNotFoundError(
            "Required geography source files are missing:\n"
            + details
        )

    output_paths = {
        "legacy_country_geography": (
            legacy_prepared_dir / "country_geography.parquet"
        ),
        "legacy_pair_geography": (
            legacy_prepared_dir / "country_pair_geography.parquet"
        ),
        "gravity_pair_geography": (
            gravity_prepared_dir / "country_pair_geography.parquet"
        ),
        "vacoda_country_geography": (
            vacoda_prepared_dir / "country_geography.parquet"
        ),
        "vacoda_geography_crosswalk": (
            vacoda_prepared_dir
            / "comtrade_geography_crosswalk.parquet"
        ),
    }

    if verbose:
        print(f"Project root: {project_root}")
        print("\nValidated geography source files:")
        for label, path in source_paths.items():
            print(f"  {label}: {path}")

    prepare_legacy_country_geography(
        input_path=source_paths["CEPII legacy country source"],
        output_path=output_paths["legacy_country_geography"],
        verbose=verbose,
    )

    prepare_legacy_pair_geography(
        input_path=source_paths["CEPII legacy pair source"],
        output_path=output_paths["legacy_pair_geography"],
        verbose=verbose,
    )

    prepare_gravity_pair_geography(
        input_path=source_paths["CEPII Gravity source"],
        output_path=output_paths["gravity_pair_geography"],
        reference_year=gravity_reference_year,
        threads=threads,
        verbose=verbose,
    )

    prepare_vacoda_country_geography(
        input_path=source_paths["VACODA country extension source"],
        output_path=output_paths["vacoda_country_geography"],
        verbose=verbose,
    )

    prepare_vacoda_geography_crosswalk(
        input_path=source_paths["VACODA geography crosswalk source"],
        output_path=output_paths["vacoda_geography_crosswalk"],
        verbose=verbose,
    )

    production_paths = build_production_geography_references(
        legacy_country_path=output_paths[
        "legacy_country_geography"
    ],
        gravity_pair_path=output_paths[
        "gravity_pair_geography"
    ],
        vacoda_country_path=output_paths[
        "vacoda_country_geography"
    ],
        vacoda_crosswalk_path=output_paths[
        "vacoda_geography_crosswalk"
    ],
    production_dir=production_dir,
    verbose=verbose,
    )

    output_paths.update(
        {
        f"production_{name}": path
        for name, path in production_paths.items()
    }
    )

    if verbose:
        print("\nAll source-specific geography references prepared:")
        for label, path in output_paths.items():
            print(f"  {label}: {path}")

    return output_paths
