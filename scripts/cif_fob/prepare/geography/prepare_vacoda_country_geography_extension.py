#!/usr/bin/env python3
"""
Prepare VACODA-owned geography extension references.
"""

from pathlib import Path

from vacoda.cif_fob.references.geography.vacoda_extension import (
    prepare_vacoda_country_geography,
    prepare_vacoda_geography_crosswalk,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

SOURCE_DIR = (
    PROJECT_ROOT
    / "references"
    / "geography"
    / "vacoda_extensions"
    / "source"
)

PREPARED_DIR = (
    PROJECT_ROOT
    / "references"
    / "geography"
    / "vacoda_extensions"
    / "prepared"
)


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")

    prepare_vacoda_country_geography(
        input_path=(
            SOURCE_DIR
            / "country_geography_extensions.csv"
        ),
        output_path=(
            PREPARED_DIR
            / "country_geography.parquet"
        ),
    )

    prepare_vacoda_geography_crosswalk(
        input_path=(
            SOURCE_DIR
            / "comtrade_geography_crosswalk.csv"
        ),
        output_path=(
            PREPARED_DIR
            / "comtrade_geography_crosswalk.parquet"
        ),
    )

    print("\nVACODA geography extension preparation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())