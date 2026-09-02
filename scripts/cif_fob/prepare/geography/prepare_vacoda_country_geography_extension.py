#!/usr/bin/env python3
"""
Prepare VACODA-owned geography extension references.
"""

from vacoda.steps.cif_fob.references.geography.vacoda_extension import (
    prepare_vacoda_country_geography,
    prepare_vacoda_geography_crosswalk,
)

from vacoda.config.paths import get_geography_vacoda_extension_dir


SOURCE_DIR = get_geography_vacoda_extension_dir() / "source"
PREPARED_DIR = get_geography_vacoda_extension_dir() / "prepared"


def main() -> int:
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Prepared directory: {PREPARED_DIR}")

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