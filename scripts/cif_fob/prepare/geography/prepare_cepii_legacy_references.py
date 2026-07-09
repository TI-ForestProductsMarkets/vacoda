#!/usr/bin/env python3
"""Prepare all CEPII legacy geography references."""
from __future__ import annotations

from vacoda.cif_fob.references.geography.cepii_legacy import (
    prepare_legacy_country_geography,
    prepare_legacy_pair_geography,
)
from vacoda.config.paths import get_geography_cepii_legacy_dir

SOURCE_DIR = get_geography_cepii_legacy_dir() / "source"
PREPARED_DIR = get_geography_cepii_legacy_dir() / "prepared"


def main() -> int:
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Prepared directory: {PREPARED_DIR}")

    prepare_legacy_country_geography(
        input_path=SOURCE_DIR / "geo_cepii.xls",
        output_path=PREPARED_DIR / "country_geography.parquet",
    )
    prepare_legacy_pair_geography(
        input_path=SOURCE_DIR / "dist_cepii.xls",
        output_path=PREPARED_DIR / "country_pair_geography.parquet",
    )

    print("\nCEPII legacy geography preparation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
