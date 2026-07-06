#!/usr/bin/env python3
"""Prepare all CEPII legacy geography references."""
from __future__ import annotations

from pathlib import Path

from vacoda.cif_fob.references.geography.cepii_legacy import (
    prepare_legacy_country_geography,
    prepare_legacy_pair_geography,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_DIR = PROJECT_ROOT / "references" / "geography" / "cepii_legacy" / "source"
PREPARED_DIR = PROJECT_ROOT / "references" / "geography" / "cepii_legacy" / "prepared"


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")

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
