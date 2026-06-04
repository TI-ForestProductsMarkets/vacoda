"""
scripts/hs_sync/02_audit_harvard_generator_inputs.py

* Check whether the staged/exported files are readable by the Harvard generator.
* Checks raw_parquet, aggregated files, product codes, matrix completeness, and Harvard dimension tests.

It audits:
- raw_parquet marker files
- aggregated_by_year_not_converted parquet files
- reporter/partner identifier style
- product-code coverage
- generated matrix completeness, if matrices exist

How to run:
    python -m scripts.hs_sync.02_audit_harvard_generator_inputs
       or
    python scripts/hs_sync/02_audit_harvard_generator_inputs.py \
            --source-hs H5 \
            --source-year 2021 \
            --target-hs H6 \
            --target-year 2022

Important:
1. it should be run after staging (01_stage_comtrade_for_harvard_generator.py) and after generating matrices via harvard generator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vacoda.hs_sync.harvard_audit import run_harvard_generator_audit


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--harvard-data-root",
        type=Path,
        default=Path("/storage/vacoda/data/external/harvard_comtrade_bulk"),
    )
    parser.add_argument(
        "--generator-root",
        type=Path,
        default=Path(
            "/storage/vacoda/data/references/comtrade-conversion-weights/generator"
        ),
    )
    parser.add_argument("--source-hs", default="H4")
    parser.add_argument("--source-year", type=int, default=2016)
    parser.add_argument("--target-hs", default="H5")
    parser.add_argument("--target-year", type=int, default=2017)

    args = parser.parse_args()

    run_harvard_generator_audit(
        harvard_data_root=args.harvard_data_root,
        generator_root=args.generator_root,
        source_hs=args.source_hs,
        source_year=args.source_year,
        target_hs=args.target_hs,
        target_year=args.target_year,
    )

if __name__ == "__main__":
    main()