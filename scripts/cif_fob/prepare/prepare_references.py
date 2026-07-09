"""
Prepare all reference data required by the CIF/FOB workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from vacoda.cif_fob.references.geography.workflow import prepare_geography_references
from vacoda.cif_fob.references.macroeconomic import prepare_annual_oil_prices
from vacoda.config.paths import get_references_macroeconomic_dir

N_THREADS = 58
GRAVITY_REFERENCE_YEAR = 2021

def prepare_cif_fob_references(
    *,
    gravity_reference_year: int = GRAVITY_REFERENCE_YEAR,
    threads: int = N_THREADS,
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Prepare all CIF/FOB reference files.

    This includes:

    - source-specific geography references;
    - final production geography references;
    - annual crude-oil prices.
    """

    outputs = prepare_geography_references(
        gravity_reference_year=gravity_reference_year,
        threads=threads,
        verbose=verbose,
    )


    oil_input_path = get_references_macroeconomic_dir() / "source" / "CMO-Historical-Data-Annual.xlsx"
    oil_output_path = get_references_macroeconomic_dir() / "prepared" / "annual_oil_prices.parquet"

    prepare_annual_oil_prices(
        input_path=oil_input_path,
        output_path=oil_output_path,
        verbose=verbose,
    )

    outputs["annual_oil_prices"] = oil_output_path

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare all source-specific references required by "
            "the VACODA CIF/FOB workflow."
        )
    )

    parser.add_argument(
        "--gravity-reference-year",
        type=int,
        default=GRAVITY_REFERENCE_YEAR,
        help=(
            f"CEPII Gravity reference year used for bilateral "
            f"geography. Default: {GRAVITY_REFERENCE_YEAR}."
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=N_THREADS,
        help=f"DuckDB thread count. Default: {N_THREADS}.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed preparation diagnostics.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.threads <= 0:
        raise ValueError("--threads must be greater than zero.")

    outputs = prepare_cif_fob_references(
        gravity_reference_year=args.gravity_reference_year,
        threads=args.threads,
        verbose=not args.quiet,
    )

    print("\nCIF/FOB reference preparation completed.")

    print("\nPrepared artifacts:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())