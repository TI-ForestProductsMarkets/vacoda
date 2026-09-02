#!/usr/bin/env python3
"""
Prepare all production level geography references in one run.

This includes:

- CEPII legacy prepared references;
- CEPII Gravity prepared references;
- VACODA extension prepared references;
- final production geography references.
"""

from __future__ import annotations

import argparse
from vacoda.steps.cif_fob.references.geography.workflow import prepare_geography_references

N_THREADS = 54
DEFAULT_GRAVITY_REFERENCE_YEAR = 2021


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare source-specific and production geography "
            "references for the VACODA CIF/FOB workflow."
        )
    )

    parser.add_argument(
        "--gravity-reference-year",
        type=int,
        default=DEFAULT_GRAVITY_REFERENCE_YEAR,
        help=(
            "CEPII Gravity reference year. "
            f"Default: {DEFAULT_GRAVITY_REFERENCE_YEAR}."
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=N_THREADS,
        help=(
            "DuckDB thread count. "
            f"Default: {N_THREADS}."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress preparation diagnostics.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    outputs = prepare_geography_references(
        gravity_reference_year=args.gravity_reference_year,
        threads=args.threads,
        verbose=not args.quiet,
    )

    print("\nGeography reference preparation completed.")

    print("\nPrepared geography artifacts:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())