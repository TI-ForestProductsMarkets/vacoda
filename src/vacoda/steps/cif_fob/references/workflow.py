"""
Prepare all reference data required by the CIF/FOB workflow.
"""

from __future__ import annotations

from pathlib import Path

from vacoda.steps.cif_fob.references.geography.workflow import  prepare_geography_references
from vacoda.steps.cif_fob.references.macroeconomic import prepare_annual_oil_prices
from vacoda.config.paths import get_repo_macroeconomic_dir


def prepare_cif_fob_references(
    *,
    gravity_reference_year: int = 2021,
    threads: int = 16,
    verbose: bool = True,
) -> dict[str, Path]:
    """Prepare all CIF/FOB source-specific reference files."""

    outputs = prepare_geography_references(
        gravity_reference_year=gravity_reference_year,
        threads=threads,
        verbose=verbose,
    )

    oil_input_path = get_repo_macroeconomic_dir() / "source" / "CMO-Historical-Data-Annual.xlsx"
    oil_output_path = get_repo_macroeconomic_dir() / "prepared" / "annual_oil_prices.parquet"
    

    prepare_annual_oil_prices(
        input_path= oil_input_path,
        output_path= oil_output_path,
        verbose=verbose,
    )

    outputs["annual_oil_prices"] = oil_output_path

    return outputs