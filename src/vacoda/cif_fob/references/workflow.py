"""
Prepare all reference data required by the CIF/FOB workflow.
"""

from __future__ import annotations

from pathlib import Path

from vacoda.cif_fob.references.geography.workflow import (
    prepare_geography_references,
)
from vacoda.cif_fob.references.macroeconomic import (
    prepare_annual_oil_prices,
)


def prepare_cif_fob_references(
    project_root: Path,
    *,
    gravity_reference_year: int = 2021,
    threads: int = 16,
    verbose: bool = True,
) -> dict[str, Path]:
    """Prepare all CIF/FOB source-specific reference files."""

    project_root = Path(project_root).resolve()

    outputs = prepare_geography_references(
        project_root=project_root,
        gravity_reference_year=gravity_reference_year,
        threads=threads,
        verbose=verbose,
    )

    oil_output = (
        project_root
        / "references"
        / "macroeconomic"
        / "prepared"
        / "annual_oil_prices.parquet"
    )

    prepare_annual_oil_prices(
        input_path=(
            project_root
            / "references"
            / "macroeconomic"
            / "source"
            / "CMO-Historical-Data-Annual.xlsx"
        ),
        output_path=oil_output,
        verbose=verbose,
    )

    outputs["annual_oil_prices"] = oil_output

    return outputs