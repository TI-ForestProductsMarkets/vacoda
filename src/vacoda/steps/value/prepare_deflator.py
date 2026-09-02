"""
vacoda.steps.value.prepare_deflator

Prepare the annual US GDP deflator reference series used by the Value
workflow.

The source is the World Bank wide-format CSV for indicator
NY.GDP.DEFL.ZS. This step:

1. reads the raw World Bank export;
2. selects the configured country and indicator;
3. reshapes annual values from wide to long format;
4. validates the canonical annual series;
5. writes the prepared reference Parquet.

No value deflation is performed here. The configured reference year is
applied later by the Value transformation step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from vacoda.config.steps.value import ValueConfig
from vacoda.pipeline.checkpoint import (
    file_fingerprint,
    semantic_config_fingerprint,
    validate_local_checkpoint,
    write_local_checkpoint,
)

# ======================================================================
# Source reading
# ======================================================================


def _find_world_bank_header_row(
    source_path: Path,
) -> int:
    """
    Return the zero-based row containing the World Bank CSV header.

    World Bank downloads commonly contain metadata lines before the
    actual table, so the header position is discovered rather than
    hard-coded.
    """
    with source_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as handle:
        for row_number, line in enumerate(handle):
            if (
                '"Country Name"' in line
                and '"Country Code"' in line
                and '"Indicator Code"' in line
            ):
                return row_number

    raise RuntimeError(
        "Could not locate the World Bank data header in "
        f"{source_path}."
    )


def _read_world_bank_source(
    source_path: Path,
) -> pd.DataFrame:
    """Read the World Bank CSV after resolving its true header row."""
    header_row = _find_world_bank_header_row(
        source_path
    )

    return pd.read_csv(
        source_path,
        skiprows=header_row,
    )


# ======================================================================
# Canonical preparation
# ======================================================================


def _year_columns(
    frame: pd.DataFrame,
) -> list[str]:
    """Return World Bank annual columns in chronological order."""
    years = [
        str(column)
        for column in frame.columns
        if str(column).isdigit()
        and len(str(column)) == 4
    ]

    if not years:
        raise RuntimeError(
            "World Bank GDP-deflator source contains no annual columns."
        )

    return sorted(
        years,
        key=int,
    )


def _prepare_annual_deflator(
    *,
    source: pd.DataFrame,
    country_code: str,
    indicator_code: str,
) -> pd.DataFrame:
    """Build the canonical annual GDP-deflator reference table."""
    required_columns = {
        "Country Name",
        "Country Code",
        "Indicator Name",
        "Indicator Code",
    }

    missing = sorted(
        required_columns - set(source.columns)
    )

    if missing:
        raise RuntimeError(
            "World Bank GDP-deflator source is missing required "
            "columns: "
            + ", ".join(missing)
        )

    selected = source.loc[
        source["Country Code"]
        .astype("string")
        .str.upper()
        .eq(country_code.upper())
        & source["Indicator Code"]
        .astype("string")
        .eq(indicator_code)
    ].copy()

    if selected.empty:
        raise RuntimeError(
            "Configured World Bank GDP-deflator series was not found: "
            f"country_code={country_code!r}, "
            f"indicator_code={indicator_code!r}."
        )

    if len(selected) != 1:
        raise RuntimeError(
            "Expected exactly one World Bank GDP-deflator series for "
            f"country_code={country_code!r} and "
            f"indicator_code={indicator_code!r}, "
            f"found {len(selected):,}."
        )

    years = _year_columns(selected)

    annual = (
        selected[
            [
                "Country Code",
                "Indicator Code",
                *years,
            ]
        ]
        .melt(
            id_vars=[
                "Country Code",
                "Indicator Code",
            ],
            value_vars=years,
            var_name="year",
            value_name="gdp_deflator",
        )
        .rename(
            columns={
                "Country Code": "country_code",
                "Indicator Code": "indicator_code",
            }
        )
    )

    annual["year"] = pd.to_numeric(
        annual["year"],
        errors="raise",
    ).astype("int32")

    annual["gdp_deflator"] = pd.to_numeric(
        annual["gdp_deflator"],
        errors="coerce",
    )

    # Years with no published World Bank value are not part of the
    # usable prepared annual series.
    annual = annual.loc[
        annual["gdp_deflator"].notna()
    ].copy()

    annual["country_code"] = (
        annual["country_code"]
        .astype("string")
        .str.upper()
    )

    annual["indicator_code"] = (
        annual["indicator_code"]
        .astype("string")
    )

    annual = annual.sort_values(
        "year"
    ).reset_index(drop=True)

    return annual


# ======================================================================
# Audit
# ======================================================================


def _audit_prepared_deflator(
    *,
    prepared: pd.DataFrame,
    country_code: str,
    indicator_code: str,
    reference_year: int,
) -> dict[str, Any]:
    """Validate the canonical prepared GDP-deflator series."""
    required_columns = {
        "year",
        "country_code",
        "indicator_code",
        "gdp_deflator",
    }

    missing = sorted(
        required_columns - set(prepared.columns)
    )

    if missing:
        raise RuntimeError(
            "Prepared GDP-deflator artifact is missing required "
            "columns: "
            + ", ".join(missing)
        )

    if prepared.empty:
        raise RuntimeError(
            "Prepared GDP-deflator artifact contains no observations."
        )

    if prepared["year"].duplicated().any():
        duplicate_years = (
            prepared.loc[
                prepared["year"].duplicated(
                    keep=False
                ),
                "year",
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        raise RuntimeError(
            "Prepared GDP-deflator artifact contains duplicate years: "
            + ", ".join(
                str(year)
                for year in duplicate_years
            )
        )

    if not prepared["country_code"].eq(
        country_code.upper()
    ).all():
        raise RuntimeError(
            "Prepared GDP-deflator artifact contains an unexpected "
            "country code."
        )

    if not prepared["indicator_code"].eq(
        indicator_code
    ).all():
        raise RuntimeError(
            "Prepared GDP-deflator artifact contains an unexpected "
            "indicator code."
        )

    if (
        prepared["gdp_deflator"].isna().any()
        or prepared["gdp_deflator"].le(0).any()
    ):
        raise RuntimeError(
            "Prepared GDP-deflator values must all be finite and "
            "strictly positive."
        )

    reference_rows = prepared.loc[
        prepared["year"].eq(reference_year)
    ]

    if len(reference_rows) != 1:
        raise RuntimeError(
            "Configured value reference year is unavailable in the "
            "prepared GDP-deflator series: "
            f"{reference_year}."
        )

    return {
        "audit_passed": True,
        "n_rows": int(len(prepared)),
        "year_min": int(prepared["year"].min()),
        "year_max": int(prepared["year"].max()),
        "reference_year": int(reference_year),
        "reference_year_deflator": float(
            reference_rows.iloc[0]["gdp_deflator"]
        ),
    }


# ======================================================================
# Public step
# ======================================================================


def prepare_us_gdp_deflator(
    *,
    cfg: ValueConfig,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Prepare and validate the configured US GDP-deflator reference.

    The output lives in the repository reference-data tree rather than
    the Value interim-output directory.
    """
    source_path = cfg.deflator_source_path()
    output_path = cfg.deflator_prepared_path()

    ## Local checkpointing 
    checkpoint_path = (
        cfg.output_dir()
        / "checkpoints"
        / "prepared_deflator.json"
    )

    config_fp = semantic_config_fingerprint(
        config=cfg,
        excluded_paths=(
            ("engine", "threads"),
        ),
    )

    input_fp = {
        "source": file_fingerprint(
            source_path,
            hash_file=True,
        ),
    }

    reusable, reason = validate_local_checkpoint(
        checkpoint_path=checkpoint_path,
        current_config=config_fp,
        current_input=input_fp,
        output_path=output_path,
    )

    if not force and reusable:
        logger.info(
            "REUSE — Prepared US GDP deflator local "
            "checkpoint matches: %s",
            output_path,
        )

        return {
            "status": "reused",
            "output_path": output_path,
            "prepared_deflator_path": output_path,
            "audit_passed": True,
            "ready_for_value_deflation": True,
        }

    logger.info(
        "REBUILD — Prepared US GDP deflator: %s",
        "forced" if force else reason,
    )

    if not source_path.is_file():
        raise FileNotFoundError(
            "World Bank GDP-deflator source CSV not found: "
            f"{source_path}"
        )

    deflator_cfg = (
        cfg.reference_data.us_gdp_deflator
    )

    logger.info(
        "Preparing US GDP-deflator reference."
    )
    logger.info(
        "  Source: %s",
        source_path,
    )
    logger.info(
        "  Country code: %s",
        deflator_cfg.country_code,
    )
    logger.info(
        "  Indicator code: %s",
        deflator_cfg.indicator_code,
    )
    logger.info(
        "  Reference year: %s",
        cfg.method.reference_year,
    )

    source = _read_world_bank_source(
        source_path
    )

    prepared = _prepare_annual_deflator(
        source=source,
        country_code=deflator_cfg.country_code,
        indicator_code=deflator_cfg.indicator_code,
    )

    audit = _audit_prepared_deflator(
        prepared=prepared,
        country_code=deflator_cfg.country_code,
        indicator_code=deflator_cfg.indicator_code,
        reference_year=cfg.method.reference_year,
    )

    audit_passed = bool(
        audit["n_rows"] > 0
        and audit["year_min"] <= cfg.method.reference_year
        and audit["year_max"] >= cfg.method.reference_year
        and audit["reference_year"] == cfg.method.reference_year
        and audit["reference_year_deflator"] > 0
    )

    ready_for_value_deflation = bool(
        audit_passed
    )

    if not audit_passed:
        raise RuntimeError(
            "Prepared GDP-deflator audit failed."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    prepared.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        output_path
    )

    write_local_checkpoint(
        checkpoint_path=checkpoint_path,
        step_name="prepared_deflator",
        run_id=run_id,
        config=config_fp,
        input_fingerprint=input_fp,
        output_path=output_path,
    )

    logger.info(
        "WRITE — Prepared US GDP deflator: %s",
        output_path,
    )
    logger.info(
        "  Years: %s-%s",
        audit["year_min"],
        audit["year_max"],
    )
    logger.info(
        "  Observations: %s",
        audit["n_rows"],
    )
    logger.info(
        "  %s deflator: %.6f",
        audit["reference_year"],
        audit["reference_year_deflator"],
    )

    return {
        "status": "completed",
        "audit_passed": audit_passed,
        "ready_for_value_deflation": ready_for_value_deflation,
        "output_path": output_path,
        "prepared_deflator_path": output_path,
        **audit,
    }