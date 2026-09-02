"""
vacoda.steps.value.deflate

Convert synchronized nominal USD trade values into constant-price USD
using the prepared US GDP-deflator reference series.

The canonical nominal Trade Sync value is preserved. This step adds:

- usd_gdp_deflator
- usd_deflation_factor
- trade_value_reconciled_constant_usd

The constant-price value is expressed in USD purchasing-power terms
of the configured reference year.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.steps.value import ValueConfig
from vacoda.utils.sql import sql_literal
from vacoda.pipeline.checkpoint import (
    file_fingerprint,
    semantic_config_fingerprint,
    validate_local_checkpoint,
    write_local_checkpoint,
)

DEFLATOR_COLUMN = "usd_gdp_deflator"
DEFLATION_FACTOR_COLUMN = "usd_deflation_factor"
CONSTANT_VALUE_COLUMN = "trade_value_reconciled_constant_usd"


# ======================================================================
# Input validation
# ======================================================================


def _require_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    required_columns: set[str],
    label: str,
) -> None:
    """Require columns to exist in a Parquet artifact."""
    parquet_sql = sql_literal(parquet_path)

    columns = {
        str(row[0])
        for row in con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet({parquet_sql})
            """
        ).fetchall()
    }

    missing = sorted(
        required_columns - columns
    )

    if missing:
        raise RuntimeError(
            f"{label} is missing required columns: "
            + ", ".join(missing)
        )


def _validate_deflator_reference(
    *,
    con: duckdb.DuckDBPyConnection,
    deflator_path: Path,
    cfg: ValueConfig,
) -> float:
    """
    Validate the prepared deflator and return the reference-year value.
    """
    deflator_sql = sql_literal(deflator_path)

    _require_columns(
        con=con,
        parquet_path=deflator_path,
        required_columns={
            "country_code",
            "indicator_code",
            "year",
            "gdp_deflator",
        },
        label="Prepared GDP-deflator reference",
    )

    duplicate_years = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT year
            FROM read_parquet({deflator_sql})
            GROUP BY year
            HAVING COUNT(*) <> 1
        )
        """
    ).fetchone()[0]

    if duplicate_years:
        raise RuntimeError(
            "Prepared GDP-deflator reference contains duplicate years."
        )

    invalid_values = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet({deflator_sql})
        WHERE gdp_deflator IS NULL
           OR NOT isfinite(gdp_deflator)
           OR gdp_deflator <= 0
        """
    ).fetchone()[0]

    if invalid_values:
        raise RuntimeError(
            "Prepared GDP-deflator reference contains missing, "
            "non-finite, or non-positive values."
        )

    deflator_cfg = cfg.reference_data.us_gdp_deflator

    unexpected_series = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet({deflator_sql})
        WHERE country_code <> ?
           OR indicator_code <> ?
        """,
        [
            deflator_cfg.country_code,
            deflator_cfg.indicator_code,
        ],
    ).fetchone()[0]

    if unexpected_series:
        raise RuntimeError(
            "Prepared GDP-deflator reference contains observations "
            "outside the configured country/indicator series."
        )

    reference_rows = con.execute(
        f"""
        SELECT gdp_deflator
        FROM read_parquet({deflator_sql})
        WHERE year = ?
        """,
        [cfg.method.reference_year],
    ).fetchall()

    if len(reference_rows) != 1:
        raise RuntimeError(
            "Expected exactly one GDP-deflator observation for "
            f"reference year {cfg.method.reference_year}, "
            f"found {len(reference_rows)}."
        )

    reference_deflator = float(
        reference_rows[0][0]
    )

    if reference_deflator <= 0:
        raise RuntimeError(
            "Reference-year GDP deflator must be positive."
        )

    return reference_deflator


# ======================================================================
# Deflation
# ======================================================================


def deflate_trade_values(
    *,
    cfg: ValueConfig,
    input_path: Path,
    deflator_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Convert nominal Trade Sync values to constant-reference-year USD.

    Formula
    -------
    constant_value_t =
        nominal_value_t
        * reference_year_deflator
        / year_t_deflator

    Nominal values are preserved unchanged.
    """
    input_path = Path(input_path)
    deflator_path = Path(deflator_path)

    output_path = cfg.product_path(
        "deflated_trade"
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Value input parquet not found: {input_path}"
        )

    if not deflator_path.is_file():
        raise FileNotFoundError(
            "Prepared GDP-deflator parquet not found: "
            f"{deflator_path}"
        )

    ## Local checkpointing
    checkpoint_path = (
        cfg.output_dir()
        / "checkpoints"
        / "deflated_trade.json"
    )

    config_fp = semantic_config_fingerprint(
        config=cfg,
        excluded_paths=(
            ("engine", "threads"),
        ),
    )

    input_fp = {
        "trade_sync_input": file_fingerprint(
            input_path,
            hash_file=True,
        ),
        "prepared_deflator": file_fingerprint(
            deflator_path,
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
            "REUSE — Deflated Value trade local "
            "checkpoint matches: %s",
            output_path,
        )

        return {
            "status": "reused",
            "output_path": output_path,
            "deflated_trade_path": output_path,
            "audit_passed": True,
            "ready_for_value_audit": True,
        }

    logger.info(
        "REBUILD — Deflated Value trade: %s",
        "forced" if force else reason,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    input_sql = sql_literal(input_path)
    deflator_sql = sql_literal(deflator_path)
    temporary_sql = sql_literal(temporary_path)

    year_column = cfg.input.columns.year
    nominal_value_column = cfg.input.columns.nominal_value

    con = duckdb.connect()

    try:
        con.execute(
            f"PRAGMA threads={int(cfg.engine.threads)}"
        )

        # --------------------------------------------------------------
        # Validate canonical Trade Sync input.
        # --------------------------------------------------------------

        _require_columns(
            con=con,
            parquet_path=input_path,
            required_columns={
                year_column,
                nominal_value_column,
            },
            label="Trade Sync Value input",
        )

        # Avoid silently overwriting columns if this step is accidentally
        # applied to an already-deflated artifact.
        _require_no_existing_value_columns(
            con=con,
            input_path=input_path,
        )

        reference_deflator = (
            _validate_deflator_reference(
                con=con,
                deflator_path=deflator_path,
                cfg=cfg,
            )
        )

        # --------------------------------------------------------------
        # Require complete deflator coverage for the configured scope.
        # --------------------------------------------------------------

        missing_years = con.execute(
            f"""
            SELECT DISTINCT
                                CAST(t.{year_column} AS INTEGER) AS year
            FROM read_parquet({input_sql}) AS t

            LEFT JOIN read_parquet({deflator_sql}) AS d
                            ON CAST(t.{year_column} AS INTEGER) = d.year

            WHERE
                                t.{year_column}
                    BETWEEN ? AND ?
                AND d.year IS NULL

            ORDER BY year
            """,
            [
                cfg.scope.min_year,
                cfg.scope.max_year,
            ],
        ).fetchall()

        if missing_years:
            missing_text = ", ".join(
                str(row[0])
                for row in missing_years
            )

            raise RuntimeError(
                "GDP-deflator reference does not cover all years "
                "present in the Value input: "
                f"{missing_text}."
            )

        logger.info(
            "Deflating Trade Sync values."
        )
        logger.info(
            "  Input: %s",
            input_path,
        )
        logger.info(
            "  Deflator: %s",
            deflator_path,
        )
        logger.info(
            "  Reference year: %s",
            cfg.method.reference_year,
        )
        logger.info(
            "  Reference-year deflator: %.9f",
            reference_deflator,
        )
        logger.info(
            "  Nominal value column: %s",
            nominal_value_column,
        )
        logger.info(
            "  Constant value column: %s",
            CONSTANT_VALUE_COLUMN,
        )

        # --------------------------------------------------------------
        # Write constant-price output.
        # --------------------------------------------------------------

        con.execute(
            f"""
            COPY (
                SELECT
                    t.*,

                    CAST(
                        d.gdp_deflator
                        AS DOUBLE
                    ) AS {DEFLATOR_COLUMN},

                    CAST(
                        ? / d.gdp_deflator
                        AS DOUBLE
                    ) AS {DEFLATION_FACTOR_COLUMN},

                    CASE
                        WHEN t.{nominal_value_column} IS NULL
                            THEN NULL

                        ELSE
                            CAST(
                                t.{nominal_value_column}
                                * (
                                    ? / d.gdp_deflator
                                )
                                AS DOUBLE
                            )
                    END
                        AS {CONSTANT_VALUE_COLUMN}

                FROM read_parquet({input_sql}) AS t

                LEFT JOIN read_parquet({deflator_sql}) AS d
                                    ON CAST(t.{year_column} AS INTEGER) = d.year
            )
            TO {temporary_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """,
            [
                reference_deflator,
                reference_deflator,
            ],
        )

        # --------------------------------------------------------------
        # Lightweight step-level validation.
        # Final methodological auditing belongs to audit.py.
        # --------------------------------------------------------------

        output_sql = sql_literal(temporary_path)

        input_rows = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({input_sql})
                """
            ).fetchone()[0]
        )

        output_rows = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({output_sql})
                """
            ).fetchone()[0]
        )

        row_count_preserved = (
            output_rows == input_rows
        )

        if not row_count_preserved:
            raise RuntimeError(
                "Value deflation changed row count: "
                f"input={input_rows:,}, "
                f"output={output_rows:,}."
            )

        missing_output_deflator = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({output_sql})
                                WHERE {year_column}
                          BETWEEN ? AND ?
                  AND {DEFLATOR_COLUMN} IS NULL
                """,
                [
                    cfg.scope.min_year,
                    cfg.scope.max_year,
                ],
            ).fetchone()[0]
        )

        deflator_coverage_complete = (
            missing_output_deflator == 0
        )

        if not deflator_coverage_complete:
            raise RuntimeError(
                "Value output contains rows without a GDP deflator "
                "inside the configured year range."
            )

        bad_reference_factor = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({output_sql})
                                WHERE {year_column} = ?
                  AND ABS(
                        {DEFLATION_FACTOR_COLUMN} - 1.0
                      ) > 1e-12
                """,
                [cfg.method.reference_year],
            ).fetchone()[0]
        )

        reference_factor_valid = (
            bad_reference_factor == 0
        )

        if not reference_factor_valid:
            raise RuntimeError(
                "Reference-year Value rows do not have "
                "usd_deflation_factor = 1."
            )

        step_audit_passed = bool(
            row_count_preserved
            and deflator_coverage_complete
            and reference_factor_valid
        )

        ready_for_value_audit = (
            step_audit_passed
        )

        temporary_path.replace(
            output_path
        )

        write_local_checkpoint(
            checkpoint_path=checkpoint_path,
            step_name="deflated_trade",
            run_id=run_id,
            config=config_fp,
            input_fingerprint=input_fp,
            output_path=output_path,
        )

        logger.info(
            "WRITE — Deflated trade values: %s",
            output_path,
        )
        logger.info(
            "  Rows: %s",
            f"{output_rows:,}",
        )
        logger.info(
            "  Step audit passed: %s",
            step_audit_passed,
        )
        logger.info(
            "  Ready for Value audit: %s",
            ready_for_value_audit,
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "output_path": output_path,
            "deflated_trade_path": output_path,
            "input_rows": input_rows,
            "output_rows": output_rows,
            "reference_year": int(
                cfg.method.reference_year
            ),
            "reference_year_deflator": (
                reference_deflator
            ),
            "nominal_value_column": (
                nominal_value_column
            ),
            "constant_value_column": (
                CONSTANT_VALUE_COLUMN
            ),
            "row_count_preserved": (
                row_count_preserved
            ),
            "deflator_coverage_complete": (
                deflator_coverage_complete
            ),
            "reference_factor_valid": (
                reference_factor_valid
            ),
            "audit_passed": (
                step_audit_passed
            ),
            "ready_for_value_audit": (
                ready_for_value_audit
            ),
        }

    finally:
        con.close()


def _require_no_existing_value_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
) -> None:
    """Prevent accidental second deflation of an already processed file."""
    input_sql = sql_literal(input_path)

    columns = {
        str(row[0])
        for row in con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet({input_sql})
            """
        ).fetchall()
    }

    generated_columns = {
        DEFLATOR_COLUMN,
        DEFLATION_FACTOR_COLUMN,
        CONSTANT_VALUE_COLUMN,
    }

    existing = sorted(
        generated_columns & columns
    )

    if existing:
        raise RuntimeError(
            "Value input already contains generated deflation "
            "columns: "
            + ", ".join(existing)
        )