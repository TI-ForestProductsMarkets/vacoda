"""
vacoda.value.audit

Final integrity audit for Value deflation.

The audit independently verifies that the Value output:

- preserves the Trade Sync population;
- preserves nominal synchronized trade values;
- joins exactly one valid GDP deflator per year;
- applies the configured reference-year deflation formula correctly;
- preserves NULL monetary values;
- contains valid constant-price values;
- is ready for downstream analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from vacoda.config.value import ValueConfig
from vacoda.utils.sql import (
    sql_identifier,
    sql_literal,
)

DEFLATOR_COLUMN = "usd_gdp_deflator"
DEFLATION_FACTOR_COLUMN = "usd_deflation_factor"
CONSTANT_VALUE_COLUMN = "trade_value_reconciled_constant_usd"


# ======================================================================
# Audit helpers
# ======================================================================


def _audit_row(
    *,
    check: str,
    passed: bool,
    value: Any = None,
    expected: Any = None,
    details: str = "",
) -> dict[str, Any]:
    """Build one canonical Value audit row."""
    return {
        "check": check,
        "passed": bool(passed),
        "value": value,
        "expected": expected,
        "details": details,
    }


def _require_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    required_columns: set[str],
    label: str,
) -> set[str]:
    """Require columns to exist and return the full column set."""
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

    return columns


# ======================================================================
# Public audit
# ======================================================================


def run_value_output_audit(
    *,
    cfg: ValueConfig,
    input_path: Path,
    deflated_trade_path: Path,
    deflator_path: Path,
    run_id: str,
    logger,
) -> dict[str, Any]:
    """
    Run the final Value integrity audit.

    Parameters
    ----------
    cfg:
        Parsed Value configuration.
    input_path:
        Canonical Trade Sync input.
    deflated_trade_path:
        Final Value output.
    deflator_path:
        Prepared annual US GDP-deflator reference.
    run_id:
        Master-pipeline run identifier.
    logger:
        Configured workflow logger.
    """
    input_path = Path(input_path)
    deflated_trade_path = Path(
        deflated_trade_path
    )
    deflator_path = Path(deflator_path)

    audit_path = cfg.product_path("audit")

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Value audit input not found: {input_path}"
        )

    if not deflated_trade_path.is_file():
        raise FileNotFoundError(
            "Value deflated output not found: "
            f"{deflated_trade_path}"
        )

    if not deflator_path.is_file():
        raise FileNotFoundError(
            "Prepared GDP-deflator reference not found: "
            f"{deflator_path}"
        )

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    year_column = cfg.input.columns.year
    nominal_value_column = (
        cfg.input.columns.nominal_value
    )

    input_sql = sql_literal(input_path)
    output_sql = sql_literal(
        deflated_trade_path
    )
    deflator_sql = sql_literal(
        deflator_path
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"PRAGMA threads={int(cfg.engine.threads)}"
        )

        # ==============================================================
        # Structural validation
        # ==============================================================

        input_columns = _require_columns(
            con=con,
            parquet_path=input_path,
            required_columns={
                year_column,
                nominal_value_column,
            },
            label="Trade Sync Value input",
        )

        output_columns = _require_columns(
            con=con,
            parquet_path=deflated_trade_path,
            required_columns={
                year_column,
                nominal_value_column,
                DEFLATOR_COLUMN,
                DEFLATION_FACTOR_COLUMN,
                CONSTANT_VALUE_COLUMN,
            },
            label="Value deflated output",
        )

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

        audit_rows: list[
            dict[str, Any]
        ] = []

        # ==============================================================
        # 1. Output exists and is non-empty
        # ==============================================================

        output_rows = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({output_sql})
                """
            ).fetchone()[0]
        )

        audit_rows.append(
            _audit_row(
                check="output_non_empty",
                passed=output_rows > 0,
                value=output_rows,
                expected="> 0",
            )
        )

        # ==============================================================
        # 2. Row count preserved
        # ==============================================================

        input_rows = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({input_sql})
                """
            ).fetchone()[0]
        )

        audit_rows.append(
            _audit_row(
                check="row_count_preserved",
                passed=(
                    input_rows == output_rows
                ),
                value=output_rows,
                expected=input_rows,
            )
        )

        # ==============================================================
        # 3. Original input columns preserved
        # ==============================================================

        missing_original_columns = sorted(
            input_columns - output_columns
        )

        audit_rows.append(
            _audit_row(
                check="input_columns_preserved",
                passed=(
                    not missing_original_columns
                ),
                value=(
                    ", ".join(
                        missing_original_columns
                    )
                    if missing_original_columns
                    else ""
                ),
                expected="no missing input columns",
            )
        )

        # ==============================================================
        # 4. Deflator uniqueness by year
        # ==============================================================

        duplicate_deflator_years = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT year
                    FROM read_parquet(
                        {deflator_sql}
                    )
                    GROUP BY year
                    HAVING COUNT(*) <> 1
                )
                """
            ).fetchone()[0]
        )

        audit_rows.append(
            _audit_row(
                check="deflator_unique_by_year",
                passed=(
                    duplicate_deflator_years == 0
                ),
                value=duplicate_deflator_years,
                expected=0,
            )
        )

        # ==============================================================
        # 5. Deflator values valid
        # ==============================================================

        invalid_deflators = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    {deflator_sql}
                )
                WHERE gdp_deflator IS NULL
                   OR NOT isfinite(
                       gdp_deflator
                   )
                   OR gdp_deflator <= 0
                """
            ).fetchone()[0]
        )

        audit_rows.append(
            _audit_row(
                check="deflator_values_valid",
                passed=(
                    invalid_deflators == 0
                ),
                value=invalid_deflators,
                expected=0,
            )
        )

        # ==============================================================
        # 6. Configured years covered
        # ==============================================================

        missing_deflator_rows = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    {output_sql}
                )
                WHERE {year_column}
                          BETWEEN ? AND ?
                  AND {DEFLATOR_COLUMN}
                          IS NULL
                """,
                [
                    cfg.scope.min_year,
                    cfg.scope.max_year,
                ],
            ).fetchone()[0]
        )

        audit_rows.append(
            _audit_row(
                check="deflator_coverage_complete",
                passed=(
                    missing_deflator_rows == 0
                ),
                value=missing_deflator_rows,
                expected=0,
            )
        )

        # ==============================================================
        # 7. Output deflator matches reference
        # ==============================================================

        deflator_mismatches = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    {output_sql}
                ) AS o

                JOIN read_parquet(
                    {deflator_sql}
                ) AS d
                  ON CAST(
                      o.{year_column}
                      AS INTEGER
                  ) = d.year

                WHERE
                    o.{DEFLATOR_COLUMN}
                        IS NULL
                    OR ABS(
                        o.{DEFLATOR_COLUMN}
                        - d.gdp_deflator
                    ) > 1e-12
                """
            ).fetchone()[0]
        )

        audit_rows.append(
            _audit_row(
                check=(
                    "output_deflator_matches_reference"
                ),
                passed=(
                    deflator_mismatches == 0
                ),
                value=deflator_mismatches,
                expected=0,
            )
        )

        # ==============================================================
        # 8. Reference-year deflator exists exactly once
        # ==============================================================

        reference_rows = con.execute(
            f"""
            SELECT gdp_deflator
            FROM read_parquet(
                {deflator_sql}
            )
            WHERE year = ?
            """,
            [
                cfg.method.reference_year
            ],
        ).fetchall()

        reference_year_valid = (
            len(reference_rows) == 1
        )

        reference_deflator = (
            float(reference_rows[0][0])
            if reference_year_valid
            else None
        )

        audit_rows.append(
            _audit_row(
                check=(
                    "reference_year_deflator_available"
                ),
                passed=reference_year_valid,
                value=len(reference_rows),
                expected=1,
                details=(
                    ""
                    if reference_deflator is None
                    else (
                        "reference_deflator="
                        f"{reference_deflator:.12f}"
                    )
                ),
            )
        )

        if reference_deflator is None:
            raise RuntimeError(
                "Cannot complete Value audit because "
                "the configured reference-year "
                "deflator is unavailable."
            )

        # ==============================================================
        # 9. Deflation factor arithmetic
        # ==============================================================

        factor_mismatches = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    {output_sql}
                )
                WHERE
                    {DEFLATOR_COLUMN}
                        IS NOT NULL
                    AND (
                        {DEFLATION_FACTOR_COLUMN}
                            IS NULL
                        OR ABS(
                            {DEFLATION_FACTOR_COLUMN}
                            - (
                                ?
                                / {DEFLATOR_COLUMN}
                            )
                        ) > 1e-12
                    )
                """,
                [reference_deflator],
            ).fetchone()[0]
        )

        audit_rows.append(
            _audit_row(
                check="deflation_factor_correct",
                passed=(
                    factor_mismatches == 0
                ),
                value=factor_mismatches,
                expected=0,
            )
        )

        # ==============================================================
        # 10. Reference-year factor = 1
        # ==============================================================

        bad_reference_factors = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    {output_sql}
                )
                WHERE {year_column} = ?
                  AND (
                      {DEFLATION_FACTOR_COLUMN}
                          IS NULL
                      OR ABS(
                          {DEFLATION_FACTOR_COLUMN}
                          - 1.0
                      ) > 1e-12
                  )
                """,
                [
                    cfg.method.reference_year
                ],
            ).fetchone()[0]
        )

        audit_rows.append(
            _audit_row(
                check=(
                    "reference_year_factor_is_one"
                ),
                passed=(
                    bad_reference_factors == 0
                ),
                value=bad_reference_factors,
                expected=0,
            )
        )

        # ==============================================================
        # 11. Complete Trade Sync rows preserved exactly
        #
        # Parquet scan order is not a stable row identity. Compare the
        # complete multiset of original input columns instead.
        #
        # EXCEPT ALL preserves duplicate multiplicity, so this verifies
        # both directions without relying on row order or a synthetic
        # row number.
        # ==============================================================

        input_column_rows = con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet({input_sql})
            """
        ).fetchall()

        ordered_input_columns = [
            str(row[0])
            for row in input_column_rows
        ]

        input_projection = ", ".join(
            sql_identifier(column)
            for column in ordered_input_columns
        )

        missing_from_output = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT {input_projection}
                    FROM read_parquet({input_sql})

                    EXCEPT ALL

                    SELECT {input_projection}
                    FROM read_parquet({output_sql})
                )
                """
            ).fetchone()[0]
        )

        extra_in_output = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT {input_projection}
                    FROM read_parquet({output_sql})

                    EXCEPT ALL

                    SELECT {input_projection}
                    FROM read_parquet({input_sql})
                )
                """
            ).fetchone()[0]
        )

        preserved_row_mismatches = (
            missing_from_output
            + extra_in_output
        )

        audit_rows.append(
            _audit_row(
                check="input_rows_preserved_exactly",
                passed=(
                    preserved_row_mismatches == 0
                ),
                value=preserved_row_mismatches,
                expected=0,
                details=(
                    f"missing_from_output={missing_from_output}; "
                    f"extra_in_output={extra_in_output}"
                ),
            )
        )
        # ==============================================================
        # 12. NULL nominal -> NULL constant value
        # ==============================================================

        null_propagation_failures = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    {output_sql}
                )
                WHERE
                    {nominal_value_column}
                        IS NULL
                    AND
                    {CONSTANT_VALUE_COLUMN}
                        IS NOT NULL
                """
            ).fetchone()[0]
        )

        audit_rows.append(
            _audit_row(
                check=(
                    "null_nominal_propagates_to_constant"
                ),
                passed=(
                    null_propagation_failures == 0
                ),
                value=(
                    null_propagation_failures
                ),
                expected=0,
            )
        )

        # ==============================================================
        # 13. Non-null nominal -> correct constant value
        # ==============================================================

        constant_value_mismatches = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    {output_sql}
                )
                WHERE
                    {nominal_value_column}
                        IS NOT NULL
                    AND (
                        {CONSTANT_VALUE_COLUMN}
                            IS NULL
                        OR ABS(
                            {CONSTANT_VALUE_COLUMN}
                            - (
                                {nominal_value_column}
                                *
                                {DEFLATION_FACTOR_COLUMN}
                            )
                        ) >
                            GREATEST(
                                1e-9,
                                ABS(
                                    {nominal_value_column}
                                    *
                                    {DEFLATION_FACTOR_COLUMN}
                                )
                                * 1e-12
                            )
                    )
                """
            ).fetchone()[0]
        )

        audit_rows.append(
            _audit_row(
                check="constant_value_correct",
                passed=(
                    constant_value_mismatches
                    == 0
                ),
                value=(
                    constant_value_mismatches
                ),
                expected=0,
            )
        )

        # ==============================================================
        # 14. Positive nominal values remain positive
        # ==============================================================

        nonpositive_constant_values = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    {output_sql}
                )
                WHERE
                    {nominal_value_column} > 0
                    AND (
                        {CONSTANT_VALUE_COLUMN}
                            IS NULL
                        OR
                        {CONSTANT_VALUE_COLUMN}
                            <= 0
                    )
                """
            ).fetchone()[0]
        )

        audit_rows.append(
            _audit_row(
                check=(
                    "positive_nominal_remains_positive"
                ),
                passed=(
                    nonpositive_constant_values
                    == 0
                ),
                value=(
                    nonpositive_constant_values
                ),
                expected=0,
            )
        )

        # ==============================================================
        # 15. Generated numeric columns finite
        # ==============================================================

        invalid_generated_values = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    {output_sql}
                )
                WHERE
                    (
                        {DEFLATOR_COLUMN}
                            IS NOT NULL
                        AND NOT isfinite(
                            {DEFLATOR_COLUMN}
                        )
                    )
                    OR
                    (
                        {DEFLATION_FACTOR_COLUMN}
                            IS NOT NULL
                        AND NOT isfinite(
                            {DEFLATION_FACTOR_COLUMN}
                        )
                    )
                    OR
                    (
                        {CONSTANT_VALUE_COLUMN}
                            IS NOT NULL
                        AND NOT isfinite(
                            {CONSTANT_VALUE_COLUMN}
                        )
                    )
                """
            ).fetchone()[0]
        )

        audit_rows.append(
            _audit_row(
                check="generated_values_finite",
                passed=(
                    invalid_generated_values
                    == 0
                ),
                value=(
                    invalid_generated_values
                ),
                expected=0,
            )
        )

        # ==============================================================
        # Write audit
        # ==============================================================

        audit_frame = pd.DataFrame(
            audit_rows
        )

        audit_frame.to_csv(
            audit_path,
            index=False,
        )

        failed = audit_frame.loc[
            ~audit_frame["passed"]
        ]

        n_failed_audit_rows = int(
            len(failed)
        )

        audit_passed = (
            n_failed_audit_rows == 0
        )

        logger.info(
            "WRITE — Value output audit: %s",
            audit_path,
        )
        logger.info(
            "  Checks: %s",
            len(audit_frame),
        )
        logger.info(
            "  Passed: %s",
            int(
                audit_frame["passed"].sum()
            ),
        )
        logger.info(
            "  Failed: %s",
            n_failed_audit_rows,
        )

        if not audit_passed:
            failed_names = ", ".join(
                failed["check"].astype(str)
            )

            raise RuntimeError(
                "Value output audit failed: "
                f"{failed_names}. "
                f"Audit: {audit_path}"
            )

        logger.info(
            "Value output audit passed."
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "audit_path": audit_path,
            "n_audit_checks": int(
                len(audit_frame)
            ),
            "n_failed_audit_rows": (
                n_failed_audit_rows
            ),
            "audit_passed": True,
            "ready_for_validation": True,
        }

    finally:
        con.close()