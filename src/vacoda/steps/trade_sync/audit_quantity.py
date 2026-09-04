"""
src.vacoda.steps.trade_sync.quantity_audit

Integrity audit for Trade Sync bilateral quantity reconciliation.

This stage validates the reconciled-quantity product produced after
trade-value reconciliation has already passed its audit.

The audit checks structural and arithmetic invariants only. It does not
judge whether the quantity-reconciliation methodology is economically
or scientifically optimal; that belongs to diagnostics and reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.steps.trade_sync import TradeSyncConfig
from vacoda.utils.sql import sql_literal

# ======================================================================
# Helpers
# ======================================================================



def _valid_output(path: Path) -> bool:
    """Return True when an existing output appears usable."""
    return path.is_file() and path.stat().st_size > 0


def _require_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    required_columns: set[str],
    label: str,
) -> None:
    """Validate required columns in a Parquet input."""

    parquet_sql = sql_literal(parquet_path)

    available_columns = {
        str(row[0])
        for row in con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet({parquet_sql})
            """
        ).fetchall()
    }

    missing_columns = sorted(
        required_columns - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            f"{label} is missing required columns: "
            + ", ".join(missing_columns)
        )


# ======================================================================
# Main audit
# ======================================================================


def audit_quantity_reconciliation(
    *,
    cfg: TradeSyncConfig,
    reconciled_trade_path: Path,
    reconciled_quantity_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Audit the Trade Sync reconciled bilateral quantity output.

    Parameters
    ----------
    cfg
        Trade Sync configuration.

    reconciled_trade_path
        Audited reconciled-value Trade Sync Parquet used as input to
        quantity reconciliation.

    reconciled_quantity_path
        Reconciled bilateral quantity Parquet to validate.

    run_id
        Pipeline run identifier.

    force
        Rebuild the audit product even when an existing valid artifact
        is present.

    logger
        Pipeline logger.

    Returns
    -------
    dict
        Audit metadata including audit path and pass/fail status.
    """

    reconciled_trade_path = Path(
        reconciled_trade_path
    )
    reconciled_quantity_path = Path(
        reconciled_quantity_path
    )

    if not reconciled_trade_path.is_file():
        raise FileNotFoundError(
            "Audited reconciled Trade Sync input not found: "
            f"{reconciled_trade_path}"
        )

    if not reconciled_quantity_path.is_file():
        raise FileNotFoundError(
            "Reconciled quantity output not found: "
            f"{reconciled_quantity_path}"
        )

    audit_path = cfg.product_path(
        "reconciliation_audit_quantity"
    )

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "🔎 CHECK — Auditing Trade Sync quantity reconciliation."
    )

    input_sql = sql_literal(
        reconciled_trade_path
    )
    quantity_sql = sql_literal(
        reconciled_quantity_path
    )
    audit_sql = sql_literal(
        audit_path
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        # --------------------------------------------------------------
        # Input contracts
        # --------------------------------------------------------------

        required_trade_columns = {
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",
            "trade_value_reconciled",
        }

        required_quantity_columns = {
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",

            "trade_value_reconciled",

            "export_synchronized_quantity_kg",
            "import_synchronized_quantity_kg",

            "has_export_quantity_kg",
            "has_import_quantity_kg",
            "quantity_availability_case",

            "quantity_relative_difference",
            "quantity_simple_mean_kg",
            "quantity_value_weighted_kg",
            "quantity_reliability_weighted_kg",

            "quantity_reconciled_kg",
            "quantity_reconciliation_method",
        }

        _require_columns(
            con=con,
            parquet_path=reconciled_trade_path,
            required_columns=required_trade_columns,
            label="Reconciled Trade Sync input",
        )

        _require_columns(
            con=con,
            parquet_path=reconciled_quantity_path,
            required_columns=required_quantity_columns,
            label="Reconciled quantity output",
        )

        # --------------------------------------------------------------
        # Row counts
        # --------------------------------------------------------------

        input_row_count = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({input_sql})
                """
            ).fetchone()[0]
        )

        quantity_row_count = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({quantity_sql})
                """
            ).fetchone()[0]
        )

        row_count_preserved = (
            input_row_count
            == quantity_row_count
        )

        # --------------------------------------------------------------
        # Bilateral-key uniqueness
        # --------------------------------------------------------------

        input_duplicate_key_rows = int(
            con.execute(
                f"""
                SELECT
                    COALESCE(
                        SUM(n_rows - 1),
                        0
                    )
                FROM (
                    SELECT
                        year,
                        exporter_iso,
                        importer_iso,
                        target_cmd,
                        COUNT(*) AS n_rows

                    FROM read_parquet({input_sql})

                    GROUP BY
                        year,
                        exporter_iso,
                        importer_iso,
                        target_cmd

                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )

        input_bilateral_key_unique = (
            input_duplicate_key_rows == 0
        )

        duplicate_key_rows = int(
            con.execute(
                f"""
                SELECT
                    COALESCE(
                        SUM(n_rows - 1),
                        0
                    )
                FROM (
                    SELECT
                        year,
                        exporter_iso,
                        importer_iso,
                        target_cmd,
                        COUNT(*) AS n_rows

                    FROM read_parquet({quantity_sql})

                    GROUP BY
                        year,
                        exporter_iso,
                        importer_iso,
                        target_cmd

                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )

        bilateral_key_unique = (
            duplicate_key_rows == 0
        )

        # --------------------------------------------------------------
        # Availability partition
        # --------------------------------------------------------------

        availability = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE quantity_availability_case
                        = 'both_quantity'
                ),

                COUNT(*) FILTER (
                    WHERE quantity_availability_case
                        = 'export_only'
                ),

                COUNT(*) FILTER (
                    WHERE quantity_availability_case
                        = 'import_only'
                ),

                COUNT(*) FILTER (
                    WHERE quantity_availability_case
                        = 'no_quantity'
                ),

                COUNT(*) FILTER (
                    WHERE quantity_availability_case NOT IN (
                        'both_quantity',
                        'export_only',
                        'import_only',
                        'no_quantity'
                    )
                    OR quantity_availability_case IS NULL
                )

            FROM read_parquet({quantity_sql})
            """
        ).fetchone()

        (
            n_both_quantity,
            n_export_only,
            n_import_only,
            n_no_quantity,
            n_invalid_availability_case,
        ) = [
            int(value or 0)
            for value in availability
        ]

        availability_partition_rows = (
            n_both_quantity
            + n_export_only
            + n_import_only
            + n_no_quantity
        )

        availability_partition_valid = (
            availability_partition_rows
            == quantity_row_count
            and n_invalid_availability_case == 0
        )

        # --------------------------------------------------------------
        # Availability flags agree with classification
        # --------------------------------------------------------------

        availability_flag_mismatches = int(
            con.execute(
                f"""
                SELECT COUNT(*)

                FROM read_parquet({quantity_sql})

                WHERE
                    (
                        quantity_availability_case
                            = 'both_quantity'
                        AND NOT (
                            has_export_quantity_kg
                            AND has_import_quantity_kg
                        )
                    )

                    OR

                    (
                        quantity_availability_case
                            = 'export_only'
                        AND NOT (
                            has_export_quantity_kg
                            AND NOT has_import_quantity_kg
                        )
                    )

                    OR

                    (
                        quantity_availability_case
                            = 'import_only'
                        AND NOT (
                            has_import_quantity_kg
                            AND NOT has_export_quantity_kg
                        )
                    )

                    OR

                    (
                        quantity_availability_case
                            = 'no_quantity'
                        AND (
                            has_export_quantity_kg
                            OR has_import_quantity_kg
                        )
                    )
                """
            ).fetchone()[0]
        )

        availability_flags_valid = (
            availability_flag_mismatches == 0
        )

        # --------------------------------------------------------------
        # Reconciled quantity validity
        # --------------------------------------------------------------

        invalid_reconciled_quantity_rows = int(
            con.execute(
                f"""
                SELECT COUNT(*)

                FROM read_parquet({quantity_sql})

                WHERE quantity_reconciled_kg IS NOT NULL
                  AND (
                        NOT isfinite(
                            quantity_reconciled_kg
                        )
                        OR quantity_reconciled_kg <= 0
                  )
                """
            ).fetchone()[0]
        )

        reconciled_quantities_valid = (
            invalid_reconciled_quantity_rows == 0
        )

        # --------------------------------------------------------------
        # Rows with usable quantity must receive a reconciled quantity
        # --------------------------------------------------------------

        missing_expected_reconciled_quantity = int(
            con.execute(
                f"""
                SELECT COUNT(*)

                FROM read_parquet({quantity_sql})

                WHERE quantity_availability_case IN (
                    'both_quantity',
                    'export_only',
                    'import_only'
                )
                  AND quantity_reconciled_kg IS NULL
                """
            ).fetchone()[0]
        )

        expected_quantities_present = (
            missing_expected_reconciled_quantity == 0
        )

        # --------------------------------------------------------------
        # No-quantity rows must remain missing
        # --------------------------------------------------------------

        no_quantity_nonmissing_rows = int(
            con.execute(
                f"""
                SELECT COUNT(*)

                FROM read_parquet({quantity_sql})

                WHERE quantity_availability_case
                        = 'no_quantity'
                  AND quantity_reconciled_kg IS NOT NULL
                """
            ).fetchone()[0]
        )

        no_quantity_rows_valid = (
            no_quantity_nonmissing_rows == 0
        )

        # --------------------------------------------------------------
        # Export-only preservation
        # --------------------------------------------------------------

        export_only_mismatches = int(
            con.execute(
                f"""
                SELECT COUNT(*)

                FROM read_parquet({quantity_sql})

                WHERE quantity_availability_case
                        = 'export_only'

                  AND (
                        quantity_reconciled_kg IS NULL
                        OR
                        ABS(
                            quantity_reconciled_kg
                            - export_synchronized_quantity_kg
                        )
                        >
                        (
                            1e-9
                            * GREATEST(
                                ABS(
                                    export_synchronized_quantity_kg
                                ),
                                1.0
                            )
                        )
                  )
                """
            ).fetchone()[0]
        )

        export_only_preserved = (
            export_only_mismatches == 0
        )

        # --------------------------------------------------------------
        # Import-only preservation
        # --------------------------------------------------------------

        import_only_mismatches = int(
            con.execute(
                f"""
                SELECT COUNT(*)

                FROM read_parquet({quantity_sql})

                WHERE quantity_availability_case
                        = 'import_only'

                  AND (
                        quantity_reconciled_kg IS NULL
                        OR
                        ABS(
                            quantity_reconciled_kg
                            - import_synchronized_quantity_kg
                        )
                        >
                        (
                            1e-9
                            * GREATEST(
                                ABS(
                                    import_synchronized_quantity_kg
                                ),
                                1.0
                            )
                        )
                  )
                """
            ).fetchone()[0]
        )

        import_only_preserved = (
            import_only_mismatches == 0
        )

        # --------------------------------------------------------------
        # Both-quantity candidate integrity
        # --------------------------------------------------------------

        both_quantity_candidate_mismatches = int(
            con.execute(
                f"""
                SELECT COUNT(*)

                FROM read_parquet({quantity_sql})

                WHERE quantity_availability_case
                        = 'both_quantity'

                  AND (
                        quantity_simple_mean_kg
                            IS NULL

                        OR quantity_relative_difference
                            IS NULL

                        OR NOT isfinite(
                            quantity_simple_mean_kg
                        )

                        OR quantity_simple_mean_kg <= 0

                        OR NOT isfinite(
                            quantity_relative_difference
                        )

                        OR quantity_relative_difference < 0

                        OR quantity_relative_difference > 1
                  )
                """
            ).fetchone()[0]
        )

        both_quantity_candidates_valid = (
            both_quantity_candidate_mismatches == 0
        )

        # --------------------------------------------------------------
        # Method / status combinations
        # --------------------------------------------------------------

        invalid_method_rows = int(
            con.execute(
                f"""
                SELECT COUNT(*)

                FROM read_parquet({quantity_sql})

                WHERE NOT (

                    (
                        quantity_availability_case
                            = 'both_quantity'

                        AND quantity_reconciliation_method IN (
                            'quantity_reliability_weighted',
                            'bilateral_simple_mean_fallback'
                        )
                    )

                    OR

                    (
                        quantity_availability_case
                            = 'export_only'

                        AND quantity_reconciliation_method
                            = 'export_quantity_retained'
                    )

                    OR

                    (
                        quantity_availability_case
                            = 'import_only'

                        AND quantity_reconciliation_method
                            = 'import_quantity_retained'
                    )

                    OR

                    (
                        quantity_availability_case
                            = 'no_quantity'

                        AND quantity_reconciliation_method
                            = 'no_usable_quantity'
                    )
                )
                """
            ).fetchone()[0]
        )

        reconciliation_methods_valid = (
            invalid_method_rows == 0
        )

        # --------------------------------------------------------------
        # Method arithmetic
        # --------------------------------------------------------------

        weighted_method_mismatches = int(
            con.execute(
                f"""
                SELECT COUNT(*)

                FROM read_parquet({quantity_sql})

                WHERE quantity_reconciliation_method
                        = 'quantity_reliability_weighted'

                  AND (
                        quantity_reliability_weighted_kg
                            IS NULL

                        OR quantity_reconciled_kg
                            IS NULL

                        OR ABS(
                            quantity_reconciled_kg
                            - quantity_reliability_weighted_kg
                        )
                        >
                        (
                            1e-9
                            * GREATEST(
                                ABS(
                                    quantity_reliability_weighted_kg
                                ),
                                1.0
                            )
                        )
                  )
                """
            ).fetchone()[0]
        )

        mean_fallback_mismatches = int(
            con.execute(
                f"""
                SELECT COUNT(*)

                FROM read_parquet({quantity_sql})

                WHERE quantity_reconciliation_method
                        = 'bilateral_simple_mean_fallback'

                  AND (
                        quantity_simple_mean_kg
                            IS NULL

                        OR quantity_reconciled_kg
                            IS NULL

                        OR ABS(
                            quantity_reconciled_kg
                            - quantity_simple_mean_kg
                        )
                        >
                        (
                            1e-9
                            * GREATEST(
                                ABS(
                                    quantity_simple_mean_kg
                                ),
                                1.0
                            )
                        )
                  )
                """
            ).fetchone()[0]
        )

        reconciliation_arithmetic_valid = (
            weighted_method_mismatches == 0
            and mean_fallback_mismatches == 0
        )

        # --------------------------------------------------------------
        # Existing reconciled trade value must remain unchanged
        # --------------------------------------------------------------

        trade_value_mismatches = int(
            con.execute(
                f"""
                SELECT COUNT(*)

                FROM read_parquet({input_sql}) AS original

                INNER JOIN
                    read_parquet({quantity_sql}) AS quantity

                  USING (
                      year,
                      exporter_iso,
                      importer_iso,
                      target_cmd
                  )

                WHERE
                    (
                        original.trade_value_reconciled
                            IS NULL
                        AND
                        quantity.trade_value_reconciled
                            IS NOT NULL
                    )

                    OR

                    (
                        original.trade_value_reconciled
                            IS NOT NULL
                        AND
                        quantity.trade_value_reconciled
                            IS NULL
                    )

                    OR

                    (
                        original.trade_value_reconciled
                            IS NOT NULL
                        AND
                        quantity.trade_value_reconciled
                            IS NOT NULL

                        AND ABS(
                            original.trade_value_reconciled
                            - quantity.trade_value_reconciled
                        )
                        >
                        (
                            1e-9
                            * GREATEST(
                                ABS(
                                    original.trade_value_reconciled
                                ),
                                1.0
                            )
                        )
                    )
                """
            ).fetchone()[0]
        )

        trade_value_preserved = (
            trade_value_mismatches == 0
        )

        # --------------------------------------------------------------
        # Join coverage against audited reconciled-trade input
        # --------------------------------------------------------------

        unmatched_quantity_rows = int(
            con.execute(
                f"""
                SELECT COUNT(*)

                FROM read_parquet({quantity_sql}) AS quantity

                ANTI JOIN
                    read_parquet({input_sql}) AS original

                  ON original.year
                     = quantity.year

                 AND original.exporter_iso
                     = quantity.exporter_iso

                 AND original.importer_iso
                     = quantity.importer_iso

                 AND original.target_cmd
                     = quantity.target_cmd
                """
            ).fetchone()[0]
        )

        unmatched_input_rows = int(
            con.execute(
                f"""
                SELECT COUNT(*)

                FROM read_parquet({input_sql}) AS original

                ANTI JOIN
                    read_parquet({quantity_sql}) AS quantity

                  ON quantity.year
                     = original.year

                 AND quantity.exporter_iso
                     = original.exporter_iso

                 AND quantity.importer_iso
                     = original.importer_iso

                 AND quantity.target_cmd
                     = original.target_cmd
                """
            ).fetchone()[0]
        )

        bilateral_population_preserved = (
            unmatched_quantity_rows == 0
            and unmatched_input_rows == 0
        )

        # --------------------------------------------------------------
        # Final audit decision
        # --------------------------------------------------------------

        audit_passed = all(
            (
                row_count_preserved,
                input_bilateral_key_unique,
                bilateral_key_unique,
                availability_partition_valid,
                availability_flags_valid,
                reconciled_quantities_valid,
                expected_quantities_present,
                no_quantity_rows_valid,
                export_only_preserved,
                import_only_preserved,
                both_quantity_candidates_valid,
                reconciliation_methods_valid,
                reconciliation_arithmetic_valid,
                trade_value_preserved,
                bilateral_population_preserved,
            )
        )

        # --------------------------------------------------------------
        # Persist one-row audit product
        # --------------------------------------------------------------

        con.execute(
            f"""
            COPY (
                SELECT

                    {sql_literal(run_id)}
                        AS run_id,

                    {input_row_count}
                        AS input_row_count,

                    {quantity_row_count}
                        AS quantity_row_count,

                    {str(row_count_preserved).upper()}
                        AS row_count_preserved,

                    {input_duplicate_key_rows}
                        AS input_duplicate_key_rows,

                    {str(input_bilateral_key_unique).upper()}
                        AS input_bilateral_key_unique,

                    {duplicate_key_rows}
                        AS duplicate_key_rows,

                    {str(bilateral_key_unique).upper()}
                        AS bilateral_key_unique,

                    {n_both_quantity}
                        AS n_both_quantity,

                    {n_export_only}
                        AS n_export_only,

                    {n_import_only}
                        AS n_import_only,

                    {n_no_quantity}
                        AS n_no_quantity,

                    {n_invalid_availability_case}
                        AS n_invalid_availability_case,

                    {str(availability_partition_valid).upper()}
                        AS availability_partition_valid,

                    {availability_flag_mismatches}
                        AS availability_flag_mismatches,

                    {str(availability_flags_valid).upper()}
                        AS availability_flags_valid,

                    {invalid_reconciled_quantity_rows}
                        AS invalid_reconciled_quantity_rows,

                    {str(reconciled_quantities_valid).upper()}
                        AS reconciled_quantities_valid,

                    {missing_expected_reconciled_quantity}
                        AS missing_expected_reconciled_quantity,

                    {str(expected_quantities_present).upper()}
                        AS expected_quantities_present,

                    {no_quantity_nonmissing_rows}
                        AS no_quantity_nonmissing_rows,

                    {str(no_quantity_rows_valid).upper()}
                        AS no_quantity_rows_valid,

                    {export_only_mismatches}
                        AS export_only_mismatches,

                    {str(export_only_preserved).upper()}
                        AS export_only_preserved,

                    {import_only_mismatches}
                        AS import_only_mismatches,

                    {str(import_only_preserved).upper()}
                        AS import_only_preserved,

                    {both_quantity_candidate_mismatches}
                        AS both_quantity_candidate_mismatches,

                    {str(both_quantity_candidates_valid).upper()}
                        AS both_quantity_candidates_valid,

                    {invalid_method_rows}
                        AS invalid_method_rows,

                    {str(reconciliation_methods_valid).upper()}
                        AS reconciliation_methods_valid,

                    {weighted_method_mismatches}
                        AS weighted_method_mismatches,

                    {mean_fallback_mismatches}
                        AS mean_fallback_mismatches,

                    {str(reconciliation_arithmetic_valid).upper()}
                        AS reconciliation_arithmetic_valid,

                    {trade_value_mismatches}
                        AS trade_value_mismatches,

                    {str(trade_value_preserved).upper()}
                        AS trade_value_preserved,

                    {unmatched_quantity_rows}
                        AS unmatched_quantity_rows,

                    {unmatched_input_rows}
                        AS unmatched_input_rows,

                    {str(bilateral_population_preserved).upper()}
                        AS bilateral_population_preserved,

                    {str(audit_passed).upper()}
                        AS audit_passed
            )
            TO {audit_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        if not _valid_output(audit_path):
            raise RuntimeError(
                "Quantity reconciliation audit did not produce "
                f"a usable artifact: {audit_path}"
            )

        logger.info(
            "  Input rows: %s",
            f"{input_row_count:,}",
        )
        logger.info(
            "  Quantity rows: %s",
            f"{quantity_row_count:,}",
        )
        logger.info(
            "  Both quantities: %s",
            f"{n_both_quantity:,}",
        )
        logger.info(
            "  Export only: %s",
            f"{n_export_only:,}",
        )
        logger.info(
            "  Import only: %s",
            f"{n_import_only:,}",
        )
        logger.info(
            "  No usable quantity: %s",
            f"{n_no_quantity:,}",
        )
        logger.info(
            "  Input duplicate bilateral keys: %s",
            f"{input_duplicate_key_rows:,}",
        )
        logger.info(
            "  Output duplicate bilateral keys: %s",
            f"{duplicate_key_rows:,}",
        )
        logger.info(
            "  Availability-flag mismatches: %s",
            f"{availability_flag_mismatches:,}",
        )
        logger.info(
            "  Invalid reconciled quantities: %s",
            f"{invalid_reconciled_quantity_rows:,}",
        )
        logger.info(
            "  Missing expected reconciled quantities: %s",
            f"{missing_expected_reconciled_quantity:,}",
        )
        logger.info(
            "  Method/arithmetic mismatches: %s",
            f"{invalid_method_rows + weighted_method_mismatches + mean_fallback_mismatches:,}",
        )
        logger.info(
            "  Trade-value mismatches: %s",
            f"{trade_value_mismatches:,}",
        )
        logger.info(
            "  Unmatched quantity/input rows: %s / %s",
            f"{unmatched_quantity_rows:,}",
            f"{unmatched_input_rows:,}",
        )
        logger.info(
            "✍ WRITE — Quantity reconciliation audit: %s",
            audit_path,
        )

        if not audit_passed:
            failed_checks = []

            checks = {
                "row_count_preserved": row_count_preserved,
                "input_bilateral_key_unique": (
                    input_bilateral_key_unique
                ),
                "bilateral_key_unique": bilateral_key_unique,
                "availability_partition_valid": (
                    availability_partition_valid
                ),
                "availability_flags_valid": (
                    availability_flags_valid
                ),
                "reconciled_quantities_valid": (
                    reconciled_quantities_valid
                ),
                "expected_quantities_present": (
                    expected_quantities_present
                ),
                "no_quantity_rows_valid": (
                    no_quantity_rows_valid
                ),
                "export_only_preserved": (
                    export_only_preserved
                ),
                "import_only_preserved": (
                    import_only_preserved
                ),
                "both_quantity_candidates_valid": (
                    both_quantity_candidates_valid
                ),
                "reconciliation_methods_valid": (
                    reconciliation_methods_valid
                ),
                "reconciliation_arithmetic_valid": (
                    reconciliation_arithmetic_valid
                ),
                "trade_value_preserved": (
                    trade_value_preserved
                ),
                "bilateral_population_preserved": (
                    bilateral_population_preserved
                ),
            }

            failed_checks = [
                name
                for name, passed in checks.items()
                if not passed
            ]

            raise RuntimeError(
                "Trade Sync quantity reconciliation audit failed: "
                + ", ".join(failed_checks)
            )

        logger.info(
            "✅ AUDIT — Trade Sync quantity reconciliation passed."
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "quantity_reconciliation_audit_path": audit_path,
            "reconciliation_audit_quantity_path": audit_path,
            "audit_passed": True,
            "input_row_count": input_row_count,
            "quantity_row_count": quantity_row_count,
            "input_duplicate_key_rows": input_duplicate_key_rows,
            "duplicate_key_rows": duplicate_key_rows,
            "n_both_quantity": n_both_quantity,
            "n_export_only": n_export_only,
            "n_import_only": n_import_only,
            "n_no_quantity": n_no_quantity,
        }

    finally:
        con.close()