"""
vacoda.steps.trade_sync.audit_trade

Integrity audit for the Trade Sync reconciled-value product.

This stage runs after bilateral trade-value reconciliation.

It validates:

1. reliability-weight completeness and bounds;
2. adjusted weight sums;
3. preservation of the prepared bilateral population;
4. reconciliation-status partitioning;
5. bilateral-key uniqueness and completeness;
6. validity of reconciled trade values;
7. expected versus unexpected reconciled values.

The audit writes one Parquet row containing the complete validation
summary and raises immediately if any hard invariant fails.
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

    parquet_sql = sql_literal(
        parquet_path
    )

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


def audit_trade_reconciliation(
    *,
    cfg: TradeSyncConfig,
    prepared_input_path: Path,
    reconciled_trade_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Audit the reconciled Trade Sync value product.

    Parameters
    ----------
    cfg
        Trade Sync configuration.

    prepared_input_path
        Canonical prepared Trade Sync population used by reconciliation.

    reconciled_trade_path
        Reconciled-value Parquet produced by ``reconcile_trade``.

    run_id
        Master pipeline run identifier.

    force
        Rebuild the audit product even when it already exists.

    logger
        Pipeline logger.

    Returns
    -------
    dict
        Audit metadata and audit artifact path.

    Notes
    -----
    This function raises RuntimeError when any hard audit invariant fails.
    Therefore successful return implies ``audit_passed=True``.
    """

    prepared_input_path = Path(
        prepared_input_path
    )
    reconciled_trade_path = Path(
        reconciled_trade_path
    )

    if not prepared_input_path.is_file():
        raise FileNotFoundError(
            "Prepared Trade Sync input not found: "
            f"{prepared_input_path}"
        )

    if not reconciled_trade_path.is_file():
        raise FileNotFoundError(
            "Reconciled Trade Sync output not found: "
            f"{reconciled_trade_path}"
        )

    audit_path = cfg.product_path(
        "reconciliation_audit"
    )

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    logger.info(
        "🔨 REBUILD — Trade Sync reconciliation audit."
    )

    prepared_sql = sql_literal(
        prepared_input_path
    )
    reconciled_sql = sql_literal(
        reconciled_trade_path
    )
    audit_sql = sql_literal(
        audit_path
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        # ==============================================================
        # INPUT CONTRACT
        # ==============================================================

        required_reconciled_columns = {
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",

            "export_value_fob",
            "import_value_fob_corrected",

            "reconciliation_export_value",
            "reconciliation_import_value",

            "reliability_evidence_case",

            "exporter_adjusted_weight",
            "importer_adjusted_weight",

            "trade_value_reconciled",
            "reconciliation_status",
            "reconciliation_method_applied",
        }

        _require_columns(
            con=con,
            parquet_path=reconciled_trade_path,
            required_columns=required_reconciled_columns,
            label="Reconciled Trade Sync output",
        )

        # ==============================================================
        # WEIGHT AUDIT
        # ==============================================================

        weight_audit_row = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE reliability_evidence_case
                        IS NOT NULL
                ) AS n_weighted_rows,

                COUNT(*) FILTER (
                    WHERE reliability_evidence_case
                        IS NOT NULL
                      AND (
                            exporter_adjusted_weight IS NULL
                         OR importer_adjusted_weight IS NULL
                      )
                ) AS n_missing_weights,

                COUNT(*) FILTER (
                    WHERE reliability_evidence_case
                        IS NOT NULL
                      AND (
                            exporter_adjusted_weight < 0
                         OR exporter_adjusted_weight > 1
                         OR importer_adjusted_weight < 0
                         OR importer_adjusted_weight > 1
                      )
                ) AS n_out_of_bounds_weights,

                COUNT(*) FILTER (
                    WHERE reliability_evidence_case
                        IS NOT NULL
                      AND ABS(
                            exporter_adjusted_weight
                            + importer_adjusted_weight
                            - 1.0
                      ) > 1e-10
                ) AS n_invalid_weight_sums,

                MIN(exporter_adjusted_weight)
                    FILTER (
                        WHERE reliability_evidence_case
                            IS NOT NULL
                    )
                    AS minimum_exporter_weight,

                MAX(exporter_adjusted_weight)
                    FILTER (
                        WHERE reliability_evidence_case
                            IS NOT NULL
                    )
                    AS maximum_exporter_weight,

                MIN(importer_adjusted_weight)
                    FILTER (
                        WHERE reliability_evidence_case
                            IS NOT NULL
                    )
                    AS minimum_importer_weight,

                MAX(importer_adjusted_weight)
                    FILTER (
                        WHERE reliability_evidence_case
                            IS NOT NULL
                    )
                    AS maximum_importer_weight

            FROM read_parquet({reconciled_sql})
            """
        ).fetchone()

        if weight_audit_row is None:
            raise RuntimeError(
                "Could not inspect Trade Sync "
                "reconciliation weights."
            )

        (
            n_weighted_rows,
            n_missing_weights,
            n_out_of_bounds_weights,
            n_invalid_weight_sums,
            minimum_exporter_weight,
            maximum_exporter_weight,
            minimum_importer_weight,
            maximum_importer_weight,
        ) = weight_audit_row

        n_weighted_rows = int(
            n_weighted_rows or 0
        )
        n_missing_weights = int(
            n_missing_weights or 0
        )
        n_out_of_bounds_weights = int(
            n_out_of_bounds_weights or 0
        )
        n_invalid_weight_sums = int(
            n_invalid_weight_sums or 0
        )

        weights_complete = (
            n_missing_weights == 0
        )

        weights_in_bounds = (
            n_out_of_bounds_weights == 0
        )

        weight_sums_valid = (
            n_invalid_weight_sums == 0
        )

        # ==============================================================
        # OUTPUT AUDIT
        # ==============================================================

        output_audit_row = con.execute(
            f"""
            WITH output AS (
                SELECT *
                FROM read_parquet({reconciled_sql})
            ),

            duplicate_keys AS (
                SELECT
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    COUNT(*) AS n_rows

                FROM output

                GROUP BY
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd

                HAVING COUNT(*) > 1
            )

            SELECT
                COUNT(*)
                    AS n_output_rows,

                COUNT(*) FILTER (
                    WHERE reconciliation_status
                        = 'bilateral_reconciled'
                )
                    AS n_bilateral_positive_reconciled,

                COUNT(*) FILTER (
                    WHERE reconciliation_status
                        = (
                            'bilateral_reconciled_'
                            'import_implicit_zero'
                        )
                )
                    AS n_import_implicit_zero_reconciled,

                COUNT(*) FILTER (
                    WHERE reconciliation_status
                        = (
                            'bilateral_reconciled_'
                            'export_implicit_zero'
                        )
                )
                    AS n_export_implicit_zero_reconciled,

                COUNT(*) FILTER (
                    WHERE reconciliation_status
                        = 'export_only'
                )
                    AS n_export_only,

                COUNT(*) FILTER (
                    WHERE reconciliation_status
                        = 'import_only'
                )
                    AS n_import_only,

                COUNT(*) FILTER (
                    WHERE reconciliation_status
                        = 'not_eligible'
                )
                    AS n_not_eligible,

                COUNT(*) FILTER (
                    WHERE trade_value_reconciled IS NOT NULL
                      AND (
                            NOT isfinite(
                                trade_value_reconciled
                            )
                         OR trade_value_reconciled < 0
                      )
                )
                    AS n_invalid_reconciled_values,

                COUNT(*) FILTER (
                    WHERE reconciliation_status IN (
                        'bilateral_reconciled',
                        'bilateral_reconciled_import_implicit_zero',
                        'bilateral_reconciled_export_implicit_zero'
                    )
                    AND (
                            trade_value_reconciled IS NULL
                        OR reconciliation_export_value IS NULL
                        OR reconciliation_import_value IS NULL
                        OR exporter_adjusted_weight IS NULL
                        OR importer_adjusted_weight IS NULL
                        OR ABS(
                                trade_value_reconciled
                                -
                                (
                                    reconciliation_export_value
                                    * exporter_adjusted_weight
                                    +
                                    reconciliation_import_value
                                    * importer_adjusted_weight
                                )
                            )
                            >
                            (
                                1e-9
                                * GREATEST(
                                    ABS(trade_value_reconciled),
                                    1.0
                                )
                            )
                    )
                ) AS n_bilateral_formula_mismatches,

                COUNT(*) FILTER (
                    WHERE reconciliation_status = 'export_only'
                    AND (
                            trade_value_reconciled IS NULL
                        OR export_value_fob IS NULL
                        OR ABS(
                                trade_value_reconciled
                                - export_value_fob
                            )
                            >
                            (
                                1e-9
                                * GREATEST(
                                    ABS(export_value_fob),
                                    1.0
                                )
                            )
                    )
                ) AS n_export_only_formula_mismatches,

                COUNT(*) FILTER (
                    WHERE reconciliation_status = 'import_only'
                    AND (
                            trade_value_reconciled IS NULL
                        OR import_value_fob_corrected IS NULL
                        OR ABS(
                                trade_value_reconciled
                                - import_value_fob_corrected
                            )
                            >
                            (
                                1e-9
                                * GREATEST(
                                    ABS(import_value_fob_corrected),
                                    1.0
                                )
                            )
                    )
                ) AS n_import_only_formula_mismatches,

                COUNT(*) FILTER (
                    WHERE reconciliation_status
                            <> 'not_eligible'
                      AND trade_value_reconciled IS NULL
                )
                    AS n_missing_expected_reconciled_values,

                COUNT(*) FILTER (
                    WHERE reconciliation_status
                            = 'not_eligible'
                      AND trade_value_reconciled IS NOT NULL
                )
                    AS n_unexpected_not_eligible_values,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                )
                    AS n_duplicate_keys,

                COUNT(*) FILTER (
                    WHERE year IS NULL
                       OR exporter_iso IS NULL
                       OR importer_iso IS NULL
                       OR target_cmd IS NULL
                )
                    AS n_missing_key_rows,

                SUM(trade_value_reconciled)
                    FILTER (
                        WHERE trade_value_reconciled
                            IS NOT NULL
                    )
                    AS total_reconciled_value,

                MEDIAN(trade_value_reconciled)
                    FILTER (
                        WHERE trade_value_reconciled
                            IS NOT NULL
                    )
                    AS median_reconciled_value

            FROM output
            """
        ).fetchone()

        if output_audit_row is None:
            raise RuntimeError(
                "Could not inspect reconciled "
                "Trade Sync output."
            )

        (
            n_output_rows,
            n_bilateral_positive_reconciled,
            n_import_implicit_zero_reconciled,
            n_export_implicit_zero_reconciled,
            n_export_only,
            n_import_only,
            n_not_eligible,
            n_invalid_reconciled_values,
            n_missing_expected_reconciled_values,
            n_unexpected_not_eligible_values,
            n_bilateral_formula_mismatches,
            n_export_only_formula_mismatches,
            n_import_only_formula_mismatches,
            n_duplicate_keys,
            n_missing_key_rows,
            total_reconciled_value,
            median_reconciled_value,
            
        ) = output_audit_row

        n_output_rows = int(
            n_output_rows or 0
        )

        n_bilateral_positive_reconciled = int(
            n_bilateral_positive_reconciled or 0
        )

        n_import_implicit_zero_reconciled = int(
            n_import_implicit_zero_reconciled or 0
        )

        n_export_implicit_zero_reconciled = int(
            n_export_implicit_zero_reconciled or 0
        )

        n_export_only = int(
            n_export_only or 0
        )

        n_import_only = int(
            n_import_only or 0
        )

        n_not_eligible = int(
            n_not_eligible or 0
        )

        n_invalid_reconciled_values = int(
            n_invalid_reconciled_values or 0
        )

        n_missing_expected_reconciled_values = int(
            n_missing_expected_reconciled_values or 0
        )

        n_unexpected_not_eligible_values = int(
            n_unexpected_not_eligible_values or 0
        )

        n_bilateral_formula_mismatches = int(
            n_bilateral_formula_mismatches or 0
        )

        n_export_only_formula_mismatches = int(
            n_export_only_formula_mismatches or 0
        )

        n_import_only_formula_mismatches = int(
            n_import_only_formula_mismatches or 0
        )
        n_duplicate_keys = int(
            n_duplicate_keys or 0
        )

        n_missing_key_rows = int(
            n_missing_key_rows or 0
        )

        # ==============================================================
        # PREPARED POPULATION
        # ==============================================================

        n_prepared_rows = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({prepared_sql})
                """
            ).fetchone()[0]
        )

        status_total = (
            n_bilateral_positive_reconciled
            + n_import_implicit_zero_reconciled
            + n_export_implicit_zero_reconciled
            + n_export_only
            + n_import_only
            + n_not_eligible
        )

        # ==============================================================
        # HARD AUDIT CHECKS
        # ==============================================================

        row_count_preserved = (
            n_output_rows
            == n_prepared_rows
        )

        status_partition_valid = (
            status_total
            == n_output_rows
        )

        bilateral_key_unique = (
            n_duplicate_keys == 0
        )

        bilateral_keys_complete = (
            n_missing_key_rows == 0
        )

        reconciled_values_valid = (
            n_invalid_reconciled_values == 0
        )

        expected_values_present = (
            n_missing_expected_reconciled_values == 0
        )

        not_eligible_values_valid = (
            n_unexpected_not_eligible_values == 0
        )

        bilateral_formula_valid = (
            n_bilateral_formula_mismatches == 0
        )

        export_only_formula_valid = (
            n_export_only_formula_mismatches == 0
        )

        import_only_formula_valid = (
            n_import_only_formula_mismatches == 0
        )

        audit_passed = all(
            (
                weights_complete,
                weights_in_bounds,
                weight_sums_valid,
                row_count_preserved,
                status_partition_valid,
                bilateral_key_unique,
                bilateral_keys_complete,
                reconciled_values_valid,
                expected_values_present,
                not_eligible_values_valid,
                bilateral_formula_valid,
                export_only_formula_valid,
                import_only_formula_valid,
            )
        )

        # ==============================================================
        # WRITE AUDIT PRODUCT
        # ==============================================================

        con.execute(
            f"""
            COPY (
                SELECT

                    {sql_literal(run_id)}
                        AS run_id,

                    {n_prepared_rows}
                        AS n_prepared_rows,

                    {n_output_rows}
                        AS n_output_rows,

                    {str(row_count_preserved).upper()}
                        AS row_count_preserved,

                    {n_weighted_rows}
                        AS n_weighted_rows,

                    {n_missing_weights}
                        AS n_missing_weights,

                    {str(weights_complete).upper()}
                        AS weights_complete,

                    {n_out_of_bounds_weights}
                        AS n_out_of_bounds_weights,

                    {str(weights_in_bounds).upper()}
                        AS weights_in_bounds,

                    {n_invalid_weight_sums}
                        AS n_invalid_weight_sums,

                    {str(weight_sums_valid).upper()}
                        AS weight_sums_valid,

                    {
                        "NULL"
                        if minimum_exporter_weight is None
                        else float(minimum_exporter_weight)
                    }
                        AS minimum_exporter_weight,

                    {
                        "NULL"
                        if maximum_exporter_weight is None
                        else float(maximum_exporter_weight)
                    }
                        AS maximum_exporter_weight,

                    {
                        "NULL"
                        if minimum_importer_weight is None
                        else float(minimum_importer_weight)
                    }
                        AS minimum_importer_weight,

                    {
                        "NULL"
                        if maximum_importer_weight is None
                        else float(maximum_importer_weight)
                    }
                        AS maximum_importer_weight,

                    {n_bilateral_positive_reconciled}
                        AS n_bilateral_positive_reconciled,

                    {n_import_implicit_zero_reconciled}
                        AS n_import_implicit_zero_reconciled,

                    {n_export_implicit_zero_reconciled}
                        AS n_export_implicit_zero_reconciled,

                    {n_export_only}
                        AS n_export_only,

                    {n_import_only}
                        AS n_import_only,

                    {n_not_eligible}
                        AS n_not_eligible,

                    {str(status_partition_valid).upper()}
                        AS status_partition_valid,

                    {n_duplicate_keys}
                        AS n_duplicate_keys,

                    {str(bilateral_key_unique).upper()}
                        AS bilateral_key_unique,

                    {n_missing_key_rows}
                        AS n_missing_key_rows,

                    {str(bilateral_keys_complete).upper()}
                        AS bilateral_keys_complete,

                    {n_invalid_reconciled_values}
                        AS n_invalid_reconciled_values,

                    {str(reconciled_values_valid).upper()}
                        AS reconciled_values_valid,

                    {n_missing_expected_reconciled_values}
                        AS n_missing_expected_reconciled_values,

                    {str(expected_values_present).upper()}
                        AS expected_values_present,

                    {n_unexpected_not_eligible_values}
                        AS n_unexpected_not_eligible_values,

                    {str(not_eligible_values_valid).upper()}
                        AS not_eligible_values_valid,

                    {n_bilateral_formula_mismatches}
                        AS n_bilateral_formula_mismatches,

                    {str(bilateral_formula_valid).upper()}
                        AS bilateral_formula_valid,

                    {n_export_only_formula_mismatches}
                        AS n_export_only_formula_mismatches,

                    {str(export_only_formula_valid).upper()}
                        AS export_only_formula_valid,

                    {n_import_only_formula_mismatches}
                        AS n_import_only_formula_mismatches,

                    {str(import_only_formula_valid).upper()}
                        AS import_only_formula_valid,
                    {
                        "NULL"
                        if total_reconciled_value is None
                        else float(total_reconciled_value)
                    }
                        AS total_reconciled_value,

                    {
                        "NULL"
                        if median_reconciled_value is None
                        else float(median_reconciled_value)
                    }
                        AS median_reconciled_value,

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


        # ==============================================================
        # LOG SUMMARY
        # ==============================================================

        logger.info(
            "  Prepared/output rows: %s",
            f"{n_output_rows:,}",
        )
        logger.info(
            "  Weighted evidence rows: %s",
            f"{n_weighted_rows:,}",
        )
        logger.info(
            "  Positive-positive reconciled: %s",
            f"{n_bilateral_positive_reconciled:,}",
        )
        logger.info(
            "  Import implicit-zero reconciled: %s",
            f"{n_import_implicit_zero_reconciled:,}",
        )
        logger.info(
            "  Export implicit-zero reconciled: %s",
            f"{n_export_implicit_zero_reconciled:,}",
        )
        logger.info(
            "  Export-only retained: %s",
            f"{n_export_only:,}",
        )
        logger.info(
            "  Import-only retained: %s",
            f"{n_import_only:,}",
        )
        logger.info(
            "  Bilateral formula mismatches: %s",
            f"{n_bilateral_formula_mismatches:,}",
        )

        logger.info(
            "  Export-only formula mismatches: %s",
            f"{n_export_only_formula_mismatches:,}",
        )

        logger.info(
            "  Import-only formula mismatches: %s",
            f"{n_import_only_formula_mismatches:,}",
        )
        logger.info(
            "  Not eligible: %s",
            f"{n_not_eligible:,}",
        )

        if minimum_exporter_weight is not None:
            logger.info(
                "  Adjusted exporter-weight range: %.6f–%.6f",
                float(minimum_exporter_weight),
                float(maximum_exporter_weight),
            )

        if minimum_importer_weight is not None:
            logger.info(
                "  Adjusted importer-weight range: %.6f–%.6f",
                float(minimum_importer_weight),
                float(maximum_importer_weight),
            )

        logger.info(
            "✍ WRITE — Trade Sync reconciliation audit: %s",
            audit_path,
        )

        # ==============================================================
        # FAILURE
        # ==============================================================

        if not audit_passed:
            checks = {
                "weights_complete": (
                    weights_complete
                ),
                "weights_in_bounds": (
                    weights_in_bounds
                ),
                "weight_sums_valid": (
                    weight_sums_valid
                ),
                "row_count_preserved": (
                    row_count_preserved
                ),
                "status_partition_valid": (
                    status_partition_valid
                ),
                "bilateral_key_unique": (
                    bilateral_key_unique
                ),
                "bilateral_keys_complete": (
                    bilateral_keys_complete
                ),
                "reconciled_values_valid": (
                    reconciled_values_valid
                ),
                "expected_values_present": (
                    expected_values_present
                ),
                "not_eligible_values_valid": (
                    not_eligible_values_valid
                ),
                "bilateral_formula_valid": (
                    bilateral_formula_valid
                ),
                "export_only_formula_valid": (
                    export_only_formula_valid
                ),
                "import_only_formula_valid": (
                    import_only_formula_valid
                ),
            }

            failed_checks = [
                name
                for name, passed in checks.items()
                if not passed
            ]

            raise RuntimeError(
                "Trade Sync reconciliation audit failed: "
                + ", ".join(failed_checks)
            )

        logger.info(
            "✅ AUDIT — Trade Sync value reconciliation passed."
        )

        return {
            "status": "completed",
            "output_path": audit_path,
            "reconciliation_audit_path": audit_path,
            "audit_passed": True,
            "n_output_rows": n_output_rows,
            "n_weighted_rows": n_weighted_rows,
            "n_bilateral_positive_reconciled": (
                n_bilateral_positive_reconciled
            ),
            "n_import_implicit_zero_reconciled": (
                n_import_implicit_zero_reconciled
            ),
            "n_export_implicit_zero_reconciled": (
                n_export_implicit_zero_reconciled
            ),
            "n_export_only": n_export_only,
            "n_import_only": n_import_only,
            "n_not_eligible": n_not_eligible,
            "total_reconciled_value": float(
                total_reconciled_value or 0.0
            ),
            "median_reconciled_value": float(
                median_reconciled_value or 0.0
            ),
        }

    finally:
        con.close()