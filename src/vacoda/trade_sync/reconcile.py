"""
vacoda.trade_sync.reconcile

Reconcile exporter FOB and importer FOB-equivalent reports using annual
country-direction reliability scores.

The implementation follows Bremer, Perger, and Schumacher:

1. country reliability is represented by mean bilateral deviation;
2. lower deviation means greater reliability;
3. each report receives a weight based on the partner report's mean
   deviation;
4. 0.10 weight is transferred from the exporter report to the importer
   report;
5. genuinely missing reports are not interpreted as zero;
6. implicit-zero reports participate in weighted reconciliation;
7. unmatched usable reports are retained when configured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.trade_sync import TradeSyncConfig
from vacoda.utils.display import get_symbols
from vacoda.pipeline.checkpoint import validate_local_checkpoint, file_fingerprint, write_local_checkpoint, semantic_config_fingerprint


IMPORT_WEIGHT_BONUS = 0.10


# ======================================================================
# Small helpers
# ======================================================================


def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def valid_output(path: Path) -> bool:
    """Return True when an existing output appears usable."""
    return path.is_file() and path.stat().st_size > 0


# ======================================================================
# Main reconciliation function
# ======================================================================


def reconcile_trade(
    *,
    cfg: TradeSyncConfig,
    prepared_input_path: Path,
    reliability_evidence_path: Path,
    country_reliability_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Build the reliability-weighted harmonized Trade Sync dataset.

    The full prepared bilateral population is retained when
    retain_unmatched_flows is enabled.
    """

    prepared_input_path = Path(prepared_input_path)
    reliability_evidence_path = Path(reliability_evidence_path)
    country_reliability_path = Path(country_reliability_path)

    output_path = cfg.product_path("reconciled_trade")
    audit_path = cfg.product_path("reconciliation_audit")

    for label, path in (
        ("prepared Trade Sync input", prepared_input_path),
        ("reliability evidence", reliability_evidence_path),
        ("country reliability", country_reliability_path),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{label.capitalize()} not found: {path}"
            )

    checkpoint_path = (
        cfg.output_dir()
        / "checkpoints"
        / "reconciled_trade.json"
    )

    config_fp = semantic_config_fingerprint(
        config=cfg,
        excluded_paths=(
            ("engine", "threads"),
        ),
    )

    input_fp = {
        "prepared_input": file_fingerprint(
            prepared_input_path,
            hash_file=True,
        ),
        "reliability_evidence": file_fingerprint(
            reliability_evidence_path,
            hash_file=True,
        ),
        "country_reliability": file_fingerprint(
            country_reliability_path,
            hash_file=True,
        ),
    }

    reusable, reason = validate_local_checkpoint(
        checkpoint_path=checkpoint_path,
        current_config=config_fp,
        current_input=input_fp,
        output_path=output_path,
    )

    if not force and reusable and valid_output(audit_path):
        logger.info(
            "REUSE — Reconciled Trade Sync local checkpoint matches: %s",
            output_path,
        )

        return {
            "status": "reused",
            "output_path": output_path,
            "reconciled_trade_path": output_path,
            "audit_path": audit_path,
            "reconciliation_audit_path": audit_path,
            "audit_passed": True,
            "ready_for_diagnostics": True,
            "ready_for_processed_export": True,
        }

    logger.info(
        "REBUILD — Reconciled Trade Sync output: %s",
        "forced" if force else reason,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared_sql = _sql_literal(prepared_input_path)
    evidence_sql = _sql_literal(reliability_evidence_path)
    reliability_sql = _sql_literal(country_reliability_path)
    output_sql = _sql_literal(output_path)
    audit_sql = _sql_literal(audit_path)

    retain_unmatched = bool(
        cfg.method.reconciliation.retain_unmatched_flows
    )

    info = get_symbols("info")
    ok = get_symbols("check")

    logger.info(
        "%s Reconciling bilateral trade using country reliability.",
        info,
    )
    logger.info("  Prepared input: %s", prepared_input_path)
    logger.info("  Reliability evidence: %s", reliability_evidence_path)
    logger.info("  Country reliability: %s", country_reliability_path)
    logger.info("  Output: %s", output_path)
    logger.info(
        "  Import weight bonus: %.2f",
        IMPORT_WEIGHT_BONUS,
    )
    logger.info(
        "  Retain unmatched usable reports: %s",
        retain_unmatched,
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        # ==============================================================
        # INPUT SCHEMA VALIDATION
        # ==============================================================

        prepared_columns = {
            str(row[0])
            for row in con.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_parquet({prepared_sql})
                """
            ).fetchall()
        }

        required_prepared_columns = {
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",
            "export_value_fob",
            "import_value_cif",
            "import_value_fob_corrected",
            "trade_sync_has_export_fob",
            "trade_sync_has_import_fob_equivalent",
            "trade_sync_availability",
        }

        missing_prepared_columns = sorted(
            required_prepared_columns - prepared_columns
        )

        if missing_prepared_columns:
            raise RuntimeError(
                "Prepared Trade Sync input is missing required columns: "
                + ", ".join(missing_prepared_columns)
            )

        reliability_columns = {
            str(row[0])
            for row in con.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_parquet({reliability_sql})
                """
            ).fetchall()
        }

        required_reliability_columns = {
            "year",
            "country_iso",
            "reporting_direction",
            "n_evidence_observations",
            "mean_reliability_deviation",
            "reliability_index",
            "implicit_zero_share",
        }

        missing_reliability_columns = sorted(
            required_reliability_columns - reliability_columns
        )

        if missing_reliability_columns:
            raise RuntimeError(
                "Country-reliability input is missing required columns: "
                + ", ".join(missing_reliability_columns)
            )

        # ==============================================================
        # RECONCILIATION QUERY
        # ==============================================================

        reconciliation_query = f"""
        WITH prepared AS (
            SELECT *
            FROM read_parquet({prepared_sql})
        ),

        evidence AS (
            SELECT
                year,
                exporter_iso,
                importer_iso,
                target_cmd,
                reliability_evidence_case,
                reliability_deviation

            FROM read_parquet({evidence_sql})
        ),

        export_scores AS (
            SELECT
                year,
                country_iso,
                n_evidence_observations
                    AS exporter_reliability_n,
                mean_reliability_deviation
                    AS exporter_mean_deviation,
                reliability_index
                    AS exporter_reliability_index,
                implicit_zero_share
                    AS exporter_implicit_zero_share

            FROM read_parquet({reliability_sql})

            WHERE reporting_direction = 'export'
        ),

        import_scores AS (
            SELECT
                year,
                country_iso,
                n_evidence_observations
                    AS importer_reliability_n,
                mean_reliability_deviation
                    AS importer_mean_deviation,
                reliability_index
                    AS importer_reliability_index,
                implicit_zero_share
                    AS importer_implicit_zero_share

            FROM read_parquet({reliability_sql})

            WHERE reporting_direction = 'import'
        ),

        joined AS (
            SELECT
                prepared.*,

                evidence.reliability_evidence_case,
                evidence.reliability_deviation,

                export_scores.exporter_reliability_n,
                export_scores.exporter_mean_deviation,
                export_scores.exporter_reliability_index,
                export_scores.exporter_implicit_zero_share,

                import_scores.importer_reliability_n,
                import_scores.importer_mean_deviation,
                import_scores.importer_reliability_index,
                import_scores.importer_implicit_zero_share

            FROM prepared

            LEFT JOIN evidence
                USING (
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd
                )

            LEFT JOIN export_scores
                ON prepared.year = export_scores.year
               AND prepared.exporter_iso
                   = export_scores.country_iso

            LEFT JOIN import_scores
                ON prepared.year = import_scores.year
               AND prepared.importer_iso
                   = import_scores.country_iso
        ),

        valued AS (
            SELECT
                *,

                CASE
                    WHEN reliability_evidence_case
                         = 'export_positive_import_implicit_zero'
                        THEN 0.0

                    WHEN trade_sync_has_import_fob_equivalent
                        THEN import_value_fob_corrected

                    ELSE NULL
                END AS reconciliation_import_value,

                CASE
                    WHEN reliability_evidence_case
                         = 'import_positive_export_implicit_zero'
                        THEN 0.0

                    WHEN trade_sync_has_export_fob
                        THEN export_value_fob

                    ELSE NULL
                END AS reconciliation_export_value

            FROM joined
        ),

        base_weights AS (
            SELECT
                *,

                CASE
                    WHEN reliability_evidence_case IN (
                        'both_positive',
                        'export_positive_import_implicit_zero',
                        'import_positive_export_implicit_zero'
                    )
                     AND exporter_mean_deviation IS NOT NULL
                     AND importer_mean_deviation IS NOT NULL
                     AND (
                         exporter_mean_deviation
                         + importer_mean_deviation
                     ) > 0
                        THEN (
                            importer_mean_deviation
                            /
                            (
                                exporter_mean_deviation
                                + importer_mean_deviation
                            )
                        )

                    -- Perfect agreement history for both sides gives a
                    -- zero denominator. Use equal base weights before
                    -- applying the import preference.
                    WHEN reliability_evidence_case IN (
                        'both_positive',
                        'export_positive_import_implicit_zero',
                        'import_positive_export_implicit_zero'
                    )
                     AND exporter_mean_deviation = 0
                     AND importer_mean_deviation = 0
                        THEN 0.5

                    ELSE NULL
                END AS exporter_base_weight,

                CASE
                    WHEN reliability_evidence_case IN (
                        'both_positive',
                        'export_positive_import_implicit_zero',
                        'import_positive_export_implicit_zero'
                    )
                     AND exporter_mean_deviation IS NOT NULL
                     AND importer_mean_deviation IS NOT NULL
                     AND (
                         exporter_mean_deviation
                         + importer_mean_deviation
                     ) > 0
                        THEN (
                            exporter_mean_deviation
                            /
                            (
                                exporter_mean_deviation
                                + importer_mean_deviation
                            )
                        )

                    WHEN reliability_evidence_case IN (
                        'both_positive',
                        'export_positive_import_implicit_zero',
                        'import_positive_export_implicit_zero'
                    )
                     AND exporter_mean_deviation = 0
                     AND importer_mean_deviation = 0
                        THEN 0.5

                    ELSE NULL
                END AS importer_base_weight

            FROM valued
        ),

        adjusted_weights AS (
            SELECT
                *,

                CASE
                    WHEN exporter_base_weight IS NOT NULL
                        THEN (
                            exporter_base_weight
                            - {IMPORT_WEIGHT_BONUS}
                        )
                    ELSE NULL
                END AS exporter_adjusted_weight,

                CASE
                    WHEN importer_base_weight IS NOT NULL
                        THEN (
                            importer_base_weight
                            + {IMPORT_WEIGHT_BONUS}
                        )
                    ELSE NULL
                END AS importer_adjusted_weight

            FROM base_weights
        ),

        reconciled AS (
            SELECT
                *,

                CASE
                    -- ------------------------------------------
                    -- Two reports, including implicit-zero cases
                    -- ------------------------------------------
                    WHEN reliability_evidence_case IN (
                        'both_positive',
                        'export_positive_import_implicit_zero',
                        'import_positive_export_implicit_zero'
                    )
                     AND exporter_adjusted_weight IS NOT NULL
                     AND importer_adjusted_weight IS NOT NULL
                        THEN (
                            reconciliation_export_value
                            * exporter_adjusted_weight
                            +
                            reconciliation_import_value
                            * importer_adjusted_weight
                        )

                    -- ------------------------------------------
                    -- One genuinely available exporter report
                    -- ------------------------------------------
                    WHEN {str(retain_unmatched).upper()}
                     AND trade_sync_has_export_fob
                     AND NOT trade_sync_has_import_fob_equivalent
                     AND reliability_evidence_case IS NULL
                        THEN export_value_fob

                    -- ------------------------------------------
                    -- One genuinely available importer report
                    -- ------------------------------------------
                    WHEN {str(retain_unmatched).upper()}
                     AND trade_sync_has_import_fob_equivalent
                     AND NOT trade_sync_has_export_fob
                     AND reliability_evidence_case IS NULL
                        THEN import_value_fob_corrected

                    ELSE NULL
                END AS trade_value_reconciled,

                CASE
                    WHEN reliability_evidence_case = 'both_positive'
                        THEN 'bilateral_reconciled'

                    WHEN reliability_evidence_case
                         = 'export_positive_import_implicit_zero'
                        THEN 'bilateral_reconciled_import_implicit_zero'

                    WHEN reliability_evidence_case
                         = 'import_positive_export_implicit_zero'
                        THEN 'bilateral_reconciled_export_implicit_zero'

                    WHEN trade_sync_has_export_fob
                     AND NOT trade_sync_has_import_fob_equivalent
                     AND reliability_evidence_case IS NULL
                        THEN 'export_only'

                    WHEN trade_sync_has_import_fob_equivalent
                     AND NOT trade_sync_has_export_fob
                     AND reliability_evidence_case IS NULL
                        THEN 'import_only'

                    ELSE 'not_eligible'
                END AS reconciliation_status,

                CASE
                    WHEN reliability_evidence_case IS NOT NULL
                        THEN 'reliability_weighted'

                    WHEN trade_sync_has_export_fob
                     AND NOT trade_sync_has_import_fob_equivalent
                        THEN 'export_report_retained'

                    WHEN trade_sync_has_import_fob_equivalent
                     AND NOT trade_sync_has_export_fob
                        THEN 'import_report_retained'

                    ELSE 'no_usable_fob_basis_value'
                END AS reconciliation_method_applied

            FROM adjusted_weights
        )

        SELECT *
        FROM reconciled
        """

        # ==============================================================
        # PRE-WRITE WEIGHT AUDIT
        #
        # The paper does not specify clipping. We therefore validate the
        # exact adjusted weights before writing the production result.
        # ==============================================================

        weight_audit_row = con.execute(
            f"""
            WITH result AS (
                {reconciliation_query}
            )

            SELECT
                COUNT(*) FILTER (
                    WHERE reliability_evidence_case IS NOT NULL
                ) AS n_weighted_rows,

                COUNT(*) FILTER (
                    WHERE reliability_evidence_case IS NOT NULL
                      AND (
                            exporter_adjusted_weight IS NULL
                         OR importer_adjusted_weight IS NULL
                      )
                ) AS n_missing_weights,

                COUNT(*) FILTER (
                    WHERE reliability_evidence_case IS NOT NULL
                      AND (
                            exporter_adjusted_weight < 0
                         OR exporter_adjusted_weight > 1
                         OR importer_adjusted_weight < 0
                         OR importer_adjusted_weight > 1
                      )
                ) AS n_out_of_bounds_weights,

                COUNT(*) FILTER (
                    WHERE reliability_evidence_case IS NOT NULL
                      AND ABS(
                          exporter_adjusted_weight
                          + importer_adjusted_weight
                          - 1.0
                      ) > 1e-10
                ) AS n_invalid_weight_sums,

                MIN(exporter_adjusted_weight)
                    FILTER (
                        WHERE reliability_evidence_case IS NOT NULL
                    ) AS minimum_exporter_weight,

                MAX(exporter_adjusted_weight)
                    FILTER (
                        WHERE reliability_evidence_case IS NOT NULL
                    ) AS maximum_exporter_weight,

                MIN(importer_adjusted_weight)
                    FILTER (
                        WHERE reliability_evidence_case IS NOT NULL
                    ) AS minimum_importer_weight,

                MAX(importer_adjusted_weight)
                    FILTER (
                        WHERE reliability_evidence_case IS NOT NULL
                    ) AS maximum_importer_weight

            FROM result
            """
        ).fetchone()

        if weight_audit_row is None:
            raise RuntimeError(
                "Could not inspect Trade Sync reconciliation weights."
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

        if int(n_missing_weights) > 0:
            raise RuntimeError(
                "Reliability-weighted reconciliation contains "
                f"{int(n_missing_weights):,} rows without both country "
                "reliability scores."
            )

        if int(n_out_of_bounds_weights) > 0:
            raise RuntimeError(
                "The paper's unbounded 0.10 import-weight adjustment "
                "produced "
                f"{int(n_out_of_bounds_weights):,} rows with weights "
                "outside [0, 1]. Inspect the score distribution before "
                "deciding whether to clip or modify the bonus."
            )

        if int(n_invalid_weight_sums) > 0:
            raise RuntimeError(
                "Reconciliation contains "
                f"{int(n_invalid_weight_sums):,} rows whose adjusted "
                "weights do not sum to one."
            )

        # ==============================================================
        # WRITE RECONCILED OUTPUT
        # ==============================================================

        con.execute(
            f"""
            COPY (
                {reconciliation_query}
            )
            TO {output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        # ==============================================================
        # OUTPUT AUDIT
        # ==============================================================

        audit_query = f"""
        WITH output AS (
            SELECT *
            FROM read_parquet({output_sql})
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
            COUNT(*) AS n_output_rows,

            COUNT(*) FILTER (
                WHERE reconciliation_status
                    = 'bilateral_reconciled'
            ) AS n_bilateral_positive_reconciled,

            COUNT(*) FILTER (
                WHERE reconciliation_status
                    = 'bilateral_reconciled_import_implicit_zero'
            ) AS n_import_implicit_zero_reconciled,

            COUNT(*) FILTER (
                WHERE reconciliation_status
                    = 'bilateral_reconciled_export_implicit_zero'
            ) AS n_export_implicit_zero_reconciled,

            COUNT(*) FILTER (
                WHERE reconciliation_status = 'export_only'
            ) AS n_export_only,

            COUNT(*) FILTER (
                WHERE reconciliation_status = 'import_only'
            ) AS n_import_only,

            COUNT(*) FILTER (
                WHERE reconciliation_status = 'not_eligible'
            ) AS n_not_eligible,

            COUNT(*) FILTER (
                WHERE trade_value_reconciled IS NOT NULL
                  AND (
                        NOT isfinite(trade_value_reconciled)
                     OR trade_value_reconciled < 0
                  )
            ) AS n_invalid_reconciled_values,

            COUNT(*) FILTER (
                WHERE reconciliation_status <> 'not_eligible'
                  AND trade_value_reconciled IS NULL
            ) AS n_missing_expected_reconciled_values,

            COUNT(*) FILTER (
                WHERE reconciliation_status = 'not_eligible'
                  AND trade_value_reconciled IS NOT NULL
            ) AS n_unexpected_not_eligible_values,

            (
                SELECT COUNT(*)
                FROM duplicate_keys
            ) AS n_duplicate_keys,

            COUNT(*) FILTER (
                WHERE year IS NULL
                   OR exporter_iso IS NULL
                   OR importer_iso IS NULL
                   OR target_cmd IS NULL
            ) AS n_missing_key_rows,

            SUM(trade_value_reconciled)
                FILTER (
                    WHERE trade_value_reconciled IS NOT NULL
                ) AS total_reconciled_value,

            MEDIAN(trade_value_reconciled)
                FILTER (
                    WHERE trade_value_reconciled IS NOT NULL
                ) AS median_reconciled_value

        FROM output
        """

        audit_row = con.execute(
            audit_query
        ).fetchone()

        if audit_row is None:
            raise RuntimeError(
                "Could not inspect reconciled Trade Sync output."
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
            n_duplicate_keys,
            n_missing_key_rows,
            total_reconciled_value,
            median_reconciled_value,
        ) = audit_row

        n_prepared_rows = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet({prepared_sql})
            """
        ).fetchone()[0]

        status_total = (
            int(n_bilateral_positive_reconciled)
            + int(n_import_implicit_zero_reconciled)
            + int(n_export_implicit_zero_reconciled)
            + int(n_export_only)
            + int(n_import_only)
            + int(n_not_eligible)
        )

        if int(n_output_rows) != int(n_prepared_rows):
            raise RuntimeError(
                "Reconciliation did not preserve the full prepared "
                "population: "
                f"prepared={int(n_prepared_rows):,}, "
                f"output={int(n_output_rows):,}."
            )

        if status_total != int(n_output_rows):
            raise RuntimeError(
                "Reconciliation statuses do not partition the full "
                "output population."
            )

        if int(n_duplicate_keys) > 0:
            raise RuntimeError(
                "Reconciled Trade Sync output contains "
                f"{int(n_duplicate_keys):,} duplicate bilateral keys."
            )

        if int(n_missing_key_rows) > 0:
            raise RuntimeError(
                "Reconciled Trade Sync output contains "
                f"{int(n_missing_key_rows):,} incomplete bilateral keys."
            )

        if int(n_invalid_reconciled_values) > 0:
            raise RuntimeError(
                "Reconciled Trade Sync output contains "
                f"{int(n_invalid_reconciled_values):,} invalid values."
            )

        if int(n_missing_expected_reconciled_values) > 0:
            raise RuntimeError(
                "Reconciled Trade Sync output contains "
                f"{int(n_missing_expected_reconciled_values):,} eligible "
                "rows without reconciled values."
            )

        if int(n_unexpected_not_eligible_values) > 0:
            raise RuntimeError(
                "Reconciled Trade Sync output contains "
                f"{int(n_unexpected_not_eligible_values):,} not-eligible "
                "rows with reconciled values."
            )

        # ==============================================================
        # WRITE AUDIT PRODUCT
        # ==============================================================

        con.execute(
            f"""
            COPY (
                {audit_query}
            )
            TO {audit_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        audit_passed = True
        ready_for_diagnostics = True
        ready_for_processed_export = True

        write_local_checkpoint(
            checkpoint_path=checkpoint_path,
            step_name="reconciled_trade",
            run_id=run_id,
            config=config_fp,
            input_fingerprint=input_fp,
            output_path=output_path,
        )


        logger.info(
            "%s Reliability-weighted trade reconciliation created.",
            ok,
        )
        logger.info(
            "  Prepared/output rows: %s",
            f"{int(n_output_rows):,}",
        )
        logger.info(
            "  Weighted evidence rows: %s",
            f"{int(n_weighted_rows):,}",
        )
        logger.info(
            "  Positive-positive reconciled: %s",
            f"{int(n_bilateral_positive_reconciled):,}",
        )
        logger.info(
            "  Import implicit-zero reconciled: %s",
            f"{int(n_import_implicit_zero_reconciled):,}",
        )
        logger.info(
            "  Export implicit-zero reconciled: %s",
            f"{int(n_export_implicit_zero_reconciled):,}",
        )
        logger.info(
            "  Export-only retained: %s",
            f"{int(n_export_only):,}",
        )
        logger.info(
            "  Import-only retained: %s",
            f"{int(n_import_only):,}",
        )
        logger.info(
            "  Not eligible: %s",
            f"{int(n_not_eligible):,}",
        )
        logger.info(
            "  Adjusted exporter-weight range: %.6f–%.6f",
            float(minimum_exporter_weight),
            float(maximum_exporter_weight),
        )
        logger.info(
            "  Adjusted importer-weight range: %.6f–%.6f",
            float(minimum_importer_weight),
            float(maximum_importer_weight),
        )
        logger.info(
            "  Total reconciled value: %.6f",
            float(total_reconciled_value or 0.0),
        )
        logger.info(
            "  Median reconciled value: %.6f",
            float(median_reconciled_value or 0.0),
        )
        logger.info("  Output: %s", output_path)
        logger.info("  Audit: %s", audit_path)
        logger.info("  Step audit passed: %s", audit_passed)

        return {
            "status": "completed",
            "output_path": output_path,
            "reconciled_trade_path": output_path,
            "audit_path": audit_path,
            "reconciliation_audit_path": audit_path,
            "audit_passed": audit_passed,
            "ready_for_diagnostics": ready_for_diagnostics,
            "ready_for_processed_export": ready_for_processed_export,
            "n_output_rows": int(n_output_rows),
            "n_weighted_rows": int(n_weighted_rows),
            "n_bilateral_positive_reconciled": int(
                n_bilateral_positive_reconciled
            ),
            "n_import_implicit_zero_reconciled": int(
                n_import_implicit_zero_reconciled
            ),
            "n_export_implicit_zero_reconciled": int(
                n_export_implicit_zero_reconciled
            ),
            "n_export_only": int(n_export_only),
            "n_import_only": int(n_import_only),
            "n_not_eligible": int(n_not_eligible),
            "minimum_exporter_weight": float(
                minimum_exporter_weight
            ),
            "maximum_exporter_weight": float(
                maximum_exporter_weight
            ),
            "minimum_importer_weight": float(
                minimum_importer_weight
            ),
            "maximum_importer_weight": float(
                maximum_importer_weight
            ),
            "total_reconciled_value": float(
                total_reconciled_value or 0.0
            ),
            "median_reconciled_value": float(
                median_reconciled_value or 0.0
            ),
        }

    finally:
        con.close()