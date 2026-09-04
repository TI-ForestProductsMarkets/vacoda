"""
vacoda.steps.trade_sync.reconcile_trade

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

This module owns reconciliation only.

Validation of the resulting reconciled-value product is performed by
``value_audit.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.steps.trade_sync import TradeSyncConfig
from vacoda.pipeline.checkpoint import (
    file_fingerprint,
    semantic_config_fingerprint,
    validate_local_checkpoint,
    write_local_checkpoint,
)
from vacoda.utils.sql import sql_literal


IMPORT_WEIGHT_BONUS = 0.10


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
    """Validate the minimum input schema required by reconciliation."""

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
# Main reconciliation
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
    Build the reliability-weighted harmonized Trade Sync value dataset.

    This function creates the reconciled value product only.

    The resulting product must subsequently pass
    ``audit_trade_reconciliation`` before downstream use.
    """

    prepared_input_path = Path(
        prepared_input_path
    )
    reliability_evidence_path = Path(
        reliability_evidence_path
    )
    country_reliability_path = Path(
        country_reliability_path
    )

    output_path = cfg.product_path(
        "reconciled_trade"
    )

    # ------------------------------------------------------------------
    # Required inputs
    # ------------------------------------------------------------------

    for label, path in (
        (
            "prepared Trade Sync input",
            prepared_input_path,
        ),
        (
            "reliability evidence",
            reliability_evidence_path,
        ),
        (
            "country reliability",
            country_reliability_path,
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label.capitalize()} not found: {path}"
            )

    # ------------------------------------------------------------------
    # Local checkpoint
    # ------------------------------------------------------------------

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

    if (
        not force
        and reusable
        and _valid_output(output_path)
    ):
        logger.info(
            "🔄 REUSE — Reconciled Trade Sync output: %s",
            output_path,
        )

        return {
            "status": "reused",
            "output_path": output_path,
            "reconciled_trade_path": output_path,
        }

    logger.info(
        "🔨 REBUILD — Reconciled Trade Sync output: %s",
        "forced" if force else reason,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared_sql = sql_literal(
        prepared_input_path
    )
    evidence_sql = sql_literal(
        reliability_evidence_path
    )
    reliability_sql = sql_literal(
        country_reliability_path
    )
    output_sql = sql_literal(
        output_path
    )

    retain_unmatched = bool(
        cfg.method.reconciliation.retain_unmatched_flows
    )

    logger.info(
        "  Prepared input: %s",
        prepared_input_path,
    )
    logger.info(
        "  Reliability evidence: %s",
        reliability_evidence_path,
    )
    logger.info(
        "  Country reliability: %s",
        country_reliability_path,
    )
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

        required_evidence_columns = {
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",
            "reliability_evidence_case",
            "reliability_deviation",
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

        _require_columns(
            con=con,
            parquet_path=prepared_input_path,
            required_columns=required_prepared_columns,
            label="Prepared Trade Sync input",
        )

        _require_columns(
            con=con,
            parquet_path=reliability_evidence_path,
            required_columns=required_evidence_columns,
            label="Trade Sync reliability evidence",
        )

        _require_columns(
            con=con,
            parquet_path=country_reliability_path,
            required_columns=required_reliability_columns,
            label="Country-reliability input",
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
                    -- ----------------------------------------------
                    -- Bilateral evidence
                    -- ----------------------------------------------
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

                    -- ----------------------------------------------
                    -- Exporter-only report
                    -- ----------------------------------------------
                    WHEN {str(retain_unmatched).upper()}
                     AND trade_sync_has_export_fob
                     AND NOT trade_sync_has_import_fob_equivalent
                     AND reliability_evidence_case IS NULL
                        THEN export_value_fob

                    -- ----------------------------------------------
                    -- Importer-only report
                    -- ----------------------------------------------
                    WHEN {str(retain_unmatched).upper()}
                     AND trade_sync_has_import_fob_equivalent
                     AND NOT trade_sync_has_export_fob
                     AND reliability_evidence_case IS NULL
                        THEN import_value_fob_corrected

                    ELSE NULL
                END AS trade_value_reconciled,

                CASE
                    WHEN reliability_evidence_case
                         = 'both_positive'
                        THEN 'bilateral_reconciled'

                    WHEN reliability_evidence_case
                         = 'export_positive_import_implicit_zero'
                        THEN (
                            'bilateral_reconciled_'
                            'import_implicit_zero'
                        )

                    WHEN reliability_evidence_case
                         = 'import_positive_export_implicit_zero'
                        THEN (
                            'bilateral_reconciled_'
                            'export_implicit_zero'
                        )

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

        if not _valid_output(output_path):
            raise RuntimeError(
                "Trade Sync reconciliation did not produce "
                f"a usable output: {output_path}"
            )

        # ==============================================================
        # CHECKPOINT
        # ==============================================================

        write_local_checkpoint(
            checkpoint_path=checkpoint_path,
            step_name="reconciled_trade",
            run_id=run_id,
            config=config_fp,
            input_fingerprint=input_fp,
            output_path=output_path,
        )

        logger.info(
            "✍ WRITE — Reconciled Trade Sync output: %s",
            output_path,
        )

        return {
            "status": "completed",
            "output_path": output_path,
            "reconciled_trade_path": output_path,
        }

    finally:
        con.close()