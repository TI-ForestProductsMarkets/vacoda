"""
vacoda.steps.trade_sync.diagnostics

Build diagnostic summaries for Trade Sync reconciliation.

The diagnostics step does not modify reconciled trade values or quantities. It evaluates:

- reconciliation coverage;
- totals before and after value reconciliation;
- composition by value-reconciliation status;
- value reliability-weight distributions;
- disagreement between exporter and importer value reports;
- effect of value reliability weighting relative to a simple average;
- agreement-tolerance incidence;
- country reliability evidence strength;
- country-year-direction scores dominated by implicit zeros;
- bilateral quantity availability and disagreement;
- quantity-specific reporting reliability;
- quantity-reconciliation method composition and effects;
- implied unit-value plausibility;
- plausibility evidence under extreme quantity disagreement;
- a plausibility-first quantity counterfactual versus the current
  reliability-weighted reconciliation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.steps.trade_sync import TradeSyncConfig
from vacoda.utils.display import get_symbols
from vacoda.pipeline.checkpoint import validate_local_checkpoint, file_fingerprint, write_local_checkpoint, semantic_config_fingerprint
from vacoda.utils.sql import sql_literal

# ======================================================================
# Helpers
# ======================================================================

def valid_output(path: Path) -> bool:
    """Return True when an existing output appears usable."""
    return path.is_file() and path.stat().st_size > 0


def _get_nested_attr(
    obj: Any,
    *names: str,
    default: Any = None,
) -> Any:
    """Safely read a nested attribute path."""
    current = obj

    for name in names:
        if current is None or not hasattr(current, name):
            return default

        current = getattr(current, name)

    return current


def _agreement_tolerance_settings(
    cfg: TradeSyncConfig,
) -> tuple[bool, float | None]:
    """Read the configured reconciliation agreement tolerance."""

    tolerance_cfg = _get_nested_attr(
        cfg,
        "diagnostics",
        "agreement_tolerance",
        default=None,
    )

    if tolerance_cfg is None:
        return False, None

    enabled = bool(
        getattr(tolerance_cfg, "enabled", False)
    )

    relative_difference = getattr(
        tolerance_cfg,
        "relative_difference",
        None,
    )

    if relative_difference is None:
        return enabled, None

    relative_difference = float(relative_difference)

    if not 0 <= relative_difference <= 1:
        raise ValueError(
            "Agreement-tolerance relative difference must lie "
            f"within [0, 1], received {relative_difference}."
        )

    return enabled, relative_difference


# ======================================================================
# Main diagnostics function
# ======================================================================


def build_trade_sync_diagnostics(
    *,
    cfg: TradeSyncConfig,
    reconciled_trade_path: Path,
    country_reliability_path: Path,
    reconciliation_audit_path: Path,
    quantity_reconciliation_audit_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Build Trade Sync reconciliation diagnostics.

    Parameters
    ----------
    cfg:
        Parsed Trade Sync configuration.
    reconciled_trade_path:
        Full reconciled Trade Sync parquet.
    country_reliability_path:
        Country-year-direction reliability parquet.
    reconciliation_audit_path:
        Value-reconciliation audit parquet.
    quantity_reconciliation_audit_path:
        Quantity-reconciliation audit parquet.
    force:
        Rebuild an existing diagnostics product when True.
    logger:
        Configured Trade Sync logger.
    """

    reconciled_trade_path = Path(reconciled_trade_path)
    country_reliability_path = Path(country_reliability_path)
    reconciliation_audit_path = Path(reconciliation_audit_path)
    quantity_reconciliation_audit_path = Path(quantity_reconciliation_audit_path)
    output_path = cfg.product_path("diagnostics")

    for label, path in (
        ("reconciled Trade Sync output", reconciled_trade_path),
        ("country reliability", country_reliability_path),
        ("reconciliation audit", reconciliation_audit_path),
        ("quantity reconciliation audit", quantity_reconciliation_audit_path)
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{label.capitalize()} not found: {path}"
            )

    checkpoint_path = (
        cfg.output_dir()
        / "checkpoints"
        / "diagnostics.json"
    )

    config_fp = semantic_config_fingerprint(
        config=cfg,
        excluded_paths=(
            ("engine", "threads"),
        ),
    )

    input_fp = {
        "reconciled_trade": file_fingerprint(
            reconciled_trade_path,
            hash_file=True,
        ),
        "country_reliability": file_fingerprint(
            country_reliability_path,
            hash_file=True,
        ),
        "reconciliation_audit": file_fingerprint(
            reconciliation_audit_path,
            hash_file=True,
        ),
        "quantity_reconciliation_audit": file_fingerprint(
            quantity_reconciliation_audit_path,
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
            "REUSE — Trade Sync diagnostics local checkpoint matches: %s",
            output_path,
        )

        return {
            "status": "reused",
            "output_path": output_path,
            "diagnostics_path": output_path,
            "audit_passed": True,
            "ready_for_decision_report": True,
        }

    logger.info(
        "REBUILD — Trade Sync diagnostics: %s",
        "forced" if force else reason,
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reconciled_sql = sql_literal(reconciled_trade_path)
    reliability_sql = sql_literal(country_reliability_path)
    reconciliation_audit_sql = sql_literal(reconciliation_audit_path)
    quantity_reconciliation_audit_sql = sql_literal(quantity_reconciliation_audit_path)
    output_sql = sql_literal(output_path)
    (
        agreement_tolerance_enabled,
        agreement_tolerance,
    ) = _agreement_tolerance_settings(cfg)

    tolerance_available = (
        agreement_tolerance_enabled
        and agreement_tolerance is not None
    )

    tolerance_sql = (
        str(float(agreement_tolerance))
        if tolerance_available
        else "NULL"
    )

    info = get_symbols("info")
    ok = get_symbols("check")

    logger.info(
        "%s Building Trade Sync reconciliation diagnostics.",
        info,
    )
    logger.info(
        "  Reconciled trade: %s",
        reconciled_trade_path,
    )
    logger.info(
        "  Country reliability: %s",
        country_reliability_path,
    )
    logger.info(
        "  Reconciliation audit: %s",
        reconciliation_audit_path,
    )
    logger.info(
        "  Quantity reconciliation audit: %s",
        quantity_reconciliation_audit_path,
    )
    logger.info(
        "  Output: %s",
        output_path,
    )
    logger.info(
        "  Agreement tolerance enabled: %s",
        agreement_tolerance_enabled,
    )

    if tolerance_available:
        logger.info(
            "  Agreement relative-difference threshold: %.6f",
            float(agreement_tolerance),
        )
    else:
        logger.info(
            "  Agreement-tolerance diagnostics available: False"
        )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        # ==============================================================
        # SCHEMA VALIDATION
        # ==============================================================

        reconciled_columns = {
            str(row[0])
            for row in con.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_parquet({reconciled_sql})
                """
            ).fetchall()
        }

        required_reconciled_columns = {
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",
            "export_value_fob",
            "import_value_fob_corrected",
            "reconciliation_export_value",
            "reconciliation_import_value",
            "trade_value_reconciled",
            "reconciliation_status",
            "reconciliation_method_applied",
            "reliability_evidence_case",
            "reliability_deviation",
            "exporter_adjusted_weight",
            "importer_adjusted_weight",
            "exporter_mean_deviation",
            "importer_mean_deviation",

            "export_synchronized_quantity_kg",
            "import_synchronized_quantity_kg",
            "export_is_synchronized_quantity_estimated",
            "import_is_synchronized_quantity_estimated",

            "quantity_availability_case",

            "quantity_simple_mean_kg",
            "quantity_value_weighted_kg",
            "quantity_reliability_weighted_kg",

            "quantity_relative_difference",
            "quantity_import_export_ratio",
            "quantity_absolute_log_gap",

            "export_quantity_reliability",
            "import_quantity_reliability",
            "export_quantity_reliability_weight",
            "import_quantity_reliability_weight",

            "export_implied_unit_value",
            "import_implied_unit_value",
            "unit_value_reference_observations",
            "product_year_median_unit_value",
            "product_year_log_unit_value_mad",

            "export_unit_value_log_distance",
            "import_unit_value_log_distance",
            "export_unit_value_robust_distance",
            "import_unit_value_robust_distance",

            "quantity_reconciled_kg",
            "quantity_reconciliation_method",

            "mean_implied_unit_value",
            "mean_unit_value_log_distance",

            "trade_synced_quantity_kg",
            "trade_synced_quantity_method",
            "trade_synced_quantity_status",
        }

        missing_reconciled_columns = sorted(
            required_reconciled_columns
            - reconciled_columns
        )

        if missing_reconciled_columns:
            raise RuntimeError(
                "Reconciled Trade Sync input is missing required "
                "columns: "
                + ", ".join(missing_reconciled_columns)
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
            "n_distinct_partners",
            "n_distinct_products",
            "mean_reliability_deviation",
            "reliability_index",
            "implicit_zero_share",
            "both_positive_share",
        }

        missing_reliability_columns = sorted(
            required_reliability_columns
            - reliability_columns
        )

        if missing_reliability_columns:
            raise RuntimeError(
                "Country-reliability input is missing required "
                "columns: "
                + ", ".join(missing_reliability_columns)
            )

        # ==============================================================
        # RECONCILIATION AUDIT VALIDATION
        # ==============================================================

        audit_row_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet({reconciliation_audit_sql})
            """
        ).fetchone()[0]

        if int(audit_row_count) != 1:
            raise RuntimeError(
                "Reconciliation audit must contain exactly one row; "
                f"found {int(audit_row_count):,}."
            )

        quantity_audit_row_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet({quantity_reconciliation_audit_sql})
            """
        ).fetchone()[0]

        if int(quantity_audit_row_count) != 1:
            raise RuntimeError(
                "Quantity reconciliation audit must contain exactly one row; "
                f"found {int(quantity_audit_row_count):,}."
            )

        value_audit_passed = con.execute(
            f"""
            SELECT audit_passed
            FROM read_parquet({reconciliation_audit_sql})
            LIMIT 1
            """
        ).fetchone()[0]

        quantity_audit_passed = con.execute(
            f"""
            SELECT audit_passed
            FROM read_parquet({quantity_reconciliation_audit_sql})
            LIMIT 1
            """
        ).fetchone()[0]

        if not bool(value_audit_passed):
            raise RuntimeError(
                "Trade Sync value reconciliation audit did not pass."
            )

        if not bool(quantity_audit_passed):
            raise RuntimeError(
                "Trade Sync quantity reconciliation audit did not pass."
            )

        # ==============================================================
        # DIAGNOSTIC BASE
        # ==============================================================

        reconciliation_base_query = f"""
        WITH reconciled AS (
            SELECT *
            FROM read_parquet({reconciled_sql})
        ),

        base AS (
            SELECT *
            FROM reconciled
        )

        SELECT
            *,

            -- --------------------------------------------------------------
            -- Quantity-reconciliation diagnostics
            -- --------------------------------------------------------------

            CASE
                WHEN quantity_simple_mean_kg > 0
                 AND quantity_reliability_weighted_kg IS NOT NULL
                    THEN (
                        ABS(
                            quantity_reliability_weighted_kg
                            - quantity_simple_mean_kg
                        )
                        / quantity_simple_mean_kg
                    )
                ELSE NULL
            END AS quantity_reliability_vs_mean_fraction,
        
            CASE
                WHEN quantity_simple_mean_kg > 0
                 AND quantity_value_weighted_kg IS NOT NULL
                    THEN (
                        ABS(
                            quantity_value_weighted_kg
                            - quantity_simple_mean_kg
                        )
                        / quantity_simple_mean_kg
                    )
                ELSE NULL
            END AS quantity_value_weighted_vs_mean_fraction,

            CASE
                WHEN quantity_simple_mean_kg > 0
                 AND trade_synced_quantity_kg IS NOT NULL
                    THEN (
                        ABS(
                            trade_synced_quantity_kg
                            - quantity_simple_mean_kg
                        )
                        / quantity_simple_mean_kg
                    )
                ELSE NULL
            END AS quantity_trade_synced_vs_mean_fraction,
            
            CASE
                WHEN quantity_value_weighted_kg > 0
                 AND quantity_reliability_weighted_kg IS NOT NULL
                    THEN (
                        ABS(
                            quantity_reliability_weighted_kg
                            - quantity_value_weighted_kg
                        )
                        / quantity_value_weighted_kg
                    )
                ELSE NULL
            END AS quantity_reliability_vs_value_weighted_fraction,

            CASE
                WHEN quantity_simple_mean_kg > 0
                 AND quantity_reconciled_kg IS NOT NULL
                    THEN (
                        ABS(
                            quantity_reconciled_kg
                            - quantity_simple_mean_kg
                        )
                        / quantity_simple_mean_kg
                    )
                ELSE NULL
            END AS quantity_reconciled_vs_mean_fraction,

            CASE
                WHEN export_unit_value_robust_distance IS NOT NULL
                 AND import_unit_value_robust_distance IS NOT NULL
                 AND ABS(
                     export_unit_value_robust_distance
                     - import_unit_value_robust_distance
                 ) >= 1.0
                 AND export_unit_value_robust_distance
                     < import_unit_value_robust_distance
                    THEN 'export'

                WHEN export_unit_value_robust_distance IS NOT NULL
                 AND import_unit_value_robust_distance IS NOT NULL
                 AND ABS(
                     export_unit_value_robust_distance
                     - import_unit_value_robust_distance
                 ) >= 1.0
                 AND import_unit_value_robust_distance
                     < export_unit_value_robust_distance
                    THEN 'import'

                ELSE NULL
            END AS quantity_plausibility_preferred_side,

            CASE
                WHEN export_unit_value_robust_distance IS NOT NULL
                 AND import_unit_value_robust_distance IS NOT NULL
                 AND ABS(
                     export_unit_value_robust_distance
                     - import_unit_value_robust_distance
                 ) >= 1.0
                 AND export_unit_value_robust_distance
                     < import_unit_value_robust_distance
                    THEN export_synchronized_quantity_kg

                WHEN export_unit_value_robust_distance IS NOT NULL
                 AND import_unit_value_robust_distance IS NOT NULL
                 AND ABS(
                     export_unit_value_robust_distance
                     - import_unit_value_robust_distance
                 ) >= 1.0
                 AND import_unit_value_robust_distance
                     < export_unit_value_robust_distance
                    THEN import_synchronized_quantity_kg

                ELSE quantity_reconciled_kg
            END AS quantity_plausibility_first_kg,
    
            CASE
                WHEN trade_value_reconciled > 0
                 AND quantity_simple_mean_kg > 0
                 AND product_year_median_unit_value > 0
                 AND product_year_log_unit_value_mad > 0
                    THEN (
                        ABS(
                            LN(
                                trade_value_reconciled
                                / quantity_simple_mean_kg
                            )
                            - LN(product_year_median_unit_value)
                        )
                        / product_year_log_unit_value_mad
                    )
                ELSE NULL
            END AS simple_mean_unit_value_robust_distance,

            CASE
                WHEN trade_value_reconciled > 0
                 AND quantity_value_weighted_kg > 0
                 AND product_year_median_unit_value > 0
                 AND product_year_log_unit_value_mad > 0
                    THEN (
                        ABS(
                            LN(
                                trade_value_reconciled
                                / quantity_value_weighted_kg
                            )
                            - LN(product_year_median_unit_value)
                        )
                        / product_year_log_unit_value_mad
                    )
                ELSE NULL
            END AS value_weighted_unit_value_robust_distance,

            CASE
                WHEN trade_value_reconciled > 0
                 AND quantity_reliability_weighted_kg > 0
                 AND product_year_median_unit_value > 0
                 AND product_year_log_unit_value_mad > 0
                    THEN (
                        ABS(
                            LN(
                                trade_value_reconciled
                                / quantity_reliability_weighted_kg
                            )
                            - LN(product_year_median_unit_value)
                        )
                        / product_year_log_unit_value_mad
                    )
                ELSE NULL
            END AS quantity_reliability_unit_value_robust_distance,

            CASE
                WHEN trade_value_reconciled > 0
                 AND trade_synced_quantity_kg > 0
                 AND product_year_median_unit_value > 0
                 AND product_year_log_unit_value_mad > 0
                    THEN (
                        ABS(
                            LN(
                                trade_value_reconciled
                                / trade_synced_quantity_kg
                            )
                            - LN(product_year_median_unit_value)
                        )
                        / product_year_log_unit_value_mad
                    )
                ELSE NULL
            END AS trade_synced_unit_value_robust_distance,

            -- --------------------------------------------------------------
            -- Existing value diagnostics
            -- --------------------------------------------------------------

            CASE
                WHEN reconciliation_export_value IS NOT NULL
                 AND reconciliation_import_value IS NOT NULL
                 AND GREATEST(
                    reconciliation_export_value,
                    reconciliation_import_value
                 ) > 0
                    THEN (
                        ABS(
                            reconciliation_export_value
                            - reconciliation_import_value
                        )
                        /
                        GREATEST(
                            reconciliation_export_value,
                            reconciliation_import_value
                        )
                    )
                ELSE NULL
            END AS relative_difference_from_larger,

            CASE
                WHEN reconciliation_export_value IS NOT NULL
                 AND reconciliation_import_value IS NOT NULL
                 AND (
                    reconciliation_export_value
                    + reconciliation_import_value
                 ) > 0
                    THEN (
                        ABS(
                            reconciliation_export_value
                            - reconciliation_import_value
                        )
                        /
                        (
                            (
                                reconciliation_export_value
                                + reconciliation_import_value
                            ) / 2.0
                        )
                    )
                ELSE NULL
            END AS symmetric_relative_difference,

            CASE
                WHEN reconciliation_export_value IS NOT NULL
                 AND reconciliation_import_value IS NOT NULL
                 AND (
                    reconciliation_export_value
                    + reconciliation_import_value
                 ) > 0
                    THEN (
                        trade_value_reconciled
                        -
                        (
                            reconciliation_export_value
                            + reconciliation_import_value
                        ) / 2.0
                    )
                    /
                    (
                        (
                            reconciliation_export_value
                            + reconciliation_import_value
                        ) / 2.0
                    )
                ELSE NULL
            END AS reconciled_vs_simple_average_fraction,

            CASE
                WHEN export_value_fob IS NOT NULL
                 AND isfinite(export_value_fob)
                 AND export_value_fob > 0
                 AND trade_value_reconciled IS NOT NULL
                    THEN (
                        trade_value_reconciled
                        - export_value_fob
                    ) / export_value_fob
                ELSE NULL
            END AS reconciled_vs_export_fraction,

            CASE
                WHEN import_value_fob_corrected IS NOT NULL
                 AND isfinite(import_value_fob_corrected)
                 AND import_value_fob_corrected > 0
                 AND trade_value_reconciled IS NOT NULL
                    THEN (
                        trade_value_reconciled
                        - import_value_fob_corrected
                    ) / import_value_fob_corrected
                ELSE NULL
            END AS reconciled_vs_import_fraction,

            CASE
                WHEN {str(tolerance_available).upper()}
                 AND reconciliation_export_value IS NOT NULL
                 AND reconciliation_import_value IS NOT NULL
                 AND GREATEST(
                    reconciliation_export_value,
                    reconciliation_import_value
                 ) > 0
                    THEN (
                        ABS(
                            reconciliation_export_value
                            - reconciliation_import_value
                        )
                        /
                        GREATEST(
                            reconciliation_export_value,
                            reconciliation_import_value
                        )
                    ) <= {tolerance_sql}
                ELSE NULL
            END AS within_agreement_tolerance

        FROM base
        """

        # ==============================================================
        # LONG-FORM DIAGNOSTIC METRICS
        # ==============================================================

        diagnostics_query = f"""
        WITH reconciliation_base AS (
            {reconciliation_base_query}
        ),

        plausibility_counterfactual AS (
            SELECT
                *,

                CASE
                    WHEN trade_value_reconciled > 0
                     AND quantity_reconciled_kg > 0
                     AND product_year_median_unit_value > 0
                     AND product_year_log_unit_value_mad > 0
                        THEN (
                            ABS(
                                LN(
                                    trade_value_reconciled
                                    / quantity_reconciled_kg
                                )
                                - LN(
                                    product_year_median_unit_value
                                )
                            )
                            / product_year_log_unit_value_mad
                        )
                    ELSE NULL
                END
                    AS quantity_reconciled_unit_value_robust_distance,

                CASE
                    WHEN trade_value_reconciled > 0
                     AND quantity_plausibility_first_kg > 0
                     AND product_year_median_unit_value > 0
                     AND product_year_log_unit_value_mad > 0
                        THEN (
                            ABS(
                                LN(
                                    trade_value_reconciled
                                    / quantity_plausibility_first_kg
                                )
                                - LN(
                                    product_year_median_unit_value
                                )
                            )
                            / product_year_log_unit_value_mad
                        )
                    ELSE NULL
                END
                    AS quantity_plausibility_first_robust_distance,

                CASE
                    WHEN quantity_reconciled_kg > 0
                     AND quantity_plausibility_first_kg IS NOT NULL
                        THEN (
                            ABS(
                                quantity_plausibility_first_kg
                                - quantity_reconciled_kg
                            )
                            / quantity_reconciled_kg
                        )
                    ELSE NULL
                END
                    AS quantity_plausibility_first_vs_current_fraction

            FROM reconciliation_base
        ),
    
        -- ==============================================================
        -- Common evaluation of the four bilateral quantity candidates
        -- ==============================================================

        quantity_candidate_long AS (

            SELECT
                year,
                quantity_relative_difference,

                exporter_iso,
                importer_iso,
                target_cmd,
                export_synchronized_quantity_kg,
                import_synchronized_quantity_kg,

                export_implied_unit_value,
                import_implied_unit_value,

                trade_value_reconciled,
                product_year_median_unit_value,
                product_year_log_unit_value_mad,

                'simple_mean' AS candidate_method,
                quantity_simple_mean_kg AS candidate_quantity_kg

            FROM plausibility_counterfactual
            WHERE quantity_availability_case = 'both_quantity'

            UNION ALL

            SELECT
                year,
                quantity_relative_difference,

                exporter_iso,
                importer_iso,
                target_cmd,

                export_synchronized_quantity_kg,
                import_synchronized_quantity_kg,

                export_implied_unit_value,
                import_implied_unit_value,

                trade_value_reconciled,
                product_year_median_unit_value,
                product_year_log_unit_value_mad,

                'value_weighted' AS candidate_method,
                quantity_value_weighted_kg AS candidate_quantity_kg

            FROM plausibility_counterfactual
            WHERE quantity_availability_case = 'both_quantity'

            UNION ALL

            SELECT
                year,
                quantity_relative_difference,

                exporter_iso,
                importer_iso,
                target_cmd,

                export_synchronized_quantity_kg,
                import_synchronized_quantity_kg,

                export_implied_unit_value,
                import_implied_unit_value,

                trade_value_reconciled,
                product_year_median_unit_value,
                product_year_log_unit_value_mad,

                'quantity_reliability_weighted' AS candidate_method,
                quantity_reliability_weighted_kg AS candidate_quantity_kg

            FROM plausibility_counterfactual
            WHERE quantity_availability_case = 'both_quantity'

            UNION ALL

            SELECT
                year,
                quantity_relative_difference,

                exporter_iso,
                importer_iso,
                target_cmd,

                export_synchronized_quantity_kg,
                import_synchronized_quantity_kg,

                export_implied_unit_value,
                import_implied_unit_value,

                trade_value_reconciled,
                product_year_median_unit_value,
                product_year_log_unit_value_mad,

                'unit_value_guided' AS candidate_method,
                trade_synced_quantity_kg AS candidate_quantity_kg

            FROM plausibility_counterfactual
            WHERE quantity_availability_case = 'both_quantity'
        ),

        quantity_candidate_post_sync AS (
            SELECT
                *,

                CASE
                    WHEN trade_value_reconciled > 0
                     AND candidate_quantity_kg > 0
                        THEN (
                            trade_value_reconciled
                            / candidate_quantity_kg
                        )
                    ELSE NULL
                END AS candidate_unit_value

            FROM quantity_candidate_long
        ),

        quantity_candidate_evaluated AS (
            SELECT
                *,

                -- ------------------------------------------------------
                -- 1. Preservation of original reporter-pair structure
                -- ------------------------------------------------------

                CASE
                    WHEN candidate_unit_value > 0
                     AND export_implied_unit_value > 0
                     AND import_implied_unit_value > 0
                        THEN (
                            candidate_unit_value
                            BETWEEN
                                LEAST(
                                    export_implied_unit_value,
                                    import_implied_unit_value
                                )
                            AND
                                GREATEST(
                                    export_implied_unit_value,
                                    import_implied_unit_value
                                )
                        )
                    ELSE NULL
                END AS unit_value_inside_reporter_range,

                CASE
                    WHEN candidate_unit_value > 0
                     AND export_implied_unit_value > 0
                     AND import_implied_unit_value > 0
                     AND candidate_unit_value
                         < LEAST(
                             export_implied_unit_value,
                             import_implied_unit_value
                         )
                        THEN ABS(
                            LN(
                                candidate_unit_value
                                / LEAST(
                                    export_implied_unit_value,
                                    import_implied_unit_value
                                )
                            )
                        )

                    WHEN candidate_unit_value > 0
                     AND export_implied_unit_value > 0
                     AND import_implied_unit_value > 0
                     AND candidate_unit_value
                         > GREATEST(
                             export_implied_unit_value,
                             import_implied_unit_value
                         )
                        THEN ABS(
                            LN(
                                candidate_unit_value
                                / GREATEST(
                                    export_implied_unit_value,
                                    import_implied_unit_value
                                )
                            )
                        )

                    WHEN candidate_unit_value > 0
                     AND export_implied_unit_value > 0
                     AND import_implied_unit_value > 0
                        THEN 0.0

                    ELSE NULL
                END AS unit_value_outside_reporter_log_distance,

                -- ------------------------------------------------------
                -- 2. Consistency with product-year value/kg reference
                -- ------------------------------------------------------

                CASE
                    WHEN candidate_unit_value > 0
                     AND product_year_median_unit_value > 0
                     AND product_year_log_unit_value_mad > 0
                        THEN (
                            ABS(
                                LN(candidate_unit_value)
                                - LN(product_year_median_unit_value)
                            )
                            / product_year_log_unit_value_mad
                        )
                    ELSE NULL
                END AS reference_robust_distance,

                -- ------------------------------------------------------
                -- 4. Amount of transformation away from observations
                -- ------------------------------------------------------

                CASE
                    WHEN candidate_quantity_kg > 0
                     AND export_synchronized_quantity_kg > 0
                     AND import_synchronized_quantity_kg > 0
                        THEN (
                            LEAST(
                                ABS(
                                    candidate_quantity_kg
                                    - export_synchronized_quantity_kg
                                ),
                                ABS(
                                    candidate_quantity_kg
                                    - import_synchronized_quantity_kg
                                )
                            )
                            /
                            GREATEST(
                                export_synchronized_quantity_kg,
                                import_synchronized_quantity_kg
                            )
                        )
                    ELSE NULL
                END AS nearest_report_quantity_fraction

            FROM quantity_candidate_post_sync
        ),

                quantity_three_candidate_match AS (
            SELECT
                year,
                exporter_iso,
                importer_iso,
                target_cmd,
                candidate_method,
                candidate_quantity_kg,
                candidate_unit_value,
                reference_robust_distance,

                MIN(reference_robust_distance) OVER (
                    PARTITION BY
                        year,
                        exporter_iso,
                        importer_iso,
                        target_cmd
                ) AS minimum_reference_distance

            FROM quantity_candidate_evaluated

            WHERE candidate_method IN (
                'simple_mean',
                'value_weighted',
                'quantity_reliability_weighted'
            )
              AND reference_robust_distance IS NOT NULL
        ),

        quantity_three_candidate_classified AS (
            SELECT
                *,

                CASE
                    WHEN ABS(
                        reference_robust_distance
                        - minimum_reference_distance
                    ) < 1e-12
                        THEN 1
                    ELSE 0
                END AS is_best_candidate

            FROM quantity_three_candidate_match
        ),

        quantity_three_candidate_row_result AS (
            SELECT
                year,
                exporter_iso,
                importer_iso,
                target_cmd,

                SUM(is_best_candidate) AS n_best_candidates,

                MAX(
                    CASE
                        WHEN is_best_candidate = 1
                            THEN candidate_method
                        ELSE NULL
                    END
                ) AS best_candidate_if_unique

            FROM quantity_three_candidate_classified

            GROUP BY
                year,
                exporter_iso,
                importer_iso,
                target_cmd
        ),
        
        quantity_three_candidate_summary_wide AS (
            SELECT

                CAST(
                    COUNT(*)
                    AS DOUBLE
                ) AS n_comparable_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE n_best_candidates = 1
                          AND best_candidate_if_unique = 'simple_mean'
                    )
                    AS DOUBLE
                ) AS n_simple_mean_best,

                CAST(
                    COUNT(*) FILTER (
                        WHERE n_best_candidates = 1
                          AND best_candidate_if_unique = 'value_weighted'
                    )
                    AS DOUBLE
                ) AS n_value_weighted_best,

                CAST(
                    COUNT(*) FILTER (
                        WHERE n_best_candidates = 1
                          AND best_candidate_if_unique
                                = 'quantity_reliability_weighted'
                    )
                    AS DOUBLE
                ) AS n_quantity_reliability_best,

                CAST(
                    COUNT(*) FILTER (
                        WHERE n_best_candidates > 1
                    )
                    AS DOUBLE
                ) AS n_ties,

                CAST(
                    AVG(
                        CASE
                            WHEN n_best_candidates = 1
                             AND best_candidate_if_unique = 'simple_mean'
                                THEN 1.0
                            WHEN n_best_candidates = 1
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS simple_mean_best_share,

                CAST(
                    AVG(
                        CASE
                            WHEN n_best_candidates = 1
                             AND best_candidate_if_unique = 'value_weighted'
                                THEN 1.0
                            WHEN n_best_candidates = 1
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS value_weighted_best_share,

                CAST(
                    AVG(
                        CASE
                            WHEN n_best_candidates = 1
                             AND best_candidate_if_unique
                                    = 'quantity_reliability_weighted'
                                THEN 1.0
                            WHEN n_best_candidates = 1
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS quantity_reliability_best_share

            FROM quantity_three_candidate_row_result
        ),

        quantity_three_candidate_extreme_result AS (
            SELECT
                classified.year,
                classified.exporter_iso,
                classified.importer_iso,
                classified.target_cmd,

                SUM(classified.is_best_candidate)
                    AS n_best_candidates,

                MAX(
                    CASE
                        WHEN classified.is_best_candidate = 1
                            THEN classified.candidate_method
                        ELSE NULL
                    END
                ) AS best_candidate_if_unique

            FROM quantity_three_candidate_classified AS classified

            INNER JOIN quantity_candidate_evaluated AS source
                ON classified.year = source.year
               AND classified.exporter_iso = source.exporter_iso
               AND classified.importer_iso = source.importer_iso
               AND classified.target_cmd = source.target_cmd
               AND source.candidate_method = 'simple_mean'

            WHERE source.quantity_relative_difference > 0.90

            GROUP BY
                classified.year,
                classified.exporter_iso,
                classified.importer_iso,
                classified.target_cmd
        ),

        quantity_three_candidate_extreme_wide AS (
            SELECT
                CAST(COUNT(*) AS DOUBLE)
                    AS n_comparable_rows,

                CAST(
                    AVG(
                        CASE
                            WHEN n_best_candidates = 1
                             AND best_candidate_if_unique = 'simple_mean'
                                THEN 1.0
                            WHEN n_best_candidates = 1
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS simple_mean_best_share,

                CAST(
                    AVG(
                        CASE
                            WHEN n_best_candidates = 1
                             AND best_candidate_if_unique = 'value_weighted'
                                THEN 1.0
                            WHEN n_best_candidates = 1
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS value_weighted_best_share,

                CAST(
                    AVG(
                        CASE
                            WHEN n_best_candidates = 1
                             AND best_candidate_if_unique
                                    = 'quantity_reliability_weighted'
                                THEN 1.0
                            WHEN n_best_candidates = 1
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS quantity_reliability_best_share,

                CAST(
                    AVG(
                        CASE
                            WHEN n_best_candidates > 1
                                THEN 1.0
                            ELSE 0.0
                        END
                    )
                    AS DOUBLE
                ) AS tie_share

            FROM quantity_three_candidate_extreme_result
        ),
        quantity_original_pair_wide AS (
            SELECT

                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_availability_case = 'both_quantity'
                          AND export_implied_unit_value > 0
                          AND import_implied_unit_value > 0
                    )
                    AS DOUBLE
                ) AS n_original_pair_rows,

                CAST(
                    AVG(
                        CASE
                            WHEN export_unit_value_robust_distance
                                    < import_unit_value_robust_distance
                                THEN 1.0
                            WHEN export_unit_value_robust_distance IS NOT NULL
                             AND import_unit_value_robust_distance IS NOT NULL
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS export_closer_to_reference_share,

                CAST(
                    AVG(
                        CASE
                            WHEN import_unit_value_robust_distance
                                    < export_unit_value_robust_distance
                                THEN 1.0
                            WHEN export_unit_value_robust_distance IS NOT NULL
                             AND import_unit_value_robust_distance IS NOT NULL
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS import_closer_to_reference_share,

                CAST(
                    AVG(
                        CASE
                            WHEN export_unit_value_robust_distance IS NOT NULL
                             AND import_unit_value_robust_distance IS NOT NULL
                             AND ABS(
                                 export_unit_value_robust_distance
                                 - import_unit_value_robust_distance
                             ) >= 1.0
                                THEN 1.0
                            WHEN export_unit_value_robust_distance IS NOT NULL
                             AND import_unit_value_robust_distance IS NOT NULL
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS clear_reference_preference_share,

                CAST(
                    MEDIAN(
                        ABS(
                            LN(
                                export_implied_unit_value
                                / import_implied_unit_value
                            )
                        )
                    ) FILTER (
                        WHERE export_implied_unit_value > 0
                          AND import_implied_unit_value > 0
                    )
                    AS DOUBLE
                ) AS median_original_unit_value_log_gap

            FROM plausibility_counterfactual
        ),

        quantity_candidate_overall_wide AS (
            SELECT
                candidate_method,

                CAST(COUNT(*) AS DOUBLE) AS n_rows,

                CAST(
                    AVG(
                        CASE
                            WHEN unit_value_inside_reporter_range
                                THEN 1.0
                            WHEN unit_value_inside_reporter_range IS NOT NULL
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS reporter_unit_value_range_share,

                CAST(
                    MEDIAN(
                        unit_value_outside_reporter_log_distance
                    ) FILTER (
                        WHERE unit_value_outside_reporter_log_distance > 0
                    )
                    AS DOUBLE
                ) AS median_outside_reporter_log_distance,

                CAST(
                    MEDIAN(reference_robust_distance)
                    AS DOUBLE
                ) AS median_reference_robust_distance,

                CAST(
                    QUANTILE_CONT(
                        reference_robust_distance,
                        0.90
                    )
                    AS DOUBLE
                ) AS p90_reference_robust_distance,

                CAST(
                    MEDIAN(nearest_report_quantity_fraction)
                    AS DOUBLE
                ) AS median_nearest_report_quantity_fraction,

                CAST(
                    QUANTILE_CONT(
                        nearest_report_quantity_fraction,
                        0.90
                    )
                    AS DOUBLE
                ) AS p90_nearest_report_quantity_fraction

            FROM quantity_candidate_evaluated

            WHERE candidate_quantity_kg IS NOT NULL
              AND candidate_quantity_kg > 0

            GROUP BY candidate_method
        ),

        quantity_candidate_extreme_wide AS (
            SELECT
                candidate_method,

                CAST(COUNT(*) AS DOUBLE) AS n_rows,

                CAST(
                    AVG(
                        CASE
                            WHEN unit_value_inside_reporter_range
                                THEN 1.0
                            WHEN unit_value_inside_reporter_range IS NOT NULL
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS reporter_unit_value_range_share,

                CAST(
                    MEDIAN(reference_robust_distance)
                    AS DOUBLE
                ) AS median_reference_robust_distance,

                CAST(
                    MEDIAN(nearest_report_quantity_fraction)
                    AS DOUBLE
                ) AS median_nearest_report_quantity_fraction

            FROM quantity_candidate_evaluated

            WHERE quantity_relative_difference > 0.90
              AND candidate_quantity_kg IS NOT NULL
              AND candidate_quantity_kg > 0

            GROUP BY candidate_method
        ),

        quantity_reference_target AS (
            SELECT
                *,

                CASE
                    WHEN trade_value_reconciled IS NOT NULL
                    AND isfinite(trade_value_reconciled)
                    AND trade_value_reconciled > 0
                    AND product_year_median_unit_value IS NOT NULL
                    AND isfinite(product_year_median_unit_value)
                    AND product_year_median_unit_value > 0
                        THEN (
                            trade_value_reconciled
                            / product_year_median_unit_value
                        )
                    ELSE NULL
                END AS quantity_reference_kg

            FROM plausibility_counterfactual
        ),

        quantity_reference_base AS (
            SELECT
                *,

                CASE
                    WHEN quantity_reference_kg > 0
                    AND export_synchronized_quantity_kg > 0
                        THEN ABS(
                            LN(
                                export_synchronized_quantity_kg
                                / quantity_reference_kg
                            )
                        )
                    ELSE NULL
                END AS export_quantity_reference_log_distance,

                CASE
                    WHEN quantity_reference_kg > 0
                    AND import_synchronized_quantity_kg > 0
                        THEN ABS(
                            LN(
                                import_synchronized_quantity_kg
                                / quantity_reference_kg
                            )
                        )
                    ELSE NULL
                END AS import_quantity_reference_log_distance,

                CASE
                    WHEN quantity_reference_kg > 0
                    AND quantity_reconciled_kg > 0
                        THEN ABS(
                            LN(
                                quantity_reconciled_kg
                                / quantity_reference_kg
                            )
                        )
                    ELSE NULL
                END AS current_quantity_reference_log_distance,

                CASE
                    WHEN quantity_reference_kg > 0
                    AND quantity_reconciled_kg > 0
                        THEN (
                            ABS(
                                quantity_reference_kg
                                - quantity_reconciled_kg
                            )
                            / quantity_reconciled_kg
                        )
                    ELSE NULL
                END AS quantity_reference_vs_current_fraction,

                CASE
                    WHEN quantity_reference_kg > 0
                    AND export_synchronized_quantity_kg > 0
                    AND import_synchronized_quantity_kg > 0
                        THEN (
                            quantity_reference_kg
                            BETWEEN
                                LEAST(
                                    export_synchronized_quantity_kg,
                                    import_synchronized_quantity_kg
                                )
                            AND
                                GREATEST(
                                    export_synchronized_quantity_kg,
                                    import_synchronized_quantity_kg
                                )
                        )
                    ELSE NULL
                END AS quantity_reference_within_reporter_range

            FROM quantity_reference_target
        ),

        quantity_reference_classified AS (
            SELECT
                *,

                CASE
                    WHEN export_quantity_reference_log_distance IS NULL
                    OR import_quantity_reference_log_distance IS NULL
                    OR current_quantity_reference_log_distance IS NULL
                        THEN NULL

                    WHEN current_quantity_reference_log_distance
                            <= export_quantity_reference_log_distance
                    AND current_quantity_reference_log_distance
                            <= import_quantity_reference_log_distance
                        THEN 'current'

                    WHEN export_quantity_reference_log_distance
                            < current_quantity_reference_log_distance
                    AND export_quantity_reference_log_distance
                            <= import_quantity_reference_log_distance
                        THEN 'export'

                    WHEN import_quantity_reference_log_distance
                            < current_quantity_reference_log_distance
                    AND import_quantity_reference_log_distance
                            < export_quantity_reference_log_distance
                        THEN 'import'

                    ELSE 'tie'
                END AS quantity_reference_closest_candidate

            FROM quantity_reference_base
        ),

        reliability AS (
                    SELECT *
                    FROM read_parquet({reliability_sql})
                ),

        overall_wide AS (
            SELECT
                CAST(COUNT(*) AS DOUBLE)
                    AS n_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE trade_value_reconciled IS NOT NULL
                    )
                    AS DOUBLE
                ) AS n_reconciled_value_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE reconciliation_status = 'not_eligible'
                    )
                    AS DOUBLE
                ) AS n_not_eligible_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE reliability_evidence_case IS NOT NULL
                    )
                    AS DOUBLE
                ) AS n_reliability_weighted_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE reconciliation_method_applied IN (
                            'export_report_retained',
                            'import_report_retained'
                        )
                    )
                    AS DOUBLE
                ) AS n_single_report_retained_rows,

                CAST(
                    SUM(trade_value_reconciled) FILTER (
                        WHERE trade_value_reconciled IS NOT NULL
                    )
                    AS DOUBLE
                ) AS total_reconciled_value,

                CAST(
                    SUM(export_value_fob) FILTER (
                        WHERE export_value_fob IS NOT NULL
                        AND isfinite(export_value_fob)
                        AND export_value_fob > 0
                    )
                    AS DOUBLE
                ) AS total_reported_export_fob,

                CAST(
                    SUM(import_value_fob_corrected) FILTER (
                        WHERE import_value_fob_corrected IS NOT NULL
                        AND isfinite(import_value_fob_corrected)
                        AND import_value_fob_corrected > 0
                    )
                    AS DOUBLE
                ) AS total_corrected_import_fob,

                CAST(
                    MEDIAN(trade_value_reconciled) FILTER (
                        WHERE trade_value_reconciled IS NOT NULL
                    )
                    AS DOUBLE
                ) AS median_reconciled_value,

                CAST(
                    MEDIAN(relative_difference_from_larger) FILTER (
                        WHERE relative_difference_from_larger IS NOT NULL
                    )
                    AS DOUBLE
                ) AS median_relative_difference,

                CAST(
                    QUANTILE_CONT(
                        relative_difference_from_larger,
                        0.90
                    ) FILTER (
                        WHERE relative_difference_from_larger IS NOT NULL
                    )
                    AS DOUBLE
                ) AS p90_relative_difference,

                CAST(
                    MEDIAN(
                        ABS(reconciled_vs_simple_average_fraction)
                    ) FILTER (
                        WHERE reconciled_vs_simple_average_fraction
                            IS NOT NULL
                    )
                    AS DOUBLE
                ) AS median_absolute_change_from_simple_average,

                CAST(
                    QUANTILE_CONT(
                        ABS(reconciled_vs_simple_average_fraction),
                        0.90
                    ) FILTER (
                        WHERE reconciled_vs_simple_average_fraction
                            IS NOT NULL
                    )
                    AS DOUBLE
                ) AS p90_absolute_change_from_simple_average,

                CAST(
                    AVG(
                        CASE
                            WHEN within_agreement_tolerance
                                THEN 1.0
                            WHEN within_agreement_tolerance IS NOT NULL
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS within_agreement_tolerance_share

            FROM plausibility_counterfactual
        ),

        quantity_overall_wide AS (
            SELECT
                CAST(COUNT(*) AS DOUBLE)
                    AS n_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE export_synchronized_quantity_kg IS NOT NULL
                          AND isfinite(export_synchronized_quantity_kg)
                          AND export_synchronized_quantity_kg > 0
                    )
                    AS DOUBLE
                ) AS n_export_quantity_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE import_synchronized_quantity_kg IS NOT NULL
                          AND isfinite(import_synchronized_quantity_kg)
                          AND import_synchronized_quantity_kg > 0
                    )
                    AS DOUBLE
                ) AS n_import_quantity_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_relative_difference IS NOT NULL
                    )
                    AS DOUBLE
                ) AS n_both_quantity_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE export_synchronized_quantity_kg IS NOT NULL
                          AND isfinite(export_synchronized_quantity_kg)
                          AND export_synchronized_quantity_kg > 0
                          AND (
                              import_synchronized_quantity_kg IS NULL
                              OR NOT isfinite(import_synchronized_quantity_kg)
                              OR import_synchronized_quantity_kg <= 0
                          )
                    )
                    AS DOUBLE
                ) AS n_export_quantity_only_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE import_synchronized_quantity_kg IS NOT NULL
                          AND isfinite(import_synchronized_quantity_kg)
                          AND import_synchronized_quantity_kg > 0
                          AND (
                              export_synchronized_quantity_kg IS NULL
                              OR NOT isfinite(export_synchronized_quantity_kg)
                              OR export_synchronized_quantity_kg <= 0
                          )
                    )
                    AS DOUBLE
                ) AS n_import_quantity_only_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE (
                            export_synchronized_quantity_kg IS NULL
                            OR NOT isfinite(export_synchronized_quantity_kg)
                            OR export_synchronized_quantity_kg <= 0
                        )
                        AND (
                            import_synchronized_quantity_kg IS NULL
                            OR NOT isfinite(import_synchronized_quantity_kg)
                            OR import_synchronized_quantity_kg <= 0
                        )
                    )
                    AS DOUBLE
                ) AS n_no_quantity_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_reconciliation_method
                            = 'quantity_reliability_weighted'
                    )
                    AS DOUBLE
                ) AS n_quantity_reliability_weighted,

                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_reconciliation_method
                            = 'bilateral_simple_mean_fallback'
                    )
                    AS DOUBLE
                ) AS n_quantity_simple_mean_fallback,

                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_reconciliation_method
                            = 'export_quantity_retained'
                    )
                    AS DOUBLE
                ) AS n_quantity_export_retained,

                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_reconciliation_method
                            = 'import_quantity_retained'
                    )
                    AS DOUBLE
                ) AS n_quantity_import_retained,

                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_reconciliation_method
                            = 'no_usable_quantity'
                    )
                    AS DOUBLE
                ) AS n_quantity_missing,

                CAST(
                    MEDIAN(quantity_relative_difference)
                    AS DOUBLE
                ) AS median_quantity_relative_difference,

                CAST(
                    QUANTILE_CONT(
                        quantity_relative_difference,
                        0.75
                    )
                    AS DOUBLE
                ) AS p75_quantity_relative_difference,

                CAST(
                    QUANTILE_CONT(
                        quantity_relative_difference,
                        0.90
                    )
                    AS DOUBLE
                ) AS p90_quantity_relative_difference,

                CAST(
                    QUANTILE_CONT(
                        quantity_relative_difference,
                        0.95
                    )
                    AS DOUBLE
                ) AS p95_quantity_relative_difference,

                CAST(
                    QUANTILE_CONT(
                        quantity_relative_difference,
                        0.99
                    )
                    AS DOUBLE
                ) AS p99_quantity_relative_difference,

                CAST(
                    MEDIAN(quantity_import_export_ratio)
                    AS DOUBLE
                ) AS median_quantity_import_export_ratio,

                CAST(
                    MEDIAN(quantity_absolute_log_gap)
                    AS DOUBLE
                ) AS median_quantity_absolute_log_gap,

                CAST(
                    MEDIAN(quantity_reconciled_vs_mean_fraction)
                    AS DOUBLE
                ) AS median_quantity_reconciled_vs_mean_fraction,

                CAST(
                    QUANTILE_CONT(
                        quantity_reconciled_vs_mean_fraction,
                        0.90
                    )
                    AS DOUBLE
                ) AS p90_quantity_reconciled_vs_mean_fraction,

                CAST(
                    MEDIAN(
                        quantity_value_weighted_vs_mean_fraction
                    )
                    AS DOUBLE
                ) AS median_quantity_value_weighted_vs_mean_fraction,

                CAST(
                    QUANTILE_CONT(
                        quantity_value_weighted_vs_mean_fraction,
                        0.90
                    )
                    AS DOUBLE
                ) AS p90_quantity_value_weighted_vs_mean_fraction,

                CAST(
                    MEDIAN(
                        quantity_trade_synced_vs_mean_fraction
                    )
                    AS DOUBLE
                ) AS median_quantity_trade_synced_vs_mean_fraction,

                CAST(
                    QUANTILE_CONT(
                        quantity_trade_synced_vs_mean_fraction,
                        0.90
                    )
                    AS DOUBLE
                ) AS p90_quantity_trade_synced_vs_mean_fraction,

                CAST(
                    QUANTILE_CONT(
                        simple_mean_unit_value_robust_distance,
                        0.90
                    )
                    AS DOUBLE
                ) AS p90_simple_mean_unit_value_distance,


                CAST(
                    QUANTILE_CONT(
                        value_weighted_unit_value_robust_distance,
                        0.90
                    )
                    AS DOUBLE
                ) AS p90_value_weighted_unit_value_distance,

                CAST(
                    MEDIAN(
                        quantity_reliability_unit_value_robust_distance
                    )
                    AS DOUBLE
                ) AS median_quantity_reliability_unit_value_distance,

                CAST(
                    QUANTILE_CONT(
                        quantity_reliability_unit_value_robust_distance,
                        0.90
                    )
                    AS DOUBLE
                ) AS p90_quantity_reliability_unit_value_distance,

                CAST(
                    MEDIAN(trade_synced_unit_value_robust_distance)
                    AS DOUBLE
                ) AS median_trade_synced_unit_value_distance,

                CAST(
                    QUANTILE_CONT(
                        trade_synced_unit_value_robust_distance,
                        0.90
                    )
                    AS DOUBLE
                ) AS p90_trade_synced_unit_value_distance,

                CAST(
                    MEDIAN(quantity_reliability_vs_value_weighted_fraction)
                    AS DOUBLE
                ) AS median_quantity_reliability_vs_value_weighted_fraction,

                CAST(
                    QUANTILE_CONT(
                        quantity_reliability_vs_value_weighted_fraction,
                        0.90
                    )
                    AS DOUBLE
                ) AS p90_quantity_reliability_vs_value_weighted_fraction,

                CAST(
                    MEDIAN(export_quantity_reliability)
                    AS DOUBLE
                ) AS median_export_quantity_reliability,

                CAST(
                    MEDIAN(import_quantity_reliability)
                    AS DOUBLE
                ) AS median_import_quantity_reliability,

                CAST(
                    MEDIAN(
                        ABS(
                            export_quantity_reliability
                            - import_quantity_reliability
                        )
                    )
                    AS DOUBLE
                ) AS median_quantity_reliability_gap,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_relative_difference <= 0.05
                                THEN 1.0
                            WHEN quantity_relative_difference IS NOT NULL
                                THEN 0.0
                        END
                    )
                    AS DOUBLE
                ) AS quantity_within_5pct_share,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_relative_difference <= 0.10
                                THEN 1.0
                            WHEN quantity_relative_difference IS NOT NULL
                                THEN 0.0
                        END
                    )
                    AS DOUBLE
                ) AS quantity_within_10pct_share,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_relative_difference <= 0.20
                                THEN 1.0
                            WHEN quantity_relative_difference IS NOT NULL
                                THEN 0.0
                        END
                    )
                    AS DOUBLE
                ) AS quantity_within_20pct_share,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_relative_difference <= 0.50
                                THEN 1.0
                            WHEN quantity_relative_difference IS NOT NULL
                                THEN 0.0
                        END
                    )
                    AS DOUBLE
                ) AS quantity_within_50pct_share

            FROM plausibility_counterfactual
        ),

        quantity_plausibility_wide AS (
            SELECT
                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_availability_case = 'both_quantity'
                          AND export_unit_value_robust_distance IS NOT NULL
                          AND import_unit_value_robust_distance IS NOT NULL
                    )
                    AS DOUBLE
                ) AS n_plausibility_comparable_rows,

                CAST(
                    AVG(
                        CASE
                            WHEN export_unit_value_robust_distance IS NOT NULL
                             AND import_unit_value_robust_distance IS NOT NULL
                             AND export_unit_value_robust_distance
                                 < import_unit_value_robust_distance
                                THEN 1.0
                            WHEN export_unit_value_robust_distance IS NOT NULL
                             AND import_unit_value_robust_distance IS NOT NULL
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS export_more_plausible_share,

                CAST(
                    AVG(
                        CASE
                            WHEN export_unit_value_robust_distance IS NOT NULL
                             AND import_unit_value_robust_distance IS NOT NULL
                             AND import_unit_value_robust_distance
                                 < export_unit_value_robust_distance
                                THEN 1.0
                            WHEN export_unit_value_robust_distance IS NOT NULL
                             AND import_unit_value_robust_distance IS NOT NULL
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS import_more_plausible_share

            FROM plausibility_counterfactual
        ),

        quantity_extreme_plausibility_wide AS (
            SELECT
                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_relative_difference > 0.90
                    )
                    AS DOUBLE
                ) AS n_extreme_quantity_disagreement,

                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_relative_difference > 0.90
                          AND export_unit_value_robust_distance IS NOT NULL
                          AND import_unit_value_robust_distance IS NOT NULL
                    )
                    AS DOUBLE
                ) AS n_extreme_with_plausibility_evidence,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_relative_difference > 0.90
                             AND export_unit_value_robust_distance IS NOT NULL
                             AND import_unit_value_robust_distance IS NOT NULL
                             AND ABS(
                                 export_unit_value_robust_distance
                                 - import_unit_value_robust_distance
                             ) >= 1.0
                                THEN 1.0

                            WHEN quantity_relative_difference > 0.90
                             AND export_unit_value_robust_distance IS NOT NULL
                             AND import_unit_value_robust_distance IS NOT NULL
                                THEN 0.0

                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS extreme_clear_plausibility_preference_share

            FROM reconciliation_base
        ),

        quantity_plausibility_first_wide AS (
            SELECT
                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_plausibility_preferred_side
                            IS NOT NULL
                    )
                    AS DOUBLE
                ) AS n_plausibility_decisive_rows,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_plausibility_preferred_side
                                IS NOT NULL
                                THEN 1.0
                            WHEN quantity_availability_case
                                = 'both_quantity'
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS plausibility_decisive_share,

                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_plausibility_preferred_side
                            = 'export'
                    )
                    AS DOUBLE
                ) AS n_plausibility_export_selected,

                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_plausibility_preferred_side
                            = 'import'
                    )
                    AS DOUBLE
                ) AS n_plausibility_import_selected,

                CAST(
                    MEDIAN(
                        quantity_plausibility_first_vs_current_fraction
                    ) FILTER (
                        WHERE quantity_plausibility_preferred_side
                            IS NOT NULL
                    )
                    AS DOUBLE
                ) AS median_plausibility_first_change,

                CAST(
                    QUANTILE_CONT(
                        quantity_plausibility_first_vs_current_fraction,
                        0.90
                    ) FILTER (
                        WHERE quantity_plausibility_preferred_side
                            IS NOT NULL
                    )
                    AS DOUBLE
                ) AS p90_plausibility_first_change,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_plausibility_preferred_side
                                    IS NOT NULL
                             AND quantity_plausibility_first_robust_distance
                                    IS NOT NULL
                             AND quantity_reconciled_unit_value_robust_distance
                                    IS NOT NULL
                             AND quantity_plausibility_first_robust_distance
                                 <
                                 quantity_reconciled_unit_value_robust_distance
                                THEN 1.0

                            WHEN quantity_plausibility_preferred_side
                                    IS NOT NULL
                             AND quantity_plausibility_first_robust_distance
                                    IS NOT NULL
                             AND quantity_reconciled_unit_value_robust_distance
                                    IS NOT NULL
                                THEN 0.0

                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS plausibility_first_improves_share,

                CAST(
                    MEDIAN(
                        quantity_reconciled_unit_value_robust_distance
                        - quantity_plausibility_first_robust_distance
                    ) FILTER (
                        WHERE quantity_plausibility_preferred_side
                                IS NOT NULL
                          AND quantity_reconciled_unit_value_robust_distance
                                IS NOT NULL
                          AND quantity_plausibility_first_robust_distance
                                IS NOT NULL
                    )
                    AS DOUBLE
                ) AS median_robust_distance_improvement

            FROM plausibility_counterfactual
        ),

        quantity_reference_wide AS (
            SELECT
                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_availability_case = 'both_quantity'
                        AND quantity_reference_kg IS NOT NULL
                    )
                    AS DOUBLE
                ) AS n_reference_comparable_rows,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_availability_case = 'both_quantity'
                            AND quantity_reference_within_reporter_range
                                THEN 1.0
                            WHEN quantity_availability_case = 'both_quantity'
                            AND quantity_reference_within_reporter_range IS NOT NULL
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS reference_between_reporters_share,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_reference_closest_candidate = 'current'
                                THEN 1.0
                            WHEN quantity_reference_closest_candidate IS NOT NULL
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS current_closest_to_reference_share,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_reference_closest_candidate = 'export'
                                THEN 1.0
                            WHEN quantity_reference_closest_candidate IS NOT NULL
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS export_closest_to_reference_share,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_reference_closest_candidate = 'import'
                                THEN 1.0
                            WHEN quantity_reference_closest_candidate IS NOT NULL
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS import_closest_to_reference_share,

                CAST(
                    MEDIAN(quantity_reference_vs_current_fraction)
                        FILTER (
                            WHERE quantity_availability_case = 'both_quantity'
                            AND quantity_reference_vs_current_fraction IS NOT NULL
                        )
                    AS DOUBLE
                ) AS median_reference_vs_current_fraction,

                CAST(
                    QUANTILE_CONT(
                        quantity_reference_vs_current_fraction,
                        0.90
                    ) FILTER (
                        WHERE quantity_availability_case = 'both_quantity'
                        AND quantity_reference_vs_current_fraction IS NOT NULL
                    )
                    AS DOUBLE
                ) AS p90_reference_vs_current_fraction,

                CAST(
                    MEDIAN(current_quantity_reference_log_distance)
                        FILTER (
                            WHERE quantity_availability_case = 'both_quantity'
                            AND current_quantity_reference_log_distance IS NOT NULL
                        )
                    AS DOUBLE
                ) AS median_current_reference_log_distance,

                CAST(
                    MEDIAN(export_quantity_reference_log_distance)
                        FILTER (
                            WHERE quantity_availability_case = 'both_quantity'
                            AND export_quantity_reference_log_distance IS NOT NULL
                        )
                    AS DOUBLE
                ) AS median_export_reference_log_distance,

                CAST(
                    MEDIAN(import_quantity_reference_log_distance)
                        FILTER (
                            WHERE quantity_availability_case = 'both_quantity'
                            AND import_quantity_reference_log_distance IS NOT NULL
                        )
                    AS DOUBLE
                ) AS median_import_reference_log_distance,

                CAST(
                    MEDIAN(unit_value_reference_observations)
                        FILTER (
                            WHERE quantity_availability_case = 'both_quantity'
                            AND quantity_reference_kg IS NOT NULL
                            AND unit_value_reference_observations IS NOT NULL
                        )
                    AS DOUBLE
                ) AS median_reference_observations

            FROM quantity_reference_classified
        ),
    
        
        quantity_trade_sync_wide AS (
            SELECT
                CAST(
                    COUNT(*) FILTER (
                        WHERE trade_synced_quantity_kg IS NOT NULL
                    )
                    AS DOUBLE
                ) AS n_trade_synced_quantity_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE trade_synced_quantity_method
                            = 'unit_value_guided_export'
                    )
                    AS DOUBLE
                ) AS n_unit_value_guided_export,

                CAST(
                    COUNT(*) FILTER (
                        WHERE trade_synced_quantity_method
                            = 'unit_value_guided_import'
                    )
                    AS DOUBLE
                ) AS n_unit_value_guided_import,

                CAST(
                    COUNT(*) FILTER (
                        WHERE trade_synced_quantity_method
                            = 'unit_value_guided_mean'
                    )
                    AS DOUBLE
                ) AS n_unit_value_guided_mean,

                CAST(
                    COUNT(*) FILTER (
                        WHERE trade_synced_quantity_method
                            = 'bilateral_mean_no_unit_value_reference'
                    )
                    AS DOUBLE
                ) AS n_bilateral_mean_no_reference,

                CAST(
                    COUNT(*) FILTER (
                        WHERE trade_synced_quantity_method
                            = 'export_quantity_retained'
                    )
                    AS DOUBLE
                ) AS n_export_quantity_retained,

                CAST(
                    COUNT(*) FILTER (
                        WHERE trade_synced_quantity_method
                            = 'import_quantity_retained'
                    )
                    AS DOUBLE
                ) AS n_import_quantity_retained,

                CAST(
                    COUNT(*) FILTER (
                        WHERE trade_synced_quantity_method
                            = 'no_usable_quantity'
                    )
                    AS DOUBLE
                ) AS n_no_usable_quantity,

                -- Shares among bilateral rows only
                CAST(
                    AVG(
                        CASE
                            WHEN quantity_availability_case = 'both_quantity'
                             AND trade_synced_quantity_method
                                    = 'unit_value_guided_export'
                                THEN 1.0
                            WHEN quantity_availability_case = 'both_quantity'
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS unit_value_guided_export_share,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_availability_case = 'both_quantity'
                             AND trade_synced_quantity_method
                                    = 'unit_value_guided_import'
                                THEN 1.0
                            WHEN quantity_availability_case = 'both_quantity'
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS unit_value_guided_import_share,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_availability_case = 'both_quantity'
                             AND trade_synced_quantity_method
                                    = 'unit_value_guided_mean'
                                THEN 1.0
                            WHEN quantity_availability_case = 'both_quantity'
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS unit_value_guided_mean_share,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_availability_case = 'both_quantity'
                             AND trade_synced_quantity_method
                                    = 'bilateral_mean_no_unit_value_reference'
                                THEN 1.0
                            WHEN quantity_availability_case = 'both_quantity'
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS bilateral_mean_no_reference_share

            FROM plausibility_counterfactual
        ),

        quantity_disagreement_band_wide AS (
            SELECT
                CASE
                    WHEN quantity_relative_difference <= 0.10
                        THEN '00_10'
                    WHEN quantity_relative_difference <= 0.50
                        THEN '10_50'
                    WHEN quantity_relative_difference <= 0.90
                        THEN '50_90'
                    ELSE '90_100'
                END AS quantity_disagreement_band,

                COUNT(*) AS n_rows,

                MEDIAN(
                    ABS(
                        export_quantity_reliability
                        - import_quantity_reliability
                    )
                ) FILTER (
                    WHERE export_quantity_reliability IS NOT NULL
                      AND import_quantity_reliability IS NOT NULL
                ) AS median_quantity_reliability_gap,

                MEDIAN(
                    quantity_reconciled_vs_mean_fraction
                ) AS median_reconciled_vs_mean_fraction,

                                MEDIAN(
                    quantity_value_weighted_vs_mean_fraction
                ) AS median_value_weighted_vs_mean_fraction,

                MEDIAN(
                    quantity_reliability_vs_mean_fraction
                ) AS median_quantity_reliability_vs_mean_fraction,

                MEDIAN(
                    quantity_trade_synced_vs_mean_fraction
                ) AS median_trade_synced_vs_mean_fraction,
                
                AVG(
                    CASE
                        WHEN export_unit_value_robust_distance IS NOT NULL
                         AND import_unit_value_robust_distance IS NOT NULL
                            THEN 1.0
                        ELSE 0.0
                    END
                ) AS plausibility_evidence_share,

                AVG(
                    CASE
                        WHEN export_unit_value_robust_distance IS NOT NULL
                         AND import_unit_value_robust_distance IS NOT NULL
                         AND export_unit_value_robust_distance
                             < import_unit_value_robust_distance
                            THEN 1.0
                        WHEN export_unit_value_robust_distance IS NOT NULL
                         AND import_unit_value_robust_distance IS NOT NULL
                            THEN 0.0
                        ELSE NULL
                    END
                ) AS export_more_plausible_share,

                AVG(
                    CASE
                        WHEN export_unit_value_robust_distance IS NOT NULL
                         AND import_unit_value_robust_distance IS NOT NULL
                         AND import_unit_value_robust_distance
                             < export_unit_value_robust_distance
                            THEN 1.0
                        WHEN export_unit_value_robust_distance IS NOT NULL
                         AND import_unit_value_robust_distance IS NOT NULL
                            THEN 0.0
                        ELSE NULL
                    END
                ) AS import_more_plausible_share,

                AVG(
                    CASE
                        WHEN export_unit_value_robust_distance IS NOT NULL
                         AND import_unit_value_robust_distance IS NOT NULL
                         AND ABS(
                             export_unit_value_robust_distance
                             - import_unit_value_robust_distance
                         ) >= 1.0
                            THEN 1.0
                        WHEN export_unit_value_robust_distance IS NOT NULL
                         AND import_unit_value_robust_distance IS NOT NULL
                            THEN 0.0
                        ELSE NULL
                    END
                ) AS clear_plausibility_preference_share,

                AVG(
                    CASE
                        WHEN quantity_plausibility_preferred_side
                            IS NOT NULL
                            THEN 1.0
                        ELSE 0.0
                    END
                ) AS plausibility_first_trigger_share,

                AVG(
                    CASE
                        WHEN quantity_plausibility_preferred_side
                                IS NOT NULL
                         AND quantity_plausibility_first_robust_distance
                                IS NOT NULL
                         AND quantity_reconciled_unit_value_robust_distance
                                IS NOT NULL
                         AND quantity_plausibility_first_robust_distance
                             <
                             quantity_reconciled_unit_value_robust_distance
                            THEN 1.0

                        WHEN quantity_plausibility_preferred_side
                                IS NOT NULL
                         AND quantity_plausibility_first_robust_distance
                                IS NOT NULL
                         AND quantity_reconciled_unit_value_robust_distance
                                IS NOT NULL
                            THEN 0.0

                        ELSE NULL
                    END
                ) AS plausibility_first_improves_share,

                MEDIAN(
                    quantity_plausibility_first_vs_current_fraction
                ) FILTER (
                    WHERE quantity_plausibility_preferred_side
                        IS NOT NULL
                ) AS median_plausibility_first_change,

                                AVG(
                    CASE
                        WHEN trade_synced_quantity_method
                                = 'unit_value_guided_export'
                            THEN 1.0
                        ELSE 0.0
                    END
                ) AS trade_synced_export_share,

                AVG(
                    CASE
                        WHEN trade_synced_quantity_method
                                = 'unit_value_guided_import'
                            THEN 1.0
                        ELSE 0.0
                    END
                ) AS trade_synced_import_share,

                AVG(
                    CASE
                        WHEN trade_synced_quantity_method
                                = 'unit_value_guided_mean'
                            THEN 1.0
                        ELSE 0.0
                    END
                ) AS trade_synced_mean_share,

                AVG(
                    CASE
                        WHEN trade_synced_quantity_method
                                = 'bilateral_mean_no_unit_value_reference'
                            THEN 1.0
                        ELSE 0.0
                    END
                ) AS trade_synced_no_reference_share,
                AVG(
                    CASE
                        WHEN quantity_reference_within_reporter_range
                            THEN 1.0
                        WHEN quantity_reference_within_reporter_range IS NOT NULL
                            THEN 0.0
                        ELSE NULL
                    END
                ) AS reference_between_reporters_share,

                AVG(
                    CASE
                        WHEN quantity_reference_closest_candidate = 'current'
                            THEN 1.0
                        WHEN quantity_reference_closest_candidate IS NOT NULL
                            THEN 0.0
                        ELSE NULL
                    END
                ) AS current_closest_to_reference_share,

                AVG(
                    CASE
                        WHEN quantity_reference_closest_candidate = 'export'
                            THEN 1.0
                        WHEN quantity_reference_closest_candidate IS NOT NULL
                            THEN 0.0
                        ELSE NULL
                    END
                ) AS export_closest_to_reference_share,

                AVG(
                    CASE
                        WHEN quantity_reference_closest_candidate = 'import'
                            THEN 1.0
                        WHEN quantity_reference_closest_candidate IS NOT NULL
                            THEN 0.0
                        ELSE NULL
                    END
                ) AS import_closest_to_reference_share,

                MEDIAN(
                    quantity_reference_vs_current_fraction
                ) FILTER (
                    WHERE quantity_reference_vs_current_fraction IS NOT NULL
                ) AS median_reference_vs_current_fraction

            FROM quantity_reference_classified

            WHERE quantity_relative_difference IS NOT NULL

            GROUP BY
                CASE
                    WHEN quantity_relative_difference <= 0.10
                        THEN '00_10'
                    WHEN quantity_relative_difference <= 0.50
                        THEN '10_50'
                    WHEN quantity_relative_difference <= 0.90
                        THEN '50_90'
                    ELSE '90_100'
                END
        ),

        quantity_year_wide AS (
            SELECT
                year,

                CAST(COUNT(*) AS DOUBLE)
                    AS n_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE quantity_relative_difference IS NOT NULL
                    )
                    AS DOUBLE
                ) AS n_both_quantity_rows,

                CAST(
                    MEDIAN(quantity_relative_difference)
                    AS DOUBLE
                ) AS median_quantity_relative_difference,

                CAST(
                    QUANTILE_CONT(
                        quantity_relative_difference,
                        0.90
                    )
                    AS DOUBLE
                ) AS p90_quantity_relative_difference,

                CAST(
                    MEDIAN(quantity_import_export_ratio)
                    AS DOUBLE
                ) AS median_quantity_import_export_ratio,

                CAST(
                    MEDIAN(quantity_reconciled_vs_mean_fraction)
                    AS DOUBLE
                ) AS median_quantity_reconciled_vs_mean_fraction,

                CAST(
                    QUANTILE_CONT(
                        quantity_reconciled_vs_mean_fraction,
                        0.90
                    )
                    AS DOUBLE
                ) AS p90_quantity_reconciled_vs_mean_fraction,

                CAST(
                    MEDIAN(export_quantity_reliability)
                    AS DOUBLE
                ) AS median_export_quantity_reliability,

                CAST(
                    MEDIAN(import_quantity_reliability)
                    AS DOUBLE
                ) AS median_import_quantity_reliability,

                CAST(
                    AVG(
                        CASE
                            WHEN quantity_relative_difference <= 0.10
                                THEN 1.0
                            WHEN quantity_relative_difference IS NOT NULL
                                THEN 0.0
                        END
                    )
                    AS DOUBLE
                ) AS quantity_within_10pct_share

            FROM plausibility_counterfactual
            GROUP BY year
        ),

        year_wide AS (
            SELECT
                year,

                CAST(COUNT(*) AS DOUBLE)
                    AS n_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE trade_value_reconciled IS NOT NULL
                    )
                    AS DOUBLE
                ) AS n_reconciled_value_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE reconciliation_status = 'not_eligible'
                    )
                    AS DOUBLE
                ) AS n_not_eligible_rows,

                CAST(
                    SUM(trade_value_reconciled) FILTER (
                        WHERE trade_value_reconciled IS NOT NULL
                    )
                    AS DOUBLE
                ) AS total_reconciled_value,

                CAST(
                    MEDIAN(relative_difference_from_larger) FILTER (
                        WHERE reliability_evidence_case = 'both_positive'
                            AND relative_difference_from_larger IS NOT NULL
                    )
                    AS DOUBLE
                ) AS median_relative_difference,

                CAST(
                    MEDIAN(exporter_adjusted_weight) FILTER (
                        WHERE exporter_adjusted_weight IS NOT NULL
                    )
                    AS DOUBLE
                ) AS median_exporter_weight,

                CAST(
                    MEDIAN(importer_adjusted_weight) FILTER (
                        WHERE importer_adjusted_weight IS NOT NULL
                    )
                    AS DOUBLE
                ) AS median_importer_weight,

                CAST(
                    MEDIAN(
                        ABS(reconciled_vs_simple_average_fraction)
                    ) FILTER (
                        WHERE reconciled_vs_simple_average_fraction
                            IS NOT NULL
                    )
                    AS DOUBLE
                ) AS median_absolute_change_from_simple_average,

                CAST(
                    AVG(
                        CASE
                            WHEN within_agreement_tolerance
                                THEN 1.0
                            WHEN within_agreement_tolerance IS NOT NULL
                                THEN 0.0
                            ELSE NULL
                        END
                    )
                    AS DOUBLE
                ) AS within_agreement_tolerance_share

            FROM reconciliation_base

            GROUP BY year
        ),

        year_status_wide AS (
            SELECT
                year,
                reconciliation_status,

                CAST(COUNT(*) AS DOUBLE)
                    AS n_rows,

                CAST(
                    COUNT(*) FILTER (
                        WHERE trade_value_reconciled IS NOT NULL
                    )
                    AS DOUBLE
                ) AS n_reconciled_value_rows,

                CAST(
                    SUM(trade_value_reconciled) FILTER (
                        WHERE trade_value_reconciled IS NOT NULL
                    )
                    AS DOUBLE
                ) AS total_reconciled_value,

                CAST(
                    MEDIAN(trade_value_reconciled) FILTER (
                        WHERE trade_value_reconciled IS NOT NULL
                    )
                    AS DOUBLE
                ) AS median_reconciled_value,

                CAST(
                    MEDIAN(relative_difference_from_larger) FILTER (
                        WHERE relative_difference_from_larger IS NOT NULL
                    )
                    AS DOUBLE
                ) AS median_relative_difference,

                CAST(
                    MEDIAN(exporter_adjusted_weight) FILTER (
                        WHERE exporter_adjusted_weight IS NOT NULL
                    )
                    AS DOUBLE
                ) AS median_exporter_weight,

                CAST(
                    MEDIAN(importer_adjusted_weight) FILTER (
                        WHERE importer_adjusted_weight IS NOT NULL
                    )
                    AS DOUBLE
                ) AS median_importer_weight

            FROM reconciliation_base

            GROUP BY
                year,
                reconciliation_status
        ),

        reliability_wide AS (
            SELECT
                year,
                reporting_direction,

                CAST(COUNT(*) AS DOUBLE)
                    AS n_country_scores,

                CAST(
                    COUNT(*) FILTER (
                        WHERE n_evidence_observations < 10
                    )
                    AS DOUBLE
                ) AS n_scores_below_10_observations,

                CAST(
                    COUNT(*) FILTER (
                        WHERE n_evidence_observations < 30
                    )
                    AS DOUBLE
                ) AS n_scores_below_30_observations,

                CAST(
                    COUNT(*) FILTER (
                        WHERE implicit_zero_share >= 0.90
                    )
                    AS DOUBLE
                ) AS n_scores_at_least_90pct_implicit_zero,

                CAST(
                    MEDIAN(n_evidence_observations)
                    AS DOUBLE
                ) AS median_evidence_observations,

                CAST(
                    QUANTILE_CONT(
                        n_evidence_observations,
                        0.10
                    )
                    AS DOUBLE
                ) AS p10_evidence_observations,

                CAST(
                    MEDIAN(n_distinct_partners)
                    AS DOUBLE
                ) AS median_distinct_partners,

                CAST(
                    MEDIAN(n_distinct_products)
                    AS DOUBLE
                ) AS median_distinct_products,

                CAST(
                    MEDIAN(mean_reliability_deviation)
                    AS DOUBLE
                ) AS median_mean_reliability_deviation,

                CAST(
                    MEDIAN(reliability_index)
                    AS DOUBLE
                ) AS median_reliability_index,

                CAST(
                    MEDIAN(implicit_zero_share)
                    AS DOUBLE
                ) AS median_implicit_zero_share

            FROM reliability

            GROUP BY
                year,
                reporting_direction
        ),

        diagnostics AS (

            -- ==========================================================
            -- Overall metrics
            -- ==========================================================

            SELECT
                'overall'::VARCHAR AS diagnostic_scope,
                NULL::INTEGER AS year,
                NULL::VARCHAR AS reconciliation_status,
                NULL::VARCHAR AS reporting_direction,
                'n_rows'::VARCHAR AS metric,
                n_rows AS metric_value
            FROM overall_wide

            
            UNION ALL

            SELECT
                'quantity_three_candidate_match',
                NULL,
                NULL,
                NULL,
                'n_comparable_rows',
                n_comparable_rows
            FROM quantity_three_candidate_summary_wide

            UNION ALL

            SELECT
                'quantity_three_candidate_match',
                NULL,
                NULL,
                NULL,
                'n_simple_mean_best',
                n_simple_mean_best
            FROM quantity_three_candidate_summary_wide

            UNION ALL

            SELECT
                'quantity_three_candidate_match',
                NULL,
                NULL,
                NULL,
                'n_value_weighted_best',
                n_value_weighted_best
            FROM quantity_three_candidate_summary_wide

            UNION ALL

            SELECT
                'quantity_three_candidate_match',
                NULL,
                NULL,
                NULL,
                'n_quantity_reliability_best',
                n_quantity_reliability_best
            FROM quantity_three_candidate_summary_wide

            UNION ALL

            SELECT
                'quantity_three_candidate_match',
                NULL,
                NULL,
                NULL,
                'n_ties',
                n_ties
            FROM quantity_three_candidate_summary_wide

            UNION ALL

            SELECT
                'quantity_three_candidate_match',
                NULL,
                NULL,
                NULL,
                'simple_mean_best_share',
                simple_mean_best_share
            FROM quantity_three_candidate_summary_wide

            UNION ALL

            SELECT
                'quantity_three_candidate_match',
                NULL,
                NULL,
                NULL,
                'value_weighted_best_share',
                value_weighted_best_share
            FROM quantity_three_candidate_summary_wide

            UNION ALL

            SELECT
                'quantity_three_candidate_match',
                NULL,
                NULL,
                NULL,
                'quantity_reliability_best_share',
                quantity_reliability_best_share
            FROM quantity_three_candidate_summary_wide

            UNION ALL

            SELECT
                'quantity_three_candidate_extreme',
                NULL,
                NULL,
                NULL,
                'n_comparable_rows',
                n_comparable_rows
            FROM quantity_three_candidate_extreme_wide

            UNION ALL

            SELECT
                'quantity_three_candidate_extreme',
                NULL,
                NULL,
                NULL,
                'simple_mean_best_share',
                simple_mean_best_share
            FROM quantity_three_candidate_extreme_wide

            UNION ALL

            SELECT
                'quantity_three_candidate_extreme',
                NULL,
                NULL,
                NULL,
                'value_weighted_best_share',
                value_weighted_best_share
            FROM quantity_three_candidate_extreme_wide

            UNION ALL

            SELECT
                'quantity_three_candidate_extreme',
                NULL,
                NULL,
                NULL,
                'quantity_reliability_best_share',
                quantity_reliability_best_share
            FROM quantity_three_candidate_extreme_wide

            UNION ALL

            SELECT
                'quantity_three_candidate_extreme',
                NULL,
                NULL,
                NULL,
                'tie_share',
                tie_share
            FROM quantity_three_candidate_extreme_wide

            UNION ALL

            SELECT
                'overall',
                NULL,
                NULL,
                NULL,
                'n_reconciled_value_rows',
                n_reconciled_value_rows
            FROM overall_wide

            UNION ALL

            SELECT
                'overall',
                NULL,
                NULL,
                NULL,
                'n_not_eligible_rows',
                n_not_eligible_rows
            FROM overall_wide

            UNION ALL

            SELECT
                'overall',
                NULL,
                NULL,
                NULL,
                'n_reliability_weighted_rows',
                n_reliability_weighted_rows
            FROM overall_wide

            UNION ALL

            SELECT
                'overall',
                NULL,
                NULL,
                NULL,
                'n_single_report_retained_rows',
                n_single_report_retained_rows
            FROM overall_wide

            UNION ALL

            SELECT
                'overall',
                NULL,
                NULL,
                NULL,
                'total_reconciled_value',
                total_reconciled_value
            FROM overall_wide

            UNION ALL

            SELECT
                'overall',
                NULL,
                NULL,
                NULL,
                'total_reported_export_fob',
                total_reported_export_fob
            FROM overall_wide

            UNION ALL

            SELECT
                'overall',
                NULL,
                NULL,
                NULL,
                'total_corrected_import_fob',
                total_corrected_import_fob
            FROM overall_wide

            UNION ALL

            SELECT
                'overall',
                NULL,
                NULL,
                NULL,
                'median_reconciled_value',
                median_reconciled_value
            FROM overall_wide

            UNION ALL

            SELECT
                'overall',
                NULL,
                NULL,
                NULL,
                'median_relative_difference',
                median_relative_difference
            FROM overall_wide

            UNION ALL

            SELECT
                'overall',
                NULL,
                NULL,
                NULL,
                'p90_relative_difference',
                p90_relative_difference
            FROM overall_wide

            UNION ALL

            SELECT
                'overall',
                NULL,
                NULL,
                NULL,
                'median_absolute_change_from_simple_average',
                median_absolute_change_from_simple_average
            FROM overall_wide

            UNION ALL

            SELECT
                'overall',
                NULL,
                NULL,
                NULL,
                'p90_absolute_change_from_simple_average',
                p90_absolute_change_from_simple_average
            FROM overall_wide

            UNION ALL

            SELECT
                'overall',
                NULL,
                NULL,
                NULL,
                'within_agreement_tolerance_share',
                within_agreement_tolerance_share
            FROM overall_wide

            -- ==========================================================
            -- Year metrics
            -- ==========================================================

            UNION ALL

            SELECT
                'year',
                year,
                NULL,
                NULL,
                'n_rows',
                n_rows
            FROM year_wide

            UNION ALL

            SELECT
                'year',
                year,
                NULL,
                NULL,
                'n_reconciled_value_rows',
                n_reconciled_value_rows
            FROM year_wide

            UNION ALL

            SELECT
                'year',
                year,
                NULL,
                NULL,
                'n_not_eligible_rows',
                n_not_eligible_rows
            FROM year_wide

            UNION ALL

            SELECT
                'year',
                year,
                NULL,
                NULL,
                'total_reconciled_value',
                total_reconciled_value
            FROM year_wide

            UNION ALL

            SELECT
                'year',
                year,
                NULL,
                NULL,
                'median_relative_difference',
                median_relative_difference
            FROM year_wide

            UNION ALL

            SELECT
                'year',
                year,
                NULL,
                NULL,
                'median_exporter_weight',
                median_exporter_weight
            FROM year_wide

            UNION ALL

            SELECT
                'year',
                year,
                NULL,
                NULL,
                'median_importer_weight',
                median_importer_weight
            FROM year_wide

            UNION ALL

            SELECT
                'year',
                year,
                NULL,
                NULL,
                'median_absolute_change_from_simple_average',
                median_absolute_change_from_simple_average
            FROM year_wide

            UNION ALL

            SELECT
                'year',
                year,
                NULL,
                NULL,
                'within_agreement_tolerance_share',
                within_agreement_tolerance_share
            FROM year_wide

            -- ==========================================================
            -- Year-status metrics
            -- ==========================================================

            UNION ALL

            SELECT
                'year_status',
                year,
                reconciliation_status,
                NULL,
                'n_rows',
                n_rows
            FROM year_status_wide

            UNION ALL

            SELECT
                'year_status',
                year,
                reconciliation_status,
                NULL,
                'n_reconciled_value_rows',
                n_reconciled_value_rows
            FROM year_status_wide

            UNION ALL

            SELECT
                'year_status',
                year,
                reconciliation_status,
                NULL,
                'total_reconciled_value',
                total_reconciled_value
            FROM year_status_wide

            UNION ALL

            SELECT
                'year_status',
                year,
                reconciliation_status,
                NULL,
                'median_reconciled_value',
                median_reconciled_value
            FROM year_status_wide

            UNION ALL

            SELECT
                'year_status',
                year,
                reconciliation_status,
                NULL,
                'median_relative_difference',
                median_relative_difference
            FROM year_status_wide

            UNION ALL

            SELECT
                'year_status',
                year,
                reconciliation_status,
                NULL,
                'median_exporter_weight',
                median_exporter_weight
            FROM year_status_wide

            UNION ALL

            SELECT
                'year_status',
                year,
                reconciliation_status,
                NULL,
                'median_importer_weight',
                median_importer_weight
            FROM year_status_wide

            -- ==========================================================
            -- Country-reliability metrics
            -- ==========================================================

            UNION ALL

            SELECT
                'country_reliability',
                year,
                NULL,
                reporting_direction,
                'n_country_scores',
                n_country_scores
            FROM reliability_wide

            UNION ALL

            SELECT
                'country_reliability',
                year,
                NULL,
                reporting_direction,
                'n_scores_below_10_observations',
                n_scores_below_10_observations
            FROM reliability_wide

            UNION ALL

            SELECT
                'country_reliability',
                year,
                NULL,
                reporting_direction,
                'n_scores_below_30_observations',
                n_scores_below_30_observations
            FROM reliability_wide

            UNION ALL

            SELECT
                'country_reliability',
                year,
                NULL,
                reporting_direction,
                'n_scores_at_least_90pct_implicit_zero',
                n_scores_at_least_90pct_implicit_zero
            FROM reliability_wide

            UNION ALL

            SELECT
                'country_reliability',
                year,
                NULL,
                reporting_direction,
                'median_evidence_observations',
                median_evidence_observations
            FROM reliability_wide

            UNION ALL

            SELECT
                'country_reliability',
                year,
                NULL,
                reporting_direction,
                'p10_evidence_observations',
                p10_evidence_observations
            FROM reliability_wide

            UNION ALL

            SELECT
                'country_reliability',
                year,
                NULL,
                reporting_direction,
                'median_distinct_partners',
                median_distinct_partners
            FROM reliability_wide

            UNION ALL

            SELECT
                'country_reliability',
                year,
                NULL,
                reporting_direction,
                'median_distinct_products',
                median_distinct_products
            FROM reliability_wide

            UNION ALL

            SELECT
                'country_reliability',
                year,
                NULL,
                reporting_direction,
                'median_mean_reliability_deviation',
                median_mean_reliability_deviation
            FROM reliability_wide

            UNION ALL

            SELECT
                'country_reliability',
                year,
                NULL,
                reporting_direction,
                'median_reliability_index',
                median_reliability_index
            FROM reliability_wide

            UNION ALL

            SELECT
                'country_reliability',
                year,
                NULL,
                reporting_direction,
                'median_implicit_zero_share',
                median_implicit_zero_share
            FROM reliability_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'n_export_quantity_rows',
                n_export_quantity_rows
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'n_import_quantity_rows',
                n_import_quantity_rows
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'n_both_quantity_rows',
                n_both_quantity_rows
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'n_export_quantity_only_rows',
                n_export_quantity_only_rows
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'n_import_quantity_only_rows',
                n_import_quantity_only_rows
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'n_no_quantity_rows',
                n_no_quantity_rows
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'n_quantity_reliability_weighted',
                n_quantity_reliability_weighted
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'n_quantity_simple_mean_fallback',
                n_quantity_simple_mean_fallback
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'n_quantity_export_retained',
                n_quantity_export_retained
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'n_quantity_import_retained',
                n_quantity_import_retained
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'n_quantity_missing',
                n_quantity_missing
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'median_quantity_reconciled_vs_mean_fraction',
                median_quantity_reconciled_vs_mean_fraction
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'p90_quantity_reconciled_vs_mean_fraction',
                p90_quantity_reconciled_vs_mean_fraction
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'median_quantity_value_weighted_vs_mean_fraction',
                median_quantity_value_weighted_vs_mean_fraction
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'p90_quantity_value_weighted_vs_mean_fraction',
                p90_quantity_value_weighted_vs_mean_fraction
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'median_quantity_trade_synced_vs_mean_fraction',
                median_quantity_trade_synced_vs_mean_fraction
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'p90_quantity_trade_synced_vs_mean_fraction',
                p90_quantity_trade_synced_vs_mean_fraction
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'median_quantity_reliability_vs_value_weighted_fraction',
                median_quantity_reliability_vs_value_weighted_fraction
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'p90_quantity_reliability_vs_value_weighted_fraction',
                p90_quantity_reliability_vs_value_weighted_fraction
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'median_export_quantity_reliability',
                median_export_quantity_reliability
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'median_import_quantity_reliability',
                median_import_quantity_reliability
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'median_quantity_reliability_gap',
                median_quantity_reliability_gap
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'median_quantity_relative_difference',
                median_quantity_relative_difference
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'p90_quantity_relative_difference',
                p90_quantity_relative_difference
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'p75_quantity_relative_difference',
                p75_quantity_relative_difference
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'median_quantity_absolute_log_gap',
                median_quantity_absolute_log_gap
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'p95_quantity_relative_difference',
                p95_quantity_relative_difference
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'p99_quantity_relative_difference',
                p99_quantity_relative_difference
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'median_quantity_import_export_ratio',
                median_quantity_import_export_ratio
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'quantity_within_5pct_share',
                quantity_within_5pct_share
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'quantity_within_10pct_share',
                quantity_within_10pct_share
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'quantity_within_20pct_share',
                quantity_within_20pct_share
            FROM quantity_overall_wide

            UNION ALL

            SELECT
                'quantity_overall',
                NULL,
                NULL,
                NULL,
                'quantity_within_50pct_share',
                quantity_within_50pct_share
            FROM quantity_overall_wide

            -- ==========================================================
            -- Quantity plausibility metrics
            -- ==========================================================

            UNION ALL

            SELECT
                'quantity_plausibility',
                NULL,
                NULL,
                NULL,
                'n_plausibility_comparable_rows',
                n_plausibility_comparable_rows
            FROM quantity_plausibility_wide

            UNION ALL

            SELECT
                'quantity_plausibility',
                NULL,
                NULL,
                NULL,
                'export_more_plausible_share',
                export_more_plausible_share
            FROM quantity_plausibility_wide

            UNION ALL

            SELECT
                'quantity_plausibility',
                NULL,
                NULL,
                NULL,
                'import_more_plausible_share',
                import_more_plausible_share
            FROM quantity_plausibility_wide

            UNION ALL

            SELECT
                'quantity_plausibility',
                NULL,
                NULL,
                NULL,
                'n_extreme_quantity_disagreement',
                n_extreme_quantity_disagreement
            FROM quantity_extreme_plausibility_wide

            UNION ALL

            SELECT
                'quantity_plausibility',
                NULL,
                NULL,
                NULL,
                'n_extreme_with_plausibility_evidence',
                n_extreme_with_plausibility_evidence
            FROM quantity_extreme_plausibility_wide

            UNION ALL

            SELECT
                'quantity_plausibility',
                NULL,
                NULL,
                NULL,
                'extreme_clear_plausibility_preference_share',
                extreme_clear_plausibility_preference_share
            FROM quantity_extreme_plausibility_wide


            -- ==========================================================
            -- Final Trade-Synced quantity metrics
            -- ==========================================================

            UNION ALL

            SELECT
                'quantity_trade_sync',
                NULL,
                NULL,
                NULL,
                'n_trade_synced_quantity_rows',
                n_trade_synced_quantity_rows
            FROM quantity_trade_sync_wide

            UNION ALL

            SELECT
                'quantity_trade_sync',
                NULL,
                NULL,
                NULL,
                'n_unit_value_guided_export',
                n_unit_value_guided_export
            FROM quantity_trade_sync_wide

            UNION ALL

            SELECT
                'quantity_trade_sync',
                NULL,
                NULL,
                NULL,
                'n_unit_value_guided_import',
                n_unit_value_guided_import
            FROM quantity_trade_sync_wide

            UNION ALL

            SELECT
                'quantity_trade_sync',
                NULL,
                NULL,
                NULL,
                'n_unit_value_guided_mean',
                n_unit_value_guided_mean
            FROM quantity_trade_sync_wide

            UNION ALL

            SELECT
                'quantity_trade_sync',
                NULL,
                NULL,
                NULL,
                'n_bilateral_mean_no_reference',
                n_bilateral_mean_no_reference
            FROM quantity_trade_sync_wide

            UNION ALL

            SELECT
                'quantity_trade_sync',
                NULL,
                NULL,
                NULL,
                'n_export_quantity_retained',
                n_export_quantity_retained
            FROM quantity_trade_sync_wide

            UNION ALL

            SELECT
                'quantity_trade_sync',
                NULL,
                NULL,
                NULL,
                'n_import_quantity_retained',
                n_import_quantity_retained
            FROM quantity_trade_sync_wide

            UNION ALL

            SELECT
                'quantity_trade_sync',
                NULL,
                NULL,
                NULL,
                'n_no_usable_quantity',
                n_no_usable_quantity
            FROM quantity_trade_sync_wide

            UNION ALL

            SELECT
                'quantity_trade_sync',
                NULL,
                NULL,
                NULL,
                'unit_value_guided_export_share',
                unit_value_guided_export_share
            FROM quantity_trade_sync_wide

            UNION ALL

            SELECT
                'quantity_trade_sync',
                NULL,
                NULL,
                NULL,
                'unit_value_guided_import_share',
                unit_value_guided_import_share
            FROM quantity_trade_sync_wide

            UNION ALL

            SELECT
                'quantity_trade_sync',
                NULL,
                NULL,
                NULL,
                'unit_value_guided_mean_share',
                unit_value_guided_mean_share
            FROM quantity_trade_sync_wide

            UNION ALL

            SELECT
                'quantity_trade_sync',
                NULL,
                NULL,
                NULL,
                'bilateral_mean_no_reference_share',
                bilateral_mean_no_reference_share
            FROM quantity_trade_sync_wide


            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'trade_synced_export_share',
                trade_synced_export_share
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'trade_synced_import_share',
                trade_synced_import_share
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'trade_synced_mean_share',
                trade_synced_mean_share
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'trade_synced_no_reference_share',
                trade_synced_no_reference_share
            FROM quantity_disagreement_band_wide

            -- ==========================================================
            -- Plausibility-first counterfactual metrics
            -- ==========================================================

            UNION ALL

            SELECT
                'quantity_plausibility_first',
                NULL,
                NULL,
                NULL,
                'n_plausibility_decisive_rows',
                n_plausibility_decisive_rows
            FROM quantity_plausibility_first_wide

            UNION ALL

            SELECT
                'quantity_plausibility_first',
                NULL,
                NULL,
                NULL,
                'plausibility_decisive_share',
                plausibility_decisive_share
            FROM quantity_plausibility_first_wide

            UNION ALL

            SELECT
                'quantity_plausibility_first',
                NULL,
                NULL,
                NULL,
                'n_plausibility_export_selected',
                n_plausibility_export_selected
            FROM quantity_plausibility_first_wide

            UNION ALL

            SELECT
                'quantity_plausibility_first',
                NULL,
                NULL,
                NULL,
                'n_plausibility_import_selected',
                n_plausibility_import_selected
            FROM quantity_plausibility_first_wide

            UNION ALL

            SELECT
                'quantity_plausibility_first',
                NULL,
                NULL,
                NULL,
                'median_plausibility_first_change',
                median_plausibility_first_change
            FROM quantity_plausibility_first_wide

            UNION ALL

            SELECT
                'quantity_plausibility_first',
                NULL,
                NULL,
                NULL,
                'p90_plausibility_first_change',
                p90_plausibility_first_change
            FROM quantity_plausibility_first_wide

            UNION ALL

            SELECT
                'quantity_plausibility_first',
                NULL,
                NULL,
                NULL,
                'plausibility_first_improves_share',
                plausibility_first_improves_share
            FROM quantity_plausibility_first_wide

            UNION ALL

            SELECT
                'quantity_plausibility_first',
                NULL,
                NULL,
                NULL,
                'median_robust_distance_improvement',
                median_robust_distance_improvement
            FROM quantity_plausibility_first_wide


            UNION ALL

            SELECT
                'quantity_reference',
                NULL,
                NULL,
                NULL,
                'n_reference_comparable_rows',
                n_reference_comparable_rows
            FROM quantity_reference_wide

            UNION ALL

            SELECT
                'quantity_reference',
                NULL,
                NULL,
                NULL,
                'reference_between_reporters_share',
                reference_between_reporters_share
            FROM quantity_reference_wide

            UNION ALL

            SELECT
                'quantity_reference',
                NULL,
                NULL,
                NULL,
                'current_closest_to_reference_share',
                current_closest_to_reference_share
            FROM quantity_reference_wide

            UNION ALL

            SELECT
                'quantity_reference',
                NULL,
                NULL,
                NULL,
                'export_closest_to_reference_share',
                export_closest_to_reference_share
            FROM quantity_reference_wide

            UNION ALL

            SELECT
                'quantity_reference',
                NULL,
                NULL,
                NULL,
                'import_closest_to_reference_share',
                import_closest_to_reference_share
            FROM quantity_reference_wide

            UNION ALL

            SELECT
                'quantity_reference',
                NULL,
                NULL,
                NULL,
                'median_reference_vs_current_fraction',
                median_reference_vs_current_fraction
            FROM quantity_reference_wide

            UNION ALL

            SELECT
                'quantity_reference',
                NULL,
                NULL,
                NULL,
                'p90_reference_vs_current_fraction',
                p90_reference_vs_current_fraction
            FROM quantity_reference_wide

            UNION ALL

            SELECT
                'quantity_reference',
                NULL,
                NULL,
                NULL,
                'median_current_reference_log_distance',
                median_current_reference_log_distance
            FROM quantity_reference_wide

            UNION ALL

            SELECT
                'quantity_reference',
                NULL,
                NULL,
                NULL,
                'median_export_reference_log_distance',
                median_export_reference_log_distance
            FROM quantity_reference_wide

            UNION ALL

            SELECT
                'quantity_reference',
                NULL,
                NULL,
                NULL,
                'median_import_reference_log_distance',
                median_import_reference_log_distance
            FROM quantity_reference_wide

            UNION ALL

            SELECT
                'quantity_reference',
                NULL,
                NULL,
                NULL,
                'median_reference_observations',
                median_reference_observations
            FROM quantity_reference_wide

            -- ==========================================================
            -- Quantity disagreement-band metrics
            -- ==========================================================

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'n_rows',
                n_rows
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'median_quantity_reliability_gap',
                median_quantity_reliability_gap
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'median_reconciled_vs_mean_fraction',
                median_reconciled_vs_mean_fraction
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'median_value_weighted_vs_mean_fraction',
                median_value_weighted_vs_mean_fraction
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'median_quantity_reliability_vs_mean_fraction',
                median_quantity_reliability_vs_mean_fraction
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'median_trade_synced_vs_mean_fraction',
                median_trade_synced_vs_mean_fraction
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'plausibility_evidence_share',
                plausibility_evidence_share
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'export_more_plausible_share',
                export_more_plausible_share
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'import_more_plausible_share',
                import_more_plausible_share
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'clear_plausibility_preference_share',
                clear_plausibility_preference_share
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'plausibility_first_trigger_share',
                plausibility_first_trigger_share
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'plausibility_first_improves_share',
                plausibility_first_improves_share
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'median_plausibility_first_change',
                median_plausibility_first_change
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'reference_between_reporters_share',
                reference_between_reporters_share
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'current_closest_to_reference_share',
                current_closest_to_reference_share
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'export_closest_to_reference_share',
                export_closest_to_reference_share
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'import_closest_to_reference_share',
                import_closest_to_reference_share
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_disagreement_band',
                NULL,
                quantity_disagreement_band,
                NULL,
                'median_reference_vs_current_fraction',
                median_reference_vs_current_fraction
            FROM quantity_disagreement_band_wide

            UNION ALL

            SELECT
                'quantity_year',
                year,
                NULL,
                NULL,
                'n_both_quantity_rows',
                n_both_quantity_rows
            FROM quantity_year_wide

            UNION ALL

            SELECT
                'quantity_year',
                year,
                NULL,
                NULL,
                'median_quantity_relative_difference',
                median_quantity_relative_difference
            FROM quantity_year_wide

            UNION ALL

            SELECT
                'quantity_year',
                year,
                NULL,
                NULL,
                'p90_quantity_relative_difference',
                p90_quantity_relative_difference
            FROM quantity_year_wide

            UNION ALL

            SELECT
                'quantity_year',
                year,
                NULL,
                NULL,
                'median_quantity_import_export_ratio',
                median_quantity_import_export_ratio
            FROM quantity_year_wide

            UNION ALL

            SELECT
                'quantity_year',
                year,
                NULL,
                NULL,
                'median_quantity_reconciled_vs_mean_fraction',
                median_quantity_reconciled_vs_mean_fraction
            FROM quantity_year_wide

            UNION ALL

            SELECT
                'quantity_year',
                year,
                NULL,
                NULL,
                'p90_quantity_reconciled_vs_mean_fraction',
                p90_quantity_reconciled_vs_mean_fraction
            FROM quantity_year_wide

            UNION ALL

            SELECT
                'quantity_year',
                year,
                NULL,
                NULL,
                'median_export_quantity_reliability',
                median_export_quantity_reliability
            FROM quantity_year_wide

            UNION ALL

            SELECT
                'quantity_year',
                year,
                NULL,
                NULL,
                'median_import_quantity_reliability',
                median_import_quantity_reliability
            FROM quantity_year_wide

            UNION ALL

            SELECT
                'quantity_year',
                year,
                NULL,
                NULL,
                'quantity_within_10pct_share',
                quantity_within_10pct_share
            FROM quantity_year_wide

                    UNION ALL

            SELECT
                'quantity_original_pair',
                NULL,
                NULL,
                NULL,
                'n_original_pair_rows',
                n_original_pair_rows
            FROM quantity_original_pair_wide

            UNION ALL

            SELECT
                'quantity_original_pair',
                NULL,
                NULL,
                NULL,
                'export_closer_to_reference_share',
                export_closer_to_reference_share
            FROM quantity_original_pair_wide

            UNION ALL

            SELECT
                'quantity_original_pair',
                NULL,
                NULL,
                NULL,
                'import_closer_to_reference_share',
                import_closer_to_reference_share
            FROM quantity_original_pair_wide

            UNION ALL

            SELECT
                'quantity_original_pair',
                NULL,
                NULL,
                NULL,
                'clear_reference_preference_share',
                clear_reference_preference_share
            FROM quantity_original_pair_wide

            UNION ALL

            SELECT
                'quantity_original_pair',
                NULL,
                NULL,
                NULL,
                'median_original_unit_value_log_gap',
                median_original_unit_value_log_gap
            FROM quantity_original_pair_wide

            UNION ALL

            SELECT
                'quantity_candidate_overall',
                NULL,
                candidate_method,
                NULL,
                'n_rows',
                n_rows
            FROM quantity_candidate_overall_wide

            UNION ALL

            SELECT
                'quantity_candidate_overall',
                NULL,
                candidate_method,
                NULL,
                'reporter_unit_value_range_share',
                reporter_unit_value_range_share
            FROM quantity_candidate_overall_wide

            UNION ALL

            SELECT
                'quantity_candidate_overall',
                NULL,
                candidate_method,
                NULL,
                'median_outside_reporter_log_distance',
                median_outside_reporter_log_distance
            FROM quantity_candidate_overall_wide

            UNION ALL

            SELECT
                'quantity_candidate_overall',
                NULL,
                candidate_method,
                NULL,
                'median_reference_robust_distance',
                median_reference_robust_distance
            FROM quantity_candidate_overall_wide

            UNION ALL

            SELECT
                'quantity_candidate_overall',
                NULL,
                candidate_method,
                NULL,
                'p90_reference_robust_distance',
                p90_reference_robust_distance
            FROM quantity_candidate_overall_wide

            UNION ALL

            SELECT
                'quantity_candidate_overall',
                NULL,
                candidate_method,
                NULL,
                'median_nearest_report_quantity_fraction',
                median_nearest_report_quantity_fraction
            FROM quantity_candidate_overall_wide

            UNION ALL

            SELECT
                'quantity_candidate_overall',
                NULL,
                candidate_method,
                NULL,
                'p90_nearest_report_quantity_fraction',
                p90_nearest_report_quantity_fraction
            FROM quantity_candidate_overall_wide

                        UNION ALL

            SELECT
                'quantity_candidate_extreme',
                NULL,
                candidate_method,
                NULL,
                'n_rows',
                n_rows
            FROM quantity_candidate_extreme_wide

            UNION ALL

            SELECT
                'quantity_candidate_extreme',
                NULL,
                candidate_method,
                NULL,
                'reporter_unit_value_range_share',
                reporter_unit_value_range_share
            FROM quantity_candidate_extreme_wide

            UNION ALL

            SELECT
                'quantity_candidate_extreme',
                NULL,
                candidate_method,
                NULL,
                'median_reference_robust_distance',
                median_reference_robust_distance
            FROM quantity_candidate_extreme_wide

            UNION ALL

            SELECT
                'quantity_candidate_extreme',
                NULL,
                candidate_method,
                NULL,
                'median_nearest_report_quantity_fraction',
                median_nearest_report_quantity_fraction
            FROM quantity_candidate_extreme_wide
        )

        SELECT *
        FROM diagnostics
        """

        # ==============================================================
        # WRITE DIAGNOSTICS
        # ==============================================================

        con.execute(
            f"""
            COPY (
                {diagnostics_query}
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

        output_audit_row = con.execute(
            f"""
            WITH output AS (
                SELECT *
                FROM read_parquet({output_sql})
            ),

            duplicate_keys AS (
                SELECT
                    diagnostic_scope,
                    year,
                    reconciliation_status,
                    reporting_direction,
                    metric,
                    COUNT(*) AS n_rows

                FROM output

                GROUP BY
                    diagnostic_scope,
                    year,
                    reconciliation_status,
                    reporting_direction,
                    metric

                HAVING COUNT(*) > 1
            )

            SELECT
                (
                    SELECT COUNT(*)
                    FROM output
                ) AS n_diagnostic_rows,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_diagnostic_keys,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE diagnostic_scope IS NULL
                       OR metric IS NULL
                ) AS n_missing_diagnostic_keys,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE metric_value IS NOT NULL
                      AND NOT isfinite(metric_value)
                ) AS n_nonfinite_metric_values,

                (
                    SELECT COUNT(DISTINCT diagnostic_scope)
                    FROM output
                ) AS n_diagnostic_scopes,

                (
                    SELECT COUNT(DISTINCT metric)
                    FROM output
                ) AS n_distinct_metrics,

                (
                    SELECT metric_value
                    FROM output
                    WHERE diagnostic_scope = 'overall'
                      AND metric = 'n_rows'
                ) AS diagnostic_input_rows,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({reconciled_sql})
                ) AS reconciled_input_rows,

                (
                    SELECT metric_value
                    FROM output
                    WHERE diagnostic_scope = 'overall'
                      AND metric = 'n_not_eligible_rows'
                ) AS overall_not_eligible_rows,

                (
                    SELECT metric_value
                    FROM output
                    WHERE diagnostic_scope = 'overall'
                      AND metric = 'n_reliability_weighted_rows'
                ) AS overall_weighted_rows,

                (
                    SELECT metric_value
                    FROM output
                    WHERE diagnostic_scope = 'overall'
                      AND metric
                        = 'n_single_report_retained_rows'
                ) AS overall_single_report_rows,

                (
                    SELECT metric_value
                    FROM output
                    WHERE diagnostic_scope = 'overall'
                      AND metric = 'total_reconciled_value'
                ) AS overall_total_reconciled_value,

                (
                    SELECT metric_value
                    FROM output
                    WHERE diagnostic_scope = 'quantity_overall'
                      AND metric = 'n_both_quantity_rows'
                ) AS overall_both_quantity_rows

            """
        ).fetchone()

        if output_audit_row is None:
            raise RuntimeError(
                "Could not inspect Trade Sync diagnostics output."
            )

        (
            n_diagnostic_rows,
            n_duplicate_diagnostic_keys,
            n_missing_diagnostic_keys,
            n_nonfinite_metric_values,
            n_diagnostic_scopes,
            n_distinct_metrics,
            diagnostic_input_rows,
            reconciled_input_rows,
            overall_not_eligible_rows,
            overall_weighted_rows,
            overall_single_report_rows,
            overall_total_reconciled_value,
            overall_both_quantity_rows,
        ) = output_audit_row

        if int(n_diagnostic_rows) == 0:
            raise RuntimeError(
                "Trade Sync diagnostics produced no rows."
            )

        if int(n_duplicate_diagnostic_keys) > 0:
            raise RuntimeError(
                "Trade Sync diagnostics contain "
                f"{int(n_duplicate_diagnostic_keys):,} duplicate keys."
            )

        if int(n_missing_diagnostic_keys) > 0:
            raise RuntimeError(
                "Trade Sync diagnostics contain "
                f"{int(n_missing_diagnostic_keys):,} incomplete keys."
            )

        if int(n_nonfinite_metric_values) > 0:
            raise RuntimeError(
                "Trade Sync diagnostics contain "
                f"{int(n_nonfinite_metric_values):,} non-finite metrics."
            )

        actual_diagnostic_scopes = {
            str(row[0])
            for row in con.execute(
                f"""
                SELECT DISTINCT diagnostic_scope
                FROM read_parquet({output_sql})
                WHERE diagnostic_scope IS NOT NULL
                """
            ).fetchall()
        }

        expected_diagnostic_scopes = {
            "overall",
            "year",
            "year_status",
            "country_reliability",
            "quantity_overall",
            "quantity_plausibility",
            "quantity_plausibility_first",
            "quantity_disagreement_band",
            "quantity_reference",
            "quantity_trade_sync",
            "quantity_year",
            "quantity_original_pair",
            "quantity_candidate_overall",
            "quantity_candidate_extreme",
            "quantity_three_candidate_match",
            "quantity_three_candidate_extreme",
        }

        missing_scopes = (
            expected_diagnostic_scopes
            - actual_diagnostic_scopes
        )

        unexpected_scopes = (
            actual_diagnostic_scopes
            - expected_diagnostic_scopes
        )

        if missing_scopes or unexpected_scopes:
            raise RuntimeError(
                "Unexpected Trade Sync diagnostic scopes. "
                f"Missing: {sorted(missing_scopes)}; "
                f"Unexpected: {sorted(unexpected_scopes)}."
            )

        if overall_both_quantity_rows is None:
            raise RuntimeError(
                "Quantity diagnostics did not produce the bilateral "
                "quantity coverage metric."
            )

        if int(round(float(overall_both_quantity_rows))) <= 0:
            raise RuntimeError(
                "No rows with usable exporter and importer quantities "
                "were available for quantity diagnostics."
            )

        if int(round(float(diagnostic_input_rows))) != int(
            reconciled_input_rows
        ):
            raise RuntimeError(
                "Diagnostics row count does not match reconciled input: "
                f"diagnostics={int(round(float(diagnostic_input_rows))):,}, "
                f"reconciled={int(reconciled_input_rows):,}."
            )

        audit_passed = True
        ready_for_decision_report = True

        write_local_checkpoint(
            checkpoint_path=checkpoint_path,
            step_name="diagnostics",
            run_id=run_id,
            config=config_fp,
            input_fingerprint=input_fp,
            output_path=output_path,
        )

        logger.info(
            "%s Trade Sync diagnostics created.",
            ok,
        )
        logger.info(
            "  Reconciled input rows: %s",
            f"{int(reconciled_input_rows):,}",
        )
        logger.info(
            "  Diagnostic rows: %s",
            f"{int(n_diagnostic_rows):,}",
        )
        logger.info(
            "  Diagnostic scopes: %s",
            f"{int(n_diagnostic_scopes):,}",
        )
        logger.info(
            "  Distinct metrics: %s",
            f"{int(n_distinct_metrics):,}",
        )
        logger.info(
            "  Reliability-weighted rows: %s",
            f"{int(round(float(overall_weighted_rows))):,}",
        )
        logger.info(
            "  Single-report rows retained: %s",
            f"{int(round(float(overall_single_report_rows))):,}",
        )
        logger.info(
            "  Not-eligible rows: %s",
            f"{int(round(float(overall_not_eligible_rows))):,}",
        )
        logger.info(
            "  Total reconciled value: %.6f",
            float(overall_total_reconciled_value or 0.0),
        )
        logger.info(
            "  Rows with both usable quantities: %s",
            f"{int(round(float(overall_both_quantity_rows))):,}",
        )
        logger.info(
            "  Agreement-tolerance diagnostics available: %s",
            tolerance_available,
        )
        logger.info(
            "  Output: %s",
            output_path,
        )
        logger.info(
            "  Step audit passed: %s",
            audit_passed,
        )
        logger.info(
            "  Ready for decision report: %s",
            ready_for_decision_report,
        )

        return {
            "status": "completed",
            "output_path": output_path,
            "diagnostics_path": output_path,
            "audit_passed": audit_passed,
            "ready_for_decision_report": ready_for_decision_report,
            "n_reconciled_input_rows": int(reconciled_input_rows),
            "n_diagnostic_rows": int(n_diagnostic_rows),
            "n_diagnostic_scopes": int(n_diagnostic_scopes),
            "n_distinct_metrics": int(n_distinct_metrics),
            "n_reliability_weighted_rows": int(
                round(float(overall_weighted_rows))
            ),
            "n_single_report_retained_rows": int(
                round(float(overall_single_report_rows))
            ),
            "n_not_eligible_rows": int(
                round(float(overall_not_eligible_rows))
            ),
            "total_reconciled_value": float(
                overall_total_reconciled_value or 0.0
            ),
            "agreement_tolerance_enabled": (
                agreement_tolerance_enabled
            ),
            "agreement_tolerance_available": tolerance_available,
            "agreement_tolerance_relative_difference": (
                float(agreement_tolerance)
                if agreement_tolerance is not None
                else None
            ),
        }

    finally:
        con.close()