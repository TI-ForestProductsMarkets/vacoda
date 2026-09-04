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
- plausibility evidence under extreme quantity disagreement.
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

            FROM reconciliation_base
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

            FROM reconciliation_base
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

            FROM reconciliation_base
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
                ) AS clear_plausibility_preference_share

            FROM reconciliation_base

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

            FROM reconciliation_base
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

        if int(n_diagnostic_scopes) != 8:
            raise RuntimeError(
                "Trade Sync diagnostics did not produce all expected "
                "scopes: overall, year, year_status, country_reliability, "
                "quantity_overall, quantity_plausibility, "
                "quantity_disagreement_band, and quantity_year."
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