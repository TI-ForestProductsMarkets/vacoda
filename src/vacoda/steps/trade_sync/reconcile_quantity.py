"""
src.vacoda.steps.trade_sync.quantity_reconciliation

Bilateral quantity reconciliation for Trade Sync.

This stage runs after trade-value reconciliation has passed its audit.

The reconciled trade value is treated as the common reference value for
evaluating exporter- and importer-reported kg quantities.

The stage:

1. classifies bilateral quantity availability;
2. measures exporter/importer quantity disagreement;
3. estimates quantity-reporting reliability by
   country × year × reporting direction;
4. derives product-year unit-value plausibility evidence using the
   reconciled trade value as the common value anchor;
5. creates simple-mean and value-reliability-weighted quantity candidates;
6. produces a transparent first-pass synchronized quantity.

The evidence columns are intentionally retained in the output so that the
final reconciliation rule can be audited and refined without rebuilding the
underlying evidence logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.utils.sql import sql_literal
from vacoda.config.steps.trade_sync import TradeSyncConfig


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
            "Quantity reconciliation input is missing required columns: "
            + ", ".join(missing_columns)
        )


# ======================================================================
# Main stage
# ======================================================================


def reconcile_bilateral_quantity(
    *,
    cfg: TradeSyncConfig,
    reconciled_trade_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Build bilateral quantity reconciliation evidence and synchronized quantity.

    Parameters
    ----------
    cfg
        Trade Sync configuration.

    reconciled_trade_path
        Audited Trade Sync reconciled-value Parquet.

    run_id
        Pipeline run identifier.

    force
        Rebuild even when the configured output already exists.

    logger
        Pipeline logger.

    Returns
    -------
    dict
        Stage metadata including the reconciled-quantity Parquet path.
    """

    reconciled_trade_path = Path(
        reconciled_trade_path
    )

    if not reconciled_trade_path.is_file():
        raise FileNotFoundError(
            "Audited reconciled Trade Sync input not found: "
            f"{reconciled_trade_path}"
        )

    output_path = cfg.product_path(
        "reconciled_quantity"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        not force
        and _valid_output(output_path)
    ):
        logger.info(
            "🔄 REUSE — Reconciled bilateral quantity: %s",
            output_path,
        )

        return {
            "status": "reused",
            "run_id": run_id,
            "reconciled_quantity_path": output_path,
            "output_path": output_path,
        }

    logger.info(
        "🔨 REBUILD — Bilateral quantity reconciliation."
    )

    input_sql = sql_literal(
        reconciled_trade_path
    )
    output_sql = sql_literal(
        output_path
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        # --------------------------------------------------------------
        # Required input contract
        # --------------------------------------------------------------

        required_columns = {
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",
            "trade_value_reconciled",
            "reconciliation_status",
            "export_synchronized_quantity_kg",
            "import_synchronized_quantity_kg",
            "exporter_adjusted_weight",
            "importer_adjusted_weight",
        }

        _require_columns(
            con=con,
            parquet_path=reconciled_trade_path,
            required_columns=required_columns,
        )

        # --------------------------------------------------------------
        # Build reconciled quantity dataset
        # --------------------------------------------------------------
        #
        # Important:
        #
        # Quantity reliability is estimated only from genuine bilateral
        # positive kg observations.
        #
        # Unit-value plausibility uses the already reconciled trade value
        # as the common value anchor for BOTH candidate quantities.
        #
        # The product-year plausibility reference is based on pooled
        # candidate unit values and uses log-space median/MAD statistics
        # for robustness against extreme outliers.
        # --------------------------------------------------------------

        con.execute(
            f"""
            COPY (

                WITH
                -- =====================================================
                -- 1. Source rows and usable-quantity flags
                -- =====================================================

                source AS (
                    SELECT
                        *,

                        (
                            export_synchronized_quantity_kg IS NOT NULL
                            AND isfinite(
                                export_synchronized_quantity_kg
                            )
                            AND export_synchronized_quantity_kg > 0
                        ) AS has_export_quantity_kg,

                        (
                            import_synchronized_quantity_kg IS NOT NULL
                            AND isfinite(
                                import_synchronized_quantity_kg
                            )
                            AND import_synchronized_quantity_kg > 0
                        ) AS has_import_quantity_kg

                    FROM read_parquet({input_sql})
                ),

                -- =====================================================
                -- 2. Bilateral quantity evidence
                -- =====================================================

                bilateral AS (
                    SELECT
                        *,

                        CASE
                            WHEN has_export_quantity_kg
                             AND has_import_quantity_kg
                                THEN 'both_quantity'

                            WHEN has_export_quantity_kg
                             AND NOT has_import_quantity_kg
                                THEN 'export_only'

                            WHEN has_import_quantity_kg
                             AND NOT has_export_quantity_kg
                                THEN 'import_only'

                            ELSE 'no_quantity'
                        END AS quantity_availability_case,

                        CASE
                            WHEN has_export_quantity_kg
                             AND has_import_quantity_kg
                                THEN (
                                    ABS(
                                        export_synchronized_quantity_kg
                                        - import_synchronized_quantity_kg
                                    )
                                    /
                                    GREATEST(
                                        export_synchronized_quantity_kg,
                                        import_synchronized_quantity_kg
                                    )
                                )
                            ELSE NULL
                        END AS quantity_relative_difference,

                        CASE
                            WHEN has_export_quantity_kg
                             AND has_import_quantity_kg
                                THEN (
                                    import_synchronized_quantity_kg
                                    /
                                    export_synchronized_quantity_kg
                                )
                            ELSE NULL
                        END AS quantity_import_export_ratio,

                        CASE
                            WHEN has_export_quantity_kg
                             AND has_import_quantity_kg
                                THEN ABS(
                                    LN(
                                        import_synchronized_quantity_kg
                                    )
                                    -
                                    LN(
                                        export_synchronized_quantity_kg
                                    )
                                )
                            ELSE NULL
                        END AS quantity_absolute_log_gap,

                        CASE
                            WHEN has_export_quantity_kg
                             AND has_import_quantity_kg
                                THEN (
                                    export_synchronized_quantity_kg
                                    + import_synchronized_quantity_kg
                                ) / 2.0
                            ELSE NULL
                        END AS quantity_simple_mean_kg,

                        CASE
                            -- Bilateral value reconciliation:
                            -- use the same reporter weights as trade-value reconciliation.
                            WHEN has_export_quantity_kg
                            AND has_import_quantity_kg
                            AND reconciliation_status IN (
                                'bilateral_reconciled',
                                'bilateral_reconciled_import_implicit_zero',
                                'bilateral_reconciled_export_implicit_zero'
                            )
                            AND exporter_adjusted_weight IS NOT NULL
                            AND importer_adjusted_weight IS NOT NULL
                            AND isfinite(exporter_adjusted_weight)
                            AND isfinite(importer_adjusted_weight)
                                THEN (
                                    export_synchronized_quantity_kg
                                    * exporter_adjusted_weight
                                    +
                                    import_synchronized_quantity_kg
                                    * importer_adjusted_weight
                                )

                            -- Value reconciliation retained exporter report only.
                            -- Reporter influence on the value side is effectively 1 / 0.
                            WHEN has_export_quantity_kg
                            AND has_import_quantity_kg
                            AND reconciliation_status = 'export_only'
                                THEN export_synchronized_quantity_kg

                            -- Value reconciliation retained importer report only.
                            -- Reporter influence on the value side is effectively 0 / 1.
                            WHEN has_export_quantity_kg
                            AND has_import_quantity_kg
                            AND reconciliation_status = 'import_only'
                                THEN import_synchronized_quantity_kg

                            ELSE NULL
                        END AS quantity_value_weighted_kg,

                        CASE
                            WHEN trade_value_reconciled IS NOT NULL
                             AND isfinite(
                                 trade_value_reconciled
                             )
                             AND trade_value_reconciled > 0
                             AND has_export_quantity_kg
                                THEN (
                                    trade_value_reconciled
                                    /
                                    export_synchronized_quantity_kg
                                )
                            ELSE NULL
                        END AS export_implied_unit_value,

                        CASE
                            WHEN trade_value_reconciled IS NOT NULL
                             AND isfinite(
                                 trade_value_reconciled
                             )
                             AND trade_value_reconciled > 0
                             AND has_import_quantity_kg
                                THEN (
                                    trade_value_reconciled
                                    /
                                    import_synchronized_quantity_kg
                                )
                            ELSE NULL
                        END AS import_implied_unit_value

                    FROM source
                ),

                -- =====================================================
                -- 3. Quantity reliability evidence
                -- =====================================================
                --
                -- A bilateral positive pair contributes the same
                -- disagreement magnitude to each reporting side.
                --
                -- Exporter:
                --   exporter country × year × export direction
                --
                -- Importer:
                --   importer country × year × import direction
                -- =====================================================

                quantity_reliability_evidence AS (

                    SELECT
                        year,
                        exporter_iso AS country_iso,
                        'export' AS reporting_direction,
                        importer_iso AS partner_iso,
                        target_cmd AS product,
                        quantity_relative_difference
                            AS quantity_deviation

                    FROM bilateral

                    WHERE quantity_availability_case
                        = 'both_quantity'
                      AND quantity_relative_difference
                        IS NOT NULL

                    UNION ALL

                    SELECT
                        year,
                        importer_iso AS country_iso,
                        'import' AS reporting_direction,
                        exporter_iso AS partner_iso,
                        target_cmd AS product,
                        quantity_relative_difference
                            AS quantity_deviation

                    FROM bilateral

                    WHERE quantity_availability_case
                        = 'both_quantity'
                      AND quantity_relative_difference
                        IS NOT NULL
                ),

                quantity_country_reliability AS (
                    SELECT
                        year,
                        country_iso,
                        reporting_direction,

                        COUNT(*)
                            AS quantity_reliability_observations,

                        COUNT(
                            DISTINCT partner_iso
                        )
                            AS quantity_reliability_partners,

                        COUNT(
                            DISTINCT product
                        )
                            AS quantity_reliability_products,

                        AVG(
                            quantity_deviation
                        )
                            AS quantity_mean_deviation,

                        1.0
                        - AVG(
                            quantity_deviation
                        )
                            AS quantity_reliability_index

                    FROM quantity_reliability_evidence

                    GROUP BY
                        year,
                        country_iso,
                        reporting_direction
                ),

                -- =====================================================
                -- 4. Attach quantity reliability to bilateral rows
                -- =====================================================

                with_reliability AS (
                    SELECT
                        bilateral.*,

                        export_rel.quantity_reliability_index
                            AS export_quantity_reliability,

                        import_rel.quantity_reliability_index
                            AS import_quantity_reliability,

                        export_rel.quantity_mean_deviation
                            AS export_quantity_mean_deviation,

                        import_rel.quantity_mean_deviation
                            AS import_quantity_mean_deviation,

                        export_rel.quantity_reliability_observations
                            AS export_quantity_reliability_observations,

                        import_rel.quantity_reliability_observations
                            AS import_quantity_reliability_observations

                    FROM bilateral

                    LEFT JOIN quantity_country_reliability
                        AS export_rel
                      ON export_rel.year = bilateral.year
                     AND export_rel.country_iso
                         = bilateral.exporter_iso
                     AND export_rel.reporting_direction
                         = 'export'

                    LEFT JOIN quantity_country_reliability
                        AS import_rel
                      ON import_rel.year = bilateral.year
                     AND import_rel.country_iso
                         = bilateral.importer_iso
                     AND import_rel.reporting_direction
                         = 'import'
                ),

                -- =====================================================
                -- 5. Pooled candidate unit-value observations
                -- =====================================================
                --
                -- Both sides are evaluated against the same harmonized
                -- trade value.
                --
                -- The reference distribution is product × year.
                --
                -- Work in log space because unit-value distributions are
                -- strongly right-skewed.
                -- =====================================================

                unit_value_candidates AS (

                    SELECT
                        year,
                        target_cmd AS product,
                        LN(
                            export_implied_unit_value
                        ) AS log_unit_value

                    FROM with_reliability

                    WHERE export_implied_unit_value
                        IS NOT NULL
                      AND export_implied_unit_value > 0
                      AND isfinite(
                          export_implied_unit_value
                      )

                    UNION ALL

                    SELECT
                        year,
                        target_cmd AS product,
                        LN(
                            import_implied_unit_value
                        ) AS log_unit_value

                    FROM with_reliability

                    WHERE import_implied_unit_value
                        IS NOT NULL
                      AND import_implied_unit_value > 0
                      AND isfinite(
                          import_implied_unit_value
                      )
                ),

                unit_value_medians AS (
                    SELECT
                        year,
                        product,

                        COUNT(*)
                            AS unit_value_reference_observations,

                        MEDIAN(
                            log_unit_value
                        ) AS median_log_unit_value

                    FROM unit_value_candidates

                    GROUP BY
                        year,
                        product
                ),

                unit_value_reference AS (
                    SELECT
                        candidates.year,
                        candidates.product,

                        medians.unit_value_reference_observations,
                        medians.median_log_unit_value,

                        MEDIAN(
                            ABS(
                                candidates.log_unit_value
                                - medians.median_log_unit_value
                            )
                        ) AS mad_log_unit_value

                    FROM unit_value_candidates
                        AS candidates

                    INNER JOIN unit_value_medians
                        AS medians
                      ON medians.year
                         = candidates.year
                     AND medians.product
                         = candidates.product

                    GROUP BY
                        candidates.year,
                        candidates.product,
                        medians.unit_value_reference_observations,
                        medians.median_log_unit_value
                ),

                -- =====================================================
                -- 6. Unit-value plausibility distances
                -- =====================================================

                with_plausibility AS (
                    SELECT
                        rows.*,

                        reference.unit_value_reference_observations,

                        EXP(
                            reference.median_log_unit_value
                        ) AS product_year_median_unit_value,

                        CASE
                            WHEN rows.trade_value_reconciled IS NOT NULL
                            AND isfinite(rows.trade_value_reconciled)
                            AND rows.trade_value_reconciled > 0
                            AND rows.quantity_simple_mean_kg IS NOT NULL
                            AND isfinite(rows.quantity_simple_mean_kg)
                            AND rows.quantity_simple_mean_kg > 0
                                THEN (
                                    rows.trade_value_reconciled
                                    / rows.quantity_simple_mean_kg
                                )
                            ELSE NULL
                        END AS mean_implied_unit_value,
                        reference.mad_log_unit_value
                            AS product_year_log_unit_value_mad,

                        CASE
                            WHEN rows.export_implied_unit_value
                                    IS NOT NULL
                             AND rows.export_implied_unit_value > 0
                             AND reference.median_log_unit_value
                                    IS NOT NULL
                                THEN ABS(
                                    LN(
                                        rows.export_implied_unit_value
                                    )
                                    - reference.median_log_unit_value
                                )
                            ELSE NULL
                        END
                            AS export_unit_value_log_distance,

                        CASE
                            WHEN rows.import_implied_unit_value
                                    IS NOT NULL
                             AND rows.import_implied_unit_value > 0
                             AND reference.median_log_unit_value
                                    IS NOT NULL
                                THEN ABS(
                                    LN(
                                        rows.import_implied_unit_value
                                    )
                                    - reference.median_log_unit_value
                                )
                            ELSE NULL
                        END
                            AS import_unit_value_log_distance,

                        CASE
                            WHEN rows.trade_value_reconciled IS NOT NULL
                            AND isfinite(rows.trade_value_reconciled)
                            AND rows.trade_value_reconciled > 0
                            AND rows.quantity_simple_mean_kg IS NOT NULL
                            AND isfinite(rows.quantity_simple_mean_kg)
                            AND rows.quantity_simple_mean_kg > 0
                            AND reference.median_log_unit_value IS NOT NULL
                                THEN ABS(
                                    LN(
                                        rows.trade_value_reconciled
                                        / rows.quantity_simple_mean_kg
                                    )
                                    - reference.median_log_unit_value
                                )
                            ELSE NULL
                        END AS mean_unit_value_log_distance,

                        CASE
                            WHEN rows.export_implied_unit_value
                                    IS NOT NULL
                             AND rows.export_implied_unit_value > 0
                             AND reference.mad_log_unit_value > 0
                                THEN (
                                    ABS(
                                        LN(
                                            rows.export_implied_unit_value
                                        )
                                        - reference.median_log_unit_value
                                    )
                                    /
                                    reference.mad_log_unit_value
                                )
                            ELSE NULL
                        END
                            AS export_unit_value_robust_distance,

                        CASE
                            WHEN rows.import_implied_unit_value
                                    IS NOT NULL
                             AND rows.import_implied_unit_value > 0
                             AND reference.mad_log_unit_value > 0
                                THEN (
                                    ABS(
                                        LN(
                                            rows.import_implied_unit_value
                                        )
                                        - reference.median_log_unit_value
                                    )
                                    /
                                    reference.mad_log_unit_value
                                )
                            ELSE NULL
                        END
                            AS import_unit_value_robust_distance

                    FROM with_reliability
                        AS rows

                    LEFT JOIN unit_value_reference
                        AS reference
                      ON reference.year
                         = rows.year
                     AND reference.product
                         = rows.target_cmd
                ),

                -- =====================================================
                -- 7. Quantity-reliability weighted candidate
                -- =====================================================
                --
                -- This is retained as diagnostic evidence.
                --
                -- Reliability indices are normalized to sum to one.
                -- No additional importer preference is applied here.
                -- =====================================================

                with_candidates AS (
                    SELECT
                        *,

                        CASE
                            WHEN quantity_availability_case
                                    = 'both_quantity'
                             AND export_quantity_reliability
                                    IS NOT NULL
                             AND import_quantity_reliability
                                    IS NOT NULL
                             AND (
                                 export_quantity_reliability
                                 + import_quantity_reliability
                             ) > 0
                                THEN (
                                    export_quantity_reliability
                                    /
                                    (
                                        export_quantity_reliability
                                        + import_quantity_reliability
                                    )
                                )
                            ELSE NULL
                        END
                            AS export_quantity_reliability_weight,

                        CASE
                            WHEN quantity_availability_case
                                    = 'both_quantity'
                             AND export_quantity_reliability
                                    IS NOT NULL
                             AND import_quantity_reliability
                                    IS NOT NULL
                             AND (
                                 export_quantity_reliability
                                 + import_quantity_reliability
                             ) > 0
                                THEN (
                                    import_quantity_reliability
                                    /
                                    (
                                        export_quantity_reliability
                                        + import_quantity_reliability
                                    )
                                )
                            ELSE NULL
                        END
                            AS import_quantity_reliability_weight

                    FROM with_plausibility
                ),

                candidates_complete AS (
                    SELECT
                        *,

                        CASE
                            WHEN export_quantity_reliability_weight
                                    IS NOT NULL
                             AND import_quantity_reliability_weight
                                    IS NOT NULL
                                THEN (
                                    export_synchronized_quantity_kg
                                    * export_quantity_reliability_weight
                                    +
                                    import_synchronized_quantity_kg
                                    * import_quantity_reliability_weight
                                )
                            ELSE NULL
                        END
                            AS quantity_reliability_weighted_kg

                    FROM with_candidates
                ),

                -- =====================================================
                -- 8. Transparent first-pass reconciliation
                -- =====================================================
                --
                -- IMPORTANT:
                --
                -- We do not yet convert unit-value plausibility into an
                -- arbitrary mathematical weight.
                --
                -- For bilateral rows:
                --
                -- - if quantity-reliability evidence exists, use the
                --   quantity-reliability-weighted candidate;
                --
                -- - otherwise fall back to the neutral simple mean.
                --
                -- Unit-value plausibility is retained as explicit evidence
                -- for the next methodological refinement and audit.
                --
                -- Unilateral usable kg quantities are retained unchanged.
                -- =====================================================

                reconciled AS (
                    SELECT
                        *,

                        CASE
                            WHEN quantity_availability_case
                                    = 'both_quantity'
                             AND quantity_reliability_weighted_kg
                                    IS NOT NULL
                                THEN quantity_reliability_weighted_kg

                            WHEN quantity_availability_case
                                    = 'both_quantity'
                                THEN quantity_simple_mean_kg

                            WHEN quantity_availability_case
                                    = 'export_only'
                                THEN export_synchronized_quantity_kg

                            WHEN quantity_availability_case
                                    = 'import_only'
                                THEN import_synchronized_quantity_kg

                            ELSE NULL
                        END AS quantity_reconciled_kg,

                        CASE
                            WHEN quantity_availability_case
                                    = 'both_quantity'
                             AND quantity_reliability_weighted_kg
                                    IS NOT NULL
                                THEN 'quantity_reliability_weighted'

                            WHEN quantity_availability_case
                                    = 'both_quantity'
                                THEN 'bilateral_simple_mean_fallback'

                            WHEN quantity_availability_case
                                    = 'export_only'
                                THEN 'export_quantity_retained'

                            WHEN quantity_availability_case
                                    = 'import_only'
                                THEN 'import_quantity_retained'

                            ELSE 'no_usable_quantity'
                        END AS quantity_reconciliation_method,
                    

                        CASE
                            -- -------------------------------------------------------------
                            -- Bilateral quantities:
                            -- choose the candidate whose implied unit value lies closest
                            -- to the robust product-year unit-value reference.
                            --
                            -- In an exact tie, prefer the neutral bilateral mean.
                            -- -------------------------------------------------------------

                            WHEN quantity_availability_case = 'both_quantity'
                            AND mean_unit_value_log_distance IS NOT NULL
                            AND mean_unit_value_log_distance
                                    <= export_unit_value_log_distance
                            AND mean_unit_value_log_distance
                                    <= import_unit_value_log_distance
                                THEN quantity_simple_mean_kg

                            WHEN quantity_availability_case = 'both_quantity'
                            AND export_unit_value_log_distance IS NOT NULL
                            AND (
                                    import_unit_value_log_distance IS NULL
                                    OR export_unit_value_log_distance
                                        <= import_unit_value_log_distance
                                )
                                THEN export_synchronized_quantity_kg

                            WHEN quantity_availability_case = 'both_quantity'
                            AND import_unit_value_log_distance IS NOT NULL
                                THEN import_synchronized_quantity_kg

                            -- If product-year unit-value evidence is unavailable,
                            -- retain a neutral bilateral fallback.
                            WHEN quantity_availability_case = 'both_quantity'
                                THEN quantity_simple_mean_kg

                            -- Unilateral usable quantities remain unchanged.
                            WHEN quantity_availability_case = 'export_only'
                                THEN export_synchronized_quantity_kg

                            WHEN quantity_availability_case = 'import_only'
                                THEN import_synchronized_quantity_kg

                            ELSE NULL

                        END AS trade_synced_quantity_kg,

                        
                        CASE
                            WHEN quantity_availability_case = 'both_quantity'
                            AND mean_unit_value_log_distance IS NOT NULL
                            AND mean_unit_value_log_distance
                                    <= export_unit_value_log_distance
                            AND mean_unit_value_log_distance
                                    <= import_unit_value_log_distance
                                THEN 'unit_value_guided_mean'

                            WHEN quantity_availability_case = 'both_quantity'
                            AND export_unit_value_log_distance IS NOT NULL
                            AND (
                                    import_unit_value_log_distance IS NULL
                                    OR export_unit_value_log_distance
                                        <= import_unit_value_log_distance
                                )
                                THEN 'unit_value_guided_export'

                            WHEN quantity_availability_case = 'both_quantity'
                            AND import_unit_value_log_distance IS NOT NULL
                                THEN 'unit_value_guided_import'

                            WHEN quantity_availability_case = 'both_quantity'
                                THEN 'bilateral_mean_no_unit_value_reference'

                            WHEN quantity_availability_case = 'export_only'
                                THEN 'export_quantity_retained'

                            WHEN quantity_availability_case = 'import_only'
                                THEN 'import_quantity_retained'

                            ELSE 'no_usable_quantity'

                        END AS trade_synced_quantity_method,
                        
                        CASE
                            WHEN quantity_availability_case = 'both_quantity'
                                THEN 'bilateral_synced'

                            WHEN quantity_availability_case IN (
                                'export_only',
                                'import_only'
                            )
                                THEN 'unilateral_retained'

                            ELSE 'missing'

                        END AS trade_synced_quantity_status
                    FROM candidates_complete
                )

                SELECT *
                FROM reconciled

            )
            TO {output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        # --------------------------------------------------------------
        # Basic stage-level accounting
        # --------------------------------------------------------------
        #
        # Full correctness belongs to the dedicated quantity audit step.
        # Here we only verify that a non-empty artifact was produced and
        # collect useful counts for logging / workflow return metadata.
        # --------------------------------------------------------------

        summary = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,

                COUNT(*) FILTER (
                    WHERE quantity_availability_case
                        = 'both_quantity'
                ) AS n_both_quantity,

                COUNT(*) FILTER (
                    WHERE quantity_availability_case
                        = 'export_only'
                ) AS n_export_only,

                COUNT(*) FILTER (
                    WHERE quantity_availability_case
                        = 'import_only'
                ) AS n_import_only,

                COUNT(*) FILTER (
                    WHERE quantity_availability_case
                        = 'no_quantity'
                ) AS n_no_quantity,

                COUNT(*) FILTER (
                    WHERE quantity_reconciliation_method
                        = 'quantity_reliability_weighted'
                ) AS n_quantity_reliability_weighted,

                COUNT(*) FILTER (
                    WHERE quantity_reconciliation_method
                        = 'bilateral_simple_mean_fallback'
                ) AS n_simple_mean_fallback,

                COUNT(*) FILTER (
                    WHERE quantity_reconciled_kg IS NOT NULL
                ) AS n_reconciled_quantity,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_method
                        = 'unit_value_guided_export'
                ) AS n_trade_synced_export,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_method
                        = 'unit_value_guided_import'
                ) AS n_trade_synced_import,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_method
                        = 'unit_value_guided_mean'
                ) AS n_trade_synced_mean,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_method
                        = 'bilateral_mean_no_unit_value_reference'
                ) AS n_trade_synced_mean_no_reference

            FROM read_parquet({output_sql})
            """
        ).fetchone()

        if summary is None:
            raise RuntimeError(
                "Could not summarize reconciled quantity output."
            )

        (
            n_rows,
            n_both_quantity,
            n_export_only,
            n_import_only,
            n_no_quantity,
            n_quantity_reliability_weighted,
            n_simple_mean_fallback,
            n_reconciled_quantity,
            n_trade_synced_export,
            n_trade_synced_import,
            n_trade_synced_mean,
            n_trade_synced_mean_no_reference,
        ) = [
            int(value or 0)
            for value in summary
        ]

        if not _valid_output(output_path):
            raise RuntimeError(
                "Quantity reconciliation did not produce "
                f"a usable output: {output_path}"
            )

        logger.info(
            "  Rows: %s",
            f"{n_rows:,}",
        )

        logger.info(
            "  Both quantities usable: %s",
            f"{n_both_quantity:,}",
        )

        logger.info(
            "  Export quantity only: %s",
            f"{n_export_only:,}",
        )

        logger.info(
            "  Import quantity only: %s",
            f"{n_import_only:,}",
        )

        logger.info(
            "  No usable kg quantity: %s",
            f"{n_no_quantity:,}",
        )

        logger.info(
            "  Quantity-reliability weighted: %s",
            f"{n_quantity_reliability_weighted:,}",
        )

        logger.info(
            "  Simple-mean fallback: %s",
            f"{n_simple_mean_fallback:,}",
        )

        logger.info(
            "  Reconciled quantity rows: %s",
            f"{n_reconciled_quantity:,}",
        )

        logger.info(
            "  Trade synced via export guidance: %s",
            f"{n_trade_synced_export:,}",
        )

        logger.info(
            "  Trade synced via import guidance: %s",
            f"{n_trade_synced_import:,}",
        )

        logger.info(
            "  Trade synced via neutral bilateral mean: %s",
            f"{n_trade_synced_mean:,}",
        )

        logger.info(
            "  Trade synced without unit-value reference: %s",
            f"{n_trade_synced_mean_no_reference:,}",
        )

        logger.info(
            "✍ WRITE — Reconciled bilateral quantity: %s",
            output_path,
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "reconciled_quantity_path": output_path,
            "output_path": output_path,
            "n_rows": n_rows,
            "n_both_quantity": n_both_quantity,
            "n_export_only": n_export_only,
            "n_import_only": n_import_only,
            "n_no_quantity": n_no_quantity,
            "n_quantity_reliability_weighted": (
                n_quantity_reliability_weighted
            ),
            "n_simple_mean_fallback": (
                n_simple_mean_fallback
            ),
            "n_reconciled_quantity": (
                n_reconciled_quantity
            ),
            "n_trade_synced_export": n_trade_synced_export,
            "n_trade_synced_import": n_trade_synced_import,
            "n_trade_synced_mean": n_trade_synced_mean,
            "n_trade_synced_mean_no_reference": (
                n_trade_synced_mean_no_reference
            ),
        }

    finally:
        con.close()