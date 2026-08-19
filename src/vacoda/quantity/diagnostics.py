"""
vacoda.quantity.diagnostics

Build report-grade diagnostics for quantity synchronization.

This module summarizes:

- quantity-field availability and coverage;
- unit composition of quantity and alt_quantity;
- cubic-metre reporting by HS code;
- reported-versus-estimated kg evidence;
- quantity-versus-net-weight agreement;
- cumulative coverage contributed by synchronization sources;
- gross-weight proxy contribution;
- changes introduced by synchronization;
- unresolved conflicts and remaining missing kg evidence;
- annual synchronization behaviour.

The output is a long-format diagnostics Parquet designed to feed the
Quantity decision report.

No synchronization decisions are made here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.quantity import QuantityConfig


def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL literal."""
    return "'" + str(value).replace("'", "''") + "'"


def build_quantity_diagnostics(
    *,
    cfg: QuantityConfig,
    prepared_input_path: Path,
    reconciliation_evidence_path: Path,
    synchronized_quantity_path: Path,
    run_id: str,
    force: bool,
    logger,
) -> dict[str, Any]:
    """
    Build report-grade diagnostics for the Quantity workflow.

    Parameters
    ----------
    cfg:
        Parsed Quantity configuration.
    prepared_input_path:
        Prepared Quantity input Parquet.
    reconciliation_evidence_path:
        Row-level Quantity reconciliation evidence.
    synchronized_quantity_path:
        Final synchronized Quantity Parquet.
    run_id:
        Master-pipeline run identifier.
    force:
        Rebuild diagnostics even when an existing artifact is available.
    logger:
        Configured workflow logger.

    Returns
    -------
    dict
        Result metadata including ``diagnostics_path``.
    """

    prepared_input_path = Path(prepared_input_path)
    reconciliation_evidence_path = Path(
        reconciliation_evidence_path
    )
    synchronized_quantity_path = Path(
        synchronized_quantity_path
    )

    required_inputs = {
        "prepared Quantity input": prepared_input_path,
        "Quantity reconciliation evidence": (
            reconciliation_evidence_path
        ),
        "synchronized Quantity output": (
            synchronized_quantity_path
        ),
    }

    for name, path in required_inputs.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} parquet not found: {path}"
            )

    output_path = cfg.product_path("diagnostics")

    if output_path.is_file() and not force:
        logger.info(
            "Reusing existing Quantity diagnostics: %s",
            output_path,
        )

        return {
            "status": "skipped_existing",
            "run_id": run_id,
            "output_path": output_path,
            "diagnostics_path": output_path,
        }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared_sql = _sql_literal(prepared_input_path)
    evidence_sql = _sql_literal(
        reconciliation_evidence_path
    )
    synchronized_sql = _sql_literal(
        synchronized_quantity_path
    )

    agreement_tolerance = float(
        cfg.method
        .kg_reconciliation
        .agreement
        .relative_tolerance
    )

    logger.info("Building report-grade Quantity diagnostics.")
    logger.info(
        "Agreement tolerance: %.6f",
        agreement_tolerance,
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        # ==============================================================
        # Resolve HS/product column
        # ==============================================================

        available_columns = {
            row[0]
            for row in con.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_parquet({prepared_sql})
                """
            ).fetchall()
        }

        hs_column = "target_cmd"

        if hs_column not in available_columns:
            raise ValueError(
                "Quantity diagnostics require HS Sync column "
                "'target_cmd'."
            )

        logger.info(
            "HS/product diagnostics column: %s",
            hs_column,
        )

        # ==============================================================
        # Diagnostics schema
        #
        # Every query emits the same long-format structure:
        #
        # diagnostic_group
        # metric
        # value
        # percentage
        # year
        # category_1
        # category_2
        # category_3
        # ==============================================================
        queries: list[str] = []

        # ==============================================================
        # 1. Quantity field inventory
        # ==============================================================

        queries.append(
            f"""
            SELECT
                'field_inventory' AS diagnostic_group,
                field_name AS metric,
                CAST(metric_value AS DOUBLE) AS value,
                percentage,
                NULL::INTEGER AS year,
                statistic AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM (

                SELECT
                    'quantity' AS field_name,
                    'non_null_rows' AS statistic,
                    COUNT(quantity) AS metric_value,
                    100.0 * COUNT(quantity) / COUNT(*)
                        AS percentage
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'quantity',
                    'positive_rows',
                    COUNT(*) FILTER (
                        WHERE quantity > 0
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE quantity > 0
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'quantity',
                    'zero_rows',
                    COUNT(*) FILTER (
                        WHERE quantity = 0
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE quantity = 0
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'quantity',
                    'missing_rows',
                    COUNT(*) FILTER (
                        WHERE quantity IS NULL
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE quantity IS NULL
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'alt_quantity',
                    'non_null_rows',
                    COUNT(alt_quantity),
                    100.0 * COUNT(alt_quantity) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'alt_quantity',
                    'positive_rows',
                    COUNT(*) FILTER (
                        WHERE alt_quantity > 0
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE alt_quantity > 0
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'alt_quantity',
                    'zero_rows',
                    COUNT(*) FILTER (
                        WHERE alt_quantity = 0
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE alt_quantity = 0
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'alt_quantity',
                    'missing_rows',
                    COUNT(*) FILTER (
                        WHERE alt_quantity IS NULL
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE alt_quantity IS NULL
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'net_weight_kg',
                    'non_null_rows',
                    COUNT(net_weight_kg),
                    100.0 * COUNT(net_weight_kg) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'net_weight_kg',
                    'positive_rows',
                    COUNT(*) FILTER (
                        WHERE net_weight_kg > 0
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE net_weight_kg > 0
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'net_weight_kg',
                    'zero_rows',
                    COUNT(*) FILTER (
                        WHERE net_weight_kg = 0
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE net_weight_kg = 0
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'net_weight_kg',
                    'missing_rows',
                    COUNT(*) FILTER (
                        WHERE net_weight_kg IS NULL
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE net_weight_kg IS NULL
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'gross_weight_kg',
                    'non_null_rows',
                    COUNT(gross_weight_kg),
                    100.0 * COUNT(gross_weight_kg) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'gross_weight_kg',
                    'positive_rows',
                    COUNT(*) FILTER (
                        WHERE gross_weight_kg > 0
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE gross_weight_kg > 0
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'gross_weight_kg',
                    'zero_rows',
                    COUNT(*) FILTER (
                        WHERE gross_weight_kg = 0
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE gross_weight_kg = 0
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'gross_weight_kg',
                    'missing_rows',
                    COUNT(*) FILTER (
                        WHERE gross_weight_kg IS NULL
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE gross_weight_kg IS NULL
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})
            )
            """
        )

        # ==============================================================
        # 2. Reported / estimated coverage
        # ==============================================================

        queries.append(
            f"""
            SELECT
                'estimated_reported_coverage'
                    AS diagnostic_group,
                source AS metric,
                CAST(n_rows AS DOUBLE) AS value,
                percentage,
                NULL::INTEGER AS year,
                state AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM (

                SELECT
                    'quantity' AS source,
                    'reported' AS state,
                    COUNT(*) FILTER (
                        WHERE has_reported_quantity
                    ) AS n_rows,
                    100.0 * COUNT(*) FILTER (
                        WHERE has_reported_quantity
                    ) / COUNT(*) AS percentage
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'quantity',
                    'estimated',
                    COUNT(*) FILTER (
                        WHERE has_estimated_quantity
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE has_estimated_quantity
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'quantity_kg',
                    'reported',
                    COUNT(*) FILTER (
                        WHERE has_reported_quantity_kg
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE has_reported_quantity_kg
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'quantity_kg',
                    'estimated',
                    COUNT(*) FILTER (
                        WHERE has_estimated_quantity_kg
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE has_estimated_quantity_kg
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'net_weight_kg',
                    'reported',
                    COUNT(*) FILTER (
                        WHERE has_reported_net_weight_kg
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE has_reported_net_weight_kg
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'net_weight_kg',
                    'estimated',
                    COUNT(*) FILTER (
                        WHERE has_estimated_net_weight_kg
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE has_estimated_net_weight_kg
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})
            )
            """
        )

        # ==============================================================
        # 3. Quantity-unit distribution
        # ==============================================================

        if cfg.diagnostics.unit_distribution.enabled:
            queries.append(
                f"""
                SELECT
                    'quantity_unit_distribution'
                        AS diagnostic_group,
                    COALESCE(
                        CAST(quantity_unit AS VARCHAR),
                        '<NULL>'
                    ) AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER ()
                        AS percentage,
                    NULL::INTEGER AS year,
                    CAST(quantity_unit_code AS VARCHAR)
                        AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({prepared_sql})
                GROUP BY
                    quantity_unit,
                    quantity_unit_code
                """
            )

            queries.append(
                f"""
                SELECT
                    'alt_quantity_unit_distribution'
                        AS diagnostic_group,
                    COALESCE(
                        CAST(alt_quantity_unit AS VARCHAR),
                        '<NULL>'
                    ) AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER ()
                        AS percentage,
                    NULL::INTEGER AS year,
                    CAST(alt_quantity_unit_code AS VARCHAR)
                        AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({prepared_sql})
                GROUP BY
                    alt_quantity_unit,
                    alt_quantity_unit_code
                """
            )

            # ----------------------------------------------------------
            # Unit distribution by year
            # ----------------------------------------------------------

            queries.append(
                f"""
                SELECT
                    'quantity_unit_by_year'
                        AS diagnostic_group,
                    COALESCE(
                        CAST(quantity_unit AS VARCHAR),
                        '<NULL>'
                    ) AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER (
                            PARTITION BY year
                        ) AS percentage,
                    CAST(year AS INTEGER) AS year,
                    CAST(quantity_unit_code AS VARCHAR)
                        AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({prepared_sql})
                GROUP BY
                    year,
                    quantity_unit,
                    quantity_unit_code
                """
            )

            queries.append(
                f"""
                SELECT
                    'alt_quantity_unit_by_year'
                        AS diagnostic_group,
                    COALESCE(
                        CAST(alt_quantity_unit AS VARCHAR),
                        '<NULL>'
                    ) AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER (
                            PARTITION BY year
                        ) AS percentage,
                    CAST(year AS INTEGER) AS year,
                    CAST(alt_quantity_unit_code AS VARCHAR)
                        AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({prepared_sql})
                GROUP BY
                    year,
                    alt_quantity_unit,
                    alt_quantity_unit_code
                """
            )

        # ==============================================================
        # 4. m³ distribution by HS code
        # ==============================================================

        if hs_column is not None:
            queries.append(
                f"""
                SELECT
                    'quantity_m3_by_hs'
                        AS diagnostic_group,
                    CAST({hs_column} AS VARCHAR) AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER ()
                        AS percentage,
                    NULL::INTEGER AS year,
                    'quantity' AS category_1,
                    'm3' AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({prepared_sql})
                WHERE LOWER(TRIM(quantity_unit))
                    IN ('m3', 'm³')
                  AND quantity > 0
                GROUP BY {hs_column}
                """
            )

            queries.append(
                f"""
                SELECT
                    'alt_quantity_m3_by_hs'
                        AS diagnostic_group,
                    CAST({hs_column} AS VARCHAR) AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER ()
                        AS percentage,
                    NULL::INTEGER AS year,
                    'alt_quantity' AS category_1,
                    'm3' AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({prepared_sql})
                WHERE LOWER(TRIM(alt_quantity_unit))
                    IN ('m3', 'm³')
                  AND alt_quantity > 0
                GROUP BY {hs_column}
                """
            )

        # ==============================================================
        # 5. kg evidence availability
        # ==============================================================

        if cfg.diagnostics.source_coverage.enabled:
            queries.append(
                f"""
                SELECT
                    'kg_source_coverage'
                        AS diagnostic_group,
                    source AS metric,
                    CAST(n_rows AS DOUBLE) AS value,
                    percentage,
                    NULL::INTEGER AS year,
                    NULL::VARCHAR AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3
                FROM (

                    SELECT
                        'quantity_kg' AS source,
                        COUNT(*) FILTER (
                            WHERE has_quantity_kg
                        ) AS n_rows,
                        100.0 * COUNT(*) FILTER (
                            WHERE has_quantity_kg
                        ) / COUNT(*) AS percentage
                    FROM read_parquet({prepared_sql})

                    UNION ALL

                    SELECT
                        'net_weight_kg',
                        COUNT(*) FILTER (
                            WHERE has_net_weight_kg
                        ),
                        100.0 * COUNT(*) FILTER (
                            WHERE has_net_weight_kg
                        ) / COUNT(*)
                    FROM read_parquet({prepared_sql})

                    UNION ALL

                    SELECT
                        'gross_weight_kg',
                        COUNT(*) FILTER (
                            WHERE has_gross_weight_kg
                        ),
                        100.0 * COUNT(*) FILTER (
                            WHERE has_gross_weight_kg
                        ) / COUNT(*)
                    FROM read_parquet({prepared_sql})

                    UNION ALL

                    SELECT
                        'synchronized_quantity_kg',
                        COUNT(*) FILTER (
                            WHERE synchronized_quantity_kg > 0
                        ),
                        100.0 * COUNT(*) FILTER (
                            WHERE synchronized_quantity_kg > 0
                        ) / COUNT(*)
                    FROM read_parquet({synchronized_sql})
                )
                """
            )

        # ==============================================================
        # 6. Reported / estimated overlap matrix
        # ==============================================================

        if cfg.diagnostics.reported_estimated_overlap.enabled:
            queries.append(
                f"""
                SELECT
                    'kg_evidence_overlap'
                        AS diagnostic_group,
                    kg_evidence_status AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER ()
                        AS percentage,
                    NULL::INTEGER AS year,
                    quantity_kg_status AS category_1,
                    net_weight_kg_status AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({evidence_sql})
                GROUP BY
                    kg_evidence_status,
                    quantity_kg_status,
                    net_weight_kg_status
                """
            )

        # ==============================================================
        # 7. Quantity ↔ net-weight agreement
        # ==============================================================

        if cfg.diagnostics.quantity_net_weight_comparison.enabled:

            queries.append(
                f"""
                SELECT
                    'quantity_net_weight_agreement'
                        AS diagnostic_group,
                    quantity_net_weight_agreement_status
                        AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER ()
                        AS percentage,
                    NULL::INTEGER AS year,
                    NULL::VARCHAR AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({evidence_sql})
                GROUP BY
                    quantity_net_weight_agreement_status
                """
            )

            # ----------------------------------------------------------
            # Summary statistics
            # ----------------------------------------------------------

            queries.append(
                f"""
                SELECT
                    'quantity_net_weight_statistics'
                        AS diagnostic_group,
                    metric,
                    value,
                    NULL::DOUBLE AS percentage,
                    NULL::INTEGER AS year,
                    NULL::VARCHAR AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3
                FROM (

                    SELECT
                        'n_comparable' AS metric,
                        CAST(COUNT(*) AS DOUBLE) AS value
                    FROM read_parquet({evidence_sql})
                    WHERE quantity_net_weight_relative_gap
                        IS NOT NULL

                    UNION ALL

                    SELECT
                        'median_ratio',
                        MEDIAN(quantity_net_weight_ratio)
                    FROM read_parquet({evidence_sql})
                    WHERE quantity_net_weight_ratio IS NOT NULL

                    UNION ALL

                    SELECT
                        'p10_ratio',
                        QUANTILE_CONT(
                            quantity_net_weight_ratio,
                            0.10
                        )
                    FROM read_parquet({evidence_sql})
                    WHERE quantity_net_weight_ratio IS NOT NULL

                    UNION ALL

                    SELECT
                        'p90_ratio',
                        QUANTILE_CONT(
                            quantity_net_weight_ratio,
                            0.90
                        )
                    FROM read_parquet({evidence_sql})
                    WHERE quantity_net_weight_ratio IS NOT NULL

                    UNION ALL

                    SELECT
                        'median_relative_gap',
                        MEDIAN(
                            quantity_net_weight_relative_gap
                        )
                    FROM read_parquet({evidence_sql})
                    WHERE quantity_net_weight_relative_gap
                        IS NOT NULL

                    UNION ALL

                    SELECT
                        'p90_relative_gap',
                        QUANTILE_CONT(
                            quantity_net_weight_relative_gap,
                            0.90
                        )
                    FROM read_parquet({evidence_sql})
                    WHERE quantity_net_weight_relative_gap
                        IS NOT NULL

                    UNION ALL

                    SELECT
                        'p95_relative_gap',
                        QUANTILE_CONT(
                            quantity_net_weight_relative_gap,
                            0.95
                        )
                    FROM read_parquet({evidence_sql})
                    WHERE quantity_net_weight_relative_gap
                        IS NOT NULL

                    UNION ALL

                    SELECT
                        'within_1pct_share',
                        AVG(
                            CASE
                                WHEN quantity_net_weight_relative_gap
                                    <= 0.01
                                THEN 1.0
                                ELSE 0.0
                            END
                        )
                    FROM read_parquet({evidence_sql})
                    WHERE quantity_net_weight_relative_gap
                        IS NOT NULL

                    UNION ALL

                    SELECT
                        'within_5pct_share',
                        AVG(
                            CASE
                                WHEN quantity_net_weight_relative_gap
                                    <= 0.05
                                THEN 1.0
                                ELSE 0.0
                            END
                        )
                    FROM read_parquet({evidence_sql})
                    WHERE quantity_net_weight_relative_gap
                        IS NOT NULL

                    UNION ALL

                    SELECT
                        'within_10pct_share',
                        AVG(
                            CASE
                                WHEN quantity_net_weight_relative_gap
                                    <= 0.10
                                THEN 1.0
                                ELSE 0.0
                            END
                        )
                    FROM read_parquet({evidence_sql})
                    WHERE quantity_net_weight_relative_gap
                        IS NOT NULL
                )
                """
            )

            # ----------------------------------------------------------
            # Agreement by year
            # ----------------------------------------------------------

            queries.append(
                f"""
                SELECT
                    'quantity_net_weight_by_year'
                        AS diagnostic_group,
                    'agreement' AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * AVG(
                        CASE
                            WHEN quantity_net_weight_relative_gap
                                <= {agreement_tolerance}
                            THEN 1.0
                            ELSE 0.0
                        END
                    ) AS percentage,
                    CAST(year AS INTEGER) AS year,
                    CAST(
                        MEDIAN(
                            quantity_net_weight_relative_gap
                        ) AS VARCHAR
                    ) AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({evidence_sql})
                WHERE quantity_net_weight_relative_gap
                    IS NOT NULL
                GROUP BY year
                """
            )

            # ----------------------------------------------------------
            # Agreement by reported / estimated state
            # ----------------------------------------------------------

            queries.append(
                f"""
                SELECT
                    'quantity_net_weight_by_evidence_state'
                        AS diagnostic_group,
                    kg_evidence_status AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * AVG(
                        CASE
                            WHEN quantity_net_weight_relative_gap
                                <= {agreement_tolerance}
                            THEN 1.0
                            ELSE 0.0
                        END
                    ) AS percentage,
                    NULL::INTEGER AS year,
                    CAST(
                        MEDIAN(quantity_net_weight_ratio)
                        AS VARCHAR
                    ) AS category_1,
                    CAST(
                        MEDIAN(
                            quantity_net_weight_relative_gap
                        ) AS VARCHAR
                    ) AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({evidence_sql})
                WHERE quantity_net_weight_relative_gap
                    IS NOT NULL
                GROUP BY kg_evidence_status
                """
            )

        # ==============================================================
        # 8. Cumulative hierarchy / coverage contribution
        #
        # This reports the hierarchy actually applied by synchronize.py.
        # ==============================================================

        queries.append(
            f"""
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({synchronized_sql})
            ),
            steps AS (

                SELECT
                    1 AS step_order,
                    'reconciled_agreement' AS step_name,
                    COUNT(*) FILTER (
                        WHERE quantity_sync_status =
                            'reconciled_agreement'
                    ) AS newly_covered
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    2,
                    'quantity_only',
                    COUNT(*) FILTER (
                        WHERE quantity_sync_status =
                            'quantity_only'
                    )
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    3,
                    'net_weight_only',
                    COUNT(*) FILTER (
                        WHERE quantity_sync_status =
                            'net_weight_only'
                    )
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    4,
                    'gross_weight_proxy',
                    COUNT(*) FILTER (
                        WHERE quantity_sync_status =
                            'gross_weight_proxy'
                    )
                FROM read_parquet({synchronized_sql})
            ),
            cumulative AS (
                SELECT
                    step_order,
                    step_name,
                    newly_covered,
                    SUM(newly_covered) OVER (
                        ORDER BY step_order
                    ) AS cumulative_covered
                FROM steps
            )
            SELECT
                'hierarchy_coverage'
                    AS diagnostic_group,
                step_name AS metric,
                CAST(newly_covered AS DOUBLE) AS value,
                100.0 * cumulative_covered
                    / population.n_total
                    AS percentage,
                NULL::INTEGER AS year,
                CAST(step_order AS VARCHAR)
                    AS category_1,
                CAST(cumulative_covered AS VARCHAR)
                    AS category_2,
                CAST(
                    100.0 * newly_covered
                    / population.n_total
                    AS VARCHAR
                ) AS category_3
            FROM cumulative
            CROSS JOIN population
            ORDER BY step_order
            """
        )

        # ==============================================================
        # 9. Gross-weight intervention
        # ==============================================================

        queries.append(
            f"""
            SELECT
                'gross_weight_intervention'
                    AS diagnostic_group,
                metric,
                CAST(metric_value AS DOUBLE) AS value,
                percentage,
                NULL::INTEGER AS year,
                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM (

                SELECT
                    'gross_weight_available' AS metric,
                    COUNT(*) FILTER (
                        WHERE has_gross_weight_kg
                    ) AS metric_value,
                    100.0 * COUNT(*) FILTER (
                        WHERE has_gross_weight_kg
                    ) / COUNT(*) AS percentage
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'gross_weight_proxy_used',
                    COUNT(*) FILTER (
                        WHERE synchronized_quantity_source =
                            'gross_weight_proxy'
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE synchronized_quantity_source =
                            'gross_weight_proxy'
                    ) / COUNT(*)
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'gross_weight_available_after_other_sources_missing',
                    COUNT(*) FILTER (
                        WHERE NOT has_quantity_kg
                          AND NOT has_net_weight_kg
                          AND has_gross_weight_kg
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE NOT has_quantity_kg
                          AND NOT has_net_weight_kg
                          AND has_gross_weight_kg
                    ) / COUNT(*)
                FROM read_parquet({synchronized_sql})
            )
            """
        )

        # ==============================================================
        # 10. Synchronization status
        # ==============================================================

        queries.append(
            f"""
            SELECT
                'synchronization_status'
                    AS diagnostic_group,
                quantity_sync_status AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * COUNT(*)
                    / SUM(COUNT(*)) OVER ()
                    AS percentage,
                NULL::INTEGER AS year,
                synchronized_quantity_source
                    AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM read_parquet({synchronized_sql})
            GROUP BY
                quantity_sync_status,
                synchronized_quantity_source
            """
        )

        # ==============================================================
        # 11. Synchronization outcome by year
        # ==============================================================

        queries.append(
            f"""
            SELECT
                'synchronization_status_by_year'
                    AS diagnostic_group,
                quantity_sync_status AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * COUNT(*)
                    / SUM(COUNT(*)) OVER (
                        PARTITION BY year
                    ) AS percentage,
                CAST(year AS INTEGER) AS year,
                synchronized_quantity_source
                    AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM read_parquet({synchronized_sql})
            GROUP BY
                year,
                quantity_sync_status,
                synchronized_quantity_source
            """
        )

        # ==============================================================
        # 12. Before-versus-after modification
        # ==============================================================

        queries.append(
            f"""
            SELECT
                'synchronization_change'
                    AS diagnostic_group,
                metric,
                CAST(metric_value AS DOUBLE) AS value,
                percentage,
                NULL::INTEGER AS year,
                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM (

                SELECT
                    'synchronized_positive_rows' AS metric,
                    COUNT(*) FILTER (
                        WHERE synchronized_quantity_kg > 0
                    ) AS metric_value,
                    100.0 * COUNT(*) FILTER (
                        WHERE synchronized_quantity_kg > 0
                    ) / COUNT(*) AS percentage
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'unchanged_from_quantity',
                    COUNT(*) FILTER (
                        WHERE synchronized_quantity_kg IS NOT NULL
                          AND quantity_kg_candidate IS NOT NULL
                          AND synchronized_quantity_kg =
                              quantity_kg_candidate
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE synchronized_quantity_kg IS NOT NULL
                          AND quantity_kg_candidate IS NOT NULL
                          AND synchronized_quantity_kg =
                              quantity_kg_candidate
                    ) / COUNT(*)
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'unchanged_from_net_weight',
                    COUNT(*) FILTER (
                        WHERE synchronized_quantity_kg IS NOT NULL
                          AND net_weight_kg_candidate IS NOT NULL
                          AND synchronized_quantity_kg =
                              net_weight_kg_candidate
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE synchronized_quantity_kg IS NOT NULL
                          AND net_weight_kg_candidate IS NOT NULL
                          AND synchronized_quantity_kg =
                              net_weight_kg_candidate
                    ) / COUNT(*)
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'reconciled_mean_rows',
                    COUNT(*) FILTER (
                        WHERE quantity_sync_status =
                            'reconciled_agreement'
                          AND quantity_kg_candidate <>
                              net_weight_kg_candidate
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE quantity_sync_status =
                            'reconciled_agreement'
                          AND quantity_kg_candidate <>
                              net_weight_kg_candidate
                    ) / COUNT(*)
                FROM read_parquet({synchronized_sql})
            )
            """
        )

        # ==============================================================
        # 13. Magnitude of numeric change for reconciled rows
        # ==============================================================

        queries.append(
            f"""
            SELECT
                'reconciliation_change_statistics'
                    AS diagnostic_group,
                metric,
                value,
                NULL::DOUBLE AS percentage,
                NULL::INTEGER AS year,
                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM (

                SELECT
                    'median_relative_change_from_quantity'
                        AS metric,
                    MEDIAN(
                        ABS(
                            synchronized_quantity_kg
                            - quantity_kg_candidate
                        )
                        / quantity_kg_candidate
                    ) AS value
                FROM read_parquet({synchronized_sql})
                WHERE quantity_sync_status =
                    'reconciled_agreement'
                  AND quantity_kg_candidate > 0

                UNION ALL

                SELECT
                    'p95_relative_change_from_quantity',
                    QUANTILE_CONT(
                        ABS(
                            synchronized_quantity_kg
                            - quantity_kg_candidate
                        )
                        / quantity_kg_candidate,
                        0.95
                    )
                FROM read_parquet({synchronized_sql})
                WHERE quantity_sync_status =
                    'reconciled_agreement'
                  AND quantity_kg_candidate > 0

                UNION ALL

                SELECT
                    'median_relative_change_from_net_weight',
                    MEDIAN(
                        ABS(
                            synchronized_quantity_kg
                            - net_weight_kg_candidate
                        )
                        / net_weight_kg_candidate
                    )
                FROM read_parquet({synchronized_sql})
                WHERE quantity_sync_status =
                    'reconciled_agreement'
                  AND net_weight_kg_candidate > 0

                UNION ALL

                SELECT
                    'p95_relative_change_from_net_weight',
                    QUANTILE_CONT(
                        ABS(
                            synchronized_quantity_kg
                            - net_weight_kg_candidate
                        )
                        / net_weight_kg_candidate,
                        0.95
                    )
                FROM read_parquet({synchronized_sql})
                WHERE quantity_sync_status =
                    'reconciled_agreement'
                  AND net_weight_kg_candidate > 0
            )
            """
        )

        # ==============================================================
        # 14. Remaining kg gaps
        # ==============================================================

        queries.append(
            f"""
            SELECT
                'remaining_kg_gap'
                    AS diagnostic_group,
                gap_type AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * COUNT(*)
                    / SUM(COUNT(*)) OVER ()
                    AS percentage,
                NULL::INTEGER AS year,
                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM (
                SELECT
                    CASE
                        WHEN quantity_sync_status =
                            'unresolved_conflict'
                        THEN 'unresolved_conflict'

                        WHEN synchronized_quantity_kg IS NULL
                             AND (
                                 has_alt_quantity
                                 OR has_quantity
                             )
                        THEN 'supplementary_quantity_but_no_kg'

                        WHEN synchronized_quantity_kg IS NULL
                        THEN 'no_physical_quantity_evidence'

                        ELSE 'resolved'
                    END AS gap_type
                FROM read_parquet({synchronized_sql})
            )
            GROUP BY gap_type
            """
        )

        # ==============================================================
        # 15. Unresolved conflict severity
        # ==============================================================

        queries.append(
            f"""
            SELECT
                'unresolved_conflict_statistics'
                    AS diagnostic_group,
                metric,
                value,
                NULL::DOUBLE AS percentage,
                NULL::INTEGER AS year,
                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM (

                SELECT
                    'n_conflicts' AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value
                FROM read_parquet({synchronized_sql})
                WHERE quantity_sync_status =
                    'unresolved_conflict'

                UNION ALL

                SELECT
                    'median_relative_gap',
                    MEDIAN(
                        quantity_net_weight_relative_gap
                    )
                FROM read_parquet({synchronized_sql})
                WHERE quantity_sync_status =
                    'unresolved_conflict'

                UNION ALL

                SELECT
                    'p90_relative_gap',
                    QUANTILE_CONT(
                        quantity_net_weight_relative_gap,
                        0.90
                    )
                FROM read_parquet({synchronized_sql})
                WHERE quantity_sync_status =
                    'unresolved_conflict'

                UNION ALL

                SELECT
                    'p95_relative_gap',
                    QUANTILE_CONT(
                        quantity_net_weight_relative_gap,
                        0.95
                    )
                FROM read_parquet({synchronized_sql})
                WHERE quantity_sync_status =
                    'unresolved_conflict'
            )
            """
        )

        # ==============================================================
        # 16. Conflict distribution by HS
        # ==============================================================

        if hs_column is not None:
            queries.append(
                f"""
                SELECT
                    'unresolved_conflicts_by_hs'
                        AS diagnostic_group,
                    CAST({hs_column} AS VARCHAR) AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER ()
                        AS percentage,
                    NULL::INTEGER AS year,
                    NULL::VARCHAR AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({synchronized_sql})
                WHERE quantity_sync_status =
                    'unresolved_conflict'
                GROUP BY {hs_column}
                """
            )

            queries.append(
                f"""
                SELECT
                    'missing_kg_by_hs'
                        AS diagnostic_group,
                    CAST({hs_column} AS VARCHAR) AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER ()
                        AS percentage,
                    NULL::INTEGER AS year,
                    NULL::VARCHAR AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({synchronized_sql})
                WHERE synchronized_quantity_kg IS NULL
                GROUP BY {hs_column}
                """
            )

        # ==============================================================
        # Write diagnostics
        # ==============================================================

        union_query = "\nUNION ALL\n".join(
            f"({query})"
            for query in queries
        )

        con.execute(
            f"""
            COPY (
                {union_query}
            )
            TO ?
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """,
            [str(output_path)],
        )

        # ==============================================================
        # Final workflow summary
        # ==============================================================

        summary = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,

                COUNT(*) FILTER (
                    WHERE synchronized_quantity_kg > 0
                ) AS n_synchronized,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status =
                        'unresolved_conflict'
                ) AS n_unresolved_conflict,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status =
                        'no_kg_evidence'
                ) AS n_no_kg_evidence,

                COUNT(*) FILTER (
                    WHERE synchronized_quantity_source =
                        'gross_weight_proxy'
                ) AS n_gross_weight_proxy

            FROM read_parquet({synchronized_sql})
            """
        ).fetchone()

        (
            n_rows,
            n_synchronized,
            n_unresolved_conflict,
            n_no_kg_evidence,
            n_gross_weight_proxy,
        ) = summary

        synchronized_pct = (
            100.0 * n_synchronized / n_rows
            if n_rows
            else 0.0
        )

        unresolved_pct = (
            100.0 * n_unresolved_conflict / n_rows
            if n_rows
            else 0.0
        )

        gross_weight_proxy_pct = (
            100.0 * n_gross_weight_proxy / n_rows
            if n_rows
            else 0.0
        )

        n_diagnostic_rows = con.execute(
            """
            SELECT COUNT(*)
            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()[0]

        logger.info(
            "Quantity diagnostic rows written: %s",
            f"{n_diagnostic_rows:,}",
        )
        logger.info(
            "Synchronized kg coverage: %s / %s (%.2f%%)",
            f"{n_synchronized:,}",
            f"{n_rows:,}",
            synchronized_pct,
        )
        logger.info(
            "Unresolved conflicts: %s (%.4f%%)",
            f"{n_unresolved_conflict:,}",
            unresolved_pct,
        )
        logger.info(
            "Rows without kg evidence: %s",
            f"{n_no_kg_evidence:,}",
        )
        logger.info(
            "Gross-weight proxy rows: %s (%.4f%%)",
            f"{n_gross_weight_proxy:,}",
            gross_weight_proxy_pct,
        )
        logger.info(
            "Quantity diagnostics: %s",
            output_path,
        )

    finally:
        con.close()

    if not output_path.is_file():
        raise FileNotFoundError(
            "Quantity diagnostics were not created: "
            f"{output_path}"
        )

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": output_path,
        "diagnostics_path": output_path,

        "n_rows": int(n_rows),
        "n_synchronized": int(n_synchronized),
        "n_unresolved_conflict": int(
            n_unresolved_conflict
        ),
        "n_no_kg_evidence": int(
            n_no_kg_evidence
        ),
        "n_gross_weight_proxy": int(
            n_gross_weight_proxy
        ),

        "synchronized_pct": float(
            synchronized_pct
        ),
        "unresolved_conflict_pct": float(
            unresolved_pct
        ),
        "gross_weight_proxy_pct": float(
            gross_weight_proxy_pct
        ),

        "ready_for_reporting": True,
    }