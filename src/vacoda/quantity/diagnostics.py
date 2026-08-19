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
from vacoda.utils.sql import sql_literal


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

    prepared_sql = sql_literal(prepared_input_path)
    evidence_sql = sql_literal(
        reconciliation_evidence_path
    )
    synchronized_sql = sql_literal(
        synchronized_quantity_path
    )
    output_sql = sql_literal(output_path)

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
                    'total_rows' AS statistic,
                    COUNT(*) AS metric_value,
                    100.0 AS percentage
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
                    'negative_rows',
                    COUNT(*) FILTER (
                        WHERE quantity < 0
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE quantity < 0
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
                    'total_rows',
                    COUNT(*),
                    100.0
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
                    'negative_rows',
                    COUNT(*) FILTER (
                        WHERE alt_quantity < 0
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE alt_quantity < 0
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
                    'total_rows',
                    COUNT(*),
                    100.0
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
                    'negative_rows',
                    COUNT(*) FILTER (
                        WHERE net_weight_kg < 0
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE net_weight_kg < 0
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
                    'total_rows',
                    COUNT(*),
                    100.0
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
                    'negative_rows',
                    COUNT(*) FILTER (
                        WHERE gross_weight_kg < 0
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE gross_weight_kg < 0
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
                    CASE
                        WHEN alt_quantity_unit_code IN (8, 15, 21)
                            THEN 'kg'
                        ELSE COALESCE(
                            CAST(alt_quantity_unit AS VARCHAR),
                            '<NULL>'
                        )
                    END AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER ()
                        AS percentage,
                    NULL::INTEGER AS year,
                    CASE
                        WHEN (
                            alt_quantity_unit_code = 8
                            AND LOWER(TRIM(alt_quantity_unit)) = 'kg'
                        )
                        OR (
                            alt_quantity_unit_code = 21
                            AND UPPER(TRIM(alt_quantity_unit)) = '1000 KG'
                        )
                        OR (
                            alt_quantity_unit_code = 15
                            AND LOWER(TRIM(alt_quantity_unit)) = 'g'
                        )
                            THEN '8'

                        ELSE CAST(alt_quantity_unit_code AS VARCHAR)
                    END AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({prepared_sql})
                GROUP BY
                    CASE
                        WHEN alt_quantity_unit_code IN (8, 15, 21)
                            THEN 'kg'
                        ELSE COALESCE(
                            CAST(alt_quantity_unit AS VARCHAR),
                            '<NULL>'
                        )
                    END,
                    CASE
                        WHEN (
                            alt_quantity_unit_code = 8
                            AND LOWER(TRIM(alt_quantity_unit)) = 'kg'
                        )
                        OR (
                            alt_quantity_unit_code = 21
                            AND UPPER(TRIM(alt_quantity_unit)) = '1000 KG'
                        )
                        OR (
                            alt_quantity_unit_code = 15
                            AND LOWER(TRIM(alt_quantity_unit)) = 'g'
                        )
                            THEN '8'

                        ELSE CAST(alt_quantity_unit_code AS VARCHAR)
                    END
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
                    CASE
                        WHEN (
                            alt_quantity_unit_code = 8
                            AND LOWER(TRIM(alt_quantity_unit)) = 'kg'
                        )
                        OR (
                            alt_quantity_unit_code = 21
                            AND UPPER(TRIM(alt_quantity_unit)) = '1000 KG'
                        )
                        OR (
                            alt_quantity_unit_code = 15
                            AND LOWER(TRIM(alt_quantity_unit)) = 'g'
                        )
                            THEN 'kg'

                        ELSE COALESCE(
                            CAST(alt_quantity_unit AS VARCHAR),
                            '<NULL>'
                        )
                    END AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER (
                            PARTITION BY year
                        ) AS percentage,
                    CAST(year AS INTEGER) AS year,
                    CASE
                        WHEN (
                            alt_quantity_unit_code = 8
                            AND LOWER(TRIM(alt_quantity_unit)) = 'kg'
                        )
                        OR (
                            alt_quantity_unit_code = 21
                            AND UPPER(TRIM(alt_quantity_unit)) = '1000 KG'
                        )
                        OR (
                            alt_quantity_unit_code = 15
                            AND LOWER(TRIM(alt_quantity_unit)) = 'g'
                        )
                            THEN '8'

                        ELSE CAST(alt_quantity_unit_code AS VARCHAR)
                    END AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({prepared_sql})
                GROUP BY
                    year,
                    CASE
                        WHEN (
                        alt_quantity_unit_code = 8
                        AND LOWER(TRIM(alt_quantity_unit)) = 'kg'
                    )
                    OR (
                        alt_quantity_unit_code = 21
                        AND UPPER(TRIM(alt_quantity_unit)) = '1000 KG'
                    )
                    OR (
                        alt_quantity_unit_code = 15
                        AND LOWER(TRIM(alt_quantity_unit)) = 'g'
                    )
                        THEN 'kg'

                    ELSE COALESCE(
                        CAST(alt_quantity_unit AS VARCHAR),
                        '<NULL>'
                    )
                    END,
                    CASE
                        WHEN (
                        alt_quantity_unit_code = 8
                        AND LOWER(TRIM(alt_quantity_unit)) = 'kg'
                    )
                    OR (
                        alt_quantity_unit_code = 21
                        AND UPPER(TRIM(alt_quantity_unit)) = '1000 KG'
                    )
                    OR (
                        alt_quantity_unit_code = 15
                        AND LOWER(TRIM(alt_quantity_unit)) = 'g'
                    )
                        THEN '8'

                    ELSE CAST(alt_quantity_unit_code AS VARCHAR)
                    END
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
                WHERE quantity_unit_code = 12
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
                WHERE alt_quantity_unit_code = 12
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
                        'alt_quantity_kg',
                        COUNT(*) FILTER (
                            WHERE alt_quantity_kg_candidate > 0
                        ),
                        100.0 * COUNT(*) FILTER (
                            WHERE alt_quantity_kg_candidate > 0
                        ) / COUNT(*)
                    FROM read_parquet({evidence_sql})

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
        # 8. Alternate-quantity kg evidence
        # ==============================================================

        queries.append(
            f"""
            SELECT
                'alt_quantity_kg_candidate_coverage'
                    AS diagnostic_group,
                alt_quantity_kg_source AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()
                    AS percentage,
                NULL::INTEGER AS year,
                alt_quantity_kg_status AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM read_parquet({evidence_sql})
            WHERE alt_quantity_kg_candidate > 0
            GROUP BY
                alt_quantity_kg_source,
                alt_quantity_kg_status
            """
        )

        queries.append(
            f"""
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({evidence_sql})
            )
            SELECT
                'alt_quantity_kg_incremental_coverage'
                    AS diagnostic_group,
                alt_quantity_kg_source AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * COUNT(*) / population.n_total
                    AS percentage,
                NULL::INTEGER AS year,
                alt_quantity_kg_status AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM read_parquet({evidence_sql})
            CROSS JOIN population
            WHERE alt_quantity_kg_candidate > 0
              AND NOT COALESCE(quantity_kg_candidate > 0, FALSE)
              AND NOT COALESCE(net_weight_kg_candidate > 0, FALSE)
            GROUP BY
                alt_quantity_kg_source,
                alt_quantity_kg_status,
                population.n_total
            """
        )

        # --------------------------------------------------------------
        # Alt quantity ↔ quantity / net-weight agreement
        # --------------------------------------------------------------

        queries.append(
            f"""
            WITH comparisons AS (
                SELECT
                    'alt_vs_quantity' AS comparison,
                    alt_quantity_kg_candidate AS left_value,
                    quantity_kg_candidate AS right_value,
                    alt_quantity_kg_status AS left_status,
                    quantity_kg_status AS right_status
                FROM read_parquet({evidence_sql})
                WHERE alt_quantity_kg_candidate > 0
                  AND quantity_kg_candidate > 0

                UNION ALL

                SELECT
                    'alt_vs_net_weight',
                    alt_quantity_kg_candidate,
                    net_weight_kg_candidate,
                    alt_quantity_kg_status,
                    net_weight_kg_status
                FROM read_parquet({evidence_sql})
                WHERE alt_quantity_kg_candidate > 0
                  AND net_weight_kg_candidate > 0
            ),
            measured AS (
                SELECT
                    *,
                    left_value / right_value AS ratio,
                    ABS(left_value - right_value)
                    / ((ABS(left_value) + ABS(right_value)) / 2.0)
                        AS relative_gap
                FROM comparisons
            )
            SELECT
                'alt_quantity_kg_agreement'
                    AS diagnostic_group,
                comparison AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * AVG(
                    CASE
                        WHEN relative_gap <= {agreement_tolerance}
                            THEN 1.0
                        ELSE 0.0
                    END
                ) AS percentage,
                NULL::INTEGER AS year,
                left_status || '_' || right_status
                    AS category_1,
                CAST(MEDIAN(ratio) AS VARCHAR)
                    AS category_2,
                CAST(MEDIAN(relative_gap) AS VARCHAR)
                    AS category_3
            FROM measured
            GROUP BY
                comparison,
                left_status,
                right_status
            """
        )

        queries.append(
            f"""
            WITH comparisons AS (
                SELECT
                    'alt_vs_quantity' AS comparison,
                    alt_quantity_kg_candidate AS left_value,
                    quantity_kg_candidate AS right_value
                FROM read_parquet({evidence_sql})
                WHERE alt_quantity_kg_candidate > 0
                  AND quantity_kg_candidate > 0

                UNION ALL

                SELECT
                    'alt_vs_net_weight',
                    alt_quantity_kg_candidate,
                    net_weight_kg_candidate
                FROM read_parquet({evidence_sql})
                WHERE alt_quantity_kg_candidate > 0
                  AND net_weight_kg_candidate > 0
            ),
            measured AS (
                SELECT
                    *,
                    left_value / right_value AS ratio,
                    ABS(left_value - right_value)
                    / ((ABS(left_value) + ABS(right_value)) / 2.0)
                        AS relative_gap
                FROM comparisons
            )
            SELECT
                'alt_quantity_kg_statistics'
                    AS diagnostic_group,
                comparison AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                NULL::DOUBLE AS percentage,
                NULL::INTEGER AS year,
                CAST(MEDIAN(ratio) AS VARCHAR)
                    AS category_1,
                CAST(
                    100.0 * AVG(
                        CASE WHEN relative_gap <= 0.01
                            THEN 1.0 ELSE 0.0 END
                    ) AS VARCHAR
                ) AS category_2,
                CAST(
                    100.0 * AVG(
                        CASE WHEN relative_gap <= 0.05
                            THEN 1.0 ELSE 0.0 END
                    ) AS VARCHAR
                ) AS category_3
            FROM measured
            GROUP BY comparison
            """
        )

        # ==============================================================
        # 9. Net weight with supplementary physical quantities
        # ==============================================================

        queries.append(
            f"""
            SELECT
                'net_weight_quantity_supplementary_overlap'
                    AS diagnostic_group,
                COALESCE(CAST(quantity_unit AS VARCHAR), '<NULL>')
                    AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()
                    AS percentage,
                NULL::INTEGER AS year,
                CAST(quantity_unit_code AS VARCHAR)
                    AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM read_parquet({prepared_sql})
            WHERE net_weight_kg > 0
              AND quantity > 0
              AND quantity_unit_code <> 8
            GROUP BY
                quantity_unit,
                quantity_unit_code
            """
        )

        queries.append(
            f"""
            SELECT
                'net_weight_alt_quantity_supplementary_overlap'
                    AS diagnostic_group,
                COALESCE(CAST(alt_quantity_unit AS VARCHAR), '<NULL>')
                    AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()
                    AS percentage,
                NULL::INTEGER AS year,
                CAST(alt_quantity_unit_code AS VARCHAR)
                    AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM read_parquet({prepared_sql})
            WHERE net_weight_kg > 0
              AND alt_quantity > 0
              AND NOT (
                    (
                        alt_quantity_unit_code = 8
                        AND LOWER(TRIM(alt_quantity_unit)) = 'kg'
                    )
                    OR (
                        alt_quantity_unit_code = 21
                        AND UPPER(TRIM(alt_quantity_unit)) = '1000 KG'
                    )
                    OR (
                        alt_quantity_unit_code = 15
                        AND LOWER(TRIM(alt_quantity_unit)) = 'g'
                    )
                )
            GROUP BY
                alt_quantity_unit,
                alt_quantity_unit_code
            """
        )

        if hs_column is not None:
            queries.append(
                f"""
                SELECT
                    'net_weight_supplementary_by_hs'
                        AS diagnostic_group,
                    CAST({hs_column} AS VARCHAR) AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER (
                            PARTITION BY quantity_unit_code
                        ) AS percentage,
                    NULL::INTEGER AS year,
                    CAST(quantity_unit_code AS VARCHAR)
                        AS category_1,
                    CAST(quantity_unit AS VARCHAR)
                        AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({prepared_sql})
                WHERE net_weight_kg > 0
                  AND quantity > 0
                  AND quantity_unit_code IN (2, 5, 12)
                GROUP BY
                    {hs_column},
                    quantity_unit_code,
                    quantity_unit
                """
            )

        # ==============================================================
        # 10. Net-weight diagnostics
        # ==============================================================

        queries.append(
            f"""
            SELECT
                'net_weight_by_year' AS diagnostic_group,
                CASE
                    WHEN COALESCE(is_net_weight_estimated, FALSE)
                        THEN 'estimated'
                    ELSE 'reported'
                END AS metric,
                CAST(COUNT(*) FILTER (
                    WHERE net_weight_kg > 0
                ) AS DOUBLE) AS value,
                100.0 * COUNT(*) FILTER (
                    WHERE net_weight_kg > 0
                ) / COUNT(*) AS percentage,
                CAST(year AS INTEGER) AS year,
                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM read_parquet({prepared_sql})
            GROUP BY
                year,
                CASE
                    WHEN COALESCE(is_net_weight_estimated, FALSE)
                        THEN 'estimated'
                    ELSE 'reported'
                END
            """
        )

        queries.append(
            f"""
            SELECT
                'net_weight_distribution_by_year'
                    AS diagnostic_group,
                'log10_net_weight_kg' AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                NULL::DOUBLE AS percentage,
                CAST(year AS INTEGER) AS year,
                CAST(QUANTILE_CONT(LOG10(net_weight_kg), 0.10)
                    AS VARCHAR) AS category_1,
                CAST(MEDIAN(LOG10(net_weight_kg))
                    AS VARCHAR) AS category_2,
                CAST(QUANTILE_CONT(LOG10(net_weight_kg), 0.90)
                    AS VARCHAR) AS category_3
            FROM read_parquet({prepared_sql})
            WHERE net_weight_kg > 0
            GROUP BY year
            """
        )

        # ==============================================================
        # 11. Cumulative hierarchy / coverage contribution
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
        # 12. Gross-weight diagnostics
        # ==============================================================

        queries.append(
            f"""
            SELECT
                'gross_weight_intervention' AS diagnostic_group,
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
                        WHERE gross_weight_kg > 0
                    ) AS metric_value,
                    100.0 * COUNT(*) FILTER (
                        WHERE gross_weight_kg > 0
                    ) / COUNT(*) AS percentage
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'gross_weight_with_net_weight',
                    COUNT(*) FILTER (
                        WHERE gross_weight_kg > 0
                          AND net_weight_kg > 0
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE gross_weight_kg > 0
                          AND net_weight_kg > 0
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'gross_weight_without_net_weight',
                    COUNT(*) FILTER (
                        WHERE gross_weight_kg > 0
                          AND NOT COALESCE(net_weight_kg > 0, FALSE)
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE gross_weight_kg > 0
                          AND NOT COALESCE(net_weight_kg > 0, FALSE)
                    ) / COUNT(*)
                FROM read_parquet({prepared_sql})

                UNION ALL

                SELECT
                    'gross_weight_available_after_quantity_kg_and_net_weight_missing',
                    COUNT(*) FILTER (
                        WHERE gross_weight_kg > 0
                          AND quantity_kg_candidate IS NULL
                          AND net_weight_kg_candidate IS NULL
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE gross_weight_kg > 0
                          AND quantity_kg_candidate IS NULL
                          AND net_weight_kg_candidate IS NULL
                    ) / COUNT(*)
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'gross_weight_available_after_quantity_kg_net_weight_and_alt_kg_missing',
                    COUNT(*) FILTER (
                        WHERE gross_weight_kg > 0
                          AND quantity_kg_candidate IS NULL
                          AND net_weight_kg_candidate IS NULL
                          AND alt_quantity_kg_candidate IS NULL
                    ),
                    100.0 * COUNT(*) FILTER (
                        WHERE gross_weight_kg > 0
                          AND quantity_kg_candidate IS NULL
                          AND net_weight_kg_candidate IS NULL
                          AND alt_quantity_kg_candidate IS NULL
                    ) / COUNT(*)
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
            )
            """
        )

        queries.append(
            f"""
            SELECT
                'gross_net_weight_relationship' AS diagnostic_group,
                metric,
                value,
                CASE
                    WHEN metric = 'gross_below_net_rows'
                    THEN 100.0 * value / NULLIF(
                        MAX(CASE WHEN metric = 'comparable_rows'
                            THEN value END) OVER (),
                        0
                    )
                    ELSE NULL
                END AS percentage,
                NULL::INTEGER AS year,
                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM (
                SELECT
                    'comparable_rows' AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value
                FROM read_parquet({prepared_sql})
                WHERE gross_weight_kg > 0
                  AND net_weight_kg > 0

                UNION ALL

                SELECT
                    'gross_below_net_rows',
                    CAST(COUNT(*) FILTER (
                        WHERE gross_weight_kg < net_weight_kg
                    ) AS DOUBLE)
                FROM read_parquet({prepared_sql})
                WHERE gross_weight_kg > 0
                  AND net_weight_kg > 0

                UNION ALL

                SELECT
                    'median_ratio',
                    MEDIAN(gross_weight_kg / net_weight_kg)
                FROM read_parquet({prepared_sql})
                WHERE gross_weight_kg > 0
                  AND net_weight_kg > 0

                UNION ALL

                SELECT
                    'p10_ratio',
                    QUANTILE_CONT(
                        gross_weight_kg / net_weight_kg, 0.10
                    )
                FROM read_parquet({prepared_sql})
                WHERE gross_weight_kg > 0
                  AND net_weight_kg > 0

                UNION ALL

                SELECT
                    'p90_ratio',
                    QUANTILE_CONT(
                        gross_weight_kg / net_weight_kg, 0.90
                    )
                FROM read_parquet({prepared_sql})
                WHERE gross_weight_kg > 0
                  AND net_weight_kg > 0
            )
            """
        )

        queries.append(
            f"""
            SELECT
                'gross_net_weight_by_evidence_state'
                    AS diagnostic_group,
                CASE
                    WHEN NOT COALESCE(
                        is_gross_weight_estimated, FALSE
                    ) AND NOT COALESCE(
                        is_net_weight_estimated, FALSE
                    ) THEN 'reported_reported'
                    WHEN NOT COALESCE(
                        is_gross_weight_estimated, FALSE
                    ) AND COALESCE(
                        is_net_weight_estimated, FALSE
                    ) THEN 'reported_estimated'
                    WHEN COALESCE(
                        is_gross_weight_estimated, FALSE
                    ) AND NOT COALESCE(
                        is_net_weight_estimated, FALSE
                    ) THEN 'estimated_reported'
                    ELSE 'estimated_estimated'
                END AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * COUNT(*) FILTER (
                    WHERE gross_weight_kg < net_weight_kg
                ) / COUNT(*) AS percentage,
                NULL::INTEGER AS year,
                CAST(MEDIAN(
                    gross_weight_kg / net_weight_kg
                ) AS VARCHAR) AS category_1,
                CAST(QUANTILE_CONT(
                    gross_weight_kg / net_weight_kg, 0.10
                ) AS VARCHAR) AS category_2,
                CAST(QUANTILE_CONT(
                    gross_weight_kg / net_weight_kg, 0.90
                ) AS VARCHAR) AS category_3
            FROM read_parquet({prepared_sql})
            WHERE gross_weight_kg > 0
              AND net_weight_kg > 0
            GROUP BY
                CASE
                    WHEN NOT COALESCE(
                        is_gross_weight_estimated, FALSE
                    ) AND NOT COALESCE(
                        is_net_weight_estimated, FALSE
                    ) THEN 'reported_reported'
                    WHEN NOT COALESCE(
                        is_gross_weight_estimated, FALSE
                    ) AND COALESCE(
                        is_net_weight_estimated, FALSE
                    ) THEN 'reported_estimated'
                    WHEN COALESCE(
                        is_gross_weight_estimated, FALSE
                    ) AND NOT COALESCE(
                        is_net_weight_estimated, FALSE
                    ) THEN 'estimated_reported'
                    ELSE 'estimated_estimated'
                END
            """
        )

        queries.append(
            f"""
            SELECT
                'gross_weight_without_net_by_year'
                    AS diagnostic_group,
                'gross_without_net' AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()
                    AS percentage,
                CAST(year AS INTEGER) AS year,
                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM read_parquet({prepared_sql})
            WHERE gross_weight_kg > 0
              AND NOT COALESCE(net_weight_kg > 0, FALSE)
            GROUP BY year
            """
        )

        if hs_column is not None:
            queries.append(
                f"""
                SELECT
                    'gross_weight_without_net_by_hs'
                        AS diagnostic_group,
                    CAST({hs_column} AS VARCHAR) AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()
                        AS percentage,
                    NULL::INTEGER AS year,
                    NULL::VARCHAR AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3
                FROM read_parquet({prepared_sql})
                WHERE gross_weight_kg > 0
                  AND NOT COALESCE(net_weight_kg > 0, FALSE)
                GROUP BY {hs_column}
                """
            )

        # ==============================================================
        # 13. Synchronization status
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
        # 14. Synchronization outcome by year
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
        # 15. Before-versus-after modification
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
        # 16. Magnitude of numeric change for reconciled rows
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
        # 17. Remaining kg gaps
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
        # 18. Unresolved conflict severity
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
        # 19. Conflict distribution by HS
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
            TO {output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
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
            "Rows unresolved by current quantity/net-weight synchronization: %s",
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