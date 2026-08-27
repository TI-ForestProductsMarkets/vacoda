"""
vacoda.quantity.diagnostics

Build report-grade diagnostics for quantity synchronization.

This module summarizes:

- quantity-field availability and coverage;
- unit composition of quantity and alt_quantity;
- cubic-metre reporting by HS code;
- reported-versus-estimated kg evidence;
- quantity-versus-net-weight agreement;
- the adopted synchronization-methodology funnel;
- detailed row counts for each applied synchronization rule;
- gross-weight diagnostics;
- changes introduced by synchronization;
- remaining missing kg evidence;
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
from vacoda.pipeline.checkpoint import (
    validate_local_checkpoint,
    file_fingerprint,
    write_local_checkpoint,
    semantic_config_fingerprint,
)

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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ## Local checkpointing
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
        "prepared_input": file_fingerprint(
            prepared_input_path,
            hash_file=True,
        ),
        "reconciliation_evidence": file_fingerprint(
            reconciliation_evidence_path,
            hash_file=True,
        ),
        "synchronized_quantity": file_fingerprint(
            synchronized_quantity_path,
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
            "REUSE — Quantity diagnostics local checkpoint matches: %s",
            output_path,
        )

        return {
            "status": "reused",
            "run_id": run_id,
            "output_path": output_path,
            "diagnostics_path": output_path,
            "audit_passed": True,
        }

    logger.info(
        "REBUILD — Quantity diagnostics: %s",
        "forced" if force else reason,
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
        #    - HS6 detailed composition
        #    - HS4 broad product-family composition
        # ==============================================================

        if hs_column is not None:

            # ----------------------------------------------------------
            # 4a. quantity: m³ distribution by HS6
            # ----------------------------------------------------------
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
                    'hs6' AS category_3
                FROM read_parquet({prepared_sql})
                WHERE quantity_unit_code = 12
                AND quantity > 0
                GROUP BY {hs_column}
                """
            )

            # ----------------------------------------------------------
            # 4b. alt_quantity: m³ distribution by HS6
            # ----------------------------------------------------------
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
                    'hs6' AS category_3
                FROM read_parquet({prepared_sql})
                WHERE alt_quantity_unit_code = 12
                AND alt_quantity > 0
                GROUP BY {hs_column}
                """
            )

            # ----------------------------------------------------------
            # 4c. quantity: m³ distribution by HS4
            # ----------------------------------------------------------
            queries.append(
                f"""
                SELECT
                    'quantity_m3_by_hs4'
                        AS diagnostic_group,
                    LEFT(
                        LPAD(CAST({hs_column} AS VARCHAR), 6, '0'),
                        4
                    ) AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER ()
                        AS percentage,
                    NULL::INTEGER AS year,
                    'quantity' AS category_1,
                    'm3' AS category_2,
                    'hs4' AS category_3
                FROM read_parquet({prepared_sql})
                WHERE quantity_unit_code = 12
                AND quantity > 0
                GROUP BY
                    LEFT(
                        LPAD(CAST({hs_column} AS VARCHAR), 6, '0'),
                        4
                    )
                """
            )

            # ----------------------------------------------------------
            # 4d. alt_quantity: m³ distribution by HS4
            # ----------------------------------------------------------
            queries.append(
                f"""
                SELECT
                    'alt_quantity_m3_by_hs4'
                        AS diagnostic_group,
                    LEFT(
                        LPAD(CAST({hs_column} AS VARCHAR), 6, '0'),
                        4
                    ) AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER ()
                        AS percentage,
                    NULL::INTEGER AS year,
                    'alt_quantity' AS category_1,
                    'm3' AS category_2,
                    'hs4' AS category_3
                FROM read_parquet({prepared_sql})
                WHERE alt_quantity_unit_code = 12
                AND alt_quantity > 0
                GROUP BY
                    LEFT(
                        LPAD(CAST({hs_column} AS VARCHAR), 6, '0'),
                        4
                    )
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
                        )
                """
            )

            queries.append(
                f"""
                WITH population AS (
                    SELECT COUNT(*) AS n_total
                    FROM read_parquet({evidence_sql})
                ),

                comparison AS (
                    SELECT
                        quantity_net_weight_relative_gap
                    FROM read_parquet({evidence_sql})
                    WHERE quantity_net_weight_relative_gap IS NOT NULL
                )

                SELECT
                    'quantity_net_weight_population_agreement'
                        AS diagnostic_group,
                    metric,
                    value,
                    percentage,
                    NULL::INTEGER AS year,
                    NULL::VARCHAR AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3

                FROM (

                    SELECT
                        'comparable' AS metric,
                        CAST(COUNT(*) AS DOUBLE) AS value,
                        100.0 * COUNT(*) / population.n_total
                            AS percentage
                    FROM comparison
                    CROSS JOIN population
                    GROUP BY population.n_total

                    UNION ALL

                    SELECT
                        'within_1pct',
                        CAST(
                            COUNT(*) FILTER (
                                WHERE quantity_net_weight_relative_gap <= 0.01
                            ) AS DOUBLE
                        ),
                        100.0 * COUNT(*) FILTER (
                            WHERE quantity_net_weight_relative_gap <= 0.01
                        ) / population.n_total
                    FROM comparison
                    CROSS JOIN population
                    GROUP BY population.n_total

                    UNION ALL

                    SELECT
                        'within_5pct',
                        CAST(
                            COUNT(*) FILTER (
                                WHERE quantity_net_weight_relative_gap <= 0.05
                            ) AS DOUBLE
                        ),
                        100.0 * COUNT(*) FILTER (
                            WHERE quantity_net_weight_relative_gap <= 0.05
                        ) / population.n_total
                    FROM comparison
                    CROSS JOIN population
                    GROUP BY population.n_total

                    UNION ALL

                    SELECT
                        'within_10pct',
                        CAST(
                            COUNT(*) FILTER (
                                WHERE quantity_net_weight_relative_gap <= 0.10
                            ) AS DOUBLE
                        ),
                        100.0 * COUNT(*) FILTER (
                            WHERE quantity_net_weight_relative_gap <= 0.10
                        ) / population.n_total
                    FROM comparison
                    CROSS JOIN population
                    GROUP BY population.n_total
                )
                """
            )

            # ----------------------------------------------------------
            # Agreement by year
            # ----------------------------------------------------------

            queries.append(
                f"""
                WITH population AS (
                    SELECT COUNT(*) AS n_total
                    FROM read_parquet({evidence_sql})
                )

                SELECT
                    'quantity_net_weight_by_year'
                        AS diagnostic_group,

                    'within_tolerance' AS metric,

                    CAST(
                        COUNT(*) FILTER (
                            WHERE quantity_net_weight_relative_gap
                                <= {agreement_tolerance}
                        ) AS DOUBLE
                    ) AS value,

                    100.0 * COUNT(*) FILTER (
                        WHERE quantity_net_weight_relative_gap
                            <= {agreement_tolerance}
                    ) / population.n_total
                        AS percentage,

                    CAST(year AS INTEGER) AS year,

                    CAST(
                        MEDIAN(
                            quantity_net_weight_relative_gap
                        ) AS VARCHAR
                    ) AS category_1,

                    CAST(COUNT(*) AS VARCHAR)
                        AS category_2,

                    NULL::VARCHAR AS category_3

                FROM read_parquet({evidence_sql})
                CROSS JOIN population

                WHERE quantity_net_weight_relative_gap IS NOT NULL

                GROUP BY
                    year,
                    population.n_total
                """
            )

            # ----------------------------------------------------------
            # Quantity/net-weight comparison by evidence state
            # All percentages use the full Quantity population.
            # ----------------------------------------------------------

            queries.append(
                f"""
                WITH population AS (
                    SELECT COUNT(*) AS n_total
                    FROM read_parquet({evidence_sql})
                )

                SELECT
                    'quantity_net_weight_by_evidence_state'
                        AS diagnostic_group,

                    kg_evidence_status AS metric,

                    CAST(COUNT(*) AS DOUBLE) AS value,

                    100.0 * COUNT(*) / population.n_total
                        AS percentage,

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
                CROSS JOIN population

                WHERE quantity_net_weight_relative_gap IS NOT NULL

                GROUP BY
                    kg_evidence_status,
                    population.n_total
                """
            )

            queries.append(
                f"""
                WITH population AS (
                    SELECT COUNT(*) AS n_total
                    FROM read_parquet({evidence_sql})
                )

                SELECT
                    'quantity_net_weight_exact_match_by_evidence_state'
                        AS diagnostic_group,

                    kg_evidence_status AS metric,

                    CAST(
                        COUNT(*) FILTER (
                            WHERE quantity_kg_candidate
                                = net_weight_kg_candidate
                        ) AS DOUBLE
                    ) AS value,

                    100.0 * COUNT(*) FILTER (
                        WHERE quantity_kg_candidate
                            = net_weight_kg_candidate
                    ) / population.n_total
                        AS percentage,

                    NULL::INTEGER AS year,

                    NULL::VARCHAR AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3

                FROM read_parquet({evidence_sql})
                CROSS JOIN population

                WHERE quantity_net_weight_relative_gap IS NOT NULL

                GROUP BY
                    kg_evidence_status,
                    population.n_total
                """
            )


            queries.append(
                f"""
                WITH population AS (
                    SELECT COUNT(*) AS n_total
                    FROM read_parquet({evidence_sql})
                )

                SELECT
                    'quantity_net_weight_within_tolerance_by_evidence_state'
                        AS diagnostic_group,

                    kg_evidence_status AS metric,

                    CAST(
                        COUNT(*) FILTER (
                            WHERE quantity_net_weight_relative_gap
                                <= {agreement_tolerance}
                        ) AS DOUBLE
                    ) AS value,

                    100.0 * COUNT(*) FILTER (
                        WHERE quantity_net_weight_relative_gap
                            <= {agreement_tolerance}
                    ) / population.n_total
                        AS percentage,

                    NULL::INTEGER AS year,

                    NULL::VARCHAR AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3

                FROM read_parquet({evidence_sql})
                CROSS JOIN population

                WHERE quantity_net_weight_relative_gap IS NOT NULL

                GROUP BY
                    kg_evidence_status,
                    population.n_total
                """
            )
        # ==============================================================
        # 8. Alternate-quantity kg evidence
        # ==============================================================

        queries.append(
            f"""
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({evidence_sql})
            )

            SELECT
                'alt_quantity_kg_candidate_coverage'
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

            GROUP BY
                alt_quantity_kg_source,
                alt_quantity_kg_status,
                population.n_total
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
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({evidence_sql})
            ),

            comparisons AS (
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

            -- ----------------------------------------------------------
            -- Rows available for comparison
            -- ----------------------------------------------------------
            SELECT
                'alt_quantity_kg_agreement'
                    AS diagnostic_group,
                comparison AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * COUNT(*) / population.n_total
                    AS percentage,
                NULL::INTEGER AS year,
                left_status || '_' || right_status
                    AS category_1,
                CAST(MEDIAN(ratio) AS VARCHAR)
                    AS category_2,
                CAST(MEDIAN(relative_gap) AS VARCHAR)
                    AS category_3

            FROM measured
            CROSS JOIN population

            GROUP BY
                comparison,
                left_status,
                right_status,
                population.n_total

            UNION ALL

            -- ----------------------------------------------------------
            -- Exact matches
            -- ----------------------------------------------------------
            SELECT
                'alt_quantity_kg_exact_match'
                    AS diagnostic_group,
                comparison AS metric,
                CAST(
                    COUNT(*) FILTER (
                        WHERE left_value = right_value
                    ) AS DOUBLE
                ) AS value,
                100.0 * COUNT(*) FILTER (
                    WHERE left_value = right_value
                ) / population.n_total
                    AS percentage,
                NULL::INTEGER AS year,
                left_status || '_' || right_status
                    AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3

            FROM measured
            CROSS JOIN population

            GROUP BY
                comparison,
                left_status,
                right_status,
                population.n_total

            UNION ALL

            -- ----------------------------------------------------------
            -- Agreement within configured tolerance
            -- ----------------------------------------------------------
            SELECT
                'alt_quantity_kg_within_tolerance'
                    AS diagnostic_group,
                comparison AS metric,
                CAST(
                    COUNT(*) FILTER (
                        WHERE relative_gap <= {agreement_tolerance}
                    ) AS DOUBLE
                ) AS value,
                100.0 * COUNT(*) FILTER (
                    WHERE relative_gap <= {agreement_tolerance}
                ) / population.n_total
                    AS percentage,
                NULL::INTEGER AS year,
                left_status || '_' || right_status
                    AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3

            FROM measured
            CROSS JOIN population

            GROUP BY
                comparison,
                left_status,
                right_status,
                population.n_total
            """
        )

        # ==============================================================
        # Three-source kg consensus + provenance diagnostics
        # ==============================================================

        queries.append(
            f"""
            WITH three_source AS (
                SELECT
                    quantity_kg_candidate AS q,
                    net_weight_kg_candidate AS n,
                    alt_quantity_kg_candidate AS a,

                    quantity_kg_status AS q_status,
                    net_weight_kg_status AS n_status,
                    alt_quantity_kg_status AS a_status,

                    ABS(
                        quantity_kg_candidate
                        - net_weight_kg_candidate
                    )
                    / (
                        (
                            ABS(quantity_kg_candidate)
                            + ABS(net_weight_kg_candidate)
                        ) / 2.0
                    ) AS q_n_gap,

                    ABS(
                        quantity_kg_candidate
                        - alt_quantity_kg_candidate
                    )
                    / (
                        (
                            ABS(quantity_kg_candidate)
                            + ABS(alt_quantity_kg_candidate)
                        ) / 2.0
                    ) AS q_a_gap,

                    ABS(
                        net_weight_kg_candidate
                        - alt_quantity_kg_candidate
                    )
                    / (
                        (
                            ABS(net_weight_kg_candidate)
                            + ABS(alt_quantity_kg_candidate)
                        ) / 2.0
                    ) AS n_a_gap

                FROM read_parquet({evidence_sql})

                WHERE quantity_kg_candidate > 0
                  AND net_weight_kg_candidate > 0
                  AND alt_quantity_kg_candidate > 0
            ),

            classified AS (
                SELECT
                    *,
                    CASE
                        WHEN q_n_gap <= {agreement_tolerance}
                         AND q_a_gap <= {agreement_tolerance}
                         AND n_a_gap <= {agreement_tolerance}
                            THEN 'all_three_agree'

                        WHEN q_n_gap <= {agreement_tolerance}
                         AND q_a_gap > {agreement_tolerance}
                         AND n_a_gap > {agreement_tolerance}
                            THEN 'quantity_net_agree_alt_outlier'

                        WHEN q_a_gap <= {agreement_tolerance}
                         AND q_n_gap > {agreement_tolerance}
                         AND n_a_gap > {agreement_tolerance}
                            THEN 'quantity_alt_agree_net_outlier'

                        WHEN n_a_gap <= {agreement_tolerance}
                         AND q_n_gap > {agreement_tolerance}
                         AND q_a_gap > {agreement_tolerance}
                            THEN 'net_alt_agree_quantity_outlier'

                        ELSE 'no_pair_agrees'
                    END AS consensus_status

                FROM three_source
            ),

            consensus_summary AS (
                SELECT
                    consensus_status,
                    COUNT(*) AS n_rows
                FROM classified
                GROUP BY consensus_status
            ),

            provenance_summary AS (
                SELECT
                    consensus_status,
                    COALESCE(q_status, 'unknown') AS q_status,
                    COALESCE(n_status, 'unknown') AS n_status,
                    COALESCE(a_status, 'unknown') AS a_status,
                    COUNT(*) AS n_rows
                FROM classified
                GROUP BY
                    consensus_status,
                    COALESCE(q_status, 'unknown'),
                    COALESCE(n_status, 'unknown'),
                    COALESCE(a_status, 'unknown')
            )

            -- ----------------------------------------------------------
            -- Overall three-source consensus
            -- ----------------------------------------------------------

            SELECT
                'three_source_consensus'
                    AS diagnostic_group,

                consensus_status AS metric,

                CAST(n_rows AS DOUBLE) AS value,

                100.0 * n_rows
                    / SUM(n_rows) OVER ()
                    AS percentage,

                NULL::INTEGER AS year,

                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3

            FROM consensus_summary

            UNION ALL

            -- ----------------------------------------------------------
            -- Provenance within each consensus outcome
            -- ----------------------------------------------------------

            SELECT
                'three_source_consensus_by_evidence_state'
                    AS diagnostic_group,

                p.consensus_status AS metric,

                CAST(p.n_rows AS DOUBLE) AS value,

                100.0 * p.n_rows
                    / NULLIF(c.n_rows, 0)
                    AS percentage,

                NULL::INTEGER AS year,

                p.q_status AS category_1,
                p.n_status AS category_2,
                p.a_status AS category_3

            FROM provenance_summary AS p

            INNER JOIN consensus_summary AS c
                ON p.consensus_status = c.consensus_status
            """
        )


        # ==============================================================
        # 9. Net weight with supplementary physical quantities
        # ==============================================================

        queries.append(
            f"""
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({prepared_sql})
            ),

            overlap AS (
                SELECT
                    quantity_unit,
                    quantity_unit_code,
                    COUNT(*) AS n_rows
                FROM read_parquet({prepared_sql})
                WHERE net_weight_kg > 0
                AND quantity > 0
                AND quantity_unit_code <> 8
                GROUP BY
                    quantity_unit,
                    quantity_unit_code
            ),

            overlap_total AS (
                SELECT SUM(n_rows) AS n_overlap
                FROM overlap
            )

            SELECT
                'net_weight_quantity_supplementary_overlap'
                    AS diagnostic_group,

                COALESCE(
                    CAST(quantity_unit AS VARCHAR),
                    '<NULL>'
                ) AS metric,

                CAST(n_rows AS DOUBLE) AS value,

                100.0 * n_rows / population.n_total
                    AS percentage,

                NULL::INTEGER AS year,

                CAST(quantity_unit_code AS VARCHAR)
                    AS category_1,

                CAST(
                    100.0 * n_rows
                    / NULLIF(overlap_total.n_overlap, 0)
                    AS VARCHAR
                ) AS category_2,

                NULL::VARCHAR AS category_3

            FROM overlap
            CROSS JOIN population
            CROSS JOIN overlap_total
            """
        )

        queries.append(
            f"""
            WITH examples AS (
                SELECT
                    CAST({hs_column} AS VARCHAR) AS hs_code,
                    CAST(year AS INTEGER) AS year,
                    net_weight_kg,
                    quantity,
                    quantity_unit,
                    ROW_NUMBER() OVER (
                        PARTITION BY quantity_unit_code
                        ORDER BY
                            year DESC,
                            {hs_column},
                            net_weight_kg DESC
                    ) AS rn
                FROM read_parquet({prepared_sql})
                WHERE net_weight_kg > 0
                AND quantity > 0
                AND quantity_unit_code IN (2, 5, 12)
            )

            SELECT
                'net_weight_quantity_supplementary_examples'
                    AS diagnostic_group,

                COALESCE(quantity_unit, '<NULL>') AS metric,

                CAST(quantity AS DOUBLE) AS value,

                NULL::DOUBLE AS percentage,

                year,

                hs_code AS category_1,

                CAST(net_weight_kg AS VARCHAR)
                    AS category_2,

                CAST(quantity AS VARCHAR)
                    AS category_3

            FROM examples
            WHERE rn = 1
            """
        )


        # --------------------------------------------------------------
        # Net-weight relationship for quantity reported in m³, by HS6
        # --------------------------------------------------------------

        if hs_column is not None:
            queries.append(
                f"""
                WITH comparable AS (
                    SELECT
                        CAST({hs_column} AS VARCHAR) AS hs_code,
                        net_weight_kg / quantity AS kg_per_m3
                    FROM read_parquet({prepared_sql})
                    WHERE net_weight_kg > 0
                    AND quantity > 0
                    AND quantity_unit_code = 12
                    AND isfinite(net_weight_kg / quantity)
                )

                SELECT
                    'net_weight_quantity_m3_by_hs'
                        AS diagnostic_group,

                    hs_code AS metric,

                    CAST(COUNT(*) AS DOUBLE) AS value,

                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER ()
                        AS percentage,

                    NULL::INTEGER AS year,

                    CAST(
                        MEDIAN(kg_per_m3)
                        AS VARCHAR
                    ) AS category_1,

                    CAST(
                        QUANTILE_CONT(kg_per_m3, 0.10)
                        AS VARCHAR
                    ) AS category_2,

                    CAST(
                        QUANTILE_CONT(kg_per_m3, 0.90)
                        AS VARCHAR
                    ) AS category_3

                FROM comparable
                GROUP BY hs_code
                """
            )


        queries.append(
            f"""
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({prepared_sql})
            ),

            overlap AS (
                SELECT
                    alt_quantity_unit,
                    alt_quantity_unit_code,
                    COUNT(*) AS n_rows
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
            ),

            overlap_total AS (
                SELECT SUM(n_rows) AS n_overlap
                FROM overlap
            )

            SELECT
                'net_weight_alt_quantity_supplementary_overlap'
                    AS diagnostic_group,

                COALESCE(
                    CAST(alt_quantity_unit AS VARCHAR),
                    '<NULL>'
                ) AS metric,

                CAST(n_rows AS DOUBLE) AS value,

                100.0 * n_rows / population.n_total
                    AS percentage,

                NULL::INTEGER AS year,

                CAST(alt_quantity_unit_code AS VARCHAR)
                    AS category_1,

                CAST(
                    100.0 * n_rows
                    / NULLIF(overlap_total.n_overlap, 0)
                    AS VARCHAR
                ) AS category_2,

                NULL::VARCHAR AS category_3

            FROM overlap
            CROSS JOIN population
            CROSS JOIN overlap_total
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


        # --------------------------------------------------------------
        # Net-weight relationship for alt_quantity reported in m³,
        # by HS6
        # --------------------------------------------------------------

        if hs_column is not None:
            queries.append(
                f"""
                WITH comparable AS (
                    SELECT
                        CAST({hs_column} AS VARCHAR) AS hs_code,
                        net_weight_kg / alt_quantity AS kg_per_m3
                    FROM read_parquet({prepared_sql})
                    WHERE net_weight_kg > 0
                    AND alt_quantity > 0
                    AND alt_quantity_unit_code = 12
                    AND isfinite(net_weight_kg / alt_quantity)
                )

                SELECT
                    'net_weight_alt_quantity_m3_by_hs'
                        AS diagnostic_group,

                    hs_code AS metric,

                    CAST(COUNT(*) AS DOUBLE) AS value,

                    100.0 * COUNT(*)
                        / SUM(COUNT(*)) OVER ()
                        AS percentage,

                    NULL::INTEGER AS year,

                    CAST(
                        MEDIAN(kg_per_m3)
                        AS VARCHAR
                    ) AS category_1,

                    CAST(
                        QUANTILE_CONT(kg_per_m3, 0.10)
                        AS VARCHAR
                    ) AS category_2,

                    CAST(
                        QUANTILE_CONT(kg_per_m3, 0.90)
                        AS VARCHAR
                    ) AS category_3

                FROM comparable
                GROUP BY hs_code
                """
            )

        # ==============================================================
        # Supplementary quantity candidate overlap
        # ==============================================================

        queries.append(
            f"""
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({evidence_sql})
            ),

            candidates AS (
                SELECT
                    *,
                    (
                        quantity > 0
                        AND quantity_unit_code <> 8
                        AND quantity_unit_code <> -1
                    ) AS has_quantity_nonkg,

                    (
                        alt_quantity > 0
                        AND alt_quantity_unit_code <> -1
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
                    ) AS has_alt_nonkg
                FROM read_parquet({evidence_sql})
            ),

            classified AS (
                SELECT
                    *,
                    CASE
                        WHEN has_quantity_nonkg
                             AND has_alt_nonkg
                             AND quantity_unit_code = alt_quantity_unit_code
                            THEN 'both_same_unit'
                        WHEN has_quantity_nonkg
                             AND has_alt_nonkg
                             AND quantity_unit_code <> alt_quantity_unit_code
                            THEN 'both_different_unit'
                        WHEN has_quantity_nonkg
                             AND NOT has_alt_nonkg
                            THEN 'quantity_only'
                        WHEN NOT has_quantity_nonkg
                             AND has_alt_nonkg
                            THEN 'alt_quantity_only'
                        ELSE 'neither'
                    END AS supplementary_overlap_status
                FROM candidates
            )

            SELECT
                'supplementary_candidate_overlap' AS diagnostic_group,
                supplementary_overlap_status AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * COUNT(*) / population.n_total AS percentage,
                NULL::INTEGER AS year,
                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM classified
            CROSS JOIN population
            GROUP BY supplementary_overlap_status, population.n_total
            """
        )

        queries.append(
            f"""
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({evidence_sql})
            )

            SELECT
                'supplementary_unit_pair' AS diagnostic_group,
                COALESCE(quantity_unit, '<NULL>')
                    || ' + '
                    || COALESCE(alt_quantity_unit, '<NULL>') AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * COUNT(*) / population.n_total AS percentage,
                NULL::INTEGER AS year,
                CAST(quantity_unit_code AS VARCHAR) AS category_1,
                CAST(alt_quantity_unit_code AS VARCHAR) AS category_2,
                CASE
                    WHEN quantity_unit_code = alt_quantity_unit_code
                        THEN 'same_unit'
                    ELSE 'different_unit'
                END AS category_3
            FROM read_parquet({evidence_sql})
            CROSS JOIN population
            WHERE quantity > 0
              AND quantity_unit_code <> 8
              AND quantity_unit_code <> -1
              AND alt_quantity > 0
              AND alt_quantity_unit_code <> -1
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
                quantity_unit,
                quantity_unit_code,
                alt_quantity_unit,
                alt_quantity_unit_code,
                population.n_total
            ORDER BY COUNT(*) DESC
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
        # 11. Synchronization methodology performance
        #
        # The adopted hierarchy is:
        #
        #   reported exact 3-way agreement
        #       -> reported exact pair agreement
        #       -> reported source precedence (Q > A > N)
        #       -> estimated fallback (Q > A > N)
        #       -> no usable Q/A/N kg evidence
        #
        # All percentages below use the full Quantity population.
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
                    'reported_three_exact' AS step_name,
                    COUNT(*) FILTER (
                        WHERE quantity_sync_status =
                            'reported_three_exact'
                    ) AS newly_covered
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    2,
                    'reported_pair_exact',
                    COUNT(*) FILTER (
                        WHERE quantity_sync_status LIKE
                            'reported_pair_exact_%'
                    )
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    3,
                    'reported_precedence',
                    COUNT(*) FILTER (
                        WHERE quantity_sync_status LIKE
                            'reported_%_precedence'
                    )
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    4,
                    'estimated_fallback',
                    COUNT(*) FILTER (
                        WHERE quantity_sync_status LIKE
                            'estimated_%_fallback'
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
                'methodology_funnel'
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

        # --------------------------------------------------------------
        # Exact synchronization rule used on every row
        # --------------------------------------------------------------

        queries.append(
            f"""
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({synchronized_sql})
            )

            SELECT
                'methodology_rule_detail'
                    AS diagnostic_group,

                quantity_sync_status AS metric,

                CAST(COUNT(*) AS DOUBLE) AS value,

                100.0 * COUNT(*) / population.n_total
                    AS percentage,

                NULL::INTEGER AS year,

                synchronized_quantity_source
                    AS category_1,

                CASE
                    WHEN quantity_sync_status LIKE 'reported_%'
                        THEN 'reported'
                    WHEN quantity_sync_status LIKE 'estimated_%'
                        THEN 'estimated'
                    ELSE 'unresolved'
                END AS category_2,

                CASE
                    WHEN quantity_sync_status =
                        'reported_three_exact'
                        THEN 'exact_three'
                    WHEN quantity_sync_status LIKE
                        'reported_pair_exact_%'
                        THEN 'exact_pair'
                    WHEN quantity_sync_status LIKE
                        'reported_%_precedence'
                        THEN 'precedence'
                    WHEN quantity_sync_status LIKE
                        'estimated_%_fallback'
                        THEN 'fallback'
                    ELSE 'missing'
                END AS category_3

            FROM read_parquet({synchronized_sql})
            CROSS JOIN population

            GROUP BY
                quantity_sync_status,
                synchronized_quantity_source,
                population.n_total
            """
        )

        # --------------------------------------------------------------
        # High-level evidence quality of the final synchronized values
        # --------------------------------------------------------------

        queries.append(
            f"""
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({synchronized_sql})
            )

            SELECT
                'methodology_summary'
                    AS diagnostic_group,
                metric,
                CAST(metric_value AS DOUBLE) AS value,
                100.0 * metric_value / population.n_total
                    AS percentage,
                NULL::INTEGER AS year,
                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3

            FROM (
                SELECT
                    'resolved_by_reported_evidence' AS metric,
                    COUNT(*) FILTER (
                        WHERE quantity_sync_status LIKE 'reported_%'
                    ) AS metric_value
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'resolved_by_exact_reported_agreement',
                    COUNT(*) FILTER (
                        WHERE quantity_sync_status =
                            'reported_three_exact'
                           OR quantity_sync_status LIKE
                            'reported_pair_exact_%'
                    )
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'resolved_by_reported_precedence',
                    COUNT(*) FILTER (
                        WHERE quantity_sync_status LIKE
                            'reported_%_precedence'
                    )
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'resolved_by_estimated_fallback',
                    COUNT(*) FILTER (
                        WHERE quantity_sync_status LIKE
                            'estimated_%_fallback'
                    )
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'no_usable_kg_evidence',
                    COUNT(*) FILTER (
                        WHERE quantity_sync_status =
                            'no_kg_evidence'
                    )
                FROM read_parquet({synchronized_sql})
            ) AS summary
            CROSS JOIN population
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
        # 15. Selection audit
        #
        # These diagnostics show which canonical candidate the final
        # synchronized value equals. Counts can overlap on exact-agreement
        # rows and are therefore diagnostic rather than a partition.
        # ==============================================================

        queries.append(
            f"""
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({synchronized_sql})
            )

            SELECT
                'selection_audit'
                    AS diagnostic_group,
                metric,
                CAST(metric_value AS DOUBLE) AS value,
                100.0 * metric_value / population.n_total
                    AS percentage,
                NULL::INTEGER AS year,
                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3

            FROM (
                SELECT
                    'synchronized_positive_rows' AS metric,
                    COUNT(*) FILTER (
                        WHERE synchronized_quantity_kg > 0
                    ) AS metric_value
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'final_equals_quantity_candidate',
                    COUNT(*) FILTER (
                        WHERE synchronized_quantity_kg > 0
                          AND quantity_kg_candidate > 0
                          AND synchronized_quantity_kg =
                              quantity_kg_candidate
                    )
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'final_equals_alt_quantity_candidate',
                    COUNT(*) FILTER (
                        WHERE synchronized_quantity_kg > 0
                          AND alt_quantity_kg_candidate > 0
                          AND synchronized_quantity_kg =
                              alt_quantity_kg_candidate
                    )
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'final_equals_net_weight_candidate',
                    COUNT(*) FILTER (
                        WHERE synchronized_quantity_kg > 0
                          AND net_weight_kg_candidate > 0
                          AND synchronized_quantity_kg =
                              net_weight_kg_candidate
                    )
                FROM read_parquet({synchronized_sql})
            ) AS audit
            CROSS JOIN population
            """
        )

        # ==============================================================
        # 16. Remaining kg gaps
        #
        # Disagreement among usable Q/A/N candidates is no longer left
        # unresolved: it is handled by the adopted source precedence.
        # Therefore the only unresolved rows are rows with no usable Q/A/N
        # kg evidence.
        # ==============================================================

        queries.append(
            f"""
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({synchronized_sql})
            )

            SELECT
                'remaining_kg_gap'
                    AS diagnostic_group,
                gap_type AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * COUNT(*) / population.n_total
                    AS percentage,
                NULL::INTEGER AS year,
                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3

            FROM (
                SELECT
                    CASE
                        WHEN synchronized_quantity_kg > 0
                            THEN 'resolved'

                        WHEN (
                            COALESCE(has_alt_quantity, FALSE)
                            OR COALESCE(has_quantity, FALSE)
                        )
                            THEN 'supplementary_quantity_but_no_kg'

                        ELSE 'no_physical_quantity_evidence'
                    END AS gap_type
                FROM read_parquet({synchronized_sql})
            ) AS gaps

            CROSS JOIN population
            GROUP BY
                gap_type,
                population.n_total
            """
        )

        # ==============================================================
        # 17. Missing kg distribution by HS
        # ==============================================================

        if hs_column is not None:
            queries.append(
                f"""
                WITH population AS (
                    SELECT COUNT(*) AS n_total
                    FROM read_parquet({synchronized_sql})
                )

                SELECT
                    'missing_kg_by_hs'
                        AS diagnostic_group,
                    CAST({hs_column} AS VARCHAR) AS metric,
                    CAST(COUNT(*) AS DOUBLE) AS value,
                    100.0 * COUNT(*) / population.n_total
                        AS percentage,
                    NULL::INTEGER AS year,
                    NULL::VARCHAR AS category_1,
                    NULL::VARCHAR AS category_2,
                    NULL::VARCHAR AS category_3

                FROM read_parquet({synchronized_sql})
                CROSS JOIN population

                WHERE quantity_sync_status =
                    'no_kg_evidence'

                GROUP BY
                    {hs_column},
                    population.n_total
                """
            )


        # ==============================================================
        # Final supplementary output
        # ==============================================================

        queries.append(
            f"""
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({synchronized_sql})
            ),

            coverage AS (
                SELECT
                    CASE
                        WHEN synchronized_quantity_kg > 0
                            THEN 'canonical_kg'

                        WHEN (
                            NOT COALESCE(
                                synchronized_quantity_kg > 0,
                                FALSE
                            )
                            AND supplementary_quantity > 0
                            AND supplementary_unit IS NOT NULL
                        )
                            THEN 'supplementary_without_kg'

                        ELSE 'no_physical_quantity'
                    END AS coverage_state,

                    COUNT(*) AS n_rows

                FROM read_parquet({synchronized_sql})

                GROUP BY coverage_state
            )

            SELECT
                'final_physical_information_coverage'
                    AS diagnostic_group,

                coverage_state AS metric,

                CAST(n_rows AS DOUBLE) AS value,

                100.0 * n_rows / population.n_total
                    AS percentage,

                NULL::INTEGER AS year,
                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3

            FROM coverage
            CROSS JOIN population
            """
        )

        queries.append(
            f"""
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({synchronized_sql})
            )

            SELECT
                'supplementary_output_summary' AS diagnostic_group,
                metric,
                CAST(metric_value AS DOUBLE) AS value,
                100.0 * metric_value / population.n_total AS percentage,
                NULL::INTEGER AS year,
                NULL::VARCHAR AS category_1,
                NULL::VARCHAR AS category_2,
                NULL::VARCHAR AS category_3
            FROM (
                SELECT
                    'supplementary_positive_rows' AS metric,
                    COUNT(*) FILTER (
                        WHERE supplementary_quantity > 0
                          AND supplementary_unit IS NOT NULL
                    ) AS metric_value
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'supplementary_missing_rows',
                    COUNT(*) FILTER (
                        WHERE supplementary_quantity IS NULL
                          AND supplementary_unit IS NULL
                    )
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'selected_from_quantity',
                    COUNT(*) FILTER (
                        WHERE supplementary_quantity_source = 'quantity'
                    )
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'selected_from_alt_quantity',
                    COUNT(*) FILTER (
                        WHERE supplementary_quantity_source = 'alt_quantity'
                    )
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'reported_supplementary',
                    COUNT(*) FILTER (
                        WHERE supplementary_quantity > 0
                          AND is_supplementary_quantity_estimated = FALSE
                    )
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'estimated_supplementary',
                    COUNT(*) FILTER (
                        WHERE supplementary_quantity > 0
                          AND is_supplementary_quantity_estimated = TRUE
                    )
                FROM read_parquet({synchronized_sql})

                UNION ALL

                SELECT
                    'quantity_unit_null_mismatch',
                    COUNT(*) FILTER (
                        WHERE (
                            supplementary_quantity IS NULL
                            AND supplementary_unit IS NOT NULL
                        )
                        OR (
                            supplementary_quantity IS NOT NULL
                            AND supplementary_unit IS NULL
                        )
                    )
                FROM read_parquet({synchronized_sql})
            ) AS summary
            CROSS JOIN population
            """
        )

        queries.append(
            f"""
            WITH population AS (
                SELECT COUNT(*) AS n_total
                FROM read_parquet({synchronized_sql})
            )

            SELECT
                'supplementary_output_unit_distribution' AS diagnostic_group,
                supplementary_unit AS metric,
                CAST(COUNT(*) AS DOUBLE) AS value,
                100.0 * COUNT(*) / population.n_total AS percentage,
                NULL::INTEGER AS year,
                supplementary_quantity_source AS category_1,
                CASE
                    WHEN is_supplementary_quantity_estimated
                        THEN 'estimated'
                    ELSE 'reported'
                END AS category_2,
                NULL::VARCHAR AS category_3
            FROM read_parquet({synchronized_sql})
            CROSS JOIN population
            WHERE supplementary_quantity > 0
              AND supplementary_unit IS NOT NULL
            GROUP BY
                supplementary_unit,
                supplementary_quantity_source,
                is_supplementary_quantity_estimated,
                population.n_total
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
                        'reported_three_exact'
                ) AS n_reported_three_exact,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status LIKE
                        'reported_pair_exact_%'
                ) AS n_reported_pair_exact,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status LIKE
                        'reported_%_precedence'
                ) AS n_reported_precedence,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status LIKE
                        'estimated_%_fallback'
                ) AS n_estimated_fallback,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status =
                        'no_kg_evidence'
                ) AS n_no_kg_evidence

            FROM read_parquet({synchronized_sql})
            """
        ).fetchone()

        (
            n_rows,
            n_synchronized,
            n_reported_three_exact,
            n_reported_pair_exact,
            n_reported_precedence,
            n_estimated_fallback,
            n_no_kg_evidence,
        ) = summary

        synchronized_pct = (
            100.0 * n_synchronized / n_rows
            if n_rows
            else 0.0
        )

        no_kg_pct = (
            100.0 * n_no_kg_evidence / n_rows
            if n_rows
            else 0.0
        )

        methodology_partition = (
            n_reported_three_exact
            + n_reported_pair_exact
            + n_reported_precedence
            + n_estimated_fallback
            + n_no_kg_evidence
        )

        if methodology_partition != n_rows:
            raise RuntimeError(
                "Quantity methodology accounting failed: "
                f"{methodology_partition:,} classified rows for "
                f"{n_rows:,} input rows."
            )

        n_diagnostic_rows = con.execute(
            """
            SELECT COUNT(*)
            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()[0]


        # ==============================================================
        # DIAGNOSTICS AUDIT
        # ==============================================================

        if not output_path.is_file():
            raise FileNotFoundError(
                "Quantity diagnostics were not created: "
                f"{output_path}"
            )

        diagnostic_columns = {
            str(row[0])
            for row in con.execute(
                """
                DESCRIBE
                SELECT *
                FROM read_parquet(?)
                """,
                [str(output_path)],
            ).fetchall()
        }

        required_diagnostic_columns = {
            "diagnostic_group",
            "metric",
            "value",
            "percentage",
            "year",
            "category_1",
            "category_2",
            "category_3",
        }

        missing_diagnostic_columns = sorted(
            required_diagnostic_columns - diagnostic_columns
        )

        if missing_diagnostic_columns:
            raise RuntimeError(
                "Quantity diagnostics are missing required columns: "
                + ", ".join(missing_diagnostic_columns)
            )

        diagnostic_audit_row = con.execute(
            """
            SELECT
                COUNT(*) AS n_diagnostic_rows,

                COUNT(*) FILTER (
                    WHERE diagnostic_group IS NULL
                       OR TRIM(diagnostic_group) = ''
                ) AS n_missing_diagnostic_group,

                COUNT(*) FILTER (
                    WHERE metric IS NULL
                       OR TRIM(metric) = ''
                ) AS n_missing_metric

            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()

        if diagnostic_audit_row is None:
            raise RuntimeError(
                "Could not inspect Quantity diagnostics."
            )

        (
            n_diagnostic_rows,
            n_missing_diagnostic_group,
            n_missing_metric,
        ) = diagnostic_audit_row

        n_diagnostic_rows = int(n_diagnostic_rows)
        n_missing_diagnostic_group = int(
            n_missing_diagnostic_group
        )
        n_missing_metric = int(n_missing_metric)

        # ==============================================================
        # HARD AUDIT CONDITIONS
        # ==============================================================

        if int(n_rows) == 0:
            raise RuntimeError(
                "Synchronized Quantity input contains no rows."
            )

        if n_diagnostic_rows == 0:
            raise RuntimeError(
                "Quantity diagnostics contain no rows."
            )

        if n_missing_diagnostic_group > 0:
            raise RuntimeError(
                "Quantity diagnostics contain "
                f"{n_missing_diagnostic_group:,} rows without "
                "a diagnostic_group."
            )

        if n_missing_metric > 0:
            raise RuntimeError(
                "Quantity diagnostics contain "
                f"{n_missing_metric:,} rows without a metric."
            )

        audit_passed = True
        ready_for_reporting = audit_passed

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
            "Reported three-source exact agreement: %s",
            f"{n_reported_three_exact:,}",
        )
        logger.info(
            "Reported exact pair agreement: %s",
            f"{n_reported_pair_exact:,}",
        )
        logger.info(
            "Reported precedence resolutions: %s",
            f"{n_reported_precedence:,}",
        )
        logger.info(
            "Estimated fallback resolutions: %s",
            f"{n_estimated_fallback:,}",
        )
        logger.info(
            "Rows with no usable Q/A/N kg evidence: %s (%.4f%%)",
            f"{n_no_kg_evidence:,}",
            no_kg_pct,
        )
        logger.info(
            "Quantity diagnostics: %s",
            output_path,
        )

        logger.info("  Output: %s", output_path)
        logger.info("  Step audit passed: %s", audit_passed,)
        logger.info("  Ready for reporting: %s", ready_for_reporting,)

    finally:
        con.close()

    if not output_path.is_file():
        raise FileNotFoundError(
            "Quantity diagnostics were not created: "
            f"{output_path}"
        )


    write_local_checkpoint(
        checkpoint_path=checkpoint_path,
        step_name="diagnostics",
        run_id=run_id,
        config=config_fp,
        input_fingerprint=input_fp,
        output_path=output_path,
    )
    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": output_path,
        "diagnostics_path": output_path,

        "audit_passed": audit_passed,
        "ready_for_reporting": ready_for_reporting,

        "n_rows": int(n_rows),
        "n_diagnostic_rows": n_diagnostic_rows,
        "n_synchronized": int(n_synchronized),
        "n_reported_three_exact": int(
            n_reported_three_exact
        ),
        "n_reported_pair_exact": int(
            n_reported_pair_exact
        ),
        "n_reported_precedence": int(
            n_reported_precedence
        ),
        "n_estimated_fallback": int(
            n_estimated_fallback
        ),
        "n_no_kg_evidence": int(
            n_no_kg_evidence
        ),

        "synchronized_pct": float(
            synchronized_pct
        ),
        "no_kg_pct": float(
            no_kg_pct
        ),
    }
