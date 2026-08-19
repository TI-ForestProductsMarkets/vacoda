"""
vacoda.quantity.reconciliation_evidence

Build row-level evidence for quantity synchronization.

This step compares usable kg quantity observations with net weight,
records reported/estimated source states, measures agreement, and
creates the evidence required by the synchronization step.

No canonical quantity source is selected here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
from vacoda.config.quantity import QuantityConfig
from vacoda.utils.sql import sql_literal


def build_quantity_reconciliation_evidence(
    *,
    cfg: QuantityConfig,
    input_path: Path,
    run_id: str,
    force: bool,
    logger,
) -> dict[str, Any]:
    """
    Build row-level reconciliation evidence for Quantity.

    Parameters
    ----------
    cfg:
        Parsed Quantity configuration.
    input_path:
        Prepared Quantity input Parquet.
    run_id:
        Master-pipeline run identifier.
    force:
        Rebuild the evidence artifact even when it already exists.
    logger:
        Configured workflow logger.

    Returns
    -------
    dict
        Result metadata including ``reconciliation_evidence_path``.
    """

    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            "Prepared Quantity input parquet not found: "
            f"{input_path}"
        )

    output_path = cfg.product_path(
        "reconciliation_evidence"
    )

    if output_path.is_file() and not force:
        logger.info(
            "Reusing existing Quantity reconciliation evidence: %s",
            output_path,
        )

        return {
            "status": "skipped_existing",
            "run_id": run_id,
            "output_path": output_path,
            "reconciliation_evidence_path": output_path,
        }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    quantity_unit = (
        cfg.method.kg_reconciliation.quantity_unit
    )

    agreement_tolerance = (
        cfg.method
        .kg_reconciliation
        .agreement
        .relative_tolerance
    )

    input_sql = sql_literal(input_path)
    output_sql = sql_literal(output_path)

    logger.info(
        "Building Quantity reconciliation evidence."
    )
    logger.info(
        "Quantity reconciliation unit: %s",
        quantity_unit,
    )
    logger.info(
        "Agreement relative tolerance: %.6f",
        agreement_tolerance,
    )

    con = duckdb.connect()

    try:

        query = f"""
            COPY (
                WITH evidence AS (

                    SELECT
                        *,

                        -- ==============================================
                        -- Canonical candidate values
                        -- ==============================================

                        CASE
                            WHEN has_quantity_kg
                            THEN quantity
                            ELSE NULL
                        END AS quantity_kg_candidate,

                        CASE
                            WHEN alt_quantity_unit_code = 8
                                AND LOWER(TRIM(alt_quantity_unit)) = 'kg'
                                AND alt_quantity > 0
                                THEN alt_quantity

                            WHEN alt_quantity_unit_code = 21
                                AND UPPER(TRIM(alt_quantity_unit)) = '1000 KG'
                                AND alt_quantity > 0
                                THEN alt_quantity * 1000.0

                            WHEN alt_quantity_unit_code = 15
                                AND LOWER(TRIM(alt_quantity_unit)) = 'g'
                                AND alt_quantity > 0
                                THEN alt_quantity * 0.001

                            ELSE NULL
                        END AS alt_quantity_kg_candidate,

                        CASE
                            WHEN has_net_weight_kg
                            THEN net_weight_kg
                            ELSE NULL
                        END AS net_weight_kg_candidate,

                        CASE
                            WHEN alt_quantity_unit_code = 8
                                AND LOWER(TRIM(alt_quantity_unit)) = 'kg'
                                AND alt_quantity > 0
                                THEN 'kg'

                            WHEN alt_quantity_unit_code = 21
                                AND UPPER(TRIM(alt_quantity_unit)) = '1000 KG'
                                AND alt_quantity > 0
                                THEN '1000kg_converted'

                            WHEN alt_quantity_unit_code = 15
                                AND LOWER(TRIM(alt_quantity_unit)) = 'g'
                                AND alt_quantity > 0
                                THEN 'g_converted'

                            ELSE NULL
                        END AS alt_quantity_kg_source,

                        CASE
                            WHEN (
                                    (
                                        alt_quantity_unit_code = 8
                                        AND LOWER(TRIM(alt_quantity_unit)) = 'kg'
                                    )
                                    OR
                                    (
                                        alt_quantity_unit_code = 21
                                        AND UPPER(TRIM(alt_quantity_unit)) = '1000 KG'
                                    )
                                    OR
                                    (
                                        alt_quantity_unit_code = 15
                                        AND LOWER(TRIM(alt_quantity_unit)) = 'g'
                                    )
                                )
                                AND alt_quantity > 0
                                AND is_alt_quantity_estimated = FALSE
                                THEN 'reported'

                            WHEN (
                                    (
                                        alt_quantity_unit_code = 8
                                        AND LOWER(TRIM(alt_quantity_unit)) = 'kg'
                                    )
                                    OR
                                    (
                                        alt_quantity_unit_code = 21
                                        AND UPPER(TRIM(alt_quantity_unit)) = '1000 KG'
                                    )
                                    OR
                                    (
                                        alt_quantity_unit_code = 15
                                        AND LOWER(TRIM(alt_quantity_unit)) = 'g'
                                    )
                                )
                                AND alt_quantity > 0
                                AND is_alt_quantity_estimated = TRUE
                                THEN 'estimated'

                            WHEN (
                                    (
                                        alt_quantity_unit_code = 8
                                        AND LOWER(TRIM(alt_quantity_unit)) = 'kg'
                                    )
                                    OR
                                    (
                                        alt_quantity_unit_code = 21
                                        AND UPPER(TRIM(alt_quantity_unit)) = '1000 KG'
                                    )
                                    OR
                                    (
                                        alt_quantity_unit_code = 15
                                        AND LOWER(TRIM(alt_quantity_unit)) = 'g'
                                    )
                                )
                                AND alt_quantity > 0
                                AND is_alt_quantity_estimated IS NULL
                                THEN 'unknown'

                            ELSE 'unavailable'
                        END AS alt_quantity_kg_status,

                        -- ==============================================
                        -- Candidate source state
                        -- ==============================================

                        CASE
                            WHEN has_reported_quantity_kg
                                THEN 'reported'
                            WHEN has_estimated_quantity_kg
                                THEN 'estimated'
                            ELSE 'unavailable'
                        END AS quantity_kg_status,

                        CASE
                            WHEN has_reported_net_weight_kg
                                THEN 'reported'
                            WHEN has_estimated_net_weight_kg
                                THEN 'estimated'
                            ELSE 'unavailable'
                        END AS net_weight_kg_status,

                        -- ==============================================
                        -- Availability pattern
                        -- ==============================================

                        CASE
                            WHEN has_quantity_kg
                                 AND has_net_weight_kg
                                THEN 'both'
                            WHEN has_quantity_kg
                                 AND NOT has_net_weight_kg
                                THEN 'quantity_only'
                            WHEN NOT has_quantity_kg
                                 AND has_net_weight_kg
                                THEN 'net_weight_only'
                            ELSE 'neither'
                        END AS kg_evidence_availability,

                        -- ==============================================
                        -- Reported / estimated combination
                        -- ==============================================

                        CASE
                            WHEN has_reported_quantity_kg
                                 AND has_reported_net_weight_kg
                                THEN 'reported_reported'

                            WHEN has_reported_quantity_kg
                                 AND has_estimated_net_weight_kg
                                THEN 'reported_estimated'

                            WHEN has_estimated_quantity_kg
                                 AND has_reported_net_weight_kg
                                THEN 'estimated_reported'

                            WHEN has_estimated_quantity_kg
                                 AND has_estimated_net_weight_kg
                                THEN 'estimated_estimated'

                            WHEN has_reported_quantity_kg
                                 AND NOT has_net_weight_kg
                                THEN 'reported_quantity_only'

                            WHEN has_estimated_quantity_kg
                                 AND NOT has_net_weight_kg
                                THEN 'estimated_quantity_only'

                            WHEN NOT has_quantity_kg
                                 AND has_reported_net_weight_kg
                                THEN 'reported_net_weight_only'

                            WHEN NOT has_quantity_kg
                                 AND has_estimated_net_weight_kg
                                THEN 'estimated_net_weight_only'

                            ELSE 'no_kg_evidence'
                        END AS kg_evidence_status,

                        -- ==============================================
                        -- Direct comparison
                        -- ==============================================

                        CASE
                            WHEN has_quantity_kg
                                 AND has_net_weight_kg
                                 AND net_weight_kg > 0
                            THEN quantity / net_weight_kg
                            ELSE NULL
                        END AS quantity_net_weight_ratio,

                        CASE
                            WHEN has_quantity_kg
                                 AND has_net_weight_kg
                            THEN quantity - net_weight_kg
                            ELSE NULL
                        END AS quantity_net_weight_difference,

                        -- Symmetric relative difference:
                        --
                        -- |q - w| / ((|q| + |w|) / 2)
                        --
                        -- This avoids arbitrarily treating either
                        -- quantity or net weight as the denominator.
                        CASE
                            WHEN has_quantity_kg
                                 AND has_net_weight_kg
                                 AND (
                                     abs(quantity)
                                     + abs(net_weight_kg)
                                 ) > 0
                            THEN
                                abs(
                                    quantity - net_weight_kg
                                )
                                /
                                (
                                    (
                                        abs(quantity)
                                        + abs(net_weight_kg)
                                    )
                                    / 2.0
                                )
                            ELSE NULL
                        END AS quantity_net_weight_relative_gap

                    FROM read_parquet({input_sql})
                )

                SELECT
                    *,

                    -- ==============================================
                    -- Agreement classification
                    -- ==============================================

                    CASE
                        WHEN quantity_net_weight_relative_gap
                             IS NULL
                            THEN NULL

                        WHEN quantity_net_weight_relative_gap <= {agreement_tolerance}
                            THEN TRUE

                        ELSE FALSE
                    END AS quantity_net_weight_agree,

                    CASE
                        WHEN quantity_net_weight_relative_gap
                             IS NULL
                            THEN 'not_comparable'

                        WHEN quantity_net_weight_relative_gap <= {agreement_tolerance}
                            THEN 'within_tolerance'

                        ELSE 'outside_tolerance'
                    END AS quantity_net_weight_agreement_status

                FROM evidence
            )
            TO {output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
        """

        con.execute(query)

        # --------------------------------------------------------------
        # Basic run statistics
        # --------------------------------------------------------------

        stats = con.execute(
            """
            SELECT
                count(*) AS n_rows,

                count(*) FILTER (
                    WHERE has_quantity_kg
                ) AS n_quantity_kg,

                count(*) FILTER (
                    WHERE has_net_weight_kg
                ) AS n_net_weight_kg,

                COUNT(*) FILTER (
                    WHERE alt_quantity_kg_candidate > 0
                ) AS n_alt_quantity_kg,

                COUNT(*) FILTER (
                    WHERE alt_quantity_kg_candidate > 0
                    AND quantity_kg_candidate IS NULL
                    AND net_weight_kg_candidate IS NULL
                ) AS n_incremental_alt_kg,

                count(*) FILTER (
                    WHERE has_quantity_kg
                      AND has_net_weight_kg
                ) AS n_both,

                count(*) FILTER (
                    WHERE quantity_net_weight_agree
                ) AS n_agree,

                count(*) FILTER (
                    WHERE quantity_net_weight_agree = FALSE
                ) AS n_disagree

            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()

        (
            n_rows,
            n_quantity_kg,
            n_net_weight_kg,
            n_alt_quantity_kg,
            n_incremental_alt_kg,
            n_both,
            n_agree,
            n_disagree,
        ) = stats

        logger.info(
            "Quantity reconciliation evidence rows: %s",
            f"{n_rows:,}",
        )
        logger.info(
            "Usable quantity kg rows: %s",
            f"{n_quantity_kg:,}",
        )
        logger.info(
            "Usable net-weight kg rows: %s",
            f"{n_net_weight_kg:,}",
        )
        logger.info(
            "Rows with both kg sources: %s",
            f"{n_both:,}",
        )
        logger.info(
            "Rows agreeing within tolerance: %s",
            f"{n_agree:,}",
        )
        logger.info(
            "Rows outside tolerance: %s",
            f"{n_disagree:,}",
        )
        logger.info(
            "Usable alt-quantity kg candidate rows: %s",
            f"{n_alt_quantity_kg:,}",
        )
        logger.info(
            "Incremental alt-quantity kg rows after quantity/net weight: %s",
            f"{n_incremental_alt_kg:,}",
        )
        logger.info(
            "Quantity reconciliation evidence: %s",
            output_path,
        )

    finally:
        con.close()

    if not output_path.is_file():
        raise FileNotFoundError(
            "Quantity reconciliation evidence was not created: "
            f"{output_path}"
        )

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": output_path,
        "reconciliation_evidence_path": output_path,
        "n_rows": int(n_rows),
        "n_quantity_kg": int(n_quantity_kg),
        "n_net_weight_kg": int(n_net_weight_kg),
        "n_alt_quantity_kg": int(n_alt_quantity_kg),
        "n_incremental_alt_kg": int(n_incremental_alt_kg),
        "n_both": int(n_both),
        "n_agree": int(n_agree),
        "n_disagree": int(n_disagree),
        "ready_for_synchronization": True,
    }