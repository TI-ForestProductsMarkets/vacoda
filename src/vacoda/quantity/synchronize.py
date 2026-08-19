"""
vacoda.quantity.synchronize

Synchronize kg-based quantity evidence into a canonical quantity field.

The synchronization is deliberately conservative:

- use the sole available kg source when only one exists;
- reconcile both sources when they agree within tolerance;
- leave conflicting observations unresolved when both sources disagree;
- do not use gross weight as a quantity proxy unless explicitly enabled.

No fixed quantity-versus-net-weight source hierarchy is imposed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.quantity import QuantityConfig
from vacoda.utils.sql import sql_literal

def synchronize_quantity(
    *,
    cfg: QuantityConfig,
    prepared_input_path: Path,
    reconciliation_evidence_path: Path,
    run_id: str,
    force: bool,
    logger,
) -> dict[str, Any]:
    """
    Build the synchronized Quantity dataset.

    Parameters
    ----------
    cfg:
        Parsed Quantity configuration.
    prepared_input_path:
        Prepared Quantity input Parquet.
    reconciliation_evidence_path:
        Row-level reconciliation evidence Parquet.
    run_id:
        Master-pipeline run identifier.
    force:
        Rebuild the synchronized artifact even when it already exists.
    logger:
        Configured workflow logger.

    Returns
    -------
    dict
        Result metadata including ``synchronized_quantity_path``.
    """

    prepared_input_path = Path(prepared_input_path)
    reconciliation_evidence_path = Path(
        reconciliation_evidence_path
    )

    if not prepared_input_path.is_file():
        raise FileNotFoundError(
            "Prepared Quantity input parquet not found: "
            f"{prepared_input_path}"
        )

    if not reconciliation_evidence_path.is_file():
        raise FileNotFoundError(
            "Quantity reconciliation evidence parquet not found: "
            f"{reconciliation_evidence_path}"
        )

    output_path = cfg.product_path(
        "synchronized_quantity"
    )

    if output_path.is_file() and not force:
        logger.info(
            "Reusing existing synchronized Quantity output: %s",
            output_path,
        )

        return {
            "status": "skipped_existing",
            "run_id": run_id,
            "output_path": output_path,
            "synchronized_quantity_path": output_path,
        }

    evidence_sql = sql_literal(
        reconciliation_evidence_path
    )
    output_sql = sql_literal(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_selection_method = (
        cfg.method
        .kg_reconciliation
        .source_selection
        .method
    )

    use_gross_weight_proxy = (
        cfg.method
        .gross_weight
        .use_as_quantity_proxy
    )

    if source_selection_method != "data_driven":
        raise ValueError(
            "Unsupported Quantity source-selection method: "
            f"{source_selection_method!r}"
        )

    logger.info(
        "Synchronizing Quantity observations."
    )
    logger.info(
        "Configured source-selection method: %s",
        source_selection_method,
    )
    logger.info(
        "Use gross weight as fallback proxy: %s",
        use_gross_weight_proxy,
    )

    con = duckdb.connect()

    try:

        # --------------------------------------------------------------
        # The reconciliation evidence already contains all original
        # prepared-input columns, so synchronization can operate directly
        # on that artifact.
        #
        # No join back to prepared_input is required.
        # --------------------------------------------------------------

        if use_gross_weight_proxy:
            gross_weight_value_sql = """
                WHEN NOT has_quantity_kg
                     AND NOT has_net_weight_kg
                     AND has_gross_weight_kg
                THEN gross_weight_kg
            """

            gross_weight_source_sql = """
                WHEN NOT has_quantity_kg
                     AND NOT has_net_weight_kg
                     AND has_gross_weight_kg
                THEN 'gross_weight_proxy'
            """

            gross_weight_status_sql = """
                WHEN NOT has_quantity_kg
                     AND NOT has_net_weight_kg
                     AND has_gross_weight_kg
                THEN 'gross_weight_proxy'
            """

            gross_weight_estimated_sql = """
                WHEN NOT has_quantity_kg
                     AND NOT has_net_weight_kg
                     AND has_gross_weight_kg
                THEN NULL
            """
        else:
            gross_weight_value_sql = ""
            gross_weight_source_sql = ""
            gross_weight_status_sql = ""
            gross_weight_estimated_sql = ""

        query = f"""
            COPY (
                SELECT
                    *,

                    -- ==============================================
                    -- Canonical synchronized kg quantity
                    -- ==============================================

                    CASE

                        -- Both sources exist and agree.
                        --
                        -- Neither source is treated as intrinsically
                        -- superior; their mean is used as the
                        -- reconciled value.
                        WHEN has_quantity_kg
                             AND has_net_weight_kg
                             AND quantity_net_weight_agree = TRUE
                        THEN
                            (
                                quantity_kg_candidate
                                + net_weight_kg_candidate
                            ) / 2.0

                        -- Quantity is the only usable kg source.
                        WHEN has_quantity_kg
                             AND NOT has_net_weight_kg
                        THEN quantity_kg_candidate

                        -- Net weight is the only usable kg source.
                        WHEN NOT has_quantity_kg
                             AND has_net_weight_kg
                        THEN net_weight_kg_candidate

                        {gross_weight_value_sql}

                        -- Both sources exist but disagree:
                        -- deliberately unresolved.
                        ELSE NULL

                    END AS synchronized_quantity_kg,


                    -- ==============================================
                    -- Source provenance
                    -- ==============================================

                    CASE

                        WHEN has_quantity_kg
                             AND has_net_weight_kg
                             AND quantity_net_weight_agree = TRUE
                        THEN 'quantity_and_net_weight'

                        WHEN has_quantity_kg
                             AND NOT has_net_weight_kg
                        THEN 'quantity'

                        WHEN NOT has_quantity_kg
                             AND has_net_weight_kg
                        THEN 'net_weight'

                        {gross_weight_source_sql}

                        WHEN has_quantity_kg
                             AND has_net_weight_kg
                             AND quantity_net_weight_agree = FALSE
                        THEN 'conflict'

                        ELSE 'unavailable'

                    END AS synchronized_quantity_source,


                    -- ==============================================
                    -- Synchronization status
                    -- ==============================================

                    CASE

                        WHEN has_quantity_kg
                             AND has_net_weight_kg
                             AND quantity_net_weight_agree = TRUE
                        THEN 'reconciled_agreement'

                        WHEN has_quantity_kg
                             AND NOT has_net_weight_kg
                        THEN 'quantity_only'

                        WHEN NOT has_quantity_kg
                             AND has_net_weight_kg
                        THEN 'net_weight_only'

                        {gross_weight_status_sql}

                        WHEN has_quantity_kg
                             AND has_net_weight_kg
                             AND quantity_net_weight_agree = FALSE
                        THEN 'unresolved_conflict'

                        ELSE 'no_kg_evidence'

                    END AS quantity_sync_status,


                    -- ==============================================
                    -- Estimated-state provenance
                    -- ==============================================

                    CASE

                        -- Both candidates are present and reconciled.
                        WHEN has_quantity_kg
                             AND has_net_weight_kg
                             AND quantity_net_weight_agree = TRUE
                        THEN
                            (
                                coalesce(
                                    is_quantity_estimated,
                                    FALSE
                                )
                                OR
                                coalesce(
                                    is_net_weight_estimated,
                                    FALSE
                                )
                            )

                        -- Sole quantity source.
                        WHEN has_quantity_kg
                             AND NOT has_net_weight_kg
                        THEN
                            coalesce(
                                is_quantity_estimated,
                                FALSE
                            )

                        -- Sole net-weight source.
                        WHEN NOT has_quantity_kg
                             AND has_net_weight_kg
                        THEN
                            coalesce(
                                is_net_weight_estimated,
                                FALSE
                            )

                        {gross_weight_estimated_sql}

                        ELSE NULL

                    END AS is_synchronized_quantity_estimated,


                    -- ==============================================
                    -- Resolution flag
                    -- ==============================================

                    CASE

                        WHEN has_quantity_kg
                             AND has_net_weight_kg
                             AND quantity_net_weight_agree = FALSE
                        THEN FALSE

                        WHEN has_quantity_kg
                             OR has_net_weight_kg
                        THEN TRUE

                        {"WHEN has_gross_weight_kg THEN TRUE"
                         if use_gross_weight_proxy
                         else ""}

                        ELSE FALSE

                    END AS quantity_sync_resolved

                FROM read_parquet({evidence_sql})
            )
            TO {output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
        """

        con.execute(query)

        # --------------------------------------------------------------
        # Synchronization summary
        # --------------------------------------------------------------

        summary = con.execute(
            """
            SELECT
                count(*) AS n_rows,

                count(*) FILTER (
                    WHERE synchronized_quantity_kg > 0
                ) AS n_synchronized,

                count(*) FILTER (
                    WHERE quantity_sync_status =
                          'reconciled_agreement'
                ) AS n_reconciled_agreement,

                count(*) FILTER (
                    WHERE quantity_sync_status =
                          'quantity_only'
                ) AS n_quantity_only,

                count(*) FILTER (
                    WHERE quantity_sync_status =
                          'net_weight_only'
                ) AS n_net_weight_only,

                count(*) FILTER (
                    WHERE quantity_sync_status =
                          'unresolved_conflict'
                ) AS n_unresolved_conflict,

                count(*) FILTER (
                    WHERE quantity_sync_status =
                          'no_kg_evidence'
                ) AS n_no_kg_evidence

            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()

        (
            n_rows,
            n_synchronized,
            n_reconciled_agreement,
            n_quantity_only,
            n_net_weight_only,
            n_unresolved_conflict,
            n_no_kg_evidence,
        ) = summary

        synchronized_pct = (
            100.0 * n_synchronized / n_rows
            if n_rows
            else 0.0
        )

        conflict_pct = (
            100.0 * n_unresolved_conflict / n_rows
            if n_rows
            else 0.0
        )

        logger.info(
            "Quantity rows: %s",
            f"{n_rows:,}",
        )
        logger.info(
            "Rows with synchronized kg quantity: %s (%.2f%%)",
            f"{n_synchronized:,}",
            synchronized_pct,
        )
        logger.info(
            "Reconciled agreeing sources: %s",
            f"{n_reconciled_agreement:,}",
        )
        logger.info(
            "Quantity-only rows: %s",
            f"{n_quantity_only:,}",
        )
        logger.info(
            "Net-weight-only rows: %s",
            f"{n_net_weight_only:,}",
        )
        logger.info(
            "Unresolved conflicts: %s (%.4f%%)",
            f"{n_unresolved_conflict:,}",
            conflict_pct,
        )
        logger.info(
            "Rows unresolved by current quantity/net-weight synchronization: %s",
            f"{n_no_kg_evidence:,}",
        )
        logger.info(
            "Synchronized Quantity output: %s",
            output_path,
        )

    finally:
        con.close()

    if not output_path.is_file():
        raise FileNotFoundError(
            "Synchronized Quantity output was not created: "
            f"{output_path}"
        )

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": output_path,
        "synchronized_quantity_path": output_path,

        "n_rows": int(n_rows),
        "n_synchronized": int(n_synchronized),
        "n_reconciled_agreement": int(
            n_reconciled_agreement
        ),
        "n_quantity_only": int(n_quantity_only),
        "n_net_weight_only": int(n_net_weight_only),
        "n_unresolved_conflict": int(
            n_unresolved_conflict
        ),
        "n_no_kg_evidence": int(n_no_kg_evidence),

        "synchronized_pct": float(
            synchronized_pct
        ),
        "conflict_pct": float(conflict_pct),

        "ready_for_diagnostics": True,
    }