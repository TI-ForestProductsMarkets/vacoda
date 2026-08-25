"""
vacoda.quantity.synchronize

Synchronize canonical kg evidence into ``synchronized_quantity_kg``.

Selection methodology
---------------------
1. Exact unit normalization is completed upstream and is not a
   source-selection decision.

2. Reported kg evidence always has precedence over estimated evidence.

3. Among reported candidates:
   - if quantity, alternate quantity, and net weight are all reported
     and exactly equal, use the common value;
   - otherwise, if any two reported candidates are exactly equal,
     use that common value;
   - otherwise resolve reported evidence using the fixed source order:

         quantity > alt_quantity > net_weight

4. Estimated evidence is considered only when no reported kg candidate
   exists. Estimated candidates use the same source order:

         quantity > alt_quantity > net_weight

5. Gross weight does not participate in ordinary quantity synchronization
   and remains diagnostic-only.

The configured tolerance is retained for diagnostics but does not affect
source selection.

Supplementary quantity methodology
-----------------------------------
Non-kg physical quantities are preserved separately from canonical kg.

- usable non-kg ``quantity`` has precedence;
- otherwise usable non-kg ``alt_quantity`` is used;
- canonical kg-equivalent values and unknown-unit values are excluded;
- original quantity and alternate-quantity fields remain preserved.

The resulting canonical fields are:

    synchronized_quantity_kg
    supplementary_quantity
    supplementary_unit
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
    reconciliation_evidence_path = Path(reconciliation_evidence_path)

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

    ## local checkpointing
    checkpoint_path = (
        cfg.output_dir()
        / "checkpoints"
        / "synchronized.json"
    )

    config_fp = semantic_config_fingerprint(config=cfg, 
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
    }

    reusable, reason = validate_local_checkpoint(
        checkpoint_path=checkpoint_path,
        current_config=config_fp,
        current_input=input_fp,
        output_path=output_path,
    )

    if not force and reusable:
        logger.info(
            "REUSE — Synchronized Quantity local checkpoint matches: %s",
            output_path,
        )

        return {
            "status": "reused",
            "run_id": run_id,
            "output_path": output_path,
            "synchronized_quantity_path": output_path,
            "audit_passed": True,
        }

    logger.info(
        "REBUILD — Synchronized Quantity: %s",
        "forced" if force else reason,
    ) 

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

    if use_gross_weight_proxy:
        raise ValueError(
            "Gross weight is diagnostic-only under the adopted "
            "Quantity synchronization methodology and cannot be "
            "used as a quantity proxy."
        )

    logger.info(
        "Synchronizing Quantity observations."
    )
    logger.info(
        "Configured source-selection method: %s",
        source_selection_method,
    )
    logger.info(
        "Reported-source precedence: quantity > alt_quantity > net_weight"
    )
    logger.info(
        "Estimated fallback precedence: quantity > alt_quantity > net_weight"
    )
    logger.info(
        "Gross weight participates in synchronization: False"
    )

    con = duckdb.connect()

    try:

        # --------------------------------------------------------------
        # The reconciliation evidence already contains all original
        # prepared-input columns plus canonical Q/A/N kg candidates and
        # reported/estimated provenance.
        #
        # Synchronization therefore operates directly on that artifact.
        # No join back to prepared_input is required.
        # --------------------------------------------------------------

        query = f"""
            COPY (
                WITH evidence AS (
                    SELECT
                        *,

                        -- ------------------------------------------
                        -- Reported kg candidates
                        -- ------------------------------------------

                        (
                            quantity_kg_candidate > 0
                            AND quantity_kg_status = 'reported'
                        ) AS q_reported,

                        (
                            alt_quantity_kg_candidate > 0
                            AND alt_quantity_kg_status = 'reported'
                        ) AS a_reported,

                        (
                            net_weight_kg_candidate > 0
                            AND net_weight_kg_status = 'reported'
                        ) AS n_reported,

                        -- ------------------------------------------
                        -- Estimated kg candidates
                        -- ------------------------------------------

                        (
                            quantity_kg_candidate > 0
                            AND quantity_kg_status = 'estimated'
                        ) AS q_estimated,

                        (
                            alt_quantity_kg_candidate > 0
                            AND alt_quantity_kg_status = 'estimated'
                        ) AS a_estimated,

                        (
                            net_weight_kg_candidate > 0
                            AND net_weight_kg_status = 'estimated'
                        ) AS n_estimated,

                        -- ------------------------------------------
                        -- Non-kg supplementary candidates
                        -- ------------------------------------------

                        (
                            quantity > 0
                            AND quantity_unit_code <> 8
                            AND quantity_unit_code <> -1
                        ) AS has_quantity_supplementary,

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
                        ) AS has_alt_quantity_supplementary
                        

                    FROM read_parquet({evidence_sql})
                ),

                classified AS (
                    SELECT
                        *,

                        CASE

                            -- ======================================
                            -- LEVEL 1: REPORTED EXACT AGREEMENT
                            -- ======================================

                            -- All three reported and exactly equal.
                            WHEN q_reported
                                 AND a_reported
                                 AND n_reported
                                 AND quantity_kg_candidate
                                     = alt_quantity_kg_candidate
                                 AND quantity_kg_candidate
                                     = net_weight_kg_candidate
                            THEN 'reported_three_exact'

                            -- Quantity + alt quantity exactly equal.
                            WHEN q_reported
                                 AND a_reported
                                 AND quantity_kg_candidate
                                     = alt_quantity_kg_candidate
                            THEN 'reported_pair_exact_quantity_alt'

                            -- Quantity + net weight exactly equal.
                            WHEN q_reported
                                 AND n_reported
                                 AND quantity_kg_candidate
                                     = net_weight_kg_candidate
                            THEN 'reported_pair_exact_quantity_net_weight'

                            -- Alt quantity + net weight exactly equal.
                            WHEN a_reported
                                 AND n_reported
                                 AND alt_quantity_kg_candidate
                                     = net_weight_kg_candidate
                            THEN 'reported_pair_exact_alt_net_weight'

                            -- ======================================
                            -- LEVEL 2: REPORTED SOURCE PRECEDENCE
                            -- quantity > alt_quantity > net_weight
                            -- ======================================

                            WHEN q_reported
                            THEN 'reported_quantity_precedence'

                            WHEN a_reported
                            THEN 'reported_alt_quantity_precedence'

                            WHEN n_reported
                            THEN 'reported_net_weight_precedence'

                            -- ======================================
                            -- LEVEL 3: ESTIMATED FALLBACK
                            -- quantity > alt_quantity > net_weight
                            -- ======================================

                            WHEN q_estimated
                            THEN 'estimated_quantity_fallback'

                            WHEN a_estimated
                            THEN 'estimated_alt_quantity_fallback'

                            WHEN n_estimated
                            THEN 'estimated_net_weight_fallback'

                            -- ======================================
                            -- NO USABLE KG EVIDENCE
                            -- ======================================

                            ELSE 'no_kg_evidence'

                        END AS quantity_sync_rule

                    FROM evidence
                )

                SELECT
                    * EXCLUDE (
                        q_reported,
                        a_reported,
                        n_reported,
                        q_estimated,
                        a_estimated,
                        n_estimated,
                        has_quantity_supplementary,
                        has_alt_quantity_supplementary,
                        quantity_sync_rule
                    ),

                    -- ==============================================
                    -- Canonical synchronized kg quantity
                    -- ==============================================

                    CASE quantity_sync_rule

                        WHEN 'reported_three_exact'
                        THEN quantity_kg_candidate

                        WHEN 'reported_pair_exact_quantity_alt'
                        THEN quantity_kg_candidate

                        WHEN 'reported_pair_exact_quantity_net_weight'
                        THEN quantity_kg_candidate

                        WHEN 'reported_pair_exact_alt_net_weight'
                        THEN alt_quantity_kg_candidate

                        WHEN 'reported_quantity_precedence'
                        THEN quantity_kg_candidate

                        WHEN 'reported_alt_quantity_precedence'
                        THEN alt_quantity_kg_candidate

                        WHEN 'reported_net_weight_precedence'
                        THEN net_weight_kg_candidate

                        WHEN 'estimated_quantity_fallback'
                        THEN quantity_kg_candidate

                        WHEN 'estimated_alt_quantity_fallback'
                        THEN alt_quantity_kg_candidate

                        WHEN 'estimated_net_weight_fallback'
                        THEN net_weight_kg_candidate

                        ELSE NULL

                    END AS synchronized_quantity_kg,

                    -- ==============================================
                    -- Canonical supplementary quantity
                    --
                    -- Non-kg quantity has precedence over non-kg
                    -- alt_quantity. Original source fields remain
                    -- preserved in the output.
                    -- ==============================================

                    CASE
                        WHEN has_quantity_supplementary
                        THEN quantity

                        WHEN has_alt_quantity_supplementary
                        THEN alt_quantity

                        ELSE NULL
                    END AS supplementary_quantity,


                    -- ==============================================
                    -- Canonical supplementary unit
                    -- ==============================================

                    CASE
                        WHEN has_quantity_supplementary
                        THEN quantity_unit

                        WHEN has_alt_quantity_supplementary
                        THEN alt_quantity_unit

                        ELSE NULL
                    END AS supplementary_unit,


                    -- ==============================================
                    -- Supplementary source provenance
                    -- ==============================================

                    CASE
                        WHEN has_quantity_supplementary
                        THEN 'quantity'

                        WHEN has_alt_quantity_supplementary
                        THEN 'alt_quantity'

                        ELSE 'unavailable'
                    END AS supplementary_quantity_source,


                    -- ==============================================
                    -- Supplementary estimated-state provenance
                    -- ==============================================

                    CASE
                        WHEN has_quantity_supplementary
                        THEN COALESCE(
                            is_quantity_estimated,
                            FALSE
                        )

                        WHEN has_alt_quantity_supplementary
                        THEN COALESCE(
                            is_alt_quantity_estimated,
                            FALSE
                        )

                        ELSE NULL
                    END AS is_supplementary_quantity_estimated,

                    -- ==============================================
                    -- Source provenance
                    -- ==============================================

                    CASE quantity_sync_rule

                        WHEN 'reported_three_exact'
                        THEN 'quantity_alt_quantity_net_weight'

                        WHEN 'reported_pair_exact_quantity_alt'
                        THEN 'quantity_alt_quantity'

                        WHEN 'reported_pair_exact_quantity_net_weight'
                        THEN 'quantity_net_weight'

                        WHEN 'reported_pair_exact_alt_net_weight'
                        THEN 'alt_quantity_net_weight'

                        WHEN 'reported_quantity_precedence'
                        THEN 'quantity'

                        WHEN 'reported_alt_quantity_precedence'
                        THEN 'alt_quantity'

                        WHEN 'reported_net_weight_precedence'
                        THEN 'net_weight'

                        WHEN 'estimated_quantity_fallback'
                        THEN 'quantity'

                        WHEN 'estimated_alt_quantity_fallback'
                        THEN 'alt_quantity'

                        WHEN 'estimated_net_weight_fallback'
                        THEN 'net_weight'

                        ELSE 'unavailable'

                    END AS synchronized_quantity_source,

                    -- ==============================================
                    -- Synchronization status / applied rule
                    -- ==============================================

                    quantity_sync_rule AS quantity_sync_status,

                    -- ==============================================
                    -- Estimated-state provenance
                    -- ==============================================

                    CASE
                        WHEN quantity_sync_rule LIKE 'reported_%'
                        THEN FALSE

                        WHEN quantity_sync_rule LIKE 'estimated_%'
                        THEN TRUE

                        ELSE NULL
                    END AS is_synchronized_quantity_estimated,

                    -- ==============================================
                    -- Resolution flag
                    -- ==============================================

                    CASE
                        WHEN quantity_sync_rule = 'no_kg_evidence'
                        THEN FALSE

                        ELSE TRUE
                    END AS quantity_sync_resolved

                FROM classified
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
                          'reported_three_exact'
                ) AS n_reported_three_exact,

                count(*) FILTER (
                    WHERE quantity_sync_status LIKE
                          'reported_pair_exact_%'
                ) AS n_reported_pair_exact,

                count(*) FILTER (
                    WHERE quantity_sync_status LIKE
                          'reported_%_precedence'
                ) AS n_reported_precedence,

                count(*) FILTER (
                    WHERE quantity_sync_status LIKE
                          'estimated_%_fallback'
                ) AS n_estimated_fallback,

                count(*) FILTER (
                    WHERE quantity_sync_status =
                          'no_kg_evidence'
                ) AS n_no_kg_evidence,

                count(*) FILTER (
                    WHERE supplementary_quantity > 0
                      AND supplementary_unit IS NOT NULL
                ) AS n_supplementary,

                count(*) FILTER (
                    WHERE supplementary_quantity_source = 'quantity'
                ) AS n_supplementary_from_quantity,

                count(*) FILTER (
                    WHERE supplementary_quantity_source = 'alt_quantity'
                ) AS n_supplementary_from_alt_quantity

            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()

        (
            n_rows,
            n_synchronized,
            n_reported_three_exact,
            n_reported_pair_exact,
            n_reported_precedence,
            n_estimated_fallback,
            n_no_kg_evidence,
            n_supplementary,
            n_supplementary_from_quantity,
            n_supplementary_from_alt_quantity,
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

        supplementary_pct = (
            100.0 * n_supplementary / n_rows
            if n_rows
            else 0.0
        )

        supplementary_source_rows = (
            n_supplementary_from_quantity
            + n_supplementary_from_alt_quantity
        )

        if supplementary_source_rows != n_supplementary:
            raise RuntimeError(
                "Supplementary source accounting failed: "
                f"{supplementary_source_rows:,} sourced rows for "
                f"{n_supplementary:,} populated supplementary rows."
            )

        supplementary_null_mismatch = con.execute(
            """
            SELECT COUNT(*)
            FROM read_parquet(?)
            WHERE (
                supplementary_quantity IS NULL
                AND supplementary_unit IS NOT NULL
            )
            OR (
                supplementary_quantity IS NOT NULL
                AND supplementary_unit IS NULL
            )
            """,
            [str(output_path)],
        ).fetchone()[0]

        if supplementary_null_mismatch:
            raise RuntimeError(
                "Supplementary quantity/unit null-consistency failed: "
                f"{supplementary_null_mismatch:,} mismatched rows."
            )

        classified_rows = (
            n_reported_three_exact
            + n_reported_pair_exact
            + n_reported_precedence
            + n_estimated_fallback
            + n_no_kg_evidence
        )

        if classified_rows != n_rows:
            raise RuntimeError(
                "Quantity synchronization accounting failed: "
                f"classified {classified_rows:,} of {n_rows:,} rows."
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
            "Reported three-source exact agreement: %s",
            f"{n_reported_three_exact:,}",
        )
        logger.info(
            "Reported exact pair agreement: %s",
            f"{n_reported_pair_exact:,}",
        )
        logger.info(
            "Reported source-precedence resolutions: %s",
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
            "Rows with supplementary quantity: %s (%.2f%%)",
            f"{n_supplementary:,}",
            supplementary_pct,
        )
        logger.info(
            "Supplementary quantity selected from quantity: %s",
            f"{n_supplementary_from_quantity:,}",
        )
        logger.info(
            "Supplementary quantity selected from alt_quantity: %s",
            f"{n_supplementary_from_alt_quantity:,}",
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

    write_local_checkpoint(
        checkpoint_path=checkpoint_path,
        step_name="synchronized_quantity",
        run_id=run_id,
        config=config_fp,
        input_fingerprint=input_fp,
        output_path=output_path,
    )

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": output_path,
        "synchronized_quantity_path": output_path,

        "n_rows": int(n_rows),
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
        "n_supplementary": int(
            n_supplementary
        ),
        "n_supplementary_from_quantity": int(
            n_supplementary_from_quantity
        ),
        "n_supplementary_from_alt_quantity": int(
            n_supplementary_from_alt_quantity
        ),

        "synchronized_pct": float(
            synchronized_pct
        ),
        "no_kg_pct": float(no_kg_pct),
        "supplementary_pct": float(
            supplementary_pct
        ),

        "ready_for_diagnostics": True,
    }
