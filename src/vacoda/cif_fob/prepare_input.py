"""
vacoda.cif_fob.prepare_input

Prepare Quantity-synchronized trade data for CIF/FOB analysis.

Policy
------
The source trade table has already passed HS synchronization and Quantity
synchronization. For CIF/FOB analysis, this module:

1. keeps only rows that are:
   - within the configured year scope;
   - canonical import/export flows (M/X);
   - explicitly reported;
   - non-aggregate;

2. consumes the canonical kilogram measure produced upstream by the
   Quantity module, together with its source and estimated-state provenance;

3. applies the configured HS allocation weight to the synchronized kilogram
   measure where required, producing the physical measure used by CIF/FOB;

4. derives paired direct CIF/FOB values only when both values are positive
   on the same source row;

5. aggregates reported additive components to exactly one row per:

       year
       reporter
       partner
       flow
       target commodity

6. preserves and summarizes physical-quantity provenance across aggregated
   components;

7. audits source selection, aggregation, value preservation, physical-weight
   coverage, direct CIF/FOB consistency, and canonical-key uniqueness.

The CIF/FOB module does not independently reconstruct a quantity hierarchy.
Canonical kilogram source selection is owned by the upstream Quantity module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.cif_fob import CifFobConfig
from vacoda.utils.display import get_symbols, color_text
from vacoda.pipeline.checkpoint import validate_local_checkpoint, file_fingerprint, write_local_checkpoint, semantic_config_fingerprint


# ======================================================================
# Small SQL helpers
# ======================================================================


def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _quote_identifier(value: str) -> str:
    """Return a safely quoted SQL identifier."""
    return '"' + value.replace('"', '""') + '"'


def valid_output(path: Path) -> bool:
    """Return True when an existing output appears usable."""
    return path.is_file() and path.stat().st_size > 0


# ======================================================================
# Main preparation function
# ======================================================================


def prepare_cif_fob_input(
    *,
    cfg: CifFobConfig,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Create the canonical prepared CIF/FOB input parquet.

    The output contains exactly one row per:

        year
        reporter
        partner
        flow
        target commodity
    """

    input_path = cfg.input_path()
    output_path = cfg.product_path("prepared_input")

    if not input_path.exists():
        raise FileNotFoundError(
            f"CIF/FOB source parquet not found: {input_path}"
        )

    checkpoint_path = (
        cfg.output_dir()
        / "checkpoints"
        / "prepared_input.json"
    )

    config_fp = semantic_config_fingerprint(
            config=cfg,
            excluded_paths=(
                ("engine", "threads"),
            ),
        )
    input_fp = {
        "source": file_fingerprint(
            input_path,
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
            "REUSE — Prepared CIF/FOB input checkpoint matches: %s",
            output_path,
        )

        return {
            "status": "reused",
            "output_path": output_path,
            "prepared_input_path": output_path,
            "audit_passed": True,
            "ready_for_mirror_pairs": True,
        }

    logger.info(
        "%s REBUILD — Prepared CIF/FOB input: %s",
        "forced" if force else reason,
        get_symbols("recycle"),
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = cfg.source_data
    columns = source.columns
    flows = source.supported_flows

    # ------------------------------------------------------------------
    # Configured column names
    # ------------------------------------------------------------------

    year_col = _quote_identifier(
        columns.year
    )
    reporter_col = _quote_identifier(
        columns.reporter
    )
    partner_col = _quote_identifier(
        columns.partner
    )
    flow_col = _quote_identifier(
        columns.flow
    )
    commodity_col = _quote_identifier(
        columns.commodity
    )

    canonical_value_col = _quote_identifier(
        columns.canonical_value
    )

    direct_cif_col = _quote_identifier(
        columns.direct_cif_value
    )
    direct_fob_col = _quote_identifier(
        columns.direct_fob_value
    )

    is_reported_col = _quote_identifier(
        columns.is_reported
    )
    is_aggregate_col = _quote_identifier(
        columns.is_aggregate
    )

    allocation_col = _quote_identifier(
        columns.allocation_weight
    )

    quantity_col = _quote_identifier(
        columns.quantity
    )
    quantity_unit_col = _quote_identifier(
        columns.quantity_unit
    )
    quantity_estimated_col = _quote_identifier(
        columns.quantity_estimated
    )

    synchronized_quantity_col = _quote_identifier(
        columns.synchronized_quantity
    )

    synchronized_quantity_source_col = _quote_identifier(
        columns.synchronized_quantity_source
    )

    synchronized_quantity_estimated_col = _quote_identifier(
        columns.synchronized_quantity_estimated
    )

    alt_quantity_col = _quote_identifier(
        columns.alt_quantity
    )
    alt_quantity_unit_col = _quote_identifier(
        columns.alt_quantity_unit
    )
    alt_quantity_estimated_col = _quote_identifier(
        columns.alt_quantity_estimated
    )

    net_weight_col = _quote_identifier(
        columns.net_weight
    )
    net_weight_estimated_col = _quote_identifier(
        columns.net_weight_estimated
    )

    gross_weight_col = _quote_identifier(
        columns.gross_weight
    )
    gross_weight_estimated_col = _quote_identifier(
        columns.gross_weight_estimated
    )

    # ------------------------------------------------------------------
    # Known Comtrade / HS-sync audit dimensions
    # ------------------------------------------------------------------

    partner2_col = _quote_identifier(
        "partner2_iso"
    )
    mot_col = _quote_identifier(
        "mot_code"
    )
    customs_col = _quote_identifier(
        "customs_code"
    )
    mos_col = _quote_identifier(
        "mos_code"
    )
    source_hs_col = _quote_identifier(
        "source_hs"
    )
    source_cmd_col = _quote_identifier(
        "source_cmd"
    )
    source_cmd_original_col = _quote_identifier(
        "source_cmd_original"
    )

    # ------------------------------------------------------------------
    # Scope and SQL-safe values
    # ------------------------------------------------------------------

    import_flow = _sql_literal(
        flows.import_code
    )
    export_flow = _sql_literal(
        flows.export_code
    )

    input_sql = _sql_literal(
        input_path
    )
    output_sql = _sql_literal(
        output_path
    )

    min_year = int(
        cfg.scope.min_year
    )
    max_year = int(
        cfg.scope.max_year
    )

    if source.allocate_supplementary_measures:
        allocation_factor_sql = (
            f"COALESCE("
            f"CAST({allocation_col} AS DOUBLE), "
            f"1.0"
            f")"
        )
    else:
        allocation_factor_sql = "1.0"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    info = get_symbols("info")
    ok = get_symbols("check")
    logger.info(
        "%s Preparing canonical CIF/FOB source data.",
        info,
    )
    logger.info(
        "  Input: %s",
        input_path,
    )
    logger.info(
        "  Output: %s",
        output_path,
    )
    logger.info(
        "  Scope: %s, %s-%s",
        cfg.scope.target_hs,
        min_year,
        max_year,
    )
    logger.info(
        "  Canonical flows: import=%s export=%s",
        flows.import_code,
        flows.export_code,
    )
    logger.info(
        "  Retain reported rows only: True"
    )
    logger.info(
        "  Exclude aggregate rows: True"
    )
    logger.info(
        "  Aggregate reported components to canonical trade grain: True"
    )
    logger.info(
        "  Allocate physical measures by HS weight: %s",
        source.allocate_supplementary_measures,
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        # ==============================================================
        # SOURCE INPUT AUDIT
        #
        # This reads from the original HS-sync input.
        # ==============================================================

        input_audit_row = con.execute(
            f"""
            WITH selected AS (
                SELECT *
                FROM read_parquet({input_sql})

                WHERE {year_col}
                      BETWEEN {min_year}
                          AND {max_year}

                  AND {flow_col} IN (
                      {import_flow},
                      {export_flow}
                  )

                  AND COALESCE(
                      CAST(
                          {is_reported_col}
                          AS BOOLEAN
                      ),
                      FALSE
                  ) = TRUE

                  AND COALESCE(
                      CAST(
                          {is_aggregate_col}
                          AS BOOLEAN
                      ),
                      FALSE
                  ) = FALSE
            ),

            selected_groups AS (
                SELECT
                    {year_col},
                    {reporter_col},
                    {partner_col},
                    {flow_col},
                    {commodity_col},

                    COUNT(*) AS n_source_rows

                FROM selected

                GROUP BY
                    {year_col},
                    {reporter_col},
                    {partner_col},
                    {flow_col},
                    {commodity_col}
            )

            SELECT
                (
                    SELECT COUNT(*)
                    FROM read_parquet({input_sql})
                ) AS n_input_rows,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({input_sql})
                    WHERE {year_col}
                          BETWEEN {min_year}
                              AND {max_year}
                ) AS n_rows_in_scope,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({input_sql})
                    WHERE {year_col}
                          BETWEEN {min_year}
                              AND {max_year}
                      AND {flow_col} IN (
                          {import_flow},
                          {export_flow}
                      )
                ) AS n_rows_in_scope_and_flows,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({input_sql})
                    WHERE {year_col}
                          BETWEEN {min_year}
                              AND {max_year}
                      AND {flow_col} IN (
                          {import_flow},
                          {export_flow}
                      )
                      AND COALESCE(
                          CAST(
                              {is_reported_col}
                              AS BOOLEAN
                          ),
                          FALSE
                      ) = TRUE
                ) AS n_rows_in_scope_flows_reported,

                (
                    SELECT COUNT(*)
                    FROM selected
                ) AS n_selected_source_rows,

                (
                    SELECT COUNT(*)
                    FROM selected_groups
                ) AS n_selected_canonical_groups,

                (
                    SELECT COUNT(*)
                    FROM selected_groups
                    WHERE n_source_rows = 1
                ) AS n_single_source_groups,

                (
                    SELECT COUNT(*)
                    FROM selected_groups
                    WHERE n_source_rows > 1
                ) AS n_multi_source_groups,

                (
                    SELECT MAX(n_source_rows)
                    FROM selected_groups
                ) AS max_source_rows_per_group,

                (
                    SELECT COALESCE(
                        SUM(
                            CAST(
                                {canonical_value_col}
                                AS DOUBLE
                            )
                        ),
                        0.0
                    )
                    FROM selected
                ) AS selected_source_value_sum
            """
        ).fetchone()

        input_metric_names = [
            "n_input_rows",
            "n_rows_in_scope",
            "n_rows_in_scope_and_flows",
            "n_rows_in_scope_flows_reported",
            "n_selected_source_rows",
            "n_selected_canonical_groups",
            "n_single_source_groups",
            "n_multi_source_groups",
            "max_source_rows_per_group",
            "selected_source_value_sum",
        ]

        input_metrics = dict(
            zip(
                input_metric_names,
                input_audit_row,
                strict=True,
            )
        )

        # ==============================================================
        # CANONICAL PREPARATION QUERY
        # ==============================================================

        preparation_query = f"""
        WITH filtered AS (
            SELECT *
            FROM read_parquet({input_sql})

            WHERE {year_col}
                  BETWEEN {min_year}
                      AND {max_year}

              AND {flow_col} IN (
                  {import_flow},
                  {export_flow}
              )

              AND COALESCE(
                  CAST(
                      {is_reported_col}
                      AS BOOLEAN
                  ),
                  FALSE
              ) = TRUE

              AND COALESCE(
                  CAST(
                      {is_aggregate_col}
                      AS BOOLEAN
                  ),
                  FALSE
              ) = FALSE
        ),

        row_canonicalized AS (
            SELECT
                *,

                -- --------------------------------------------------
                -- Canonical weight value
                --
                -- Quantity synchronization is performed upstream by
                -- the Quantity module. CIF/FOB therefore consumes the
                -- approved synchronized kg value rather than deriving
                -- an independent weight hierarchy.
                -- --------------------------------------------------

                CAST(
                    {synchronized_quantity_col}
                    AS DOUBLE
                ) AS row_canonical_weight_kg,

                -- --------------------------------------------------
                -- Canonical weight source
                -- --------------------------------------------------

                CAST(
                    {synchronized_quantity_source_col}
                    AS VARCHAR
                ) AS row_canonical_weight_source,

                -- --------------------------------------------------
                -- Paired direct CIF/FOB values
                --
                -- Both must be positive on the same source row.
                -- --------------------------------------------------

                CASE
                    WHEN {flow_col} = {import_flow}
                    AND CAST(
                        {direct_cif_col}
                        AS DOUBLE
                    ) > 0
                     AND CAST(
                        {direct_fob_col}
                        AS DOUBLE
                    ) > 0
                        THEN (
                            CAST(
                                {direct_cif_col}
                                AS DOUBLE
                            )
                            * {allocation_factor_sql}
                        )
                    ELSE NULL
                END AS row_paired_cif_value,

                CASE
                    WHEN {flow_col} = {import_flow} 
                    AND CAST(
                        {direct_cif_col}
                        AS DOUBLE
                    ) > 0
                     AND CAST(
                        {direct_fob_col}
                        AS DOUBLE
                    ) > 0
                        THEN (
                            CAST(
                                {direct_fob_col}
                                AS DOUBLE
                            )
                            * {allocation_factor_sql}
                        )
                    ELSE NULL
                END AS row_paired_fob_value,

                CASE
                    WHEN {flow_col} = {import_flow}
                    AND CAST(
                        {direct_cif_col}
                        AS DOUBLE
                    ) > 0
                     AND CAST(
                        {direct_fob_col}
                        AS DOUBLE
                    ) > 0
                        THEN CAST(
                            {canonical_value_col}
                            AS DOUBLE
                        )
                    ELSE NULL
                END AS row_paired_primary_value

            FROM filtered
        ),

        row_enriched AS (
            SELECT
                *,

                CAST(
                    {synchronized_quantity_estimated_col}
                    AS BOOLEAN
                ) AS row_canonical_weight_estimated,

                FALSE AS row_canonical_weight_is_proxy,

                CASE
                    WHEN row_canonical_weight_kg > 0
                        THEN (
                            row_canonical_weight_kg
                            * {allocation_factor_sql}
                        )
                    ELSE NULL
                END AS row_canonical_weight_hs_synced,

                (
                    row_paired_cif_value > 0
                    AND row_paired_fob_value > 0
                ) AS row_has_paired_direct_values

            FROM row_canonicalized
        ),

        aggregated AS (
            SELECT
                -- Canonical trade key
                {year_col},
                {reporter_col},
                {partner_col},
                {flow_col},
                {commodity_col},

                -- ----------------------------------------------
                -- Canonical value and physical measure
                -- ----------------------------------------------

                SUM(
                    CAST(
                        {canonical_value_col}
                        AS DOUBLE
                    )
                ) AS {canonical_value_col},

                SUM(
                    row_canonical_weight_kg
                ) AS canonical_weight_kg,

                SUM(
                    row_canonical_weight_hs_synced
                ) AS canonical_weight_hs_synced,

                CASE
                    WHEN COUNT(
                        DISTINCT
                        row_canonical_weight_source
                    ) = 1
                        THEN MIN(
                            row_canonical_weight_source
                        )
                    ELSE 'mixed'
                END AS canonical_weight_source,

                BOOL_OR(
                    COALESCE(
                        row_canonical_weight_estimated,
                        FALSE
                    )
                ) AS canonical_weight_estimated,

                BOOL_OR(
                    COALESCE(
                        row_canonical_weight_is_proxy,
                        FALSE
                    )
                ) AS canonical_weight_is_proxy,

                -- ----------------------------------------------
                -- Paired direct CIF/FOB values
                --
                -- These output columns deliberately contain only
                -- same-source-row paired values.
                -- ----------------------------------------------

                SUM(
                    row_paired_cif_value
                ) AS {direct_cif_col},

                SUM(
                    row_paired_fob_value
                ) AS {direct_fob_col},

                SUM(
                    row_paired_cif_value
                )
                /
                NULLIF(
                    SUM(
                        row_paired_fob_value
                    ),
                    0
                ) AS direct_cif_fob_factor,

                SUM(
                    row_paired_primary_value
                ) AS direct_factor_covered_primary_value,

                SUM(
                    row_paired_primary_value
                )
                /
                NULLIF(
                    SUM(
                        CAST(
                            {canonical_value_col}
                            AS DOUBLE
                        )
                    ),
                    0
                ) AS direct_factor_value_coverage,

                -- ----------------------------------------------
                -- Source-component audit metadata
                -- ----------------------------------------------

                COUNT(*) AS source_row_count,

                COUNT(*) FILTER (
                    WHERE row_has_paired_direct_values
                ) AS direct_pair_source_row_count,

                COUNT(DISTINCT {partner2_col})
                    AS source_partner2_count,

                COUNT(DISTINCT {mot_col})
                    AS source_mot_count,

                COUNT(DISTINCT {customs_col})
                    AS source_customs_count,

                COUNT(DISTINCT {mos_col})
                    AS source_mos_count,

                COUNT(DISTINCT {source_hs_col})
                    AS source_hs_count,

                COUNT(DISTINCT {source_cmd_col})
                    AS source_cmd_count,

                COUNT(
                    DISTINCT
                    {source_cmd_original_col}
                ) AS source_cmd_original_count,

                COUNT(*) FILTER (
                    WHERE row_canonical_weight_estimated
                ) AS estimated_weight_source_row_count,

                COUNT(*) FILTER (
                    WHERE row_canonical_weight_is_proxy
                ) AS proxy_weight_source_row_count,

                COUNT(*) FILTER (
                    WHERE row_canonical_weight_kg IS NULL
                       OR row_canonical_weight_kg <= 0
                ) AS missing_weight_source_row_count,

                'reported_non_aggregate_components'
                    AS canonical_source_policy

            FROM row_enriched

            GROUP BY
                {year_col},
                {reporter_col},
                {partner_col},
                {flow_col},
                {commodity_col}
        )

        SELECT *
        FROM aggregated
        """

        # ==============================================================
        # WRITE CANONICAL OUTPUT
        # ==============================================================

        con.execute(
            f"""
            COPY (
                {preparation_query}
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
            WITH duplicate_keys AS (
                SELECT
                    {year_col},
                    {reporter_col},
                    {partner_col},
                    {flow_col},
                    {commodity_col},
                    COUNT(*) AS n_rows

                FROM read_parquet({output_sql})

                GROUP BY
                    {year_col},
                    {reporter_col},
                    {partner_col},
                    {flow_col},
                    {commodity_col}

                HAVING COUNT(*) > 1
            )

            SELECT
                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                ) AS n_output_rows,

                (
                    SELECT COALESCE(
                        SUM(source_row_count),
                        0
                    )
                    FROM read_parquet({output_sql})
                ) AS n_represented_source_rows,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_canonical_keys,

                (
                    SELECT COALESCE(
                        SUM(
                            CAST(
                                {canonical_value_col}
                                AS DOUBLE
                            )
                        ),
                        0.0
                    )
                    FROM read_parquet({output_sql})
                ) AS output_value_sum,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE canonical_weight_hs_synced > 0
                ) AS n_with_canonical_weight,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE canonical_weight_hs_synced IS NULL
                       OR canonical_weight_hs_synced <= 0
                ) AS n_without_canonical_weight,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE canonical_weight_estimated
                ) AS n_groups_using_estimated_weight,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE canonical_weight_is_proxy
                ) AS n_groups_using_proxy_weight,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE source_row_count = 1
                ) AS n_single_component_output_groups,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE source_row_count > 1
                ) AS n_multi_component_output_groups,

                (
                    SELECT MAX(source_row_count)
                    FROM read_parquet({output_sql})
                ) AS max_components_per_output_group,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE direct_pair_source_row_count > 0
                ) AS n_groups_with_direct_pair,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE direct_cif_fob_factor > 0
                ) AS n_groups_with_direct_factor

            """
        ).fetchone()

        output_metric_names = [
            "n_output_rows",
            "n_represented_source_rows",
            "n_duplicate_canonical_keys",
            "output_value_sum",
            "n_with_canonical_weight",
            "n_without_canonical_weight",
            "n_groups_using_estimated_weight",
            "n_groups_using_proxy_weight",
            "n_single_component_output_groups",
            "n_multi_component_output_groups",
            "max_components_per_output_group",
            "n_groups_with_direct_pair",
            "n_groups_with_direct_factor",
        ]

        output_metrics = dict(
            zip(
                output_metric_names,
                output_audit_row,
                strict=True,
            )
        )

        metrics = {
            **input_metrics,
            **output_metrics,
        }

        # ==============================================================
        # DERIVED AUDIT METRICS
        # ==============================================================

        selected_share_percent = (
            100.0
            * metrics["n_selected_source_rows"]
            / metrics["n_input_rows"]
            if metrics["n_input_rows"]
            else 0.0
        )

        canonical_weight_coverage_percent = (
            100.0
            * metrics["n_with_canonical_weight"]
            / metrics["n_output_rows"]
            if metrics["n_output_rows"]
            else 0.0
        )

        aggregation_reduction_percent = (
            100.0
            * (
                1.0
                - (
                    metrics["n_output_rows"]
                    / metrics["n_selected_source_rows"]
                )
            )
            if metrics["n_selected_source_rows"]
            else 0.0
        )

        source_value_sum = float(
            metrics["selected_source_value_sum"]
            or 0.0
        )
        output_value_sum = float(
            metrics["output_value_sum"]
            or 0.0
        )

        value_difference = (
            output_value_sum
            - source_value_sum
        )

        value_relative_difference = (
            value_difference
            / source_value_sum
            if source_value_sum != 0
            else 0.0
        )

        metrics[
            "selected_input_share_percent"
        ] = selected_share_percent

        metrics[
            "canonical_weight_coverage_percent"
        ] = canonical_weight_coverage_percent

        metrics[
            "aggregation_reduction_percent"
        ] = aggregation_reduction_percent

        metrics[
            "value_difference"
        ] = value_difference

        metrics[
            "value_relative_difference"
        ] = value_relative_difference

        # ==============================================================
        # HARD STOP CONDITIONS
        # ==============================================================

        if (
            metrics["n_output_rows"]
            != metrics["n_selected_canonical_groups"]
        ):
            raise RuntimeError(
                "Canonical CIF/FOB output row count does not "
                "match the number of selected canonical groups: "
                f"output={metrics['n_output_rows']:,}, "
                f"expected={metrics['n_selected_canonical_groups']:,}."
            )

        if (
            metrics["n_represented_source_rows"]
            != metrics["n_selected_source_rows"]
        ):
            raise RuntimeError(
                "Canonical CIF/FOB output does not represent every "
                "selected source row exactly once: "
                f"represented={metrics['n_represented_source_rows']:,}, "
                f"selected={metrics['n_selected_source_rows']:,}."
            )

        if metrics["n_duplicate_canonical_keys"] > 0:
            raise RuntimeError(
                "Canonical CIF/FOB output contains "
                f"{metrics['n_duplicate_canonical_keys']:,} "
                "duplicate canonical trade keys."
            )

        # Allow only tiny floating-point summation differences.
        if abs(value_relative_difference) > 1e-10:
            raise RuntimeError(
                "Canonical CIF/FOB aggregation did not preserve "
                "the selected source value total: "
                f"source={source_value_sum:.6f}, "
                f"output={output_value_sum:.6f}, "
                f"relative_difference="
                f"{value_relative_difference:.12e}."
            )

        if metrics["n_output_rows"] == 0:
            raise RuntimeError(
                "Canonical CIF/FOB preparation produced no rows."
            )


        ## Additional checks on direct CIF/FOB paired values and factors
        direct_pair_audit = con.execute(
            f"""
            SELECT COUNT(*) FILTER (
                WHERE direct_pair_source_row_count > source_row_count
            ) AS invalid_pair_row_counts,

            COUNT(*) FILTER (
                WHERE direct_cif_fob_factor IS NOT NULL
                    AND (
                    {direct_cif_col} <= 0
                    OR {direct_fob_col} <= 0
                )
                ) AS invalid_direct_factors,
            
            COUNT(*) FILTER (
                WHERE direct_factor_value_coverage < -1e-12
                OR direct_factor_value_coverage
                  > 1.000000001
                ) AS invalid_direct_coverage,


            COUNT(*) FILTER (
                WHERE direct_cif_fob_factor IS NOT NULL
                  AND {flow_col} <> {import_flow}
            ) AS direct_factors_on_non_import_rows,

            COUNT(*) FILTER (
            WHERE {flow_col} <> {import_flow}
              AND (
                    {direct_cif_col} IS NOT NULL
                    OR {direct_fob_col} IS NOT NULL
                    OR direct_factor_covered_primary_value IS NOT NULL
                )
            ) AS paired_values_on_non_import_rows

        FROM read_parquet({output_sql})
        """
        ).fetchone()
        
        (
        invalid_pair_row_counts,
        invalid_direct_factors,
        invalid_direct_coverage,
        direct_factors_on_non_import_rows,
        paired_values_on_non_import_rows,
        ) = direct_pair_audit
        
        if invalid_pair_row_counts > 0:
            raise RuntimeError(
        f"Direct CIF/FOB paired-row count exceeds source-row count, invalid count: {invalid_pair_row_counts}."
        )

        if invalid_direct_coverage > 0:
            raise RuntimeError(
                    "Direct-factor primary-value coverage falls "
                    "outside [0, 1], "
                    f"invalid count: {invalid_direct_coverage}."
        )

        if paired_values_on_non_import_rows > 0:
            raise RuntimeError(
                "Paired direct CIF/FOB fields were retained "
                "for non-import groups, "
                f"invalid count: {paired_values_on_non_import_rows}."
            )
           
        if invalid_direct_factors > 0:
            raise RuntimeError(
        f"Prepared output contains invalid direct CIF/FOB factors, invalid count: {invalid_direct_factors}."
         )

        if direct_factors_on_non_import_rows > 0:
            raise RuntimeError(
        f"Direct CIF/FOB factors found on non-import rows, invalid count: {direct_factors_on_non_import_rows}."
        )


        direct_coverage_summary = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER  (
            WHERE direct_cif_fob_factor > 0
            ) AS n_direct_groups,

            MEDIAN(
                direct_factor_value_coverage
            ) FILTER (
                WHERE direct_cif_fob_factor > 0
            ) AS median_direct_coverage,

            QUANTILE_CONT(
                direct_factor_value_coverage,
                0.05
            ) FILTER (
                WHERE direct_cif_fob_factor > 0
            ) AS p05_direct_coverage,

            COUNT(*) FILTER (
                WHERE direct_cif_fob_factor > 0
                AND direct_factor_value_coverage >= 0.90
            ) AS n_direct_groups_coverage_90

        FROM read_parquet({output_sql})
        """
        ).fetchone()
        (
            n_direct_groups,
            median_direct_coverage,
            p05_direct_coverage,
            n_direct_groups_coverage_90,
        ) = direct_coverage_summary

        logger.info(
            f"{info} Direct CIF/FOB factor audit for groups with direct factors: "
            f"n_direct_groups={n_direct_groups}, "
            f"median_direct_coverage={median_direct_coverage:.4%}, "
            f"p05_direct_coverage={p05_direct_coverage:.4%}, "
            f"n_direct_groups_coverage_90={n_direct_groups_coverage_90}."
        )


        audit_passed = True
        ready_for_mirror_pairs = audit_passed

        write_local_checkpoint(
                checkpoint_path=checkpoint_path,
                step_name="prepared_input",
                run_id=run_id,
                config=config_fp,
                input_fingerprint=input_fp,
                output_path=output_path,
            )

        # ==============================================================
        # LOG RESULTS
        # ==============================================================

        logger.info(
            "%s Canonical CIF/FOB input created.",
            ok,
        )

        logger.info(
            "  Source input rows: %s",
            metrics["n_input_rows"],
        )
        logger.info(
            "  Rows in year scope: %s",
            metrics["n_rows_in_scope"],
        )
        logger.info(
            "  Rows in supported flows: %s",
            metrics["n_rows_in_scope_and_flows"],
        )
        logger.info(
            "  Reported rows in supported scope: %s",
            metrics[
                "n_rows_in_scope_flows_reported"
            ],
        )
        logger.info(
            "  Selected reported non-aggregate rows: %s",
            metrics["n_selected_source_rows"],
        )
        logger.info(
            "  Selected share of source input: %.4f%%",
            selected_share_percent,
        )

        logger.info(
            "  Canonical trade groups: %s",
            metrics["n_selected_canonical_groups"],
        )
        logger.info(
            "  Single-component source groups: %s",
            metrics["n_single_source_groups"],
        )
        logger.info(
            "  Multi-component source groups: %s",
            metrics["n_multi_source_groups"],
        )
        logger.info(
            "  Maximum source rows per group: %s",
            metrics["max_source_rows_per_group"],
        )

        logger.info(
            "  Output rows: %s",
            metrics["n_output_rows"],
        )
        logger.info(
            "  Represented source rows: %s",
            metrics["n_represented_source_rows"],
        )
        logger.info(
            "  Aggregation row reduction: %.4f%%",
            aggregation_reduction_percent,
        )
        logger.info(
            "  Duplicate canonical keys: %s",
            metrics["n_duplicate_canonical_keys"],
        )

        logger.info(
            "  Source selected value sum: %.6f",
            source_value_sum,
        )
        logger.info(
            "  Canonical output value sum: %.6f",
            output_value_sum,
        )
        logger.info(
            "  Value relative difference: %.12e",
            value_relative_difference,
        )

        logger.info(
            "  Groups with canonical weight: %s",
            metrics["n_with_canonical_weight"],
        )
        logger.info(
            "  Canonical weight coverage: %.4f%%",
            canonical_weight_coverage_percent,
        )
        logger.info(
            "  Groups without canonical weight: %s",
            metrics["n_without_canonical_weight"],
        )
        logger.info(
            "  Groups using estimated weight: %s",
            metrics["n_groups_using_estimated_weight"],
        )
        logger.info(
            "  Groups using gross-weight proxy: %s",
            metrics["n_groups_using_proxy_weight"],
        )

        logger.info(
            "  Single-component output groups: %s",
            metrics["n_single_component_output_groups"],
        )
        logger.info(
            "  Multi-component output groups: %s",
            metrics["n_multi_component_output_groups"],
        )
        logger.info(
            "  Maximum components per output group: %s",
            metrics["max_components_per_output_group"],
        )

        logger.info(
            "  Groups with paired direct CIF/FOB rows: %s",
            metrics["n_groups_with_direct_pair"],
        )
        logger.info(
            "  Groups with direct CIF/FOB factor: %s",
            metrics["n_groups_with_direct_factor"],
        )

        logger.info(
            "  Output: %s",
            output_path,
        )


        logger.info("  Step 1 audit passed: %s", audit_passed)
        logger.info("  Ready for mirror-pair construction: %s", ready_for_mirror_pairs)


        return {
            "status": "completed",
            "output_path": output_path,
            "audit_passed": audit_passed,
            "ready_for_mirror_pairs": ready_for_mirror_pairs,
            "prepared_input_path": output_path,
            **metrics,
        }

    finally:
        con.close()
