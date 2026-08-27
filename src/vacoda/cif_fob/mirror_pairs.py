"""
vacoda.cif_fob.mirror_pairs

Build canonical bilateral import/export mirror pairs for CIF/FOB analysis.

Input contract
--------------
The prepared CIF/FOB input must contain exactly one canonical reporter-side
row per:

    year
    reporter_iso
    partner_iso
    flow_code
    target_cmd

Step 1 has already:

- retained reported, non-aggregate Comtrade evidence;
- aggregated additive Partner2 / transport / HS components;
- allocated supplementary measures during HS synchronization;
- consumed the synchronized kg quantity produced by the Quantity module
  and applied HS allocation where required;
- derived importer-side direct CIF/FOB factors;
- passed structural audits.

Step 2 therefore does not aggregate or allocate the source again. It:

1. validates the canonical Step 1 input;
2. orients exports and imports to:

       year
       exporter_iso
       importer_iso
       target_cmd

3. full-outer-joins both reporting sides;
4. calculates mirror diagnostics;
5. preserves Step 1 quality metadata from each side;
6. audits mirror-pair uniqueness and join accounting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.cif_fob import CifFobConfig
from vacoda.pipeline.checkpoint import validate_local_checkpoint, file_fingerprint, write_local_checkpoint, semantic_config_fingerprint
from vacoda.utils.logger import log_checkpoint_reuse, log_checkpoint_rebuild, log_semantic_check, log_checkpoint_skip

# ======================================================================
# SQL helpers
# ======================================================================


def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _quote_identifier(value: str) -> str:
    """Return a safely quoted SQL identifier."""
    return '"' + value.replace('"', '""') + '"'


# ======================================================================
# Existing-output inspection
# ======================================================================


def _inspect_existing_mirror_pairs(
    *,
    output_path: Path,
    threads: int,
) -> dict[str, Any]:
    """Read and validate summary metrics from an existing mirror output."""

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(threads)}"
        )

        summary = con.execute(
            """
            WITH duplicate_keys AS (
                SELECT
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    COUNT(*) AS n_rows

                FROM read_parquet(?)

                GROUP BY
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd

                HAVING COUNT(*) > 1
            )

            SELECT
                (
                    SELECT COUNT(*)
                    FROM read_parquet(?)
                ) AS n_mirror_groups,

                (
                    SELECT COUNT(*)
                    FROM read_parquet(?)
                    WHERE has_export_report
                ) AS n_with_export,

                (
                    SELECT COUNT(*)
                    FROM read_parquet(?)
                    WHERE has_import_report
                ) AS n_with_import,

                (
                    SELECT COUNT(*)
                    FROM read_parquet(?)
                    WHERE has_both_mirrors
                ) AS n_with_both_mirrors,

                (
                    SELECT COUNT(*)
                    FROM read_parquet(?)
                    WHERE training_eligible
                ) AS n_training_eligible,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_mirror_keys,

                (
                    SELECT MEDIAN(raw_value_ratio)
                    FROM read_parquet(?)
                    WHERE raw_value_ratio IS NOT NULL
                ) AS median_raw_value_ratio,

                (
                    SELECT MEDIAN(unit_value_ratio)
                    FROM read_parquet(?)
                    WHERE unit_value_ratio IS NOT NULL
                ) AS median_unit_value_ratio,

                (
                    SELECT MEDIAN(weight_concordance)
                    FROM read_parquet(?)
                    WHERE weight_concordance IS NOT NULL
                ) AS median_weight_concordance

            """,
            [
                str(output_path),
                str(output_path),
                str(output_path),
                str(output_path),
                str(output_path),
                str(output_path),
                str(output_path),
                str(output_path),
                str(output_path),
            ],
        ).fetchone()

    finally:
        con.close()

    if summary is None:
        raise RuntimeError(
            f"Could not inspect mirror-pair output: {output_path}"
        )

    columns_out = [
        "n_mirror_groups",
        "n_with_export",
        "n_with_import",
        "n_with_both_mirrors",
        "n_training_eligible",
        "n_duplicate_mirror_keys",
        "median_raw_value_ratio",
        "median_unit_value_ratio",
        "median_weight_concordance",
    ]

    metrics = dict(
        zip(
            columns_out,
            summary,
            strict=True,
        )
    )

    if metrics["n_duplicate_mirror_keys"] > 0:
        raise RuntimeError(
            "Existing mirror-pair output contains "
            f"{metrics['n_duplicate_mirror_keys']:,} "
            "duplicate bilateral keys."
        )

    return {
        "status": "reused",
        "audit_passed": True,
        "ready_for_training_sample": True,
        "output_path": output_path,
        "mirror_pairs_path": output_path,
        **metrics,
    }


# ======================================================================
# Main builder
# ======================================================================


def build_mirror_pairs(
    *,
    cfg: CifFobConfig,
    input_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Build one bilateral mirror diagnostic row per:

        year
        exporter_iso
        importer_iso
        target_cmd
    """

    output_path = cfg.product_path(
        "mirror_pairs"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Prepared CIF/FOB input parquet not found: {input_path}"
        )


    checkpoint_path = (
        cfg.output_dir()
        / "checkpoints"
        / "mirror_pairs.json"
    )

    config_fp = semantic_config_fingerprint(
        config=cfg,
        excluded_paths=(
            ("engine", "threads"),
        ),
    )

    input_fp = {
        "prepared_input": file_fingerprint(
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
        log_checkpoint_reuse(
                    logger,
                    "CIF/FOB mirror Pair",
                    reason=reason,
                    cached_output_path=output_path,
                )


        return _inspect_existing_mirror_pairs(
            output_path=output_path,
            threads=cfg.engine.threads,
        )


    log_checkpoint_rebuild(
        logger,
        "CIF/FOB Mirror Pair",
        reason=reason if not force else "forced",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = cfg.source_data
    columns = source.columns
    flows = source.supported_flows

    # ------------------------------------------------------------------
    # Configured canonical columns
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
    value_col = _quote_identifier(
        columns.canonical_value
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

    direct_cif_col = _quote_identifier(
        columns.direct_cif_value
    )
    direct_fob_col = _quote_identifier(
        columns.direct_fob_value
    )

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

    minimum_concordance = float(
        cfg.method.training.minimum_quantity_concordance
    )

    logger.info(
        "Building canonical CIF/FOB mirror-pair diagnostics."
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
        "  Input already canonicalized by Step 1: True"
    )
    logger.info(
        "  Re-aggregate prepared input: False"
    )
    logger.info(
        "  Minimum weight concordance: %.3f",
        minimum_concordance,
    )
    logger.info(
        "  DuckDB threads: %s",
        cfg.engine.threads,
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        # ==============================================================
        # STEP 1 INPUT CONTRACT AUDIT
        # ==============================================================

        input_audit = con.execute(
            f"""
            WITH input_rows AS (
                SELECT *
                FROM read_parquet({input_sql})

                WHERE {year_col}
                      BETWEEN {min_year}
                          AND {max_year}

                  AND {flow_col} IN (
                      {import_flow},
                      {export_flow}
                  )
            ),

            duplicate_keys AS (
                SELECT
                    {year_col},
                    {reporter_col},
                    {partner_col},
                    {flow_col},
                    {commodity_col},
                    COUNT(*) AS n_rows

                FROM input_rows

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
                    FROM input_rows
                ) AS n_input_rows,

                (
                    SELECT COUNT(*)
                    FROM input_rows
                    WHERE {flow_col} = {export_flow}
                ) AS n_export_rows,

                (
                    SELECT COUNT(*)
                    FROM input_rows
                    WHERE {flow_col} = {import_flow}
                ) AS n_import_rows,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_input_keys,

                (
                    SELECT COUNT(*)
                    FROM input_rows
                    WHERE {reporter_col} IS NULL
                       OR {partner_col} IS NULL
                       OR {commodity_col} IS NULL
                       OR {year_col} IS NULL
                ) AS n_null_key_rows,

                (
                    SELECT COUNT(*)
                    FROM input_rows
                    WHERE {reporter_col} = {partner_col}
                ) AS n_self_trade_rows,

                (
                    SELECT COUNT(*)
                    FROM input_rows
                    WHERE {flow_col} = {export_flow}
                      AND (
                          {direct_cif_col} IS NOT NULL
                          OR {direct_fob_col} IS NOT NULL
                          OR direct_cif_fob_factor IS NOT NULL
                      )
                ) AS n_export_rows_with_direct_fields,

                (
                    SELECT COALESCE(
                        SUM(source_row_count),
                        0
                    )
                    FROM input_rows
                ) AS n_underlying_source_components
            """
        ).fetchone()

        input_metric_names = [
            "n_input_rows",
            "n_export_rows",
            "n_import_rows",
            "n_duplicate_input_keys",
            "n_null_key_rows",
            "n_self_trade_rows",
            "n_export_rows_with_direct_fields",
            "n_underlying_source_components",
        ]

        input_metrics = dict(
            zip(
                input_metric_names,
                input_audit,
                strict=True,
            )
        )

        if input_metrics["n_input_rows"] == 0:
            raise RuntimeError(
                "Prepared CIF/FOB input contains no in-scope M/X rows."
            )

        if input_metrics["n_duplicate_input_keys"] > 0:
            raise RuntimeError(
                "Prepared CIF/FOB input violates the Step 1 canonical "
                "grain. Duplicate reporter-side keys: "
                f"{input_metrics['n_duplicate_input_keys']:,}."
            )

        if input_metrics["n_null_key_rows"] > 0:
            raise RuntimeError(
                "Prepared CIF/FOB input contains rows with null "
                "canonical keys: "
                f"{input_metrics['n_null_key_rows']:,}."
            )

        if input_metrics["n_export_rows_with_direct_fields"] > 0:
            raise RuntimeError(
                "Prepared CIF/FOB input contains direct CIF/FOB "
                "fields on export rows: "
                f"{input_metrics['n_export_rows_with_direct_fields']:,}."
            )

        # Self-trade is not useful for bilateral mirror construction.
        # It is excluded below, but logged rather than treated as a
        # Step 1 structural failure.
        logger.info(
            "  Prepared input rows: %s",
            input_metrics["n_input_rows"],
        )
        logger.info(
            "  Prepared export rows: %s",
            input_metrics["n_export_rows"],
        )
        logger.info(
            "  Prepared import rows: %s",
            input_metrics["n_import_rows"],
        )
        logger.info(
            "  Underlying Step 1 source components: %s",
            input_metrics["n_underlying_source_components"],
        )
        logger.info(
            "  Duplicate prepared-input keys: %s",
            input_metrics["n_duplicate_input_keys"],
        )
        logger.info(
            "  Self-trade rows excluded from mirror construction: %s",
            input_metrics["n_self_trade_rows"],
        )

        # ==============================================================
        # MIRROR-PAIR QUERY
        #
        # No GROUP BY is needed. Step 1 already guarantees one row per
        # reporter-side canonical key.
        # ==============================================================

        query = f"""
        WITH source_rows AS (
            SELECT
                {year_col} AS year,
                {reporter_col} AS reporter_iso,
                {partner_col} AS partner_iso,
                {flow_col} AS flow_code,
                {commodity_col} AS target_cmd,

                CAST(
                    {value_col}
                    AS DOUBLE
                ) AS primary_value_hs_synced,

                {synchronized_quantity_col},

                synchronized_quantity_kg_hs_synced,

                {synchronized_quantity_estimated_col},

                {synchronized_quantity_source_col},

                CAST(
                    canonical_weight_is_proxy
                    AS BOOLEAN
                ) AS canonical_weight_is_proxy,

                CAST(
                    {direct_cif_col}
                    AS DOUBLE
                ) AS direct_cif_value,

                CAST(
                    {direct_fob_col}
                    AS DOUBLE
                ) AS direct_fob_value,

                CAST(
                    direct_cif_fob_factor
                    AS DOUBLE
                ) AS direct_cif_fob_factor,

                CAST(
                    direct_factor_covered_primary_value
                    AS DOUBLE
                ) AS direct_factor_covered_primary_value,

                CAST(
                    direct_factor_value_coverage
                    AS DOUBLE
                ) AS direct_factor_value_coverage,

                CAST(
                    source_row_count
                    AS BIGINT
                ) AS source_row_count,

                CAST(
                    direct_pair_source_row_count
                    AS BIGINT
                ) AS direct_pair_source_row_count,

                CAST(
                    source_partner2_count
                    AS BIGINT
                ) AS source_partner2_count,

                CAST(
                    source_mot_count
                    AS BIGINT
                ) AS source_mot_count,

                CAST(
                    source_customs_count
                    AS BIGINT
                ) AS source_customs_count,

                CAST(
                    source_mos_count
                    AS BIGINT
                ) AS source_mos_count,

                CAST(
                    source_hs_count
                    AS BIGINT
                ) AS source_hs_count,

                CAST(
                    source_cmd_count
                    AS BIGINT
                ) AS source_cmd_count,

                CAST(
                    source_cmd_original_count
                    AS BIGINT
                ) AS source_cmd_original_count,

                CAST(
                    estimated_weight_source_row_count
                    AS BIGINT
                ) AS estimated_weight_source_row_count,

                CAST(
                    proxy_weight_source_row_count
                    AS BIGINT
                ) AS proxy_weight_source_row_count,

                CAST(
                    missing_weight_source_row_count
                    AS BIGINT
                ) AS missing_weight_source_row_count,

                canonical_source_policy

            FROM read_parquet({input_sql})

            WHERE {year_col}
                  BETWEEN {min_year}
                      AND {max_year}

              AND {flow_col} IN (
                  {import_flow},
                  {export_flow}
              )

              AND {reporter_col} IS NOT NULL
              AND {partner_col} IS NOT NULL
              AND {commodity_col} IS NOT NULL

              AND {reporter_col} <> {partner_col}
        ),

        export_reports AS (
            SELECT
                year,
                reporter_iso AS exporter_iso,
                partner_iso AS importer_iso,
                target_cmd,

                primary_value_hs_synced
                    AS export_value_fob,

                {synchronized_quantity_col}
                    AS export_synchronized_quantity_kg,

                synchronized_quantity_kg_hs_synced
                    AS export_weight_kg,

                {synchronized_quantity_source_col}
                    AS export_synchronized_quantity_source,

                {synchronized_quantity_estimated_col}
                    AS export_is_synchronized_quantity_estimated,

                canonical_weight_is_proxy
                    AS export_weight_is_proxy,

                source_row_count
                    AS export_component_row_count,

                source_partner2_count
                    AS export_partner2_count,

                source_mot_count
                    AS export_mot_count,

                source_customs_count
                    AS export_customs_count,

                source_mos_count
                    AS export_mos_count,

                source_hs_count
                    AS export_source_hs_count,

                source_cmd_count
                    AS export_source_cmd_count,

                source_cmd_original_count
                    AS export_source_cmd_original_count,

                estimated_weight_source_row_count
                    AS export_estimated_weight_component_count,

                proxy_weight_source_row_count
                    AS export_proxy_weight_component_count,

                missing_weight_source_row_count
                    AS export_missing_weight_component_count,

                canonical_source_policy
                    AS export_source_policy

            FROM source_rows

            WHERE flow_code = {export_flow}
        ),

        import_reports AS (
            SELECT
                year,
                partner_iso AS exporter_iso,
                reporter_iso AS importer_iso,
                target_cmd,

                primary_value_hs_synced
                    AS import_value_cif,

                {synchronized_quantity_col}
                    AS import_synchronized_quantity_kg,

                synchronized_quantity_kg_hs_synced
                    AS import_weight_kg,

                {synchronized_quantity_source_col}
                    AS import_synchronized_quantity_source,

                {synchronized_quantity_estimated_col}
                    AS import_is_synchronized_quantity_estimated,


                canonical_weight_is_proxy
                    AS import_weight_is_proxy,

                direct_cif_value
                    AS import_direct_cif_value,

                direct_fob_value
                    AS import_direct_fob_value,

                direct_cif_fob_factor
                    AS import_direct_cif_fob_factor,

                direct_factor_covered_primary_value
                    AS import_direct_covered_primary_value,

                direct_factor_value_coverage
                    AS import_direct_value_coverage,

                direct_pair_source_row_count
                    AS import_direct_pair_component_count,

                source_row_count
                    AS import_component_row_count,

                source_partner2_count
                    AS import_partner2_count,

                source_mot_count
                    AS import_mot_count,

                source_customs_count
                    AS import_customs_count,

                source_mos_count
                    AS import_mos_count,

                source_hs_count
                    AS import_source_hs_count,

                source_cmd_count
                    AS import_source_cmd_count,

                source_cmd_original_count
                    AS import_source_cmd_original_count,

                estimated_weight_source_row_count
                    AS import_estimated_weight_component_count,

                proxy_weight_source_row_count
                    AS import_proxy_weight_component_count,

                missing_weight_source_row_count
                    AS import_missing_weight_component_count,

                canonical_source_policy
                    AS import_source_policy

            FROM source_rows

            WHERE flow_code = {import_flow}
        ),

        paired AS (
            SELECT
                COALESCE(
                    x.year,
                    m.year
                ) AS year,

                COALESCE(
                    x.exporter_iso,
                    m.exporter_iso
                ) AS exporter_iso,

                COALESCE(
                    x.importer_iso,
                    m.importer_iso
                ) AS importer_iso,

                COALESCE(
                    x.target_cmd,
                    m.target_cmd
                ) AS target_cmd,

                -- Reporter-side values
                x.export_value_fob,
                m.import_value_cif,

                x.export_synchronized_quantity_kg,
                m.import_synchronized_quantity_kg,

                x.export_weight_kg,
                m.import_weight_kg,

                x.export_synchronized_quantity_source,
                m.import_synchronized_quantity_source,

                x.export_is_synchronized_quantity_estimated,
                m.import_is_synchronized_quantity_estimated,

                x.export_weight_is_proxy,
                m.import_weight_is_proxy,

                -- Importer-side direct evidence
                m.import_direct_cif_value,
                m.import_direct_fob_value,
                m.import_direct_cif_fob_factor,
                m.import_direct_covered_primary_value,
                m.import_direct_value_coverage,
                m.import_direct_pair_component_count,

                -- Step 1 component metadata
                x.export_component_row_count,
                m.import_component_row_count,

                x.export_partner2_count,
                m.import_partner2_count,

                x.export_mot_count,
                m.import_mot_count,

                x.export_customs_count,
                m.import_customs_count,

                x.export_mos_count,
                m.import_mos_count,

                x.export_source_hs_count,
                m.import_source_hs_count,

                x.export_source_cmd_count,
                m.import_source_cmd_count,

                x.export_source_cmd_original_count,
                m.import_source_cmd_original_count,

                x.export_estimated_weight_component_count,
                m.import_estimated_weight_component_count,

                x.export_proxy_weight_component_count,
                m.import_proxy_weight_component_count,

                x.export_missing_weight_component_count,
                m.import_missing_weight_component_count,

                x.export_source_policy,
                m.import_source_policy,

                x.year IS NOT NULL
                    AS has_export_report,

                m.year IS NOT NULL
                    AS has_import_report

            FROM export_reports AS x

            FULL OUTER JOIN import_reports AS m
                ON x.year = m.year
               AND x.exporter_iso = m.exporter_iso
               AND x.importer_iso = m.importer_iso
               AND x.target_cmd = m.target_cmd
        ),

        diagnostics AS (
            SELECT
                *,

                (
                    has_export_report
                    AND has_import_report
                ) AS has_both_mirrors,

                CASE
                    WHEN export_value_fob > 0
                     AND import_value_cif > 0
                        THEN (
                            import_value_cif
                            / export_value_fob
                        )
                END AS raw_value_ratio,

                CASE
                    WHEN export_value_fob > 0
                     AND export_weight_kg > 0
                        THEN (
                            export_value_fob
                            / export_weight_kg
                        )
                END AS export_unit_value_usd_per_kg,

                CASE
                    WHEN import_value_cif > 0
                     AND import_weight_kg > 0
                        THEN (
                            import_value_cif
                            / import_weight_kg
                        )
                END AS import_unit_value_usd_per_kg,

                CASE
                    WHEN export_value_fob > 0
                     AND import_value_cif > 0
                     AND export_weight_kg > 0
                     AND import_weight_kg > 0
                        THEN (
                            (
                                import_value_cif
                                / import_weight_kg
                            )
                            /
                            (
                                export_value_fob
                                / export_weight_kg
                            )
                        )
                END AS unit_value_ratio,

                CASE
                    WHEN export_weight_kg > 0
                     AND import_weight_kg > 0
                        THEN (
                            LEAST(
                                export_weight_kg,
                                import_weight_kg
                            )
                            /
                            GREATEST(
                                export_weight_kg,
                                import_weight_kg
                            )
                        )
                END AS weight_concordance,

                (
                    import_direct_cif_fob_factor IS NOT NULL
                ) AS has_import_direct_factor

            FROM paired AS m
        )

        SELECT
            *,

            CASE
                WHEN NOT has_export_report
                    THEN 'missing_export_report'

                WHEN NOT has_import_report
                    THEN 'missing_import_report'

                WHEN export_value_fob IS NULL
                  OR export_value_fob <= 0
                    THEN 'nonpositive_export_value'

                WHEN import_value_cif IS NULL
                  OR import_value_cif <= 0
                    THEN 'nonpositive_import_value'

                WHEN export_weight_kg IS NULL
                  OR export_weight_kg <= 0
                    THEN 'missing_or_nonpositive_export_weight'

                WHEN import_weight_kg IS NULL
                  OR import_weight_kg <= 0
                    THEN 'missing_or_nonpositive_import_weight'

                WHEN weight_concordance < {minimum_concordance}
                    THEN 'low_weight_concordance'

                ELSE 'eligible'
            END AS exclusion_reason,

            (
                has_both_mirrors
                AND export_value_fob > 0
                AND import_value_cif > 0
                AND export_weight_kg > 0
                AND import_weight_kg > 0
                AND weight_concordance
                    >= {minimum_concordance}
            ) AS training_eligible

        FROM diagnostics
        """

        # ==============================================================
        # WRITE OUTPUT
        # ==============================================================

        con.execute(
            f"""
            COPY (
                {query}
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

        output_audit = con.execute(
            f"""
            WITH duplicate_keys AS (
                SELECT
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    COUNT(*) AS n_rows

                FROM read_parquet({output_sql})

                GROUP BY
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd

                HAVING COUNT(*) > 1
            )

            SELECT
                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                ) AS n_mirror_groups,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE has_export_report
                ) AS n_with_export,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE has_import_report
                ) AS n_with_import,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE has_both_mirrors
                ) AS n_with_both_mirrors,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE has_export_report
                      AND NOT has_import_report
                ) AS n_export_only,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE has_import_report
                      AND NOT has_export_report
                ) AS n_import_only,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE training_eligible
                ) AS n_training_eligible,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE has_import_direct_factor
                ) AS n_with_import_direct_factor,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_mirror_keys,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE exporter_iso IS NULL
                       OR importer_iso IS NULL
                       OR target_cmd IS NULL
                       OR year IS NULL
                ) AS n_null_mirror_keys,

                (
                    SELECT MEDIAN(raw_value_ratio)
                    FROM read_parquet({output_sql})
                    WHERE raw_value_ratio IS NOT NULL
                ) AS median_raw_value_ratio,

                (
                    SELECT MEDIAN(unit_value_ratio)
                    FROM read_parquet({output_sql})
                    WHERE unit_value_ratio IS NOT NULL
                ) AS median_unit_value_ratio,

                (
                    SELECT MEDIAN(weight_concordance)
                    FROM read_parquet({output_sql})
                    WHERE weight_concordance IS NOT NULL
                ) AS median_weight_concordance
            """
        ).fetchone()

        output_metric_names = [
            "n_mirror_groups",
            "n_with_export",
            "n_with_import",
            "n_with_both_mirrors",
            "n_export_only",
            "n_import_only",
            "n_training_eligible",
            "n_with_import_direct_factor",
            "n_duplicate_mirror_keys",
            "n_null_mirror_keys",
            "median_raw_value_ratio",
            "median_unit_value_ratio",
            "median_weight_concordance",
        ]

        output_metrics = dict(
            zip(
                output_metric_names,
                output_audit,
                strict=True,
            )
        )

        metrics = {
            **input_metrics,
            **output_metrics,
        }

        # ==============================================================
        # HARD STOP CONDITIONS
        # ==============================================================


        # The self-trade count above includes both flows. Recalculate
        # flow-specific eligible input counts for exact accounting.
        oriented_input_counts = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE {flow_col} = {export_flow}
                      AND {reporter_col} <> {partner_col}
                ) AS n_oriented_export_rows,

                COUNT(*) FILTER (
                    WHERE {flow_col} = {import_flow}
                      AND {reporter_col} <> {partner_col}
                ) AS n_oriented_import_rows

            FROM read_parquet({input_sql})

            WHERE {year_col}
                  BETWEEN {min_year}
                      AND {max_year}

              AND {flow_col} IN (
                  {import_flow},
                  {export_flow}
              )

              AND {reporter_col} IS NOT NULL
              AND {partner_col} IS NOT NULL
              AND {commodity_col} IS NOT NULL
            """
        ).fetchone()

        (
            n_oriented_export_rows,
            n_oriented_import_rows,
        ) = oriented_input_counts

        if output_metrics["n_duplicate_mirror_keys"] > 0:
            raise RuntimeError(
                "Mirror-pair output contains duplicate bilateral keys: "
                f"{output_metrics['n_duplicate_mirror_keys']:,}."
            )

        if output_metrics["n_null_mirror_keys"] > 0:
            raise RuntimeError(
                "Mirror-pair output contains null bilateral keys: "
                f"{output_metrics['n_null_mirror_keys']:,}."
            )

        if (
            output_metrics["n_with_export"]
            != n_oriented_export_rows
        ):
            raise RuntimeError(
                "Mirror-pair full join did not preserve every oriented "
                "export report exactly once: "
                f"output={output_metrics['n_with_export']:,}, "
                f"expected={n_oriented_export_rows:,}."
            )

        if (
            output_metrics["n_with_import"]
            != n_oriented_import_rows
        ):
            raise RuntimeError(
                "Mirror-pair full join did not preserve every oriented "
                "import report exactly once: "
                f"output={output_metrics['n_with_import']:,}, "
                f"expected={n_oriented_import_rows:,}."
            )

        expected_union_rows = (
            output_metrics["n_with_export"]
            + output_metrics["n_with_import"]
            - output_metrics["n_with_both_mirrors"]
        )

        if (
            output_metrics["n_mirror_groups"]
            != expected_union_rows
        ):
            raise RuntimeError(
                "Mirror-pair output violates full-join accounting: "
                f"rows={output_metrics['n_mirror_groups']:,}, "
                f"expected={expected_union_rows:,}."
            )

        if (
            output_metrics["n_training_eligible"]
            > output_metrics["n_with_both_mirrors"]
        ):
            raise RuntimeError(
                "Training-eligible mirror rows exceed rows with both "
                "reporting sides."
            )

        audit_passed = True
        ready_for_training_sample = True

        write_local_checkpoint(
                checkpoint_path=checkpoint_path,
                step_name="mirror_pairs",
                run_id=run_id,
                config=config_fp,
                input_fingerprint=input_fp,
                output_path=output_path,
            )

        # ==============================================================
        # LOG SUMMARY
        # ==============================================================

        logger.info(
            "Canonical CIF/FOB mirror-pair table built."
        )
        logger.info(
            "  Mirror groups: %s",
            metrics["n_mirror_groups"],
        )
        logger.info(
            "  Groups with exporter report: %s",
            metrics["n_with_export"],
        )
        logger.info(
            "  Groups with importer report: %s",
            metrics["n_with_import"],
        )
        logger.info(
            "  Groups with both mirrors: %s",
            metrics["n_with_both_mirrors"],
        )
        logger.info(
            "  Export-only groups: %s",
            metrics["n_export_only"],
        )
        logger.info(
            "  Import-only groups: %s",
            metrics["n_import_only"],
        )
        logger.info(
            "  Training eligible: %s",
            metrics["n_training_eligible"],
        )
        logger.info(
            "  Groups with importer direct factor: %s",
            metrics["n_with_import_direct_factor"],
        )
        logger.info(
            "  Duplicate mirror keys: %s",
            metrics["n_duplicate_mirror_keys"],
        )
        logger.info(
            "  Median raw value ratio: %s",
            metrics["median_raw_value_ratio"],
        )
        logger.info(
            "  Median unit-value ratio: %s",
            metrics["median_unit_value_ratio"],
        )
        logger.info(
            "  Median weight concordance: %s",
            metrics["median_weight_concordance"],
        )
        logger.info(
            "  Step 2 audit passed: %s",
            audit_passed,
        )
        logger.info(
            "  Ready for training-sample construction: %s",
            ready_for_training_sample,
        )
        logger.info(
            "  Output: %s",
            output_path,
        )

        return {
            "status": "completed",
            "audit_passed": audit_passed,
            "ready_for_training_sample": (
                ready_for_training_sample
            ),
            "output_path": output_path,
            "mirror_pairs_path": output_path,
            **metrics,
        }

    finally:
        con.close()

