"""
vacoda.cif_fob.mirror_pairs

Build canonical bilateral import/export mirror pairs for CIF/FOB analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.cif_fob import CifFobConfig
from vacoda.cif_fob.prepare_input import valid_output


def _inspect_existing_mirror_pairs(
    *,
    output_path: Path,
    threads: int,
) -> dict[str, Any]:
    """Read summary metrics from an existing mirror-pair output."""

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(threads)}")

        summary = con.execute(
            """
            SELECT
                COUNT(*) AS n_mirror_groups,

                COUNT(*) FILTER (
                    WHERE has_export_report
                ) AS n_with_export,

                COUNT(*) FILTER (
                    WHERE has_import_report
                ) AS n_with_import,

                COUNT(*) FILTER (
                    WHERE has_both_mirrors
                ) AS n_with_both_mirrors,

                COUNT(*) FILTER (
                    WHERE training_eligible
                ) AS n_training_eligible,

                MEDIAN(raw_value_ratio)
                    FILTER (
                        WHERE raw_value_ratio IS NOT NULL
                    )
                    AS median_raw_value_ratio,

                MEDIAN(unit_value_ratio)
                    FILTER (
                        WHERE unit_value_ratio IS NOT NULL
                    )
                    AS median_unit_value_ratio,

                MEDIAN(weight_concordance)
                    FILTER (
                        WHERE weight_concordance IS NOT NULL
                    )
                    AS median_weight_concordance

            FROM read_parquet(?)
            """,
            [str(output_path)],
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
        "median_raw_value_ratio",
        "median_unit_value_ratio",
        "median_weight_concordance",
    ]

    metrics = dict(
        zip(columns_out, summary, strict=True)
    )

    return {
        "status": "reused",
        "output_path": output_path,
        "mirror_pairs_path": output_path,
        **metrics,
    }

def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _quote_identifier(value: str) -> str:
    """Return a safely quoted SQL identifier."""
    return '"' + value.replace('"', '""') + '"'


def build_mirror_pairs(
    *,
    cfg: CifFobConfig,
    input_path: Path,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """Build the canonical CIF/FOB mirror-pair diagnostic table.

    The table contains one row per:

        year, exporter, importer, target commodity

    Only canonical total-import and total-export flows are included.
    """
    output_path = cfg.product_path("mirror_pairs")

    if not force and valid_output(output_path):
        logger.info(
        "REUSE — Mirror-pair output already exists: %s",
        output_path,
    )
        return _inspect_existing_mirror_pairs(
            output_path=output_path,
            threads=cfg.engine.threads,
        )


    if not input_path.exists():
        raise FileNotFoundError(
            f"CIF/FOB input parquet not found: {input_path}"
        )
    else:
        logger.info("CIF/FOB input parquet found: %s", input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Adjust these attribute names only if your parser uses different names.
    source = cfg.source_data
    columns = source.columns
    flows = source.supported_flows

    year_col = _quote_identifier(columns.year)
    reporter_col = _quote_identifier(columns.reporter)
    partner_col = _quote_identifier(columns.partner)
    flow_col = _quote_identifier(columns.flow)
    commodity_col = _quote_identifier(columns.commodity)
    value_col = _quote_identifier(columns.canonical_value)
    allocation_col = _quote_identifier(columns.allocation_weight)
    direct_cif_col = _quote_identifier(columns.direct_cif_value)
    direct_fob_col = _quote_identifier(columns.direct_fob_value)

    import_flow = _sql_literal(flows.import_code)
    export_flow = _sql_literal(flows.export_code)

    input_sql = _sql_literal(input_path)
    output_sql = _sql_literal(output_path)

    min_year = int(cfg.scope.min_year)
    max_year = int(cfg.scope.max_year)
    minimum_concordance = float(
        cfg.method.training.minimum_quantity_concordance
    )

    logger.info("Building CIF/FOB mirror-pair diagnostics.")
    logger.info("  Input: %s", input_path)
    logger.info("  Output: %s", output_path)
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
        "  Minimum weight concordance: %.3f",
        minimum_concordance,
    )
    logger.info("  DuckDB threads: %s", cfg.engine.threads)

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        query = f"""
        WITH source_rows AS (
            SELECT
                {year_col} AS year,
                {reporter_col} AS reporter_iso,
                {partner_col} AS partner_iso,
                {flow_col} AS flow_code,
                {commodity_col} AS target_cmd,

                CAST({value_col} AS DOUBLE)
                    AS primary_value_hs_synced,

                CAST(canonical_weight_hs_synced AS DOUBLE)
                    AS canonical_weight_hs_synced,

                CAST(canonical_weight_estimated AS BOOLEAN)
                    AS canonical_weight_estimated,

                CAST(canonical_weight_is_proxy AS BOOLEAN)
                    AS canonical_weight_is_proxy,

                CAST({direct_cif_col} AS DOUBLE)
                    * COALESCE(
                        CAST({allocation_col} AS DOUBLE),
                        1.0
                    )
                    AS direct_cif_value_hs_synced,

                CAST({direct_fob_col} AS DOUBLE)
                    * COALESCE(
                        CAST({allocation_col} AS DOUBLE),
                        1.0
                    )
                    AS direct_fob_value_hs_synced

            FROM read_parquet({input_sql})

            WHERE {year_col} BETWEEN {min_year} AND {max_year}
              AND {flow_col} IN ({import_flow}, {export_flow})
              AND {reporter_col} IS NOT NULL
              AND {partner_col} IS NOT NULL
              AND {reporter_col} <> {partner_col}
              AND {commodity_col} IS NOT NULL
              AND COALESCE(is_aggregate, FALSE) = FALSE
        ),

        export_reports AS (
            SELECT
                year,
                reporter_iso AS exporter_iso,
                partner_iso AS importer_iso,
                target_cmd,

                SUM(primary_value_hs_synced)
                    AS export_value_fob,

                SUM(
                    CASE
                        WHEN canonical_weight_hs_synced > 0
                        THEN canonical_weight_hs_synced
                    END
                ) AS export_weight_kg,

                SUM(
                    CASE
                        WHEN direct_fob_value_hs_synced > 0
                        THEN direct_fob_value_hs_synced
                    END
                ) AS export_direct_fob_value,

                SUM(
                    CASE
                        WHEN direct_cif_value_hs_synced > 0
                        THEN direct_cif_value_hs_synced
                    END
                ) AS export_direct_cif_value,

                COUNT(*) AS export_source_rows,

                COUNT(*) FILTER (
                    WHERE primary_value_hs_synced > 0
                ) AS export_positive_value_rows,

                COUNT(*) FILTER (
                    WHERE canonical_weight_hs_synced > 0
                ) AS export_positive_weight_rows,

                COUNT(*) FILTER (
                    WHERE canonical_weight_is_proxy
                ) AS export_proxy_weight_rows,

                COUNT(*) FILTER (
                    WHERE canonical_weight_estimated
                ) AS export_estimated_weight_rows

            FROM source_rows
            WHERE flow_code = {export_flow}

            GROUP BY
                year,
                reporter_iso,
                partner_iso,
                target_cmd
        ),

        import_reports AS (
            SELECT
                year,
                partner_iso AS exporter_iso,
                reporter_iso AS importer_iso,
                target_cmd,

                SUM(primary_value_hs_synced)
                    AS import_value_cif,

                SUM(
                    CASE
                        WHEN canonical_weight_hs_synced > 0
                        THEN canonical_weight_hs_synced
                    END
                ) AS import_weight_kg,

                SUM(
                    CASE
                        WHEN direct_cif_value_hs_synced > 0
                        THEN direct_cif_value_hs_synced
                    END
                ) AS import_direct_cif_value,

                SUM(
                    CASE
                        WHEN direct_fob_value_hs_synced > 0
                        THEN direct_fob_value_hs_synced
                    END
                ) AS import_direct_fob_value,

                COUNT(*) AS import_source_rows,

                COUNT(*) FILTER (
                    WHERE primary_value_hs_synced > 0
                ) AS import_positive_value_rows,

                COUNT(*) FILTER (
                    WHERE canonical_weight_hs_synced > 0
                ) AS import_positive_weight_rows,

                COUNT(*) FILTER (
                    WHERE canonical_weight_is_proxy
                ) AS import_proxy_weight_rows,

                COUNT(*) FILTER (
                    WHERE canonical_weight_estimated
                ) AS import_estimated_weight_rows

            FROM source_rows
            WHERE flow_code = {import_flow}

            GROUP BY
                year,
                partner_iso,
                reporter_iso,
                target_cmd
        ),

        paired AS (
            SELECT
                COALESCE(x.year, m.year) AS year,
                COALESCE(x.exporter_iso, m.exporter_iso)
                    AS exporter_iso,
                COALESCE(x.importer_iso, m.importer_iso)
                    AS importer_iso,
                COALESCE(x.target_cmd, m.target_cmd)
                    AS target_cmd,

                x.export_value_fob,
                m.import_value_cif,

                x.export_weight_kg,
                m.import_weight_kg,

                x.export_direct_fob_value,
                x.export_direct_cif_value,
                m.import_direct_cif_value,
                m.import_direct_fob_value,

                x.export_source_rows,
                m.import_source_rows,

                x.export_positive_value_rows,
                m.import_positive_value_rows,

                x.export_positive_weight_rows,
                m.import_positive_weight_rows,

                x.export_proxy_weight_rows,
                m.import_proxy_weight_rows,

                x.export_estimated_weight_rows,
                m.import_estimated_weight_rows,

                x.year IS NOT NULL AS has_export_report,
                m.year IS NOT NULL AS has_import_report

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

                has_export_report
                    AND has_import_report
                    AS has_both_mirrors,

                CASE
                    WHEN export_value_fob > 0
                     AND import_value_cif > 0
                    THEN import_value_cif / export_value_fob
                END AS raw_value_ratio,

                CASE
                    WHEN export_value_fob > 0
                     AND export_weight_kg > 0
                    THEN export_value_fob
                         / export_weight_kg
                END AS export_unit_value_usd_per_kg,

                CASE
                    WHEN import_value_cif > 0
                     AND import_weight_kg > 0
                    THEN import_value_cif
                         / import_weight_kg
                END AS import_unit_value_usd_per_kg,

                CASE
                    WHEN export_value_fob > 0
                     AND import_value_cif > 0
                     AND export_weight_kg > 0
                     AND import_weight_kg > 0
                    THEN
                        (
                            import_value_cif
                            / import_weight_kg
                        )
                        /
                        (
                            export_value_fob
                            / export_weight_kg
                        )
                END AS unit_value_ratio,

                CASE
                    WHEN export_weight_kg > 0
                     AND import_weight_kg > 0
                    THEN
                        LEAST(
                            export_weight_kg,
                            import_weight_kg
                        )
                        /
                        GREATEST(
                            export_weight_kg,
                            import_weight_kg
                        )
                END AS weight_concordance

            FROM paired
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
                AND weight_concordance >= {minimum_concordance}
            ) AS training_eligible

        FROM diagnostics
        """

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

        summary = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_mirror_groups,

                COUNT(*) FILTER (
                    WHERE has_export_report
                ) AS n_with_export,

                COUNT(*) FILTER (
                    WHERE has_import_report
                ) AS n_with_import,

                COUNT(*) FILTER (
                    WHERE has_both_mirrors
                ) AS n_with_both_mirrors,

                COUNT(*) FILTER (
                    WHERE training_eligible
                ) AS n_training_eligible,

                MEDIAN(raw_value_ratio)
                    FILTER (
                        WHERE raw_value_ratio IS NOT NULL
                    )
                    AS median_raw_value_ratio,

                MEDIAN(unit_value_ratio)
                    FILTER (
                        WHERE unit_value_ratio IS NOT NULL
                    )
                    AS median_unit_value_ratio,

                MEDIAN(weight_concordance)
                    FILTER (
                        WHERE weight_concordance IS NOT NULL
                    )
                    AS median_weight_concordance

            FROM read_parquet({output_sql})
            """
        ).fetchone()

        columns_out = [
            "n_mirror_groups",
            "n_with_export",
            "n_with_import",
            "n_with_both_mirrors",
            "n_training_eligible",
            "median_raw_value_ratio",
            "median_unit_value_ratio",
            "median_weight_concordance",
        ]

        metrics = dict(zip(columns_out, summary, strict=True))

        logger.info("CIF/FOB mirror-pair table built.")
        logger.info("  Rows: %s", metrics["n_mirror_groups"])
        logger.info(
            "  Both mirrors: %s",
            metrics["n_with_both_mirrors"],
        )
        logger.info(
            "  Training eligible: %s",
            metrics["n_training_eligible"],
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
        logger.info("  Output: %s", output_path)

        return {
            "status": "completed",
            "output_path": output_path,
            "mirror_pairs_path": output_path,
            **metrics,
        }

    finally:
        con.close()