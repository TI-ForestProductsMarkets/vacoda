"""
vacoda.cif_fob.prepare_input

Prepare HS-synced trade data for CIF/FOB analysis.

This module derives a canonical kilogram measure using the hierarchy:

1. reported net weight
2. reported quantity in kg
3. reported alternative quantity in kg
4. estimated net weight
5. estimated quantity in kg
6. estimated alternative quantity in kg
7. reported gross weight
8. estimated gross weight
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import duckdb

from vacoda.config.cif_fob import CifFobConfig


def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _quote_identifier(value: str) -> str:
    """Return a safely quoted SQL identifier."""
    return '"' + value.replace('"', '""') + '"'


def valid_output(path: Path) -> bool:
    """Return True when an existing output appears usable."""
    return path.is_file() and path.stat().st_size > 0


def prepare_cif_fob_input(
    *,
    cfg: CifFobConfig,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """Create the prepared CIF/FOB input parquet."""
    input_path = cfg.input_path()
    output_path = cfg.product_path("prepared_input")

    if not input_path.exists():
        raise FileNotFoundError(
            f"CIF/FOB source parquet not found: {input_path}"
        )

    if not force and valid_output(output_path):
        logger.info(
            "REUSE — Prepared CIF/FOB input already exists: %s",
            output_path,
        )

        return {
            "status": "reused",
            "prepared_input_path": output_path,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = cfg.source_data
    columns = source.columns
    flows = source.supported_flows

    year_col = _quote_identifier(columns.year)
    flow_col = _quote_identifier(columns.flow)

    allocation_col = _quote_identifier(
        columns.allocation_weight
    )

    quantity_col = _quote_identifier(columns.quantity)
    quantity_unit_col = _quote_identifier(
        columns.quantity_unit
    )
    quantity_estimated_col = _quote_identifier(
        columns.quantity_estimated
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

    import_flow = _sql_literal(flows.import_code)
    export_flow = _sql_literal(flows.export_code)

    input_sql = _sql_literal(input_path)
    output_sql = _sql_literal(output_path)

    min_year = int(cfg.scope.min_year)
    max_year = int(cfg.scope.max_year)

    if source.allocate_supplementary_measures:
        allocation_factor_sql = (
            f"COALESCE(CAST({allocation_col} AS DOUBLE), 1.0)"
        )
    else:
        allocation_factor_sql = "1.0"

    logger.info("Preparing CIF/FOB source data.")
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
        "  Allocate physical measures by HS weight: %s",
        source.allocate_supplementary_measures,
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        query = f"""
        WITH filtered AS (
            SELECT *
            FROM read_parquet({input_sql})
            WHERE {year_col} BETWEEN {min_year} AND {max_year}
              AND {flow_col} IN ({import_flow}, {export_flow})
        ),

        canonicalized AS (
            SELECT
                *,

                CASE
                    WHEN CAST({net_weight_col} AS DOUBLE) > 0
                     AND COALESCE(
                         CAST(
                             {net_weight_estimated_col}
                             AS BOOLEAN
                         ),
                         FALSE
                     ) = FALSE
                        THEN CAST(
                            {net_weight_col}
                            AS DOUBLE
                        )

                    WHEN CAST({quantity_col} AS DOUBLE) > 0
                     AND LOWER(
                         TRIM(
                             CAST(
                                 {quantity_unit_col}
                                 AS VARCHAR
                             )
                         )
                     ) = 'kg'
                     AND COALESCE(
                         CAST(
                             {quantity_estimated_col}
                             AS BOOLEAN
                         ),
                         FALSE
                     ) = FALSE
                        THEN CAST(
                            {quantity_col}
                            AS DOUBLE
                        )

                    WHEN CAST(
                        {alt_quantity_col}
                        AS DOUBLE
                    ) > 0
                     AND LOWER(
                         TRIM(
                             CAST(
                                 {alt_quantity_unit_col}
                                 AS VARCHAR
                             )
                         )
                     ) = 'kg'
                     AND COALESCE(
                         CAST(
                             {alt_quantity_estimated_col}
                             AS BOOLEAN
                         ),
                         FALSE
                     ) = FALSE
                        THEN CAST(
                            {alt_quantity_col}
                            AS DOUBLE
                        )

                    WHEN CAST({net_weight_col} AS DOUBLE) > 0
                        THEN CAST(
                            {net_weight_col}
                            AS DOUBLE
                        )

                    WHEN CAST({quantity_col} AS DOUBLE) > 0
                     AND LOWER(
                         TRIM(
                             CAST(
                                 {quantity_unit_col}
                                 AS VARCHAR
                             )
                         )
                     ) = 'kg'
                        THEN CAST(
                            {quantity_col}
                            AS DOUBLE
                        )

                    WHEN CAST(
                        {alt_quantity_col}
                        AS DOUBLE
                    ) > 0
                     AND LOWER(
                         TRIM(
                             CAST(
                                 {alt_quantity_unit_col}
                                 AS VARCHAR
                             )
                         )
                     ) = 'kg'
                        THEN CAST(
                            {alt_quantity_col}
                            AS DOUBLE
                        )

                    WHEN CAST(
                        {gross_weight_col}
                        AS DOUBLE
                    ) > 0
                     AND COALESCE(
                         CAST(
                             {gross_weight_estimated_col}
                             AS BOOLEAN
                         ),
                         FALSE
                     ) = FALSE
                        THEN CAST(
                            {gross_weight_col}
                            AS DOUBLE
                        )

                    WHEN CAST(
                        {gross_weight_col}
                        AS DOUBLE
                    ) > 0
                        THEN CAST(
                            {gross_weight_col}
                            AS DOUBLE
                        )

                    ELSE NULL
                END AS canonical_weight_kg,

                CASE
                    WHEN CAST({net_weight_col} AS DOUBLE) > 0
                     AND COALESCE(
                         CAST(
                             {net_weight_estimated_col}
                             AS BOOLEAN
                         ),
                         FALSE
                     ) = FALSE
                        THEN 'reported_net_weight_kg'

                    WHEN CAST({quantity_col} AS DOUBLE) > 0
                     AND LOWER(
                         TRIM(
                             CAST(
                                 {quantity_unit_col}
                                 AS VARCHAR
                             )
                         )
                     ) = 'kg'
                     AND COALESCE(
                         CAST(
                             {quantity_estimated_col}
                             AS BOOLEAN
                         ),
                         FALSE
                     ) = FALSE
                        THEN 'reported_quantity_kg'

                    WHEN CAST(
                        {alt_quantity_col}
                        AS DOUBLE
                    ) > 0
                     AND LOWER(
                         TRIM(
                             CAST(
                                 {alt_quantity_unit_col}
                                 AS VARCHAR
                             )
                         )
                     ) = 'kg'
                     AND COALESCE(
                         CAST(
                             {alt_quantity_estimated_col}
                             AS BOOLEAN
                         ),
                         FALSE
                     ) = FALSE
                        THEN 'reported_alt_quantity_kg'

                    WHEN CAST({net_weight_col} AS DOUBLE) > 0
                        THEN 'estimated_net_weight_kg'

                    WHEN CAST({quantity_col} AS DOUBLE) > 0
                     AND LOWER(
                         TRIM(
                             CAST(
                                 {quantity_unit_col}
                                 AS VARCHAR
                             )
                         )
                     ) = 'kg'
                        THEN 'estimated_quantity_kg'

                    WHEN CAST(
                        {alt_quantity_col}
                        AS DOUBLE
                    ) > 0
                     AND LOWER(
                         TRIM(
                             CAST(
                                 {alt_quantity_unit_col}
                                 AS VARCHAR
                             )
                         )
                     ) = 'kg'
                        THEN 'estimated_alt_quantity_kg'

                    WHEN CAST(
                        {gross_weight_col}
                        AS DOUBLE
                    ) > 0
                     AND COALESCE(
                         CAST(
                             {gross_weight_estimated_col}
                             AS BOOLEAN
                         ),
                         FALSE
                     ) = FALSE
                        THEN 'reported_gross_weight_kg'

                    WHEN CAST(
                        {gross_weight_col}
                        AS DOUBLE
                    ) > 0
                        THEN 'estimated_gross_weight_kg'

                    ELSE 'missing'
                END AS canonical_weight_source

            FROM filtered
        ),

        enriched AS (
            SELECT
                *,

                CASE
                    WHEN canonical_weight_source IN (
                        'reported_net_weight_kg',
                        'reported_quantity_kg',
                        'reported_alt_quantity_kg'
                    )
                        THEN FALSE

                    WHEN canonical_weight_source IN (
                        'estimated_net_weight_kg',
                        'estimated_quantity_kg',
                        'estimated_alt_quantity_kg',
                        'estimated_gross_weight_kg'
                    )
                        THEN TRUE

                    WHEN canonical_weight_source =
                         'reported_gross_weight_kg'
                        THEN FALSE

                    ELSE NULL
                END AS canonical_weight_estimated,

                canonical_weight_source IN (
                    'reported_gross_weight_kg',
                    'estimated_gross_weight_kg'
                ) AS canonical_weight_is_proxy,

                CASE
                    WHEN canonical_weight_kg > 0
                        THEN canonical_weight_kg
                             * {allocation_factor_sql}
                    ELSE NULL
                END AS canonical_weight_hs_synced

            FROM canonicalized
        )

        SELECT *
        FROM enriched
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

        summary_row = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,

                COUNT(*) FILTER (
                    WHERE canonical_weight_kg > 0
                ) AS n_with_canonical_weight,

                COUNT(*) FILTER (
                    WHERE canonical_weight_source =
                          'reported_net_weight_kg'
                ) AS n_reported_net_weight,

                COUNT(*) FILTER (
                    WHERE canonical_weight_source =
                          'reported_quantity_kg'
                ) AS n_reported_quantity_kg,

                COUNT(*) FILTER (
                    WHERE canonical_weight_source =
                          'reported_alt_quantity_kg'
                ) AS n_reported_alt_quantity_kg,

                COUNT(*) FILTER (
                    WHERE canonical_weight_source =
                          'estimated_net_weight_kg'
                ) AS n_estimated_net_weight,

                COUNT(*) FILTER (
                    WHERE canonical_weight_source =
                          'estimated_quantity_kg'
                ) AS n_estimated_quantity_kg,

                COUNT(*) FILTER (
                    WHERE canonical_weight_source =
                          'estimated_alt_quantity_kg'
                ) AS n_estimated_alt_quantity_kg,

                COUNT(*) FILTER (
                    WHERE canonical_weight_is_proxy
                ) AS n_gross_weight_proxy,

                COUNT(*) FILTER (
                    WHERE canonical_weight_source = 'missing'
                ) AS n_missing_weight

            FROM read_parquet({output_sql})
            """
        ).fetchone()

        metric_names = [
            "n_rows",
            "n_with_canonical_weight",
            "n_reported_net_weight",
            "n_reported_quantity_kg",
            "n_reported_alt_quantity_kg",
            "n_estimated_net_weight",
            "n_estimated_quantity_kg",
            "n_estimated_alt_quantity_kg",
            "n_gross_weight_proxy",
            "n_missing_weight",
        ]

        metrics = dict(
            zip(metric_names, summary_row, strict=True)
        )

        coverage_percent = (
            100.0
            * metrics["n_with_canonical_weight"]
            / metrics["n_rows"]
            if metrics["n_rows"]
            else 0.0
        )

        metrics["canonical_weight_coverage_percent"] = (
            coverage_percent
        )

        logger.info("Prepared CIF/FOB input created.")
        logger.info("  Rows: %s", metrics["n_rows"])
        logger.info(
            "  Rows with canonical weight: %s",
            metrics["n_with_canonical_weight"],
        )
        logger.info(
            "  Canonical weight coverage: %.4f%%",
            coverage_percent,
        )
        logger.info(
            "  Gross-weight proxy rows: %s",
            metrics["n_gross_weight_proxy"],
        )
        logger.info(
            "  Missing canonical weight: %s",
            metrics["n_missing_weight"],
        )
        logger.info("  Output: %s", output_path)

        return {
            "status": "completed",
            "output_path": output_path,
            "prepared_input_path": output_path,
            **metrics,
        }

    finally:
        con.close()