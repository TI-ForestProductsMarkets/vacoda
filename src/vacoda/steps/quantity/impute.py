"""Estimate missing reporting-side kg from an HS6 unit-value reference."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb

from vacoda.config.steps.quantity import QuantityConfig
from vacoda.utils.sql import sql_literal


def impute_missing_quantities(
    *,
    cfg: QuantityConfig,
    synchronized_quantity_path: Path,
    run_id: str,
    logger,
) -> dict[str, Any]:
    """Write and check the final Quantity artifact."""
    input_path = Path(synchronized_quantity_path)
    output_path = cfg.product_path("imputed_quantity")
    min_reference_rows = cfg.method.imputation.min_reference_rows

    if not input_path.is_file():
        raise FileNotFoundError(f"Synchronized Quantity input missing: {input_path}")

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Quantity synchronization and imputation need distinct paths.")

    if min_reference_rows < 1:
        raise ValueError("min_reference_rows must be positive.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_sql = sql_literal(input_path)
    method = "hs6_year_flow_median_unit_value"

    with TemporaryDirectory(dir=output_path.parent) as temp_dir:
        staged_path = Path(temp_dir) / output_path.name
        staged_sql = sql_literal(staged_path)

        with duckdb.connect() as con:
            con.execute(f"SET threads = {int(cfg.engine.threads)}")
            con.execute(f"""
                COPY (
                    WITH unit_values AS (
                        SELECT
                            year,
                            target_cmd,
                            flow_code,
                            COUNT(*) AS n_reference,
                            MEDIAN(
                                LN(primary_value_hs_synced)
                                - LN(synchronized_quantity_kg)
                            ) AS median_log_unit_value
                        FROM read_parquet({input_sql})
                        WHERE isfinite(synchronized_quantity_kg)
                          AND synchronized_quantity_kg > 0
                          AND is_synchronized_quantity_estimated IS FALSE
                          AND CAST(is_reported AS BOOLEAN) IS TRUE
                          AND CAST(is_aggregate AS BOOLEAN) IS NOT TRUE
                          AND isfinite(primary_value_hs_synced)
                          AND primary_value_hs_synced > 0
                        GROUP BY year, target_cmd, flow_code
                        HAVING COUNT(*) >= {min_reference_rows}
                    ),
                    candidates AS (
                        SELECT
                            q.*,
                            r.n_reference,
                            CASE
                                WHEN isfinite(q.primary_value_hs_synced)
                                 AND q.primary_value_hs_synced > 0
                                THEN EXP(
                                    LN(q.primary_value_hs_synced)
                                    - r.median_log_unit_value
                                )
                            END AS candidate_kg
                        FROM read_parquet({input_sql}) AS q
                        LEFT JOIN unit_values AS r
                          ON q.year = r.year
                         AND q.target_cmd = r.target_cmd
                         AND q.flow_code = r.flow_code
                         AND NOT COALESCE(
                             isfinite(q.synchronized_quantity_kg)
                             AND q.synchronized_quantity_kg > 0,
                             FALSE
                         )
                    ),
                    accepted AS (
                        SELECT
                            * EXCLUDE (candidate_kg),
                            CASE
                                WHEN isfinite(candidate_kg) AND candidate_kg > 0
                                THEN candidate_kg
                            END AS quantity_imputed_kg
                        FROM candidates
                    )
                    SELECT
                        * EXCLUDE (n_reference) REPLACE (
                            COALESCE(
                                quantity_imputed_kg, synchronized_quantity_kg
                            ) AS synchronized_quantity_kg,
                            CASE
                                WHEN quantity_imputed_kg IS NOT NULL THEN TRUE
                                ELSE is_synchronized_quantity_estimated
                            END AS is_synchronized_quantity_estimated,
                            CASE
                                WHEN quantity_imputed_kg IS NOT NULL THEN '{method}'
                                ELSE synchronized_quantity_source
                            END AS synchronized_quantity_source
                        ),
                        synchronized_quantity_kg
                            AS synchronized_quantity_kg_before_imputation,
                        CASE WHEN quantity_imputed_kg IS NOT NULL THEN '{method}' END
                            AS quantity_imputation_method,
                        CASE WHEN quantity_imputed_kg IS NOT NULL THEN n_reference END
                            AS quantity_imputation_reference_count
                    FROM accepted
                ) TO {staged_sql} (FORMAT PARQUET, COMPRESSION ZSTD)
            """)

            input_rows = con.execute(
                f"SELECT COUNT(*) FROM read_parquet({input_sql})"
            ).fetchone()[0]

            output_rows, imputed_rows, missing_rows, invalid_rows = con.execute(f"""
                SELECT
                    COUNT(*),
                    COUNT(quantity_imputed_kg),
                    COUNT(*) FILTER (
                        WHERE NOT COALESCE(
                            isfinite(synchronized_quantity_kg)
                            AND synchronized_quantity_kg > 0,
                            FALSE
                        )
                    ),
                    COUNT(*) FILTER (
                        WHERE CASE
                            WHEN quantity_imputed_kg IS NULL THEN
                                synchronized_quantity_kg IS DISTINCT FROM
                                    synchronized_quantity_kg_before_imputation
                            ELSE
                                COALESCE(
                                    isfinite(synchronized_quantity_kg_before_imputation)
                                    AND synchronized_quantity_kg_before_imputation > 0,
                                    FALSE
                                )
                                OR synchronized_quantity_kg IS DISTINCT FROM
                                    quantity_imputed_kg
                                OR is_synchronized_quantity_estimated IS NOT TRUE
                                OR quantity_imputation_method IS DISTINCT FROM '{method}'
                                OR COALESCE(quantity_imputation_reference_count, 0)
                                    < {min_reference_rows}
                        END
                    )
                FROM read_parquet({staged_sql})
            """).fetchone()

        if input_rows == 0 or output_rows != input_rows or invalid_rows:
            raise RuntimeError(
                "Quantity imputation output failed checks "
                f"(input rows={input_rows:,}, output rows={output_rows:,}, "
                f"invalid rows={invalid_rows:,})."
            )

        staged_path.replace(output_path)

    logger.info("Quantity rows imputed: %s", f"{imputed_rows:,}")
    logger.info("Quantity rows still without usable kg: %s", f"{missing_rows:,}")

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": output_path,
        "imputed_quantity_path": output_path,
        "audit_passed": True,
        "imputed_rows": imputed_rows,
        "remaining_missing_kg_rows": missing_rows,
    }