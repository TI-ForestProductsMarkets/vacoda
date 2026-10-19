"""Descriptive diagnostics for published VACODA Release datasets.

Step 4 of Release. These summaries are informational only and never
authorize publication or change the published datasets.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb

from vacoda.config.steps.release import ReleaseConfig
from vacoda.utils.sql import sql_literal


def build_release_diagnostics(
    *,
    cfg: ReleaseConfig,
    standard_path: Path,
    enriched_path: Path,
    audit_path: Path,
    run_id: str,
    release_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """Write one Parquet table with overall and per-year Release summaries.

    The audit is a provenance input, not an approval decision at this stage:
    approval was enforced earlier by the Release workflow and publisher.
    """
    start = perf_counter()
    standard_path = Path(standard_path)
    enriched_path = Path(enriched_path)
    audit_path = Path(audit_path)
    output_path = cfg.product_path("diagnostics", release_id=release_id)

    expected_standard = cfg.product_path("release", release_id=release_id)
    expected_enriched = cfg.enriched_path(release_id=release_id)
    expected_audit = cfg.product_path("release_audit", release_id=release_id)
    if (standard_path != expected_standard
            or enriched_path != expected_enriched
            or audit_path != expected_audit):
        raise ValueError("Release diagnostics input paths do not match release identity.")

    for label, path in (
        ("standard dataset", standard_path),
        ("enriched dataset", enriched_path),
        ("Release audit", audit_path),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Release {label} missing or empty: {path}")

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Release diagnostics already exist (force required): {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    standard_sql = sql_literal(standard_path)
    enriched_sql = sql_literal(enriched_path)
    audit_sql = sql_literal(audit_path)

    con = duckdb.connect()
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        con.execute(f"PRAGMA threads={int(cfg.engine.threads)}")

        # Only read the published artifacts; do not materialize the full dataset
        # into pandas, and do not assign publication permissions here.
        audit_row = con.execute(f"""
            SELECT run_id, release_id, audit_passed
            FROM read_parquet({audit_sql})
        """).fetchall()
        if len(audit_row) != 1 or audit_row[0] != (run_id, release_id, True):
            raise RuntimeError("Release audit identity/status does not match.")

        con.execute(f"""
            CREATE TEMP VIEW standard_release AS
            SELECT * FROM read_parquet({standard_sql})
        """)
        con.execute(f"""
            CREATE TEMP VIEW enriched_release AS
            SELECT * FROM read_parquet({enriched_sql})
        """)

        # Two aggregate sets permit descriptive cross-product comparisons
        # without joining millions of underlying bilateral observations.
        query = """
            WITH s AS (
                SELECT
                    CASE WHEN GROUPING(year) = 1 THEN 'all_years'
                         ELSE 'year' END AS period_type,
                    CASE WHEN GROUPING(year) = 1 THEN NULL
                         ELSE year END AS year,
                    COUNT(*) AS standard_rows,
                    COUNT(DISTINCT target_cmd) AS hs6_codes,
                    COUNT(*) FILTER (
                        WHERE trade_value_reconciled IS NOT NULL
                    ) AS nominal_value_rows,
                    COUNT(*) FILTER (
                        WHERE trade_synced_quantity_kg IS NOT NULL
                    ) AS synced_quantity_rows,
                    COUNT(*) FILTER (
                        WHERE model_prediction_available IS TRUE
                    ) AS model_prediction_available_rows,
                    COUNT(*) FILTER (
                        WHERE model_prediction_quantity_kg IS NOT NULL
                    ) AS standard_prediction_rows,
                    COUNT(*) FILTER (
                        WHERE manual_review_required IS TRUE
                    ) AS manual_review_rows,
                    COUNT(DISTINCT target_cmd) FILTER (
                        WHERE manual_review_required IS TRUE
                    ) AS manual_review_hs6_codes,
                    MEDIAN(trade_value_reconciled) AS median_trade_value_usd,
                    MEDIAN(trade_synced_quantity_kg) AS median_synced_quantity_kg,
                    MEDIAN(model_prediction_quantity_kg) AS median_standard_prediction_kg
                FROM standard_release
                GROUP BY GROUPING SETS ((year), ())
            ),
            e AS (
                SELECT
                    CASE WHEN GROUPING(year) = 1 THEN 'all_years'
                         ELSE 'year' END AS period_type,
                    CASE WHEN GROUPING(year) = 1 THEN NULL
                         ELSE year END AS year,
                    COUNT(*) AS enriched_rows,
                    COUNT(*) FILTER (
                        WHERE model_prediction_quantity_kg IS NOT NULL
                    ) AS raw_model_prediction_rows,
                    COUNT(*) FILTER (
                        WHERE model_input_valid IS TRUE
                    ) AS model_input_valid_rows,
                    COUNT(*) FILTER (
                        WHERE training_selected IS TRUE
                    ) AS training_selected_rows,
                    COUNT(*) FILTER (
                        WHERE hs6_seen IS FALSE
                    ) AS unseen_hs6_rows,
                    COUNT(*) FILTER (
                        WHERE hs4_seen IS FALSE
                    ) AS unseen_hs4_rows,
                    COUNT(*) FILTER (
                        WHERE hs6_seen IS NULL OR hs4_seen IS NULL
                    ) AS support_not_evaluated_rows,
                    COUNT(*) FILTER (
                        WHERE quantity_abs_log_residual IS NOT NULL
                    ) AS comparable_residual_rows,
                    MEDIAN(quantity_abs_log_residual) AS median_abs_log_residual,
                    MEDIAN(trade_sync_uv_usd_per_kg) AS median_trade_sync_uv_usd_per_kg,
                    MEDIAN(model_uv_usd_per_kg) AS median_model_uv_usd_per_kg,
                    MEDIAN(trade_sync_uv_mad_distance) AS median_trade_sync_uv_mad_distance,
                    MEDIAN(model_uv_mad_distance) AS median_model_uv_mad_distance
                FROM enriched_release
                GROUP BY GROUPING SETS ((year), ())
            )
            SELECT
                s.period_type,
                s.year,
                s.standard_rows,
                e.enriched_rows,
                s.hs6_codes,
                s.nominal_value_rows,
                s.synced_quantity_rows,
                s.model_prediction_available_rows,
                s.standard_prediction_rows,
                e.raw_model_prediction_rows,
                s.manual_review_rows,
                s.manual_review_hs6_codes,
                e.model_input_valid_rows,
                e.training_selected_rows,
                e.unseen_hs6_rows,
                e.unseen_hs4_rows,
                e.support_not_evaluated_rows,
                e.comparable_residual_rows,
                s.median_trade_value_usd,
                s.median_synced_quantity_kg,
                s.median_standard_prediction_kg,
                e.median_abs_log_residual,
                e.median_trade_sync_uv_usd_per_kg,
                e.median_model_uv_usd_per_kg,
                e.median_trade_sync_uv_mad_distance,
                e.median_model_uv_mad_distance
            FROM s JOIN e
              ON s.period_type = e.period_type
             AND s.year IS NOT DISTINCT FROM e.year
            ORDER BY CASE WHEN s.period_type = 'all_years' THEN 1 ELSE 0 END,
                     s.year
        """

        # Create a temporary Parquet before promoting the diagnostics artifact.
        temporary_path.unlink(missing_ok=True)
        con.execute(
            f"COPY ({query}) TO {sql_literal(temporary_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )

        summary = con.execute(f"""
            SELECT standard_rows, enriched_rows, manual_review_rows,
                   manual_review_hs6_codes, standard_prediction_rows,
                   raw_model_prediction_rows
            FROM read_parquet({sql_literal(temporary_path)})
            WHERE period_type = 'all_years'
        """).fetchall()
        if len(summary) != 1:
            raise RuntimeError("Release diagnostics missing all-years summary.")
        (n_standard, n_enriched, n_review, n_review_codes,
         n_standard_prediction, n_raw_prediction) = summary[0]

        if not n_standard or not n_enriched:
            raise RuntimeError("Published Release dataset is empty.")

        temporary_path.replace(output_path)
        logger.info("Release diagnostics: %s", output_path)
        logger.info(
            "  Records: standard=%s enriched=%s", f"{n_standard:,}", f"{n_enriched:,}"
        )
        logger.info(
            "  Manual review: %s rows (%s HS6 codes)",
            f"{n_review:,}", f"{n_review_codes:,}"
        )
        logger.info(
            "  Predictions: standard=%s enriched raw=%s",
            f"{n_standard_prediction:,}", f"{n_raw_prediction:,}"
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "release_id": release_id,
            "output_path": output_path,
            "diagnostics_path": output_path,
            "n_rows": int(n_standard),
            "n_manual_review_rows": int(n_review),
            "n_manual_review_hs6_codes": int(n_review_codes),
            "duration_seconds": perf_counter() - start,
        }
    finally:
        temporary_path.unlink(missing_ok=True)
        con.close()
