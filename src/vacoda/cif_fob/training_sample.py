"""
vacoda.cif_fob.training_sample

Build the CIF/FOB model-training sample from canonical bilateral
mirror pairs.

The training sample preserves multiple eligibility levels:

1. value_pair_eligible
2. unit_value_pair_eligible
3. high_weight_concordance
4. model_eligible

Outlier removal is deliberately deferred. This module preserves the
raw observed factors and creates diagnostic flags so that later model
selection remains transparent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.cif_fob import CifFobConfig
from vacoda.cif_fob.prepare_input import valid_output

def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def build_training_sample(
    *,
    cfg: CifFobConfig,
    input_path: Path,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """Build the CIF/FOB training-sample parquet.

    Parameters
    ----------
    cfg:
        Parsed CIF/FOB configuration.
    input_path:
        Path to the mirror-pair parquet.
    logger:
        Pipeline logger.

    Returns
    -------
    dict[str, Any]
        Output paths and training-sample audit metrics.
    """
    output_path = cfg.product_path("training_sample")

    if not force and valid_output(output_path):
        logger.info(
            "REUSE — CIF/FOB training sample already exists: %s",
            output_path,
        )

        return {
            "output_path": output_path,
            "training_sample_path": output_path,
            "status": "reused",
        }

    if not input_path.exists():
        raise FileNotFoundError(
            f"CIF/FOB mirror-pair parquet not found: {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_sql = _sql_literal(input_path)
    output_sql = _sql_literal(output_path)

    minimum_weight_concordance = float(
        cfg.method.training.minimum_quantity_concordance
    )

    logger.info("Building CIF/FOB training sample.")
    logger.info("  Input: %s", input_path)
    logger.info("  Output: %s", output_path)
    logger.info(
        "  Minimum weight concordance: %.3f",
        minimum_weight_concordance,
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        query = f"""
        WITH classified AS (
            SELECT
                *,

                (
                    has_both_mirrors
                    AND export_value_fob > 0
                    AND import_value_cif > 0
                ) AS value_pair_eligible,

                (
                    has_both_mirrors
                    AND export_value_fob > 0
                    AND import_value_cif > 0
                    AND export_weight_kg > 0
                    AND import_weight_kg > 0
                ) AS unit_value_pair_eligible,

                (
                    has_both_mirrors
                    AND export_value_fob > 0
                    AND import_value_cif > 0
                    AND export_weight_kg > 0
                    AND import_weight_kg > 0
                    AND weight_concordance
                        >= {minimum_weight_concordance}
                ) AS high_weight_concordance,

                CASE
                    WHEN import_direct_cif_value > 0
                     AND import_direct_fob_value > 0
                    THEN
                        import_direct_cif_value
                        / import_direct_fob_value
                END AS direct_cif_fob_factor,

                CASE
                    WHEN unit_value_ratio > 0
                    THEN unit_value_ratio
                END AS mirror_unit_value_factor,

                CASE
                    WHEN raw_value_ratio > 0
                    THEN raw_value_ratio
                END AS mirror_raw_value_factor,

                COALESCE(export_proxy_weight_rows, 0) > 0
                    AS export_uses_weight_proxy,

                COALESCE(import_proxy_weight_rows, 0) > 0
                    AS import_uses_weight_proxy,

                COALESCE(export_estimated_weight_rows, 0) > 0
                    AS export_uses_estimated_weight,

                COALESCE(import_estimated_weight_rows, 0) > 0
                    AS import_uses_estimated_weight

            FROM read_parquet({input_sql})
        ),

        factor_selected AS (
            SELECT
                *,

                CASE
                    WHEN direct_cif_fob_factor > 0
                        THEN direct_cif_fob_factor

                    WHEN high_weight_concordance
                     AND mirror_unit_value_factor > 0
                        THEN mirror_unit_value_factor

                    ELSE NULL
                END AS observed_cif_fob_factor,

                CASE
                    WHEN direct_cif_fob_factor > 0
                        THEN 'direct_import_cif_fob'

                    WHEN high_weight_concordance
                     AND mirror_unit_value_factor > 0
                        THEN 'mirror_unit_value'

                    ELSE NULL
                END AS factor_source

            FROM classified
        ),

        diagnosed AS (
            SELECT
                *,

                CASE
                    WHEN observed_cif_fob_factor > 0
                    THEN LN(observed_cif_fob_factor)
                END AS log_observed_cif_fob_factor,

                observed_cif_fob_factor < 1.0
                    AS factor_below_one,

                observed_cif_fob_factor > 1.5
                    AS factor_above_one_point_five,

                observed_cif_fob_factor > 2.0
                    AS factor_above_two,

                (
                    export_uses_weight_proxy
                    OR import_uses_weight_proxy
                ) AS pair_uses_weight_proxy,

                (
                    export_uses_estimated_weight
                    OR import_uses_estimated_weight
                ) AS pair_uses_estimated_weight

            FROM factor_selected
        )

        SELECT
            *,

            (
                observed_cif_fob_factor IS NOT NULL
                AND observed_cif_fob_factor > 0
            ) AS model_eligible,

            CASE
                WHEN direct_cif_fob_factor > 0
                    THEN 'eligible_direct_factor'

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

                WHEN weight_concordance
                     < {minimum_weight_concordance}
                    THEN 'low_weight_concordance'

                WHEN mirror_unit_value_factor IS NULL
                  OR mirror_unit_value_factor <= 0
                    THEN 'invalid_mirror_unit_value_factor'

                ELSE 'eligible_mirror_factor'
            END AS model_eligibility_reason

        FROM diagnosed
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
                    WHERE value_pair_eligible
                ) AS n_value_pair_eligible,

                COUNT(*) FILTER (
                    WHERE unit_value_pair_eligible
                ) AS n_unit_value_pair_eligible,

                COUNT(*) FILTER (
                    WHERE high_weight_concordance
                ) AS n_high_weight_concordance,

                COUNT(*) FILTER (
                    WHERE direct_cif_fob_factor > 0
                ) AS n_direct_factors,

                COUNT(*) FILTER (
                    WHERE factor_source = 'mirror_unit_value'
                ) AS n_mirror_factors,

                COUNT(*) FILTER (
                    WHERE model_eligible
                ) AS n_model_eligible,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND factor_below_one
                ) AS n_factor_below_one,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND factor_above_one_point_five
                ) AS n_factor_above_one_point_five,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND pair_uses_weight_proxy
                ) AS n_model_pairs_using_proxy_weight,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND pair_uses_estimated_weight
                ) AS n_model_pairs_using_estimated_weight,

                MEDIAN(observed_cif_fob_factor)
                    FILTER (
                        WHERE model_eligible
                    )
                    AS median_observed_factor,

                QUANTILE_CONT(
                    observed_cif_fob_factor,
                    0.05
                ) FILTER (
                    WHERE model_eligible
                ) AS observed_factor_p05,

                QUANTILE_CONT(
                    observed_cif_fob_factor,
                    0.95
                ) FILTER (
                    WHERE model_eligible
                ) AS observed_factor_p95

            FROM read_parquet({output_sql})
            """
        ).fetchone()

        metric_names = [
            "n_rows",
            "n_value_pair_eligible",
            "n_unit_value_pair_eligible",
            "n_high_weight_concordance",
            "n_direct_factors",
            "n_mirror_factors",
            "n_model_eligible",
            "n_factor_below_one",
            "n_factor_above_one_point_five",
            "n_model_pairs_using_proxy_weight",
            "n_model_pairs_using_estimated_weight",
            "median_observed_factor",
            "observed_factor_p05",
            "observed_factor_p95",
        ]

        metrics = dict(
            zip(metric_names, summary_row, strict=True)
        )

        model_eligible = int(metrics["n_model_eligible"] or 0)

        metrics["share_factor_below_one_percent"] = (
            100.0
            * int(metrics["n_factor_below_one"] or 0)
            / model_eligible
            if model_eligible
            else 0.0
        )

        metrics["share_factor_above_one_point_five_percent"] = (
            100.0
            * int(
                metrics["n_factor_above_one_point_five"] or 0
            )
            / model_eligible
            if model_eligible
            else 0.0
        )

        logger.info("CIF/FOB training sample built.")
        logger.info("  Rows: %s", metrics["n_rows"])
        logger.info(
            "  Value-pair eligible: %s",
            metrics["n_value_pair_eligible"],
        )
        logger.info(
            "  Unit-value eligible: %s",
            metrics["n_unit_value_pair_eligible"],
        )
        logger.info(
            "  High weight concordance: %s",
            metrics["n_high_weight_concordance"],
        )
        logger.info(
            "  Direct factors: %s",
            metrics["n_direct_factors"],
        )
        logger.info(
            "  Mirror factors selected: %s",
            metrics["n_mirror_factors"],
        )
        logger.info(
            "  Model eligible: %s",
            metrics["n_model_eligible"],
        )
        logger.info(
            "  Median observed factor: %s",
            metrics["median_observed_factor"],
        )
        logger.info(
            "  Observed factor p05/p95: %s / %s",
            metrics["observed_factor_p05"],
            metrics["observed_factor_p95"],
        )
        logger.info(
            "  Factor below 1: %.2f%%",
            metrics["share_factor_below_one_percent"],
        )
        logger.info(
            "  Factor above 1.5: %.2f%%",
            metrics[
                "share_factor_above_one_point_five_percent"
            ],
        )
        logger.info("  Output: %s", output_path)

        return {
            "status": "completed",
            "output_path": output_path,
            "training_sample_path": output_path,
            **metrics,
        }

    finally:
        con.close()