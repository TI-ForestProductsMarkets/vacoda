"""
vacoda.cif_fob.training_sample

Build the CIF/FOB factor-candidate sample from canonical bilateral
mirror pairs.

Evidence hierarchy
------------------
For each canonical bilateral group:

1. Prefer importer-reported direct CIF/FOB evidence when:
   - paired direct CIF and FOB values are positive;
   - the direct factor is positive;
   - the paired source rows cover a sufficient share of the canonical
     importer-side primary value.

2. Otherwise use mirror unit-value evidence when:
   - both reporting sides exist;
   - values and weights are positive;
   - reported weights satisfy the configured concordance threshold;
   - the mirror unit-value factor is positive.

3. Preserve factor-quality diagnostic bands without removing outliers.

The resulting table is a transparent candidate sample for later rate
estimation. Final outlier treatment is deliberately deferred.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.cif_fob import CifFobConfig
from vacoda.cif_fob.prepare_input import valid_output


# ======================================================================
# SQL helpers
# ======================================================================


def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


# ======================================================================
# Existing-output inspection
# ======================================================================


def _inspect_existing_training_sample(
    *,
    output_path: Path,
    threads: int,
) -> dict[str, Any]:
    """Inspect and structurally validate an existing training sample."""

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
                ) AS n_rows,

                (
                    SELECT COUNT(*)
                    FROM read_parquet(?)
                    WHERE direct_eligible
                ) AS n_direct_eligible,

                (
                    SELECT COUNT(*)
                    FROM read_parquet(?)
                    WHERE mirror_eligible
                ) AS n_mirror_eligible,

                (
                    SELECT COUNT(*)
                    FROM read_parquet(?)
                    WHERE factor_source = 'direct'
                ) AS n_direct_selected,

                (
                    SELECT COUNT(*)
                    FROM read_parquet(?)
                    WHERE factor_source = 'mirror'
                ) AS n_mirror_selected,

                (
                    SELECT COUNT(*)
                    FROM read_parquet(?)
                    WHERE model_eligible
                ) AS n_model_eligible,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_keys,

                (
                    SELECT MEDIAN(observed_cif_fob_factor)
                    FROM read_parquet(?)
                    WHERE model_eligible
                ) AS median_observed_factor

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
            ],
        ).fetchone()

    finally:
        con.close()

    if summary is None:
        raise RuntimeError(
            f"Could not inspect training-sample output: {output_path}"
        )

    metric_names = [
        "n_rows",
        "n_direct_eligible",
        "n_mirror_eligible",
        "n_direct_selected",
        "n_mirror_selected",
        "n_model_eligible",
        "n_duplicate_keys",
        "median_observed_factor",
    ]

    metrics = dict(
        zip(
            metric_names,
            summary,
            strict=True,
        )
    )

    if metrics["n_duplicate_keys"] > 0:
        raise RuntimeError(
            "Existing CIF/FOB training sample contains "
            f"{metrics['n_duplicate_keys']:,} duplicate bilateral keys."
        )

    if (
        metrics["n_model_eligible"]
        != (
            metrics["n_direct_selected"]
            + metrics["n_mirror_selected"]
        )
    ):
        raise RuntimeError(
            "Existing training-sample source accounting is inconsistent."
        )

    return {
        "status": "reused",
        "audit_passed": True,
        "ready_for_training_features": True,
        "output_path": output_path,
        "training_sample_path": output_path,
        **metrics,
    }


# ======================================================================
# Main builder
# ======================================================================


def build_training_sample(
    *,
    cfg: CifFobConfig,
    input_path: Path,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Build the CIF/FOB factor-candidate parquet.

    The output preserves exactly one row per canonical bilateral key:

        year
        exporter_iso
        importer_iso
        target_cmd
    """

    output_path = cfg.product_path("training_sample")

    if not input_path.exists():
        raise FileNotFoundError(
            f"CIF/FOB mirror-pair parquet not found: {input_path}"
        )

    if not force and valid_output(output_path):
        logger.info(
            "REUSE — CIF/FOB training sample already exists: %s",
            output_path,
        )

        return _inspect_existing_training_sample(
            output_path=output_path,
            threads=cfg.engine.threads,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_sql = _sql_literal(
        input_path
    )
    output_sql = _sql_literal(
        output_path
    )

    training_cfg = cfg.method.training

    minimum_weight_concordance = float(
        training_cfg.minimum_quantity_concordance
    )

    # This can later become an explicit YAML field. Until then,
    # direct evidence must cover at least 90% of importer primary value.
    minimum_direct_value_coverage = float(
        getattr(
            training_cfg,
            "minimum_direct_value_coverage",
            0.90,
        )
    )

    logger.info(
        "Building CIF/FOB factor-candidate training sample."
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
        "  Minimum mirror weight concordance: %.3f",
        minimum_weight_concordance,
    )
    logger.info(
        "  Minimum direct value coverage: %.3f",
        minimum_direct_value_coverage,
    )
    logger.info(
        "  Evidence priority: direct importer factor, then mirror factor"
    )
    logger.info(
        "  Final outlier removal deferred: True"
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        # ==============================================================
        # INPUT CONTRACT AUDIT
        # ==============================================================

        input_audit = con.execute(
            f"""
            WITH duplicate_keys AS (
                SELECT
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    COUNT(*) AS n_rows

                FROM read_parquet({input_sql})

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
                    FROM read_parquet({input_sql})
                ) AS n_input_rows,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_input_keys,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({input_sql})
                    WHERE year IS NULL
                       OR exporter_iso IS NULL
                       OR importer_iso IS NULL
                       OR target_cmd IS NULL
                ) AS n_null_input_keys,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({input_sql})
                    WHERE training_eligible
                ) AS n_step2_training_eligible,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({input_sql})
                    WHERE has_import_direct_factor
                ) AS n_step2_direct_factor_groups

            """
        ).fetchone()

        input_metric_names = [
            "n_input_rows",
            "n_duplicate_input_keys",
            "n_null_input_keys",
            "n_step2_training_eligible",
            "n_step2_direct_factor_groups",
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
                "Mirror-pair input contains no rows."
            )

        if input_metrics["n_duplicate_input_keys"] > 0:
            raise RuntimeError(
                "Mirror-pair input contains duplicate bilateral keys: "
                f"{input_metrics['n_duplicate_input_keys']:,}."
            )

        if input_metrics["n_null_input_keys"] > 0:
            raise RuntimeError(
                "Mirror-pair input contains null bilateral keys: "
                f"{input_metrics['n_null_input_keys']:,}."
            )

        # ==============================================================
        # TRAINING-CANDIDATE QUERY
        # ==============================================================

        query = f"""
        WITH classified AS (
            SELECT
                *,

                -- --------------------------------------------------
                -- General mirror-data eligibility levels
                -- --------------------------------------------------

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
                    AND unit_value_ratio > 0
                ) AS unit_value_pair_eligible,

                (
                    has_both_mirrors
                    AND export_value_fob > 0
                    AND import_value_cif > 0
                    AND export_weight_kg > 0
                    AND import_weight_kg > 0
                    AND unit_value_ratio > 0
                    AND weight_concordance
                        >= {minimum_weight_concordance}
                ) AS high_weight_concordance,

                -- --------------------------------------------------
                -- Direct evidence
                --
                -- These values were already paired source-row-wise
                -- during Step 1 and attached to the importer side
                -- during Step 2.
                -- --------------------------------------------------

                CASE
                    WHEN import_direct_cif_value > 0
                     AND import_direct_fob_value > 0
                     AND import_direct_cif_fob_factor > 0
                        THEN import_direct_cif_fob_factor
                END AS direct_cif_fob_factor,

                (
                    import_direct_cif_value > 0
                    AND import_direct_fob_value > 0
                    AND import_direct_cif_fob_factor > 0
                    AND COALESCE(
                        import_direct_value_coverage,
                        0.0
                    ) >= {minimum_direct_value_coverage}
                ) AS direct_eligible,

                -- --------------------------------------------------
                -- Mirror evidence
                -- --------------------------------------------------

                CASE
                    WHEN unit_value_ratio > 0
                        THEN unit_value_ratio
                END AS mirror_unit_value_factor,

                CASE
                    WHEN raw_value_ratio > 0
                        THEN raw_value_ratio
                END AS mirror_raw_value_factor,

                (
                    training_eligible
                    AND unit_value_ratio > 0
                    AND weight_concordance
                        >= {minimum_weight_concordance}
                ) AS mirror_eligible,

                -- --------------------------------------------------
                -- Weight-quality metadata
                -- --------------------------------------------------

                COALESCE(
                    export_weight_is_proxy,
                    FALSE
                ) AS export_uses_weight_proxy,

                COALESCE(
                    import_weight_is_proxy,
                    FALSE
                ) AS import_uses_weight_proxy,

                COALESCE(
                    export_weight_estimated,
                    FALSE
                ) AS export_uses_estimated_weight,

                COALESCE(
                    import_weight_estimated,
                    FALSE
                ) AS import_uses_estimated_weight

            FROM read_parquet({input_sql})
        ),

        factor_selected AS (
            SELECT
                *,

                -- Direct evidence has priority. A group with both
                -- direct and mirror evidence contributes only once.
                CASE
                    WHEN direct_eligible
                        THEN direct_cif_fob_factor

                    WHEN mirror_eligible
                        THEN mirror_unit_value_factor

                    ELSE NULL
                END AS observed_cif_fob_factor,

                CASE
                    WHEN direct_eligible
                        THEN 'direct'

                    WHEN mirror_eligible
                        THEN 'mirror'

                    ELSE NULL
                END AS factor_source,

                (
                    direct_eligible
                    AND mirror_eligible
                ) AS has_both_factor_sources,

                CASE
                    WHEN direct_eligible
                     AND mirror_eligible
                     AND direct_cif_fob_factor > 0
                        THEN (
                            mirror_unit_value_factor
                            / direct_cif_fob_factor
                        )
                END AS mirror_to_direct_factor_ratio

            FROM classified
        ),

        diagnosed AS (
            SELECT
                *,

                CASE
                    WHEN observed_cif_fob_factor > 0
                        THEN LN(
                            observed_cif_fob_factor
                        )
                END AS log_observed_cif_fob_factor,

                (
                    observed_cif_fob_factor < 1.0
                ) AS factor_below_one,

                (
                    observed_cif_fob_factor > 1.5
                ) AS factor_above_one_point_five,

                (
                    observed_cif_fob_factor > 2.0
                ) AS factor_above_two,

                (
                    observed_cif_fob_factor > 3.0
                ) AS factor_above_three,

                CASE
                    WHEN observed_cif_fob_factor IS NULL
                        THEN 'missing'

                    WHEN observed_cif_fob_factor < 1.0
                        THEN 'below_one'

                    WHEN observed_cif_fob_factor <= 1.5
                        THEN 'regular_1_to_1_5'

                    WHEN observed_cif_fob_factor <= 3.0
                        THEN 'high_1_5_to_3'

                    ELSE 'extreme_above_3'
                END AS factor_quality_band,

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
                AND factor_source IS NOT NULL
            ) AS model_eligible,

            CASE
                WHEN direct_eligible
                    THEN 'eligible_direct_factor'

                WHEN mirror_eligible
                    THEN 'eligible_mirror_factor'

                WHEN has_import_direct_factor
                 AND (
                     import_direct_value_coverage IS NULL
                     OR import_direct_value_coverage
                        < {minimum_direct_value_coverage}
                 )
                    THEN 'insufficient_direct_value_coverage'

                WHEN has_import_direct_factor
                 AND (
                     import_direct_cif_value IS NULL
                     OR import_direct_cif_value <= 0
                     OR import_direct_fob_value IS NULL
                     OR import_direct_fob_value <= 0
                     OR import_direct_cif_fob_factor IS NULL
                     OR import_direct_cif_fob_factor <= 0
                 )
                    THEN 'invalid_direct_factor'

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

                WHEN weight_concordance IS NULL
                    THEN 'missing_weight_concordance'

                WHEN weight_concordance
                     < {minimum_weight_concordance}
                    THEN 'low_weight_concordance'

                WHEN mirror_unit_value_factor IS NULL
                  OR mirror_unit_value_factor <= 0
                    THEN 'invalid_mirror_unit_value_factor'

                ELSE 'no_eligible_factor'
            END AS model_eligibility_reason

        FROM diagnosed
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
        # OUTPUT SUMMARY
        # ==============================================================

        summary_row = con.execute(
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
                ) AS n_rows,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_output_keys,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE value_pair_eligible
                ) AS n_value_pair_eligible,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE unit_value_pair_eligible
                ) AS n_unit_value_pair_eligible,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE high_weight_concordance
                ) AS n_high_weight_concordance,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE direct_eligible
                ) AS n_direct_eligible,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE mirror_eligible
                ) AS n_mirror_eligible,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE has_both_factor_sources
                ) AS n_with_both_factor_sources,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE factor_source = 'direct'
                ) AS n_direct_factors,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE factor_source = 'mirror'
                ) AS n_mirror_factors,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE model_eligible
                ) AS n_model_eligible,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE model_eligible
                      AND factor_below_one
                ) AS n_factor_below_one,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE model_eligible
                      AND factor_above_one_point_five
                ) AS n_factor_above_one_point_five,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE model_eligible
                      AND factor_above_three
                ) AS n_factor_above_three,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE model_eligible
                      AND pair_uses_weight_proxy
                ) AS n_model_pairs_using_proxy_weight,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE model_eligible
                      AND pair_uses_estimated_weight
                ) AS n_model_pairs_using_estimated_weight,

                (
                    SELECT MEDIAN(observed_cif_fob_factor)
                    FROM read_parquet({output_sql})
                    WHERE model_eligible
                ) AS median_observed_factor,

                (
                    SELECT QUANTILE_CONT(
                        observed_cif_fob_factor,
                        0.05
                    )
                    FROM read_parquet({output_sql})
                    WHERE model_eligible
                ) AS observed_factor_p05,

                (
                    SELECT QUANTILE_CONT(
                        observed_cif_fob_factor,
                        0.95
                    )
                    FROM read_parquet({output_sql})
                    WHERE model_eligible
                ) AS observed_factor_p95,

                (
                    SELECT MEDIAN(
                        mirror_to_direct_factor_ratio
                    )
                    FROM read_parquet({output_sql})
                    WHERE has_both_factor_sources
                      AND mirror_to_direct_factor_ratio > 0
                ) AS median_mirror_to_direct_ratio

            """
        ).fetchone()

        metric_names = [
            "n_rows",
            "n_duplicate_output_keys",
            "n_value_pair_eligible",
            "n_unit_value_pair_eligible",
            "n_high_weight_concordance",
            "n_direct_eligible",
            "n_mirror_eligible",
            "n_with_both_factor_sources",
            "n_direct_factors",
            "n_mirror_factors",
            "n_model_eligible",
            "n_factor_below_one",
            "n_factor_above_one_point_five",
            "n_factor_above_three",
            "n_model_pairs_using_proxy_weight",
            "n_model_pairs_using_estimated_weight",
            "median_observed_factor",
            "observed_factor_p05",
            "observed_factor_p95",
            "median_mirror_to_direct_ratio",
        ]

        output_metrics = dict(
            zip(
                metric_names,
                summary_row,
                strict=True,
            )
        )

        metrics = {
            **input_metrics,
            **output_metrics,
        }

        # ==============================================================
        # DERIVED METRICS
        # ==============================================================

        model_eligible = int(
            metrics["n_model_eligible"]
            or 0
        )

        metrics[
            "share_factor_below_one_percent"
        ] = (
            100.0
            * int(
                metrics["n_factor_below_one"]
                or 0
            )
            / model_eligible
            if model_eligible
            else 0.0
        )

        metrics[
            "share_factor_above_one_point_five_percent"
        ] = (
            100.0
            * int(
                metrics[
                    "n_factor_above_one_point_five"
                ]
                or 0
            )
            / model_eligible
            if model_eligible
            else 0.0
        )

        metrics[
            "share_factor_above_three_percent"
        ] = (
            100.0
            * int(
                metrics["n_factor_above_three"]
                or 0
            )
            / model_eligible
            if model_eligible
            else 0.0
        )

        # ==============================================================
        # HARD STOP CONDITIONS
        # ==============================================================

        if (
            metrics["n_rows"]
            != metrics["n_input_rows"]
        ):
            raise RuntimeError(
                "Training-sample output did not preserve every "
                "mirror-pair row exactly once: "
                f"output={metrics['n_rows']:,}, "
                f"input={metrics['n_input_rows']:,}."
            )

        if metrics["n_duplicate_output_keys"] > 0:
            raise RuntimeError(
                "Training-sample output contains duplicate bilateral "
                f"keys: {metrics['n_duplicate_output_keys']:,}."
            )

        if (
            metrics["n_high_weight_concordance"]
            != metrics["n_step2_training_eligible"]
        ):
            raise RuntimeError(
                "Step 3 high-weight-concordance count does not match "
                "Step 2 training-eligible count: "
                f"step3={metrics['n_high_weight_concordance']:,}, "
                f"step2={metrics['n_step2_training_eligible']:,}."
            )

        if (
            metrics["n_model_eligible"]
            != (
                metrics["n_direct_factors"]
                + metrics["n_mirror_factors"]
            )
        ):
            raise RuntimeError(
                "Selected-factor accounting is inconsistent: "
                f"model={metrics['n_model_eligible']:,}, "
                f"direct={metrics['n_direct_factors']:,}, "
                f"mirror={metrics['n_mirror_factors']:,}."
            )

        invalid_source_rows = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE factor_source = 'direct'
                      AND NOT direct_eligible
                ) AS invalid_direct_selection,

                COUNT(*) FILTER (
                    WHERE factor_source = 'mirror'
                      AND (
                          NOT mirror_eligible
                          OR direct_eligible
                      )
                ) AS invalid_mirror_selection,

                COUNT(*) FILTER (
                    WHERE model_eligible
                      AND (
                          observed_cif_fob_factor IS NULL
                          OR observed_cif_fob_factor <= 0
                          OR factor_source IS NULL
                      )
                ) AS invalid_model_rows,

                COUNT(*) FILTER (
                    WHERE NOT model_eligible
                      AND factor_source IS NOT NULL
                ) AS source_on_ineligible_rows

            FROM read_parquet({output_sql})
            """
        ).fetchone()

        (
            invalid_direct_selection,
            invalid_mirror_selection,
            invalid_model_rows,
            source_on_ineligible_rows,
        ) = invalid_source_rows

        if invalid_direct_selection > 0:
            raise RuntimeError(
                "Training sample selected ineligible direct factors: "
                f"{invalid_direct_selection:,}."
            )

        if invalid_mirror_selection > 0:
            raise RuntimeError(
                "Training sample selected invalid or double-counted "
                f"mirror factors: {invalid_mirror_selection:,}."
            )

        if invalid_model_rows > 0:
            raise RuntimeError(
                "Training sample contains invalid model-eligible rows: "
                f"{invalid_model_rows:,}."
            )

        if source_on_ineligible_rows > 0:
            raise RuntimeError(
                "Training sample assigned factor sources to ineligible "
                f"rows: {source_on_ineligible_rows:,}."
            )

        audit_passed = True
        ready_for_training_features = (
            metrics["n_model_eligible"] > 0
        )

        # ==============================================================
        # LOG SUMMARY
        # ==============================================================

        logger.info(
            "CIF/FOB factor-candidate training sample built."
        )
        logger.info(
            "  Rows: %s",
            metrics["n_rows"],
        )
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
            "  Direct eligible: %s",
            metrics["n_direct_eligible"],
        )
        logger.info(
            "  Mirror eligible: %s",
            metrics["n_mirror_eligible"],
        )
        logger.info(
            "  Groups with both factor sources: %s",
            metrics["n_with_both_factor_sources"],
        )
        logger.info(
            "  Direct factors selected: %s",
            metrics["n_direct_factors"],
        )
        logger.info(
            "  Mirror factors selected: %s",
            metrics["n_mirror_factors"],
        )
        logger.info(
            "  Model-candidate rows: %s",
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
            "  Median mirror/direct ratio in overlap: %s",
            metrics["median_mirror_to_direct_ratio"],
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
        logger.info(
            "  Factor above 3: %.2f%%",
            metrics["share_factor_above_three_percent"],
        )
        logger.info(
            "  Candidate rows using proxy weight: %s",
            metrics["n_model_pairs_using_proxy_weight"],
        )
        logger.info(
            "  Candidate rows using estimated weight: %s",
            metrics["n_model_pairs_using_estimated_weight"],
        )
        logger.info(
            "  Step 3 audit passed: %s",
            audit_passed,
        )
        logger.info(
            "  Ready for rate estimation: %s",
            ready_for_training_features,
        )
        logger.info(
            "  Output: %s",
            output_path,
        )

        return {
            "status": "completed",
            "audit_passed": audit_passed,
            "ready_for_training_features": (
                ready_for_training_features
            ),
            "output_path": output_path,
            "training_sample_path": output_path,
            **metrics,
        }

    finally:
        con.close()