"""
vacoda.trade_sync.reliability_evidence

Build observation-level evidence for Trade Sync country reliability.

Method
------
The reliability methodology distinguishes three included cases:

1. Both exporter FOB and importer FOB-equivalent are positive:
       deviation = abs(export - import) / max(export, import)

2. Exporter FOB is positive, importer reports other imports in the same
   year, but reports no value for this bilateral product:
       importer value is treated as an implicit zero;
       deviation = 1.0

3. Importer FOB-equivalent is positive, exporter reports other exports
   in the same year, but reports no value for this bilateral product:
       exporter value is treated as an implicit zero;
       deviation = 1.0

A missing report is distinguished from an implicit zero by checking
whether the country reported any other trade in the same direction and
year.

The output contains only observations that contribute to reliability
estimation. All other bilateral rows remain preserved in prepared_input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.trade_sync import TradeSyncConfig
from vacoda.utils.display import get_symbols
from vacoda.pipeline.artifact_manifest import (
    file_content_sha256,
    file_stat_fingerprint,
    manifest_path_for,
    semantic_fingerprint,
    validate_manifest,
    write_manifest,
)
from vacoda.pipeline.checkpoint import validate_local_checkpoint, file_fingerprint, write_local_checkpoint, semantic_config_fingerprint


# ======================================================================
# Small helpers
# ======================================================================


def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def valid_output(path: Path) -> bool:
    """Return True when an existing output appears usable."""
    return path.is_file() and path.stat().st_size > 0


# ======================================================================
# Main reliability-evidence function
# ======================================================================


def build_trade_sync_reliability_evidence(
    *,
    cfg: TradeSyncConfig,
    input_path: Path,
    run_id: str,
    force: bool = False,
    logger) -> dict[str, Any]:
    """
    Build observation-level evidence for country reliability scores.

    Parameters
    ----------
    cfg:
        Parsed Trade Sync configuration.
    input_path:
        Prepared Trade Sync parquet containing the full bilateral
        population.
    run_id:
        Unique identifier for the current workflow run.
    force:
        Rebuild an existing product when True.
    logger:
        Configured Trade Sync logger.
    """

    input_path = Path(input_path)
    output_path = cfg.product_path("reliability_evidence")
    
    if not input_path.exists():
        raise FileNotFoundError(
            f"Prepared Trade Sync input not found: {input_path}"
        )

    checkpoint_path = (
        cfg.output_dir()
        / "checkpoints"
        / "reliability_evidence.json"
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
        logger.info(
            "REUSE — Trade Sync reliability evidence "
            "local checkpoint matches: %s",
            output_path,
        )

        return {
            "status": "reused",
            "output_path": output_path,
            "reliability_evidence_path": output_path,
            "audit_passed": True,
            "ready_for_country_reliability": True,
        }

    logger.info(
        "REBUILD — Trade Sync reliability evidence: %s",
        "forced" if force else reason,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_sql = _sql_literal(input_path)
    output_sql = _sql_literal(output_path)

    info = get_symbols("info")
    ok = get_symbols("check")

    logger.info(
        "%s Building Trade Sync country-reliability evidence.",
        info,
    )
    logger.info("  Input: %s", input_path)
    logger.info("  Output: %s", output_path)
    logger.info(
        "  Reliability grain: "
        "year × exporter_iso × importer_iso × target_cmd"
    )
    logger.info(
        "  Positive-positive deviation: "
        "absolute difference divided by larger value"
    )
    logger.info(
        "  Positive–implicit-zero deviation: 1.0"
    )
    logger.info(
        "  Missing partner reports excluded: True"
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        # ==============================================================
        # INPUT SCHEMA VALIDATION
        # ==============================================================

        available_columns = {
            str(row[0])
            for row in con.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_parquet({input_sql})
                """
            ).fetchall()
        }

        required_columns = {
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",
            "export_value_fob",
            "import_value_cif",
            "import_value_fob_corrected",
            "trade_sync_has_export_fob",
            "trade_sync_has_import_fob_equivalent",
            "trade_sync_has_both_reports",
        }

        missing_columns = sorted(
            required_columns - available_columns
        )

        if missing_columns:
            raise RuntimeError(
                "Prepared Trade Sync input is missing required columns: "
                + ", ".join(missing_columns)
            )

        # ==============================================================
        # SHARED EVIDENCE QUERY
        #
        # Reporter activity is defined separately by year and direction.
        #
        # Export activity:
        #   country reported at least one usable exporter FOB value.
        #
        # Import activity:
        #   country reported at least one usable original importer CIF
        #   value. Corrected FOB is deliberately not used here because
        #   failure to estimate a CIF/FOB factor is not non-reporting.
        # ==============================================================

        classified_query = f"""
        WITH source AS (
            SELECT *
            FROM read_parquet({input_sql})
        ),

        export_activity AS (
            SELECT
                year,
                exporter_iso AS country_iso,
                COUNT(*) AS n_reported_export_rows,
                SUM(export_value_fob)
                    FILTER (
                        WHERE export_value_fob IS NOT NULL
                          AND isfinite(export_value_fob)
                          AND export_value_fob > 0
                    ) AS reported_export_value

            FROM source

            WHERE export_value_fob IS NOT NULL
              AND isfinite(export_value_fob)
              AND export_value_fob > 0

            GROUP BY
                year,
                exporter_iso
        ),

        import_activity AS (
            SELECT
                year,
                importer_iso AS country_iso,
                COUNT(*) AS n_reported_import_rows,
                SUM(import_value_cif)
                    FILTER (
                        WHERE import_value_cif IS NOT NULL
                          AND isfinite(import_value_cif)
                          AND import_value_cif > 0
                    ) AS reported_import_cif_value

            FROM source

            WHERE import_value_cif IS NOT NULL
              AND isfinite(import_value_cif)
              AND import_value_cif > 0

            GROUP BY
                year,
                importer_iso
        ),

        enriched AS (
            SELECT
                source.*,

                export_activity.country_iso IS NOT NULL
                    AS exporter_reported_exports_that_year,

                import_activity.country_iso IS NOT NULL
                    AS importer_reported_imports_that_year,

                COALESCE(
                    export_activity.n_reported_export_rows,
                    0
                ) AS exporter_reported_export_rows_that_year,

                COALESCE(
                    import_activity.n_reported_import_rows,
                    0
                ) AS importer_reported_import_rows_that_year

            FROM source

            LEFT JOIN export_activity
                ON source.year = export_activity.year
               AND source.exporter_iso
                   = export_activity.country_iso

            LEFT JOIN import_activity
                ON source.year = import_activity.year
               AND source.importer_iso
                   = import_activity.country_iso
        ),

        classified AS (
            SELECT
                *,

                CASE
                    -- ------------------------------------------
                    -- Both reports are positive and comparable
                    -- ------------------------------------------
                    WHEN trade_sync_has_export_fob
                     AND trade_sync_has_import_fob_equivalent
                        THEN 'both_positive'

                    -- ------------------------------------------
                    -- Export exists; importer is an implicit zero
                    --
                    -- The original importer CIF field must be
                    -- absent. A positive CIF value without a
                    -- corrected FOB value is a comparability
                    -- failure, not an implicit zero.
                    -- ------------------------------------------
                    WHEN trade_sync_has_export_fob
                     AND import_value_cif IS NULL
                     AND importer_reported_imports_that_year
                        THEN 'export_positive_import_implicit_zero'

                    -- ------------------------------------------
                    -- Import exists; exporter is an implicit zero
                    --
                    -- The exporter FOB field must be absent.
                    -- Non-null invalid values are not silently
                    -- reinterpreted as implicit zeros.
                    -- ------------------------------------------
                    WHEN trade_sync_has_import_fob_equivalent
                     AND export_value_fob IS NULL
                     AND exporter_reported_exports_that_year
                        THEN 'import_positive_export_implicit_zero'

                    ELSE NULL
                END AS reliability_evidence_case

            FROM enriched
        ),

        evidence AS (
            SELECT
                *,

                CASE
                    WHEN reliability_evidence_case = 'both_positive'
                        THEN export_value_fob

                    WHEN reliability_evidence_case
                         = 'export_positive_import_implicit_zero'
                        THEN export_value_fob

                    WHEN reliability_evidence_case
                         = 'import_positive_export_implicit_zero'
                        THEN 0.0

                    ELSE NULL
                END AS reliability_export_value_fob,

                CASE
                    WHEN reliability_evidence_case = 'both_positive'
                        THEN import_value_fob_corrected

                    WHEN reliability_evidence_case
                         = 'export_positive_import_implicit_zero'
                        THEN 0.0

                    WHEN reliability_evidence_case
                         = 'import_positive_export_implicit_zero'
                        THEN import_value_fob_corrected

                    ELSE NULL
                END AS reliability_import_value_fob_equivalent,

                CASE
                    WHEN reliability_evidence_case = 'both_positive'
                        THEN (
                            ABS(
                                export_value_fob
                                - import_value_fob_corrected
                            )
                            /
                            GREATEST(
                                export_value_fob,
                                import_value_fob_corrected
                            )
                        )

                    WHEN reliability_evidence_case IN (
                        'export_positive_import_implicit_zero',
                        'import_positive_export_implicit_zero'
                    )
                        THEN 1.0

                    ELSE NULL
                END AS reliability_deviation

            FROM classified
        )

        SELECT *
        FROM evidence
        """

        # ==============================================================
        # INPUT / CLASSIFICATION AUDIT
        # ==============================================================

        classification_audit_row = con.execute(
            f"""
            WITH classified_population AS (
                {classified_query}
            )

            SELECT
                COUNT(*) AS n_prepared_rows,

                COUNT(*) FILTER (
                    WHERE reliability_evidence_case
                        = 'both_positive'
                ) AS n_both_positive,

                COUNT(*) FILTER (
                    WHERE reliability_evidence_case
                        = 'export_positive_import_implicit_zero'
                ) AS n_export_positive_import_implicit_zero,

                COUNT(*) FILTER (
                    WHERE reliability_evidence_case
                        = 'import_positive_export_implicit_zero'
                ) AS n_import_positive_export_implicit_zero,

                COUNT(*) FILTER (
                    WHERE reliability_evidence_case IS NOT NULL
                ) AS n_included_evidence_rows,

                COUNT(*) FILTER (
                    WHERE reliability_evidence_case IS NULL
                ) AS n_excluded_rows,

                -- Export exists but importer did not report any
                -- imports in the year: genuinely missing reporter.
                COUNT(*) FILTER (
                    WHERE trade_sync_has_export_fob
                      AND import_value_cif IS NULL
                      AND NOT importer_reported_imports_that_year
                ) AS n_export_positive_import_report_missing,

                -- Import exists but exporter did not report any
                -- exports in the year: genuinely missing reporter.
                COUNT(*) FILTER (
                    WHERE trade_sync_has_import_fob_equivalent
                      AND export_value_fob IS NULL
                      AND NOT exporter_reported_exports_that_year
                ) AS n_import_positive_export_report_missing,

                -- Importer reported a positive CIF value but no
                -- corrected FOB-equivalent is available.
                COUNT(*) FILTER (
                    WHERE trade_sync_has_export_fob
                      AND import_value_cif IS NOT NULL
                      AND isfinite(import_value_cif)
                      AND import_value_cif > 0
                      AND NOT trade_sync_has_import_fob_equivalent
                ) AS n_import_reported_but_not_fob_comparable,

                COUNT(DISTINCT (
                    CAST(year AS VARCHAR)
                    || '|'
                    || exporter_iso
                )) FILTER (
                    WHERE exporter_reported_exports_that_year
                ) AS n_exporter_year_activity_groups,

                COUNT(DISTINCT (
                    CAST(year AS VARCHAR)
                    || '|'
                    || importer_iso
                )) FILTER (
                    WHERE importer_reported_imports_that_year
                ) AS n_importer_year_activity_groups

            FROM classified_population
            """
        ).fetchone()

        if classification_audit_row is None:
            raise RuntimeError(
                "Could not inspect reliability-evidence population."
            )

        (
            n_prepared_rows,
            n_both_positive,
            n_export_positive_import_implicit_zero,
            n_import_positive_export_implicit_zero,
            n_included_evidence_rows,
            n_excluded_rows,
            n_export_positive_import_report_missing,
            n_import_positive_export_report_missing,
            n_import_reported_but_not_fob_comparable,
            n_exporter_year_activity_groups,
            n_importer_year_activity_groups,
        ) = classification_audit_row

        # ==============================================================
        # WRITE INCLUDED RELIABILITY EVIDENCE
        # ==============================================================

        con.execute(
            f"""
            COPY (
                WITH classified_population AS (
                    {classified_query}
                )

                SELECT *
                FROM classified_population

                WHERE reliability_evidence_case IS NOT NULL
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
            WITH output AS (
                SELECT *
                FROM read_parquet({output_sql})
            ),

            duplicate_keys AS (
                SELECT
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    COUNT(*) AS n_rows

                FROM output

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
                    FROM output
                ) AS n_output_rows,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_keys,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE year IS NULL
                       OR exporter_iso IS NULL
                       OR importer_iso IS NULL
                       OR target_cmd IS NULL
                ) AS n_missing_key_rows,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE reliability_evidence_case IS NULL
                       OR reliability_evidence_case NOT IN (
                           'both_positive',
                           'export_positive_import_implicit_zero',
                           'import_positive_export_implicit_zero'
                       )
                ) AS n_invalid_cases,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE reliability_deviation IS NULL
                       OR NOT isfinite(reliability_deviation)
                       OR reliability_deviation < 0
                       OR reliability_deviation > 1.000000001
                ) AS n_invalid_deviations,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE reliability_evidence_case
                        = 'both_positive'
                      AND (
                            reliability_export_value_fob <= 0
                         OR reliability_import_value_fob_equivalent <= 0
                      )
                ) AS n_invalid_both_positive_values,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE reliability_evidence_case IN (
                        'export_positive_import_implicit_zero',
                        'import_positive_export_implicit_zero'
                    )
                      AND ABS(
                          reliability_deviation - 1.0
                      ) > 1e-12
                ) AS n_invalid_implicit_zero_deviations,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE reliability_evidence_case
                        = 'export_positive_import_implicit_zero'
                      AND (
                            reliability_export_value_fob <= 0
                         OR reliability_import_value_fob_equivalent <> 0
                      )
                ) AS n_invalid_import_implicit_zero_values,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE reliability_evidence_case
                        = 'import_positive_export_implicit_zero'
                      AND (
                            reliability_export_value_fob <> 0
                         OR reliability_import_value_fob_equivalent <= 0
                      )
                ) AS n_invalid_export_implicit_zero_values,

                (
                    SELECT MIN(reliability_deviation)
                    FROM output
                ) AS minimum_deviation,

                (
                    SELECT MEDIAN(reliability_deviation)
                    FROM output
                ) AS median_deviation,

                (
                    SELECT AVG(reliability_deviation)
                    FROM output
                ) AS mean_deviation,

                (
                    SELECT QUANTILE_CONT(
                        reliability_deviation,
                        0.90
                    )
                    FROM output
                ) AS p90_deviation,

                (
                    SELECT MAX(reliability_deviation)
                    FROM output
                ) AS maximum_deviation

            """
        ).fetchone()

        if output_audit_row is None:
            raise RuntimeError(
                "Could not inspect reliability-evidence output."
            )

        (
            n_output_rows,
            n_duplicate_keys,
            n_missing_key_rows,
            n_invalid_cases,
            n_invalid_deviations,
            n_invalid_both_positive_values,
            n_invalid_implicit_zero_deviations,
            n_invalid_import_implicit_zero_values,
            n_invalid_export_implicit_zero_values,
            minimum_deviation,
            median_deviation,
            mean_deviation,
            p90_deviation,
            maximum_deviation,
        ) = output_audit_row

        included_share_percent = (
            100.0
            * int(n_output_rows)
            / int(n_prepared_rows)
            if n_prepared_rows
            else 0.0
        )

        implicit_zero_rows = (
            int(n_export_positive_import_implicit_zero)
            + int(n_import_positive_export_implicit_zero)
        )

        implicit_zero_share_percent = (
            100.0
            * implicit_zero_rows
            / int(n_output_rows)
            if n_output_rows
            else 0.0
        )

        # ==============================================================
        # HARD AUDIT CONDITIONS
        # ==============================================================

        if int(n_prepared_rows) == 0:
            raise RuntimeError(
                "Prepared Trade Sync input contains no rows."
            )

        if (
            int(n_included_evidence_rows)
            + int(n_excluded_rows)
            != int(n_prepared_rows)
        ):
            raise RuntimeError(
                "Reliability evidence inclusion and exclusion counts "
                "do not partition the prepared population."
            )

        if int(n_output_rows) != int(n_included_evidence_rows):
            raise RuntimeError(
                "Reliability-evidence output row count does not match "
                "the included evidence population: "
                f"output={int(n_output_rows):,}, "
                f"expected={int(n_included_evidence_rows):,}."
            )

        if int(n_output_rows) == 0:
            raise RuntimeError(
                "Reliability-evidence preparation produced no rows."
            )

        if int(n_duplicate_keys) > 0:
            raise RuntimeError(
                "Reliability-evidence output contains "
                f"{int(n_duplicate_keys):,} duplicate bilateral keys."
            )

        if int(n_missing_key_rows) > 0:
            raise RuntimeError(
                "Reliability-evidence output contains "
                f"{int(n_missing_key_rows):,} incomplete bilateral keys."
            )

        if int(n_invalid_cases) > 0:
            raise RuntimeError(
                "Reliability-evidence output contains "
                f"{int(n_invalid_cases):,} invalid evidence cases."
            )

        if int(n_invalid_deviations) > 0:
            raise RuntimeError(
                "Reliability-evidence output contains "
                f"{int(n_invalid_deviations):,} deviations outside [0, 1]."
            )

        invalid_value_rows = (
            int(n_invalid_both_positive_values)
            + int(n_invalid_implicit_zero_deviations)
            + int(n_invalid_import_implicit_zero_values)
            + int(n_invalid_export_implicit_zero_values)
        )

        if invalid_value_rows > 0:
            raise RuntimeError(
                "Reliability-evidence output contains internally "
                "inconsistent values or implicit-zero cases: "
                f"{invalid_value_rows:,} rows."
            )

        audit_passed = True
        ready_for_country_reliability = True
        write_local_checkpoint(
            checkpoint_path=checkpoint_path,
            step_name="reliability_evidence",
            run_id=run_id,
            config=config_fp,
            input_fingerprint=input_fp,
            output_path=output_path,
        )

        # ==============================================================
        # LOG RESULTS
        # ==============================================================

        logger.info(
            "%s Trade Sync reliability evidence created.",
            ok,
        )
        logger.info(
            "  Prepared input rows: %s",
            f"{int(n_prepared_rows):,}",
        )
        logger.info(
            "  Included evidence rows: %s",
            f"{int(n_output_rows):,}",
        )
        logger.info(
            "  Evidence coverage: %.4f%%",
            included_share_percent,
        )
        logger.info(
            "  Both reports positive: %s",
            f"{int(n_both_positive):,}",
        )
        logger.info(
            "  Export positive / import implicit zero: %s",
            f"{int(n_export_positive_import_implicit_zero):,}",
        )
        logger.info(
            "  Import positive / export implicit zero: %s",
            f"{int(n_import_positive_export_implicit_zero):,}",
        )
        logger.info(
            "  Total implicit-zero evidence rows: %s",
            f"{implicit_zero_rows:,}",
        )
        logger.info(
            "  Implicit-zero share of evidence: %.4f%%",
            implicit_zero_share_percent,
        )
        logger.info(
            "  Excluded rows: %s",
            f"{int(n_excluded_rows):,}",
        )
        logger.info(
            "    Export positive / importer reporter missing: %s",
            f"{int(n_export_positive_import_report_missing):,}",
        )
        logger.info(
            "    Import positive / exporter reporter missing: %s",
            f"{int(n_import_positive_export_report_missing):,}",
        )
        logger.info(
            "    Import reported but not FOB-comparable: %s",
            f"{int(n_import_reported_but_not_fob_comparable):,}",
        )
        logger.info(
            "  Exporter-year activity groups: %s",
            f"{int(n_exporter_year_activity_groups):,}",
        )
        logger.info(
            "  Importer-year activity groups: %s",
            f"{int(n_importer_year_activity_groups):,}",
        )
        logger.info(
            "  Minimum reliability deviation: %.6f",
            float(minimum_deviation),
        )
        logger.info(
            "  Median reliability deviation: %.6f",
            float(median_deviation),
        )
        logger.info(
            "  Mean reliability deviation: %.6f",
            float(mean_deviation),
        )
        logger.info(
            "  P90 reliability deviation: %.6f",
            float(p90_deviation),
        )
        logger.info(
            "  Maximum reliability deviation: %.6f",
            float(maximum_deviation),
        )
        logger.info(
            "  Duplicate bilateral keys: %s",
            f"{int(n_duplicate_keys):,}",
        )
        logger.info("  Output: %s", output_path)
        logger.info(
            "  Step audit passed: %s",
            audit_passed,
        )
        logger.info(
            "  Ready for country-reliability estimation: %s",
            ready_for_country_reliability,
        )

        return {
            "status": "completed",
            "output_path": output_path,
            "reliability_evidence_path": output_path,
            "audit_passed": audit_passed,
            "ready_for_country_reliability": (
                ready_for_country_reliability
            ),
            "n_prepared_rows": int(n_prepared_rows),
            "n_reliability_evidence_rows": int(n_output_rows),
            "n_excluded_rows": int(n_excluded_rows),
            "n_both_positive": int(n_both_positive),
            "n_export_positive_import_implicit_zero": int(
                n_export_positive_import_implicit_zero
            ),
            "n_import_positive_export_implicit_zero": int(
                n_import_positive_export_implicit_zero
            ),
            "n_implicit_zero_rows": implicit_zero_rows,
            "evidence_coverage_percent": included_share_percent,
            "implicit_zero_share_percent": (
                implicit_zero_share_percent
            ),
            "n_export_positive_import_report_missing": int(n_export_positive_import_report_missing),
            "n_import_positive_export_report_missing": int(n_import_positive_export_report_missing),
            "n_import_reported_but_not_fob_comparable": int(n_import_reported_but_not_fob_comparable),
            "minimum_deviation": float(minimum_deviation),
            "median_deviation": float(median_deviation),
            "mean_deviation": float(mean_deviation),
            "p90_deviation": float(p90_deviation),
            "maximum_deviation": float(maximum_deviation),
        }

    finally:
        con.close()