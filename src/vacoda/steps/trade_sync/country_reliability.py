"""
vacoda.steps.trade_sync.country_reliability

Estimate annual country reliability from bilateral trade deviations.

Method
------
Each reliability-evidence observation contributes to two scores:

1. the export reliability of the exporting country;
2. the import reliability of the importing country.

Scores are calculated separately at:

    year
    country_iso
    reporting_direction

Following Bremer, Perger, and Schumacher, the primary score is the
unweighted arithmetic mean of bilateral deviations across all products
and partner countries.

Because a lower mean deviation indicates greater reliability, this module
also derives:

    reliability_index = 1 - mean_reliability_deviation

The mean deviation is retained for the harmonization equations. The
reliability index is provided as a more intuitive reporting measure, where
higher values indicate greater agreement with mirror reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.steps.trade_sync import TradeSyncConfig
from vacoda.utils.display import get_symbols
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
# Main country-reliability function
# ======================================================================


def estimate_trade_sync_country_reliability(
    *,
    cfg: TradeSyncConfig,
    input_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Estimate annual import and export reliability for each country.

    Parameters
    ----------
    cfg:
        Parsed Trade Sync configuration.
    input_path:
        Observation-level reliability-evidence parquet.
    force:
        Rebuild an existing product when True.
    logger:
        Configured Trade Sync logger.
    """

    input_path = Path(input_path)
    output_path = cfg.product_path("country_reliability")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Trade Sync reliability evidence not found: {input_path}"
        )

    checkpoint_path = (
        cfg.output_dir()
        / "checkpoints"
        / "country_reliability.json"
    )

    config_fp = semantic_config_fingerprint(
        config=cfg,
        excluded_paths=(
            ("engine", "threads"),
        ),
    )

    input_fp = {
        "reliability_evidence": file_fingerprint(
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
            "REUSE — Trade Sync country reliability "
            "local checkpoint matches: %s",
            output_path,
        )

        return {
            "status": "reused",
            "output_path": output_path,
            "country_reliability_path": output_path,
            "audit_passed": True,
            "ready_for_reconciliation": True,
        }

    logger.info(
        "REBUILD — Trade Sync country reliability: %s",
        "forced" if force else reason,
    )

    input_sql = _sql_literal(input_path)
    output_sql = _sql_literal(output_path)

    info = get_symbols("info")
    ok = get_symbols("check")

    logger.info(
        "%s Estimating Trade Sync country reliability.",
        info,
    )
    logger.info("  Input: %s", input_path)
    logger.info("  Output: %s", output_path)
    logger.info(
        "  Reliability grain: "
        "year × country_iso × reporting_direction"
    )
    logger.info(
        "  Commodity-specific reliability scores: False"
    )
    logger.info(
        "  Partner-specific reliability scores: False"
    )
    logger.info(
        "  Primary score: unweighted mean bilateral deviation"
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
            "reliability_evidence_case",
            "reliability_deviation",
        }

        missing_columns = sorted(
            required_columns - available_columns
        )

        if missing_columns:
            raise RuntimeError(
                "Reliability-evidence input is missing required columns: "
                + ", ".join(missing_columns)
            )

        # ==============================================================
        # INPUT AUDIT
        # ==============================================================

        input_audit_row = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_evidence_rows,

                COUNT(*) FILTER (
                    WHERE reliability_evidence_case = 'both_positive'
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
                    WHERE reliability_deviation IS NULL
                       OR NOT isfinite(reliability_deviation)
                       OR reliability_deviation < 0
                       OR reliability_deviation > 1.000000001
                ) AS n_invalid_input_deviations,

                MIN(year) AS minimum_year,
                MAX(year) AS maximum_year

            FROM read_parquet({input_sql})
            """
        ).fetchone()

        if input_audit_row is None:
            raise RuntimeError(
                "Could not inspect Trade Sync reliability evidence."
            )

        (
            n_evidence_rows,
            n_both_positive,
            n_export_positive_import_implicit_zero,
            n_import_positive_export_implicit_zero,
            n_invalid_input_deviations,
            minimum_year,
            maximum_year,
        ) = input_audit_row

        # ==============================================================
        # COUNTRY-DIRECTION EVIDENCE
        #
        # Every bilateral evidence row contributes:
        #
        #   exporter -> export score
        #   importer -> import score
        #
        # The same bilateral deviation is used for both reporters because
        # it measures disagreement between the two complementary reports.
        # ==============================================================

        directional_evidence_query = f"""
        WITH evidence AS (
            SELECT *
            FROM read_parquet({input_sql})
        ),

        directional_evidence AS (
            SELECT
                year,
                exporter_iso AS country_iso,
                'export' AS reporting_direction,
                importer_iso AS partner_iso,
                target_cmd,
                reliability_evidence_case,
                reliability_deviation

            FROM evidence

            UNION ALL

            SELECT
                year,
                importer_iso AS country_iso,
                'import' AS reporting_direction,
                exporter_iso AS partner_iso,
                target_cmd,
                reliability_evidence_case,
                reliability_deviation

            FROM evidence
        )

        SELECT *
        FROM directional_evidence
        """

        # ==============================================================
        # AGGREGATE COUNTRY RELIABILITY
        # ==============================================================

        country_reliability_query = f"""
        WITH directional_evidence AS (
            {directional_evidence_query}
        ),

        aggregated AS (
            SELECT
                year,
                country_iso,
                reporting_direction,

                COUNT(*) AS n_evidence_observations,

                COUNT(DISTINCT partner_iso)
                    AS n_distinct_partners,

                COUNT(DISTINCT target_cmd)
                    AS n_distinct_products,

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
                    WHERE reliability_evidence_case IN (
                        'export_positive_import_implicit_zero',
                        'import_positive_export_implicit_zero'
                    )
                ) AS n_implicit_zero_observations,

                AVG(
                    reliability_deviation
                ) AS mean_reliability_deviation,

                MEDIAN(
                    reliability_deviation
                ) AS median_reliability_deviation,

                QUANTILE_CONT(
                    reliability_deviation,
                    0.10
                ) AS p10_reliability_deviation,

                QUANTILE_CONT(
                    reliability_deviation,
                    0.90
                ) AS p90_reliability_deviation,

                MIN(
                    reliability_deviation
                ) AS minimum_reliability_deviation,

                MAX(
                    reliability_deviation
                ) AS maximum_reliability_deviation,

                AVG(
                    reliability_deviation
                ) FILTER (
                    WHERE reliability_evidence_case
                        = 'both_positive'
                ) AS mean_positive_positive_deviation,

                MEDIAN(
                    reliability_deviation
                ) FILTER (
                    WHERE reliability_evidence_case
                        = 'both_positive'
                ) AS median_positive_positive_deviation

            FROM directional_evidence

            GROUP BY
                year,
                country_iso,
                reporting_direction
        )

        SELECT
            *,

            1.0
            -
            mean_reliability_deviation
                AS reliability_index,

            n_implicit_zero_observations
            /
            NULLIF(
                CAST(n_evidence_observations AS DOUBLE),
                0
            ) AS implicit_zero_share,

            n_both_positive
            /
            NULLIF(
                CAST(n_evidence_observations AS DOUBLE),
                0
            ) AS both_positive_share

        FROM aggregated
        """

        # ==============================================================
        # WRITE OUTPUT
        # ==============================================================

        con.execute(
            f"""
            COPY (
                {country_reliability_query}
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
                    country_iso,
                    reporting_direction,
                    COUNT(*) AS n_rows

                FROM output

                GROUP BY
                    year,
                    country_iso,
                    reporting_direction

                HAVING COUNT(*) > 1
            )

            SELECT
                (
                    SELECT COUNT(*)
                    FROM output
                ) AS n_country_year_direction_scores,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_score_keys,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE year IS NULL
                       OR country_iso IS NULL
                       OR reporting_direction IS NULL
                ) AS n_missing_score_keys,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE reporting_direction NOT IN (
                        'import',
                        'export'
                    )
                ) AS n_invalid_directions,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE n_evidence_observations <= 0
                ) AS n_empty_score_groups,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE mean_reliability_deviation IS NULL
                       OR NOT isfinite(
                           mean_reliability_deviation
                       )
                       OR mean_reliability_deviation < 0
                       OR mean_reliability_deviation
                            > 1.000000001
                ) AS n_invalid_mean_deviations,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE reliability_index IS NULL
                       OR NOT isfinite(reliability_index)
                       OR reliability_index < -0.000000001
                       OR reliability_index > 1.000000001
                ) AS n_invalid_reliability_indices,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE implicit_zero_share IS NULL
                       OR NOT isfinite(implicit_zero_share)
                       OR implicit_zero_share < 0
                       OR implicit_zero_share > 1.000000001
                ) AS n_invalid_implicit_zero_shares,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE both_positive_share IS NULL
                       OR NOT isfinite(both_positive_share)
                       OR both_positive_share < 0
                       OR both_positive_share > 1.000000001
                ) AS n_invalid_both_positive_shares,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE (
                        n_both_positive
                        + n_implicit_zero_observations
                    ) <> n_evidence_observations
                ) AS n_invalid_case_totals,

                (
                    SELECT COALESCE(
                        SUM(n_evidence_observations),
                        0
                    )
                    FROM output
                    WHERE reporting_direction = 'export'
                ) AS n_export_score_observations,

                (
                    SELECT COALESCE(
                        SUM(n_evidence_observations),
                        0
                    )
                    FROM output
                    WHERE reporting_direction = 'import'
                ) AS n_import_score_observations,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE reporting_direction = 'export'
                ) AS n_export_scores,

                (
                    SELECT COUNT(*)
                    FROM output
                    WHERE reporting_direction = 'import'
                ) AS n_import_scores,

                (
                    SELECT COUNT(DISTINCT country_iso)
                    FROM output
                ) AS n_countries,

                (
                    SELECT COUNT(DISTINCT year)
                    FROM output
                ) AS n_years,

                (
                    SELECT MIN(n_evidence_observations)
                    FROM output
                ) AS minimum_evidence_observations,

                (
                    SELECT MEDIAN(n_evidence_observations)
                    FROM output
                ) AS median_evidence_observations,

                (
                    SELECT QUANTILE_CONT(
                        n_evidence_observations,
                        0.90
                    )
                    FROM output
                ) AS p90_evidence_observations,

                (
                    SELECT MAX(n_evidence_observations)
                    FROM output
                ) AS maximum_evidence_observations,

                (
                    SELECT MIN(mean_reliability_deviation)
                    FROM output
                ) AS minimum_mean_deviation,

                (
                    SELECT MEDIAN(mean_reliability_deviation)
                    FROM output
                ) AS median_mean_deviation,

                (
                    SELECT AVG(mean_reliability_deviation)
                    FROM output
                ) AS average_mean_deviation,

                (
                    SELECT MAX(mean_reliability_deviation)
                    FROM output
                ) AS maximum_mean_deviation,

                (
                    SELECT MIN(reliability_index)
                    FROM output
                ) AS minimum_reliability_index,

                (
                    SELECT MEDIAN(reliability_index)
                    FROM output
                ) AS median_reliability_index,

                (
                    SELECT AVG(reliability_index)
                    FROM output
                ) AS average_reliability_index,

                (
                    SELECT MAX(reliability_index)
                    FROM output
                ) AS maximum_reliability_index

            """
        ).fetchone()

        if output_audit_row is None:
            raise RuntimeError(
                "Could not inspect country-reliability output."
            )

        (
            n_country_year_direction_scores,
            n_duplicate_score_keys,
            n_missing_score_keys,
            n_invalid_directions,
            n_empty_score_groups,
            n_invalid_mean_deviations,
            n_invalid_reliability_indices,
            n_invalid_implicit_zero_shares,
            n_invalid_both_positive_shares,
            n_invalid_case_totals,
            n_export_score_observations,
            n_import_score_observations,
            n_export_scores,
            n_import_scores,
            n_countries,
            n_years,
            minimum_evidence_observations,
            median_evidence_observations,
            p90_evidence_observations,
            maximum_evidence_observations,
            minimum_mean_deviation,
            median_mean_deviation,
            average_mean_deviation,
            maximum_mean_deviation,
            minimum_reliability_index,
            median_reliability_index,
            average_reliability_index,
            maximum_reliability_index,
        ) = output_audit_row

        expected_directional_observations = (
            2 * int(n_evidence_rows)
        )

        actual_directional_observations = (
            int(n_export_score_observations)
            + int(n_import_score_observations)
        )

        # ==============================================================
        # HARD AUDIT CONDITIONS
        # ==============================================================

        if int(n_evidence_rows) == 0:
            raise RuntimeError(
                "Reliability-evidence input contains no rows."
            )

        if int(n_invalid_input_deviations) > 0:
            raise RuntimeError(
                "Reliability-evidence input contains "
                f"{int(n_invalid_input_deviations):,} invalid deviations."
            )

        if int(n_country_year_direction_scores) == 0:
            raise RuntimeError(
                "Country-reliability estimation produced no scores."
            )

        if int(n_duplicate_score_keys) > 0:
            raise RuntimeError(
                "Country-reliability output contains "
                f"{int(n_duplicate_score_keys):,} duplicate "
                "country-year-direction keys."
            )


        if int(n_missing_score_keys) > 0:
            raise RuntimeError(
                "Country-reliability output contains "
                f"{int(n_missing_score_keys):,} rows with incomplete "
                "country-year-direction keys."
            )

        if int(n_invalid_directions) > 0:
            raise RuntimeError(
                "Country-reliability output contains "
                f"{int(n_invalid_directions):,} invalid reporting "
                "directions."
            )

        if int(n_empty_score_groups) > 0:
            raise RuntimeError(
                "Country-reliability output contains "
                f"{int(n_empty_score_groups):,} empty score groups."
            )

        if int(n_invalid_mean_deviations) > 0:
            raise RuntimeError(
                "Country-reliability output contains "
                f"{int(n_invalid_mean_deviations):,} invalid mean "
                "deviations."
            )

        if int(n_invalid_reliability_indices) > 0:
            raise RuntimeError(
                "Country-reliability output contains "
                f"{int(n_invalid_reliability_indices):,} invalid "
                "reliability indices."
            )

        if int(n_invalid_implicit_zero_shares) > 0:
            raise RuntimeError(
                "Country-reliability output contains "
                f"{int(n_invalid_implicit_zero_shares):,} invalid "
                "implicit-zero shares."
            )

        if int(n_invalid_both_positive_shares) > 0:
            raise RuntimeError(
                "Country-reliability output contains "
                f"{int(n_invalid_both_positive_shares):,} invalid "
                "both-positive shares."
            )

        if int(n_invalid_case_totals) > 0:
            raise RuntimeError(
                "Country-reliability output contains "
                f"{int(n_invalid_case_totals):,} rows with evidence-case "
                "totals that do not match n_evidence_observations."
            )

        if actual_directional_observations != expected_directional_observations:
            raise RuntimeError(
                "Country-reliability directional evidence accounting "
                "mismatch: "
                f"actual={actual_directional_observations:,}, "
                f"expected={expected_directional_observations:,}."
            )

        audit_passed = True
        ready_for_reconciliation = True

        write_local_checkpoint(
            checkpoint_path=checkpoint_path,
            step_name="country_reliability",
            run_id=run_id,
            config=config_fp,
            input_fingerprint=input_fp,
            output_path=output_path,
        )

        both_positive_share_percent = (
            100.0 * int(n_both_positive) / int(n_evidence_rows)
            if int(n_evidence_rows)
            else 0.0
        )

        implicit_zero_share_percent = (
            100.0
            * (
                int(n_export_positive_import_implicit_zero)
                + int(n_import_positive_export_implicit_zero)
            )
            / int(n_evidence_rows)
            if int(n_evidence_rows)
            else 0.0
        )

        logger.info(
            "%s Trade Sync country reliability created.",
            ok,
        )
        logger.info(
            "  Reliability-evidence rows: %s",
            f"{int(n_evidence_rows):,}",
        )
        logger.info(
            "  Country-year-direction scores: %s",
            f"{int(n_country_year_direction_scores):,}",
        )
        logger.info(
            "  Countries covered: %s",
            f"{int(n_countries):,}",
        )
        logger.info(
            "  Years covered: %s (%s-%s)",
            f"{int(n_years):,}",
            minimum_year,
            maximum_year,
        )
        logger.info(
            "  Directional evidence observations (expected/actual): %s / %s",
            f"{expected_directional_observations:,}",
            f"{actual_directional_observations:,}",
        )
        logger.info(
            "  Input evidence cases: both_positive=%s, "
            "export_positive_import_implicit_zero=%s, "
            "import_positive_export_implicit_zero=%s",
            f"{int(n_both_positive):,}",
            f"{int(n_export_positive_import_implicit_zero):,}",
            f"{int(n_import_positive_export_implicit_zero):,}",
        )
        logger.info(
            "  Input both-positive share: %.4f%%",
            both_positive_share_percent,
        )
        logger.info(
            "  Input implicit-zero share: %.4f%%",
            implicit_zero_share_percent,
        )
        logger.info("  Output: %s", output_path)
        logger.info("  Step audit passed: %s", audit_passed)
        logger.info("  Ready for reconciliation: %s", ready_for_reconciliation,)

        return {
            "status": "completed",
            "output_path": output_path,
            "country_reliability_path": output_path,
            "audit_passed": audit_passed,
            "ready_for_reconciliation": ready_for_reconciliation,
            "n_reliability_evidence_rows": int(n_evidence_rows),
            "n_country_year_direction_scores": int(n_country_year_direction_scores),
            "n_countries": int(n_countries),
            "n_years": int(n_years),
            "minimum_year": int(minimum_year) if minimum_year is not None else None,
            "maximum_year": int(maximum_year) if maximum_year is not None else None,
            "n_both_positive": int(n_both_positive),
            "n_export_positive_import_implicit_zero": int(n_export_positive_import_implicit_zero),
            "n_import_positive_export_implicit_zero": int(n_import_positive_export_implicit_zero),
            "both_positive_share_percent": both_positive_share_percent,
            "implicit_zero_share_percent": implicit_zero_share_percent,
            "minimum_evidence_observations": int(minimum_evidence_observations),
            "median_evidence_observations": int(median_evidence_observations),
            "p90_evidence_observations": int(p90_evidence_observations),
            "maximum_evidence_observations": int(maximum_evidence_observations),
            "minimum_mean_deviation": float(minimum_mean_deviation),
            "median_mean_deviation": float(median_mean_deviation),
            "average_mean_deviation": float(average_mean_deviation),
            "maximum_mean_deviation": float(maximum_mean_deviation),
            "minimum_reliability_index": float(minimum_reliability_index),
            "median_reliability_index": float(median_reliability_index),
            "average_reliability_index": float(average_reliability_index),
            "maximum_reliability_index": float(maximum_reliability_index),
        }

    finally:
        con.close()
