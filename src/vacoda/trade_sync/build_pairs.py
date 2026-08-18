"""
vacoda.trade_sync.build_pairs

Build the two-sided bilateral comparison population for Trade Sync.

Policy
------
The prepared Trade Sync input contains the complete bilateral population.
This step retains only rows where both of the following are usable:

    export_value_fob
    import_value_fob_corrected

The output remains unique at:

    year
    exporter_iso
    importer_iso
    target_cmd

It also derives initial mirror-report comparison measures used by later
reliability and reconciliation steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.trade_sync import TradeSyncConfig
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
# Main pair-building function
# ======================================================================


def build_trade_sync_pairs(
    *,
    cfg: TradeSyncConfig,
    input_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Build the Trade Sync bilateral-pairs parquet.

    Only rows containing both usable FOB-basis reports are retained.
    """

    input_path = Path(input_path)
    output_path = cfg.product_path("bilateral_pairs")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Prepared Trade Sync input not found: {input_path}"
        )
    
    checkpoint_path = (
        cfg.output_dir()
        / "checkpoints"
        / "bilateral_pairs.json"
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
            "REUSE — Trade Sync bilateral-pairs local checkpoint matches: %s",
            output_path,
        )

        return {
            "status": "reused",
            "output_path": output_path,
            "bilateral_pairs_path": output_path,
            "audit_passed": True,
            "ready_for_reconciliation": True,
        }

    logger.info(
        "REBUILD — Trade Sync bilateral pairs: %s",
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
        "%s Building Trade Sync bilateral comparison pairs.",
        info,
    )
    logger.info("  Input: %s", input_path)
    logger.info("  Output: %s", output_path)
    logger.info(
        "  Canonical grain: "
        "year × exporter_iso × importer_iso × target_cmd"
    )
    logger.info(
        "  Pair eligibility: usable exporter FOB and "
        "usable corrected importer FOB-equivalent"
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
            "import_value_fob_corrected",
            "trade_sync_has_export_fob",
            "trade_sync_has_import_fob_equivalent",
            "trade_sync_has_both_reports",
            "trade_sync_availability",
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
        # INPUT POPULATION AUDIT
        # ==============================================================

        input_audit_row = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_prepared_rows,

                COUNT(*) FILTER (
                    WHERE trade_sync_has_both_reports
                ) AS n_pair_eligible_rows,

                COUNT(*) FILTER (
                    WHERE NOT trade_sync_has_both_reports
                ) AS n_excluded_rows,

                COUNT(*) FILTER (
                    WHERE trade_sync_availability = 'exporter_only'
                ) AS n_exporter_only,

                COUNT(*) FILTER (
                    WHERE trade_sync_availability
                        = 'corrected_importer_only'
                ) AS n_corrected_importer_only,

                COUNT(*) FILTER (
                    WHERE trade_sync_availability = 'neither'
                ) AS n_neither

            FROM read_parquet({input_sql})
            """
        ).fetchone()

        if input_audit_row is None:
            raise RuntimeError(
                "Could not inspect prepared Trade Sync input."
            )

        (
            n_prepared_rows,
            n_pair_eligible_rows,
            n_excluded_rows,
            n_exporter_only,
            n_corrected_importer_only,
            n_neither,
        ) = input_audit_row

        # ==============================================================
        # BUILD BILATERAL PAIRS
        # ==============================================================

        pair_query = f"""
        WITH eligible AS (
            SELECT *
            FROM read_parquet({input_sql})

            WHERE trade_sync_has_both_reports
        ),

        compared AS (
            SELECT
                *,

                import_value_fob_corrected
                /
                NULLIF(
                    export_value_fob,
                    0
                ) AS import_export_ratio,

                LN(
                    import_value_fob_corrected
                )
                -
                LN(
                    export_value_fob
                ) AS log_import_export_ratio,

                ABS(
                    LN(
                        import_value_fob_corrected
                    )
                    -
                    LN(
                        export_value_fob
                    )
                ) AS absolute_log_gap,

                ABS(
                    import_value_fob_corrected
                    -
                    export_value_fob
                ) AS absolute_value_gap,

                ABS(
                    import_value_fob_corrected
                    -
                    export_value_fob
                )
                /
                NULLIF(
                    (
                        import_value_fob_corrected
                        +
                        export_value_fob
                    )
                    / 2.0,
                    0
                ) AS symmetric_relative_gap,

                CASE
                    WHEN import_value_fob_corrected
                         > export_value_fob
                        THEN 'import_above_export'

                    WHEN import_value_fob_corrected
                         < export_value_fob
                        THEN 'import_below_export'

                    ELSE 'approximately_equal'
                END AS mirror_value_direction

            FROM eligible
        )

        SELECT *
        FROM compared
        """

        con.execute(
            f"""
            COPY (
                {pair_query}
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
                ) AS n_output_rows,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_keys,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE year IS NULL
                       OR exporter_iso IS NULL
                       OR importer_iso IS NULL
                       OR target_cmd IS NULL
                ) AS n_missing_key_rows,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE export_value_fob IS NULL
                       OR NOT isfinite(export_value_fob)
                       OR export_value_fob <= 0
                       OR import_value_fob_corrected IS NULL
                       OR NOT isfinite(
                           import_value_fob_corrected
                       )
                       OR import_value_fob_corrected <= 0
                ) AS n_invalid_pair_values,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE import_export_ratio IS NULL
                       OR NOT isfinite(import_export_ratio)
                       OR import_export_ratio <= 0
                ) AS n_invalid_ratios,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE absolute_log_gap IS NULL
                       OR NOT isfinite(absolute_log_gap)
                       OR absolute_log_gap < 0
                ) AS n_invalid_absolute_log_gaps,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE symmetric_relative_gap IS NULL
                       OR NOT isfinite(
                           symmetric_relative_gap
                       )
                       OR symmetric_relative_gap < 0
                       OR symmetric_relative_gap > 2.000000001
                ) AS n_invalid_symmetric_gaps,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE mirror_value_direction
                        = 'import_above_export'
                ) AS n_import_above_export,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE mirror_value_direction
                        = 'import_below_export'
                ) AS n_import_below_export,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE mirror_value_direction
                        = 'approximately_equal'
                ) AS n_approximately_equal,

                (
                    SELECT MEDIAN(
                        import_export_ratio
                    )
                    FROM read_parquet({output_sql})
                ) AS median_import_export_ratio,

                (
                    SELECT MEDIAN(
                        absolute_log_gap
                    )
                    FROM read_parquet({output_sql})
                ) AS median_absolute_log_gap,

                (
                    SELECT QUANTILE_CONT(
                        absolute_log_gap,
                        0.90
                    )
                    FROM read_parquet({output_sql})
                ) AS p90_absolute_log_gap,

                (
                    SELECT MEDIAN(
                        symmetric_relative_gap
                    )
                    FROM read_parquet({output_sql})
                ) AS median_symmetric_relative_gap,

                (
                    SELECT QUANTILE_CONT(
                        symmetric_relative_gap,
                        0.90
                    )
                    FROM read_parquet({output_sql})
                ) AS p90_symmetric_relative_gap

            """
        ).fetchone()

        if output_audit_row is None:
            raise RuntimeError(
                "Could not inspect Trade Sync bilateral-pairs output."
            )

        (
            n_output_rows,
            n_duplicate_keys,
            n_missing_key_rows,
            n_invalid_pair_values,
            n_invalid_ratios,
            n_invalid_absolute_log_gaps,
            n_invalid_symmetric_gaps,
            n_import_above_export,
            n_import_below_export,
            n_approximately_equal,
            median_import_export_ratio,
            median_absolute_log_gap,
            p90_absolute_log_gap,
            median_symmetric_relative_gap,
            p90_symmetric_relative_gap,
        ) = output_audit_row

        direction_total = (
            int(n_import_above_export)
            + int(n_import_below_export)
            + int(n_approximately_equal)
        )

        pair_coverage_percent = (
            100.0
            * int(n_output_rows)
            / int(n_prepared_rows)
            if n_prepared_rows
            else 0.0
        )

        # ==============================================================
        # HARD AUDIT CONDITIONS
        # ==============================================================

        if int(n_prepared_rows) == 0:
            raise RuntimeError(
                "Prepared Trade Sync input contains no rows."
            )

        if int(n_pair_eligible_rows) == 0:
            raise RuntimeError(
                "Prepared Trade Sync input contains no eligible "
                "two-sided bilateral pairs."
            )

        if (
            int(n_pair_eligible_rows)
            + int(n_excluded_rows)
            != int(n_prepared_rows)
        ):
            raise RuntimeError(
                "Pair-eligibility categories do not partition the "
                "prepared Trade Sync population."
            )

        if int(n_output_rows) != int(n_pair_eligible_rows):
            raise RuntimeError(
                "Trade Sync bilateral-pairs output does not match "
                "the eligible prepared-input population: "
                f"output={int(n_output_rows):,}, "
                f"eligible={int(n_pair_eligible_rows):,}."
            )

        if int(n_duplicate_keys) > 0:
            raise RuntimeError(
                "Trade Sync bilateral-pairs output contains "
                f"{int(n_duplicate_keys):,} duplicate bilateral keys."
            )

        if int(n_missing_key_rows) > 0:
            raise RuntimeError(
                "Trade Sync bilateral-pairs output contains "
                f"{int(n_missing_key_rows):,} rows with incomplete keys."
            )

        if int(n_invalid_pair_values) > 0:
            raise RuntimeError(
                "Trade Sync bilateral-pairs output contains "
                f"{int(n_invalid_pair_values):,} invalid paired values."
            )

        if int(n_invalid_ratios) > 0:
            raise RuntimeError(
                "Trade Sync bilateral-pairs output contains "
                f"{int(n_invalid_ratios):,} invalid import/export ratios."
            )

        if int(n_invalid_absolute_log_gaps) > 0:
            raise RuntimeError(
                "Trade Sync bilateral-pairs output contains "
                f"{int(n_invalid_absolute_log_gaps):,} invalid "
                "absolute log gaps."
            )

        if int(n_invalid_symmetric_gaps) > 0:
            raise RuntimeError(
                "Trade Sync bilateral-pairs output contains "
                f"{int(n_invalid_symmetric_gaps):,} invalid symmetric "
                "relative gaps."
            )

        if direction_total != int(n_output_rows):
            raise RuntimeError(
                "Mirror-value direction categories do not partition "
                "the bilateral-pairs output."
            )

        audit_passed = True
        ready_for_reconciliation = True

        write_local_checkpoint(
            checkpoint_path=checkpoint_path,
            step_name="bilateral_pairs",
            run_id=run_id,
            config=config_fp,
            input_fingerprint=input_fp,
            output_path=output_path,
        )

        # ==============================================================
        # LOG RESULTS
        # ==============================================================

        logger.info(
            "%s Trade Sync bilateral pairs created.",
            ok,
        )
        logger.info(
            "  Prepared input rows: %s",
            f"{int(n_prepared_rows):,}",
        )
        logger.info(
            "  Eligible bilateral pairs: %s",
            f"{int(n_pair_eligible_rows):,}",
        )
        logger.info(
            "  Excluded from paired analysis: %s",
            f"{int(n_excluded_rows):,}",
        )
        logger.info(
            "    Exporter only: %s",
            f"{int(n_exporter_only):,}",
        )
        logger.info(
            "    Corrected importer only: %s",
            f"{int(n_corrected_importer_only):,}",
        )
        logger.info(
            "    Neither usable: %s",
            f"{int(n_neither):,}",
        )
        logger.info(
            "  Paired-analysis coverage: %.4f%%",
            pair_coverage_percent,
        )
        logger.info(
            "  Duplicate bilateral keys: %s",
            f"{int(n_duplicate_keys):,}",
        )
        logger.info(
            "  Import above export: %s",
            f"{int(n_import_above_export):,}",
        )
        logger.info(
            "  Import below export: %s",
            f"{int(n_import_below_export):,}",
        )
        logger.info(
            "  Approximately equal: %s",
            f"{int(n_approximately_equal):,}",
        )
        logger.info(
            "  Median import/export ratio: %.6f",
            float(median_import_export_ratio),
        )
        logger.info(
            "  Median absolute log gap: %.6f",
            float(median_absolute_log_gap),
        )
        logger.info(
            "  P90 absolute log gap: %.6f",
            float(p90_absolute_log_gap),
        )
        logger.info(
            "  Median symmetric relative gap: %.6f",
            float(median_symmetric_relative_gap),
        )
        logger.info(
            "  P90 symmetric relative gap: %.6f",
            float(p90_symmetric_relative_gap),
        )
        logger.info("  Output: %s", output_path)
        logger.info(
            "  Step 2 audit passed: %s",
            audit_passed,
        )
        logger.info(
            "  Ready for reconciliation: %s",
            ready_for_reconciliation,
        )

        return {
            "status": "completed",
            "output_path": output_path,
            "bilateral_pairs_path": output_path,
            "audit_passed": audit_passed,
            "ready_for_reconciliation": ready_for_reconciliation,
            "n_prepared_rows": int(n_prepared_rows),
            "n_bilateral_pairs": int(n_output_rows),
            "n_excluded_from_pair_analysis": int(
                n_excluded_rows
            ),
            "n_exporter_only": int(n_exporter_only),
            "n_corrected_importer_only": int(
                n_corrected_importer_only
            ),
            "n_neither": int(n_neither),
            "pair_coverage_percent": pair_coverage_percent,
            "n_import_above_export": int(
                n_import_above_export
            ),
            "n_import_below_export": int(
                n_import_below_export
            ),
            "n_approximately_equal": int(
                n_approximately_equal
            ),
            "median_import_export_ratio": float(
                median_import_export_ratio
            ),
            "median_absolute_log_gap": float(
                median_absolute_log_gap
            ),
            "p90_absolute_log_gap": float(
                p90_absolute_log_gap
            ),
            "median_symmetric_relative_gap": float(
                median_symmetric_relative_gap
            ),
            "p90_symmetric_relative_gap": float(
                p90_symmetric_relative_gap
            ),
        }

    finally:
        con.close()