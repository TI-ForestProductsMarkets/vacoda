"""
vacoda.steps.trade_sync.prepare_input

Prepare the CIF/FOB-corrected bilateral trade data for Trade Sync.

Policy
------
The CIF/FOB output is already at the canonical bilateral grain:

    year
    exporter_iso
    importer_iso
    target_cmd

This step therefore performs no further aggregation or matching. It:

1. preserves the full bilateral population;
2. validates the canonical bilateral key;
3. derives explicit exporter/importer usability flags;
4. assigns each row to one mutually exclusive availability category;
5. audits row preservation, key uniqueness, and category completeness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.steps.trade_sync import TradeSyncConfig
from vacoda.utils.display import get_symbols, color_text
from vacoda.pipeline.checkpoint import validate_local_checkpoint, file_fingerprint, write_local_checkpoint, semantic_config_fingerprint
from vacoda.utils.sql import sql_literal

# ======================================================================
# Small helpers
# ======================================================================

def valid_output(path: Path) -> bool:
    """Return True when an existing output appears usable."""
    return path.is_file() and path.stat().st_size > 0


# ======================================================================
# Main preparation function
# ======================================================================


def prepare_trade_sync_input(
    *,
    cfg: TradeSyncConfig,
    input_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Create the canonical prepared Trade Sync input parquet.

    The output preserves exactly one row per:

        year
        exporter_iso
        importer_iso
        target_cmd
    """

    input_path = Path(input_path)
    output_path = cfg.product_path("prepared_input")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Trade Sync source parquet not found: {input_path}"
        )

    ## Local Checkpoint 
    checkpoint_path = (
        cfg.output_dir()
        / "checkpoints"
        / "prepared_input.json"
    )

    config_fp = semantic_config_fingerprint(
        config=cfg,
        excluded_paths=(
            ("engine", "threads"),
        ),
    )

    input_fp = {
        "source": file_fingerprint(
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
            "REUSE — Prepared Trade Sync input local checkpoint matches: %s",
            output_path,
        )

        return {
            "status": "reused",
            "output_path": output_path,
            "prepared_input_path": output_path,
            "audit_passed": True,
            "ready_for_bilateral_pairs": True,
        }

    logger.info(
        "REBUILD — Prepared Trade Sync input: %s",
        "forced" if force else reason,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_sql = sql_literal(input_path)
    output_sql = sql_literal(output_path)

    info = color_text(get_symbols("info"), "red")
    ok = get_symbols("check")

    logger.info(
        "%s Preparing canonical Trade Sync input.",
        info,
    )
    logger.info("  Input: %s", input_path)
    logger.info("  Output: %s", output_path)
    logger.info(
        "  Canonical grain: "
        "year × exporter_iso × importer_iso × target_cmd"
    )
    logger.info(
        "  Preserve full bilateral population: True"
    )
    logger.info(
        "  Perform additional aggregation: False"
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
            "cif_fob_factor_applied_source",

            # Physical quantities propagated from Quantity through CIF/FOB.
            "export_synchronized_quantity_kg",
            "import_synchronized_quantity_kg",
            "export_weight_kg",
            "import_weight_kg",
            "export_synchronized_quantity_source",
            "import_synchronized_quantity_source",
            "export_is_synchronized_quantity_estimated",
            "import_is_synchronized_quantity_estimated",
            "export_weight_is_proxy",
            "import_weight_is_proxy",
        }

        missing_columns = sorted(
            required_columns - available_columns
        )

        if missing_columns:
            raise RuntimeError(
                "Trade Sync input is missing required columns: "
                + ", ".join(missing_columns)
            )

        # ==============================================================
        # INPUT AUDIT
        # ==============================================================

        input_audit_row = con.execute(
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
                ) AS n_input_duplicate_keys,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({input_sql})
                    WHERE year IS NULL
                       OR exporter_iso IS NULL
                       OR importer_iso IS NULL
                       OR target_cmd IS NULL
                ) AS n_input_missing_key_rows,

                (
                    SELECT MIN(year)
                    FROM read_parquet({input_sql})
                ) AS minimum_year,

                (
                    SELECT MAX(year)
                    FROM read_parquet({input_sql})
                ) AS maximum_year
            """
        ).fetchone()

        if input_audit_row is None:
            raise RuntimeError(
                "Could not inspect Trade Sync input."
            )

        (
            n_input_rows,
            n_input_duplicate_keys,
            n_input_missing_key_rows,
            minimum_year,
            maximum_year,
        ) = input_audit_row

        # ==============================================================
        # PREPARE OUTPUT
        # ==============================================================

        preparation_query = f"""
        WITH prepared AS (
            SELECT
                *,

                -- ----------------------------------------------
                -- Exporter-side usable FOB value
                -- ----------------------------------------------

                (
                    export_value_fob IS NOT NULL
                    AND isfinite(export_value_fob)
                    AND export_value_fob > 0
                ) AS trade_sync_has_export_fob,

                -- ----------------------------------------------
                -- Original importer-side usable CIF value
                -- ----------------------------------------------

                (
                    import_value_cif IS NOT NULL
                    AND isfinite(import_value_cif)
                    AND import_value_cif > 0
                ) AS trade_sync_has_import_cif,

                -- ----------------------------------------------
                -- Importer-side usable FOB-equivalent value
                -- ----------------------------------------------

                (
                    import_value_fob_corrected IS NOT NULL
                    AND isfinite(import_value_fob_corrected)
                    AND import_value_fob_corrected > 0
                ) AS trade_sync_has_import_fob_equivalent

            FROM read_parquet({input_sql})
        )

        SELECT
            *,

            (
                trade_sync_has_export_fob
                AND trade_sync_has_import_fob_equivalent
            ) AS trade_sync_has_both_reports,

            (
                trade_sync_has_export_fob
                OR trade_sync_has_import_fob_equivalent
            ) AS trade_sync_has_usable_value,

            CASE
                WHEN trade_sync_has_export_fob
                 AND trade_sync_has_import_fob_equivalent
                    THEN 'both'

                WHEN trade_sync_has_export_fob
                 AND NOT trade_sync_has_import_fob_equivalent
                    THEN 'exporter_only'

                WHEN NOT trade_sync_has_export_fob
                 AND trade_sync_has_import_fob_equivalent
                    THEN 'corrected_importer_only'

                ELSE 'neither'
            END AS trade_sync_availability

        FROM prepared
        """

        con.execute(
            f"""
            COPY (
                {preparation_query}
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
                ) AS n_output_duplicate_keys,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE year IS NULL
                       OR exporter_iso IS NULL
                       OR importer_iso IS NULL
                       OR target_cmd IS NULL
                ) AS n_output_missing_key_rows,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE trade_sync_availability = 'both'
                ) AS n_both_reports,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE trade_sync_availability
                        = 'corrected_importer_only'
                ) AS n_corrected_importer_only,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE trade_sync_availability = 'exporter_only'
                ) AS n_exporter_only,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE trade_sync_availability = 'neither'
                ) AS n_neither,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE trade_sync_has_usable_value
                ) AS n_with_usable_value,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE trade_sync_availability NOT IN (
                        'both',
                        'corrected_importer_only',
                        'exporter_only',
                        'neither'
                    )
                    OR trade_sync_availability IS NULL
                ) AS n_invalid_availability
            """
        ).fetchone()

        if output_audit_row is None:
            raise RuntimeError(
                "Could not inspect prepared Trade Sync output."
            )

        (
            n_output_rows,
            n_output_duplicate_keys,
            n_output_missing_key_rows,
            n_both_reports,
            n_corrected_importer_only,
            n_exporter_only,
            n_neither,
            n_with_usable_value,
            n_invalid_availability,
        ) = output_audit_row

        categorized_rows = (
            int(n_both_reports)
            + int(n_corrected_importer_only)
            + int(n_exporter_only)
            + int(n_neither)
        )

        usable_value_coverage_percent = (
            100.0
            * int(n_with_usable_value)
            / int(n_output_rows)
            if n_output_rows
            else 0.0
        )

        # ==============================================================
        # HARD AUDIT CONDITIONS
        # ==============================================================

        if int(n_input_rows) == 0:
            raise RuntimeError(
                "Trade Sync input contains no rows."
            )

        if int(n_input_duplicate_keys) > 0:
            raise RuntimeError(
                "Trade Sync input contains "
                f"{int(n_input_duplicate_keys):,} duplicate bilateral keys."
            )

        if int(n_input_missing_key_rows) > 0:
            raise RuntimeError(
                "Trade Sync input contains "
                f"{int(n_input_missing_key_rows):,} rows with an "
                "incomplete bilateral key."
            )

        if int(n_output_rows) != int(n_input_rows):
            raise RuntimeError(
                "Trade Sync preparation did not preserve the full "
                "input population: "
                f"input={int(n_input_rows):,}, "
                f"output={int(n_output_rows):,}."
            )

        if int(n_output_duplicate_keys) > 0:
            raise RuntimeError(
                "Prepared Trade Sync output contains "
                f"{int(n_output_duplicate_keys):,} duplicate "
                "bilateral keys."
            )

        if int(n_output_missing_key_rows) > 0:
            raise RuntimeError(
                "Prepared Trade Sync output contains "
                f"{int(n_output_missing_key_rows):,} rows with an "
                "incomplete bilateral key."
            )

        if categorized_rows != int(n_output_rows):
            raise RuntimeError(
                "Trade Sync availability categories do not partition "
                "the complete output population: "
                f"categorized={categorized_rows:,}, "
                f"output={int(n_output_rows):,}."
            )

        if int(n_invalid_availability) > 0:
            raise RuntimeError(
                "Prepared Trade Sync output contains "
                f"{int(n_invalid_availability):,} invalid availability "
                "classifications."
            )

        audit_passed = True
        ready_for_bilateral_pairs = True
        write_local_checkpoint(
            checkpoint_path=checkpoint_path,
            step_name="prepared_input",
            run_id=run_id,
            config=config_fp,
            input_fingerprint=input_fp,
            output_path=output_path,
        )       

        # ==============================================================
        # LOG RESULTS
        # ==============================================================

        logger.info(
            "%s Canonical Trade Sync input created.",
            ok,
        )
        logger.info(
            "  Input rows: %s",
            f"{int(n_input_rows):,}",
        )
        logger.info(
            "  Output rows: %s",
            f"{int(n_output_rows):,}",
        )
        logger.info(
            "  Year range: %s-%s",
            minimum_year,
            maximum_year,
        )
        logger.info(
            "  Duplicate bilateral keys: %s",
            f"{int(n_output_duplicate_keys):,}",
        )
        logger.info(
            "  Rows with incomplete bilateral keys: %s",
            f"{int(n_output_missing_key_rows):,}",
        )
        logger.info(
            "  Both exporter and corrected importer: %s",
            f"{int(n_both_reports):,}",
        )
        logger.info(
            "  Corrected importer only: %s",
            f"{int(n_corrected_importer_only):,}",
        )
        logger.info(
            "  Exporter only: %s",
            f"{int(n_exporter_only):,}",
        )
        logger.info(
            "  Neither usable: %s",
            f"{int(n_neither):,}",
        )
        logger.info(
            "  Rows with at least one usable FOB-basis value: %s",
            f"{int(n_with_usable_value):,}",
        )
        logger.info(
            "  Usable-value coverage: %.4f%%",
            usable_value_coverage_percent,
        )
        logger.info("  Output: %s", output_path)
        logger.info(
            "  Step 1 audit passed: %s",
            audit_passed,
        )
        logger.info(
            "  Ready for bilateral-pair preparation: %s",
            ready_for_bilateral_pairs,
        )

        return {
            "status": "completed",
            "output_path": output_path,
            "prepared_input_path": output_path,
            "audit_passed": audit_passed,
            "ready_for_bilateral_pairs": (
                ready_for_bilateral_pairs
            ),
            "n_input_rows": int(n_input_rows),
            "n_output_rows": int(n_output_rows),
            "minimum_year": minimum_year,
            "maximum_year": maximum_year,
            "n_duplicate_bilateral_keys": int(
                n_output_duplicate_keys
            ),
            "n_missing_key_rows": int(
                n_output_missing_key_rows
            ),
            "n_both_reports": int(n_both_reports),
            "n_corrected_importer_only": int(
                n_corrected_importer_only
            ),
            "n_exporter_only": int(n_exporter_only),
            "n_neither": int(n_neither),
            "n_with_usable_value": int(
                n_with_usable_value
            ),
            "usable_value_coverage_percent": (
                usable_value_coverage_percent
            ),
        }

    finally:
        con.close()