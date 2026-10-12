"""
vacoda.steps.validation.prepare_input

Prepare the canonical Value output for the final VACODA Validation workflow.

This module performs technical preparation only. Scientific validation,
model-based quantity assessment, correction, imputation, and release
decisions belong in later Validation workflow steps.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
from vacoda.pipeline.checkpoint import (
    validate_local_checkpoint,
    file_fingerprint,
    write_local_checkpoint,
    semantic_config_fingerprint,
)
from vacoda.config.steps.validation import ValidationConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.sql import sql_literal


def prepare_validation_input(
    *,
    cfg: ValidationConfig,
    input_path: Path,
    run_id: str,
    force: bool,
    logger,
) -> dict[str, Any]:
    """
    Prepare the canonical Validation input from the audited Value-step output

    The preparation step:

    - verifies that the configured input exists;
    - checks that required Validation fields are present;
    - restricts rows to the configured pipeline year scope;
    - writes the canonical prepared Validation parquet;
    - performs only technical integrity checks required for downstream
      Validation steps.

    No observations are removed for statistical or economic reasons here.
    """

    start = perf_counter()

    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Validation input parquet not found: {input_path}"
        )

    output_path = cfg.product_path("prepared_input")

    required_fields = tuple(
        cfg.method.structural.required_fields
    )

    checkpoint_path = (
        cfg.output_dir()
        / "checkpoints"
        / "prepared_input.json"
    )

    config_fp = semantic_config_fingerprint(
        config={
            "target_hs": cfg.scope.target_hs,
            "min_year": int(cfg.scope.min_year),
            "max_year": int(cfg.scope.max_year),
            "required_fields": list(required_fields),
        }
    )

    input_fp = {
        "validation_input": file_fingerprint(
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

    ready_for_structural_audit = True

    if not force and reusable:

        logger.info(
            "%s REUSE — Prepared Validation input checkpoint matches: %s",
            get_symbols("reuse"),
            output_path,
        )

        return {
            "status": "reused",
            "output_path": output_path,
            "prepared_input_path": output_path,
            "reused": True,
            "ready_for_structural_audit": ready_for_structural_audit,
            "run_id": run_id,
        }

    logger.info(
        "%s REBUILD — Prepared Validation input: %s",
        get_symbols("rebuilt"),
        "forced" if force else reason,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    logger.info("Preparing Validation input.")
    logger.info("  Input: %s", input_path)
    logger.info("  Output: %s", output_path)
    logger.info(
        "  Scope: %s, %s-%s",
        cfg.scope.target_hs,
        cfg.scope.min_year,
        cfg.scope.max_year,
    )


    con = duckdb.connect()

    try:
        con.execute(
            f"PRAGMA threads={int(cfg.engine.threads)}"
        )

        # --------------------------------------------------------------
        # Inspect source schema
        # --------------------------------------------------------------

        source_columns = {
            row[0]
            for row in con.execute(
                """
                DESCRIBE
                SELECT *
                FROM read_parquet(?)
                """,
                [str(input_path)],
            ).fetchall()
        }

        missing_fields = sorted(
            set(required_fields) - source_columns
        )

        if missing_fields:
            raise ValueError(
                "Validation input is missing required field(s): "
                + ", ".join(missing_fields)
            )

        logger.info(
            "  Required Validation columns found: %s",
            len(required_fields),
        )

        # --------------------------------------------------------------
        # Source statistics
        # --------------------------------------------------------------

        source_stats = con.execute(
            """
            SELECT
                COUNT(*) AS n_rows,
                MIN(year) AS min_year,
                MAX(year) AS max_year
            FROM read_parquet(?)
            """,
            [str(input_path)],
        ).fetchone()

        validation_stats = con.execute(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE trade_value_reconciled IS NOT NULL
                ) AS n_with_nominal_value,

                COUNT(*) FILTER (
                    WHERE trade_value_reconciled_constant_usd IS NOT NULL
                ) AS n_with_constant_value,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_kg IS NOT NULL
                ) AS n_with_trade_synced_quantity,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_kg IS NULL
                ) AS n_without_trade_synced_quantity,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_status = 'bilateral_synced'
                ) AS n_bilateral_synced,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_status = 'unilateral_retained'
                ) AS n_unilateral_retained,

                COUNT(*) FILTER (
                    WHERE trade_synced_quantity_status = 'missing'
                ) AS n_quantity_missing

            FROM read_parquet(?)
            WHERE year BETWEEN ? AND ?
            """,
            [
                str(input_path),
                int(cfg.scope.min_year),
                int(cfg.scope.max_year),
            ],
        ).fetchone()

        (
            n_with_nominal_value,
            n_with_constant_value,
            n_with_trade_synced_quantity,
            n_without_trade_synced_quantity,
            n_bilateral_synced,
            n_unilateral_retained,
            n_quantity_missing,
        ) = map(int, validation_stats)

        source_rows = int(source_stats[0])
        source_min_year = source_stats[1]
        source_max_year = source_stats[2]

        logger.info("  Source rows: %s", f"{source_rows:,}")
        logger.info(
            "  Source year range: %s-%s",
            source_min_year,
            source_max_year,
        )
        logger.info(
            "  Rows with nominal trade value: %s",
            f"{n_with_nominal_value:,}",
        )
        logger.info(
            "  Rows with constant-USD trade value: %s",
            f"{n_with_constant_value:,}",
        )
        logger.info(
            "  Rows with Trade-Synced quantity: %s",
            f"{n_with_trade_synced_quantity:,}",
        )
        logger.info(
            "  Rows without Trade-Synced quantity: %s",
            f"{n_without_trade_synced_quantity:,}",
        )
        # --------------------------------------------------------------
        # Write prepared Validation input
        # --------------------------------------------------------------
        #
        # Important:
        # Preserve the upstream dataset here.
        # Validation-specific scientific transformations happen later.
        # --------------------------------------------------------------
        input_sql = sql_literal(str(input_path))
        output_sql = sql_literal(str(output_path))

        logger.info(
            "  Preserving all upstream Value columns."
        )

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet({input_sql})
                WHERE year BETWEEN ? AND ?
            )
            TO {output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """,
            [
                int(cfg.scope.min_year),
                int(cfg.scope.max_year),
            ],
        )

        logger.info(
            "  Prepared Validation parquet written."
        )
        # --------------------------------------------------------------
        # Verify prepared output
        # --------------------------------------------------------------

        prepared_stats = con.execute(
            """
            SELECT
                COUNT(*) AS n_rows,
                MIN(year) AS min_year,
                MAX(year) AS max_year
            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()

        prepared_rows = int(prepared_stats[0])
        prepared_min_year = prepared_stats[1]
        prepared_max_year = prepared_stats[2]

    finally:
        con.close()

    ready_for_structural_audit = (
        prepared_rows > 0
        and prepared_min_year is not None
        and prepared_max_year is not None
        and prepared_min_year >= cfg.scope.min_year
        and prepared_max_year <= cfg.scope.max_year
    )

    if prepared_rows <= 0:
        raise RuntimeError(
            "Prepared Validation input contains no rows."
        )

    if prepared_min_year < cfg.scope.min_year:
        raise RuntimeError(
            "Prepared Validation input contains rows below "
            "the configured minimum year."
        )

    if prepared_max_year > cfg.scope.max_year:
        raise RuntimeError(
            "Prepared Validation input contains rows above "
            "the configured maximum year."
        )

    if not ready_for_structural_audit:
        raise RuntimeError(
            "Prepared Validation input failed readiness checks "
            "for structural audit."
        )

    write_local_checkpoint(
        checkpoint_path=checkpoint_path,
        step_name="prepared_input",
        run_id=run_id,
        config=config_fp,
        input_fingerprint=input_fp,
        output_path=output_path,
    )

    elapsed = perf_counter() - start

    logger.info(
        "%s Prepared Validation input created.",
        get_symbols("success"),
    )
    logger.info(
        "  Prepared rows: %s",
        f"{prepared_rows:,}",
    )
    logger.info(
        "  Prepared year range: %s-%s",
        prepared_min_year,
        prepared_max_year,
    )
    logger.info("  Output: %s", output_path)
    logger.info("  Ready for structural audit: %s", ready_for_structural_audit)

    return {
        "status": "completed",
        "run_id": run_id,

        "output_path": output_path,
        "prepared_input_path": output_path,

        "source_rows": source_rows,
        "prepared_rows": prepared_rows,

        "source_min_year": source_min_year,
        "source_max_year": source_max_year,
        "prepared_min_year": prepared_min_year,
        "prepared_max_year": prepared_max_year,

        "required_fields": required_fields,
        "reused": False,

        "ready_for_structural_audit": ready_for_structural_audit,
        "duration_seconds": elapsed,
        
        "n_with_nominal_value": n_with_nominal_value,
        "n_with_constant_value": n_with_constant_value,
        "n_with_trade_synced_quantity": n_with_trade_synced_quantity,
        "n_without_trade_synced_quantity": n_without_trade_synced_quantity,
        "n_bilateral_synced": n_bilateral_synced,
        "n_unilateral_retained": n_unilateral_retained,
        "n_quantity_missing": n_quantity_missing,
    }