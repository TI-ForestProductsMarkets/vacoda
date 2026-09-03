"""
vacoda.steps.validation.prepare_input

Prepare the canonical input for the final VACODA Validation workflow.

This module performs technical preparation only. Scientific validation,
outlier detection, missingness analysis, imputation, and release decisions
belong in later Validation workflow steps.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb

from vacoda.config.steps.validation import ValidationConfig
from vacoda.utils.display import get_symbols


def prepare_validation_input(
    *,
    cfg: ValidationConfig,
    input_path: Path,
    run_id: str,
    force: bool,
    logger,
) -> dict[str, Any]:
    """
    Prepare the canonical Validation input from the Value-step output.

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
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        not force
        and output_path.is_file()
        and output_path.stat().st_size > 0
    ):
        logger.info(
            "%s REUSE — Prepared Validation input: %s",
            get_symbols("reuse"),
            output_path,
        )

        return {
            "status": "completed",
            "output_path": output_path,
            "prepared_input_path": output_path,
            "reused": True,
            "ready_for_structural_audit": True,
            "run_id": run_id,
        }

    logger.info(
        "%s REBUILD — Prepared Validation input",
        get_symbols("rebuilt"),
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

    required_fields = tuple(
        cfg.method.structural.required_fields
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

        source_rows = int(source_stats[0])
        source_min_year = source_stats[1]
        source_max_year = source_stats[2]

        logger.info("  Source rows: %s", f"{source_rows:,}")
        logger.info(
            "  Source year range: %s-%s",
            source_min_year,
            source_max_year,
        )

        # --------------------------------------------------------------
        # Write prepared Validation input
        # --------------------------------------------------------------
        #
        # Important:
        # Preserve the upstream dataset here.
        # Validation-specific scientific transformations happen later.
        # --------------------------------------------------------------

        con.execute(
            """
            COPY (
                SELECT *
                FROM read_parquet(?)
                WHERE year BETWEEN ? AND ?
            )
            TO ?
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """,
            [
                str(input_path),
                int(cfg.scope.min_year),
                int(cfg.scope.max_year),
                str(output_path),
            ],
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
    logger.info("  Ready for structural audit: True")

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

        "ready_for_structural_audit": True,
        "duration_seconds": elapsed,
    }