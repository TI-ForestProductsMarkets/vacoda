"""
vacoda.quantity.prepare_input

Prepare the HS-synchronized dataset for quantity reconciliation.

This step validates the quantity-related input schema, derives basic
quantity availability flags, and writes a canonical prepared Parquet
for downstream Quantity workflow steps.

No source-selection or reconciliation decision is made here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.quantity import QuantityConfig
from vacoda.utils.sql import sql_literal
from vacoda.pipeline.checkpoint import (
    validate_local_checkpoint,
    file_fingerprint,
    write_local_checkpoint,
    semantic_config_fingerprint,
)

REQUIRED_QUANTITY_COLUMNS = (
    "quantity",
    "quantity_unit",
    "quantity_unit_code",
    "is_quantity_estimated",

    "alt_quantity",
    "alt_quantity_unit",
    "alt_quantity_unit_code",
    "is_alt_quantity_estimated",

    "net_weight_kg",
    "is_net_weight_estimated",

    "gross_weight_kg",
    "is_gross_weight_estimated",

    "target_hs",
    "target_cmd",
)

def _quote_identifier(name: str) -> str:
    """Quote a DuckDB identifier safely."""
    return '"' + name.replace('"', '""') + '"'


def _parquet_columns(
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
) -> set[str]:
    """
    Return the column names available in a Parquet file.
    """
    rows = con.execute(
        """
        DESCRIBE
        SELECT *
        FROM read_parquet(?)
        """,
        [str(input_path)],
    ).fetchall()

    return {str(row[0]) for row in rows}


def _validate_required_columns(
    available_columns: set[str],
) -> None:
    """
    Validate that the HS-sync output contains the quantity fields
    required by the Quantity module.
    """
    missing = sorted(
        set(REQUIRED_QUANTITY_COLUMNS) - available_columns
    )

    if missing:
        raise ValueError(
            "Quantity input is missing required columns: "
            + ", ".join(missing)
        )


def prepare_quantity_input(
    *,
    cfg: QuantityConfig,
    input_path: Path,
    run_id: str,
    force: bool,
    logger,
) -> dict[str, Any]:
    """
    Prepare canonical input for the Quantity workflow.

    Parameters
    ----------
    cfg:
        Parsed Quantity configuration.
    input_path:
        HS-synchronized input Parquet.
    run_id:
        Master-pipeline run identifier.
    force:
        Rebuild the prepared artifact even when it already exists.
    logger:
        Configured workflow logger.

    Returns
    -------
    dict
        Result metadata including ``prepared_input_path``.
    """

    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Quantity input parquet not found: {input_path}"
        )

    output_path = cfg.product_path("prepared_input")
    quantity_unit = (cfg.method.kg_reconciliation.quantity_unit)
    require_positive = (cfg.method.kg_reconciliation.require_positive_values)

    input_sql = sql_literal(input_path)
    output_sql = sql_literal(output_path)
    quantity_unit_sql = sql_literal(quantity_unit.lower())

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ### Local checkpointing for prepared input
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
            "REUSE — Prepared Quantity input local checkpoint matches: %s",
            output_path,
        )

        return {
            "status": "reused",
            "run_id": run_id,
            "output_path": output_path,
            "prepared_input_path": output_path,
            "audit_passed": True,
            "ready_for_reconciliation_evidence": True,
        }

    logger.info(
        "REBUILD — Prepared Quantity input: %s",
        "forced" if force else reason,
    )

    con = duckdb.connect()

    try:
        available_columns = _parquet_columns(
            con,
            input_path,
        )

        _validate_required_columns(
            available_columns,
        )

        logger.info(
            "Preparing Quantity input from: %s",
            input_path,
        )
        logger.info(
            "Required quantity columns found: %s",
            len(REQUIRED_QUANTITY_COLUMNS),
        )
        logger.info(
            "Configured reconciliation unit: %s",
            quantity_unit,
        )
        logger.info(
            "Require positive values: %s",
            require_positive,
        )

        if require_positive:
            quantity_valid_expr = """
                quantity IS NOT NULL
                AND quantity > 0
            """
            net_weight_valid_expr = """
                net_weight_kg IS NOT NULL
                AND net_weight_kg > 0
            """
            gross_weight_valid_expr = """
                gross_weight_kg IS NOT NULL
                AND gross_weight_kg > 0
            """
            alt_quantity_valid_expr = """
                alt_quantity IS NOT NULL
                AND alt_quantity > 0
            """
        else:
            quantity_valid_expr = """
                quantity IS NOT NULL
            """
            net_weight_valid_expr = """
                net_weight_kg IS NOT NULL
            """
            gross_weight_valid_expr = """
                gross_weight_kg IS NOT NULL
            """
            alt_quantity_valid_expr = """
                alt_quantity IS NOT NULL
            """

        query = f"""
            COPY (
                SELECT
                    *,

                    -- --------------------------------------------------
                    -- Quantity availability
                    -- --------------------------------------------------

                    (
                        {quantity_valid_expr}
                    ) AS has_quantity,

                    (
                        {quantity_valid_expr}
                        AND lower(trim(quantity_unit)) = {quantity_unit_sql}
                    ) AS has_quantity_kg,

                    (
                        {alt_quantity_valid_expr}
                    ) AS has_alt_quantity,

                    (
                        {net_weight_valid_expr}
                    ) AS has_net_weight_kg,

                    (
                        {gross_weight_valid_expr}
                    ) AS has_gross_weight_kg,

                    -- --------------------------------------------------
                    -- Reported / estimated state
                    -- --------------------------------------------------

                    (
                        {quantity_valid_expr}
                        AND coalesce(is_quantity_estimated, FALSE) = FALSE
                    ) AS has_reported_quantity,

                    (
                        {quantity_valid_expr}
                        AND coalesce(is_quantity_estimated, FALSE) = TRUE
                    ) AS has_estimated_quantity,

                    (
                        {net_weight_valid_expr}
                        AND coalesce(is_net_weight_estimated, FALSE) = FALSE
                    ) AS has_reported_net_weight_kg,

                    (
                        {net_weight_valid_expr}
                        AND coalesce(is_net_weight_estimated, FALSE) = TRUE
                    ) AS has_estimated_net_weight_kg,

                    (
                        {alt_quantity_valid_expr}
                        AND coalesce(is_alt_quantity_estimated, FALSE) = FALSE
                    ) AS has_reported_alt_quantity,

                    (
                        {alt_quantity_valid_expr}
                        AND coalesce(is_alt_quantity_estimated, FALSE) = TRUE
                    ) AS has_estimated_alt_quantity,

                    (
                        {gross_weight_valid_expr}
                        AND coalesce(is_gross_weight_estimated, FALSE) = FALSE
                    ) AS has_reported_gross_weight_kg,

                    (
                        {gross_weight_valid_expr}
                        AND coalesce(is_gross_weight_estimated, FALSE) = TRUE
                    ) AS has_estimated_gross_weight_kg,

                    -- --------------------------------------------------
                    -- kg-specific quantity state
                    -- --------------------------------------------------

                    (
                        {quantity_valid_expr}
                        AND lower(trim(quantity_unit)) = {quantity_unit_sql}
                        AND coalesce(is_quantity_estimated, FALSE) = FALSE
                    ) AS has_reported_quantity_kg,

                    (
                        {quantity_valid_expr}
                        AND lower(trim(quantity_unit)) = {quantity_unit_sql}
                        AND coalesce(is_quantity_estimated, FALSE) = TRUE
                    ) AS has_estimated_quantity_kg

                FROM read_parquet({input_sql})
            )
            TO {output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
        """

        con.execute(query)

        # ==============================================================
        # OUTPUT EXISTENCE
        # ==============================================================

        if not output_path.is_file():
            raise FileNotFoundError(
                "Prepared Quantity output was not created: "
                f"{output_path}"
            )

        # ==============================================================
        # INPUT / OUTPUT AUDIT
        # ==============================================================

        input_audit_row = con.execute(
            """
            SELECT COUNT(*) AS n_input_rows
            FROM read_parquet(?)
            """,
            [str(input_path)],
        ).fetchone()

        if input_audit_row is None:
            raise RuntimeError(
                "Could not inspect Quantity input."
            )

        output_audit_row = con.execute(
            """
            SELECT
                COUNT(*) AS n_output_rows,

                COUNT(*) FILTER (
                    WHERE has_quantity_kg
                      AND NOT has_quantity
                ) AS n_invalid_quantity_kg,

                COUNT(*) FILTER (
                    WHERE has_reported_quantity_kg
                      AND NOT has_quantity_kg
                ) AS n_invalid_reported_quantity_kg,

                COUNT(*) FILTER (
                    WHERE has_estimated_quantity_kg
                      AND NOT has_quantity_kg
                ) AS n_invalid_estimated_quantity_kg,

                COUNT(*) FILTER (
                    WHERE has_reported_quantity_kg
                      AND has_estimated_quantity_kg
                ) AS n_quantity_kg_state_overlap

            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()

        if output_audit_row is None:
            raise RuntimeError(
                "Could not inspect prepared Quantity output."
            )

        n_input_rows = int(input_audit_row[0])

        (
            n_output_rows,
            n_invalid_quantity_kg,
            n_invalid_reported_quantity_kg,
            n_invalid_estimated_quantity_kg,
            n_quantity_kg_state_overlap,
        ) = output_audit_row

        n_output_rows = int(n_output_rows)
        n_invalid_quantity_kg = int(n_invalid_quantity_kg)
        n_invalid_reported_quantity_kg = int(
            n_invalid_reported_quantity_kg
        )
        n_invalid_estimated_quantity_kg = int(
            n_invalid_estimated_quantity_kg
        )
        n_quantity_kg_state_overlap = int(
            n_quantity_kg_state_overlap
        )

        # ==============================================================
        # HARD AUDIT CONDITIONS
        # ==============================================================

        if n_input_rows == 0:
            raise RuntimeError(
                "Quantity input contains no rows."
            )

        if n_output_rows != n_input_rows:
            raise RuntimeError(
                "Quantity preparation did not preserve the full "
                "input population: "
                f"input={n_input_rows:,}, "
                f"output={n_output_rows:,}."
            )

        if n_invalid_quantity_kg > 0:
            raise RuntimeError(
                "Prepared Quantity output contains "
                f"{n_invalid_quantity_kg:,} rows where "
                "has_quantity_kg=True but has_quantity=False."
            )

        if n_invalid_reported_quantity_kg > 0:
            raise RuntimeError(
                "Prepared Quantity output contains "
                f"{n_invalid_reported_quantity_kg:,} invalid "
                "reported-kg states."
            )

        if n_invalid_estimated_quantity_kg > 0:
            raise RuntimeError(
                "Prepared Quantity output contains "
                f"{n_invalid_estimated_quantity_kg:,} invalid "
                "estimated-kg states."
            )

        if n_quantity_kg_state_overlap > 0:
            raise RuntimeError(
                "Prepared Quantity output contains "
                f"{n_quantity_kg_state_overlap:,} rows classified "
                "as both reported and estimated kg quantity."
            )

        audit_passed = True
        ready_for_reconciliation_evidence = audit_passed

        logger.info(
            "Prepared Quantity rows: %s",
            f"{n_output_rows:,}",
        )
        logger.info(
            "Prepared Quantity input: %s",
            output_path,
        )
        logger.info(
            "  Step 1 audit passed: %s",
            audit_passed,
        )
        logger.info(
            "  Ready for reconciliation evidence: %s",
            ready_for_reconciliation_evidence,
        )

    finally:
        con.close()

    write_local_checkpoint(
        checkpoint_path=checkpoint_path,
        step_name="prepared_input",
        run_id=run_id,
        config=config_fp,
        input_fingerprint=input_fp,
        output_path=output_path,
    )

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": output_path,
        "prepared_input_path": output_path,
        "audit_passed": audit_passed,
        "ready_for_reconciliation_evidence": ready_for_reconciliation_evidence,
        "n_input_rows": n_input_rows,
        "n_output_rows": int(n_output_rows),
        "required_columns": REQUIRED_QUANTITY_COLUMNS,
    }