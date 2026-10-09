"""
vacoda.steps.validation.training_sample

Build the model-training sample and HS x year unit-value evidence.

This step prepares the canonical value-weighted quantity/value data, defines the robust
HS × year unit-value reference, derives row-level plausibility evidence,
and applies the configured model-training population rule.

All Validation rows are retained in the output. The final model-training
population is identified by the boolean ``training_selected`` column.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb

from vacoda.config.steps.validation import ValidationConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.sql import sql_literal
from vacoda.pipeline.checkpoint import (
    validate_local_checkpoint,
    file_fingerprint,
    write_local_checkpoint,
    semantic_config_fingerprint,
)

# ======================================================================
# Helpers
# ======================================================================
def _format_training_parameter(
    value: float | int,
) -> str:
    """Return a compact filesystem-safe parameter value."""

    if isinstance(value, int):
        return str(value)

    return (
        f"{float(value):g}"
        .replace(".", "p")
        .replace("-", "m")
    )


def training_definition_name(
    *,
    cfg: ValidationConfig,
) -> str:
    """Return the human-readable training-definition directory name."""

    training_cfg = (
        cfg.model.training_sample
    )

    before = int(
        training_cfg.reference_year_window.before
    )

    after = int(
        training_cfg.reference_year_window.after
    )

    min_reference_n = int(
        training_cfg.min_hs_year_reference_observations
    )

    max_uv_distance = (
        _format_training_parameter(
            training_cfg.max_normalized_uv_distance_mad
        )
    )

    return (
        f"refwin_{before}_{after}"
        f"__n{min_reference_n}"
        f"__uv{max_uv_distance}"
    )


def _require_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    required_columns: set[str],
) -> None:
    """Require the columns needed to build the training sample."""

    input_sql = sql_literal(input_path)

    available = {
        str(row[0])
        for row in con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet({input_sql})
            """
        ).fetchall()
    }

    missing = sorted(
        required_columns - available
    )

    if missing:
        raise RuntimeError(
            "Validation training-sample input is missing required "
            "column(s): "
            + ", ".join(missing)
        )


# ======================================================================
# Public step
# ======================================================================


def build_training_sample(
    *,
    cfg: ValidationConfig,
    input_path: Path,
    run_id: str,
    output_path: Path | None = None,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Build the candidate model-training sample.


    This step:

    1. preserves all upstream columns;
    2. derives the canonical value-weighted unit value V / Q_canonical;
    4. derives robust HS × year unit-value reference statistics;
    5. measures each canonical value-weighted quantity/value observation relative to that
       reference;
    6. writes the resulting evidence for Diagnostics.

    """

    start = perf_counter()

    input_path = Path(input_path)


    if not input_path.is_file():
        raise FileNotFoundError(
            "Validation training-sample input not found: "
            f"{input_path}"
        )

    if output_path is not None:
        output_path = Path(
            output_path
        )

    else:

        base_output_path = (
            cfg.product_path(
                "training_sample"
            )
        )

        training_definition = (
            training_definition_name(
                cfg=cfg
            )
        )

        output_path = (
            base_output_path.parent
            / training_definition
            / base_output_path.name
        )


    ## local checkpoints
    diagnostics_cfg = cfg.diagnostics
    columns_cfg = diagnostics_cfg.columns

    model_training_cfg = (
        cfg.model.training_sample
    )

    min_reference_observations = int(
        model_training_cfg.min_hs_year_reference_observations
    )

    reference_year_before = int(
        model_training_cfg.reference_year_window.before
    )

    reference_year_after = int(
        model_training_cfg.reference_year_window.after
    )

    max_normalized_uv_distance_mad = float(
        model_training_cfg.max_normalized_uv_distance_mad
    )

    year_col = str(
        columns_cfg.year
    )

    product_col = str(
        columns_cfg.product
    )

    nominal_value_col = str(
        columns_cfg.nominal_value
    )

    quantity_col = str(
        cfg.model.target.column
    )

    required_columns = {
        year_col,
        product_col,
        "exporter_iso",
        "importer_iso",
        nominal_value_col,
        quantity_col,
    }

    checkpoint_path = (
        output_path.parent
        / "checkpoints"
        / "training_sample.json"
    )

    config_fp = semantic_config_fingerprint(
        config={
            "columns": {
                "year": year_col,
                "product": product_col,
                "nominal_value": nominal_value_col,
                "quantity": quantity_col,
            },
            "training_sample": {
                "reference_year_before": (
                    reference_year_before
                ),
                "reference_year_after": (
                    reference_year_after
                ),
                "min_reference_observations": (
                    min_reference_observations
                ),
                "max_normalized_uv_distance_mad": (
                    max_normalized_uv_distance_mad
                ),
            },
        }
    )

    input_fp = {
        "quantity_selection": file_fingerprint(
            input_path,
            hash_file=True,
        ),
    }

    # ------------------------------------------------------------------
    # Reuse
    # ------------------------------------------------------------------

    reusable, reason = validate_local_checkpoint(
        checkpoint_path=checkpoint_path,
        current_config=config_fp,
        current_input=input_fp,
        output_path=output_path,
    )

    if not force and reusable:

        logger.info(
            "%s REUSE — Validation training-sample "
            "local checkpoint matches: %s",
            get_symbols("reuse"),
            output_path,
        )

        return {
            "status": "reused",
            "run_id": run_id,
            "output_path": output_path,
            "training_sample_path": output_path,
            "reused": True,
        }

    logger.info(
        "%s REBUILD — Validation training sample: %s",
        get_symbols("rebuilt"),
        "forced" if force else reason,
    )

    logger.info(
        "  Output: %s",
        output_path,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    # ==================================================================
    # Configuration
    # ==================================================================

    diagnostics_cfg = cfg.diagnostics
    columns_cfg = diagnostics_cfg.columns

    model_training_cfg = (
        cfg.model.training_sample
    )

    min_reference_observations = int(
        model_training_cfg.min_hs_year_reference_observations
    )

    reference_year_before = int(
        model_training_cfg.reference_year_window.before
    )

    reference_year_after = int(
        model_training_cfg.reference_year_window.after
    )

    max_normalized_uv_distance_mad = float(
        model_training_cfg.max_normalized_uv_distance_mad
    )


    ### Logging Training-Sample Configuration
    logger.info(
        "  Training-sample configuration:"
    )

    logger.info(
        "    Reference-year window: -%s / +%s year(s)",
        reference_year_before,
        reference_year_after,
    )

    logger.info(
        "    Minimum HS × year reference observations: %s",
        min_reference_observations,
    )

    logger.info(
        "    Maximum normalized UV distance (MAD): %s",
        f"{max_normalized_uv_distance_mad:g}",
    )

    year_col = str(
        columns_cfg.year
    )

    product_col = str(
        columns_cfg.product
    )

    nominal_value_col = str(
        columns_cfg.nominal_value
    )

    quantity_col = str(cfg.model.target.column)

    required_columns = {
        year_col,
        product_col,
        "exporter_iso",
        "importer_iso",
        nominal_value_col,
        quantity_col,
    }

    # ==================================================================
    # Build
    # ==================================================================

    con = duckdb.connect()

    try:
        con.execute(
            f"PRAGMA threads={int(cfg.engine.threads)}"
        )

        input_sql = sql_literal(
            input_path
        )

        # --------------------------------------------------------------
        # Input contract
        # --------------------------------------------------------------

        _require_columns(
            con=con,
            input_path=input_path,
            required_columns=required_columns,
        )

        # --------------------------------------------------------------
        # Base population
        # --------------------------------------------------------------
        #
        # The selected unit value is defined from the canonical bilateral
        # canonical value-weighted quantity and reconciled nominal trade value:
        #
        #     UV = V_reconciled / Q_selected
        #
        # training_eligible identifies observations with a usable positive,
        # finite canonical value-weighted quantity, reconciled value, and selected unit value.
        # These observations form the population from which HS × year
        # unit-value reference statistics are constructed.
        # --------------------------------------------------------------

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW
                validation_training_base
            AS

            SELECT
                *,

                CASE
                    WHEN
                        {quantity_col} IS NOT NULL
                        AND {quantity_col} > 0

                        AND {nominal_value_col} IS NOT NULL
                        AND {nominal_value_col} > 0

                        AND isfinite(
                            CAST(
                                {quantity_col}
                                AS DOUBLE
                            )
                        )

                        AND isfinite(
                            CAST(
                                {nominal_value_col}
                                AS DOUBLE
                            )
                        )

                    THEN
                        CAST(
                            {nominal_value_col}
                            AS DOUBLE
                        )
                        /
                        CAST(
                            {quantity_col}
                            AS DOUBLE
                        )

                    ELSE NULL
                END
                    AS selected_unit_value,

                (
                    {quantity_col} IS NOT NULL
                    AND {quantity_col} > 0
                    AND isfinite(
                        CAST(
                            {quantity_col}
                            AS DOUBLE
                        )
                    )

                    AND {nominal_value_col} IS NOT NULL
                    AND {nominal_value_col} > 0
                    AND isfinite(
                        CAST(
                            {nominal_value_col}
                            AS DOUBLE
                        )
                    )

                    AND selected_unit_value IS NOT NULL
                    AND selected_unit_value > 0
                    AND isfinite(selected_unit_value)
                )
                    AS training_eligible

            FROM read_parquet({input_sql})
            """
        )

        # --------------------------------------------------------------
        # HS × focal-year reference population
        # --------------------------------------------------------------
        #
        # For every focal HS × year, construct the robust unit-value
        # reference using eligible observations from the configured
        # surrounding-year window:
        #
        #   [year - reference_year_before,
        #    year + reference_year_after]
        #
        # The HS product remains exact. Only the year dimension is pooled.
        #
        # Examples:
        #
        #   before=0, after=0
        #       -> exact HS × year reference
        #
        #   before=1, after=1
        #       -> HS × (year-1, year, year+1) reference
        #
        # At dataset boundaries only available years contribute.
        #
        # Work in log(unit value):
        #
        #   log(UV) = log(V / Q)
        #
        # The robust centre is the median log unit value.
        # Dispersion is the median absolute deviation around that centre.
        # --------------------------------------------------------------

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW
                validation_training_uv_reference
            AS

            WITH eligible_reference_rows AS (

                SELECT
                    {year_col}
                        AS source_year,

                    {product_col}
                        AS target_cmd,

                    LN(
                        selected_unit_value
                    )
                        AS log_unit_value

                FROM validation_training_base

                WHERE training_eligible

                AND selected_unit_value IS NOT NULL
                AND selected_unit_value > 0

                AND isfinite(
                    selected_unit_value
                )
            ),

            focal_groups AS (

                SELECT DISTINCT
                    {year_col}
                        AS focal_year,

                    {product_col}
                        AS target_cmd

                FROM validation_training_base
            ),

            window_reference_rows AS (

                SELECT
                    f.focal_year,
                    f.target_cmd,

                    r.source_year,
                    r.log_unit_value

                FROM focal_groups AS f

                INNER JOIN eligible_reference_rows AS r
                    ON r.target_cmd = f.target_cmd

                AND r.source_year
                    BETWEEN
                        f.focal_year - {reference_year_before}
                    AND
                        f.focal_year + {reference_year_after}
            ),

            centres AS (

                SELECT
                    focal_year,
                    target_cmd,

                    COUNT(*)
                        AS hs_year_uv_reference_n,

                    MEDIAN(
                        log_unit_value
                    )
                        AS median_log_unit_value

                FROM window_reference_rows

                GROUP BY
                    focal_year,
                    target_cmd
            ),

            deviations AS (

                SELECT
                    r.focal_year,
                    r.target_cmd,

                    c.hs_year_uv_reference_n,
                    c.median_log_unit_value,

                    ABS(
                        r.log_unit_value
                        - c.median_log_unit_value
                    )
                        AS log_absolute_deviation

                FROM window_reference_rows AS r

                INNER JOIN centres AS c
                    USING (
                        focal_year,
                        target_cmd
                    )
            )

            SELECT
                focal_year
                    AS year,

                target_cmd,

                MAX(
                    hs_year_uv_reference_n
                )
                    AS hs_year_uv_reference_n,

                EXP(
                    MAX(
                        median_log_unit_value
                    )
                )
                    AS hs_year_uv_reference,

                MAX(
                    median_log_unit_value
                )
                    AS hs_year_uv_reference_log,

                MEDIAN(
                    log_absolute_deviation
                )
                    AS hs_year_uv_log_mad

            FROM deviations

            GROUP BY
                focal_year,
                target_cmd
            """
        )

        # --------------------------------------------------------------
        # Attach reference evidence
        # --------------------------------------------------------------
        #
        # No statistical cutoff is applied.
        #
        # uv_evidence_available only indicates that the row contains the
        # mechanically necessary information for later training-population
        # evaluation:
        #
        #   - canonical value-weighted quantity
        #   - reconciled value
        #   - selected unit value
        #   - HS × year reference
        #
        # It does NOT mean that the row is approved for model training.
        # --------------------------------------------------------------

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        if temporary_path.exists():
            temporary_path.unlink()

        temporary_sql = sql_literal(
            temporary_path
        )

        con.execute(
            f"""
            COPY (

                SELECT
                    evidence.*,

                    CASE
                        WHEN
                            evidence.selected_uv_log_distance IS NOT NULL

                            AND evidence.hs_year_uv_log_mad IS NOT NULL
                            AND evidence.hs_year_uv_log_mad > 0

                            AND isfinite(
                                evidence.selected_uv_log_distance
                            )

                            AND isfinite(
                                evidence.hs_year_uv_log_mad
                            )

                        THEN
                            evidence.selected_uv_log_distance
                            /
                            evidence.hs_year_uv_log_mad

                        ELSE NULL
                    END
                        AS training_normalized_uv_distance,

                    (
                        evidence.hs_year_uv_reference_n IS NOT NULL
                        AND evidence.hs_year_uv_reference_n
                            >= {min_reference_observations}

                        AND evidence.hs_year_uv_log_mad IS NOT NULL
                        AND evidence.hs_year_uv_log_mad > 0
                        AND isfinite(
                            evidence.hs_year_uv_log_mad
                        )

                        AND evidence.selected_uv_log_distance IS NOT NULL
                        AND isfinite(
                            evidence.selected_uv_log_distance
                        )

                        AND (
                            evidence.selected_uv_log_distance
                            /
                            evidence.hs_year_uv_log_mad
                        ) <= {max_normalized_uv_distance_mad}
                    )
                        AS training_selected

                FROM (

                    SELECT
                        b.*,

                        r.hs_year_uv_reference,
                        r.hs_year_uv_reference_log,
                        r.hs_year_uv_reference_n,
                        r.hs_year_uv_log_mad,

                        CASE
                            WHEN
                                b.selected_unit_value IS NOT NULL
                                AND b.selected_unit_value > 0

                                AND r.hs_year_uv_reference IS NOT NULL
                                AND r.hs_year_uv_reference > 0

                                AND isfinite(
                                    b.selected_unit_value
                                )

                                AND isfinite(
                                    r.hs_year_uv_reference
                                )

                            THEN
                                ABS(
                                    LN(
                                        b.selected_unit_value
                                    )
                                    -
                                    LN(
                                        r.hs_year_uv_reference
                                    )
                                )

                            ELSE NULL
                        END
                            AS selected_uv_log_distance,

                        (
                            b.{quantity_col} IS NOT NULL
                            AND b.{quantity_col} > 0

                            AND isfinite(
                                CAST(
                                    b.{quantity_col}
                                    AS DOUBLE
                                )
                            )

                            AND b.{nominal_value_col} IS NOT NULL
                            AND b.{nominal_value_col} > 0

                            AND isfinite(
                                CAST(
                                    b.{nominal_value_col}
                                    AS DOUBLE
                                )
                            )

                            AND b.selected_unit_value IS NOT NULL
                            AND b.selected_unit_value > 0

                            AND isfinite(
                                b.selected_unit_value
                            )

                            AND r.hs_year_uv_reference IS NOT NULL
                            AND r.hs_year_uv_reference > 0

                            AND isfinite(
                                r.hs_year_uv_reference
                            )

                            AND r.hs_year_uv_reference_n IS NOT NULL
                        )
                            AS uv_evidence_available

                    FROM validation_training_base AS b

                    LEFT JOIN validation_training_uv_reference AS r
                    ON r.year = b.{year_col}
                    AND r.target_cmd = b.{product_col}

                ) AS evidence
            )
            TO {temporary_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )
        temporary_path.replace(
            output_path
        )

        output_sql = sql_literal(
            output_path
        )

        # --------------------------------------------------------------
        # Basic output accounting
        # --------------------------------------------------------------

        counts = con.execute(
            f"""
            SELECT
                COUNT(*)
                    AS n_rows,

                COUNT(*) FILTER (
                    WHERE training_eligible
                )
                    AS n_reference_rows,

                COUNT(*) FILTER (
                    WHERE uv_evidence_available
                )
                    AS n_uv_evidence_available_rows,

                COUNT(*) FILTER (
                    WHERE selected_unit_value IS NOT NULL
                )
                    AS n_selected_unit_value_rows,

                COUNT(*) FILTER (
                    WHERE hs_year_uv_reference IS NOT NULL
                )
                    AS n_rows_with_reference,

                COUNT(*) FILTER (
                    WHERE selected_uv_log_distance IS NOT NULL
                )
                    AS n_rows_with_uv_distance,

                COUNT(*) FILTER (
                    WHERE training_selected
                )
                    AS n_training_selected_rows

            FROM read_parquet({output_sql})
            """
        ).fetchone()

        (
            n_rows,
            n_reference_rows,
            n_uv_evidence_available_rows,
            n_selected_unit_value_rows,
            n_rows_with_reference,
            n_rows_with_uv_distance,
            n_training_selected_rows,
        ) = map(
            int,
            counts,
        )

        n_reference_groups = int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM validation_training_uv_reference
                """
            ).fetchone()[0]
        )

    finally:
        con.close()

    # ==================================================================
    # Logging
    # ==================================================================

    elapsed = perf_counter() - start

    logger.info(
        "  Output rows: %s",
        f"{n_rows:,}",
    )

    logger.info(
        "  Reference-population rows: %s",
        f"{n_reference_rows:,}",
    )

    logger.info(
        "  HS × year reference groups: %s",
        f"{n_reference_groups:,}",
    )

    logger.info(
        "  Reference-year window: -%s / +%s year(s)",
        reference_year_before,
        reference_year_after,
    )

    logger.info(
        "  Rows with selected unit value: %s",
        f"{n_selected_unit_value_rows:,}",
    )

    logger.info(
        "  Rows with HS × year UV reference: %s",
        f"{n_rows_with_reference:,}",
    )

    logger.info(
        "  Rows with UV-distance evidence: %s",
        f"{n_rows_with_uv_distance:,}",
    )

    logger.info(
        "  Rows with complete UV evidence: %s",
        f"{n_uv_evidence_available_rows:,}",
    )

    logger.info(
        "  Selected training rows: %s",
        f"{n_training_selected_rows:,}",
    )
    logger.info(
        "  Output: %s",
        output_path,
    )

    logger.info(
        "%s CHECK — Training sample built.",
        get_symbols("check"),
    )

    return {
        "status": "completed",
        "run_id": run_id,

        "output_path": output_path,
        "training_sample_path": output_path,

        "n_rows": n_rows,
        "n_reference_rows": n_reference_rows,
        "n_reference_groups": n_reference_groups,
        "n_selected_unit_value_rows": (
            n_selected_unit_value_rows
        ),
        "n_rows_with_reference": (
            n_rows_with_reference
        ),
        "n_rows_with_uv_distance": (
            n_rows_with_uv_distance
        ),
        "n_uv_evidence_available_rows": (
            n_uv_evidence_available_rows
        ),

        "n_training_selected_rows": (
            n_training_selected_rows
        ),
        "duration_seconds": elapsed,
        "reused": False,
    }