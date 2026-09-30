"""
vacoda.steps.validation.diagnostics

Build post-selection diagnostics used by the Validation workflow.

"""

from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import pandas as pd

from vacoda.config.steps.validation import ValidationConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.sql import sql_literal


# ======================================================================
# Helpers
# ======================================================================


def _safe_identifier(
    name: str,
) -> str:
    """Return a safely quoted SQL identifier."""

    if not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*",
        str(name),
    ):
        raise ValueError(
            f"Unsafe Validation diagnostics column identifier: {name!r}"
        )

    return f'"{name}"'


def _share(
    numerator: int,
    denominator: int,
) -> float | None:
    """Return numerator / denominator, or NULL for an empty denominator."""

    if denominator <= 0:
        return None

    return float(numerator) / float(denominator)


def _require_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    required_columns: set[str],
) -> None:
    """Require configured diagnostics-development columns in the input."""

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
            "Validation diagnostics input is missing required column(s): "
            + ", ".join(missing)
        )


# ======================================================================
# Public step
# ======================================================================


def build_diagnostics(
    *,
    cfg: ValidationConfig,
    input_path: Path,
    structural_audit_path: Path,
    quantity_selection_audit_path: Path,
    model_predictions_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Build post-selection quantity diagnostics for reporting.

    """

    start = perf_counter()

    input_path = Path(input_path)
    structural_audit_path = Path(
        structural_audit_path
    )
    quantity_selection_audit_path = Path(
        quantity_selection_audit_path
    )

    model_predictions_path = Path(
        model_predictions_path
    )

    if not model_predictions_path.is_file():
        raise FileNotFoundError(
            "Validation model-prediction artifact not found: "
            f"{model_predictions_path}"
        )
    
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Validation diagnostics input not found: {input_path}"
        )

    if not structural_audit_path.is_file():
        raise FileNotFoundError(
            "Validation diagnostics structural-audit artifact not found: "
            f"{structural_audit_path}"
        )

    if not quantity_selection_audit_path.is_file():
        raise FileNotFoundError(
            "Validation diagnostics quantity-selection audit artifact not found: "
            f"{quantity_selection_audit_path}"
        )

    audit_con = duckdb.connect()

    try:
        audit_passed_row = audit_con.execute(
            f"""
            SELECT audit_passed
            FROM read_parquet({sql_literal(quantity_selection_audit_path)})
            LIMIT 1
            """
        ).fetchone()
    finally:
        audit_con.close()

    if audit_passed_row is None or not bool(audit_passed_row[0]):
        raise RuntimeError(
            "Validation diagnostics require quantity_selection_audit.audit_passed == True. "
            f"Audit file: {quantity_selection_audit_path}"
        )

    output_path = cfg.product_path(
        "diagnostics"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Reuse
    # ------------------------------------------------------------------

    if (
        not force
        and output_path.is_file()
        and output_path.stat().st_size > 0
    ):
        logger.info(
            "%s REUSE — Validation diagnostics: %s",
            get_symbols("reuse"),
            output_path,
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "output_path": output_path,
            "diagnostics_path": output_path,
            "reused": True,
            "fitted": False,
            "ready_for_training_decision": True,
            "selection_already_applied": True,
        }

    logger.info(
        "%s REBUILD — Validation diagnostics",
        get_symbols("rebuilt"),
    )
    logger.info(
        "Building post-selection quantity diagnostics for reporting."
    )
    logger.info(
        "  Input: %s",
        input_path,
    )

    logger.info(
        "  Model predictions: %s",
        model_predictions_path,
    )

    logger.info(
        "  Structural audit: %s",
        structural_audit_path,
    )
    logger.info(
        "  Output: %s",
        output_path,
    )

    # ==================================================================
    # Configuration
    # ==================================================================

    diagnostics_cfg = cfg.diagnostics
    columns_cfg = diagnostics_cfg.columns
    training_cfg = diagnostics_cfg.training_population_sensitivity

    year_col = str(
        columns_cfg.year
    )
    product_col = str(
        columns_cfg.product
    )

    # The selector now owns the final quantity methodology. Diagnostics
    # evaluates the selected result and the retained alternatives.
    quantity_col = "quantity_selected_kg"
    selected_candidate_col = "quantity_selected_candidate"
    selected_source_col = "quantity_selected_source"

    # Fixed quantity candidates retained from Trade Sync.
    # These are compared in Validation; none is treated as final here.
    simple_mean_col = "quantity_simple_mean_kg"
    value_weighted_col = "quantity_value_weighted_kg"
    quantity_reliability_weighted_col = (
        "quantity_reliability_weighted_kg"
    )

    nominal_value_col = str(
        columns_cfg.nominal_value
    )

    constant_value_col = str(
        columns_cfg.constant_value
    )

    export_quantity_col = str(
        columns_cfg.export_quantity
    )

    import_quantity_col = str(
        columns_cfg.import_quantity
    )

    export_estimated_col = str(
        columns_cfg.export_estimated
    )

    import_estimated_col = str(
        columns_cfg.import_estimated
    )

    relative_difference_col = str(
        columns_cfg.quantity_relative_difference
    )

    uv_reference_col = str(
        columns_cfg.unit_value_reference
    )

    uv_reference_n_col = str(
        columns_cfg.unit_value_reference_observations
    )

    disagreement_thresholds = tuple(
        training_cfg.disagreement_thresholds
    )

    uv_support_thresholds = tuple(
        training_cfg.unit_value_reference_min_observations
    )

    uv_distance_thresholds = tuple(
        training_cfg.unit_value_log_distance_thresholds
    )

    require_bilateral = bool(
        training_cfg.require_bilateral_quantities
    )

    require_non_estimated = bool(
        training_cfg.require_non_estimated_quantities
    )


    logger.info(
        "  Require bilateral quantities: %s",
        require_bilateral,
    )
    logger.info(
        "  Require non-estimated quantities: %s",
        require_non_estimated,
    )
    logger.info(
        "  Mirror-disagreement thresholds: %s",
        ", ".join(
            f"{value:.0%}"
            for value in disagreement_thresholds
        ),
    )
    logger.info(
        "  Unit-value support thresholds: %s",
        ", ".join(
            f"{value:,}"
            for value in uv_support_thresholds
        ),
    )
    logger.info(
        "  Unit-value log-distance thresholds: %s",
        ", ".join(
            f"{value:g}"
            for value in uv_distance_thresholds
        ),
    )
    # ==================================================================
    # Input contract
    # ==================================================================

    required_columns = {
        year_col,
        product_col,
        "exporter_iso",
        "importer_iso",
        quantity_col,
        simple_mean_col,
        value_weighted_col,
        quantity_reliability_weighted_col,
        nominal_value_col,
        constant_value_col,
        export_quantity_col,
        import_quantity_col,
        export_estimated_col,
        import_estimated_col,
        relative_difference_col,
        uv_reference_col,
        uv_reference_n_col,
        selected_candidate_col,
        selected_source_col,
        "candidate_simple_mean_uv_log_distance",
        "candidate_value_weighted_uv_log_distance",
        "candidate_quantity_reliability_weighted_uv_log_distance",
        "candidate_simple_mean_export_log_gap",
        "candidate_simple_mean_import_log_gap",
        "candidate_value_weighted_export_log_gap",
        "candidate_value_weighted_import_log_gap",
        "candidate_quantity_reliability_weighted_export_log_gap",
        "candidate_quantity_reliability_weighted_import_log_gap",
        "candidate_simple_mean_conditioned_uv_abs_residual",
        "candidate_value_weighted_conditioned_uv_abs_residual",
        "candidate_quantity_reliability_weighted_conditioned_uv_abs_residual",
        "quantity_conditioned_model_n_rows",
        "quantity_conditioned_log_uv_intercept",
        "quantity_conditioned_log_uv_slope",
        "quantity_conditioned_log_uv_r2",

        # Training-sample evidence
        "selected_unit_value",
        "training_reference_eligible",
        "hs_year_uv_reference",
        "hs_year_uv_reference_log",
        "hs_year_uv_reference_n",
        "hs_year_uv_log_mad",
        "selected_uv_log_distance",
        "training_hard_eligible",
    }

    con = duckdb.connect()

    try:
        con.execute(
            f"PRAGMA threads={int(cfg.engine.threads)}"
        )

        _require_columns(
            con=con,
            input_path=input_path,
            required_columns=required_columns,
        )

        input_sql = sql_literal(
            input_path
        )

        model_predictions_sql = sql_literal(
            model_predictions_path
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW
                validation_model_predictions
            AS

            SELECT
                model_name,
                model_type,
                held_out_year,

                target_transformed,
                prediction_transformed,

                target_quantity_kg,
                prediction_quantity_kg,

                prediction_transformed
                - target_transformed
                    AS log_error,

                ABS(
                    prediction_transformed
                    - target_transformed
                )
                    AS absolute_log_error,

                POWER(
                    prediction_transformed
                    - target_transformed,
                    2
                )
                    AS squared_log_error

            FROM read_parquet(
                {model_predictions_sql}
            )

            WHERE target_transformed IS NOT NULL
            AND prediction_transformed IS NOT NULL
            AND isfinite(target_transformed)
            AND isfinite(prediction_transformed)
            """
        )
        # --------------------------------------------------------------
        # Safe SQL identifiers
        # --------------------------------------------------------------

        year_sql = _safe_identifier(
            year_col
        )

        product_sql = _safe_identifier(
            product_col
        )

        quantity_sql = _safe_identifier(
            quantity_col
        )

        simple_mean_sql = _safe_identifier(
            simple_mean_col
        )

        value_weighted_sql = _safe_identifier(
            value_weighted_col
        )

        quantity_reliability_weighted_sql = _safe_identifier(
            quantity_reliability_weighted_col
        )

        nominal_value_sql = _safe_identifier(
            nominal_value_col
        )

        constant_value_sql = _safe_identifier(
            constant_value_col
        )

        export_quantity_sql = _safe_identifier(
            export_quantity_col
        )

        import_quantity_sql = _safe_identifier(
            import_quantity_col
        )

        export_estimated_sql = _safe_identifier(
            export_estimated_col
        )

        import_estimated_sql = _safe_identifier(
            import_estimated_col
        )

        relative_difference_sql = _safe_identifier(
            relative_difference_col
        )

        uv_reference_sql = _safe_identifier(
            uv_reference_col
        )

        uv_reference_n_sql = _safe_identifier(
            uv_reference_n_col
        )

        # ==================================================================
        # Canonical diagnostics-development view
        # ==================================================================
        #
        # The unit-value plausibility comparison intentionally uses
        # nominal reconciled trade value.
        #
        # The product-year unit-value reference was derived from nominal
        # within-year trade values. Using constant-price USD against that
        # nominal reference would mix two price bases.
        #
        # Constant-price value is retained here for later predictive model
        # development, where cross-year comparability becomes relevant.
        # ==================================================================

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW
                validation_diagnostics_base
            AS

            SELECT
                {year_sql}
                    AS year,

                exporter_iso,
                importer_iso,

                {product_sql}
                    AS target_cmd,

                {quantity_sql}
                    AS trade_synced_quantity_kg,

                {simple_mean_sql}
                    AS quantity_simple_mean_kg,

                {value_weighted_sql}
                    AS quantity_value_weighted_kg,

                {quantity_reliability_weighted_sql}
                    AS quantity_reliability_weighted_kg,

                {nominal_value_sql}
                    AS trade_value_reconciled,

                {constant_value_sql}
                    AS trade_value_reconciled_constant_usd,

                {export_quantity_sql}
                    AS export_quantity_kg,

                {import_quantity_sql}
                    AS import_quantity_kg,

                {export_estimated_sql}
                    AS export_quantity_estimated,

                {import_estimated_sql}
                    AS import_quantity_estimated,

                {relative_difference_sql}
                    AS quantity_relative_difference,

                                    CASE
                    WHEN
                        {export_quantity_sql} IS NOT NULL
                        AND {import_quantity_sql} IS NOT NULL

                        AND {export_quantity_sql} > 0
                        AND {import_quantity_sql} > 0

                        AND isfinite(
                            CAST(
                                {export_quantity_sql}
                                AS DOUBLE
                            )
                        )

                        AND isfinite(
                            CAST(
                                {import_quantity_sql}
                                AS DOUBLE
                            )
                        )

                    THEN
                        (
                            CAST(
                                {export_quantity_sql}
                                AS DOUBLE
                            )
                            -
                            CAST(
                                {import_quantity_sql}
                                AS DOUBLE
                            )
                        )
                        /
                        CAST(
                            {import_quantity_sql}
                            AS DOUBLE
                        )

                    ELSE NULL
                END
                    AS quantity_difference_vs_import,

                CASE
                    WHEN
                        {export_quantity_sql} IS NOT NULL
                        AND {import_quantity_sql} IS NOT NULL

                        AND {export_quantity_sql} > 0
                        AND {import_quantity_sql} > 0

                        AND isfinite(
                            CAST(
                                {export_quantity_sql}
                                AS DOUBLE
                            )
                        )

                        AND isfinite(
                            CAST(
                                {import_quantity_sql}
                                AS DOUBLE
                            )
                        )

                    THEN
                        ABS(
                            CAST(
                                {export_quantity_sql}
                                AS DOUBLE
                            )
                            -
                            CAST(
                                {import_quantity_sql}
                                AS DOUBLE
                            )
                        )
                        /
                        CAST(
                            {import_quantity_sql}
                            AS DOUBLE
                        )

                    ELSE NULL
                END
                    AS quantity_absolute_difference_vs_import,

                CASE
                    WHEN
                        {export_quantity_sql} IS NOT NULL
                        AND {import_quantity_sql} IS NOT NULL
                        AND {export_quantity_sql} > 0
                        AND {import_quantity_sql} > 0
                        AND isfinite(
                            CAST(
                                {export_quantity_sql}
                                AS DOUBLE
                            )
                        )
                        AND isfinite(
                            CAST(
                                {import_quantity_sql}
                                AS DOUBLE
                            )
                        )
                    THEN SQRT(
                        CAST(
                            {export_quantity_sql}
                            AS DOUBLE
                        )
                        * CAST(
                            {import_quantity_sql}
                            AS DOUBLE
                        )
                    )
                END
                    AS bilateral_quantity_kg,

                CASE
                    WHEN
                        {export_quantity_sql} IS NOT NULL
                        AND {import_quantity_sql} IS NOT NULL
                        AND {export_quantity_sql} > 0
                        AND {import_quantity_sql} > 0
                        AND {nominal_value_sql} IS NOT NULL
                        AND {nominal_value_sql} > 0
                        AND isfinite(CAST({export_quantity_sql} AS DOUBLE))
                        AND isfinite(CAST({import_quantity_sql} AS DOUBLE))
                        AND isfinite(CAST({nominal_value_sql} AS DOUBLE))
                    THEN
                        CAST({nominal_value_sql} AS DOUBLE)
                        /
                        SQRT(
                            CAST({export_quantity_sql} AS DOUBLE)
                            * CAST({import_quantity_sql} AS DOUBLE)
                        )
                END
                    AS bilateral_unit_value,

                CASE
                    WHEN
                        {export_quantity_sql} > 0
                        AND {import_quantity_sql} > 0
                    THEN LN(
                        SQRT(
                            CAST({export_quantity_sql} AS DOUBLE)
                            * CAST({import_quantity_sql} AS DOUBLE)
                        )
                    )
                END
                    AS log_bilateral_quantity,

                CASE
                    WHEN
                        {export_quantity_sql} > 0
                        AND {import_quantity_sql} > 0
                        AND {nominal_value_sql} > 0
                    THEN LN(
                        CAST({nominal_value_sql} AS DOUBLE)
                        /
                        SQRT(
                            CAST({export_quantity_sql} AS DOUBLE)
                            * CAST({import_quantity_sql} AS DOUBLE)
                        )
                    )
                END
                    AS log_bilateral_unit_value,

                {uv_reference_sql}
                    AS product_year_median_unit_value,

                {uv_reference_n_sql}
                    AS unit_value_reference_observations,

                (
                    {quantity_sql} IS NOT NULL
                    AND {quantity_sql} > 0
                    AND isfinite(
                        CAST(
                            {quantity_sql}
                            AS DOUBLE
                        )
                    )
                )
                    AS has_synced_quantity,

                (
                    {export_quantity_sql} IS NOT NULL
                    AND {export_quantity_sql} > 0
                    AND isfinite(
                        CAST(
                            {export_quantity_sql}
                            AS DOUBLE
                        )
                    )
                )
                    AS has_export_quantity,

                (
                    {import_quantity_sql} IS NOT NULL
                    AND {import_quantity_sql} > 0
                    AND isfinite(
                        CAST(
                            {import_quantity_sql}
                            AS DOUBLE
                        )
                    )
                )
                    AS has_import_quantity,

                (
                    {export_quantity_sql} IS NOT NULL
                    AND {import_quantity_sql} IS NOT NULL

                    AND {export_quantity_sql} > 0
                    AND {import_quantity_sql} > 0

                    AND isfinite(
                        CAST(
                            {export_quantity_sql}
                            AS DOUBLE
                        )
                    )

                    AND isfinite(
                        CAST(
                            {import_quantity_sql}
                            AS DOUBLE
                        )
                    )
                )
                    AS has_both_quantities,

                (
                    {export_quantity_sql} IS NOT NULL
                    AND {import_quantity_sql} IS NOT NULL

                    AND {export_quantity_sql} > 0
                    AND {import_quantity_sql} > 0

                    AND isfinite(
                        CAST(
                            {export_quantity_sql}
                            AS DOUBLE
                        )
                    )

                    AND isfinite(
                        CAST(
                            {import_quantity_sql}
                            AS DOUBLE
                        )
                    )

                    AND COALESCE(
                        {export_estimated_sql},
                        FALSE
                    ) = FALSE

                    AND COALESCE(
                        {import_estimated_sql},
                        FALSE
                    ) = FALSE
                )
                    AS both_quantities_observed,

        CASE
            WHEN
                {export_quantity_sql} IS NOT NULL
                AND {export_quantity_sql} > 0

                AND {nominal_value_sql} IS NOT NULL
                AND {nominal_value_sql} > 0

                AND {uv_reference_sql} IS NOT NULL
                AND {uv_reference_sql} > 0

                AND isfinite(
                    CAST(
                        {export_quantity_sql}
                        AS DOUBLE
                    )
                )

                AND isfinite(
                    CAST(
                        {nominal_value_sql}
                        AS DOUBLE
                    )
                )

                AND isfinite(
                    CAST(
                        {uv_reference_sql}
                        AS DOUBLE
                    )
                )

            THEN
                ABS(
                    LN(
                        CAST(
                            {nominal_value_sql}
                            AS DOUBLE
                        )
                        /
                        CAST(
                            {export_quantity_sql}
                            AS DOUBLE
                        )
                    )
                    -
                    LN(
                        CAST(
                            {uv_reference_sql}
                            AS DOUBLE
                        )
                    )
                )

            ELSE NULL
        END
            AS export_unit_value_log_distance,

        CASE
            WHEN
                {import_quantity_sql} IS NOT NULL
                AND {import_quantity_sql} > 0

                AND {nominal_value_sql} IS NOT NULL
                AND {nominal_value_sql} > 0

                AND {uv_reference_sql} IS NOT NULL
                AND {uv_reference_sql} > 0

                AND isfinite(
                    CAST(
                        {import_quantity_sql}
                        AS DOUBLE
                    )
                )

                AND isfinite(
                    CAST(
                        {nominal_value_sql}
                        AS DOUBLE
                    )
                )

                AND isfinite(
                    CAST(
                        {uv_reference_sql}
                        AS DOUBLE
                    )
                )

            THEN
                ABS(
                    LN(
                        CAST(
                            {nominal_value_sql}
                            AS DOUBLE
                        )
                        /
                        CAST(
                            {import_quantity_sql}
                            AS DOUBLE
                        )
                    )
                    -
                    LN(
                        CAST(
                            {uv_reference_sql}
                            AS DOUBLE
                        )
                    )
                )

            ELSE NULL
        END
            AS import_unit_value_log_distance,

        CASE
            WHEN
                {export_quantity_sql} IS NOT NULL
                AND {export_quantity_sql} > 0
                AND {import_quantity_sql} IS NOT NULL
                AND {import_quantity_sql} > 0
                AND {nominal_value_sql} IS NOT NULL
                AND {nominal_value_sql} > 0
                AND {uv_reference_sql} IS NOT NULL
                AND {uv_reference_sql} > 0
                AND isfinite(CAST({export_quantity_sql} AS DOUBLE))
                AND isfinite(CAST({import_quantity_sql} AS DOUBLE))
                AND isfinite(CAST({nominal_value_sql} AS DOUBLE))
                AND isfinite(CAST({uv_reference_sql} AS DOUBLE))
            THEN GREATEST(
                ABS(
                    LN(CAST({nominal_value_sql} AS DOUBLE) / CAST({export_quantity_sql} AS DOUBLE))
                    - LN(CAST({uv_reference_sql} AS DOUBLE))
                ),
                ABS(
                    LN(CAST({nominal_value_sql} AS DOUBLE) / CAST({import_quantity_sql} AS DOUBLE))
                    - LN(CAST({uv_reference_sql} AS DOUBLE))
                )
            )
            ELSE NULL
        END
            AS bilateral_max_unit_value_log_distance,
                    
                CASE
                    WHEN
                        {quantity_sql} IS NOT NULL
                        AND {quantity_sql} > 0

                        AND {nominal_value_sql} IS NOT NULL
                        AND {nominal_value_sql} > 0

                        AND {uv_reference_sql} IS NOT NULL
                        AND {uv_reference_sql} > 0

                        AND isfinite(
                            CAST(
                                {quantity_sql}
                                AS DOUBLE
                            )
                        )

                        AND isfinite(
                            CAST(
                                {nominal_value_sql}
                                AS DOUBLE
                            )
                        )

                        AND isfinite(
                            CAST(
                                {uv_reference_sql}
                                AS DOUBLE
                            )
                        )

                    THEN
                        ABS(
                            LN(
                                CAST(
                                    {nominal_value_sql}
                                    AS DOUBLE
                                )
                                /
                                CAST(
                                    {quantity_sql}
                                    AS DOUBLE
                                )
                            )
                            -
                            LN(
                                CAST(
                                    {uv_reference_sql}
                                    AS DOUBLE
                                )
                            )
                        )

                    ELSE NULL
                END
                    AS synced_unit_value_log_distance,

                quantity_selected_candidate,
                quantity_selected_source,

                candidate_simple_mean_uv_log_distance,
                candidate_value_weighted_uv_log_distance,
                candidate_quantity_reliability_weighted_uv_log_distance,

                candidate_simple_mean_export_log_gap,
                candidate_simple_mean_import_log_gap,
                candidate_value_weighted_export_log_gap,
                candidate_value_weighted_import_log_gap,
                candidate_quantity_reliability_weighted_export_log_gap,
                candidate_quantity_reliability_weighted_import_log_gap,

                candidate_simple_mean_conditioned_uv_abs_residual,
                candidate_value_weighted_conditioned_uv_abs_residual,
                candidate_quantity_reliability_weighted_conditioned_uv_abs_residual,

                quantity_conditioned_model_n_rows,
                quantity_conditioned_log_uv_intercept,
                quantity_conditioned_log_uv_slope,
                quantity_conditioned_log_uv_r2,

                selected_unit_value,
                training_reference_eligible,
                hs_year_uv_reference,
                hs_year_uv_reference_log,
                hs_year_uv_reference_n,
                hs_year_uv_log_mad,
                selected_uv_log_distance,
                training_hard_eligible,


            FROM read_parquet({input_sql})
            """
        )

        # ==================================================================
        # Quantity-candidate evidence supplied by selector
        # ==================================================================
        #
        # Candidate definitions, unconditional UV distances, reporter gaps,
        # and quantity-conditioned UV residuals are calculated once in
        # select_quantity_candidate.py. Diagnostics only reshapes that evidence
        # into the long form required by the existing reporting scopes.
        # ==================================================================

        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW
                validation_quantity_candidates
            AS

            SELECT
                *,
                'simple_mean' AS candidate,
                quantity_simple_mean_kg AS candidate_quantity_kg,
                candidate_simple_mean_uv_log_distance
                    AS candidate_uv_log_distance,
                candidate_simple_mean_export_log_gap
                    AS candidate_export_log_gap,
                candidate_simple_mean_import_log_gap
                    AS candidate_import_log_gap,
                candidate_simple_mean_conditioned_uv_abs_residual
                    AS candidate_conditioned_uv_abs_residual,
                (
                    quantity_simple_mean_kg IS NOT NULL
                    AND quantity_simple_mean_kg > 0
                    AND isfinite(quantity_simple_mean_kg)
                ) AS candidate_available
            FROM validation_diagnostics_base

            UNION ALL

            SELECT
                *,
                'value_weighted' AS candidate,
                quantity_value_weighted_kg AS candidate_quantity_kg,
                candidate_value_weighted_uv_log_distance
                    AS candidate_uv_log_distance,
                candidate_value_weighted_export_log_gap
                    AS candidate_export_log_gap,
                candidate_value_weighted_import_log_gap
                    AS candidate_import_log_gap,
                candidate_value_weighted_conditioned_uv_abs_residual
                    AS candidate_conditioned_uv_abs_residual,
                (
                    quantity_value_weighted_kg IS NOT NULL
                    AND quantity_value_weighted_kg > 0
                    AND isfinite(quantity_value_weighted_kg)
                ) AS candidate_available
            FROM validation_diagnostics_base

            UNION ALL

            SELECT
                *,
                'quantity_reliability_weighted' AS candidate,
                quantity_reliability_weighted_kg AS candidate_quantity_kg,
                candidate_quantity_reliability_weighted_uv_log_distance
                    AS candidate_uv_log_distance,
                candidate_quantity_reliability_weighted_export_log_gap
                    AS candidate_export_log_gap,
                candidate_quantity_reliability_weighted_import_log_gap
                    AS candidate_import_log_gap,
                candidate_quantity_reliability_weighted_conditioned_uv_abs_residual
                    AS candidate_conditioned_uv_abs_residual,
                (
                    quantity_reliability_weighted_kg IS NOT NULL
                    AND quantity_reliability_weighted_kg > 0
                    AND isfinite(quantity_reliability_weighted_kg)
                ) AS candidate_available
            FROM validation_diagnostics_base
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW
                validation_quantity_candidates_conditioned
            AS

            SELECT
                *,
                quantity_conditioned_model_n_rows AS n_model_rows,
                quantity_conditioned_log_uv_intercept AS log_uv_intercept,
                quantity_conditioned_log_uv_slope AS log_uv_slope,
                quantity_conditioned_log_uv_r2 AS log_uv_r2
            FROM validation_quantity_candidates
            """
        )

        # ==================================================================
        # 1. Quantity availability by reporting side
        # ==================================================================
        #
        # This is a mutually exclusive accounting of usable reporter-side
        # quantity information. Estimated/direct-report flags are deliberately
        # ignored here. The purpose is only to establish which reporting side
        # supplies a usable quantity.
        # ==================================================================

        quantity_availability = con.execute(
            """
            SELECT
                COUNT(*)
                    AS n_all_rows,

                COUNT(*) FILTER (
                    WHERE has_export_quantity
                    AND has_import_quantity
                )
                    AS n_both,

                COUNT(*) FILTER (
                    WHERE has_export_quantity
                    AND NOT has_import_quantity
                )
                    AS n_export_only,

                COUNT(*) FILTER (
                    WHERE NOT has_export_quantity
                    AND has_import_quantity
                )
                    AS n_import_only,

                COUNT(*) FILTER (
                    WHERE NOT has_export_quantity
                    AND NOT has_import_quantity
                )
                    AS n_neither

            FROM validation_diagnostics_base
            """
        ).fetchone()

        (
            n_all_rows,
            n_quantity_both,
            n_quantity_export_only,
            n_quantity_import_only,
            n_quantity_neither,
        ) = map(int, quantity_availability)

        diagnostics_rows: list[dict[str, Any]] = []

        selected_candidates = con.execute(
            """
            SELECT DISTINCT quantity_selected_candidate
            FROM validation_diagnostics_base
            WHERE quantity_selected_candidate IS NOT NULL
            """
        ).fetchall()

        if len(selected_candidates) != 1:
            raise RuntimeError(
                "Validation diagnostics require exactly one selected "
                "bilateral quantity methodology."
            )

        selected_candidate = str(selected_candidates[0][0])

        diagnostics_rows.append(
            {
                "scope": "quantity_selection",
                "criterion": "selected_bilateral_quantity_candidate",
                "candidate": selected_candidate,
                "disagreement_threshold": None,
                "uv_support_threshold": None,
                "uv_distance_threshold": None,
                "year": None,
                "target_cmd": None,
                "variable": "selected_candidate",
                "quantile": None,
                "n_rows": n_all_rows,
                "share_of_all": 1.0,
                "share_of_parent": None,
                "value": None,
            }
        )

        quantity_availability_specs = (
            (
                "both",
                n_quantity_both,
            ),
            (
                "export_only",
                n_quantity_export_only,
            ),
            (
                "import_only",
                n_quantity_import_only,
            ),
            (
                "neither",
                n_quantity_neither,
            ),
        )

        for criterion, n_rows in quantity_availability_specs:
            diagnostics_rows.append(
                {
                    "scope": "quantity_availability",
                    "criterion": criterion,
                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,
                    "year": None,
                    "target_cmd": None,
                    "variable": None,
                    "quantile": None,
                    "n_rows": int(n_rows),
                    "share_of_all": _share(
                        n_rows,
                        n_all_rows,
                    ),
                    "share_of_parent": None,
                    "value": None,
                }
            )

        # ==================================================================
        # 1. Population funnel
        # ==================================================================
        #
        # The funnel is based on reporter-side quantity evidence. The legacy
        # Trade-Synced quantity is deliberately not used as an eligibility
        # step because no final bilateral quantity has yet been selected.
        # ==================================================================

        funnel = con.execute(
            """
            SELECT
                COUNT(*)
                    AS n_all_rows,

                COUNT(*) FILTER (
                    WHERE has_both_quantities
                )
                    AS n_both_quantities,

                COUNT(*) FILTER (
                    WHERE both_quantities_observed
                )
                    AS n_both_observed,

                COUNT(*) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_relative_difference IS NOT NULL
                )
                    AS n_both_observed_with_disagreement,

                COUNT(*) FILTER (
                    WHERE both_quantities_observed
                      AND bilateral_max_unit_value_log_distance IS NOT NULL
                )
                    AS n_both_observed_with_uv_distance,

                COUNT(*) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_relative_difference IS NOT NULL
                      AND bilateral_max_unit_value_log_distance IS NOT NULL
                      AND unit_value_reference_observations IS NOT NULL
                )
                    AS n_complete_training_evidence

            FROM validation_diagnostics_base
            """
        ).fetchone()

        (
            n_funnel_all_rows,
            n_both_quantities,
            n_both_observed,
            n_both_observed_with_disagreement,
            n_both_observed_with_uv_distance,
            n_complete_training_evidence,
        ) = map(int, funnel)

        if n_funnel_all_rows != n_all_rows:
            raise RuntimeError(
                "Quantity-availability and population-funnel row totals differ."
            )
        
        funnel_specs = (
            ("all_rows", n_all_rows, n_all_rows),
            ("both_quantities_available", n_both_quantities, n_all_rows),
            ("both_quantities_non_estimated", n_both_observed, n_both_quantities),
            (
                "both_observed_with_disagreement",
                n_both_observed_with_disagreement,
                n_both_observed,
            ),
            (
                "both_observed_with_uv_distance",
                n_both_observed_with_uv_distance,
                n_both_observed,
            ),
            (
                "complete_training_evidence",
                n_complete_training_evidence,
                n_both_observed,
            ),
        )

        for criterion, n_rows, n_parent_rows in funnel_specs:
            diagnostics_rows.append(
                {
                    "scope": "population_funnel",
                    "criterion": criterion,
                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,
                    "year": None,
                    "target_cmd": None,
                    "variable": None,
                    "quantile": None,
                    "n_rows": int(n_rows),
                    "share_of_all": _share(n_rows, n_all_rows),
                    "share_of_parent": _share(n_rows, n_parent_rows),
                    "value": None,
                }
            )

        # ==================================================================
        # Training-sample preparation summary
        # ==================================================================

        training_sample_summary = con.execute(
            """
            SELECT
                COUNT(*)
                    AS n_all_rows,

                COUNT(*) FILTER (
                    WHERE selected_unit_value IS NOT NULL
                )
                    AS n_selected_unit_value,

                COUNT(*) FILTER (
                    WHERE training_reference_eligible
                )
                    AS n_reference_rows,

                COUNT(*) FILTER (
                    WHERE hs_year_uv_reference IS NOT NULL
                )
                    AS n_with_reference,

                COUNT(*) FILTER (
                    WHERE selected_uv_log_distance IS NOT NULL
                )
                    AS n_with_uv_distance,

                COUNT(*) FILTER (
                    WHERE training_hard_eligible
                )
                    AS n_hard_eligible

            FROM validation_diagnostics_base
            """
        ).fetchone()

        (
            n_training_all,
            n_training_selected_uv,
            n_training_reference,
            n_training_with_reference,
            n_training_with_uv_distance,
            n_training_hard_eligible,
        ) = map(
            int,
            training_sample_summary,
        )

        training_summary_specs = (
            (
                "all_rows",
                n_training_all,
                n_training_all,
            ),
            (
                "selected_unit_value_available",
                n_training_selected_uv,
                n_training_all,
            ),
            (
                "reference_population",
                n_training_reference,
                n_training_all,
            ),
            (
                "hs_year_reference_available",
                n_training_with_reference,
                n_training_all,
            ),
            (
                "uv_distance_available",
                n_training_with_uv_distance,
                n_training_all,
            ),
            (
                "hard_eligible",
                n_training_hard_eligible,
                n_training_all,
            ),
        )

        for criterion, n_rows, parent_rows in training_summary_specs:

            diagnostics_rows.append(
                {
                    "scope": "training_sample_summary",
                    "criterion": criterion,

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "year": None,
                    "target_cmd": None,

                    "variable": None,
                    "quantile": None,

                    "n_rows": n_rows,
                    "share_of_all": _share(
                        n_rows,
                        n_training_all,
                    ),
                    "share_of_parent": _share(
                        n_rows,
                        parent_rows,
                    ),

                    "value": None,
                }
            )

        # ==================================================================
        # Selected-quantity UV-distance distribution
        # ==================================================================

        training_quantiles = (
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.975,
            0.99,
            0.995,
        )

        for quantile in training_quantiles:

            value = con.execute(
                f"""
                SELECT
                    QUANTILE_CONT(
                        selected_uv_log_distance,
                        {float(quantile):.17g}
                    )
                FROM validation_diagnostics_base

                WHERE training_hard_eligible
                AND selected_uv_log_distance IS NOT NULL
                """
            ).fetchone()[0]

            diagnostics_rows.append(
                {
                    "scope": (
                        "training_uv_distance_distribution"
                    ),

                    "criterion": (
                        "training_hard_eligible"
                    ),

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "year": None,
                    "target_cmd": None,

                    "variable": (
                        "selected_uv_log_distance"
                    ),

                    "quantile": float(
                        quantile
                    ),

                    "n_rows": (
                        n_training_hard_eligible
                    ),

                    "share_of_all": None,
                    "share_of_parent": None,

                    "value": (
                        float(value)
                        if value is not None
                        else None
                    ),
                }
            )


        # ==================================================================
        # HS × year reference-support distribution
        # ==================================================================

        for quantile in training_quantiles:

            value = con.execute(
                f"""
                SELECT
                    QUANTILE_CONT(
                        hs_year_uv_reference_n,
                        {float(quantile):.17g}
                    )

                FROM (
                    SELECT DISTINCT
                        year,
                        target_cmd,
                        hs_year_uv_reference_n

                    FROM validation_diagnostics_base

                    WHERE hs_year_uv_reference_n IS NOT NULL
                )
                """
            ).fetchone()[0]

            n_groups = int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT DISTINCT
                            year,
                            target_cmd
                        FROM validation_diagnostics_base
                        WHERE hs_year_uv_reference_n IS NOT NULL
                    )
                    """
                ).fetchone()[0]
            )

            diagnostics_rows.append(
                {
                    "scope": (
                        "training_reference_support_distribution"
                    ),

                    "criterion": "hs_year_groups",

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "year": None,
                    "target_cmd": None,

                    "variable": (
                        "hs_year_uv_reference_n"
                    ),

                    "quantile": float(
                        quantile
                    ),

                    "n_rows": n_groups,

                    "share_of_all": None,
                    "share_of_parent": None,

                    "value": (
                        float(value)
                        if value is not None
                        else None
                    ),
                }
            )


        # ==================================================================
        # HS × year UV-dispersion distribution
        # ==================================================================

        for quantile in training_quantiles:

            value = con.execute(
                f"""
                SELECT
                    QUANTILE_CONT(
                        hs_year_uv_log_mad,
                        {float(quantile):.17g}
                    )

                FROM (
                    SELECT DISTINCT
                        year,
                        target_cmd,
                        hs_year_uv_log_mad

                    FROM validation_diagnostics_base

                    WHERE hs_year_uv_log_mad IS NOT NULL
                )
                """
            ).fetchone()[0]

            diagnostics_rows.append(
                {
                    "scope": (
                        "training_uv_dispersion_distribution"
                    ),

                    "criterion": "hs_year_groups",

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "year": None,
                    "target_cmd": None,

                    "variable": (
                        "hs_year_uv_log_mad"
                    ),

                    "quantile": float(
                        quantile
                    ),

                    "n_rows": n_groups,

                    "share_of_all": None,
                    "share_of_parent": None,

                    "value": (
                        float(value)
                        if value is not None
                        else None
                    ),
                }
            )

        # ==================================================================
        # Dispersion-normalized selected UV-distance distribution
        # ==================================================================
        #
        # This diagnostic scales each row's selected-unit-value log distance
        # by the robust within-HS × year log-UV dispersion:
        #
        #     normalized distance
        #         = selected UV log distance / HS × year log MAD
        #
        # No threshold is applied here.
        #
        # Groups with zero or missing MAD are excluded because the ratio is
        # undefined.
        # ==================================================================

        n_normalized_uv_rows = int(
            con.execute(
                """
                SELECT COUNT(*)

                FROM validation_diagnostics_base

                WHERE training_hard_eligible

                AND selected_uv_log_distance IS NOT NULL
                AND isfinite(selected_uv_log_distance)

                AND hs_year_uv_log_mad IS NOT NULL
                AND hs_year_uv_log_mad > 0
                AND isfinite(hs_year_uv_log_mad)
                """
            ).fetchone()[0]
        )


        for quantile in training_quantiles:

            value = con.execute(
                f"""
                SELECT
                    QUANTILE_CONT(
                        selected_uv_log_distance
                        / hs_year_uv_log_mad,
                        {float(quantile):.17g}
                    )

                FROM validation_diagnostics_base

                WHERE training_hard_eligible

                AND selected_uv_log_distance IS NOT NULL
                AND isfinite(selected_uv_log_distance)

                AND hs_year_uv_log_mad IS NOT NULL
                AND hs_year_uv_log_mad > 0
                AND isfinite(hs_year_uv_log_mad)
                """
            ).fetchone()[0]

            diagnostics_rows.append(
                {
                    "scope": (
                        "training_normalized_uv_distance_distribution"
                    ),

                    "criterion": (
                        "training_hard_eligible_with_positive_mad"
                    ),

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "year": None,
                    "target_cmd": None,

                    "variable": (
                        "selected_uv_log_distance_over_mad"
                    ),

                    "quantile": float(
                        quantile
                    ),

                    "n_rows": n_normalized_uv_rows,

                    "share_of_all": _share(
                        n_normalized_uv_rows,
                        n_training_all,
                    ),

                    "share_of_parent": _share(
                        n_normalized_uv_rows,
                        n_training_hard_eligible,
                    ),

                    "value": (
                        float(value)
                        if value is not None
                        else None
                    ),
                }
            )

        training_by_year = con.execute(
            """
            SELECT
                year,

                COUNT(*)
                    AS total_rows,

                COUNT(*) FILTER (
                    WHERE training_reference_eligible
                )
                    AS n_reference_rows,

                COUNT(*) FILTER (
                    WHERE training_hard_eligible
                )
                    AS n_hard_eligible_rows,

                MEDIAN(selected_uv_log_distance)
                    FILTER (
                        WHERE training_hard_eligible
                        AND selected_uv_log_distance IS NOT NULL
                    )
                    AS median_uv_distance,

                MEDIAN(hs_year_uv_reference_n)
                    FILTER (
                        WHERE hs_year_uv_reference_n IS NOT NULL
                    )
                    AS median_reference_support

            FROM validation_diagnostics_base

            GROUP BY year
            ORDER BY year
            """
        ).fetchdf()

        for row in training_by_year.itertuples(
            index=False
        ):
            diagnostics_rows.append(
                {
                    "scope": "training_sample_by_year",
                    "criterion": "year",

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "year": int(row.year),
                    "target_cmd": None,

                    "variable": (
                        "hard_eligible_share"
                    ),

                    "quantile": None,

                    "n_rows": int(
                        row.n_hard_eligible_rows
                    ),

                    "total_rows": int(
                        row.total_rows
                    ),

                    "share_of_all": _share(
                        int(row.n_hard_eligible_rows),
                        int(row.total_rows),
                    ),

                    "share_of_parent": None,

                    "value": (
                        float(row.median_uv_distance)
                        if row.median_uv_distance
                        is not None
                        else None
                    ),

                    "median_uv_reference_observations": (
                        float(
                            row.median_reference_support
                        )
                        if row.median_reference_support
                        is not None
                        else None
                    ),
                }
            )


        # ==================================================================
        # Normalized UV-distance cutoff sensitivity
        # ==================================================================
        #
        # These thresholds are exploratory diagnostics only.
        # They do not define the final training population.
        #
        # Two possible training bases are compared:
        #
        #   1. direct bilateral reference population
        #   2. all mechanically hard-eligible rows
        #
        # Shares are calculated relative to rows in the respective population
        # for which the normalized UV-distance is actually defined.
        # ==================================================================

        normalized_uv_cutoffs = (
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            4.0,
        )


        # ------------------------------------------------------------------
        # Evaluable populations
        # ------------------------------------------------------------------

        n_reference_normalized_evaluable = int(
            con.execute(
                """
                SELECT COUNT(*)

                FROM validation_diagnostics_base

                WHERE training_reference_eligible

                AND selected_uv_log_distance IS NOT NULL
                AND isfinite(selected_uv_log_distance)

                AND hs_year_uv_log_mad IS NOT NULL
                AND hs_year_uv_log_mad > 0
                AND isfinite(hs_year_uv_log_mad)
                """
            ).fetchone()[0]
        )


        n_hard_normalized_evaluable = int(
            con.execute(
                """
                SELECT COUNT(*)

                FROM validation_diagnostics_base

                WHERE training_hard_eligible

                AND selected_uv_log_distance IS NOT NULL
                AND isfinite(selected_uv_log_distance)

                AND hs_year_uv_log_mad IS NOT NULL
                AND hs_year_uv_log_mad > 0
                AND isfinite(hs_year_uv_log_mad)
                """
            ).fetchone()[0]
        )


        for cutoff in normalized_uv_cutoffs:

            # ==============================================================
            # Reference population
            # ==============================================================

            n_reference_selected = int(
                con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM validation_diagnostics_base

                    WHERE training_reference_eligible

                    AND selected_uv_log_distance IS NOT NULL
                    AND isfinite(selected_uv_log_distance)

                    AND hs_year_uv_log_mad IS NOT NULL
                    AND hs_year_uv_log_mad > 0
                    AND isfinite(hs_year_uv_log_mad)

                    AND (
                            selected_uv_log_distance
                            / hs_year_uv_log_mad
                        ) <= {float(cutoff):.17g}
                    """
                ).fetchone()[0]
            )

            diagnostics_rows.append(
                {
                    "scope": (
                        "training_normalized_uv_cutoff_sensitivity"
                    ),
                    "criterion": "reference_population",
                    "candidate": None,

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": float(cutoff),

                    "year": None,
                    "target_cmd": None,

                    "variable": (
                        "selected_uv_log_distance_over_mad"
                    ),
                    "quantile": None,

                    "n_rows": n_reference_selected,
                    "total_rows": (
                        n_reference_normalized_evaluable
                    ),

                    "share_of_all": _share(
                        n_reference_selected,
                        n_training_all,
                    ),

                    "share_of_parent": _share(
                        n_reference_selected,
                        n_reference_normalized_evaluable,
                    ),

                    "value": float(cutoff),
                }
            )

            # ==============================================================
            # Hard-eligible population
            # ==============================================================

            n_hard_selected = int(
                con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM validation_diagnostics_base

                    WHERE training_hard_eligible

                    AND selected_uv_log_distance IS NOT NULL
                    AND isfinite(selected_uv_log_distance)

                    AND hs_year_uv_log_mad IS NOT NULL
                    AND hs_year_uv_log_mad > 0
                    AND isfinite(hs_year_uv_log_mad)

                    AND (
                            selected_uv_log_distance
                            / hs_year_uv_log_mad
                        ) <= {float(cutoff):.17g}
                    """
                ).fetchone()[0]
            )

            diagnostics_rows.append(
                {
                    "scope": (
                        "training_normalized_uv_cutoff_sensitivity"
                    ),

                    "criterion": (
                        "hard_eligible_population"
                    ),

                    "candidate": None,

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": float(
                        cutoff
                    ),

                    "year": None,
                    "target_cmd": None,

                    "variable": (
                        "selected_uv_log_distance_over_mad"
                    ),

                    "quantile": None,

                    "n_rows": n_hard_selected,

                    "total_rows": (
                        n_hard_normalized_evaluable
                    ),

                    "share_of_all": _share(
                        n_hard_selected,
                        n_training_all,
                    ),

                    "share_of_parent": _share(
                        n_hard_selected,
                        n_hard_normalized_evaluable,
                    ),

                    "value": float(
                        cutoff
                    ),
                }
            )
        

        # ==================================================================
        # Candidate final training population
        # ==================================================================
        #
        # This is still diagnostic evidence only.
        # It applies the currently preferred joint rule:
        #
        #   direct bilateral reference population
        #   + minimum HS × year support of 30
        #   + normalized UV distance <= 2.5 MAD
        #
        # ==================================================================

        n_candidate_training_selected = int(
            con.execute(
                """
                SELECT COUNT(*)

                FROM validation_diagnostics_base

                WHERE training_reference_eligible

                AND hs_year_uv_reference_n IS NOT NULL
                AND hs_year_uv_reference_n >= 30

                AND selected_uv_log_distance IS NOT NULL
                AND isfinite(selected_uv_log_distance)

                AND hs_year_uv_log_mad IS NOT NULL
                AND hs_year_uv_log_mad > 0
                AND isfinite(hs_year_uv_log_mad)

                AND (
                        selected_uv_log_distance
                        / hs_year_uv_log_mad
                    ) <= 2.5
                """
            ).fetchone()[0]
        )


        diagnostics_rows.append(
            {
                "scope": "training_candidate_final_rule",

                "criterion": (
                    "reference_population_support_30_normalized_uv_2_5"
                ),

                "candidate": None,

                "disagreement_threshold": None,
                "uv_support_threshold": 30.0,
                "uv_distance_threshold": 2.5,

                "year": None,
                "target_cmd": None,

                "variable": (
                    "selected_uv_log_distance_over_mad"
                ),

                "quantile": None,

                "n_rows": n_candidate_training_selected,

                "total_rows": (
                    n_reference_normalized_evaluable
                ),

                "share_of_all": _share(
                    n_candidate_training_selected,
                    n_training_all,
                ),

                "share_of_parent": _share(
                    n_candidate_training_selected,
                    n_reference_normalized_evaluable,
                ),

                "value": 2.5,
            }
        )
        
        training_by_product = con.execute(
            """
            SELECT
                target_cmd,

                COUNT(*)
                    AS total_rows,

                COUNT(*) FILTER (
                    WHERE training_hard_eligible
                )
                    AS n_hard_eligible,

                MEDIAN(selected_uv_log_distance)
                    FILTER (
                        WHERE training_hard_eligible
                        AND selected_uv_log_distance IS NOT NULL
                    )
                    AS median_uv_distance,

                QUANTILE_CONT(
                    selected_uv_log_distance,
                    0.95
                )
                    FILTER (
                        WHERE training_hard_eligible
                        AND selected_uv_log_distance IS NOT NULL
                    )
                    AS p95_uv_distance,

                MEDIAN(hs_year_uv_log_mad)
                    FILTER (
                        WHERE hs_year_uv_log_mad IS NOT NULL
                    )
                    AS median_uv_log_mad

            FROM validation_diagnostics_base

            GROUP BY target_cmd
            """
        ).fetchdf()

        # ==================================================================
        # 2. Individual threshold sensitivity
        # ==================================================================

        threshold_expressions: list[str] = []
        threshold_metadata: list[
            tuple[str, float | int, str]
        ] = []

        # --------------------------------------------------------------
        # Mirror disagreement
        # --------------------------------------------------------------

        for index, threshold in enumerate(
            disagreement_thresholds
        ):
            alias = (
                f"disagreement_{index}"
            )

            threshold_expressions.append(
                f"""
                COUNT(*) FILTER (
                    WHERE both_quantities_observed

                      AND quantity_relative_difference
                          IS NOT NULL

                      AND quantity_relative_difference
                          <= {float(threshold):.17g}
                ) AS {alias}
                """
            )

            threshold_metadata.append(
                (
                    "quantity_relative_difference",
                    float(threshold),
                    alias,
                )
            )

        # --------------------------------------------------------------
        # Product-year unit-value support
        # --------------------------------------------------------------

        for index, threshold in enumerate(
            uv_support_thresholds
        ):
            alias = (
                f"uv_support_{index}"
            )

            threshold_expressions.append(
                f"""
                COUNT(*) FILTER (
                    WHERE both_quantities_observed

                      AND unit_value_reference_observations
                          IS NOT NULL

                      AND unit_value_reference_observations
                          >= {int(threshold)}
                ) AS {alias}
                """
            )

            threshold_metadata.append(
                (
                    "unit_value_reference_observations",
                    int(threshold),
                    alias,
                )
            )

        # --------------------------------------------------------------
        # Worst-side reporter unit-value distance
        # --------------------------------------------------------------

        for index, threshold in enumerate(
            uv_distance_thresholds
        ):
            alias = (
                f"uv_distance_{index}"
            )

            threshold_expressions.append(
                f"""
                COUNT(*) FILTER (
                    WHERE both_quantities_observed

                      AND bilateral_max_unit_value_log_distance
                          IS NOT NULL

                      AND bilateral_max_unit_value_log_distance
                          <= {float(threshold):.17g}
                ) AS {alias}
                """
            )

            threshold_metadata.append(
                (
                    "bilateral_max_unit_value_log_distance",
                    float(threshold),
                    alias,
                )
            )

        sensitivity_row = con.execute(
            f"""
            SELECT
                {", ".join(threshold_expressions)}
            FROM validation_diagnostics_base
            """
        ).fetchone()

        for (
            metadata,
            value,
        ) in zip(
            threshold_metadata,
            sensitivity_row,):
            (
                variable,
                threshold,
                _,
            ) = metadata

            n_rows = int(value)

            diagnostics_rows.append(
                {
                    "scope": "threshold_sensitivity",
                    "criterion": variable,

                    "disagreement_threshold": (
                        float(threshold)
                        if variable
                        == "quantity_relative_difference"
                        else None
                    ),

                    "uv_support_threshold": (
                        float(threshold)
                        if variable
                        == "unit_value_reference_observations"
                        else None
                    ),

                    "uv_distance_threshold": (
                        float(threshold)
                        if variable
                        == "bilateral_max_unit_value_log_distance"
                        else None
                    ),

                    "year": None,
                    "target_cmd": None,

                    "variable": variable,
                    "quantile": None,

                    "n_rows": n_rows,

                    "share_of_all": _share(
                        n_rows,
                        n_all_rows,
                    ),

                    "share_of_parent": _share(
                        n_rows,
                        n_both_observed,
                    ),

                    "value": None,
                }
            )

        # ==================================================================
        # 3. Full candidate-training threshold grid
        # ==================================================================
        #
        # Every configured combination is evaluated.
        #
        # Nothing here is declared to be the final training population.
        # That decision is made only after diagnostics/reporting review.
        # ==================================================================

        grid_expressions: list[str] = []

        grid_metadata: list[
            tuple[
                float,
                int,
                float,
                str,
            ]
        ] = []

        grid_index = 0

        for disagreement in disagreement_thresholds:

            for support in uv_support_thresholds:

                for uv_distance in uv_distance_thresholds:

                    alias = (
                        f"training_grid_{grid_index}"
                    )

                    grid_expressions.append(
                        f"""
                        COUNT(*) FILTER (
                            WHERE both_quantities_observed

                              AND quantity_relative_difference
                                  IS NOT NULL

                              AND quantity_relative_difference
                                  <= {float(disagreement):.17g}

                              AND unit_value_reference_observations
                                  IS NOT NULL

                              AND unit_value_reference_observations
                                  >= {int(support)}

                              AND bilateral_max_unit_value_log_distance
                                  IS NOT NULL

                              AND bilateral_max_unit_value_log_distance
                                  <= {float(uv_distance):.17g}
                        ) AS {alias}
                        """
                    )

                    grid_metadata.append(
                        (
                            float(disagreement),
                            int(support),
                            float(uv_distance),
                            alias,
                        )
                    )

                    grid_index += 1

        grid_row = con.execute(
            f"""
            SELECT
                {", ".join(grid_expressions)}
            FROM validation_diagnostics_base
            """
        ).fetchone()

        for (
            metadata,
            value,
        ) in zip(
            grid_metadata,
            grid_row,
        ):
            (
                disagreement,
                support,
                uv_distance,
                _,
            ) = metadata

            n_rows = int(value)

            diagnostics_rows.append(
                {
                    "scope": "threshold_grid",

                    "criterion": (
                        "candidate_training_population"
                    ),

                    "disagreement_threshold": (
                        disagreement
                    ),

                    "uv_support_threshold": float(
                        support
                    ),

                    "uv_distance_threshold": (
                        uv_distance
                    ),

                    "year": None,
                    "target_cmd": None,

                    "variable": None,
                    "quantile": None,

                    "n_rows": n_rows,

                    "share_of_all": _share(
                        n_rows,
                        n_all_rows,
                    ),

                    "share_of_parent": _share(
                        n_rows,
                        n_both_observed,
                    ),

                    "value": None,
                }
            )

        # ==================================================================
        # 4. Mirror-agreement × bilateral unit-value plausibility matrix
        # ==================================================================
        #
        # Unlike the cumulative threshold grid above, this diagnostic uses
        # mutually exclusive bands.
        #
        # Unit-value plausibility is defined using the WORSE of the exporter
        # and importer distances. Therefore, a row can enter a low-distance
        # band only when both reported quantities independently imply
        # plausible unit values.
        # ==================================================================

        disagreement_edges = tuple(
            float(value)
            for value in disagreement_thresholds
        )

        uv_edges = tuple(
            float(value)
            for value in uv_distance_thresholds
        )

        def _band_case(
            *,
            column: str,
            edges: tuple[float, ...],
            prefix: str,
        ) -> str:

            clauses: list[str] = []

            lower = 0.0

            for index, upper in enumerate(
                edges
            ):
                if index == 0:
                    label = (
                        f"{100 * upper:.0f}% or less"
                        if prefix == "mirror"
                        else f"{upper:g} or less"
                    )

                    clauses.append(
                        f"""
                        WHEN {column} <= {upper:.17g}
                            THEN '{label}'
                        """
                    )

                else:
                    label = (
                        f"{100 * lower:.0f}–{100 * upper:.0f}%"
                        if prefix == "mirror"
                        else f"{lower:g}–{upper:g}"
                    )

                    clauses.append(
                        f"""
                        WHEN {column} <= {upper:.17g}
                            THEN '{label}'
                        """
                    )

                lower = upper

            final_label = (
                f"> {100 * edges[-1]:.0f}%"
                if prefix == "mirror"
                else f"> {edges[-1]:g}"
            )

            return (
                "CASE "
                + " ".join(clauses)
                + f" ELSE '{final_label}' END"
            )

        disagreement_band_sql = _band_case(
            column="quantity_relative_difference",
            edges=disagreement_edges,
            prefix="mirror",
        )

        bilateral_uv_distance_sql = """
            GREATEST(
                export_unit_value_log_distance,
                import_unit_value_log_distance
            )
        """

        uv_band_sql = _band_case(
            column=bilateral_uv_distance_sql,
            edges=uv_edges,
            prefix="uv",
        )

        evidence_matrix = con.execute(
            f"""
            WITH evidence AS (
                SELECT
                    {disagreement_band_sql}
                        AS disagreement_band,

                    {uv_band_sql}
                        AS uv_distance_band

                FROM validation_diagnostics_base

                WHERE both_quantities_observed

                  AND quantity_relative_difference
                      IS NOT NULL

                  AND export_unit_value_log_distance
                      IS NOT NULL

                  AND import_unit_value_log_distance
                      IS NOT NULL
            )

            SELECT
                disagreement_band,
                uv_distance_band,
                COUNT(*) AS n_rows

            FROM evidence

            GROUP BY
                disagreement_band,
                uv_distance_band
            """
        ).fetchdf()

        n_matrix_rows = int(
            evidence_matrix["n_rows"].sum()
        )

        for row in evidence_matrix.itertuples(
            index=False
        ):
            diagnostics_rows.append(
                {
                    "scope": "evidence_matrix",

                    "criterion": (
                        "mirror_agreement_x_bilateral_uv_plausibility"
                    ),

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "disagreement_band": (
                        row.disagreement_band
                    ),

                    "uv_distance_band": (
                        row.uv_distance_band
                    ),

                    "year": None,
                    "target_cmd": None,

                    "variable": None,
                    "quantile": None,

                    "n_rows": int(
                        row.n_rows
                    ),

                    "share_of_all": _share(
                        int(row.n_rows),
                        n_all_rows,
                    ),

                    "share_of_parent": _share(
                        int(row.n_rows),
                        n_matrix_rows,
                    ),

                    "value": None,
                }
            )


        # ==================================================================
        # 5. Side-specific plausibility by mirror-disagreement band
        # ==================================================================
        uv_strict = float(uv_distance_thresholds[0])
        uv_moderate = float(uv_distance_thresholds[1])
        uv_loose = float(uv_distance_thresholds[2])

        side_plausibility = con.execute(
            f"""
            WITH evidence AS (
                SELECT
                    {disagreement_band_sql}
                        AS disagreement_band,

                    export_unit_value_log_distance
                        AS export_distance,

                    import_unit_value_log_distance
                        AS import_distance,

                    GREATEST(
                        export_unit_value_log_distance,
                        import_unit_value_log_distance
                    )
                        AS bilateral_max_distance

                FROM validation_diagnostics_base

                WHERE both_quantities_observed

                  AND quantity_relative_difference
                      IS NOT NULL

                  AND export_unit_value_log_distance
                      IS NOT NULL

                  AND import_unit_value_log_distance
                      IS NOT NULL
            )

            SELECT
                disagreement_band,

                COUNT(*) AS n_rows,

                MEDIAN(
                    export_distance
                ) AS median_export_distance,

                MEDIAN(
                    import_distance
                ) AS median_import_distance,

                MEDIAN(
                    bilateral_max_distance
                ) AS median_bilateral_max_distance,

                AVG(
                    CASE
                        WHEN bilateral_max_distance <= {uv_strict:.17g}
                            THEN 1.0
                        ELSE 0.0
                    END
                ) AS both_within_025_share,

                AVG(
                    CASE
                        WHEN bilateral_max_distance <= {uv_moderate:.17g}
                            THEN 1.0
                        ELSE 0.0
                    END
                ) AS both_within_050_share,

                AVG(
                    CASE
                        WHEN bilateral_max_distance <= {uv_loose:.17g}
                            THEN 1.0
                        ELSE 0.0
                    END
                ) AS both_within_100_share

            FROM evidence

            GROUP BY disagreement_band
            """
        ).fetchdf()

        for row in side_plausibility.itertuples(
            index=False
        ):

            metrics = {
                "median_export_uv_distance": (
                    row.median_export_distance
                ),

                "median_import_uv_distance": (
                    row.median_import_distance
                ),

                "median_bilateral_max_uv_distance": (
                    row.median_bilateral_max_distance
                ),

                "both_within_025_share": (
                    row.both_within_025_share
                ),

                "both_within_050_share": (
                    row.both_within_050_share
                ),

                "both_within_100_share": (
                    row.both_within_100_share
                ),
            }

            for metric, value in metrics.items():

                diagnostics_rows.append(
                    {
                        "scope": "side_plausibility",

                        "criterion": (
                            row.disagreement_band
                        ),

                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,

                        "disagreement_band": (
                            row.disagreement_band
                        ),

                        "uv_distance_band": None,

                        "year": None,
                        "target_cmd": None,

                        "variable": metric,
                        "quantile": None,

                        "n_rows": int(
                            row.n_rows
                        ),

                        "share_of_all": None,
                        "share_of_parent": None,

                        "value": (
                            float(value)
                            if value is not None
                            else None
                        ),
                    }
                )


        # ==================================================================
        # 6a. Exporter quantity relative to importer reference
        # ==================================================================
        #
        # Import quantity is used as the reference denominator.
        #
        # Signed difference:
        #
        #     (Q_export - Q_import) / Q_import
        #
        # Positive  -> exporter quantity is larger.
        # Negative  -> exporter quantity is smaller.
        # Zero      -> exact agreement.
        #
        # The absolute version measures only the magnitude of disagreement.
        # ==================================================================

        importer_reference = con.execute(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS n_comparable_rows,

                AVG(
                    quantity_difference_vs_import
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS mean_signed_difference,

                MEDIAN(
                    quantity_difference_vs_import
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS median_signed_difference,

                MEDIAN(
                    quantity_absolute_difference_vs_import
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_absolute_difference_vs_import IS NOT NULL
                )
                    AS median_absolute_difference,

                AVG(
                    CASE
                        WHEN quantity_difference_vs_import > 0
                            THEN 1.0
                        ELSE 0.0
                    END
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS exporter_higher_share,

                AVG(
                    CASE
                        WHEN quantity_difference_vs_import < 0
                            THEN 1.0
                        ELSE 0.0
                    END
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS importer_higher_share,

                AVG(
                    CASE
                        WHEN quantity_difference_vs_import = 0
                            THEN 1.0
                        ELSE 0.0
                    END
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS exact_agreement_share

            FROM validation_diagnostics_base
            """
        ).fetchone()

        (
            n_import_reference_rows,
            mean_signed_difference,
            median_signed_difference,
            median_absolute_difference,
            exporter_higher_share,
            importer_higher_share,
            exact_agreement_share,
        ) = importer_reference

        importer_reference_metrics = {
            "mean_quantity_difference_vs_import": (
                mean_signed_difference
            ),
            "median_quantity_difference_vs_import": (
                median_signed_difference
            ),
            "median_absolute_quantity_difference_vs_import": (
                median_absolute_difference
            ),
            "exporter_quantity_higher_share": (
                exporter_higher_share
            ),
            "importer_quantity_higher_share": (
                importer_higher_share
            ),
            "exact_quantity_agreement_share": (
                exact_agreement_share
            ),
        }

        for metric, value in importer_reference_metrics.items():

            diagnostics_rows.append(
                {
                    "scope": "importer_reference",

                    "criterion": (
                        "exporter_quantity_relative_to_importer"
                    ),

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "disagreement_band": None,
                    "uv_distance_band": None,

                    "year": None,
                    "target_cmd": None,

                    "variable": metric,
                    "quantile": None,

                    "n_rows": int(
                        n_import_reference_rows
                    ),

                    "share_of_all": _share(
                        int(n_import_reference_rows),
                        n_all_rows,
                    ),

                    "share_of_parent": _share(
                        int(n_import_reference_rows),
                        n_both_observed,
                    ),

                    "value": (
                        float(value)
                        if value is not None
                        else None
                    ),
                }
            )

        # ==================================================================
        # 6b. Exporter / importer quantity-ratio distribution
        # ==================================================================

        quantity_ratio_distribution = con.execute(
            """
            WITH ratios AS (
                SELECT
                    LN(
                        export_quantity_kg
                        / import_quantity_kg
                    ) AS log_ratio

                FROM validation_diagnostics_base

                WHERE both_quantities_observed
                  AND export_quantity_kg > 0
                  AND import_quantity_kg > 0
                  AND isfinite(
                      LN(
                          export_quantity_kg
                          / import_quantity_kg
                      )
                  )
            ),

            bounds AS (
                SELECT
                    QUANTILE_CONT(
                        log_ratio,
                        0.005
                    ) AS lower_bound,

                    QUANTILE_CONT(
                        log_ratio,
                        0.995
                    ) AS upper_bound

                FROM ratios
            ),

            displayed AS (
                SELECT
                    log_ratio

                FROM ratios
                CROSS JOIN bounds

                WHERE log_ratio >= lower_bound
                  AND log_ratio <= upper_bound
            ),

            binned AS (
                SELECT
                    FLOOR(log_ratio / 0.10) * 0.10
                        AS log_ratio_bin,

                    COUNT(*) AS n_rows

                FROM displayed

                GROUP BY log_ratio_bin
            )

            SELECT
                log_ratio_bin,
                n_rows

            FROM binned

            ORDER BY log_ratio_bin
            """
        ).fetchdf()

        n_quantity_ratio_rows = int(
            quantity_ratio_distribution[
                "n_rows"
            ].sum()
        )

        for row in quantity_ratio_distribution.itertuples(
            index=False
        ):

            diagnostics_rows.append(
                {
                    "scope": "quantity_ratio_distribution",

                    "criterion": (
                        "direct_bilateral_quantity_observations"
                    ),

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "disagreement_band": None,
                    "uv_distance_band": None,

                    "year": None,
                    "target_cmd": None,

                    "variable": (
                        "export_import_log_quantity_ratio"
                    ),

                    "quantile": None,

                    "n_rows": int(
                        row.n_rows
                    ),

                    "share_of_all": _share(
                        int(row.n_rows),
                        n_all_rows,
                    ),

                    "share_of_parent": _share(
                        int(row.n_rows),
                        n_quantity_ratio_rows,
                    ),

                    "value": float(
                        row.log_ratio_bin
                    ),
                }
            )


        # ==================================================================
        # Quantity–UV relationship
        # ==================================================================
        #
        # Explore whether unit value systematically varies with trade
        # quantity. This is diagnostic only and does not currently affect
        # candidate-training selection.
        #
        # A symmetric bilateral quantity proxy is used so that neither
        # exporter nor importer quantity is treated as the true quantity.
        # ==================================================================

        quantity_uv_relationship = con.execute(
            """
            SELECT
                year,
                target_cmd,

                COUNT(*) AS n_rows,

                REGR_SLOPE(
                    log_bilateral_unit_value,
                    log_bilateral_quantity
                ) AS log_uv_quantity_slope,

                REGR_R2(
                    log_bilateral_unit_value,
                    log_bilateral_quantity
                ) AS log_uv_quantity_r2,

                CORR(
                    log_bilateral_unit_value,
                    log_bilateral_quantity
                ) AS log_uv_quantity_correlation

            FROM validation_diagnostics_base

            WHERE both_quantities_observed
              AND log_bilateral_quantity IS NOT NULL
              AND log_bilateral_unit_value IS NOT NULL

            GROUP BY
                year,
                target_cmd

            ORDER BY
                year,
                target_cmd
            """
        ).fetchdf()

        for row in quantity_uv_relationship.itertuples(
            index=False
        ):

            metrics = {
                "log_uv_quantity_slope": (
                    row.log_uv_quantity_slope
                ),
                "log_uv_quantity_r2": (
                    row.log_uv_quantity_r2
                ),
                "log_uv_quantity_correlation": (
                    row.log_uv_quantity_correlation
                ),
            }

            for variable, value in metrics.items():

                diagnostics_rows.append(
                    {
                        "scope": "quantity_uv_relationship",

                        "criterion": (
                            "direct_bilateral_quantity_observations"
                        ),

                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,

                        "disagreement_band": None,
                        "uv_distance_band": None,

                        "year": int(
                            row.year
                        ),

                        "target_cmd": str(
                            row.target_cmd
                        ),

                        "variable": variable,
                        "quantile": None,

                        "n_rows": int(
                            row.n_rows
                        ),

                        "share_of_all": None,
                        "share_of_parent": None,

                        "value": (
                            float(value)
                            if value is not None
                            and pd.notna(value)
                            else None
                        ),
                    }
                )

        quantity_uv_summary = con.execute(
            """
            WITH relationship AS (
                SELECT
                    year,
                    target_cmd,

                    COUNT(*) AS n_rows,

                    REGR_SLOPE(
                        log_bilateral_unit_value,
                        log_bilateral_quantity
                    ) AS slope,

                    REGR_R2(
                        log_bilateral_unit_value,
                        log_bilateral_quantity
                    ) AS r2

                FROM validation_diagnostics_base

                WHERE both_quantities_observed
                  AND log_bilateral_quantity IS NOT NULL
                  AND log_bilateral_unit_value IS NOT NULL

                GROUP BY
                    year,
                    target_cmd
            )

            SELECT
                COUNT(*) AS n_product_year_groups,

                MEDIAN(slope)
                    AS median_slope,

                QUANTILE_CONT(slope, 0.25)
                    AS p25_slope,

                QUANTILE_CONT(slope, 0.75)
                    AS p75_slope,

                AVG(
                    CASE
                        WHEN slope < 0 THEN 1.0
                        ELSE 0.0
                    END
                )
                    AS negative_slope_share,

                MEDIAN(r2)
                    AS median_r2

            FROM relationship

            WHERE n_rows >= 30
              AND slope IS NOT NULL
            """
        ).fetchone()

        (
            n_quantity_uv_groups,
            median_quantity_uv_slope,
            p25_quantity_uv_slope,
            p75_quantity_uv_slope,
            negative_quantity_uv_share,
            median_quantity_uv_r2,
        ) = quantity_uv_summary

        summary_metrics = {
            "n_product_year_groups": (
                n_quantity_uv_groups
            ),
            "median_log_uv_quantity_slope": (
                median_quantity_uv_slope
            ),
            "p25_log_uv_quantity_slope": (
                p25_quantity_uv_slope
            ),
            "p75_log_uv_quantity_slope": (
                p75_quantity_uv_slope
            ),
            "negative_slope_share": (
                negative_quantity_uv_share
            ),
            "median_log_uv_quantity_r2": (
                median_quantity_uv_r2
            ),
        }

        for variable, value in summary_metrics.items():

            diagnostics_rows.append(
                {
                    "scope": "quantity_uv_summary",
                    "criterion": (
                        "product_year_groups_with_at_least_30_rows"
                    ),

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "disagreement_band": None,
                    "uv_distance_band": None,

                    "year": None,
                    "target_cmd": None,

                    "variable": variable,
                    "quantile": None,

                    "n_rows": int(
                        n_quantity_uv_groups
                    ),

                    "share_of_all": None,
                    "share_of_parent": None,

                    "value": (
                        float(value)
                        if value is not None
                        else None
                    ),
                }
            )

        supported_quantity_uv = quantity_uv_relationship[
            quantity_uv_relationship["n_rows"] >= 30
        ].copy()

        for row in supported_quantity_uv.itertuples(
            index=False):

            diagnostics_rows.append(
                {
                    "scope": "quantity_uv_relationship_groups",
                    "criterion": "hs_year_group_with_min_30_rows",

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "disagreement_band": None,
                    "uv_distance_band": None,

                    "year": int(
                        row.year
                    ),

                    "target_cmd": str(
                        row.target_cmd
                    ),

                    "variable": "quantity_uv_slope",
                    "quantile": None,

                    "n_rows": int(
                        row.n_rows
                    ),

                    "share_of_all": None,
                    "share_of_parent": None,

                    "value": (
                        float(
                            row.log_uv_quantity_slope
                        )
                        if pd.notna(
                            row.log_uv_quantity_slope
                        )
                        else None
                    ),
                }
            )

            diagnostics_rows.append(
                {
                    "scope": "quantity_uv_relationship_groups",
                    "criterion": "hs_year_group_with_min_30_rows",

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "disagreement_band": None,
                    "uv_distance_band": None,

                    "year": int(
                        row.year
                    ),

                    "target_cmd": str(
                        row.target_cmd
                    ),

                    "variable": "quantity_uv_r_squared",
                    "quantile": None,

                    "n_rows": int(
                        row.n_rows
                    ),

                    "share_of_all": None,
                    "share_of_parent": None,

                    "value": (
                        float(
                            row.log_uv_quantity_r2
                        )
                        if pd.notna(
                            row.log_uv_quantity_r2
                        )
                        else None
                    ),
                }
            )
        # ==================================================================
        # 7. Year coverage
        # ==================================================================

        year_frame = con.execute(
            """
            SELECT
                year,

                COUNT(*)
                    AS n_rows,

                COUNT(*) FILTER (
                    WHERE has_both_quantities
                )
                    AS n_both_quantities,

                COUNT(*) FILTER (
                    WHERE both_quantities_observed
                )
                    AS n_both_observed,

                MEDIAN(
                    quantity_relative_difference
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_relative_difference
                          IS NOT NULL
                )
                    AS median_quantity_relative_difference,

                MEDIAN(
                    bilateral_max_unit_value_log_distance
                ) FILTER (
                    WHERE both_quantities_observed
                      AND bilateral_max_unit_value_log_distance
                          IS NOT NULL
                )
                    AS median_worst_side_uv_log_distance,

                MEDIAN(
                    unit_value_reference_observations
                ) FILTER (
                    WHERE both_quantities_observed
                      AND unit_value_reference_observations
                          IS NOT NULL
                )
                    AS median_uv_reference_observations,

                MEDIAN(
                    quantity_difference_vs_import
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS median_quantity_difference_vs_import,

                MEDIAN(
                    quantity_absolute_difference_vs_import
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_absolute_difference_vs_import IS NOT NULL
                )
                    AS median_absolute_quantity_difference_vs_import,

                AVG(
                    CASE
                        WHEN quantity_difference_vs_import > 0
                            THEN 1.0
                        ELSE 0.0
                    END
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_difference_vs_import IS NOT NULL
                )
                    AS exporter_higher_share

            FROM validation_diagnostics_base

            GROUP BY year
            ORDER BY year
            """
        ).fetchdf()

        for row in year_frame.itertuples(
            index=False
        ):
            diagnostics_rows.append(
                {
                    "scope": "year_coverage",

                    "criterion": (
                        "both_quantities_non_estimated"
                    ),

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "year": int(
                        row.year
                    ),

                    "target_cmd": None,

                    "variable": None,
                    "quantile": None,

                    "n_rows": int(
                        row.n_both_observed
                    ),

                    "share_of_all": _share(
                        int(row.n_both_observed),
                        int(row.n_rows),
                    ),

                    "share_of_parent": None,

                    "value": None,

                    "total_rows": int(
                        row.n_rows
                    ),

                    "n_both_quantities": int(
                        row.n_both_quantities
                    ),

                    "median_quantity_relative_difference": (
                        float(
                            row.median_quantity_relative_difference
                        )
                        if pd.notna(
                            row.median_quantity_relative_difference
                        )
                        else None
                    ),

                    "median_worst_side_uv_log_distance": (
                        float(
                            row.median_worst_side_uv_log_distance
                        )
                        if pd.notna(
                            row.median_worst_side_uv_log_distance
                        )
                        else None
                    ),

                    "median_quantity_difference_vs_import": (
                        float(
                            row.median_quantity_difference_vs_import
                        )
                        if pd.notna(
                            row.median_quantity_difference_vs_import
                        )
                        else None
                    ),

                    "median_absolute_quantity_difference_vs_import": (
                        float(
                            row.median_absolute_quantity_difference_vs_import
                        )
                        if pd.notna(
                            row.median_absolute_quantity_difference_vs_import
                        )
                        else None
                    ),

                    "exporter_higher_share": (
                        float(
                            row.exporter_higher_share
                        )
                        if pd.notna(
                            row.exporter_higher_share
                        )
                        else None
                    ),
                    "median_uv_reference_observations": (
                        float(
                            row.median_uv_reference_observations
                        )
                        if pd.notna(
                            row.median_uv_reference_observations
                        )
                        else None
                    ),
                }
            )

        # ==================================================================
        # 8. Product coverage
        # ==================================================================

        product_frame = con.execute(
            """
            SELECT
                target_cmd,

                COUNT(*)
                    AS n_rows,

                COUNT(*) FILTER (
                    WHERE has_both_quantities
                )
                    AS n_both_quantities,

                COUNT(*) FILTER (
                    WHERE both_quantities_observed
                )
                    AS n_both_observed,

                MEDIAN(
                    quantity_relative_difference
                ) FILTER (
                    WHERE both_quantities_observed
                      AND quantity_relative_difference
                          IS NOT NULL
                )
                    AS median_quantity_relative_difference,

                MEDIAN(
                    bilateral_max_unit_value_log_distance
                ) FILTER (
                    WHERE both_quantities_observed
                      AND bilateral_max_unit_value_log_distance
                          IS NOT NULL
                )
                    AS median_worst_side_uv_log_distance,

                MEDIAN(
                    unit_value_reference_observations
                ) FILTER (
                    WHERE both_quantities_observed
                      AND unit_value_reference_observations
                          IS NOT NULL
                )
                    AS median_uv_reference_observations

            FROM validation_diagnostics_base

            GROUP BY target_cmd
            ORDER BY target_cmd
            """
        ).fetchdf()

        for row in product_frame.itertuples(
            index=False
        ):
            diagnostics_rows.append(
                {
                    "scope": "product_coverage",

                    "criterion": (
                        "both_quantities_non_estimated"
                    ),

                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,

                    "year": None,

                    "target_cmd": str(
                        row.target_cmd
                    ),

                    "variable": None,
                    "quantile": None,

                    "n_rows": int(
                        row.n_both_observed
                    ),

                    "share_of_all": _share(
                        int(row.n_both_observed),
                        int(row.n_rows),
                    ),

                    "share_of_parent": None,

                    "value": None,

                    "total_rows": int(
                        row.n_rows
                    ),

                    "n_both_quantities": int(
                        row.n_both_quantities
                    ),

                    "median_quantity_relative_difference": (
                        float(
                            row.median_quantity_relative_difference
                        )
                        if pd.notna(
                            row.median_quantity_relative_difference
                        )
                        else None
                    ),

                    "median_worst_side_uv_log_distance": (
                        float(
                            row.median_worst_side_uv_log_distance
                        )
                        if pd.notna(
                            row.median_worst_side_uv_log_distance
                        )
                        else None
                    ),

                    "median_uv_reference_observations": (
                        float(
                            row.median_uv_reference_observations
                        )
                        if pd.notna(
                            row.median_uv_reference_observations
                        )
                        else None
                    ),
                }
            )

        # ==================================================================
        # 9. Distribution evidence
        # ==================================================================

        quantiles = (
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        )

        distribution_variables = (
            "quantity_relative_difference",

            "quantity_difference_vs_import",
            "quantity_absolute_difference_vs_import",

            "bilateral_max_unit_value_log_distance",
            "unit_value_reference_observations",
        )

        quantile_sql = ", ".join(
            f"""
            quantile_cont(
                {{variable}},
                {float(quantile):.17g}
            ) AS q_{index}
            """
            for index, quantile in enumerate(
                quantiles
            )
        )

        for variable in distribution_variables:

            variable_quantile_sql = (
                quantile_sql.replace(
                    "{variable}",
                    variable,
                )
            )

            values = con.execute(
                f"""
                SELECT
                    COUNT(*) AS n_rows,
                    {variable_quantile_sql}

                FROM validation_diagnostics_base

                WHERE both_quantities_observed
                  AND {variable} IS NOT NULL
                """
            ).fetchone()

            n_distribution_rows = int(
                values[0]
            )

            for (
                quantile,
                value,
            ) in zip(
                quantiles,
                values[1:],
            ):
                diagnostics_rows.append(
                    {
                        "scope": "distribution",

                        "criterion": (
                            "both_quantities_non_estimated"
                        ),

                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,

                        "year": None,
                        "target_cmd": None,

                        "variable": variable,

                        "quantile": float(
                            quantile
                        ),

                        "n_rows": (
                            n_distribution_rows
                        ),

                        "share_of_all": None,
                        "share_of_parent": None,

                        "value": (
                            float(value)
                            if value is not None
                            else None
                        ),
                    }
                )

        # ==================================================================
        # 10. Fixed quantity-candidate comparison
        # ==================================================================
        #
        # These diagnostics compare the three fixed quantity candidates after
        # the selector has already made the methodological choice. They are
        # retained for transparency, robustness checks, and reporting.
        # ==================================================================

        candidate_summary = con.execute(
            """
            SELECT
                candidate,
                COUNT(*) FILTER (
                    WHERE candidate_available
                ) AS n_available,
                COUNT(*) FILTER (
                    WHERE candidate_uv_log_distance IS NOT NULL
                ) AS n_with_uv_distance,
                MEDIAN(candidate_uv_log_distance)
                    FILTER (
                        WHERE candidate_uv_log_distance IS NOT NULL
                    ) AS median_uv_distance,
                QUANTILE_CONT(candidate_uv_log_distance, 0.90)
                    FILTER (
                        WHERE candidate_uv_log_distance IS NOT NULL
                    ) AS p90_uv_distance,
                QUANTILE_CONT(candidate_uv_log_distance, 0.95)
                    FILTER (
                        WHERE candidate_uv_log_distance IS NOT NULL
                    ) AS p95_uv_distance,
                QUANTILE_CONT(candidate_uv_log_distance, 0.99)
                    FILTER (
                        WHERE candidate_uv_log_distance IS NOT NULL
                    ) AS p99_uv_distance,
                MEDIAN(candidate_export_log_gap)
                    FILTER (
                        WHERE candidate_export_log_gap IS NOT NULL
                    ) AS median_export_log_gap,
                MEDIAN(candidate_import_log_gap)
                    FILTER (
                        WHERE candidate_import_log_gap IS NOT NULL
                    ) AS median_import_log_gap
            FROM validation_quantity_candidates
            WHERE both_quantities_observed
            GROUP BY candidate
            ORDER BY candidate
            """
        ).fetchdf()

        for row in candidate_summary.itertuples(index=False):
            metrics = {
                "median_uv_log_distance": row.median_uv_distance,
                "p90_uv_log_distance": row.p90_uv_distance,
                "p95_uv_log_distance": row.p95_uv_distance,
                "p99_uv_log_distance": row.p99_uv_distance,
                "median_export_log_gap": row.median_export_log_gap,
                "median_import_log_gap": row.median_import_log_gap,
            }

            for variable, value in metrics.items():
                diagnostics_rows.append(
                    {
                        "scope": "quantity_candidate_summary",
                        "criterion": "both_quantities_non_estimated",
                        "candidate": str(row.candidate),
                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,
                        "disagreement_band": None,
                        "uv_distance_band": None,
                        "year": None,
                        "target_cmd": None,
                        "variable": variable,
                        "quantile": None,
                        "n_rows": int(row.n_available),
                        "share_of_all": _share(
                            int(row.n_available),
                            n_all_rows,
                        ),
                        "share_of_parent": None,
                        "value": (
                            float(value)
                            if value is not None and pd.notna(value)
                            else None
                        ),
                        "n_candidate_uv_rows": int(row.n_with_uv_distance),
                    }
                )

        # ------------------------------------------------------------------
        # Candidate performance by mirror-disagreement band
        # ------------------------------------------------------------------

        candidate_by_disagreement = con.execute(
            f"""
            SELECT
                candidate,
                {disagreement_band_sql} AS disagreement_band,
                COUNT(*) AS n_rows,
                MEDIAN(candidate_uv_log_distance)
                    AS median_uv_distance,
                QUANTILE_CONT(candidate_uv_log_distance, 0.90)
                    AS p90_uv_distance,
                QUANTILE_CONT(candidate_uv_log_distance, 0.95)
                    AS p95_uv_distance
            FROM validation_quantity_candidates
            WHERE both_quantities_observed
              AND quantity_relative_difference IS NOT NULL
              AND candidate_uv_log_distance IS NOT NULL
            GROUP BY
                candidate,
                disagreement_band
            ORDER BY
                candidate,
                disagreement_band
            """
        ).fetchdf()

        for row in candidate_by_disagreement.itertuples(index=False):
            for variable, value in {
                "median_uv_log_distance": row.median_uv_distance,
                "p90_uv_log_distance": row.p90_uv_distance,
                "p95_uv_log_distance": row.p95_uv_distance,
            }.items():
                diagnostics_rows.append(
                    {
                        "scope": "quantity_candidate_by_disagreement",
                        "criterion": "fixed_quantity_candidate",
                        "candidate": str(row.candidate),
                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,
                        "disagreement_band": str(row.disagreement_band),
                        "uv_distance_band": None,
                        "year": None,
                        "target_cmd": None,
                        "variable": variable,
                        "quantile": None,
                        "n_rows": int(row.n_rows),
                        "share_of_all": _share(
                            int(row.n_rows),
                            n_all_rows,
                        ),
                        "share_of_parent": None,
                        "value": (
                            float(value)
                            if value is not None and pd.notna(value)
                            else None
                        ),
                    }
                )

        # ------------------------------------------------------------------
        # Closest candidate to the product-year UV reference
        # ------------------------------------------------------------------
        # Evaluate only rows where all three candidate distances exist.
        # Ties are retained explicitly rather than broken arbitrarily.
        # ------------------------------------------------------------------

        candidate_closest = con.execute(
            """
            WITH wide AS (
                SELECT
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    quantity_relative_difference,

                    MAX(candidate_uv_log_distance)
                        FILTER (WHERE candidate = 'simple_mean')
                        AS d_simple,

                    MAX(candidate_uv_log_distance)
                        FILTER (WHERE candidate = 'value_weighted')
                        AS d_value,

                    MAX(candidate_uv_log_distance)
                        FILTER (
                            WHERE candidate = 'quantity_reliability_weighted'
                        ) AS d_reliability

                FROM validation_quantity_candidates

                WHERE both_quantities_observed

                GROUP BY
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    quantity_relative_difference
            ),

            scored AS (
                SELECT
                    *,
                    LEAST(
                        d_simple,
                        d_value,
                        d_reliability
                    ) AS best_distance
                FROM wide
                WHERE d_simple IS NOT NULL
                  AND d_value IS NOT NULL
                  AND d_reliability IS NOT NULL
            ),

            classified AS (
                SELECT
                    *,
                    (
                        CAST(ABS(d_simple - best_distance) <= 1e-12 AS INTEGER)
                        + CAST(ABS(d_value - best_distance) <= 1e-12 AS INTEGER)
                        + CAST(ABS(d_reliability - best_distance) <= 1e-12 AS INTEGER)
                    ) AS n_best,

                    CASE
                        WHEN ABS(d_simple - best_distance) <= 1e-12
                         AND ABS(d_value - best_distance) > 1e-12
                         AND ABS(d_reliability - best_distance) > 1e-12
                            THEN 'simple_mean'

                        WHEN ABS(d_value - best_distance) <= 1e-12
                         AND ABS(d_simple - best_distance) > 1e-12
                         AND ABS(d_reliability - best_distance) > 1e-12
                            THEN 'value_weighted'

                        WHEN ABS(d_reliability - best_distance) <= 1e-12
                         AND ABS(d_simple - best_distance) > 1e-12
                         AND ABS(d_value - best_distance) > 1e-12
                            THEN 'quantity_reliability_weighted'

                        ELSE 'tie'
                    END AS closest_candidate

                FROM scored
            )

            SELECT
                closest_candidate,
                COUNT(*) AS n_rows
            FROM classified
            GROUP BY closest_candidate
            ORDER BY closest_candidate
            """
        ).fetchdf()

        n_closest_rows = int(candidate_closest["n_rows"].sum())

        for row in candidate_closest.itertuples(index=False):
            diagnostics_rows.append(
                {
                    "scope": "quantity_candidate_closest_reference",
                    "criterion": "all_three_candidate_uv_distances_available",
                    "candidate": str(row.closest_candidate),
                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,
                    "disagreement_band": None,
                    "uv_distance_band": None,
                    "year": None,
                    "target_cmd": None,
                    "variable": "closest_reference_share",
                    "quantile": None,
                    "n_rows": int(row.n_rows),
                    "share_of_all": _share(
                        int(row.n_rows),
                        n_all_rows,
                    ),
                    "share_of_parent": _share(
                        int(row.n_rows),
                        n_closest_rows,
                    ),
                    "value": _share(
                        int(row.n_rows),
                        n_closest_rows,
                    ),
                }
            )

        # ------------------------------------------------------------------
        # Candidate temporal stability
        # ------------------------------------------------------------------

        candidate_by_year = con.execute(
            """
            SELECT
                year,
                candidate,
                COUNT(*) FILTER (
                    WHERE candidate_uv_log_distance IS NOT NULL
                ) AS n_rows,
                MEDIAN(candidate_uv_log_distance)
                    FILTER (
                        WHERE candidate_uv_log_distance IS NOT NULL
                    ) AS median_uv_distance,
                QUANTILE_CONT(candidate_uv_log_distance, 0.90)
                    FILTER (
                        WHERE candidate_uv_log_distance IS NOT NULL
                    ) AS p90_uv_distance
            FROM validation_quantity_candidates
            WHERE both_quantities_observed
            GROUP BY year, candidate
            ORDER BY year, candidate
            """
        ).fetchdf()

        for row in candidate_by_year.itertuples(index=False):
            for variable, value in {
                "median_uv_log_distance": row.median_uv_distance,
                "p90_uv_log_distance": row.p90_uv_distance,
            }.items():
                diagnostics_rows.append(
                    {
                        "scope": "quantity_candidate_by_year",
                        "criterion": "fixed_quantity_candidate",
                        "candidate": str(row.candidate),
                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,
                        "disagreement_band": None,
                        "uv_distance_band": None,
                        "year": int(row.year),
                        "target_cmd": None,
                        "variable": variable,
                        "quantile": None,
                        "n_rows": int(row.n_rows),
                        "share_of_all": None,
                        "share_of_parent": None,
                        "value": (
                            float(value)
                            if value is not None and pd.notna(value)
                            else None
                        ),
                    }
                )

        # ------------------------------------------------------------------
        # Candidate product stability
        # ------------------------------------------------------------------

        candidate_by_product = con.execute(
            """
            SELECT
                target_cmd,
                candidate,
                COUNT(*) FILTER (
                    WHERE candidate_uv_log_distance IS NOT NULL
                ) AS n_rows,
                MEDIAN(candidate_uv_log_distance)
                    FILTER (
                        WHERE candidate_uv_log_distance IS NOT NULL
                    ) AS median_uv_distance,
                QUANTILE_CONT(candidate_uv_log_distance, 0.90)
                    FILTER (
                        WHERE candidate_uv_log_distance IS NOT NULL
                    ) AS p90_uv_distance
            FROM validation_quantity_candidates
            WHERE both_quantities_observed
            GROUP BY target_cmd, candidate
            ORDER BY target_cmd, candidate
            """
        ).fetchdf()

        for row in candidate_by_product.itertuples(index=False):
            for variable, value in {
                "median_uv_log_distance": row.median_uv_distance,
                "p90_uv_log_distance": row.p90_uv_distance,
            }.items():
                diagnostics_rows.append(
                    {
                        "scope": "quantity_candidate_by_product",
                        "criterion": "fixed_quantity_candidate",
                        "candidate": str(row.candidate),
                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,
                        "disagreement_band": None,
                        "uv_distance_band": None,
                        "year": None,
                        "target_cmd": str(row.target_cmd),
                        "variable": variable,
                        "quantile": None,
                        "n_rows": int(row.n_rows),
                        "share_of_all": None,
                        "share_of_parent": None,
                        "value": (
                            float(value)
                            if value is not None and pd.notna(value)
                            else None
                        ),
                    }
                )

        # ------------------------------------------------------------------
        # Quantity-conditioned candidate UV plausibility
        # ------------------------------------------------------------------

        candidate_conditioned_summary = con.execute(
            """
            SELECT
                candidate,
                COUNT(*) FILTER (
                    WHERE candidate_conditioned_uv_abs_residual IS NOT NULL
                ) AS n_rows,
                MEDIAN(candidate_conditioned_uv_abs_residual)
                    FILTER (
                        WHERE candidate_conditioned_uv_abs_residual IS NOT NULL
                    ) AS median_abs_residual,
                QUANTILE_CONT(candidate_conditioned_uv_abs_residual, 0.90)
                    FILTER (
                        WHERE candidate_conditioned_uv_abs_residual IS NOT NULL
                    ) AS p90_abs_residual,
                QUANTILE_CONT(candidate_conditioned_uv_abs_residual, 0.95)
                    FILTER (
                        WHERE candidate_conditioned_uv_abs_residual IS NOT NULL
                    ) AS p95_abs_residual
            FROM validation_quantity_candidates_conditioned
            WHERE both_quantities_observed
            GROUP BY candidate
            ORDER BY candidate
            """
        ).fetchdf()

        for row in candidate_conditioned_summary.itertuples(index=False):
            for variable, value in {
                "median_conditioned_uv_abs_residual": row.median_abs_residual,
                "p90_conditioned_uv_abs_residual": row.p90_abs_residual,
                "p95_conditioned_uv_abs_residual": row.p95_abs_residual,
            }.items():
                diagnostics_rows.append(
                    {
                        "scope": "quantity_candidate_conditioned_summary",
                        "criterion": "hs_year_quantity_conditioned_uv_model_min_30",
                        "candidate": str(row.candidate),
                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,
                        "disagreement_band": None,
                        "uv_distance_band": None,
                        "year": None,
                        "target_cmd": None,
                        "variable": variable,
                        "quantile": None,
                        "n_rows": int(row.n_rows),
                        "share_of_all": _share(int(row.n_rows), n_all_rows),
                        "share_of_parent": _share(int(row.n_rows), n_both_observed),
                        "value": (
                            float(value)
                            if value is not None and pd.notna(value)
                            else None
                        ),
                    }
                )

        candidate_conditioned_by_disagreement = con.execute(
            f"""
            SELECT
                {disagreement_band_sql} AS disagreement_band,
                candidate,
                COUNT(*) FILTER (
                    WHERE candidate_conditioned_uv_abs_residual IS NOT NULL
                ) AS n_rows,
                MEDIAN(candidate_conditioned_uv_abs_residual)
                    FILTER (
                        WHERE candidate_conditioned_uv_abs_residual IS NOT NULL
                    ) AS median_abs_residual,
                QUANTILE_CONT(candidate_conditioned_uv_abs_residual, 0.90)
                    FILTER (
                        WHERE candidate_conditioned_uv_abs_residual IS NOT NULL
                    ) AS p90_abs_residual,
                QUANTILE_CONT(candidate_conditioned_uv_abs_residual, 0.95)
                    FILTER (
                        WHERE candidate_conditioned_uv_abs_residual IS NOT NULL
                    ) AS p95_abs_residual
            FROM validation_quantity_candidates_conditioned
            WHERE both_quantities_observed
              AND quantity_relative_difference IS NOT NULL
            GROUP BY disagreement_band, candidate
            ORDER BY disagreement_band, candidate
            """
        ).fetchdf()

        for row in candidate_conditioned_by_disagreement.itertuples(index=False):
            for variable, value in {
                "median_conditioned_uv_abs_residual": row.median_abs_residual,
                "p90_conditioned_uv_abs_residual": row.p90_abs_residual,
                "p95_conditioned_uv_abs_residual": row.p95_abs_residual,
            }.items():
                diagnostics_rows.append(
                    {
                        "scope": "quantity_candidate_conditioned_by_disagreement",
                        "criterion": "hs_year_quantity_conditioned_uv_model_min_30",
                        "candidate": str(row.candidate),
                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,
                        "disagreement_band": str(row.disagreement_band),
                        "uv_distance_band": None,
                        "year": None,
                        "target_cmd": None,
                        "variable": variable,
                        "quantile": None,
                        "n_rows": int(row.n_rows),
                        "share_of_all": _share(int(row.n_rows), n_all_rows),
                        "share_of_parent": None,
                        "value": (
                            float(value)
                            if value is not None and pd.notna(value)
                            else None
                        ),
                    }
                )

        candidate_closest_conditioned = con.execute(
            """
            WITH wide AS (
                SELECT
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    export_quantity_kg,
                    import_quantity_kg,
                    MAX(CASE WHEN candidate = 'simple_mean'
                        THEN candidate_conditioned_uv_abs_residual END) AS d_simple,
                    MAX(CASE WHEN candidate = 'value_weighted'
                        THEN candidate_conditioned_uv_abs_residual END) AS d_value,
                    MAX(CASE WHEN candidate = 'quantity_reliability_weighted'
                        THEN candidate_conditioned_uv_abs_residual END) AS d_reliability
                FROM validation_quantity_candidates_conditioned
                WHERE both_quantities_observed
                GROUP BY
                    year, exporter_iso, importer_iso, target_cmd,
                    export_quantity_kg, import_quantity_kg
            ),
            scored AS (
                SELECT
                    *,
                    LEAST(d_simple, d_value, d_reliability) AS best_distance
                FROM wide
                WHERE d_simple IS NOT NULL
                  AND d_value IS NOT NULL
                  AND d_reliability IS NOT NULL
            ),
            classified AS (
                SELECT
                    CASE
                        WHEN ABS(d_simple - best_distance) <= 1e-12
                         AND ABS(d_value - best_distance) > 1e-12
                         AND ABS(d_reliability - best_distance) > 1e-12
                            THEN 'simple_mean'
                        WHEN ABS(d_value - best_distance) <= 1e-12
                         AND ABS(d_simple - best_distance) > 1e-12
                         AND ABS(d_reliability - best_distance) > 1e-12
                            THEN 'value_weighted'
                        WHEN ABS(d_reliability - best_distance) <= 1e-12
                         AND ABS(d_simple - best_distance) > 1e-12
                         AND ABS(d_value - best_distance) > 1e-12
                            THEN 'quantity_reliability_weighted'
                        ELSE 'tie'
                    END AS closest_candidate
                FROM scored
            )
            SELECT
                closest_candidate,
                COUNT(*) AS n_rows
            FROM classified
            GROUP BY closest_candidate
            ORDER BY closest_candidate
            """
        ).fetchdf()

        n_closest_conditioned_rows = int(
            candidate_closest_conditioned["n_rows"].sum()
        )

        for row in candidate_closest_conditioned.itertuples(index=False):
            diagnostics_rows.append(
                {
                    "scope": "quantity_candidate_closest_conditioned",
                    "criterion": "all_three_conditioned_residuals_available",
                    "candidate": str(row.closest_candidate),
                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,
                    "disagreement_band": None,
                    "uv_distance_band": None,
                    "year": None,
                    "target_cmd": None,
                    "variable": "closest_conditioned_uv_share",
                    "quantile": None,
                    "n_rows": int(row.n_rows),
                    "share_of_all": _share(int(row.n_rows), n_all_rows),
                    "share_of_parent": _share(
                        int(row.n_rows),
                        n_closest_conditioned_rows,
                    ),
                    "value": _share(
                        int(row.n_rows),
                        n_closest_conditioned_rows,
                    ),
                }
            )

        # ------------------------------------------------------------------
        # How much do the three candidates actually differ?
        # ------------------------------------------------------------------

        candidate_spread = con.execute(
            f"""
            SELECT
                {disagreement_band_sql} AS disagreement_band,
                COUNT(*) AS n_rows,
                MEDIAN(
                    (
                        GREATEST(
                            quantity_simple_mean_kg,
                            quantity_value_weighted_kg,
                            quantity_reliability_weighted_kg
                        )
                        - LEAST(
                            quantity_simple_mean_kg,
                            quantity_value_weighted_kg,
                            quantity_reliability_weighted_kg
                        )
                    )
                    /
                    GREATEST(
                        quantity_simple_mean_kg,
                        quantity_value_weighted_kg,
                        quantity_reliability_weighted_kg
                    )
                ) AS median_relative_spread,
                QUANTILE_CONT(
                    (
                        GREATEST(
                            quantity_simple_mean_kg,
                            quantity_value_weighted_kg,
                            quantity_reliability_weighted_kg
                        )
                        - LEAST(
                            quantity_simple_mean_kg,
                            quantity_value_weighted_kg,
                            quantity_reliability_weighted_kg
                        )
                    )
                    /
                    GREATEST(
                        quantity_simple_mean_kg,
                        quantity_value_weighted_kg,
                        quantity_reliability_weighted_kg
                    ),
                    0.95
                ) AS p95_relative_spread
            FROM validation_diagnostics_base
            WHERE both_quantities_observed
              AND quantity_relative_difference IS NOT NULL
              AND quantity_simple_mean_kg > 0
              AND quantity_value_weighted_kg > 0
              AND quantity_reliability_weighted_kg > 0
            GROUP BY disagreement_band
            ORDER BY disagreement_band
            """
        ).fetchdf()

        for row in candidate_spread.itertuples(index=False):
            for variable, value in {
                "median_relative_candidate_spread": row.median_relative_spread,
                "p95_relative_candidate_spread": row.p95_relative_spread,
            }.items():
                diagnostics_rows.append(
                    {
                        "scope": "quantity_candidate_spread",
                        "criterion": "all_three_candidates_available",
                        "candidate": None,
                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,
                        "disagreement_band": str(row.disagreement_band),
                        "uv_distance_band": None,
                        "year": None,
                        "target_cmd": None,
                        "variable": variable,
                        "quantile": None,
                        "n_rows": int(row.n_rows),
                        "share_of_all": _share(
                            int(row.n_rows),
                            n_all_rows,
                        ),
                        "share_of_parent": None,
                        "value": (
                            float(value)
                            if value is not None and pd.notna(value)
                            else None
                        ),
                    }
                )


        # ==================================================================
        # Model evaluation — absolute log-error distribution
        # ==================================================================

        model_error_distribution = con.execute(
            """
            SELECT
                model_name,

                COUNT(*)
                    AS n_rows,

                QUANTILE_CONT(
                    absolute_log_error,
                    0.50
                )
                    AS p50,

                QUANTILE_CONT(
                    absolute_log_error,
                    0.75
                )
                    AS p75,

                QUANTILE_CONT(
                    absolute_log_error,
                    0.90
                )
                    AS p90,

                QUANTILE_CONT(
                    absolute_log_error,
                    0.95
                )
                    AS p95,

                QUANTILE_CONT(
                    absolute_log_error,
                    0.99
                )
                    AS p99

            FROM validation_model_predictions

            GROUP BY model_name

            ORDER BY model_name
            """
        ).fetchdf()

        quantile_columns = {
            0.50: "p50",
            0.75: "p75",
            0.90: "p90",
            0.95: "p95",
            0.99: "p99",
        }

        for row in model_error_distribution.itertuples(
            index=False
        ):

            for quantile, column in (
                quantile_columns.items()
            ):

                diagnostics_rows.append(
                    {
                        "scope": (
                            "model_absolute_log_error_distribution"
                        ),
                        "criterion": (
                            "leave_one_year_out"
                        ),
                        "model_name": str(
                            row.model_name
                        ),
                        "candidate": None,

                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,

                        "disagreement_band": None,
                        "uv_distance_band": None,

                        "year": None,
                        "target_cmd": None,

                        "variable": (
                            "absolute_log_error"
                        ),
                        "quantile": float(
                            quantile
                        ),

                        "n_rows": int(
                            row.n_rows
                        ),

                        "share_of_all": None,
                        "share_of_parent": None,

                        "value": float(
                            getattr(
                                row,
                                column,
                            )
                        ),
                    }
                )
        # ==================================================================
        # Model evaluation — performance by held-out year
        # ==================================================================

        model_performance_by_year = con.execute(
            """
            SELECT
                model_name,
                held_out_year,

                COUNT(*)
                    AS n_rows,

                AVG(
                    absolute_log_error
                )
                    AS mae_log,

                SQRT(
                    AVG(
                        squared_log_error
                    )
                )
                    AS rmse_log,

                MEDIAN(
                    absolute_log_error
                )
                    AS median_absolute_log_error

            FROM validation_model_predictions

            GROUP BY
                model_name,
                held_out_year

            ORDER BY
                model_name,
                held_out_year
            """
        ).fetchdf()

        for row in model_performance_by_year.itertuples(
            index=False
        ):

            metrics = {
                "mae_log": row.mae_log,
                "rmse_log": row.rmse_log,
                "median_absolute_log_error": (
                    row.median_absolute_log_error
                ),
            }

            for variable, value in metrics.items():

                diagnostics_rows.append(
                    {
                        "scope": (
                            "model_performance_by_year"
                        ),
                        "criterion": (
                            "leave_one_year_out"
                        ),
                        "model_name": str(
                            row.model_name
                        ),
                        "candidate": None,

                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,

                        "disagreement_band": None,
                        "uv_distance_band": None,

                        "year": int(
                            row.held_out_year
                        ),
                        "target_cmd": None,

                        "variable": variable,
                        "quantile": None,

                        "n_rows": int(
                            row.n_rows
                        ),

                        "share_of_all": None,
                        "share_of_parent": None,

                        "value": float(
                            value
                        ),
                    }
                )
        # ==================================================================
        # Model evaluation — overall cross-validated performance
        # ==================================================================

        model_performance = con.execute(
            """
            SELECT
                model_name,
                model_type,

                COUNT(*)
                    AS n_rows,

                AVG(
                    absolute_log_error
                )
                    AS mae_log,

                SQRT(
                    AVG(
                        squared_log_error
                    )
                )
                    AS rmse_log,

                MEDIAN(
                    absolute_log_error
                )
                    AS median_absolute_log_error

            FROM validation_model_predictions

            GROUP BY
                model_name,
                model_type

            ORDER BY
                model_name
            """
        ).fetchdf()

        for row in model_performance.itertuples(
            index=False
        ):

            metrics = {
                "mae_log": row.mae_log,
                "rmse_log": row.rmse_log,
                "median_absolute_log_error": (
                    row.median_absolute_log_error
                ),
            }

            for variable, value in metrics.items():

                diagnostics_rows.append(
                    {
                        "scope": (
                            "model_performance_overall"
                        ),
                        "criterion": (
                            "leave_one_year_out"
                        ),
                        "model_name": str(
                            row.model_name
                        ),
                        "candidate": None,

                        "disagreement_threshold": None,
                        "uv_support_threshold": None,
                        "uv_distance_threshold": None,

                        "disagreement_band": None,
                        "uv_distance_band": None,

                        "year": None,
                        "target_cmd": None,

                        "variable": variable,
                        "quantile": None,

                        "n_rows": int(
                            row.n_rows
                        ),

                        "share_of_all": None,
                        "share_of_parent": None,

                        "value": float(
                            value
                        ),
                    }
                )
        # ==================================================================
        # Diagnostics artifact
        # ==================================================================

        diagnostics = pd.DataFrame(
            diagnostics_rows
        )

        if diagnostics.empty:
            raise RuntimeError(
                "Validation diagnostics contain no rows."
            )

        # Ensure optional columns exist consistently across scopes.
        optional_columns = (
            "candidate",
            "disagreement_band",
            "uv_distance_band",

            "total_rows",
            "n_candidate_uv_rows",
            "n_both_quantities",

            "model_name"

            "median_quantity_relative_difference",
            "median_worst_side_uv_log_distance",
            "median_uv_reference_observations",
            "median_quantity_difference_vs_import",
            "median_absolute_quantity_difference_vs_import",
            "exporter_higher_share",
        )

        for column in optional_columns:

            if column not in diagnostics.columns:
                diagnostics[column] = None

        # Stable column order.
        diagnostics = diagnostics[
            [
                "scope",
                "criterion",
                "candidate",
                "model_name",

                "disagreement_threshold",
                "uv_support_threshold",
                "uv_distance_threshold",

                "disagreement_band",
                "uv_distance_band",

                "year",
                "target_cmd",

                "variable",
                "quantile",

                "n_rows",
                "total_rows",
                "n_both_quantities",
                "n_candidate_uv_rows",

                "share_of_all",
                "share_of_parent",

                "value",

                "median_quantity_relative_difference",
                "median_worst_side_uv_log_distance",
                "median_uv_reference_observations",
                "median_quantity_difference_vs_import",
                "median_absolute_quantity_difference_vs_import",
                "exporter_higher_share",
            ]
        ]

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        if temporary_path.exists():
            temporary_path.unlink()

        temporary_sql = sql_literal(
            temporary_path
        )

        con.register(
            "_validation_diagnostics",
            diagnostics,
        )

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM _validation_diagnostics
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

    finally:
        con.close()

    # ==================================================================
    # Logging
    # ==================================================================

    elapsed = perf_counter() - start

    logger.info(
        "  All Validation rows: %s",
        f"{n_all_rows:,}",
    )

    logger.info(
        "  Both quantities available: %s",
        f"{n_both_quantities:,}",
    )

    logger.info(
        "  Both quantities non-estimated: %s",
        f"{n_both_observed:,}",
    )

    logger.info(
        "  Complete candidate-training evidence: %s",
        f"{n_complete_training_evidence:,}",
    )

    logger.info(
        "  Threshold combinations evaluated: %s",
        f"{len(grid_metadata):,}",
    )

    logger.info(
        "  Quantity-conditioned candidate rows: %s",
        f"{int(candidate_conditioned_summary['n_rows'].max()):,}",
    )

    logger.info(
        "  Diagnostics rows: %s",
        f"{len(diagnostics):,}",
    )

    logger.info(
        "  Output: %s",
        output_path,
    )

    logger.info(
        "%s CHECK — Post-selection quantity diagnostics built.",
        get_symbols("check"),
    )

    return {
        "status": "completed",
        "run_id": run_id,

        "output_path": output_path,
        "diagnostics_path": output_path,

        "n_rows": n_all_rows,

        "n_both_quantities": (
            n_both_quantities
        ),

        "n_both_observed": (
            n_both_observed
        ),

        "n_both_observed_with_disagreement": (
            n_both_observed_with_disagreement
        ),

        "n_both_observed_with_uv_distance": (
            n_both_observed_with_uv_distance
        ),

        "n_complete_training_evidence": (
            n_complete_training_evidence
        ),

        "n_threshold_combinations": len(
            grid_metadata
        ),

        "fitted": False,

        "ready_for_training_decision": True,

        "selection_already_applied": True,

        "duration_seconds": elapsed,
        "reused": False,
    }