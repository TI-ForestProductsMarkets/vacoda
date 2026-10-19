"""
vacoda.steps.validation.audit_inference

Audit internal and cross-variable consistency of Validation inference.

This stage runs after Validation inference.

It verifies:

1. inference population/key integrity;
2. HS4 derivation consistency;
3. observed-quantity availability and validity flags;
4. model-input and prediction consistency;
5. prediction inverse-transform identity;
6. residual availability and residual formulas;
7. categorical-support flag consistency;
8. fitted-model metadata consistency;
9. training-selection consistency;
10. value/quantity/unit-value algebraic consistency.

The audit does not modify Validation rows and does not create corrected
or imputed values.

It writes one Parquet row containing the complete audit summary and raises
immediately when any hard invariant fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.steps.validation import ValidationConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.sql import sql_literal


# ======================================================================
# Helpers
# ======================================================================


def _require_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    required_columns: set[str],
    label: str,
) -> None:
    """Require all columns needed by the consistency audit."""

    parquet_sql = sql_literal(
        parquet_path
    )

    available_columns = {
        str(row[0])
        for row in con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet({parquet_sql})
            """
        ).fetchall()
    }

    missing_columns = sorted(
        required_columns
        - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            f"{label} is missing required columns: "
            + ", ".join(
                missing_columns
            )
        )


# ======================================================================
# Main audit
# ======================================================================


def audit_validation_inference(
    *,
    cfg: ValidationConfig,
    inference_path: Path,
    run_id: str,
    output_path: Path | None = None,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Audit Validation inference and cross-variable consistency.

    This audit is deliberately strict on structural and mathematical
    invariants. Scientific unusualness is not itself treated as a
    pipeline failure.

    A successful return implies ``audit_passed=True``.
    """

    inference_path = Path(
        inference_path
    )

    if not inference_path.is_file():
        raise FileNotFoundError(
            "Validation inference not found: "
            f"{inference_path}"
        )

    output_path = (
        Path(output_path)
        if output_path is not None
        else cfg.product_path(
            "inference_audit"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    inference_sql = sql_literal(
        inference_path
    )

    output_sql = sql_literal(
        output_path
    )

    logger.info(
        "%s Audit cross-variable consistency",
        get_symbols("build"),
    )

    logger.info(
        "  Inference: %s",
        inference_path,
    )

    logger.info(
        "  Audit output: %s",
        output_path,
    )

    con = duckdb.connect()

    try:

        con.execute(
            f"PRAGMA threads="
            f"{int(cfg.engine.threads)}"
        )

        # ==============================================================
        # INPUT CONTRACT
        # ==============================================================

        required_columns = {
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",
            "target_hs4",

            "trade_value_reconciled_constant_usd",
            "trade_synced_quantity_kg",

            "training_selected",

            "model_input_valid",

            "model_prediction_log_quantity",
            "model_prediction_quantity_kg",
            "model_prediction_available",

            "observed_quantity_available",
            "observed_quantity_valid_for_log",

            "quantity_log_residual",
            "quantity_abs_log_residual",

            "exporter_seen",
            "importer_seen",
            "hs6_seen",
            "hs4_seen",
            "manual_review_required",

            "validation_model_name",
            "validation_model_type",
        }

        _require_columns(
            con=con,
            parquet_path=inference_path,
            required_columns=required_columns,
            label="Validation inference",
        )

        # ==============================================================
        # COMPLETE AUDIT QUERY
        #
        # Scientific thresholds do not belong here.
        # Every mismatch counted below is a structural/mathematical
        # contradiction.
        # ==============================================================

        audit_row = con.execute(
            f"""
            WITH inference AS (

                SELECT *
                FROM read_parquet(
                    {inference_sql}
                )

            ),

            duplicate_keys AS (

                SELECT
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    COUNT(*) AS n_rows

                FROM inference

                GROUP BY
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd

                HAVING COUNT(*) > 1

            ),

            checks AS (

                SELECT

                    -- --------------------------------------------------
                    -- Population
                    -- --------------------------------------------------

                    COUNT(*)
                        AS n_rows,

                    COUNT(*) FILTER (
                        WHERE year IS NULL
                           OR exporter_iso IS NULL
                           OR importer_iso IS NULL
                           OR target_cmd IS NULL
                    )
                        AS n_missing_key_rows,

                    COUNT(*) FILTER (
                        WHERE target_hs4 IS NULL
                           OR target_hs4
                              <> LEFT(
                                  target_cmd,
                                  4
                              )
                    )
                        AS n_invalid_hs4_rows,

                    COUNT(*) FILTER (
                        WHERE training_selected
                            IS NULL
                    )
                        AS n_null_training_selected,

                    COUNT(*) FILTER (
                        WHERE model_input_valid
                            IS NULL
                    )
                        AS n_null_model_input_valid,

                    -- --------------------------------------------------
                    -- Trade value
                    -- --------------------------------------------------

                    COUNT(*) FILTER (
                        WHERE
                            trade_value_reconciled_constant_usd
                                IS NOT NULL
                            AND NOT isfinite(
                                trade_value_reconciled_constant_usd
                            )
                    )
                        AS n_nonfinite_trade_values,

                    COUNT(*) FILTER (
                        WHERE model_input_valid
                          AND (
                                trade_value_reconciled_constant_usd
                                    IS NULL
                             OR NOT isfinite(
                                    trade_value_reconciled_constant_usd
                                )
                             OR trade_value_reconciled_constant_usd
                                    <= 0
                          )
                    )
                        AS n_valid_model_input_bad_trade_values,

                    -- --------------------------------------------------
                    -- Observed quantity
                    -- --------------------------------------------------

                    COUNT(*) FILTER (
                        WHERE
                            observed_quantity_available
                            IS DISTINCT FROM (
                                trade_synced_quantity_kg
                                    IS NOT NULL
                                AND isfinite(
                                    trade_synced_quantity_kg
                                )
                            )
                    )
                        AS n_observed_availability_mismatches,

                    COUNT(*) FILTER (
                        WHERE
                            observed_quantity_valid_for_log
                            IS DISTINCT FROM (
                                trade_synced_quantity_kg
                                    IS NOT NULL
                                AND isfinite(
                                    trade_synced_quantity_kg
                                )
                                AND trade_synced_quantity_kg
                                    > 0
                            )
                    )
                        AS n_observed_log_validity_mismatches,

                    COUNT(*) FILTER (
                        WHERE
                            trade_synced_quantity_kg
                                IS NOT NULL
                            AND NOT isfinite(
                                trade_synced_quantity_kg
                            )
                    )
                        AS n_nonfinite_observed_quantities,

                    COUNT(*) FILTER (
                        WHERE
                            trade_synced_quantity_kg
                                IS NOT NULL
                            AND isfinite(
                                trade_synced_quantity_kg
                            )
                            AND trade_synced_quantity_kg
                                <= 0
                    )
                        AS n_nonpositive_observed_quantities,

                    -- --------------------------------------------------
                    -- Prediction
                    -- --------------------------------------------------

                    COUNT(*) FILTER (
                        WHERE
                            model_prediction_available
                            IS DISTINCT FROM (
                                model_prediction_log_quantity
                                    IS NOT NULL
                                AND isfinite(
                                    model_prediction_log_quantity
                                )
                                AND model_prediction_quantity_kg
                                    IS NOT NULL
                                AND isfinite(
                                    model_prediction_quantity_kg
                                )
                                AND model_prediction_quantity_kg
                                    > 0
                            )
                    )
                        AS n_prediction_availability_mismatches,

                    COUNT(*) FILTER (
                        WHERE model_prediction_available
                          AND (
                                model_prediction_log_quantity
                                    IS NULL
                             OR NOT isfinite(
                                    model_prediction_log_quantity
                                )
                             OR model_prediction_quantity_kg
                                    IS NULL
                             OR NOT isfinite(
                                    model_prediction_quantity_kg
                                )
                             OR model_prediction_quantity_kg
                                    <= 0
                          )
                    )
                        AS n_invalid_available_predictions,

                    COUNT(*) FILTER (
                        WHERE
                            NOT model_input_valid
                            AND model_prediction_available
                    )
                        AS n_predictions_with_invalid_model_input,

                    COUNT(*) FILTER (
                        WHERE
                            model_input_valid
                            AND NOT model_prediction_available
                    )
                        AS n_missing_predictions_for_valid_input,

                    -- exp(log prediction) must equal Q prediction.
                    COUNT(*) FILTER (
                        WHERE model_prediction_available
                          AND ABS(
                                model_prediction_quantity_kg
                                - EXP(
                                    model_prediction_log_quantity
                                )
                              )
                              >
                              (
                                  1e-10
                                  * GREATEST(
                                      ABS(
                                          model_prediction_quantity_kg
                                      ),
                                      1.0
                                  )
                              )
                    )
                        AS n_prediction_inverse_transform_mismatches,

                    -- --------------------------------------------------
                    -- Residual
                    -- --------------------------------------------------

                    COUNT(*) FILTER (
                        WHERE
                            (
                                quantity_log_residual
                                    IS NOT NULL
                                AND isfinite(
                                    quantity_log_residual
                                )
                            )
                            IS DISTINCT FROM (
                                observed_quantity_valid_for_log
                                AND model_prediction_available
                            )
                    )
                        AS n_residual_availability_mismatches,

                    COUNT(*) FILTER (
                        WHERE
                            observed_quantity_valid_for_log
                            AND model_prediction_available
                            AND (
                                quantity_log_residual
                                    IS NULL
                                OR NOT isfinite(
                                    quantity_log_residual
                                )
                                OR ABS(
                                    quantity_log_residual
                                    - (
                                        LN(
                                            trade_synced_quantity_kg
                                        )
                                        - model_prediction_log_quantity
                                    )
                                )
                                >
                                (
                                    1e-10
                                    * GREATEST(
                                        ABS(
                                            quantity_log_residual
                                        ),
                                        1.0
                                    )
                                )
                            )
                    )
                        AS n_residual_formula_mismatches,

                    COUNT(*) FILTER (
                        WHERE
                            quantity_log_residual
                                IS NOT NULL
                            AND (
                                quantity_abs_log_residual
                                    IS NULL
                                OR NOT isfinite(
                                    quantity_abs_log_residual
                                )
                                OR ABS(
                                    quantity_abs_log_residual
                                    - ABS(
                                        quantity_log_residual
                                    )
                                )
                                >
                                (
                                    1e-12
                                    * GREATEST(
                                        ABS(
                                            quantity_abs_log_residual
                                        ),
                                        1.0
                                    )
                                )
                            )
                    )
                        AS n_absolute_residual_mismatches,

                    COUNT(*) FILTER (
                        WHERE
                            quantity_log_residual
                                IS NULL
                            AND quantity_abs_log_residual
                                IS NOT NULL
                    )
                        AS n_unexpected_absolute_residuals,

                    -- --------------------------------------------------
                    -- Support flags
                    -- --------------------------------------------------

                    COUNT(*) FILTER (
                        WHERE model_input_valid
                          AND (
                                exporter_seen IS NULL
                             OR importer_seen IS NULL
                             OR hs6_seen IS NULL
                             OR hs4_seen IS NULL
                          )
                    )
                        AS n_missing_support_flags_valid_input,

                    COUNT(*) FILTER (
                        WHERE NOT model_input_valid
                          AND (
                                exporter_seen IS NOT NULL
                             OR importer_seen IS NOT NULL
                             OR hs6_seen IS NOT NULL
                             OR hs4_seen IS NOT NULL
                          )
                    )
                        AS n_support_flags_on_invalid_input,

                    COUNT(*) FILTER (
                        WHERE
                            hs6_seen
                            AND (
                                hs4_seen IS NULL
                                OR NOT hs4_seen
                            )
                    )
                        AS n_hs6_seen_hs4_unseen,

                    COUNT(*) FILTER (
                        WHERE manual_review_required IS DISTINCT FROM (
                            hs6_seen IS FALSE AND hs4_seen IS FALSE
                        )
                    ) AS n_manual_review_rule_mismatches,
                        
                    -- --------------------------------------------------
                    -- Model metadata
                    -- --------------------------------------------------

                    COUNT(*) FILTER (
                        WHERE validation_model_name
                            IS NULL
                           OR validation_model_type
                            IS NULL
                    )
                        AS n_missing_model_metadata,

                    COUNT(
                        DISTINCT validation_model_name
                    )
                        AS n_distinct_model_names,

                    COUNT(
                        DISTINCT validation_model_type
                    )
                        AS n_distinct_model_types,

                    -- --------------------------------------------------
                    -- Training rows
                    -- --------------------------------------------------

                    COUNT(*) FILTER (
                        WHERE training_selected
                          AND NOT observed_quantity_valid_for_log
                    )
                        AS n_training_rows_invalid_observed_quantity,

                    COUNT(*) FILTER (
                        WHERE training_selected
                          AND NOT model_input_valid
                    )
                        AS n_training_rows_invalid_model_input,

                    COUNT(*) FILTER (
                        WHERE training_selected
                          AND (
                                exporter_seen IS DISTINCT FROM TRUE
                             OR importer_seen IS DISTINCT FROM TRUE
                             OR hs6_seen IS DISTINCT FROM TRUE
                             OR hs4_seen IS DISTINCT FROM TRUE
                          )
                    )
                        AS n_training_rows_unseen_categories,

                    -- --------------------------------------------------
                    -- Cross-variable unit-value identity
                    --
                    -- UV_obs   = V / Q_obs
                    -- UV_model = V / Q_model
                    --
                    -- log(UV_obs) - log(UV_model)
                    --     = -quantity_log_residual
                    -- --------------------------------------------------

                    COUNT(*) FILTER (
                        WHERE
                            trade_value_reconciled_constant_usd
                                IS NOT NULL
                            AND isfinite(
                                trade_value_reconciled_constant_usd
                            )
                            AND trade_value_reconciled_constant_usd
                                > 0
                            AND observed_quantity_valid_for_log
                            AND model_prediction_available
                    )
                        AS n_unit_value_identity_rows,

                    COUNT(*) FILTER (
                        WHERE
                            trade_value_reconciled_constant_usd
                                IS NOT NULL
                            AND isfinite(
                                trade_value_reconciled_constant_usd
                            )
                            AND trade_value_reconciled_constant_usd
                                > 0
                            AND observed_quantity_valid_for_log
                            AND model_prediction_available
                            AND ABS(
                                (
                                    LN(
                                        trade_value_reconciled_constant_usd
                                        / trade_synced_quantity_kg
                                    )
                                    -
                                    LN(
                                        trade_value_reconciled_constant_usd
                                        / model_prediction_quantity_kg
                                    )
                                )
                                + quantity_log_residual
                            )
                            >
                            (
                                1e-10
                                * GREATEST(
                                    ABS(
                                        quantity_log_residual
                                    ),
                                    1.0
                                )
                            )
                    )
                        AS n_unit_value_identity_mismatches,

                    -- --------------------------------------------------
                    -- Non-failing scientific summaries
                    -- --------------------------------------------------

                    COUNT(*) FILTER (
                        WHERE observed_quantity_available
                    )
                        AS n_observed_quantity_rows,

                    COUNT(*) FILTER (
                        WHERE NOT observed_quantity_available
                    )
                        AS n_missing_observed_quantity_rows,

                    COUNT(*) FILTER (
                        WHERE model_input_valid
                    )
                        AS n_model_input_valid_rows,

                    COUNT(*) FILTER (
                        WHERE NOT model_input_valid
                    )
                        AS n_model_input_invalid_rows,

                    COUNT(*) FILTER (
                        WHERE model_prediction_available
                    )
                        AS n_prediction_rows,

                    COUNT(*) FILTER (
                        WHERE quantity_log_residual
                            IS NOT NULL
                    )
                        AS n_residual_rows,

                    COUNT(*) FILTER (
                        WHERE hs6_seen IS FALSE
                    )
                        AS n_hs6_unseen_rows,

                    COUNT(*) FILTER (
                        WHERE hs4_seen IS FALSE
                    )
                        AS n_hs4_unseen_rows,

                    COUNT(DISTINCT target_cmd) FILTER (
                        WHERE hs6_seen IS FALSE
                    )
                        AS n_hs6_unseen_codes,

                    COUNT(DISTINCT target_hs4) FILTER (
                        WHERE hs4_seen IS FALSE
                    )
                        AS n_hs4_unseen_codes,
                        
                    COUNT(*) FILTER (
                        WHERE manual_review_required IS TRUE
                    ) AS n_manual_review_rows,

                    COUNT(DISTINCT target_cmd) FILTER (
                        WHERE manual_review_required IS TRUE
                    ) AS n_manual_review_hs6_codes,

                    COUNT(*) FILTER (
                        WHERE training_selected
                    )
                        AS n_training_selected_rows,

                    MEDIAN(
                        quantity_abs_log_residual
                    ) FILTER (
                        WHERE quantity_abs_log_residual
                            IS NOT NULL
                    )
                        AS median_abs_log_residual,

                    AVG(
                        quantity_abs_log_residual
                    ) FILTER (
                        WHERE quantity_abs_log_residual
                            IS NOT NULL
                    )
                        AS mean_abs_log_residual

                FROM inference

            )

            SELECT
                checks.*,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                )
                    AS n_duplicate_keys

            FROM checks
            """
        ).fetchone()

        if audit_row is None:
            raise RuntimeError(
                "Could not inspect Validation inference."
            )

        (
            n_rows,
            n_missing_key_rows,
            n_invalid_hs4_rows,
            n_null_training_selected,
            n_null_model_input_valid,

            n_nonfinite_trade_values,
            n_valid_model_input_bad_trade_values,

            n_observed_availability_mismatches,
            n_observed_log_validity_mismatches,
            n_nonfinite_observed_quantities,
            n_nonpositive_observed_quantities,

            n_prediction_availability_mismatches,
            n_invalid_available_predictions,
            n_predictions_with_invalid_model_input,
            n_missing_predictions_for_valid_input,
            n_prediction_inverse_transform_mismatches,

            n_residual_availability_mismatches,
            n_residual_formula_mismatches,
            n_absolute_residual_mismatches,
            n_unexpected_absolute_residuals,

            n_missing_support_flags_valid_input,
            n_support_flags_on_invalid_input,
            n_hs6_seen_hs4_unseen,
            n_manual_review_rule_mismatches,

            n_missing_model_metadata,
            n_distinct_model_names,
            n_distinct_model_types,

            n_training_rows_invalid_observed_quantity,
            n_training_rows_invalid_model_input,
            n_training_rows_unseen_categories,

            n_unit_value_identity_rows,
            n_unit_value_identity_mismatches,

            n_observed_quantity_rows,
            n_missing_observed_quantity_rows,
            n_model_input_valid_rows,
            n_model_input_invalid_rows,
            n_prediction_rows,
            n_residual_rows,
            n_hs6_unseen_rows,
            n_hs4_unseen_rows,
            n_hs6_unseen_codes,
            n_hs4_unseen_codes,
            n_manual_review_rows,
            n_manual_review_hs6_codes,
            n_training_selected_rows,
            median_abs_log_residual,
            mean_abs_log_residual,

            n_duplicate_keys,
        ) = audit_row

        # Normalize integer outputs.
        integer_names = {
            "n_rows": n_rows,
            "n_missing_key_rows": n_missing_key_rows,
            "n_invalid_hs4_rows": n_invalid_hs4_rows,
            "n_null_training_selected": n_null_training_selected,
            "n_null_model_input_valid": n_null_model_input_valid,
            "n_nonfinite_trade_values": n_nonfinite_trade_values,
            "n_valid_model_input_bad_trade_values":
                n_valid_model_input_bad_trade_values,
            "n_observed_availability_mismatches":
                n_observed_availability_mismatches,
            "n_observed_log_validity_mismatches":
                n_observed_log_validity_mismatches,
            "n_nonfinite_observed_quantities":
                n_nonfinite_observed_quantities,
            "n_nonpositive_observed_quantities":
                n_nonpositive_observed_quantities,
            "n_prediction_availability_mismatches":
                n_prediction_availability_mismatches,
            "n_invalid_available_predictions":
                n_invalid_available_predictions,
            "n_predictions_with_invalid_model_input":
                n_predictions_with_invalid_model_input,
            "n_missing_predictions_for_valid_input":
                n_missing_predictions_for_valid_input,
            "n_prediction_inverse_transform_mismatches":
                n_prediction_inverse_transform_mismatches,
            "n_residual_availability_mismatches":
                n_residual_availability_mismatches,
            "n_residual_formula_mismatches":
                n_residual_formula_mismatches,
            "n_absolute_residual_mismatches":
                n_absolute_residual_mismatches,
            "n_unexpected_absolute_residuals":
                n_unexpected_absolute_residuals,
            "n_missing_support_flags_valid_input":
                n_missing_support_flags_valid_input,
            "n_support_flags_on_invalid_input":
                n_support_flags_on_invalid_input,
            "n_hs6_seen_hs4_unseen":
                n_hs6_seen_hs4_unseen,
            "n_missing_model_metadata":
                n_missing_model_metadata,
            "n_distinct_model_names":
                n_distinct_model_names,
            "n_distinct_model_types":
                n_distinct_model_types,
            "n_training_rows_invalid_observed_quantity":
                n_training_rows_invalid_observed_quantity,
            "n_training_rows_invalid_model_input":
                n_training_rows_invalid_model_input,
            "n_training_rows_unseen_categories":
                n_training_rows_unseen_categories,
            "n_unit_value_identity_rows":
                n_unit_value_identity_rows,
            "n_unit_value_identity_mismatches":
                n_unit_value_identity_mismatches,
            "n_observed_quantity_rows":
                n_observed_quantity_rows,
            "n_missing_observed_quantity_rows":
                n_missing_observed_quantity_rows,
            "n_model_input_valid_rows":
                n_model_input_valid_rows,
            "n_model_input_invalid_rows":
                n_model_input_invalid_rows,
            "n_prediction_rows":
                n_prediction_rows,
            "n_residual_rows":
                n_residual_rows,
            "n_hs6_unseen_rows":
                n_hs6_unseen_rows,
            "n_hs4_unseen_rows":
                n_hs4_unseen_rows,
            "n_hs6_unseen_codes":
                n_hs6_unseen_codes,
            "n_hs4_unseen_codes":
                n_hs4_unseen_codes,
            "n_training_selected_rows":
                n_training_selected_rows,
            "n_duplicate_keys":
                n_duplicate_keys,
            "n_manual_review_rule_mismatches":
                n_manual_review_rule_mismatches,

            "n_manual_review_rows":
                n_manual_review_rows,

            "n_manual_review_hs6_codes":
                n_manual_review_hs6_codes,
        }

        counts = {
            name: int(
                value or 0
            )
            for name, value
            in integer_names.items()
        }

        # ==============================================================
        # HARD AUDIT CHECKS
        # ==============================================================

        checks = {
            "bilateral_key_unique":
                counts["n_duplicate_keys"] == 0,

            "bilateral_keys_complete":
                counts["n_missing_key_rows"] == 0,

            "hs4_derivation_valid":
                counts["n_invalid_hs4_rows"] == 0,

            "training_selected_complete":
                counts["n_null_training_selected"] == 0,

            "model_input_valid_complete":
                counts["n_null_model_input_valid"] == 0,

            "trade_values_finite":
                counts["n_nonfinite_trade_values"] == 0,

            "valid_model_inputs_have_valid_trade_value":
                counts[
                    "n_valid_model_input_bad_trade_values"
                ] == 0,

            "observed_availability_valid":
                counts[
                    "n_observed_availability_mismatches"
                ] == 0,

            "observed_log_validity_valid":
                counts[
                    "n_observed_log_validity_mismatches"
                ] == 0,

            "observed_quantities_finite":
                counts[
                    "n_nonfinite_observed_quantities"
                ] == 0,

            "observed_quantities_positive":
                counts[
                    "n_nonpositive_observed_quantities"
                ] == 0,

            "prediction_availability_valid":
                counts[
                    "n_prediction_availability_mismatches"
                ] == 0,

            "available_predictions_valid":
                counts[
                    "n_invalid_available_predictions"
                ] == 0,

            "invalid_inputs_have_no_prediction":
                counts[
                    "n_predictions_with_invalid_model_input"
                ] == 0,

            "valid_inputs_have_prediction":
                counts[
                    "n_missing_predictions_for_valid_input"
                ] == 0,

            "prediction_inverse_transform_valid":
                counts[
                    "n_prediction_inverse_transform_mismatches"
                ] == 0,

            "residual_availability_valid":
                counts[
                    "n_residual_availability_mismatches"
                ] == 0,

            "residual_formula_valid":
                counts[
                    "n_residual_formula_mismatches"
                ] == 0,

            "absolute_residual_formula_valid":
                counts[
                    "n_absolute_residual_mismatches"
                ] == 0,

            "absolute_residual_presence_valid":
                counts[
                    "n_unexpected_absolute_residuals"
                ] == 0,

            "support_flags_complete_for_valid_input":
                counts[
                    "n_missing_support_flags_valid_input"
                ] == 0,

            "support_flags_absent_for_invalid_input":
                counts[
                    "n_support_flags_on_invalid_input"
                ] == 0,

            "hs6_seen_implies_hs4_seen":
                counts[
                    "n_hs6_seen_hs4_unseen"
                ] == 0,

            "model_metadata_complete":
                counts[
                    "n_missing_model_metadata"
                ] == 0,

            "single_model_name":
                counts[
                    "n_distinct_model_names"
                ] == 1,

            "single_model_type":
                counts[
                    "n_distinct_model_types"
                ] == 1,

            "training_rows_have_valid_observed_quantity":
                counts[
                    "n_training_rows_invalid_observed_quantity"
                ] == 0,

            "training_rows_have_valid_model_input":
                counts[
                    "n_training_rows_invalid_model_input"
                ] == 0,

            "training_rows_have_seen_categories":
                counts[
                    "n_training_rows_unseen_categories"
                ] == 0,

            "unit_value_identity_valid":
                counts[
                    "n_unit_value_identity_mismatches"
                ] == 0,
            "manual_review_flag_consistent":
                counts["n_manual_review_rule_mismatches"] == 0,
        }

        audit_passed = all(
            checks.values()
        )

        # ==============================================================
        # WRITE ONE-ROW AUDIT PRODUCT
        # ==============================================================

        audit_values: dict[str, Any] = {
            "run_id": run_id,
            **counts,
            "median_abs_log_residual": (
                None
                if median_abs_log_residual is None
                else float(
                    median_abs_log_residual
                )
            ),
            "mean_abs_log_residual": (
                None
                if mean_abs_log_residual is None
                else float(
                    mean_abs_log_residual
                )
            ),
            **checks,
            "audit_passed": audit_passed,
        }

        columns_sql: list[str] = []

        for name, value in audit_values.items():

            if isinstance(
                value,
                bool,
            ):
                sql_value = (
                    "TRUE"
                    if value
                    else "FALSE"
                )

            elif value is None:
                sql_value = "NULL"

            elif isinstance(
                value,
                str,
            ):
                sql_value = sql_literal(
                    value
                )

            else:
                sql_value = str(
                    value
                )

            columns_sql.append(
                f"{sql_value} AS {name}"
            )

        con.execute(
            f"""
            COPY (
                SELECT
                    {", ".join(columns_sql)}
            )
            TO {output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        # ==============================================================
        # LOG SUMMARY
        # ==============================================================

        logger.info(
            "  Rows: %s",
            f"{counts['n_rows']:,}",
        )

        logger.info(
            "  Observed quantity rows: %s",
            f"{counts['n_observed_quantity_rows']:,}",
        )

        logger.info(
            "  Missing observed quantity rows: %s",
            f"{counts['n_missing_observed_quantity_rows']:,}",
        )

        logger.info(
            "  Model-input valid rows: %s",
            f"{counts['n_model_input_valid_rows']:,}",
        )

        logger.info(
            "  Prediction rows: %s",
            f"{counts['n_prediction_rows']:,}",
        )

        logger.info(
            "  Residual rows: %s",
            f"{counts['n_residual_rows']:,}",
        )

        logger.info(
            "  Rows with unseen HS6: %s",
            f"{counts['n_hs6_unseen_rows']:,}",
        )

        logger.info(
            "  Distinct unseen HS6 codes: %s",
            f"{counts['n_hs6_unseen_codes']:,}",
        )

        logger.info(
            "  HS6 unseen rows: %s",
            f"{counts['n_hs6_unseen_rows']:,}",
        )

        logger.info(
            "  HS4 unseen rows: %s",
            f"{counts['n_hs4_unseen_rows']:,}",
        )

        logger.info(
            "  Manual review rows: %s (HS6 codes: %s)",
            f"{counts['n_manual_review_rows']:,}",
            f"{counts['n_manual_review_hs6_codes']:,}",
        )

        logger.info(
            "  Manual review rule mismatches: %s",
            f"{counts['n_manual_review_rule_mismatches']:,}",
        )
        logger.info(
            "  Duplicate bilateral keys: %s",
            f"{counts['n_duplicate_keys']:,}",
        )

        logger.info(
            "  Residual formula mismatches: %s",
            f"{counts['n_residual_formula_mismatches']:,}",
        )

        logger.info(
            "  Unit-value identity mismatches: %s",
            f"{counts['n_unit_value_identity_mismatches']:,}",
        )

        logger.info(
            "%s WRITE — Validation consistency audit: %s",
            get_symbols("write"),
            output_path,
        )

        # ==============================================================
        # FAILURE
        # ==============================================================

        if not audit_passed:

            failed_checks = [
                name
                for name, passed
                in checks.items()
                if not passed
            ]

            raise RuntimeError(
                "Validation cross-variable consistency "
                "audit failed: "
                + ", ".join(
                    failed_checks
                )
            )

        logger.info(
            "%s AUDIT — Validation cross-variable consistency passed.",
            get_symbols("check"),
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "output_path": output_path,
            "inference_audit_path": output_path,
            "audit_passed": True,
            "n_rows": counts[
                "n_rows"
            ],
            "n_prediction_rows": counts[
                "n_prediction_rows"
            ],
            "n_residual_rows": counts[
                "n_residual_rows"
            ],
            "n_unit_value_identity_mismatches": counts[
                "n_unit_value_identity_mismatches"
            ],
        
            "n_manual_review_rows": counts["n_manual_review_rows"],
            "n_manual_review_hs6_codes": counts["n_manual_review_hs6_codes"],
            "n_manual_review_rule_mismatches": counts["n_manual_review_rule_mismatches"],
        }

    finally:
        con.close()